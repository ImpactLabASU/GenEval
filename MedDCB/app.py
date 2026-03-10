import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import pysindy as ps
# why: sklearn >=1.8 returns 6 values from _preprocess_data; PySINDy 1.7.5 expects 5
import pysindy.optimizers.base as _sindy_base
_orig_preprocess = _sindy_base._preprocess_data
_sindy_base._preprocess_data = lambda *a, **kw: _orig_preprocess(*a, **kw)[:5]
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

import dcb_core
from dcb_core import check_coverage, mahalanobis_bounds

app = Flask(__name__, template_folder="template")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024 * 1024  # 100GB max upload ceiling
app.config["MAX_FORM_PARTS"] = 20_000  # allow thousands of images per dataset
app.config["MAX_FORM_MEMORY_SIZE"] = 256 * 1024 * 1024  # allow large multipart headers for many files

# why: control staging location; what: keep temp jobs in workspace
JOB_STORAGE_ROOT = Path("dcb_tmp")
JOB_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
TORCH_DEVICE = None
if torch is not None:
    try:
        TORCH_DEVICE = torch.device("cuda") if torch.cuda.is_available() else None
    except Exception:
        TORCH_DEVICE = None


# why: PySINDy model for theta extraction; what: reusable model instance
SINDY_LIBRARY = ps.PolynomialLibrary(degree=5)
SINDY_OPTIMIZER = ps.STLSQ(threshold=0.1, normalize_columns=True)
SINDY_MODEL = ps.SINDy(feature_library=SINDY_LIBRARY, optimizer=SINDY_OPTIMIZER)
SINDY_LOCK = threading.Lock()

# why: background job registry; what: track async progress safely
jobs = {}
jobs_lock = threading.Lock()
# why: keep pre-uploaded datasets before pipeline starts; what: support auto-upload in parallel
staged_uploads = {}
staged_uploads_lock = threading.Lock()


def update_stage_progress(job_id, dataset_key, saved_count, total_count, filename):
    """why: surface staging progress; what: reflect saved file counts + log snippet"""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        dataset = job["datasets"].get(dataset_key)
        if not dataset:
            return
        dataset["processed"] = min(saved_count, dataset["total"])
        dataset["last_image"] = f"Staged: {filename}"
        dataset["theta_snippet"] = []
        if total_count:
            dataset["total"] = total_count
        if saved_count == total_count or saved_count % 200 == 0:
            job["log"].append(f"Staging {dataset_key}: {saved_count}/{total_count or '?'} files")
            job["log"] = job["log"][-6:]


def reset_dataset_progress(job_id):
    """why: reuse progress bars post-staging; what: zero processed counts before θ extraction"""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        for dataset in job["datasets"].values():
            dataset["processed"] = 0
            dataset["last_image"] = "θ extractor warming up..."
            dataset["theta_snippet"] = []


def close_upload_streams(file_handles):
    """why: release temp upload buffers; what: close FileStorage streams safely"""
    for handle in file_handles:
        closer = getattr(handle, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                app.logger.exception("Failed to close upload handle")


def as_bool(value):
    """why: normalize truthy flags from form/json; what: shared parser for auto-chain toggles"""
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_session_id(raw_session_id):
    """why: keep staged upload paths safe; what: allow only filesystem-safe session ids"""
    cleaned = secure_filename(str(raw_session_id or "").strip())
    return cleaned or None


def _new_staged_dataset_state():
    """why: normalize upload bookkeeping; what: shared schema for staged dataset records"""
    return {
        "images": [],
        "csv": None,
        "total": 0,
        "completed": False,
        "upload_id": None,
        "total_chunks": 1,
        "received_chunks": [],
    }


def _get_or_create_staged_upload(session_id):
    """why: reuse partial uploads; what: store per-session dataset metadata + base path"""
    with staged_uploads_lock:
        session = staged_uploads.get(session_id)
        if session:
            return session
        base_dir = Path(tempfile.mkdtemp(prefix=f"dcb_stage_{session_id}_", dir=str(JOB_STORAGE_ROOT)))
        session = {
            "session_id": session_id,
            "base_dir": str(base_dir),
            "datasets": {
                "dataset_a": _new_staged_dataset_state(),
                "dataset_b": _new_staged_dataset_state(),
            },
        }
        staged_uploads[session_id] = session
        return session


def _snapshot_staged_upload(session_id):
    """why: read staged session without races; what: copy shape needed for API checks"""
    with staged_uploads_lock:
        session = staged_uploads.get(session_id)
        if not session:
            return None
        datasets = session.get("datasets", {})
        a = datasets.get("dataset_a", {})
        b = datasets.get("dataset_b", {})
        return {
            "session_id": session.get("session_id"),
            "base_dir": session.get("base_dir"),
            "datasets": {
                "dataset_a": {
                    "images": list(a.get("images", [])),
                    "csv": a.get("csv"),
                    "total": int(a.get("total", 0)),
                    "completed": bool(a.get("completed", False)),
                },
                "dataset_b": {
                    "images": list(b.get("images", [])),
                    "csv": b.get("csv"),
                    "total": int(b.get("total", 0)),
                    "completed": bool(b.get("completed", False)),
                },
            },
        }


def _pop_staged_upload(session_id):
    """why: avoid duplicate runs from same uploads; what: consume staged session once pipeline starts"""
    with staged_uploads_lock:
        return staged_uploads.pop(session_id, None)


def _cleanup_staged_upload(session):
    """why: avoid leaked staged files on failure; what: remove session workspace safely"""
    if not session:
        return
    base_dir = session.get("base_dir")
    if base_dir:
        shutil.rmtree(base_dir, ignore_errors=True)


def _save_uploaded_dataset(session_id, dataset_key, files, upload_id=None, chunk_index=0, total_chunks=1):
    """why: persist chunked dataset uploads; what: avoid giant multipart requests for large folders"""
    if dataset_key not in {"dataset_a", "dataset_b"}:
        raise ValueError("Invalid dataset key")
    if total_chunks < 1:
        raise ValueError("total_chunks must be >= 1")
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise ValueError("chunk_index is out of range")

    images, csv_file = _split_folder_files(files)
    if not images and csv_file is None:
        raise ValueError("No files found in upload chunk")

    session = _get_or_create_staged_upload(session_id)
    base_dir = Path(session["base_dir"])
    dataset_dir = base_dir / dataset_key
    upload_token = secure_filename(str(upload_id or "").strip()) or None

    with staged_uploads_lock:
        record = staged_uploads.get(session_id)
        if not record:
            raise RuntimeError("Upload session expired")
        dataset_record = record["datasets"].get(dataset_key) or _new_staged_dataset_state()
        current_token = dataset_record.get("upload_id")
        if not upload_token:
            upload_token = current_token or f"{dataset_key}_{uuid.uuid4().hex}"
        reset_upload = current_token != upload_token or chunk_index == 0
        if reset_upload:
            dataset_record = _new_staged_dataset_state()
            dataset_record["upload_id"] = upload_token
            dataset_record["total_chunks"] = total_chunks
            record["datasets"][dataset_key] = dataset_record
        else:
            expected_chunks = int(dataset_record.get("total_chunks", total_chunks) or total_chunks)
            if total_chunks != expected_chunks:
                raise ValueError(
                    f"Mismatched chunk plan for {dataset_key}: expected {expected_chunks}, got {total_chunks}"
                )

    if reset_upload:
        shutil.rmtree(dataset_dir, ignore_errors=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for file in images:
        rel_path = (file.filename or "").replace("\\", "/")
        safe_parts = [secure_filename(p) for p in rel_path.split("/") if p]
        fallback_name = secure_filename(file.filename or "image") or "image"
        safe_rel = os.path.join(*safe_parts) if safe_parts else fallback_name
        target_path = dataset_dir / safe_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        file.save(str(target_path))
        image_paths.append(str(target_path))

    csv_target = None
    if csv_file is not None:
        csv_target = base_dir / f"{dataset_key}_labels.csv"
        csv_file.save(str(csv_target))

    with staged_uploads_lock:
        record = staged_uploads.get(session_id)
        if not record:
            raise RuntimeError("Upload session expired")
        dataset_record = record["datasets"].get(dataset_key)
        if not dataset_record or dataset_record.get("upload_id") != upload_token:
            raise RuntimeError("Upload session replaced; please retry")

        existing_paths = list(dataset_record.get("images", []))
        existing_set = set(existing_paths)
        for path in image_paths:
            if path not in existing_set:
                existing_paths.append(path)
                existing_set.add(path)
        dataset_record["images"] = existing_paths

        if csv_target is not None:
            dataset_record["csv"] = str(csv_target)

        received_chunks = set(dataset_record.get("received_chunks", []))
        received_chunks.add(chunk_index)
        dataset_record["received_chunks"] = sorted(received_chunks)
        dataset_record["total_chunks"] = total_chunks
        dataset_record["total"] = len(existing_paths)
        dataset_record["completed"] = bool(
            dataset_record["total"]
            and dataset_record.get("csv")
            and len(received_chunks) >= total_chunks
        )

        datasets = record["datasets"]
        ready = all(datasets[k].get("completed") for k in ("dataset_a", "dataset_b"))
        totals = {
            "dataset_a": int(datasets["dataset_a"].get("total", 0)),
            "dataset_b": int(datasets["dataset_b"].get("total", 0)),
        }

    return {
        "session_id": session_id,
        "dataset": dataset_key,
        "total_images": totals.get(dataset_key, 0),
        "totals": totals,
        "ready": ready,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "uploaded_chunks": len(dataset_record.get("received_chunks", [])),
        "completed": bool(dataset_record.get("completed")),
    }


def register_job(count_a, count_b):
    """why: keep async state; what: create job record with per-dataset totals"""
    job_id = uuid.uuid4().hex
    record = {
        "job_id": job_id,
        "status": "queued",
        "stage": "theta",
        "datasets": {
            "A": {"label": "Dataset A", "total": count_a, "processed": 0, "last_image": None, "theta_snippet": []},
            "B": {"label": "Dataset B", "total": count_b, "processed": 0, "last_image": None, "theta_snippet": []},
        },
        "recent_theta": [],
        "log": ["Upload accepted"],
        "result": None,
        "error": None,
        "ready_for_dcb": False,
        "theta_artifacts": None,
        "workspace": None,
    }
    with jobs_lock:
        jobs[job_id] = record
    return job_id


def set_job_status(job_id, status, note=None, payload=None):
    """why: mutate shared job safely; what: status + optional data/log update"""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job["status"] = status
        if payload:
            job.update(payload)
        if note:
            job["log"].append(note)
            job["log"] = job["log"][-6:]


def record_progress(job_id, dataset_key, image_id, theta_values):
    """why: feed UI with live samples; what: store counts + theta snippet"""
    snippet = [round(float(val), 4) for val in theta_values[:6]]
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        dataset = job["datasets"].get(dataset_key)
        if not dataset:
            return
        dataset["processed"] = min(dataset["processed"] + 1, dataset["total"])
        dataset["last_image"] = image_id
        dataset["theta_snippet"] = snippet
        job["recent_theta"] = snippet
        job["log"].append(f"{dataset_key}: {image_id} θ={snippet}")
        job["log"] = job["log"][-6:]


def snapshot_job(job_id):
    """why: serve status endpoint; what: shallow copy for JSON safety"""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "stage": job.get("stage", job["status"]),
            "datasets": job["datasets"],
            "recent_theta": job["recent_theta"],
            "log": job["log"],
            "result": job["result"],
            "error": job["error"],
            "ready_for_dcb": job.get("ready_for_dcb", False),
        }


def preprocess_image(img_path, size=(512, 512)):
    """why: prepare image for PySINDy processing; what: resize, enhance, smooth"""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        img_array = np.array(img)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return enhanced.astype(np.uint8), True
    except Exception as exc:
        return None, False


def _sample_circular_means_numpy(image_array, angle_step=10):
    """why: CPU fallback; what: original polar sampling loop"""
    h, w = image_array.shape[:2]
    center = (w // 2, h // 2)
    max_radius = max(1, min(w, h) // 2 - 2)
    rows = []
    for radius in range(max_radius):
        circle_points = []
        for angle in range(0, 360, angle_step):
            angle_rad = np.deg2rad(angle)
            x = int(center[0] + radius * np.cos(angle_rad))
            y = int(center[1] + radius * np.sin(angle_rad))
            if 0 <= x < w and 0 <= y < h:
                circle_points.append(image_array[y, x])
        if circle_points:
            rows.append(np.mean(circle_points, axis=0))
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)


def _sample_circular_means_gpu(image_array, angle_step=10):
    """why: leverage GPU when present; what: grid_sample circular ring statistics"""
    if TORCH_DEVICE is None or torch is None or F is None:
        return None
    h, w = image_array.shape[:2]
    if h < 2 or w < 2:
        return None
    max_radius = max(1, min(w, h) // 2 - 2)
    radii = torch.arange(0, max_radius, device=TORCH_DEVICE, dtype=torch.float32)
    if radii.numel() == 0:
        return None
    angles = torch.deg2rad(torch.arange(0, 360, angle_step, device=TORCH_DEVICE, dtype=torch.float32))
    if angles.numel() == 0:
        return None
    xs = (w / 2.0) + radii.unsqueeze(1) * torch.cos(angles)
    ys = (h / 2.0) + radii.unsqueeze(1) * torch.sin(angles)
    norm_x = (xs / (max(w - 1, 1))) * 2 - 1
    norm_y = (ys / (max(h - 1, 1))) * 2 - 1
    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
    tensor = torch.from_numpy(image_array).to(TORCH_DEVICE, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    samples = F.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    samples = samples.mean(dim=-1).squeeze(0).transpose(0, 1) * 255.0
    return samples.detach().cpu().numpy()


def sample_circular_means(image_array):
    """why: centralize polar stats; what: GPU first, CPU fallback"""
    if TORCH_DEVICE is not None:
        try:
            gpu_result = _sample_circular_means_gpu(image_array)
            if gpu_result is not None:
                return gpu_result
        except Exception:
            app.logger.exception("GPU theta sampling failed; falling back to CPU")
    return _sample_circular_means_numpy(image_array)


def extract_theta_from_image(img_path):
    """why: extract theta coefficients from image; what: circular sampling + PySINDy fit"""
    processed_img, success = preprocess_image(img_path)
    if not success:
        app.logger.warning(f"Failed to preprocess image: {img_path}")
        return None

    X = sample_circular_means(processed_img)
    if X is None or len(X) < 2:
        app.logger.warning(f"Failed to sample circular means for: {img_path}")
        return None

    dX = np.zeros_like(X)
    dX[:-1] = X[1:] - X[:-1]
    dX[-1] = dX[-2]
    
    try:
        with SINDY_LOCK:
            SINDY_MODEL.fit(X, t=np.arange(len(X)), x_dot=dX)
            theta = SINDY_MODEL.coefficients()
        return theta.flatten().astype(float).tolist()
    except Exception as exc:
        app.logger.warning(f"PySINDy fit failed for {img_path}: {exc}")
        return None


def _normalize_identifier(value):
    """why: compare csv/image names robustly; what: normalize slashes/case/whitespace"""
    text = str(value or "").strip().replace("\\", "/")
    if not text or text.lower() in {"nan", "none"}:
        return ""
    while text.startswith("./"):
        text = text[2:]
    return text.lower()


def _column_token(name):
    """why: tolerate header formatting; what: remove punctuation/spaces for matching"""
    return "".join(ch for ch in str(name or "").strip().lower() if ch.isalnum())


def _find_image_identifier_column(columns):
    """why: avoid hard dependency on image_id; what: pick common id/path header variants"""
    candidates = [str(col).strip() for col in columns]
    if "image_id" in candidates:
        return "image_id"
    token_map = {col: _column_token(col) for col in candidates}
    preferred = (
        "imageid",
        "image",
        "imagename",
        "filename",
        "filepath",
        "path",
        "file",
        "img",
    )
    for target in preferred:
        for col, token in token_map.items():
            if token == target:
                return col
    for col, token in token_map.items():
        if "image" in token or token.endswith("path"):
            return col
    return None


def _image_identifier_variants(relative_with_ext):
    """why: support diverse CSV id formats; what: derive path+basename variants with/without ext"""
    rel_norm = _normalize_identifier(relative_with_ext)
    if not rel_norm:
        return set()
    rel_obj = Path(rel_norm)
    variants = {
        rel_norm,
        _normalize_identifier(str(rel_obj.with_suffix(""))),
        _normalize_identifier(rel_obj.name),
        _normalize_identifier(rel_obj.stem),
    }
    return {value for value in variants if value}


def process_images_to_dataframe(image_files, csv_path, temp_dir, progress_cb=None):
    """why: convert images + CSV labels to theta dataframe; what: extract features and merge with labels"""
    labels_df = pd.read_csv(csv_path)
    labels_df.columns = [str(col).strip() for col in labels_df.columns]
    id_column = _find_image_identifier_column(labels_df.columns)
    if not id_column:
        available = ", ".join(labels_df.columns)
        raise ValueError(
            f"CSV must contain an image identifier column (e.g. image_id/image/path/filename). "
            f"Found: {available}"
        )
    
    os.makedirs(temp_dir, exist_ok=True)
    image_map = {}
    
    # why: files are already saved as paths from staging; what: build image_id to path mapping
    # image_id uses relative path stem (e.g. patient_001/slice_01) to match CSV entries
    for file in image_files:
        rel_with_ext = ""
        if hasattr(file, "save"):
            image_dir = os.path.join(temp_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(image_dir, filename)
            file.save(filepath)
            image_id = os.path.splitext(filename)[0]
            rel_with_ext = filename
        else:
            filepath = str(file)
            # why: derive image_id from path relative to dataset staging dir; what: match CSV subfolder paths
            dataset_dir = str(Path(filepath).parent.parent)
            try:
                rel = Path(filepath).relative_to(dataset_dir)
                rel_with_ext = str(rel).replace("\\", "/")
                image_id = str(rel.with_suffix(""))
            except ValueError:
                rel_with_ext = os.path.basename(filepath)
                image_id = os.path.splitext(os.path.basename(filepath))[0]
        variants = _image_identifier_variants(rel_with_ext)
        variants.add(_normalize_identifier(image_id))
        image_map[image_id] = {"path": filepath, "variants": {v for v in variants if v}}
    
    app.logger.info(f"Processing {len(image_map)} images for theta extraction")
    theta_rows = []
    processed_count = 0
    failed_count = 0
    
    # why: extract theta coefficients from each image; what: PySINDy feature extraction
    for image_id, image_meta in image_map.items():
        img_path = image_meta["path"]
        processed_count += 1
        theta = extract_theta_from_image(img_path)
        
        if theta is not None:
            row = {"image_id": image_id}
            for i, val in enumerate(theta):
                row[f"theta_{i}"] = val
            theta_rows.append(row)
            # why: update UI with successful extraction; what: fire progress callback with theta values
            if progress_cb:
                progress_cb(image_id, theta)
        else:
            failed_count += 1
            # why: show progress even for failed extractions; what: fire callback with dummy theta
            if progress_cb:
                progress_cb(image_id, [0.0] * 6)
        
        if processed_count % 100 == 0:
            app.logger.info(f"Processed {processed_count}/{len(image_map)} images ({failed_count} failed so far)")
    
    if not theta_rows:
        app.logger.error(f"No valid theta features extracted from {len(image_map)} images - all failed!")
        raise ValueError(f"No valid theta features extracted from images. All {len(image_map)} images failed extraction.")
    
    success_rate = (len(theta_rows) / len(image_map)) * 100
    app.logger.info(f"Theta extraction complete: {len(theta_rows)}/{len(image_map)} succeeded ({success_rate:.1f}%), {failed_count} failed")
    
    theta_df = pd.DataFrame(theta_rows)
    variant_lookup = {}
    ambiguous_variants = set()
    for image_id, image_meta in image_map.items():
        for variant in image_meta["variants"]:
            previous = variant_lookup.get(variant)
            if previous and previous != image_id:
                ambiguous_variants.add(variant)
            else:
                variant_lookup[variant] = image_id
    for variant in ambiguous_variants:
        variant_lookup.pop(variant, None)

    labels_aligned = labels_df.copy()
    labels_aligned["_lookup_key"] = labels_aligned[id_column].map(_normalize_identifier)
    labels_aligned["image_id"] = labels_aligned["_lookup_key"].map(variant_lookup)
    matched_rows = int(labels_aligned["image_id"].notna().sum())
    if matched_rows == 0:
        raise ValueError(
            f"Could not match CSV column '{id_column}' to uploaded image names. "
            "Ensure CSV identifiers use image filenames or paths."
        )
    if matched_rows < len(labels_aligned):
        app.logger.warning(
            "Matched %d/%d CSV rows from '%s' to uploaded images",
            matched_rows,
            len(labels_aligned),
            id_column,
        )

    labels_aligned = labels_aligned.drop(columns=["_lookup_key"])
    labels_matched = labels_aligned[labels_aligned["image_id"].notna()].copy()
    duplicate_rows = int(labels_matched["image_id"].duplicated().sum())
    if duplicate_rows:
        app.logger.warning("Dropping %d duplicate CSV rows after image_id alignment", duplicate_rows)
        labels_matched = labels_matched.drop_duplicates(subset=["image_id"], keep="first")

    result_df = theta_df.merge(labels_matched, on="image_id", how="left")
    return result_df


def persist_job_inputs(images_a, csv_a, images_b, csv_b, progress_cb=None):
    """why: reuse uploads asynchronously; what: save to unique job directory with optional staging callback"""
    job_dir = Path(tempfile.mkdtemp(prefix="dcb_job_", dir=str(JOB_STORAGE_ROOT)))
    dataset_a_dir = job_dir / "dataset_a"
    dataset_b_dir = job_dir / "dataset_b"
    dataset_a_dir.mkdir(parents=True, exist_ok=True)
    dataset_b_dir.mkdir(parents=True, exist_ok=True)

    def save_images(files, target_dir, dataset_key):
        saved_paths = []
        total = len(files) or 0
        for idx, file in enumerate(files, 1):
            # why: preserve subfolder structure from webkitdirectory upload; secure_filename strips '/'
            rel_path = file.filename.replace("\\", "/")
            safe_parts = [secure_filename(p) for p in rel_path.split("/") if p]
            safe_rel = os.path.join(*safe_parts) if safe_parts else secure_filename(file.filename)
            filepath = target_dir / safe_rel
            filepath.parent.mkdir(parents=True, exist_ok=True)
            file.save(filepath)
            saved_paths.append(str(filepath))
            if progress_cb:
                try:
                    progress_cb(dataset_key, idx, total, safe_rel)
                except Exception:
                    app.logger.exception("Staging progress callback failed")
        return saved_paths

    try:
        image_paths_a = save_images(images_a, dataset_a_dir, "A")
        image_paths_b = save_images(images_b, dataset_b_dir, "B")
        csv_a_path = job_dir / "labels_a.csv"
        csv_b_path = job_dir / "labels_b.csv"
        csv_a.save(str(csv_a_path))
        csv_b.save(str(csv_b_path))
        return {
            "base_dir": str(job_dir),
            "images_a": image_paths_a,
            "images_b": image_paths_b,
            "csv_a": str(csv_a_path),
            "csv_b": str(csv_b_path),
        }
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise


def update_job_ready_stage(job_id, payload):
    """why: snapshot staging output; what: prepare job for θ extraction"""
    if not payload:
        app.logger.error(f"Job {job_id} completed staging but payload is None")
        set_job_status(job_id, "error", "Failed to prepare job payload", {"error": "Payload is None"})
        return None
    reset_dataset_progress(job_id)
    app.logger.info(f"Job {job_id} starting theta extraction")
    return payload


def run_theta_job(job_id, payload, auto_chain=False):
    """why: extract theta features first; what: optionally auto-chain DCB for single-click runs"""
    set_job_status(job_id, "theta_running", "Extracting theta streams...", {"stage": "theta"})
    app.logger.info(f"Job {job_id} starting theta extraction for {len(payload['images_a'])} images in A, {len(payload['images_b'])} in B")
    try:
        work_a = os.path.join(payload["base_dir"], "work_a")
        work_b = os.path.join(payload["base_dir"], "work_b")
        
        app.logger.info(f"Job {job_id} processing Dataset A...")
        df_a = process_images_to_dataframe(
            payload["images_a"],
            payload["csv_a"],
            work_a,
            progress_cb=lambda image_id, theta: record_progress(job_id, "A", image_id, theta),
        )
        app.logger.info(f"Job {job_id} completed Dataset A: {len(df_a)} rows")
        
        app.logger.info(f"Job {job_id} processing Dataset B...")
        df_b = process_images_to_dataframe(
            payload["images_b"],
            payload["csv_b"],
            work_b,
            progress_cb=lambda image_id, theta: record_progress(job_id, "B", image_id, theta),
        )
        app.logger.info(f"Job {job_id} completed Dataset B: {len(df_b)} rows")
        
        theta_a_path = Path(payload["base_dir"]) / "theta_a.csv"
        theta_b_path = Path(payload["base_dir"]) / "theta_b.csv"
        df_a.to_csv(theta_a_path, index=False)
        df_b.to_csv(theta_b_path, index=False)

        # why: free disk weight; what: drop staged images after θ conversion completes
        try:
            shutil.rmtree(Path(payload["base_dir"]) / "dataset_a", ignore_errors=True)
            shutil.rmtree(Path(payload["base_dir"]) / "dataset_b", ignore_errors=True)
        except Exception:
            app.logger.exception(f"Job {job_id} failed to clean staged images")

        artifacts = {
            "dataset_a": {"path": str(theta_a_path), "rows": len(df_a)},
            "dataset_b": {"path": str(theta_b_path), "rows": len(df_b)},
        }
        app.logger.info(f"Job {job_id} theta conversion ready for SDCD")
        if auto_chain:
            set_job_status(
                job_id,
                "theta_ready",
                "θ conversion finished. Auto-starting SDCD computation.",
                {"ready_for_dcb": True, "theta_artifacts": artifacts, "stage": "theta"},
            )
            run_dcb_job(job_id)
        else:
            set_job_status(
                job_id,
                "theta_ready",
                "θ conversion finished. Ready for SDCD computation.",
                {"ready_for_dcb": True, "theta_artifacts": artifacts, "stage": "theta"},
            )
    except Exception as exc:
        app.logger.exception(f"Job {job_id} failed during theta extraction")
        set_job_status(job_id, "error", f"θ conversion failed: {exc}", {"error": str(exc)})


def run_dcb_job(job_id):
    """why: compute bounds + SDCD after theta is ready; what: reuse saved θ CSVs."""
    with jobs_lock:
        job = jobs.get(job_id)
        artifacts = job.get("theta_artifacts") if job else None
        workspace = job.get("workspace") if job else None
    if artifacts is None:
        set_job_status(job_id, "error", "SDCD requested but θ artifacts missing", {"error": "Missing theta artifacts"})
        return

    set_job_status(job_id, "dcb_running", "Computing SDCD scores...", {"stage": "dcb"})
    try:
        df_a = pd.read_csv(artifacts["dataset_a"]["path"])
        df_b = pd.read_csv(artifacts["dataset_b"]["path"])

        app.logger.info(f"Job {job_id} computing bounds...")
        bounds_a = mahalanobis_bounds(df_a)
        bounds_b = mahalanobis_bounds(df_b)
        coverage_a_on_b = check_coverage(df_a, df_b, bounds_a)
        coverage_b_on_a = check_coverage(df_b, df_a, bounds_b)

        if coverage_a_on_b["percent_in_range"] >= coverage_b_on_a["percent_in_range"]:
            best_source = "Dataset A"
            best_target = "Dataset B"
            best_coverage = coverage_a_on_b["percent_in_range"]
        else:
            best_source = "Dataset B"
            best_target = "Dataset A"
            best_coverage = coverage_b_on_a["percent_in_range"]

        result = {
            "dataset_a_bounds": bounds_a,
            "dataset_b_bounds": bounds_b,
            "coverage_a_on_b": coverage_a_on_b,
            "coverage_b_on_a": coverage_b_on_a,
            "recommendation": {
                "source": best_source,
                "target": best_target,
                "coverage_percent": round(best_coverage, 2),
            },
        }
        app.logger.info(f"Job {job_id} completed SDCD computation")
        set_job_status(
            job_id,
            "completed",
            "SDCD computation finished",
            {"result": result, "stage": "dcb", "ready_for_dcb": False},
        )
    except Exception as exc:
        app.logger.exception(f"Job {job_id} failed during SDCD computation")
        set_job_status(job_id, "error", f"Job failed: {exc}", {"error": str(exc)})
    finally:
        if workspace:
            app.logger.info(f"Job {job_id} cleaning up workspace at {workspace}")
            shutil.rmtree(workspace, ignore_errors=True)


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


def _split_folder_files(all_files):
    """why: single-folder upload mixes images + CSV; what: separate by extension"""
    images, csv_file = [], None
    for f in all_files:
        name = (f.filename or "").lower()
        if name.endswith(".csv"):
            csv_file = f
        else:
            images.append(f)
    return images, csv_file


@app.route("/upload/<session_id>/<dataset_key>", methods=["POST"])
def stage_single_dataset(session_id, dataset_key):
    """why: begin upload immediately after folder drop; what: save one dataset independently."""
    safe_session_id = _sanitize_session_id(session_id)
    if not safe_session_id:
        return jsonify({"error": "Invalid upload session id"}), 400
    files = request.files.getlist("dataset_folder")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400
    chunk_index = request.form.get("chunk_index", default=0, type=int)
    total_chunks = request.form.get("total_chunks", default=1, type=int)
    upload_id = request.form.get("upload_id", default="", type=str)
    try:
        payload = _save_uploaded_dataset(
            safe_session_id,
            dataset_key,
            files,
            upload_id=upload_id,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Failed staged upload for %s/%s", safe_session_id, dataset_key)
        return jsonify({"error": f"Failed to stage files: {exc}"}), 500
    finally:
        close_upload_streams(files)


@app.route("/theta/staged", methods=["POST"])
def start_theta_from_staged():
    """why: launch pipeline from already-uploaded folders; what: avoid re-upload on Run."""
    body = request.get_json(silent=True) or {}
    safe_session_id = _sanitize_session_id(body.get("session_id"))
    auto_chain = as_bool(body.get("auto_chain", 1))
    if not safe_session_id:
        return jsonify({"error": "session_id is required"}), 400

    snapshot = _snapshot_staged_upload(safe_session_id)
    if not snapshot:
        return jsonify({"error": "Upload session not found"}), 404

    ds_a = snapshot["datasets"]["dataset_a"]
    ds_b = snapshot["datasets"]["dataset_b"]
    if not ds_a.get("completed"):
        return jsonify({"error": "Dataset A is not fully uploaded yet"}), 409
    if not ds_b.get("completed"):
        return jsonify({"error": "Dataset B is not fully uploaded yet"}), 409

    session_payload = _pop_staged_upload(safe_session_id)
    if not session_payload:
        return jsonify({"error": "Upload session not available"}), 409

    totals = {
        "dataset_a": ds_a["total"],
        "dataset_b": ds_b["total"],
    }
    job_id = register_job(totals["dataset_a"], totals["dataset_b"])
    payload = {
        "base_dir": session_payload["base_dir"],
        "images_a": list(session_payload["datasets"]["dataset_a"]["images"]),
        "images_b": list(session_payload["datasets"]["dataset_b"]["images"]),
        "csv_a": session_payload["datasets"]["dataset_a"]["csv"],
        "csv_b": session_payload["datasets"]["dataset_b"]["csv"],
    }

    set_job_status(
        job_id,
        "staging",
        "Uploads complete. Launching θ extraction...",
        {"workspace": payload.get("base_dir"), "stage": "theta", "ready_for_dcb": False},
    )

    prepared = update_job_ready_stage(job_id, payload)
    if prepared is None:
        _cleanup_staged_upload(session_payload)
        return jsonify({"error": "Failed to prepare job payload"}), 500

    worker = threading.Thread(
        target=run_theta_job,
        args=(job_id, prepared, auto_chain),
        daemon=True,
    )
    worker.start()

    return jsonify({"job_id": job_id, "totals": totals, "auto_chain": auto_chain})


@app.route("/theta", methods=["POST"])
def start_theta_conversion():
    """Stage uploads and kick off θ conversion; optional auto-chain runs DCB afterwards."""
    files_a = request.files.getlist("dataset_a_folder")
    files_b = request.files.getlist("dataset_b_folder")
    auto_chain = as_bool(request.form.get("auto_chain", ""))

    images_a, csv_a = _split_folder_files(files_a)
    images_b, csv_b = _split_folder_files(files_b)

    if not images_a or csv_a is None:
        return jsonify({"error": "Dataset A folder must contain images and a CSV file"}), 400
    if not images_b or csv_b is None:
        return jsonify({"error": "Dataset B folder must contain images and a CSV file"}), 400

    totals = {"dataset_a": len(images_a), "dataset_b": len(images_b)}
    job_id = register_job(totals["dataset_a"], totals["dataset_b"])

    payload = None
    try:
        set_job_status(
            job_id,
            "staging",
            "Saving uploads to workspace...",
            {"stage": "theta", "ready_for_dcb": False},
        )
        payload = persist_job_inputs(
            images_a,
            csv_a,
            images_b,
            csv_b,
            progress_cb=lambda key, count, total, fname: update_stage_progress(
                job_id, key, count, total, fname
            ),
        )
        set_job_status(
            job_id,
            "staging",
            "Uploads staged. Launching θ extraction...",
            {"workspace": payload.get("base_dir"), "stage": "theta"},
        )
    except Exception as exc:
        app.logger.exception(f"Staging failed for job {job_id}")
        set_job_status(job_id, "error", f"Failed to stage uploads: {exc}", {"error": str(exc)})
        close_upload_streams(list(images_a) + list(images_b) + [csv_a, csv_b])
        return jsonify({"error": "Failed to store uploaded files"}), 500
    finally:
        close_upload_streams(list(images_a) + list(images_b) + [csv_a, csv_b])

    payload = update_job_ready_stage(job_id, payload)
    if payload is None:
        return jsonify({"error": "Failed to prepare job payload"}), 500

    worker = threading.Thread(
        target=run_theta_job,
        args=(job_id, payload, auto_chain),
        daemon=True,
    )
    worker.start()

    return jsonify({"job_id": job_id, "totals": totals, "auto_chain": auto_chain})


@app.route("/analyze/<job_id>", methods=["POST"])
def analyze_job(job_id):
    """Run SDCD computation using existing θ artifacts."""
    with jobs_lock:
        job = jobs.get(job_id)
        status = job.get("status") if job else None

    if job is None:
        return jsonify({"error": "Job not found"}), 404
    if status == "completed":
        return jsonify({"job_id": job_id, "status": status, "result": job.get("result")})
    if status == "dcb_running":
        return jsonify({"job_id": job_id, "status": status})
    if status != "theta_ready":
        return jsonify({"error": f"Job must be theta_ready before SDCD. Current status: {status}"}), 409

    worker = threading.Thread(target=run_dcb_job, args=(job_id,), daemon=True)
    worker.start()
    return jsonify({"job_id": job_id, "status": "dcb_running"})


@app.route("/status/<job_id>", methods=["GET"])
def job_status(job_id):
    """why: let UI poll async progress; what: surface current job snapshot"""
    job = snapshot_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.errorhandler(413)
def payload_too_large(exc):
    """why: avoid HTML blob in UI; what: emit JSON with size guidance"""
    limit_bytes = app.config.get("MAX_CONTENT_LENGTH") or 0
    form_parts = app.config.get("MAX_FORM_PARTS") or 0
    form_memory_bytes = app.config.get("MAX_FORM_MEMORY_SIZE") or 0
    limit_gb = round(limit_bytes / (1024**3), 2) if limit_bytes else None
    form_memory_mb = round(form_memory_bytes / (1024**2), 2) if form_memory_bytes else None
    received = request.content_length or 0
    received_gb = round(received / (1024**3), 2)
    app.logger.warning("Upload rejected: %.2f GB exceeds limit %.2f GB", received_gb, limit_gb or 0)
    reason = getattr(exc, "description", "") or ""
    if "multipart" in reason and "parts" in reason:
        message = (
            f"Upload rejected: {reason}. "
            f"Try fewer files per request or raise MAX_FORM_PARTS (current {form_parts})."
        )
    elif "memory" in reason and "form" in reason:
        message = (
            f"Upload rejected: {reason}. "
            f"Raise MAX_FORM_MEMORY_SIZE (current {form_memory_mb} MB)."
        )
    elif limit_gb and received > limit_bytes:
        message = f"Upload exceeds {limit_gb} GB limit (received ~{received_gb} GB)."
    else:
        parser_hints = []
        if form_parts:
            parser_hints.append(f"MAX_FORM_PARTS={form_parts}")
        if form_memory_mb:
            parser_hints.append(f"MAX_FORM_MEMORY_SIZE={form_memory_mb} MB")
        hint_text = f" Parser settings: {', '.join(parser_hints)}." if parser_hints else ""
        message = (
            "Upload exceeds server limit. "
            "If file count is high, increase multipart parser limits." + hint_text
        )
    return jsonify({"error": message}), 413


@app.route("/health/storage", methods=["GET"])
def health_storage():
    """Report available disk space for monitoring."""
    stat = shutil.disk_usage("/")
    return jsonify({
        "total_gb": round(stat.total / (1024**3), 2),
        "used_gb": round(stat.used / (1024**3), 2),
        "free_gb": round(stat.free / (1024**3), 2),
        "free_percent": round(100 * stat.free / stat.total, 2),
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
