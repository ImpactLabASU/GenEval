#!/usr/bin/env python
"""Cross-domain evaluation for GenEval LoRA adapters (DR only)."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
import yaml
from peft import PeftModel
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, set_seed

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

IMAGE_COLUMN_CANDIDATES = (
    "path",
    "image_path",
    "image",
    "filepath",
    "file_path",
    "filename",
)
LABEL_COLUMN_CANDIDATES = (
    "label",
    "grade",
    "retinopathy grade",
    "target",
    "class",
    "severity",
)
DIABETIC_RETINOPATHY_CLASSES = ["0", "1", "2", "3", "4"]
DIABETIC_RETINOPATHY_OPTIONS = "\n".join(DIABETIC_RETINOPATHY_CLASSES)
DIABETIC_RETINOPATHY_PROMPT = (
    "You are a medical imaging expert with 10 years of experience specialized in "
    "diabetic rethenopathy trained to assist ophthalmologists by grading the fundus "
    "image. Analyze the given retina fundus image and classify the diabetic retinopathy "
    "(DR) stage. Respond with only one number (0, 1, 2, 3, or 4) according to the "
    "following criteria where 0 is No signs of diabetic retinopathy, 1 is Mild: "
    "Microaneurysms present, 2 is Moderate: More microaneurysms, hemorrhages, or "
    "exudates, 3 is Severe: Significant hemorrhages, venous beading, or intraretinal "
    "microvascular abnormalities (IRMA), 4 is Proliferative: Neovascularization or "
    f"vitreous hemorrhage present and your option are:\n{DIABETIC_RETINOPATHY_OPTIONS} "
    "Return only the number (0–4) that best matches the image."
)
DR_LABEL_IDS = list(range(len(DIABETIC_RETINOPATHY_CLASSES)))
CHAT_TEMPLATE = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": DIABETIC_RETINOPATHY_PROMPT},
        ],
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GenEval LoRA adapters on held-out datasets."
    )
    parser.add_argument("--model_path", required=True, help="Directory containing LoRA adapter.")
    parser.add_argument(
        "--base_model",
        default="google/medgemma-4b",
        help="Base MedGemma checkpoint for loading adapters.",
    )
    parser.add_argument(
        "--test_dataset",
        required=True,
        choices=["aptos", "eyepacs", "messidor1", "messidor2"],
        help="Dataset key inside configs/dataset_config.yaml.",
    )
    parser.add_argument("--test_data_path", required=True, help="Folder with evaluation images.")
    parser.add_argument("--test_csv_path", required=True, help="CSV with eval labels.")
    parser.add_argument(
        "--output_file",
        default=None,
        help=(
            "CSV file that stores predictions with metadata. "
            "Defaults to <dataset-folder-name>.csv if omitted."
        ),
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help="Optional override for dataset config path.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Maximum generated tokens during decoding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation order.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_dataset_config(config_path: Path, dataset_key: str) -> Dict[str, Sequence[str]]:
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    datasets = data.get("datasets", {})
    if dataset_key not in datasets:
        raise ValueError(f"Dataset '{dataset_key}' not present in {config_path}.")
    dataset_cfg = datasets[dataset_key]
    if "class_names" not in dataset_cfg:
        raise ValueError(f"class_names missing for dataset '{dataset_key}'.")
    return dataset_cfg


def find_column(df: pd.DataFrame, candidates: Sequence[str], kind: str) -> str:
    normalized = {column.strip().lower(): column for column in df.columns}
    for col in candidates:
        if col in df.columns:
            return col
        key = col.strip().lower()
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Unable to infer {kind} column. CSV columns: {list(df.columns)}"
    )


def encode_label(raw_value: object, class_names: Sequence[str]) -> int:
    if isinstance(raw_value, (int, float)) and not pd.isna(raw_value):
        idx = int(raw_value)
        if 0 <= idx < len(class_names):
            return idx
    normalized = str(raw_value).strip().lower()
    for idx, name in enumerate(class_names):
        if normalized == name.lower():
            return idx
    raise ValueError(f"Encountered unknown label '{raw_value}'.")


def resolve_path(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def build_records(
    df: pd.DataFrame,
    image_col: str,
    label_col: str,
    image_root: Path,
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        raw_path = str(row[image_col]).strip()
        if not raw_path:
            continue
        img_path = resolve_path(image_root, raw_path)
        if not img_path.exists():
            logging.warning("Skipping missing file %s", img_path)
            continue
        try:
            label_idx = encode_label(row[label_col], class_names)
        except ValueError as err:
            logging.warning("Skipping row due to label error: %s", err)
            continue
        label_name = (
            class_names[label_idx]
            if 0 <= label_idx < len(class_names)
            else str(label_idx)
        )
        records.append(
            {
                "image_path": img_path,
                "label_id": label_idx,
                "label_name": label_name,
            }
        )
    if not records:
        raise RuntimeError("Evaluation dataset is empty after validation.")
    return records


def select_adapter_path(model_path: Path) -> Path:
    if (model_path / "adapter_config.json").exists():
        return model_path
    candidate = model_path / "lora_adapter"
    if (candidate / "adapter_config.json").exists():
        return candidate
    raise FileNotFoundError(
        f"No adapter_config.json found under {model_path}. Pass the directory that contains the LoRA weights."
    )


def decode_prediction(text: str) -> int:
    for char in text:
        if char.isdigit() and char in {"0", "1", "2", "3", "4"}:
            return int(char)
    tokens = text.strip().lower().split()
    str_to_int = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
    }
    for token in tokens:
        if token in str_to_int:
            return str_to_int[token]
    return -1


def infer_dataset_folder_name(data_root: Path) -> str:
    """Infer a friendly dataset name from the evaluation image directory."""
    candidate = data_root.name
    generic_names = {"image", "images", "img", "imgs"}
    if candidate.strip().lower() in generic_names and data_root.parent != data_root:
        candidate = data_root.parent.name
    return candidate or "predictions"


def determine_output_path(explicit_path: str | None, data_root: Path) -> Path:
    if explicit_path:
        return Path(explicit_path)
    dataset_folder = infer_dataset_folder_name(data_root)
    filename = f"{dataset_folder}.csv"
    logging.info(
        "No --output_file provided; saving predictions as %s", filename
    )
    return Path(filename)


def main() -> None:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    config_path = (
        Path(args.config_path)
        if args.config_path
        else project_root / "configs" / "dataset_config.yaml"
    )
    dataset_cfg = load_dataset_config(config_path, args.test_dataset)
    class_names = dataset_cfg["class_names"]

    csv_path = Path(args.test_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    data_root = Path(args.test_data_path)
    if not data_root.exists():
        raise FileNotFoundError(f"Image directory not found: {data_root}")
    output_path = determine_output_path(args.output_file, data_root)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"No rows found in {csv_path}")
    image_col = find_column(df, IMAGE_COLUMN_CANDIDATES, "image")
    label_col = find_column(df, LABEL_COLUMN_CANDIDATES, "label")
    df = df[[image_col, label_col]].dropna()
    records = build_records(df, image_col, label_col, data_root, class_names)
    logging.info("Prepared %d evaluation samples.", len(records))

    adapter_root = select_adapter_path(Path(args.model_path))
    processor_candidate = adapter_root.parent / "preprocessor_config.json"
    if processor_candidate.exists():
        processor_path = adapter_root.parent
    else:
        processor_path = Path(args.base_model)
    if not processor_path.exists():
        raise FileNotFoundError(
            f"Processor path {processor_path} does not exist. "
            "Supply --base_model pointing to a local checkpoint."
        )
    logging.info("Loading processor from %s", processor_path)
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    prompt_text = processor.apply_chat_template(
        CHAT_TEMPLATE,
        add_generation_prompt=True,
        tokenize=False,
    )

    base_model_path = Path(args.base_model)
    if not base_model_path.exists():
        raise FileNotFoundError(
            f"Base model path {base_model_path} does not exist. "
            "Ensure --base_model points to the MedGemma checkpoint."
        )
    logging.info("Loading base model %s", base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        )
    model = PeftModel.from_pretrained(base_model, adapter_root)
    model.eval()

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None and not hasattr(processor, "batch_decode"):
        raise ValueError("Processor must expose either batch_decode or tokenizer.")

    results: List[Dict[str, object]] = []
    device = next(model.parameters()).device
    for record in tqdm(records, desc="Evaluating", unit="image"):
        image_path = record["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as err:
            logging.warning("Failed to open %s: %s", image_path, err)
            results.append(
                {
                    "image_path": str(image_path),
                    "ground_truth": record["label_id"],
                    "ground_truth_name": record["label_name"],
                    "predicted_label": -1,
                    "predicted_label_name": "invalid",
                    "raw_output": f"ERROR: {err}",
                }
            )
            continue

        try:
            inputs = processor(
                text=prompt_text,
                images=[image],
                return_tensors="pt",
            )
        except Exception as err:
            logging.warning("Processor failed on %s: %s", image_path, err)
            results.append(
                {
                    "image_path": str(image_path),
                    "ground_truth": record["label_id"],
                    "ground_truth_name": record["label_name"],
                    "predicted_label": -1,
                    "predicted_label_name": "invalid",
                    "raw_output": f"ERROR: {err}",
                }
            )
            continue

        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            generation = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )
        new_tokens = generation[:, inputs["input_ids"].shape[1] :]
        if hasattr(processor, "batch_decode"):
            decoded_text = processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0].strip()
        else:
            decoded_text = tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )[0].strip()
        pred_label = decode_prediction(decoded_text)
        pred_name = (
            DIABETIC_RETINOPATHY_CLASSES[pred_label]
            if pred_label in DR_LABEL_IDS
            else "invalid"
        )
        results.append(
            {
                "image_path": str(image_path),
                "ground_truth": record["label_id"],
                "predicted_label": pred_label,
                "raw_output": decoded_text,
            }
        )

    output_parent = output_path.parent
    if output_parent and not output_parent.exists():
        output_parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logging.info("Saved predictions to %s", output_path.resolve())

    valid_df = results_df[results_df["predicted_label"] >= 0]
    if not valid_df.empty:
        y_true = valid_df["ground_truth"].astype(int)
        y_pred = valid_df["predicted_label"].astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        precision = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=DR_LABEL_IDS).tolist()
    else:
        logging.warning("No valid predictions were generated; metrics set to zero.")
        acc = f1 = precision = recall = 0.0
        cm = []

    metrics = {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "confusion_matrix": cm,
        "total_samples": len(results_df),
        "valid_predictions": int(valid_df.shape[0]),
    }
    logging.info(
        "Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
        acc,
        precision,
        recall,
        f1,
    )
    logging.info("Confusion matrix (labels %s): %s", DR_LABEL_IDS, cm)

    metrics_path = output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    logging.info("Metrics stored in %s", metrics_path.resolve())


if __name__ == "__main__":
    main()
