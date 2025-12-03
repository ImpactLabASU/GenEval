#!/usr/bin/env python
"""Fine-tune MedGemma-4B with LoRA adapters on diabetic retinopathy datasets."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
import yaml
from peft import LoraConfig, get_peft_model
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune MedGemma-4B on diabetic retinopathy datasets with LoRA adapters."
    )
    parser.add_argument(
        "--model_name",
        default="google/medgemma-4b",
        help="Base multi-modal checkpoint on Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset",
        choices=["aptos", "eyepacs", "messidor1", "messidor2"],
        help="Dataset key defined inside configs/dataset_config.yaml. Use --datasets for multi-domain training.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["aptos", "eyepacs", "messidor1", "messidor2"],
        help="Optional list of dataset keys to combine.",
    )
    parser.add_argument(
        "--data_path",
        help="Directory containing image files for --dataset.",
    )
    parser.add_argument(
        "--data_paths",
        nargs="+",
        help="Directories containing images for each dataset in --datasets.",
    )
    parser.add_argument(
        "--csv_path",
        help="CSV file with image relative paths and labels for --dataset.",
    )
    parser.add_argument(
        "--csv_paths",
        nargs="+",
        help="CSV label files aligned with --datasets.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory used to store LoRA adapters and logs.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Peak learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenized prompts.",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help="Optional override for dataset config path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting and training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Linear warmup ratio.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Trainer logging interval.",
    )
    parser.add_argument(
        "--train_subset",
        type=int,
        default=None,
        help="Limit the number of training samples (useful for smoke tests).",
    )
    parser.add_argument(
        "--val_subset",
        type=int,
        default=None,
        help="Limit the number of validation samples.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override total optimizer steps for quick validations.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def load_dataset_config(config_path: Path, dataset_key: str) -> Dict[str, Sequence[str]]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    datasets = config.get("datasets", {})
    if dataset_key not in datasets:
        raise ValueError(f"Dataset '{dataset_key}' not defined in {config_path}.")
    dataset_cfg = datasets[dataset_key]
    class_names = dataset_cfg.get("class_names")
    if not class_names:
        raise ValueError(
            f"Missing class_names for dataset '{dataset_key}' in {config_path}."
        )
    return dataset_cfg


def find_column(df: pd.DataFrame, candidates: Sequence[str], kind: str) -> str:
    normalized = {col.strip().lower(): col for col in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        key = name.strip().lower()
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Could not infer {kind} column from CSV. Available columns: {list(df.columns)}"
    )


def encode_label(raw_value: object, class_names: Sequence[str]) -> int:
    if isinstance(raw_value, (int, float)) and not pd.isna(raw_value):
        idx = int(raw_value)
        if 0 <= idx < len(class_names):
            return idx
    cleaned = str(raw_value).strip().lower()
    for idx, name in enumerate(class_names):
        if cleaned == name.lower():
            return idx
    raise ValueError(f"Unknown label '{raw_value}'. Expected one of {class_names}.")


def resolve_image_path(image_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return image_root / candidate


def build_records(
    df: pd.DataFrame,
    image_col: str,
    label_col: str,
    image_root: Path,
    class_names: Sequence[str],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    missing_files: List[str] = []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Validating samples", unit="sample"
    ):
        raw_image = str(row[image_col]).strip()
        if not raw_image:
            continue
        img_path = resolve_image_path(image_root, raw_image)
        if not img_path.exists():
            missing_files.append(str(img_path))
            continue
        try:
            label_idx = encode_label(row[label_col], class_names)
        except ValueError as err:
            logging.warning("Skipping sample %s due to label error: %s", raw_image, err)
            continue
        if 0 <= label_idx < len(DIABETIC_RETINOPATHY_CLASSES):
            label_name = DIABETIC_RETINOPATHY_CLASSES[label_idx]
        else:
            label_name = str(label_idx)
        records.append(
            {
                "image_path": img_path,
                "label_id": label_idx,
                "label_name": label_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": DIABETIC_RETINOPATHY_PROMPT},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": label_name},
                        ],
                    },
                ],
            }
        )
    if missing_files:
        logging.warning(
            "Skipped %d missing files. First missing: %s",
            len(missing_files),
            missing_files[0],
        )
    if not records:
        raise RuntimeError("No valid samples found. Check your CSV/image paths.")
    return records


def normalize_source_specs(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    dataset_keys: List[str] = list(args.datasets) if args.datasets else []
    if args.dataset:
        dataset_keys.append(args.dataset)
    data_paths: List[str] = list(args.data_paths) if args.data_paths else []
    if args.data_path:
        data_paths.append(args.data_path)
    csv_paths: List[str] = list(args.csv_paths) if args.csv_paths else []
    if args.csv_path:
        csv_paths.append(args.csv_path)

    if not dataset_keys:
        raise ValueError("Provide at least one dataset via --dataset or --datasets.")
    if not data_paths or not csv_paths:
        raise ValueError("Provide matching --data_path(s) and --csv_path(s) for every dataset.")
    if not (
        len(dataset_keys) == len(data_paths) == len(csv_paths)
    ):
        raise ValueError(
            "Mismatch between datasets, data paths, and csv paths counts "
            f"({len(dataset_keys)}, {len(data_paths)}, {len(csv_paths)})."
        )
    return list(zip(dataset_keys, data_paths, csv_paths))


def stratified_split(
    records: List[Dict[str, object]], seed: int
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    labels = [rec["label_id"] for rec in records]
    try:
        train_records, val_records = train_test_split(
            records,
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        logging.warning("Stratified split failed; falling back to random split.")
        train_records, val_records = train_test_split(
            records,
            test_size=0.2,
            random_state=seed,
        )
    return train_records, val_records


class GenEvalDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict[str, object]],
        processor: AutoProcessor,
        max_length: int,
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self.records[index]
        image = Image.open(entry["image_path"]).convert("RGB")
        chat_text = self.processor.apply_chat_template(
            entry["messages"], add_generation_prompt=False, tokenize=False
        )
        encoding = self.processor(
            text=chat_text,
            images=[image],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        labels = encoding["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100
        encoding["labels"] = labels
        return encoding


@dataclass
class MultimodalDataCollator:
    pad_token_id: int

    def __call__(
        self, features: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        batch = {}
        for key in features[0].keys():
            batch[key] = torch.stack([f[key] for f in features])
        return batch


class MetricsLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logging.info("Trainer metrics: %s", json.dumps(logs, indent=2))


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = (
        Path(args.config_path)
        if args.config_path
        else project_root / "configs" / "dataset_config.yaml"
    )
    output_dir = Path(args.output_dir)
    configure_logging(output_dir)
    set_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_name)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    try:
        source_specs = normalize_source_specs(args)
    except ValueError as err:
        logging.error("Source specification error: %s", err)
        raise SystemExit(1) from err

    combined_records: List[Dict[str, object]] = []
    dataset_stats: Dict[str, int] = {}
    for dataset_key, data_dir_str, csv_file_str in source_specs:
        dataset_cfg = load_dataset_config(config_path, dataset_key)
        class_names = dataset_cfg["class_names"]
        logging.info(
            "Preparing dataset '%s' with %d classes.", dataset_key, len(class_names)
        )

        csv_path = Path(csv_file_str)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        data_path = Path(data_dir_str)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
            raise RuntimeError(f"CSV file {csv_path} is empty.")
        image_col = find_column(df, IMAGE_COLUMN_CANDIDATES, "image path")
        label_col = find_column(df, LABEL_COLUMN_CANDIDATES, "label")
        df = df[[image_col, label_col]].dropna()

        dataset_records = build_records(
            df,
            image_col,
            label_col,
            data_path,
            class_names,
        )
        for record in dataset_records:
            record["dataset"] = dataset_key
        dataset_stats[dataset_key] = len(dataset_records)
        combined_records.extend(dataset_records)
        logging.info(
            "Added %d samples from dataset '%s'.",
            len(dataset_records),
            dataset_key,
        )

    if not combined_records:
        raise RuntimeError("No samples collected from the provided datasets.")
    if len(dataset_stats) > 1:
        logging.info("Combined multi-domain summary:")
        for name, count in dataset_stats.items():
            logging.info("  %s: %d samples", name, count)

    records = combined_records
    train_records, val_records = stratified_split(records, args.seed)
    logging.info(
        "Train samples: %d | Val samples: %d", len(train_records), len(val_records)
    )

    def apply_subset(
        subset_value: int | None, data: List[Dict[str, object]], split_name: str
    ) -> List[Dict[str, object]]:
        if subset_value is None or subset_value <= 0:
            return data
        limited = data[: min(subset_value, len(data))]
        logging.info(
            "Applying subset for %s split: %d samples (requested %d).",
            split_name,
            len(limited),
            subset_value,
        )
        return limited

    train_records = apply_subset(args.train_subset, train_records, "train")
    val_records = apply_subset(args.val_subset, val_records, "validation")

    train_dataset = GenEvalDataset(train_records, processor, args.max_length)
    val_dataset = GenEvalDataset(val_records, processor, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "logs"),
        load_best_model_at_end=True,
    )

    data_collator = MultimodalDataCollator(
        pad_token_id=processor.tokenizer.pad_token_id
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    trainer.add_callback(MetricsLoggerCallback())

    logging.info("Starting training with %d epochs.", args.num_epochs)
    train_result = trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    metrics = trainer.evaluate()
    metrics.update(train_result.metrics)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    logging.info("Saved metrics to %s", metrics_path)

    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    logging.info(
        "LoRA adapter saved to %s (adapter_model.safetensors & adapter_config.json).",
        adapter_dir,
    )


if __name__ == "__main__":
    main()
