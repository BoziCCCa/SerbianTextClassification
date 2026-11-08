#HuggingFace Trainer wrapper; invoked slot-by-slot by run_all.py.
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

LABEL_NAMES = ["negative", "neutral", "positive"]
NUM_LABELS = 3


#Config

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(model_config: Path, overrides: dict | None = None) -> dict:
    #Load a specific config
    cfg = _load_yaml(CONFIG_DIR / "base.yaml")
    cfg.update(_load_yaml(model_config))
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def output_dir_for(cfg: dict) -> Path:
    return (
        REPO_ROOT / cfg["output_root"] / cfg["run_tag"]
        / f"lr{cfg['learning_rate']}_seed{cfg['seed']}"
    )


#Data
def _load_splits(cfg: dict, tokenizer):
    #Read CSVs from disk.
    def _read(csv_path: str) -> pd.DataFrame:
        return pd.read_csv(REPO_ROOT / csv_path)

    train_df = _read(cfg["train_csv"])
    val_df = _read(cfg["val_csv"])
    test_df = _read(cfg["test_csv"])

    #Pandas -> HF Dataset -> tokenized
    #Raw `text` column is dropped (the Trainer only needs the token IDs)
    def _to_hf(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
        return ds.map(
            lambda ex: tokenizer(ex["text"], truncation=True, max_length=cfg["max_length"]),
            batched=True,
            remove_columns=["text"],
        )

    return _to_hf(train_df), _to_hf(val_df), _to_hf(test_df), test_df


#Metrics

def _compute_metrics(eval_pred):
    #Called for every eval. Accuracy + macro-F1 for imbalanced labels.
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


#Train one slot

def run(cfg: dict, out_dir: Path) -> dict:
    #Seed Python/numpy/torch for reproducibility.
    set_seed(cfg["seed"])

    #Tokenizer + pretrained model with a fresh 3-way classification head.
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=NUM_LABELS,
        id2label={i: n for i, n in enumerate(LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(LABEL_NAMES)},
    )

    #Datasets + dynamic-padding collator
    train_ds, val_ds, test_ds, test_df = _load_splits(cfg, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    #Trainer hyperparameters.
    args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        label_smoothing_factor=cfg["label_smoothing_factor"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        save_total_limit=cfg["save_total_limit"],
        bf16=cfg["bf16"],
        fp16=cfg["fp16"],
        seed=cfg["seed"],
        logging_strategy="epoch",
        report_to="none",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg["early_stopping_patience"]
        )],
    )

    #Train
    t0 = time.perf_counter()
    trainer.train()
    train_time_s = time.perf_counter() - t0

    #Test eval on the (restored) best checkpoint.
    pred_out = trainer.predict(test_ds)
    logits = pred_out.predictions
    y_true = pred_out.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    y_pred = probs.argmax(axis=-1)

    #Collect everything the notebook/report need.
    metrics = {
        "model_name": cfg["model_name"],
        "run_tag": cfg["run_tag"],
        "seed": cfg["seed"],
        "learning_rate": cfg["learning_rate"],
        "num_train_epochs": cfg["num_train_epochs"],
        "max_length": cfg["max_length"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "train_time_s": train_time_s,
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": classification_report(
            y_true, y_pred, target_names=LABEL_NAMES,
            output_dict=True, zero_division=0,
        ),
        "val_history": trainer.state.log_history,
    }

    #Write metrics.json
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    #Per-example test predictions for error analysis.
    preds_df = test_df[["comment_id", "text", "label", "polarity"]].copy()
    preds_df["pred_label"] = y_pred
    preds_df["pred_polarity"] = [LABEL_NAMES[i] for i in y_pred]
    preds_df["confidence"] = probs.max(axis=-1)
    preds_df.to_csv(out_dir / "test_predictions.csv",
                    index=False, encoding="utf-8")

    #Summary for the log.
    print(f"\n[{cfg['run_tag']} lr={cfg['learning_rate']} seed={cfg['seed']}]")
    print(f"  train_time    : {train_time_s:.1f}s")
    print(f"  test accuracy : {metrics['test_accuracy']:.4f}")
    print(f"  test f1_macro : {metrics['test_f1_macro']:.4f}")
    return metrics
