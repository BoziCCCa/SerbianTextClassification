#Load SentiComments.SR, map to 3-class polarity, write stratified splits.
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

#Allow both `python -m src.data` and `python src/data.py`.
try:
    from .text_utils import is_cyrillic, transliterate
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.text_utils import is_cyrillic, transliterate

#Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "SentiComments.SR.corr.txt"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

#Split 
RANDOM_SEED = 50
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10

#6 raw labels + sarcasm variants collapse to 3-way polarity.
#Sign is polarity (+ positive, - negative). Letter is intensity/subjectivity:
#   1  = clear, unambiguous sentiment
#   M  = mixed (both signs present; dot sign shows which dominates)
#   NS = no sentiment (objective statement; sign shows slight tilt)
#The optional `s` suffix marks sarcasm — real sentiment flips for 1/M
LABEL_TO_POLARITY: dict[str, str] = {
    "+1": "positive", "+M": "positive",
    "-1": "negative", "-M": "negative",
    "+NS": "neutral", "-NS": "neutral",
    "+1s": "negative", "+Ms": "negative",
    "-1s": "positive", "-Ms": "positive",
    "+NSs": "neutral", "-NSs": "neutral",
}

#Stable integer encoding for the Trainer.
POLARITY_TO_ID: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_POLARITY: dict[int, str] = {v: k for k, v in POLARITY_TO_ID.items()}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label_raw", "comment_id", "text"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    #Drop empty text rows after stripping whitespace.
    df = df.copy()
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 0]

    #Drop unexpected labels
    unknown = sorted(set(df["label_raw"]) - set(LABEL_TO_POLARITY))
    if unknown:
        print(f"WARN: dropping {len(df[df['label_raw'].isin(unknown)])} "
              f"rows with unexpected labels: {unknown}")
        df = df[~df["label_raw"].isin(unknown)]

    #Map raw label to polarity string to integer id.
    df["polarity"] = df["label_raw"].map(LABEL_TO_POLARITY)
    df["label"] = df["polarity"].map(POLARITY_TO_ID)

    #Transliterate Cyrillic rows 
    df["was_cyrillic"] = df["text"].map(is_cyrillic)
    df["text"] = df["text"].map(transliterate)

    #Drop exact duplicates on (text, label).
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    after = len(df)
    print(f"Rows: {before} raw -> {after} after cleanup "
          f"(dropped {before - after}).")
    return df


def stratified_split(df: pd.DataFrame, seed: int = RANDOM_SEED
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #train, val, test split
    train_val, test = train_test_split(
        df, test_size=TEST_FRACTION, stratify=df["label"], random_state=seed,
    )
    val_rel = VAL_FRACTION / (1 - TEST_FRACTION)
    train, val = train_test_split(
        train_val, test_size=val_rel, stratify=train_val["label"], random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _class_counts(df: pd.DataFrame) -> dict[str, int]:
    #Stable key order regardless of what's present in the slice
    counts = Counter(df["polarity"])
    return {k: int(counts.get(k, 0)) for k in POLARITY_TO_ID}


def summarize(splits: dict[str, pd.DataFrame]) -> dict:
    #Everything that drives downstream decisions gets dumped to JSON for the notebook and report.
    out = {
        "random_seed": RANDOM_SEED,
        "val_fraction": VAL_FRACTION,
        "test_fraction": TEST_FRACTION,
        "label_map": {k: v for k, v in POLARITY_TO_ID.items()},
        "splits": {},
    }
    for name, df in splits.items():
        out["splits"][name] = {
            "n_examples": len(df),
            "class_counts": _class_counts(df),
            "char_len": {
                "mean": float(df["text"].str.len().mean()),
                "median": float(df["text"].str.len().median()),
                "p95": float(df["text"].str.len().quantile(0.95)),
                "max": int(df["text"].str.len().max()),
            },
            "cyrillic_rows": int(df["was_cyrillic"].sum()),
        }
    return out


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    #Only columns downstream code (train.py, compare.py) actually reads.
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["comment_id", "text", "label", "polarity"]
    for name, df in splits.items():
        path = out_dir / f"{name}.csv"
        df[keep].to_csv(path, index=False, encoding="utf-8")
        print(f"Wrote {path}  ({len(df)} rows)")


def main() -> None:
    #CLI
    ap = argparse.ArgumentParser(
        description="Load SentiComments.SR, map labels to 3-class polarity, "
                    "write stratified train/val/test CSVs."
    )
    ap.add_argument("--raw", type=Path, default=RAW_PATH)
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--summary", type=Path,
                    default=RESULTS_DIR / "dataset_summary.json")
    args = ap.parse_args()

    #Pipeline: read -> clean -> split -> write CSVs -> write summary JSON.
    df = load_raw(args.raw)
    df = normalize(df)
    train, val, test = stratified_split(df)
    splits = {"train": train, "val": val, "test": test}

    save_splits(splits, args.out_dir)

    summary = summarize(splits)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"\nSummary -> {args.summary}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
