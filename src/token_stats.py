#Per-tokenizer token-length stats -> dataset_summary.json. Drives max_length.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_SUMMARY = REPO_ROOT / "results" / "dataset_summary.json"

#The two tokenizers used
TOKENIZERS = {
    "xlm-roberta-base": "xlm-roberta-base",
    "bcms-bertic": "classla/bcms-bertic",
}


def _length_stats(lengths: np.ndarray) -> dict:
    #Percentiles + buckets for the three max_length candidates (128/256/512).
    return {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p95": float(np.quantile(lengths, 0.95)),
        "p99": float(np.quantile(lengths, 0.99)),
        "max": int(lengths.max()),
        "n_over_128": int((lengths > 128).sum()),
        "n_over_256": int((lengths > 256).sum()),
        "n_over_512": int((lengths > 512).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Add token-length stats to dataset_summary.json")
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = ap.parse_args()

    #Load existing summary.
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    #For each tokenizer, measure true token counts on every split.
    for alias, hf_name in TOKENIZERS.items():
        print(f"Loading tokenizer: {hf_name}")
        tok = AutoTokenizer.from_pretrained(hf_name)
        for split_name in ("train", "val", "test"):
            df = pd.read_csv(args.processed_dir / f"{split_name}.csv")
            encs = tok(df["text"].tolist(), add_special_tokens=True,
                       padding=False, truncation=False)
            lengths = np.array([len(ids) for ids in encs["input_ids"]])
            summary["splits"][split_name].setdefault("token_lens", {})
            summary["splits"][split_name]["token_lens"][alias] = _length_stats(lengths)
            print(f"  {split_name:<5}  n={len(lengths)}  "
                  f"p95={np.quantile(lengths, 0.95):.0f}  "
                  f"max={lengths.max()}")

    #Overwrite the summary with the enriched version.
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nUpdated {args.summary}")


if __name__ == "__main__":
    main()
