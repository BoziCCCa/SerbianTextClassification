#TF-IDF + Logistic Regression baseline on SentiComments.SR polarity.
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

#Small grid; val macro-F1 picks the winner.
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
LABEL_NAMES = ["negative", "neutral", "positive"]


def _load():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train, val, test


def main() -> None:
    #Load splits.
    train, val, test = _load()
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    #Vectorize: word 1-2 grams, drop rare/near-universal tokens,
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train["text"])
    X_val = vectorizer.transform(val["text"])
    X_test = vectorizer.transform(test["text"])
    print(f"vocab size: {len(vectorizer.vocabulary_)}")

    y_train = train["label"].to_numpy()
    y_val = val["label"].to_numpy()
    y_test = test["label"].to_numpy()

    #Grid-search C on validation
    val_scores = {}
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        val_scores[c] = f1_score(y_val, pred, average="macro")
        print(f"  C={c:>5}  val_f1_macro={val_scores[c]:.4f}")

    best_c = max(val_scores, key=val_scores.get)
    print(f"\nbest C on val: {best_c}  (val_f1_macro={val_scores[best_c]:.4f})")

    #Final fit at best C
    t0 = time.perf_counter()
    clf = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0

    pred_test = clf.predict(X_test)
    prob_test = clf.predict_proba(X_test)

    #Metrics JSON
    metrics = {
        "model": "tfidf+logreg",
        "best_C": best_c,
        "val_f1_macro_grid": {str(k): float(v) for k, v in val_scores.items()},
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "test_f1_macro": float(f1_score(y_test, pred_test, average="macro")),
        "per_class": classification_report(
            y_test, pred_test, target_names=LABEL_NAMES,
            output_dict=True, zero_division=0,
        ),
        "train_time_s": train_time_s,
        "vocab_size": len(vectorizer.vocabulary_),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baseline.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    #Per-example predictions feed the error-analysis section in the notebook.
    preds_df = test[["comment_id", "text", "label", "polarity"]].copy()
    preds_df["pred_label"] = pred_test
    preds_df["pred_polarity"] = [LABEL_NAMES[i] for i in pred_test]
    preds_df["confidence"] = prob_test.max(axis=1)
    preds_df.to_csv(RESULTS_DIR / "baseline_test_predictions.csv",
                    index=False, encoding="utf-8")

    #Summary
    print(f"\ntest accuracy : {metrics['test_accuracy']:.4f}")
    print(f"test f1_macro : {metrics['test_f1_macro']:.4f}")
    print("\nclassification report:")
    print(classification_report(y_test, pred_test, target_names=LABEL_NAMES,
                                zero_division=0))
    print(f"wrote {RESULTS_DIR / 'baseline.json'}")
    print(f"wrote {RESULTS_DIR / 'baseline_test_predictions.csv'}")


if __name__ == "__main__":
    main()
