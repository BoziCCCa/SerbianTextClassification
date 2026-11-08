#Full experiment: LR search + seeded finals + summary. Resumable.
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.train import REPO_ROOT, load_config, output_dir_for, run

#Models we sweep over.
MODEL_CONFIGS = {
    "xlmr": REPO_ROOT / "configs" / "xlmr.yaml",
    "bertic": REPO_ROOT / "configs" / "bertic.yaml",
}

#Phase 1: train every (model, lr) at this one seed to pick the best LR.
LR_GRID = [2e-5, 3e-5, 5e-5]
SEARCH_SEED = 42

#Phase 2: at the chosen LR, train these seeds for mean +- std.
FINAL_SEEDS = [42, 123, 2024]

SUMMARY_PATH = REPO_ROOT / "results" / "summary.json"


#--- Slot helpers ---

def _run_slot(model_cfg_path: Path, lr: float, seed: int, *, force: bool) -> Path:
    #Resumability: skip if this slot already finished.
    cfg = load_config(model_cfg_path, {"learning_rate": lr, "seed": seed})
    out = output_dir_for(cfg)
    metrics_path = out / "metrics.json"
    if metrics_path.exists() and not force:
        print(f"[skip] {metrics_path.relative_to(REPO_ROOT)} exists")
        return out
    print(f"\n==> train {cfg['run_tag']}  lr={lr}  seed={seed}")
    run(cfg, out)
    return out


def _best_val_f1(metrics_path: Path) -> float:
    #Highest val macro-F1 across epochs for this slot.
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return max(
        (e["eval_f1_macro"] for e in data["val_history"] if "eval_f1_macro" in e),
        default=0.0,
    )


#--- Phase 1: LR search ---

def phase_lr_search(force: bool) -> dict[str, float]:
    #Train every (model, lr) at SEARCH_SEED.
    for name, path in MODEL_CONFIGS.items():
        for lr in LR_GRID:
            _run_slot(path, lr, SEARCH_SEED, force=force)

    #Pick the LR with the best val macro-F1 for each model.
    best_lrs: dict[str, float] = {}
    print("\nLR search results (val macro-F1)")
    for name, path in MODEL_CONFIGS.items():
        cfg = load_config(path, {"learning_rate": LR_GRID[0], "seed": SEARCH_SEED})
        run_dir = REPO_ROOT / cfg["output_root"] / cfg["run_tag"]
        scored = []
        for lr in LR_GRID:
            m = run_dir / f"lr{lr}_seed{SEARCH_SEED}" / "metrics.json"
            scored.append((lr, _best_val_f1(m)))
        scored.sort(key=lambda x: -x[1])
        for lr, f1 in scored:
            print(f"  {name:<7} lr={lr:<7}  val_f1={f1:.4f}")
        best_lrs[name] = scored[0][0]
        print(f"  -> best for {name}: lr={best_lrs[name]}")
    return best_lrs


#--- Phase 2: final seeded runs ---

def phase_finals(best_lrs: dict[str, float], force: bool) -> None:
    #Reuses the SEARCH_SEED run when seed=42 is in FINAL_SEEDS (skip-check).
    for name, path in MODEL_CONFIGS.items():
        for seed in FINAL_SEEDS:
            _run_slot(path, best_lrs[name], seed, force=force)


#--- Phase 3: aggregate into results/summary.json ---

def _mean_std(xs: list[float]) -> tuple[float, float]:
    #statistics.stdev crashes on n<2; return 0.
    if len(xs) < 2:
        return (xs[0] if xs else 0.0), 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def build_summary(best_lrs: dict[str, float]) -> dict:
    summary = {"baseline": None, "models": {}, "best_lrs": best_lrs}

    #Baseline (if baseline.py has been run).
    baseline_path = REPO_ROOT / "results" / "baseline.json"
    if baseline_path.exists():
        bl = json.loads(baseline_path.read_text(encoding="utf-8"))
        summary["baseline"] = {
            "name": bl["model"],
            "test_accuracy": bl["test_accuracy"],
            "test_f1_macro": bl["test_f1_macro"],
        }

    #For each model, collect test metrics across all final seeds -> mean +- std.
    for name, path in MODEL_CONFIGS.items():
        cfg = load_config(path, {"learning_rate": best_lrs[name], "seed": FINAL_SEEDS[0]})
        run_dir = REPO_ROOT / cfg["output_root"] / cfg["run_tag"]
        accs, f1s, times = [], [], []
        per_seed = []
        n_params = None
        for seed in FINAL_SEEDS:
            m_path = run_dir / f"lr{best_lrs[name]}_seed{seed}" / "metrics.json"
            if not m_path.exists():
                print(f"[warn] missing {m_path.relative_to(REPO_ROOT)}")
                continue
            m = json.loads(m_path.read_text(encoding="utf-8"))
            accs.append(m["test_accuracy"])
            f1s.append(m["test_f1_macro"])
            times.append(m["train_time_s"])
            n_params = m["n_params"]
            per_seed.append({
                "seed": seed,
                "test_accuracy": m["test_accuracy"],
                "test_f1_macro": m["test_f1_macro"],
            })

        acc_mean, acc_std = _mean_std(accs)
        f1_mean, f1_std = _mean_std(f1s)
        time_mean, _ = _mean_std(times)
        summary["models"][name] = {
            "model_name": cfg["model_name"],
            "best_lr": best_lrs[name],
            "n_params": n_params,
            "train_time_s_mean": time_mean,
            "test_accuracy": {"mean": acc_mean, "std": acc_std},
            "test_f1_macro": {"mean": f1_mean, "std": f1_std},
            "per_seed": per_seed,
        }

    #Dump — this is the one artifact the notebook + report read from.
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def print_summary(summary: dict) -> None:
    #Human-readable print of the final numbers.
    print("\nFinal results (test set, mean +- std over seeds)")
    if summary["baseline"]:
        bl = summary["baseline"]
        print(f"  {bl['name']:<22}  acc {bl['test_accuracy']:.4f}  "
              f"f1_macro {bl['test_f1_macro']:.4f}")
    for name, s in summary["models"].items():
        acc = s["test_accuracy"]
        f1 = s["test_f1_macro"]
        print(f"  {s['model_name']:<22}  "
              f"acc {acc['mean']:.4f} +- {acc['std']:.4f}  "
              f"f1_macro {f1['mean']:.4f} +- {f1['std']:.4f}  "
              f"(n_params {s['n_params']/1e6:.0f}M, lr {s['best_lr']}, "
              f"train ~{s['train_time_s_mean']:.0f}s)")


#--- CLI ---

def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full experiment sweep.")
    ap.add_argument("--force", action="store_true",
                    help="re-run every slot even if metrics.json exists")
    ap.add_argument("--only-summary", action="store_true",
                    help="skip training, rebuild results/summary.json from existing runs")
    args = ap.parse_args()

    #--only-summary: infer best LR from whatever LR-search slots exist on disk.
    if args.only_summary:
        best_lrs = {}
        for name, path in MODEL_CONFIGS.items():
            cfg = load_config(path, {"learning_rate": LR_GRID[0], "seed": SEARCH_SEED})
            run_dir = REPO_ROOT / cfg["output_root"] / cfg["run_tag"]
            scored = []
            for lr in LR_GRID:
                m = run_dir / f"lr{lr}_seed{SEARCH_SEED}" / "metrics.json"
                if m.exists():
                    scored.append((lr, _best_val_f1(m)))
            if scored:
                scored.sort(key=lambda x: -x[1])
                best_lrs[name] = scored[0][0]
        summary = build_summary(best_lrs)
        print_summary(summary)
        return

    #Normal path: LR search -> finals -> summary.
    best_lrs = phase_lr_search(args.force)
    phase_finals(best_lrs, args.force)
    summary = build_summary(best_lrs)
    print_summary(summary)


if __name__ == "__main__":
    main()
