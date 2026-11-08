# Serbian Text Classification

University NLP assignment: fine-tune one multilingual model and one
regional (BCMS) model on a Serbian sentiment dataset and compare them.

* Prose write-up with full discussion: [`report.md`](report.md)
* Plots, tables, error analysis: [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) (pre-executed)

**Headline result** on 346 held-out test examples (3-seed mean ± std):

| Model | Params | Test acc | Test macro-F1 |
|---|---:|---:|---:|
| tfidf + logreg (baseline) | ~7 k | 0.659 | 0.631 |
| xlm-roberta-base | 278 M | 0.805 ± 0.007 | 0.785 ± 0.008 |
| **classla/bcms-bertic** | **111 M** | **0.832 ± 0.013** | **0.814 ± 0.013** |

## Hardware this is set up for

* **GPU:** AMD Radeon RX 9060 XT (16 GB VRAM, RDNA 4)
* **OS:** Windows 11
* **PyTorch:** `2.9.1+rocm7.2.1` — AMD's Windows ROCm wheel, exposes the
  GPU through the regular `torch.cuda` API. Code falls back to CPU
  automatically if no GPU is visible.

The ROCm wheels do not live on PyPI, so **PyTorch is installed
separately** from the AMD repo — see Step 2 below. Everything else goes
through regular `pip install -r requirements.txt`.

## Where the environment lives

Because `C:` on this machine is almost full, the conda env is installed
into the project directory itself as `.conda/` (via `--prefix`). The pip
download cache and build temp are redirected to `.pip-cache/` and
`.tmp/` for the same reason. All three are listed in `.gitignore`.

Nothing in this repo requires conda specifically — a plain `venv` on
Python 3.12 would work. Conda just happens to be what is installed here.

## Setup

### Step 1 — create the Python environment

```powershell
# From the repo root
$env:CONDA_PKGS_DIRS = 'D:\conda-pkgs'   # keep package cache off C:
conda create --prefix .\.conda python=3.12 -y
```

### Step 2 — install PyTorch (ROCm 7.2.1, Windows)

PyTorch pulls three extra `rocm` meta-packages that only exist on
`repo.radeon.com`, so all seven URLs have to be passed on one command
line.

```powershell
# Redirect pip cache + temp off C: so the 2+ GB download does not fill it
$env:PIP_CACHE_DIR = (Resolve-Path .\.pip-cache)
$env:TMP = (Resolve-Path .\.tmp); $env:TEMP = $env:TMP; $env:TMPDIR = $env:TMP

.\.conda\python.exe -m pip install `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" `
  "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
```

### Step 3 — everything else

```powershell
.\.conda\python.exe -m pip install -r requirements.txt
```

### Step 4 — verify the GPU

```powershell
.\.conda\python.exe -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected: `AMD Radeon RX 9060 XT`. If it prints `CPU`, the ROCm install
didn't take — redo Step 2 with all seven URLs on one line.

### On a different machine

On an NVIDIA box or a CPU-only machine, replace Step 2 with the normal
PyTorch install from <https://pytorch.org/get-started/locally/>. The
rest of the setup and all code is identical — `torch.cuda.is_available()`
returns the right answer in both cases.

## Reproducing the results from scratch

HF model downloads land on D: instead of C:/user cache:

```powershell
$env:HF_HOME = (Resolve-Path .).Path + '\hf_cache'
```

Then, in order:

```powershell
.\.conda\python.exe -m src.data           # read raw -> stratified CSVs + dataset_summary.json
.\.conda\python.exe -m src.token_stats    # adds per-tokenizer length stats to the summary
.\.conda\python.exe -m src.baseline       # TF-IDF + LogReg -> results/baseline.json
.\.conda\python.exe run_all.py            # LR search + 3-seed finals -> results/summary.json
.\.conda\python.exe scripts\build_notebook.py
.\.conda\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\analysis.ipynb
```

Total runtime on this GPU: ~25 minutes, almost all in `run_all.py`.
Everything is resumable — if a training slot already has a `metrics.json`,
it's skipped. Pass `--force` to `run_all.py` to redo everything.

## Project layout

```
configs/
  base.yaml          shared hyperparameters
  xlmr.yaml          xlm-roberta-base override
  bertic.yaml        classla/bcms-bertic override

data/
  raw/               SentiComments.SR.*.txt (download; ~830 KB)
  processed/         train.csv, val.csv, test.csv (stratified, seed 50)

src/
  text_utils.py      Cyrillic -> Latin transliteration
  data.py            raw -> 3-class polarity, stratified 80/10/10 CSVs
  token_stats.py     per-tokenizer token-length stats -> summary.json
  baseline.py        TF-IDF + logistic regression (reference point)
  train.py           HF Trainer wrapper (one run per call)

run_all.py           orchestrates LR search + 3-seed finals + summary

scripts/
  build_notebook.py  generates notebooks/analysis.ipynb

notebooks/
  analysis.ipynb     pre-executed plots + tables (open in VS Code / Jupyter)

results/
  dataset_summary.json       per-split counts + length stats
  baseline.json              TF-IDF + logreg metrics
  baseline_test_predictions.csv
  summary.json               aggregated transformer results (mean ± std)

runs/                (gitignored)    checkpoints + per-slot metrics.json
hf_cache/            (gitignored)    HF model weights (~2 GB)
```

## What goes where when you run

* `data/processed/*.csv` — produced by `src.data`, stratified with
  `RANDOM_SEED=50`.
* `results/*.json` — small, committed, the notebook and report both read
  from them.
* `runs/<model>/lr<lr>_seed<seed>/` — per-training-slot output:
  `metrics.json`, `test_predictions.csv`, best `checkpoint-N/`. Hundreds
  of MB each, gitignored. Safe to delete after `summary.json` is built.
