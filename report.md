# Serbian Text Classification — Report

University NLP homework. Fine-tune one multilingual model and one
regional (BCMS) model on a Serbian sentiment dataset and compare them,
with a classic baseline for context.

For the full per-run numbers, LR sweep, and per-example predictions see
`results/summary.json` and `runs/*/*/metrics.json`. This file is a 1–2
page prose summary.

## Dataset

**SentiComments.SR** (Batanović et al., [GitHub](https://github.com/vukbatanovic/SentiComments.SR),
CC BY-NC-SA 4.0): 3,490 user comments from Serbian movie-review threads,
spell-corrected (`SentiComments.SR.corr.txt`). Format is tab-separated:
`label \t comment_id \t text`.

Labels in the raw file are a 6-way scheme — `{+1, -1, +M, -M, +NS, -NS}`
with an optional `s` suffix for sarcasm — which we collapse into
**3-class polarity**:

| Final class | Original labels | Notes |
|---|---|---|
| **positive** | `+1`, `+M`, `-1s`, `-Ms` | sarcastic negatives flip to positive |
| **negative** | `-1`, `-M`, `+1s`, `+Ms` | sarcastic positives flip to negative |
| **neutral**  | `+NS`, `-NS`, `+NSs`, `-NSs` | "no sentiment" variants, sarcasm irrelevant |

Why 3-way: six classes with ~580 examples per class on average and some
sarcasm variants below 50 would make per-class F1 noisy. The author's
own follow-up work on this dataset also collapses to 3-way polarity.

After cleanup (strip whitespace, drop empty-text rows, drop exact
duplicates), **3,460 rows** remain. Stratified 80/10/10 split with seed
50 gives **2,768 / 346 / 346**. The class distribution survives the
split:

| Split | n | negative | neutral | positive |
|---|---:|---:|---:|---:|
| train | 2,768 | 818 (30%) | 484 (17%) | 1,466 (53%) |
| val   | 346   | 102 | 61  | 183 |
| test  | 346   | 102 | 61  | 183 |

Imbalanced, so **macro-F1** is the headline metric throughout.

## Preprocessing — script handling

Serbian is written in both Cyrillic and Latin. Transliteration is
deterministic in one direction (Cyrillic → Latin) because every Cyrillic
letter has exactly one Latin equivalent; the reverse is ambiguous (a
Latin "lj" could be the single letter *љ* or two separate letters).

BERTić was pretrained mostly on Latin-script text, so Cyrillic input
produces broken subword fragments and hurts accuracy. We chose **Latin
as the canonical form** everywhere downstream. `src/text_utils.py`
transliterates Cyrillic rows; Latin rows pass through unchanged.

In practice SentiComments.SR is ~99% Latin already — only **26 rows**
(22 train, 3 val, 1 test) needed transliteration. We keep the step so
the pipeline doesn't depend on a dataset-specific assumption and would
survive a drop-in replacement dataset.

## Models

**`xlm-roberta-base`** (278 M parameters) — the multilingual baseline.
RoBERTa pretrained on 2.5 TB of CommonCrawl across 100 languages,
including Serbian in both scripts. Industry default for "I don't want
to think about which language."

**`classla/bcms-bertic`** (111 M parameters) — the regional model. An
ELECTRA model pretrained from scratch by the CLASSLA group on 8 GB of
Bosnian / Croatian / Montenegrin / Serbian web text, news, and
Wikipedia. The author of SentiComments.SR ships a `trainer_bertic.py`
in the same repo, which makes it the natural comparison pick.

Both models load via `AutoModelForSequenceClassification` with a fresh
3-way classification head on top. A third model is a one-line config
drop-in: add a new `configs/<name>.yaml` with the HF name and
`run_tag`.

**TF-IDF + Logistic Regression** baseline (`src/baseline.py`): word
1–2 grams, `min_df=2`, `max_df=0.95`, `class_weight='balanced'`, and a
grid over `C ∈ {0.25, 0.5, 1, 2, 4, 8}` picked by val macro-F1. Fits in
under a second on CPU. Standard "before you reach for transformers"
reference point.

## Training setup

**Hardware:** AMD Radeon RX 9060 XT (16 GB VRAM, RDNA 4), PyTorch
2.9.1+rocm7.2.1 on Windows. Code calls the standard `torch.cuda` API
— on ROCm, AMD is exposed through it transparently. Mixed precision
(`bf16=True`) held up on ROCm for all 12 fine-tuning runs; no fallback
was needed.

**Hyperparameters:**

| Setting | Value | Chosen because |
|---|---|---|
| max_length | 128 | p95 token length is 67–77 for both tokenizers (`results/dataset_summary.json`); 128 covers >99% of comments without truncation |
| batch_size | 16 | fits comfortably on 16 GB with bf16 |
| epochs | 5 | with early stopping (patience 2) on val macro-F1 |
| warmup_ratio | 0.1 | standard |
| weight_decay | 0.01 | standard |
| LR grid | {2e-5, 3e-5, 5e-5} | per model, picked by val macro-F1 |
| seeds | {42, 123, 2024} | 3 seeds at the chosen LR per model |

Best-checkpoint selection on validation macro-F1
(`load_best_model_at_end=True`). The test set is only touched after
training, once per run.

Both models picked **2e-5** as the best LR — the smallest value in the
grid. On a 2,768-example training set that is expected; larger LRs
overshoot when gradients are averaged over few examples.

## Results

Test set, mean ± standard deviation across 3 seeds (from
`results/summary.json`):

| Model | Params | Acc | Macro-F1 | Train time |
|---|---:|---:|---:|---:|
| tfidf + logreg (baseline) | ~7 k | 0.659 | 0.631 | <1 s |
| xlm-roberta-base | 278 M | 0.805 ± 0.007 | 0.785 ± 0.008 | 109 s |
| **classla/bcms-bertic** | **111 M** | **0.832 ± 0.013** | **0.814 ± 0.013** | **65 s** |

**BERTić wins on every axis** — ~3 F1 points higher than XLM-R, less
than half the parameters, and 60% of the training time. The baseline
trails both transformers by ~18 F1.

The gap is not seed noise. Each model's seed standard deviation is
~0.01 F1; the 0.029 F1 gap between BERTić and XLM-R is ~2–4× that. No
seed of XLM-R beat any seed of BERTić.

Per-class on the single seed-42 BERTić run:

| Class | Precision | Recall | F1 | support |
|---|---:|---:|---:|---:|
| negative | 0.796 | 0.725 | 0.759 | 102 |
| neutral  | 0.773 | 0.836 | 0.803 | 61 |
| positive | 0.872 | 0.891 | 0.881 | 183 |

Neutral was the baseline's weakest class (F1 0.57). BERTić pulls it up
to 0.80 — the hardest class is where the transformer story pays off
most.

## Where BERTić wins

Disagreement analysis on the seed-42 runs: of 346 test examples, **both
models correct** on 261, **both wrong** on 43, **only BERTić right** on
24, **only XLM-R right** on 18. BERTić has a clear per-example edge too,
not just higher aggregate numbers.

Examples where BERTić got it right and XLM-R didn't:

* `filmčina! greh je ako Mads ne dobije Oskara.` — true **positive**.
  XLM-R said negative. "Filmčina" is a Serbian intensifier noun
  ("great/powerful film") formed with the *-čina* augmentative suffix
  — exactly the kind of inflected, regionally-specific morphology
  BERTić's BCMS vocabulary handles and XLM-R's 100-language vocab
  struggles with.
* `A što samo 8? Od mene desetka ko vrata.` — true **positive**
  ("Why only 8? I'd give it a 10 flat"). XLM-R focused on the
  complaining opener "A što samo 8?" and called it negative. BERTić
  caught the *"desetka ko vrata"* idiom ("a 10 like doors" ≈
  definitive 10).
* `film je smaranje` — true **negative**. BERTić right; XLM-R wrong.
  *"Smaranje"* (slang for "a drag/boring") is colloquial Serbian; XLM-R
  likely tokenizes it into near-opaque fragments.

## Where XLM-R wins

Examples where XLM-R got it right and BERTić didn't:

* `kako ironičan naziv filma...` — true **negative**. XLM-R
  caught the ironic framing; BERTić defaulted to positive from the
  surface word "ironičan".
* `Nije loš, mada sam očekivala više...` — true **negative**. Negated
  praise ("not bad, but I expected more") — XLM-R handled the
  construction, BERTić did not.

Pattern: XLM-R's larger capacity occasionally helps on constructions
that need more global context (negation, irony flags). Not enough to
offset the morphology advantage BERTić has at word level.

## Where both struggle

Representative "both wrong" cases:

* `Lepo stoji u tekstu da se u filmu ne izgovara niti jedna reč.` —
  true **negative** (a sardonic jab), both predicted **neutral**. No
  surface sentiment vocabulary; the negativity is entirely
  context-dependent.
* `ovaj film zaslužuje mnogo veću ocenu!!!` — true **positive**, both
  predicted **negative**. The exclamation and the comparative
  "zaslužuje veću" ("deserves higher") throws both — probably learned
  as a complaint pattern.
* `Meni se ne sviđa ni prvi ni drugi, ovo je totalni promašaj oba
  filma!` — true **positive**(!!) — mislabeled in the source; both
  models called it negative, which any human would. These edge cases
  cap the achievable ceiling.

## Limitations

* **Single dataset, single domain.** SentiComments.SR is movie-review
  comments. Generalisation to tweets, news, or product reviews is not
  established.
* **Small test set (346 examples).** Each seed's test F1 is subject to
  ~±1 point noise from sampling alone. The 3-seed mean smooths this but
  the dataset simply cannot support a tighter estimate.
* **Label-collapse hides sarcasm performance.** We fold sarcasm
  variants into their inverted polarity. The models never see the
  sarcasm signal explicitly, and we never measure how often a sarcastic
  comment is correctly inverted.
* **Dataset-side mislabels** (last example above) place a ceiling on
  any model's achievable accuracy. We did not audit label quality.
* **3 seeds, 1 LR grid.** Tighter conclusions would need more seeds,
  k-fold CV, and a larger LR search. On a 2.7k-example dataset, the
  compute saved is not worth the extra confidence we'd gain.

## Improvements worth exploring

1. **Keep the 6-way scheme and train the sarcasm variants explicitly.**
   Accept sparse per-class support in exchange for a model that can
   detect sarcasm, not just absorb it into polarity.
2. **Larger BERTić** — the CLASSLA group has released a BERTić-large
   variant; comparing it against XLM-R-large would complete the 2×2.
3. **Class-weighted loss on the transformer.** The baseline uses
   `class_weight='balanced'`; the transformers use default
   cross-entropy. Weighting might lift neutral recall at the cost of
   positive precision.
4. **Domain adaptation on more Serbian movie text** before
   fine-tuning. A short MLM continue-training pass on
   `SentiComments.SR.orig.txt` (the uncorrected version we didn't use
   for training) might give BERTić another point or two.
5. **External evaluation on SerbMR** (Batanović's movie-review
   dataset). Different domain, different source — would test whether
   the BERTić-over-XLM-R gap generalises.

## Conclusion

On a 3,460-example Serbian sentiment dataset, a 111 M-parameter model
pretrained on the Serbian language family (BERTić) outperforms a 278 M
multilingual model (XLM-R) by ~3 F1 points — 0.814 ± 0.013 vs.
0.785 ± 0.008 test macro-F1, 3 seeds each. Both transformers improve
over a TF-IDF + logistic regression baseline by ~18 F1 points. The
regional-model advantage persists at less than half the parameter
count and 60% of the training time, which suggests that for
low-resource languages, matched pretraining data quality dominates raw
model scale.
