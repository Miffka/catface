# experiments/e2 — baselines

## What this is
RSCH-2 asks Q2: does a learned model beat two geometric ratios (eye aperture, ear angle) fed to logistic regression? This run trains three baseline models on cat-emotions-3, plausible rows only, the three usable classes (attentive, relaxed, uncomfortable), on identical stratified 5-fold splits with balanced class weights. Produced by `uv run python scripts/make_splits.py` then `uv run python scripts/baselines.py`.

## Input
Rows: 1967. Per-class counts:
  attentive: 1130
  relaxed: 730
  uncomfortable: 107
Split recipe: `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` over `class`, cached at `data/cache/splits.csv`.

## Results

### ratios_lr
`n_features` = 2.

| fold | macro F1 |
|---|---|
| 0 | 0.298 |
| 1 | 0.435 |
| 2 | 0.437 |
| 3 | 0.399 |
| 4 | 0.388 |
| **mean** | **0.391 +/- 0.051** |

Confusion matrix (rows = true, columns = predicted, summed over 5 folds):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 734 | 229 | 167 |
| relaxed | 299 | 270 | 161 |
| uncomfortable | 45 | 38 | 24 |

Plot: `plots/ratios_lr_confusion_matrix.png`.

### coords_lr
`n_features` = 96.

| fold | macro F1 |
|---|---|
| 0 | 0.435 |
| 1 | 0.425 |
| 2 | 0.455 |
| 3 | 0.386 |
| 4 | 0.394 |
| **mean** | **0.419 +/- 0.026** |

Confusion matrix (rows = true, columns = predicted, summed over 5 folds):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 638 | 265 | 227 |
| relaxed | 277 | 270 | 183 |
| uncomfortable | 20 | 23 | 64 |

Plot: `plots/coords_lr_confusion_matrix.png`.

### mlp
`n_features` = 96.

| fold | macro F1 |
|---|---|
| 0 | 0.426 |
| 1 | 0.417 |
| 2 | 0.413 |
| 3 | 0.422 |
| 4 | 0.376 |
| **mean** | **0.411 +/- 0.018** |

Confusion matrix (rows = true, columns = predicted, summed over 5 folds):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 735 | 217 | 178 |
| relaxed | 363 | 191 | 176 |
| uncomfortable | 23 | 18 | 66 |

Plot: `plots/mlp_confusion_matrix.png`.

## Required controls
- [x] Identical splits across all three models: all three read `data/cache/splits.csv`, written once by `scripts/make_splits.py`.
- [x] 5-fold: `n_splits=5` in `catface.ml.splits.make_splits` and `catface.ml.cv.cross_validate`.
- [x] Macro F1 reported (not accuracy alone): per-fold and mean +/- std, above.
- [x] Confusion matrix per model: above, summed over folds.
- [x] Balanced class weights: `class_weight="balanced"` (LR models), `nn.CrossEntropyLoss(weight=...)` computed via `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` (MLP).

## Q2 verdict
Model (1) ratios_lr scores mean macro F1 0.391. Model (3) mlp scores 0.411, std 0.018 across folds. The gap between them is 0.019, beyond one CV std of model (3)'s macro F1: **NO MATCH**. The MLP clears the two-ratio baseline by more than one fold's worth of its own noise, so Q2's answer here is that the learned model does find signal past eye aperture and ear angle.

## Extra readout
`muzzle_spread_ratio` (max pairwise distance among the 22 MUZZLE points, divided by inter-ocular distance) over the 1967 input rows: mean 1.239, std 0.103. This is descriptive only. None of the three baselines above use it as a model input; model (1) stays exactly eye aperture plus ear angle, per Q2 as posed. It is a geometry-only stand-in for whisker-pad spread, not a validated "tension" measure. See `docs/backlog.md` RSCH-2 grooming notes and `src/catface/core/geometry.py:muzzle_spread_ratio`.

## Citations
- CatFLW (landmark scheme, Finka et al.) and the Finka landmark scheme: see `docs/MODEL_REPORT.md` Citations.
- Feline pain-assessment descriptors, Scientific Reports, https://www.nature.com/articles/s41598-023-49031-2: 35 geometric descriptors (angles, distance ratios, area ratios by action unit) over cat facial landmarks. It reports that ear-position and orbital-tightening descriptors gave the smallest prediction error of the set, external validation that eye aperture and ear angle make a reasonable pair to baseline against here, not a requirement to add features beyond Q2's stated two.

