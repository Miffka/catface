# experiments/e2 — baselines

## What this is
Three baseline models -- ratios+LR (`ratios_lr`, three geometric ratios: eye aperture, ear angle, muzzle spread), coords+LR (`coords_lr`), coords+MLP (`mlp`) -- trained on cat-emotions-3, plausible rows only, the three usable classes (attentive, relaxed, uncomfortable), on identical stratified 5-fold splits. Balancing: random oversampling of minority classes on the training folds only (`oversample_to_balance`). Each model reports macro F1, Cohen's kappa and MCC, with a uniform-random classifier run as a chance-level reference. Produced by `uv run python scripts/make_splits.py` then `uv run python scripts/baselines.py`.

## Input
Rows: 1967. Per-class counts:
  attentive: 1130
  relaxed: 730
  uncomfortable: 107
Split recipe: `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` over `class`, cached at `data/cache/splits.csv`.

## Results

### ratios_lr
`n_features` = 3.

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.340 | 0.117 | 0.131 |
| 1 | 0.429 | 0.210 | 0.214 |
| 2 | 0.396 | 0.179 | 0.185 |
| 3 | 0.391 | 0.180 | 0.186 |
| 4 | 0.368 | 0.084 | 0.087 |
| **mean** | **0.385 +/- 0.030** | **0.154 +/- 0.046** | **0.161 +/- 0.046** |

Confusion matrix (rows = true, columns = predicted, summed over 5 folds):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 710 | 217 | 203 |
| relaxed | 284 | 245 | 201 |
| uncomfortable | 44 | 35 | 28 |

Plot: `plots/ratios_lr_confusion_matrix.png`.

### coords_lr
`n_features` = 96.

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.435 | 0.169 | 0.175 |
| 1 | 0.431 | 0.197 | 0.206 |
| 2 | 0.479 | 0.246 | 0.256 |
| 3 | 0.394 | 0.163 | 0.170 |
| 4 | 0.391 | 0.098 | 0.102 |
| **mean** | **0.426 +/- 0.032** | **0.174 +/- 0.048** | **0.182 +/- 0.050** |

Confusion matrix (rows = true, columns = predicted, summed over 5 folds):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 671 | 250 | 209 |
| relaxed | 300 | 265 | 165 |
| uncomfortable | 19 | 26 | 62 |

Plot: `plots/coords_lr_confusion_matrix.png`.

### mlp
`n_features` = 96.

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.427 | 0.155 | 0.161 |
| 1 | 0.443 | 0.211 | 0.219 |
| 2 | 0.421 | 0.157 | 0.167 |
| 3 | 0.423 | 0.179 | 0.184 |
| 4 | 0.376 | 0.073 | 0.076 |
| **mean** | **0.418 +/- 0.023** | **0.155 +/- 0.046** | **0.161 +/- 0.047** |

Confusion matrix (rows = true, columns = predicted, summed over 5 folds):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 742 | 218 | 170 |
| relaxed | 363 | 203 | 164 |
| uncomfortable | 23 | 20 | 64 |

Plot: `plots/mlp_confusion_matrix.png`.

## Random-classifier reference
`DummyClassifier(strategy="uniform", random_state=0)`, run on the identical 5-fold splits as the three models above -- a chance-level reference, excluded from the macro-F1 comparison plot.

| metric | mean +/- std |
|---|---|
| macro F1 | 0.302 +/- 0.000 |
| kappa | 0.017 +/- 0.001 |
| MCC | 0.018 +/- 0.001 |

Plot: `plots/random_baseline_confusion_matrix.png`.

## Citations
- CatFLW (landmark scheme, Finka et al.) and the Finka landmark scheme: see `docs/MODEL_REPORT.md` Citations.
- Feline pain-assessment descriptors, Scientific Reports, https://www.nature.com/articles/s41598-023-49031-2: 35 geometric descriptors (angles, distance ratios, area ratios by action unit) over cat facial landmarks. It reports that ear-position and orbital-tightening descriptors gave the smallest prediction error of the set, external validation that eye aperture and ear angle make a reasonable pair to baseline against here.

