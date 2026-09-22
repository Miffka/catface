# experiments/e4 — pose-binned check of coords_lr and gnn_identity

In-sample check of the two E2/E3 arms with the highest pooled out-of-fold kappa (`gnn_identity`, `coords_lr`, per `docs/backlog.md` RSCH-4): their committed full-dataset checkpoints are loaded and scored inside four pose bins. Not a re-run of the pre-registered out-of-fold numbers -- these checkpoints were fit on the same rows they're scored on here. No verdict is computed; the numbers below are for the Research PM to read directly. Produced by `uv run python scripts/pose_and_classes.py`.

Rows: 1967. Splits: `data/cache/splits.csv`, sha256 `488c674e6cde9a239b481a5f52091b2372cf1196d2106b6fee86851ecbecee47`.
Pose estimator: `core.geometry.yaw_foreshortening_ratio`. `r_frontal` (the 95th percentile of the observed ratio): 1.6330, 99 out-of-domain rows clamped to theta = 0.

## Bins
Four equal-count quantile bins on theta, bin 0 most frontal.

| bin | theta range (deg) |
|---|---|
| 0 | 0.0 to 29.8 |
| 1 | 29.8 to 35.3 |
| 2 | 35.3 to 39.2 |
| 3 | 39.2 to 61.5 |

## Results

| arm | checkpoint | pooled kappa (95% CI) | pooled macro F1 | pooled accuracy |
|---|---|---|---|---|
| `coords_lr` | `experiments/e2/coords_lr/checkpoint.pkl` | 0.213 (0.177 to 0.245) | 0.449 | 0.525 |
| `gnn_identity` | `experiments/e3/gnn_identity/checkpoint.pt` | 0.263 (0.228 to 0.297) | 0.491 | 0.571 |

### coords_lr

Pooled confusion matrix (rows = true, columns = predicted):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 650 | 278 | 202 |
| relaxed | 247 | 317 | 166 |
| uncomfortable | 20 | 22 | 65 |

Per bin:

| bin | n | kappa (95% CI) | macro F1 | accuracy | majority rate | n attentive | n relaxed | n uncomfortable |
|---|---|---|---|---|---|---|---|---|
| 0 | 492 | 0.159 (0.113 to 0.206) | 0.356 | 0.433 | 0.604 | 297 | 151 | 44 |
| 1 | 491 | 0.196 (0.131 to 0.263) | 0.425 | 0.532 | 0.603 | 296 | 175 | 20 |
| 2 | 492 | 0.181 (0.105 to 0.256) | 0.449 | 0.535 | 0.553 | 272 | 202 | 18 |
| 3 | 492 | 0.275 (0.200 to 0.352) | 0.522 | 0.600 | 0.539 | 265 | 202 | 25 |

### gnn_identity

Pooled confusion matrix (rows = true, columns = predicted):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 714 | 269 | 147 |
| relaxed | 256 | 342 | 132 |
| uncomfortable | 17 | 23 | 67 |

Per bin:

| bin | n | kappa (95% CI) | macro F1 | accuracy | majority rate | n attentive | n relaxed | n uncomfortable |
|---|---|---|---|---|---|---|---|---|
| 0 | 492 | 0.224 (0.162 to 0.277) | 0.428 | 0.518 | 0.604 | 297 | 151 | 44 |
| 1 | 491 | 0.266 (0.201 to 0.337) | 0.460 | 0.595 | 0.603 | 296 | 175 | 20 |
| 2 | 492 | 0.263 (0.193 to 0.336) | 0.491 | 0.589 | 0.553 | 272 | 202 | 18 |
| 3 | 492 | 0.262 (0.186 to 0.338) | 0.540 | 0.581 | 0.539 | 265 | 202 | 25 |

Plot: `plots/kappa_per_bin.png`.

