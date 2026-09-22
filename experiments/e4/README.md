# experiments/e4 — pose-binned check of coords_lr and gnn_identity

In-sample check of the two E2/E3 arms with the highest pooled out-of-fold kappa (`gnn_identity`, `coords_lr`, per `docs/backlog.md` RSCH-4): their committed full-dataset checkpoints are loaded and scored inside four pose bins. Not a re-run of the pre-registered out-of-fold numbers -- these checkpoints were fit on the same rows they're scored on here. No verdict is computed; the numbers below are for the Research PM to read directly. Produced by `uv run python scripts/pose_and_classes.py`.

**This is an in-sample check and its evidence is one-sided.** Both checkpoints were fit on the exact rows they are scored on here, so a flat-or-rising per-bin trend below does not establish that the model holds up on unseen high-yaw photos -- it only shows the checkpoints are not visibly underfitting their own high-yaw training rows. A falling trend would have been informative; this one is not proof of generalization.

Rows: 1967. Splits: `data/cache/splits.csv`, sha256 `488c674e6cde9a239b481a5f52091b2372cf1196d2106b6fee86851ecbecee47`.
Pose estimator: `core.geometry.yaw_foreshortening_ratio`. `r_frontal` (the 95th percentile of the observed ratio): 1.6330, 99 out-of-domain rows clamped to theta = 0.

## Bins
Four equal-count quantile bins on theta (arccos of the ratio), bin 0 most frontal. Reported here in raw `w_obs / v_obs` ratio units rather than degrees: this project's own internal control (`docs/backlog.md` RSCH-4, 2026-09-20 comment) found the yaw label unearned for this estimator -- Spearman rho 0.106 (CI 0.062 to 0.149) against a pre-registered 0.3 threshold -- so no degree figure is attached to it here. Ratio falls as yaw rises, so bin 0's range sits highest.

| bin | w_obs / v_obs range |
|---|---|
| 0 | 1.4177 to 2.4708 |
| 1 | 1.3328 to 1.4172 |
| 2 | 1.2663 to 1.3326 |
| 3 | 0.7791 to 1.2662 |

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

| bin | n | kappa (95% CI) | macro F1 | accuracy | majority rate | TV vs pooled prior | n attentive | n relaxed | n uncomfortable |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 492 | 0.159 (0.113 to 0.206) | 0.356 | 0.433 | 0.604 | 0.064 (over 0.05 tolerance) | 297 | 151 | 44 |
| 1 | 491 | 0.196 (0.131 to 0.263) | 0.425 | 0.532 | 0.603 | 0.028 | 296 | 175 | 20 |
| 2 | 492 | 0.181 (0.105 to 0.256) | 0.449 | 0.535 | 0.553 | 0.039 | 272 | 202 | 18 |
| 3 | 492 | 0.275 (0.200 to 0.352) | 0.522 | 0.600 | 0.539 | 0.039 | 265 | 202 | 25 |

`coords_lr`'s kappa in each bin is read against that bin's own majority-class rate as the chance floor: bin 0 0.604, bin 1 0.603, bin 2 0.553, bin 3 0.539.

### gnn_identity

Pooled confusion matrix (rows = true, columns = predicted):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 714 | 269 | 147 |
| relaxed | 256 | 342 | 132 |
| uncomfortable | 17 | 23 | 67 |

Per bin:

| bin | n | kappa (95% CI) | macro F1 | accuracy | majority rate | TV vs pooled prior | n attentive | n relaxed | n uncomfortable |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 492 | 0.224 (0.162 to 0.277) | 0.428 | 0.518 | 0.604 | 0.064 (over 0.05 tolerance) | 297 | 151 | 44 |
| 1 | 491 | 0.266 (0.201 to 0.337) | 0.460 | 0.595 | 0.603 | 0.028 | 296 | 175 | 20 |
| 2 | 492 | 0.263 (0.193 to 0.336) | 0.491 | 0.589 | 0.553 | 0.039 | 272 | 202 | 18 |
| 3 | 492 | 0.262 (0.186 to 0.338) | 0.540 | 0.581 | 0.539 | 0.039 | 265 | 202 | 25 |

`gnn_identity`'s kappa in each bin is read against that bin's own majority-class rate as the chance floor: bin 0 0.604, bin 1 0.603, bin 2 0.553, bin 3 0.539.

Per-bin total-variation distance from the pooled class prior exceeds the pre-registered 0.05 tolerance in bin 0 (0.064): that bin's class mix has drifted from the pooled prior enough that its kappa cannot be read as pose effect alone.

## Limitations

- **Facial-width confound.** `theta`/the ratio comes from dividing a bilateral span by a vertical one (family 1, foreshortening); a genuinely narrow- or wide-faced cat reads as more or less yawed at zero rotation. Breed and identity aren't labelled in cat-emotions-3 and there's one photo per cat, so a per-row correction isn't estimable at this sample size.
- **Pitch is not modelled.** The estimator only accounts for rotation about the vertical (yaw) axis; any pitch present in a photo is absorbed into the ratio uncalibrated, not removed.

Plot: `plots/kappa_per_bin.png`.

