# experiments/e4 — class structure and pose

## What this is
RSCH-4 asks Q3 (how much head pose contaminates the prediction) and Q4 (whether the three classes are separable or some collapse). Six arms on the same 1967 cat-emotions-3 rows and the same frozen five folds E2 and E3 used, each scored inside four bins of a pose estimator built from the landmark map. Method, bin count, verdict metric, thresholds and decision rules are the Research PM's two grooming passes in `docs/backlog.md`, all pre-registered; the constants and both verdict functions were committed in `src/catface/ml/pose.py` and `src/catface/ml/scoring.py` before this run produced a single metric. Produced by `uv run python scripts/pose_and_classes.py`.

## Input
Rows: 1967. Per-class counts:
  attentive: 1130
  relaxed: 730
  uncomfortable: 107
Splits: `data/cache/splits.csv`, sha256 `488c674e6cde9a239b481a5f52091b2372cf1196d2106b6fee86851ecbecee47`, reused as-is from E2 and E3, never regenerated.

## The pose estimators
The axis convention and the weak-perspective projection model are written out in `src/catface/core/geometry.py`'s module docstring and are not repeated here. Both estimators run on the same `generalized_procrustes` output E1 used, so in-plane roll is already removed. Pitch is not modelled and not removed.

**Family 1, foreshortening (primary).** `w_obs / v_obs`, the outer-eye-corner span over the chin-to-eye-corner-midpoint vertical, equals `r_frontal * |cos(theta)|`: the pair's own depth cancels, so no depth prior enters anywhere. Unsigned, monotonic on [0, 90] degrees, nothing saturates.
**Family 2, midline offset (secondary, signed).** `s`, the philtrum's offset from the eye-corner midpoint in units of the observed span, equals `d_rel * tan(theta)`. Signed and monotonic over the full range.
**The legacy centroid proxy.** E1's quantity, reported per row and correlated against the new estimator, so E1's PC2 finding stays connected to this one. Not a binning quantity and no edge is computed from it.

Rejected estimator families, recorded so the choice is on the record:

- **Family 3, mirror-Procrustes residual.** Swap all 21 bilateral pairs, align the relabelled shape to the original, read the residual. Rejected at grooming: one non-negative scalar mixing yaw with expression asymmetry, one ear forward and landmark error, with no way to decompose it, and unsigned like family 1 without family 1's freedom from a depth prior.
- **Per-row least-squares fit of theta against all 21 pairs.** Family 1 generalised, and right if a 3D reference cat existed. None does: 21 frontal spans each carry the same population-versus-individual width confound, so the extra precision buys nothing against a shared systematic error that dominates it. The obvious upgrade if a 3D model ever lands.
- **The E1 centroid proxy.** Kept as a named legacy column, not as a binning quantity. It goes as sin(2*theta), so it turns over and one value has two candidate angles; the first grooming pass's saturation machinery existed only to manage that. See below.

## r_frontal and the out-of-domain rows
Pre-registered: `r_frontal` is the 95th percentile of `w_obs / v_obs` over all 1967 rows, pooled, computed once and frozen into `config.json` before any arm was scored. Rows above it give `cos(theta) > 1`, clamp to theta = 0 and land in bin 0; they are counted, never given an invented angle.

| percentile | `r_frontal` | out-of-domain rows | share |
|---|---|---|---|
| 90.0th | 1.5334 | 197 | 10.0% |
| 95.0th | 1.6330 | 99 | 5.0% |
| 99.0th | 1.8748 | 20 | 1.0% |

## Bins
Four equal-count quantile bins on the estimator, computed at run time over all 1967 rows, pooled, and frozen into `config.json` before any arm was scored. Bin 0 is the most frontal. The ratio edge is what an app-side consumer would key on; it carries no modelling assumption at all.

| bin | ratio range (`w_obs / v_obs`) | theta range (95th pct) | rows | `uncomfortable` rows | per-fold `uncomfortable` |
|---|---|---|---|---|---|
| 0 | 1.417 to 1.633 | 0.0 to 29.8 deg | 492 | 44 | [9, 13, 7, 6, 9] |
| 1 | 1.333 to 1.417 | 29.8 to 35.3 deg | 491 | 20 | [2, 2, 5, 8, 3] |
| 2 | 1.266 to 1.333 | 35.3 to 39.2 deg | 492 | 18 | [6, 2, 4, 2, 4] |
| 3 | 0.779 to 1.266 | 39.2 to 61.5 deg | 492 | 25 | [5, 5, 5, 5, 5] |

Pre-registered 3-bin fallback: **not taken**. The smallest (bin x fold) `uncomfortable` cell holds 2 rows, none is empty, so the run stayed at four bins.

## The internal control, and what it did to the word "yaw"
Spearman rho between |s| and tan(theta) over all 1967 rows: **0.106**, 95% bootstrap CI 0.062 to 0.149 (B = 1000, seed 0). The pre-registered thresholds are 0.5 to call the binning quantity yaw and 0.3 to withdraw the label.

**The measured rho is below the withdrawal threshold.** The two estimators, built from the same landmark map on the same frame and predicted by the same geometry to agree, are very nearly unrelated on this data. That is the single most important number in this run: whatever the bins are binning, both families cannot be measuring it. Q3's verdict is therefore reported against the foreshortening ratio under its operational name, with no degree figure attached to it, and the degree columns above are kept only because the bins were computed on them. Plot: `plots/estimator_agreement.png`.

## The measured depth term, and the chin control
`d_rel = |s| / tan(theta)` over the 1862 rows above 5 degrees (below that `tan(theta) -> 0` and the ratio is noise): median **0.0517**, IQR 0.0214 to 0.1026. This is the quantity the first grooming pass would have swept as an invented constant; here it is measured per row.

The chin control, node 2, sits close to the eye-corner plane, so the geometry predicts its implied depth comes out far smaller in magnitude than the philtrum's. It does not: median 0.0449 against the philtrum's 0.0517, a ratio of 0.87. **The falsifiable prediction of the geometry is falsified.** The reading that fits both this and the rho above is that neither column is measuring depth: `|s| / tan(theta)` is a ratio of two noisy asymmetries, and a near-zero-depth point produces just as much of it as a protruding one. The synthetic round trip in `tests/test_pose.py` recovers the built-in 0.30 to 1e-6 and puts the chin below 1% of it, so the estimator is right and the data is not the geometry it assumes.

## The manual review montage
`yaw_extremes_manual_review.png`: the most-negative, near-zero and most-positive rows by family 2's signed `s`, cropped to the landmark box and rendered from the source images with points 4, 8, 16 and 2 marked. Looked at before this README was written; `yaw_extremes_manual_review.json` names the rows.

- Two of the three most-negative-`s` rows are landmark failures, not poses. In the most-negative row of all (`s` = -0.92) point 4 sits on one cat's eye and point 8 on a second cat's eye in the same photo; the next (`s` = -0.60) spreads the 48 points across three kittens in a basket. Both passed E0's plausibility filter. The extreme tail of family 2 is therefore populated by detector failures rather than by extreme yaw, and any statistic that leans on that tail is reading the detector.
- The rows that are readable do carry the sign. The third most-negative row is a single ginger cat with its head turned toward the image right, and the two readable most-positive rows are cats with their heads turned toward the image left. Read off the pictures: **positive `s` means the head is turned toward the image left**, which is the cat's own right, and negative `s` toward the image right. That is the direction `u_hat` running from point 4 (the cat's left outer eye corner, on the image right in a frontal photo) to point 8 predicts, so the convention is consistent, but it is stated here because the montage shows it, not because the algebra says it.
- The near-zero-`s` panels are where family 1 comes apart. The middle panel is a black cat facing the camera square on -- no turn a human would call yaw -- and family 1 puts it at 38.9 degrees, above the median of the whole dataset. Its face is simply narrow relative to its chin-to-eye span. This is the brachycephalic/dolichocephalic confound named in the grooming, seen in one picture, and it is the mechanism behind the low rho below.

Sign balance: 1023 rows negative, 944 positive, close to even as a population of photographs should be.

## The superseded centroid proxy, and why the estimator changed
E1's centroid proxy reaches |proxy| = 0.1265 on these 1967 rows, against the roughly 0.0315 that the first grooming pass's forward model could generate at its central depth prior d = 0.30 -- the data overruns the model by a factor of about four, so for those rows no angle exists that the model could have produced. That finding is what motivated the second grooming pass to replace the estimator rather than narrow its prior, and it is recorded here as a measurement on the superseded quantity, not as an input to any bin, threshold or verdict in this run. The proxy moves under roll, pitch, expression asymmetry and plain landmark error as well as yaw.

Correlation of the legacy proxy against this run's theta: r = -0.067. The two are essentially unrelated, which is the same disagreement the internal control found between families 1 and 2.
E1's published r = 0.90 between PC2 and the proxy re-derives as 0.896 on E1's own 2029-row plausible stack, now that the proxy is imported from `core.geometry` rather than computed inline. `experiments/e1` is byte-unchanged and E1 was not re-run.

## Arms

| arm | features | role |
|---|---|---|
| `ratios_lr` | 3 geometric ratios | E2 model arm |
| `coords_lr` | 96 aligned coordinates | E2 model arm |
| `mlp` | 96 aligned coordinates | E2 model arm |
| `gnn_identity` | 48x4 node features, `A_hat = I` | E3's winning arm |
| `yaw_only_lr` | the binning quantity alone | control (c) |
| `random_baseline` | none (uniform dummy) | control (b) |

All six on the frozen folds with `oversample=True` on training folds only, test folds untouched, exactly as E2 and E3 ran them. Each arm is called through its own module's `build_model`, not its `run()`, because E4 needs per-row out-of-fold predictions.

## Results, pooled

| arm | mean macro F1 | mean kappa | pooled out-of-fold kappa (95% CI) |
|---|---|---|---|
| `ratios_lr` | 0.385 +/- 0.030 | 0.154 +/- 0.046 | 0.154 (0.122 to 0.182) |
| `coords_lr` | 0.426 +/- 0.032 | 0.174 +/- 0.048 | 0.174 (0.141 to 0.209) |
| `mlp` | 0.418 +/- 0.023 | 0.155 +/- 0.046 | 0.155 (0.118 to 0.188) |
| `gnn_identity` | 0.466 +/- 0.012 | 0.231 +/- 0.022 | 0.231 (0.193 to 0.267) |
| `yaw_only_lr` | 0.274 +/- 0.010 | 0.041 +/- 0.011 | 0.041 (0.017 to 0.065) |
| `random_baseline` | 0.302 +/- 0.000 | 0.017 +/- 0.001 | 0.017 (-0.011 to 0.044) |

Verdict arm by the pre-registered rule (highest pooled out-of-fold kappa among the four model arms): **`gnn_identity`**.

Control (c), `yaw_only_lr`, is reported at the same detail as the model arms below, not as a footnote. Logistic regression on the binning quantity alone, same configuration as the other two LR arms, same folds, same oversampling: pooled kappa 0.041 against the uniform dummy's 0.017. Pose barely predicts the label, so a per-bin kappa that moved could not be explained away as yaw carrying label information.

### ratios_lr

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.340 | 0.117 | 0.131 |
| 1 | 0.429 | 0.210 | 0.214 |
| 2 | 0.396 | 0.179 | 0.185 |
| 3 | 0.391 | 0.180 | 0.186 |
| 4 | 0.368 | 0.084 | 0.087 |
| **mean** | **0.385 +/- 0.030** | **0.154 +/- 0.046** | **0.161 +/- 0.046** |
| **pooled out-of-fold** | | **0.154** (CI 0.122 to 0.182) | |

Per bin (bin 0 = most frontal):

| bin | n | kappa (95% CI) | macro F1 | accuracy | that bin's majority rate | n attentive | n relaxed | n uncomfortable | per-fold kappa |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 492 | 0.231 (0.165 to 0.290) | 0.437 | 0.598 | 0.604 | 297 | 151 | 44 | 0.25, 0.27, 0.21, 0.20, 0.22 |
| 1 | 491 | 0.182 (0.115 to 0.245) | 0.403 | 0.542 | 0.603 | 296 | 175 | 20 | 0.14, 0.27, 0.15, 0.20, 0.19 |
| 2 | 492 | 0.107 (0.043 to 0.168) | 0.342 | 0.457 | 0.553 | 272 | 202 | 18 | 0.08, 0.14, 0.24, 0.18, -0.07 |
| 3 | 492 | 0.085 (0.024 to 0.147) | 0.342 | 0.402 | 0.539 | 265 | 202 | 25 | 0.06, 0.12, 0.09, 0.06, 0.08 |

### coords_lr

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.435 | 0.169 | 0.175 |
| 1 | 0.431 | 0.197 | 0.206 |
| 2 | 0.479 | 0.246 | 0.256 |
| 3 | 0.394 | 0.163 | 0.170 |
| 4 | 0.391 | 0.098 | 0.102 |
| **mean** | **0.426 +/- 0.032** | **0.174 +/- 0.048** | **0.182 +/- 0.050** |
| **pooled out-of-fold** | | **0.174** (CI 0.141 to 0.209) | |

Per bin (bin 0 = most frontal):

| bin | n | kappa (95% CI) | macro F1 | accuracy | that bin's majority rate | n attentive | n relaxed | n uncomfortable | per-fold kappa |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 492 | 0.147 (0.100 to 0.194) | 0.347 | 0.423 | 0.604 | 297 | 151 | 44 | 0.18, 0.20, 0.11, 0.08, 0.16 |
| 1 | 491 | 0.140 (0.077 to 0.207) | 0.386 | 0.507 | 0.603 | 296 | 175 | 20 | 0.10, 0.21, 0.13, 0.21, 0.06 |
| 2 | 492 | 0.148 (0.070 to 0.226) | 0.434 | 0.524 | 0.553 | 272 | 202 | 18 | 0.20, 0.09, 0.29, 0.21, -0.03 |
| 3 | 492 | 0.223 (0.145 to 0.301) | 0.496 | 0.575 | 0.539 | 265 | 202 | 25 | 0.16, 0.22, 0.40, 0.07, 0.18 |

### mlp

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.427 | 0.155 | 0.161 |
| 1 | 0.443 | 0.211 | 0.219 |
| 2 | 0.421 | 0.157 | 0.167 |
| 3 | 0.423 | 0.179 | 0.184 |
| 4 | 0.376 | 0.073 | 0.076 |
| **mean** | **0.418 +/- 0.023** | **0.155 +/- 0.046** | **0.161 +/- 0.047** |
| **pooled out-of-fold** | | **0.155** (CI 0.118 to 0.188) | |

Per bin (bin 0 = most frontal):

| bin | n | kappa (95% CI) | macro F1 | accuracy | that bin's majority rate | n attentive | n relaxed | n uncomfortable | per-fold kappa |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 492 | 0.145 (0.092 to 0.194) | 0.367 | 0.419 | 0.604 | 297 | 151 | 44 | 0.16, 0.21, 0.05, 0.11, 0.20 |
| 1 | 491 | 0.161 (0.083 to 0.229) | 0.405 | 0.525 | 0.603 | 296 | 175 | 20 | 0.08, 0.35, 0.12, 0.24, 0.04 |
| 2 | 492 | 0.150 (0.078 to 0.223) | 0.425 | 0.543 | 0.553 | 272 | 202 | 18 | 0.26, 0.12, 0.20, 0.28, -0.04 |
| 3 | 492 | 0.170 (0.098 to 0.241) | 0.479 | 0.565 | 0.539 | 265 | 202 | 25 | 0.13, 0.13, 0.33, 0.17, 0.04 |

### gnn_identity

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.460 | 0.205 | 0.211 |
| 1 | 0.470 | 0.240 | 0.245 |
| 2 | 0.487 | 0.250 | 0.255 |
| 3 | 0.454 | 0.255 | 0.258 |
| 4 | 0.458 | 0.205 | 0.209 |
| **mean** | **0.466 +/- 0.012** | **0.231 +/- 0.022** | **0.235 +/- 0.021** |
| **pooled out-of-fold** | | **0.231** (CI 0.193 to 0.267) | |

Per bin (bin 0 = most frontal):

| bin | n | kappa (95% CI) | macro F1 | accuracy | that bin's majority rate | n attentive | n relaxed | n uncomfortable | per-fold kappa |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 492 | 0.191 (0.132 to 0.247) | 0.402 | 0.500 | 0.604 | 297 | 151 | 44 | 0.21, 0.22, 0.17, 0.11, 0.22 |
| 1 | 491 | 0.240 (0.170 to 0.311) | 0.440 | 0.580 | 0.603 | 296 | 175 | 20 | 0.25, 0.35, 0.06, 0.31, 0.21 |
| 2 | 492 | 0.222 (0.149 to 0.297) | 0.452 | 0.571 | 0.553 | 272 | 202 | 18 | 0.22, 0.19, 0.24, 0.30, 0.18 |
| 3 | 492 | 0.233 (0.152 to 0.311) | 0.527 | 0.573 | 0.539 | 265 | 202 | 25 | 0.11, 0.11, 0.43, 0.25, 0.21 |

### yaw_only_lr

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.289 | 0.047 | 0.062 |
| 1 | 0.271 | 0.055 | 0.075 |
| 2 | 0.264 | 0.037 | 0.050 |
| 3 | 0.280 | 0.042 | 0.052 |
| 4 | 0.264 | 0.023 | 0.030 |
| **mean** | **0.274 +/- 0.010** | **0.041 +/- 0.011** | **0.054 +/- 0.015** |
| **pooled out-of-fold** | | **0.041** (CI 0.017 to 0.065) | |

Per bin (bin 0 = most frontal):

| bin | n | kappa (95% CI) | macro F1 | accuracy | that bin's majority rate | n attentive | n relaxed | n uncomfortable | per-fold kappa |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 492 | 0.000 (0.000 to 0.000) | 0.055 | 0.089 | 0.604 | 297 | 151 | 44 | 0.00, 0.00, 0.00, 0.00, 0.00 |
| 1 | 491 | -0.006 (-0.069 to 0.062) | 0.315 | 0.420 | 0.603 | 296 | 175 | 20 | 0.00, 0.01, -0.04, 0.06, -0.05 |
| 2 | 492 | 0.000 (0.000 to 0.000) | 0.194 | 0.411 | 0.553 | 272 | 202 | 18 | 0.00, 0.00, 0.00, 0.00, 0.00 |
| 3 | 492 | 0.000 (0.000 to 0.000) | 0.194 | 0.411 | 0.539 | 265 | 202 | 25 | 0.00, 0.00, 0.00, 0.00, 0.00 |

### random_baseline

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.302 | 0.015 | 0.017 |
| 1 | 0.302 | 0.015 | 0.017 |
| 2 | 0.302 | 0.018 | 0.019 |
| 3 | 0.302 | 0.018 | 0.019 |
| 4 | 0.302 | 0.018 | 0.019 |
| **mean** | **0.302 +/- 0.000** | **0.017 +/- 0.001** | **0.018 +/- 0.001** |
| **pooled out-of-fold** | | **0.017** (CI -0.011 to 0.044) | |

Per bin (bin 0 = most frontal):

| bin | n | kappa (95% CI) | macro F1 | accuracy | that bin's majority rate | n attentive | n relaxed | n uncomfortable | per-fold kappa |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 492 | 0.036 (-0.023 to 0.096) | 0.322 | 0.376 | 0.604 | 297 | 151 | 44 | 0.04, -0.00, -0.01, 0.05, 0.10 |
| 1 | 491 | 0.018 (-0.036 to 0.077) | 0.296 | 0.348 | 0.603 | 296 | 175 | 20 | 0.04, -0.01, -0.02, -0.08, 0.14 |
| 2 | 492 | -0.018 (-0.076 to 0.039) | 0.274 | 0.337 | 0.553 | 272 | 202 | 18 | -0.08, 0.05, 0.04, 0.06, -0.13 |
| 3 | 492 | 0.028 (-0.029 to 0.086) | 0.309 | 0.362 | 0.539 | 265 | 202 | 25 | 0.07, 0.00, 0.02, 0.03, 0.02 |

Plot: `plots/kappa_per_bin.png`, every arm's kappa in every bin with its bootstrap CI.

One thing the pre-registered rule does not capture and the tables do: `ratios_lr` declines monotonically across the bins, and it is the one arm whose three features are themselves geometric ratios of the same kind as the binning quantity, so the likeliest reading is shared measurement rather than pose. The verdict arm `gnn_identity` does not, and the CIs overlap throughout, which is why the rule did not fire. Recorded here so the observation is on the page rather than left for a reader to find and over-read.

Accuracy is printed beside each bin's own majority-class rate on purpose: with a class prior this skewed, an arm gains accuracy by predicting `attentive` harder, and the verdict rests on kappa for that reason.

## Control (a): per-bin class mix
Guards against reading prior shift as pose degradation. Total-variation distance between each bin's class distribution and the overall prior; the pre-registered tolerance is 0.05.

| bin | attentive | relaxed | uncomfortable | TV from the overall prior |
|---|---|---|---|---|
| 0 | 297 | 151 | 44 | 0.064 |
| 1 | 296 | 175 | 20 | 0.028 |
| 2 | 272 | 202 | 18 | 0.039 |
| 3 | 265 | 202 | 25 | 0.039 |

## Control (b): the uniform dummy, inside each bin
`DummyClassifier(strategy="uniform", random_state=0)` on the identical folds, scored within each bin under that bin's own label prior rather than once over the pooled rows -- see the `random_baseline` per-bin table above. It sits at chance in every bin, which is what makes the model arms' per-bin kappas readable as discrimination.

## Control (d): reproduction against the committed E2/E3 runs
E2's and E3's `metrics.json` hold no per-row predictions, so every arm had to be re-run to recover them. Per-fold macro F1, kappa and MCC compared against the committed files:

| arm | source | max abs delta | exact |
|---|---|---|---|
| `ratios_lr` | `experiments/e2/ratios_lr/metrics.json` | 0.000000 | yes |
| `coords_lr` | `experiments/e2/coords_lr/metrics.json` | 0.000000 | yes |
| `mlp` | `experiments/e2/mlp/metrics.json` | 0.000000 | yes |
| `gnn_identity` | `experiments/e3/gnn_identity/metrics.json` | 0.000000 | yes |
| `yaw_only_lr` | none (new at E4) | - | - |
| `random_baseline` | `experiments/e2/random_baseline/metrics.json` | 0.000000 | yes |

All 5 checkable arms reproduce the committed per-fold macro F1, kappa and MCC bit-for-bit, so this run's out-of-fold predictions belong to the same models E2 and E3 reported.

Recorded because it was observed rather than inferred: an earlier execution of this same script, at the same commit and on the same inputs, reproduced every other arm exactly and `mlp` in four of its five folds, fold 0 differing by 0.0153 of kappa -- about five predictions out of that fold's 393. The seeding is identical to `mlp.run`'s (`torch.manual_seed(0)` immediately before the arm), so torch CPU training here is not bit-reproducible from one execution to the next. Every artifact in this directory comes from a single execution, and the reproduction table above is that execution's.

## Q3 verdict
Status: **NEGATIVE (PROVISIONAL)**.

Verdict arm: `gnn_identity`, the highest pooled out-of-fold kappa among the four model arms, picked by the pre-registered rule rather than by which arm gives the nicer pose story. Verdict metric: kappa, with macro F1 beside it.

Spearman rho between |s| (family 2) and tan(theta) (family 1) is 0.106 (95% bootstrap CI 0.062 to 0.149), below the pre-registered 0.3 threshold: **the yaw label is not earned**. The verdict is reported against the binning quantity under its operational name, the foreshortening ratio, with no degree figure attached.

Bin 0 clears chance (kappa 0.191, CI 0.132 to 0.247). **The degradation rule did not fire**: no bin b >= 1 has a kappa upper CI below bin 0's lower CI. **Q3 verdict: NEGATIVE** -- no measurable degradation over the foreshortening-ratio range this dataset covers, threshold not locatable at this sample size, and the app applies no pose penalty.

**Prior-shift tolerance exceeded** at bin 0 (TV 0.064), against the pre-registered TV = 0.05 against the overall class prior: the verdict is **PROVISIONAL** and those bins' kappas move partly for reasons that are not pose.

## Q4: class structure
Pooled out-of-fold confusion matrix, `gnn_identity` (rows = true, columns = predicted):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 711 | 280 | 139 |
| relaxed | 274 | 328 | 128 |
| uncomfortable | 19 | 33 | 55 |

Plot: `plots/pooled_confusion_matrix.png`. Full pair detail: `merge.json`.

Status: **NEGATIVE**.

Pooled out-of-fold confusion matrix from `gnn_identity` (kappa 0.231, CI 0.193 to 0.267). Pre-registered merge thresholds, both required and not either: cross-talk >= 0.25 **and** the pairwise 2x2 kappa CI contains 0.

- **attentive / relaxed**: cross-talk 0.298 (clears 0.25), pairwise 2x2 kappa 0.262 with CI 0.213 to 0.310 (excludes 0) -- not a candidate.

- **attentive / uncomfortable**: cross-talk 0.128 (below 0.25), pairwise 2x2 kappa 0.333 with CI 0.260 to 0.412 (excludes 0) -- not a candidate.

- **relaxed / uncomfortable**: cross-talk 0.192 (below 0.25), pairwise 2x2 kappa 0.240 with CI 0.151 to 0.321 (excludes 0) -- not a candidate.

No pair clears both thresholds. **Q4 verdict: NEGATIVE, no merge is warranted on the evidence**, and the three classes stay as they are. This is the pre-registered null path, written before the run so that a flat confusion matrix could not be turned into a merge out of its largest off-diagonal cell.

The retrained comparison was run anyway for **attentive / relaxed**, the highest cross-talk pair, although it is not a candidate: retrained 2-class kappa 0.148, post-hoc collapse of the 3-class predictions 0.190, 2-class uniform dummy 0.021 (`merged/gnn_identity/metrics.json`). Retraining under the merged label does not beat collapsing the predictions after the fact, so the null path rests on a measurement rather than on an argument.

The Method text's expectation that Scared, Surprised and Angry would prove interchangeable is **not testable here**: cat-emotions-7 is excluded (`docs/DECISIONS.md`, 2026-09-19). It is recorded as not testable, no other pair is substituted for it, and it stays as the prediction to check if that set ever returns.

## Limitations
- **Facial width.** Family 1 divides a bilateral span by a vertical one, so a genuinely narrow-faced cat photographed head-on reads as yawed. Breed is not labelled in cat-emotions-3, and with one photo per cat and no identity labels a per-row `r_frontal` is not estimable, so a single population value has to absorb real brachycephalic-to-dolichocephalic variation. The montage's frontal black cat at 38.9 degrees is this confound in one picture, and the low rho says it dominates rather than perturbs. This is the limitation to carry into `docs/MODEL_REPORT.md` under Q3.
- **Pitch is not modelled and not removed.** A cat looking up or down changes the chin-to-eye vertical that family 1 divides by, and nothing here separates that from a narrower face.
- **Landmark error.** The montage shows two of the six extreme rows are photographs of two cats with the 48 points split between them, both passing E0's plausibility filter. Every quantity in this run inherits that.
- **Within-dataset near-duplicates were never hashed** (`docs/MODEL_REPORT.md`), so the absolute per-bin numbers are an upper bound. It inflates every arm and every bin equally, so it does not tilt the per-bin comparison, which is what Q3 turns on.

## Required controls
- [x] Control (a): per-bin class counts and TV distance from the overall prior, above.
- [x] Control (b): uniform `DummyClassifier` scored within each bin under that bin's own prior, above.
- [x] Control (c): `yaw_only_lr` reported at the same detail as the model arms, above.
- [x] Control (d): per-fold metrics compared against the committed E2/E3 `metrics.json`, deltas in `config.json` and above.
- [x] Bootstrap as specified: within-bin over rows, B = 1000, seed 0, 2.5/97.5 percentile; per-fold kappas reported beside it as the secondary sanity check.
- [x] Bin edges frozen into `config.json` before any arm was scored, with the occupancy tables taken at the same moment.
- [x] Internal control reported against the pre-registered 0.5 / 0.3 thresholds, and the word "yaw" withdrawn by it.
- [x] Every degree figure in this run carries the percentile that produced it; the verdict carries none, because the control withdrew them.

## Files
- `config.json` — every frozen constant, the edges, the reproduction deltas, the git sha.
- `per_row.csv` — 1967 rows: both estimators, theta, the out-of-domain flag, the bin, the legacy proxy and one out-of-fold prediction column per arm. Every table above recomputes from it.
- `<arm>/metrics.json`, `merged/<arm>/metrics.json`, `merge.json`, `plots/`, `yaw_extremes_manual_review.png`.

