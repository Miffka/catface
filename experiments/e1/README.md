# experiments/e1 — shape space

## What this is
Procrustes alignment + PCA over the E0 landmark cache, cat-emotions-3 only, plausible rows only. Also writes `models/class_means.json`. Produced by
`uv run python scripts/shape_space.py`.

## Input
cat-emotions-3 total rows: 2071. Plausible (used): 2029.
Per-class plausible counts:
  attentive: 1130
  relaxed: 730
  uncomfortable: 107
  no clear emotion recognizable: 34
  sad: 22
  angry: 3
  Unlabeled: 2
  attentive uncomfortable: 1

## PCA explained variance
| PC | explained variance | cumulative |
|---|---|---|
| PC1 | 34.5% | 34.5% |
| PC2 | 26.5% | 61.0% |
| PC3 | 10.2% | 71.2% |
| PC4 | 5.3% | 76.6% |
| PC5 | 3.5% | 80.1% |
| PC6 | 2.3% | 82.4% |
Plot: `plots/pc_explained_variance.png`.

## Pose confound check
Plot threshold: |r| > 0.3.
- **ear_position**: strongest at PC1, r = 0.84.
- **head_yaw**: strongest at PC2, r = 0.90.

Plot: `plots/pc_confound_scatter.png` (the strongest offending PC vs its proxy).

## Class separation
Between-class / within-class variance ratio per PC, computed across the three
usable classes only (attentive, relaxed, uncomfortable):

| PC | ratio |
|---|---|
| PC1 | 0.028 |
| PC2 | 0.002 |
| PC3 | 0.044 |
| PC4 | 0.012 |
| PC5 | 0.008 |
| PC6 | 0.001 |

Plot: `plots/pc_scatter_grid.png`.

## models/class_means.json
Covers exactly the three usable classes: attentive, relaxed, uncomfortable.
The other five cat-emotions-3 label folders (`no clear emotion recognizable`,
`sad`, `angry`, `Unlabeled`, `attentive uncomfortable`) are single/low-double-digit
counts and, per RSCH-4's method in `docs/backlog.md`, are not treated as real
expression classes. Each entry is a 48x2 mean of the shared Procrustes-aligned
frame computed above (same alignment the PCA ran on, not a separate one).
