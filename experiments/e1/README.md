# experiments/e1 — shape space

## What this is
Procrustes alignment + PCA over the E0 landmark cache, cat-emotions-3 only,
plausible rows only. Answers RSCH-1: is there visible class signal before
training anything, and does any early PC track a pose confound? Also writes
`models/class_means.json`. Produced by `uv run python scripts/shape_space.py`.

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
Threshold: a PC is judged to visibly track a confound at |r| > 0.3.
- **ear_position**: PC1 tracks it, r = 0.84.
- **head_yaw**: PC2 tracks it, r = 0.90.

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

Classes do not visibly separate in the first six PCs — the best ratio (PC3, 0.044) is small; see `plots/pc_scatter_grid.png`.

## Stop condition
NOT triggered — RSCH-1 names no stop condition ("this is a look, not a gate"),
per `docs/backlog.md` and `docs/research_process.md`'s instruction to read the
stop condition before closing.

## models/class_means.json
Covers exactly the three usable classes: attentive, relaxed, uncomfortable.
The other five cat-emotions-3 label folders (`no clear emotion recognizable`,
`sad`, `angry`, `Unlabeled`, `attentive uncomfortable`) are single/low-double-digit
counts and, per RSCH-4's method in `docs/backlog.md`, are not treated as real
expression classes. Each entry is a 48x2 mean of the shared Procrustes-aligned
frame computed above (same alignment the PCA ran on, not a separate one).
