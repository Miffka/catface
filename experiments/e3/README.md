# experiments/e3 — GNN

## What this is
RSCH-3 asks Q1: does a dense graph conv over the anatomical adjacency beat the E2 MLP? This directory holds the re-run ordered by the Research PM's re-grooming of 2026-09-20 (`docs/backlog.md`), which sent the first run back on methodology. Five arms, all on the same cat-emotions-3 rows and the frozen splits in `data/cache/splits.csv`, all balanced by training-fold oversampling, and the four graph-conv arms all at the same depth and readout except where the arm exists to vary one of those. No PyTorch Geometric anywhere: the conv is hand-written dense matmuls over a fixed adjacency buffer, which is what makes it export to ONNX. Produced by `uv run python scripts/gnn.py`.

## Input
Rows: 1967. Per-class counts:
  attentive: 1130
  relaxed: 730
  uncomfortable: 107
Split recipe: `StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` over `class`, cached at `data/cache/splits.csv` (reused as-is from E2, not regenerated).
Node features: `(N, 48, 4)` -- Procrustes-aligned (x,y) plus offset from the mean shape (x,y).
Anatomical adjacency: 98 unique edges, parsed from `models/graph_edge_schemes/graph_edges_manual_v3.txt`, sha256 `86b29330044500b04f1a9bf38d814537aea8624da0e2fb0b18f6a75bf0a97fb4`.
Random-adjacency ablation: 98 edges, drawn once (not per fold) with seed 1.
Identity control: `A_hat = I`, 48 nodes, no edges.

## Arms

| arm | adjacency | readout | balancing |
|---|---|---|---|
| `gnn_anatomical` | v3, 98 edges | flatten | oversample |
| `gnn_random_ablation` | random, 98 edges, seed 1 | flatten | oversample |
| `gnn_identity` | `I`, no message passing | flatten | oversample |
| `gnn_anatomical_meanpool` | v3, 98 edges | mean | oversample |
| `random_baseline` | none | none | oversample |

## ONNX export check
ONNX export of an untrained GNN **succeeded** (opset 17, input shape [1, 48, 4], readout `flatten`, exporter `torch.onnx.export(dynamo=False)`).
It ran on an untrained, randomly initialised model with the flatten head, before any training loop, per RSCH-3's method and the re-grooming's first acceptance criterion: the head shape changed from `Linear(32, 3)` to `Linear(1536, 3)`, so the first run's export result does not carry over. The export target went to a temp directory and was not kept. This is not RSCH-6's export verification (parity, benchmarking, manifest), which belongs to the Export Verifier once a winning model exists.

## Results

### mlp (E2, carried over for comparison)
`n_features` = 96 (flattened total, not per-node). Mean macro F1 **0.418 +/- 0.023**, kappa **0.155 +/- 0.046**, MCC **0.161 +/- 0.047** across 5 folds. Full detail: `experiments/e2/README.md`.

### gnn_anatomical
the hypothesis: v3 hand-authored anatomical adjacency, flatten readout.
`n_features` = 4 (per-node channel count), readout `flatten`, balancing: training-fold oversampling.

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.379 | 0.101 | 0.106 |
| 1 | 0.393 | 0.127 | 0.135 |
| 2 | 0.379 | 0.086 | 0.092 |
| 3 | 0.348 | 0.062 | 0.066 |
| 4 | 0.372 | 0.103 | 0.110 |
| **mean** | **0.374 +/- 0.015** | **0.096 +/- 0.021** | **0.102 +/- 0.022** |

Confusion matrix (rows = true, columns = predicted, summed over 5 folds):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 572 | 291 | 267 |
| relaxed | 307 | 235 | 188 |
| uncomfortable | 24 | 23 | 60 |

Plot: `plots/gnn_anatomical_confusion_matrix.png`.

### gnn_random_ablation
the ablation RSCH-3 requires: a random adjacency of the same edge count, drawn once with seed 1, everything else identical.
`n_features` = 4 (per-node channel count), readout `flatten`, balancing: training-fold oversampling.

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.402 | 0.094 | 0.097 |
| 1 | 0.413 | 0.164 | 0.173 |
| 2 | 0.433 | 0.173 | 0.180 |
| 3 | 0.367 | 0.098 | 0.102 |
| 4 | 0.376 | 0.088 | 0.094 |
| **mean** | **0.398 +/- 0.024** | **0.124 +/- 0.037** | **0.129 +/- 0.039** |

Confusion matrix (rows = true, columns = predicted, summed over 5 folds):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 614 | 295 | 221 |
| relaxed | 308 | 261 | 161 |
| uncomfortable | 30 | 21 | 56 |

Plot: `plots/gnn_random_ablation_confusion_matrix.png`.

### gnn_identity
the control the first grooming missed: `A_hat = I`, so the three conv layers run at the same depth and parameter count with message passing switched off. Separates what the graph contributes from what the extra depth contributes, which the random-adjacency ablation cannot do on its own -- a random graph still passes messages.
`n_features` = 4 (per-node channel count), readout `flatten`, balancing: training-fold oversampling.

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.460 | 0.205 | 0.211 |
| 1 | 0.470 | 0.240 | 0.245 |
| 2 | 0.487 | 0.250 | 0.255 |
| 3 | 0.454 | 0.255 | 0.258 |
| 4 | 0.458 | 0.205 | 0.209 |
| **mean** | **0.466 +/- 0.012** | **0.231 +/- 0.022** | **0.235 +/- 0.021** |

Confusion matrix (rows = true, columns = predicted, summed over 5 folds):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 711 | 280 | 139 |
| relaxed | 274 | 328 | 128 |
| uncomfortable | 19 | 33 | 55 |

Plot: `plots/gnn_identity_confusion_matrix.png`.

### gnn_anatomical_meanpool
the first E3 run's defect, kept on the record: same v3 adjacency, mean pool over the 48-node axis instead of flatten. See the readout section below.
`n_features` = 4 (per-node channel count), readout `mean`, balancing: training-fold oversampling.

| fold | macro F1 | kappa | MCC |
|---|---|---|---|
| 0 | 0.278 | 0.036 | 0.046 |
| 1 | 0.297 | 0.034 | 0.040 |
| 2 | 0.278 | 0.024 | 0.031 |
| 3 | 0.253 | -0.012 | -0.014 |
| 4 | 0.238 | 0.011 | 0.015 |
| **mean** | **0.269 +/- 0.021** | **0.019 +/- 0.018** | **0.023 +/- 0.022** |

Confusion matrix (rows = true, columns = predicted, summed over 5 folds):

| true \ pred | attentive | relaxed | uncomfortable |
|---|---|---|---|
| attentive | 219 | 440 | 471 |
| relaxed | 150 | 273 | 307 |
| uncomfortable | 15 | 19 | 73 |

Plot: `plots/gnn_anatomical_meanpool_confusion_matrix.png`.

### random_baseline
`DummyClassifier(strategy="uniform", random_state=0)` on the identical folds through the same `cv.cross_validate`, reused from E2's `catface.ml.random_baseline`. A chance-level reference: any arm below this line is not classifying.

| metric | mean +/- std |
|---|---|
| macro F1 | 0.302 +/- 0.000 |
| kappa | 0.017 +/- 0.001 |
| MCC | 0.018 +/- 0.001 |

Plot: `plots/random_baseline_confusion_matrix.png`.

Comparison plots: `plots/kappa_comparison.png` (the metric the verdict uses) and `plots/macro_f1_comparison.png`.

## The readout defect that caused the re-groom
The first E3 run ended `_Net.forward` in `h.mean(dim=1)`, a mean over the 48-node axis. Node features come from `generalized_procrustes`, and `_center_scale` subtracts each shape's centroid, so the node-axis mean of the GNN's input is zero for all 1967 rows in all four channels. The Research PM measured 2.58e-33 of between-sample variance surviving that readout, against 1.64e-2 for the flattened coordinates the E2 MLP reads. The head saw a near-constant vector; only ReLU asymmetry leaked anything through.
Q1 asks what the adjacency contributes, and that run varied adjacency and readout at once with the readout term dominating, so its numbers could not answer the question. `readout` is now a constructor argument on `_Net`/`GNNClassifier`, defaulting to `flatten` (head `nn.Linear(HIDDEN * 48, N_CLASSES)`). `gnn_anatomical_meanpool` keeps the old `mean` head so the defect stays measurable rather than described: 0.269 macro F1 / 0.019 kappa against the flatten arm's 0.374 / 0.096, same adjacency, same splits, same epochs.
Two more things changed with it. `GNNClassifier.fit` no longer applies `compute_class_weight("balanced")`, and `gnn.run` now passes `oversample=True`, so every arm here balances the way E2's MLP does and the comparison is against like. The Q1 verdict is decided on kappa, not macro F1, because macro F1 between two chance-level classifiers reflects their prediction marginals rather than how well either discriminates.
`tests/test_gnn.py` pins the readout argument: the default is `flatten`, the head is sized `HIDDEN * 48` for it, and with `A_hat = I` the mean readout is permutation-invariant over nodes while the flatten readout is not.

## Required controls
- [x] Untrained-model ONNX export attempted and recorded before any training run, for the flatten head: above.
- [x] All five arms trained on the frozen splits, macro F1 + kappa + MCC + confusion matrix, 5-fold: above.
- [x] Readout identical across the four trained GNN arms except `gnn_anatomical_meanpool`, which exists to vary it; balancing identical across all arms (`oversample=True`, no in-`fit` class weights).
- [x] `gnn_identity` reported in the same detail as the other arms.
- [x] Q1 verdict computed on kappa with macro F1 beside it, both conditions checked: below.
- [x] Adjacency file path and sha256 pinned in `config.json`.
- [x] No `torch_geometric` import anywhere in the run (AST-walk tested, `tests/test_gnn.py`).

## Q1 verdict
Condition A (beats the E2 MLP by more than the anatomical arm's own CV std): anatomical GNN kappa 0.096 vs. MLP 0.155, diff -0.059, anatomical std 0.021: **DOES NOT HOLD**. Macro F1 beside it: 0.374 vs. 0.418.
Condition B (beats the random-adjacency ablation by the same margin): anatomical GNN kappa 0.096 vs. random-adjacency 0.124, diff -0.028, anatomical std 0.021: **DOES NOT HOLD**. Macro F1 beside it: 0.374 vs. 0.398.
At least one condition fails: **Q1 verdict: NO**. RSCH-3's stop condition names a random-adjacency tie, or an anatomical GNN that does not clear the MLP, as a valid answer that has to be reported, not a failed run.
Identity control (`A_hat = I`, message passing off, same depth and parameter count): kappa 0.231 +/- 0.022, macro F1 0.466 +/- 0.012 -- above both graph arms. Switching message passing off raises the score, so nothing the anatomical arm achieves can be credited to the graph: the convolution is smoothing away signal the same layers keep when they act per-node.

## Citations
- CatFLW (landmark scheme, Finka et al.) and the Finka landmark scheme: see `docs/MODEL_REPORT.md` Citations.

