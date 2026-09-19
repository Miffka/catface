"""Fold-by-fold macro F1 + summed confusion matrix -- identical scoring
logic for all three E2 models (and later RSCH-3's GNN).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    model_fn: Callable[[], Any],
    n_splits: int = 5,
) -> dict:
    """Train `model_fn()` (a fresh, untrained model each fold) on rows
    where split != k and evaluate on split == k, for k in 0..n_splits-1.
    `model_fn`'s return must implement .fit(X, y) and .predict(X)."""
    labels = sorted(set(np.asarray(y).tolist()))
    per_fold_macro_f1 = []
    total_confusion = np.zeros((len(labels), len(labels)), dtype=int)

    for k in range(n_splits):
        train_mask, test_mask = split != k, split == k
        model = model_fn()
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])

        per_fold_macro_f1.append(f1_score(y[test_mask], pred, average="macro", labels=labels))
        total_confusion += confusion_matrix(y[test_mask], pred, labels=labels)

    return {
        "per_fold_macro_f1": [float(v) for v in per_fold_macro_f1],
        "mean_macro_f1": float(np.mean(per_fold_macro_f1)),
        "std_macro_f1": float(np.std(per_fold_macro_f1)),
        "confusion_matrix": total_confusion.tolist(),
        "labels": labels,
    }
