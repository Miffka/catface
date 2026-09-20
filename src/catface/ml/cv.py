"""Fold-by-fold macro F1 + summed confusion matrix -- identical scoring
logic for all three E2 models (and later RSCH-3's GNN).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.utils import resample


def oversample_to_balance(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random oversampling with replacement, per class, up to the majority
    class's count. Majority-class rows pass through untouched. Deterministic
    given `seed`."""
    counts = {c: int((y == c).sum()) for c in np.unique(y)}
    majority = max(counts.values())
    X_blocks, y_blocks = [], []
    for c, count in counts.items():
        X_c, y_c = X[y == c], y[y == c]
        if count < majority:
            X_c, y_c = resample(X_c, y_c, replace=True, n_samples=majority, random_state=seed)
        X_blocks.append(X_c)
        y_blocks.append(y_c)
    return np.concatenate(X_blocks), np.concatenate(y_blocks)


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    model_fn: Callable[[], Any],
    n_splits: int = 5,
    oversample: bool = False,
    oversample_seed: int = 0,
) -> dict:
    """Train `model_fn()` (a fresh, untrained model each fold) on rows
    where split != k and evaluate on split == k, for k in 0..n_splits-1.
    `model_fn`'s return must implement .fit(X, y) and .predict(X). If
    `oversample`, the training fold (only) is rebalanced via
    `oversample_to_balance` before fitting; the test fold is never touched."""
    labels = sorted(set(np.asarray(y).tolist()))
    per_fold_macro_f1 = []
    per_fold_kappa = []
    per_fold_mcc = []
    total_confusion = np.zeros((len(labels), len(labels)), dtype=int)

    for k in range(n_splits):
        train_mask, test_mask = split != k, split == k
        X_train, y_train = X[train_mask], y[train_mask]
        if oversample:
            X_train, y_train = oversample_to_balance(X_train, y_train, seed=oversample_seed)
        model = model_fn()
        model.fit(X_train, y_train)
        pred = model.predict(X[test_mask])
        y_test = y[test_mask]

        per_fold_macro_f1.append(f1_score(y_test, pred, average="macro", labels=labels))
        per_fold_kappa.append(cohen_kappa_score(y_test, pred, labels=labels))
        per_fold_mcc.append(matthews_corrcoef(y_test, pred))
        total_confusion += confusion_matrix(y_test, pred, labels=labels)

    return {
        "per_fold_macro_f1": [float(v) for v in per_fold_macro_f1],
        "mean_macro_f1": float(np.mean(per_fold_macro_f1)),
        "std_macro_f1": float(np.std(per_fold_macro_f1)),
        "per_fold_kappa": [float(v) for v in per_fold_kappa],
        "mean_kappa": float(np.mean(per_fold_kappa)),
        "std_kappa": float(np.std(per_fold_kappa)),
        "per_fold_mcc": [float(v) for v in per_fold_mcc],
        "mean_mcc": float(np.mean(per_fold_mcc)),
        "std_mcc": float(np.std(per_fold_mcc)),
        "confusion_matrix": total_confusion.tolist(),
        "labels": labels,
    }
