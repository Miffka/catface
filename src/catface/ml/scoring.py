"""Binning and bootstrap-CI helpers shared by the E4 (RSCH-4) pose-bin
scoring in `scripts/pose_and_classes.py`.
"""

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 0
BOOTSTRAP_PERCENTILES = (2.5, 97.5)


def quantile_bins(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count quantile bins over all rows, pooled, not per fold.
    Returns (edges of length n_bins+1, per-row bin index).

    Ties (the out-of-domain rows all clamp to theta = 0) make the counts
    only approximately equal; a value sitting exactly on an edge goes to the
    higher bin.
    """
    edges = np.quantile(np.asarray(values, dtype=float), np.linspace(0, 1, n_bins + 1))
    index = np.searchsorted(edges[1:-1], values, side="right")
    return edges, np.clip(index, 0, n_bins - 1)


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    percentiles: tuple[float, float] = BOOTSTRAP_PERCENTILES,
) -> tuple[float, float]:
    """Percentile interval from `b` row-resamples of `n` rows.
    `statistic` takes the resampled row indices. Resamples that leave the
    statistic undefined (a bootstrap sample of one class only) are dropped
    rather than counted as zero."""
    rng = np.random.default_rng(seed)
    values = [statistic(rng.integers(0, n, n)) for _ in range(b)]
    finite = [v for v in values if np.isfinite(v)]
    lo, hi = np.percentile(finite, percentiles)
    return float(lo), float(hi)


def kappa_ci(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int], **kwargs) -> tuple[float, float]:
    """Bootstrap CI for Cohen's kappa over the given rows."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    def statistic(idx: np.ndarray) -> float:
        return cohen_kappa_score(y_true[idx], y_pred[idx], labels=list(labels))

    return bootstrap_ci(statistic, len(y_true), **kwargs)
