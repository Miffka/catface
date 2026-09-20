import tempfile
from pathlib import Path

import numpy as np
import pytest

from catface.ml import coords_lr, mlp, random_baseline, ratios_lr, splits
from catface.ml.cv import oversample_to_balance
from catface.ml.features import coord_features, ratio_features
from catface.ml.plausibility import EAR, EYE, MUZZLE

requires_splits_cache = pytest.mark.skipif(
    not splits.SPLITS_PATH.exists(),
    reason="data/cache/splits.csv not generated yet -- run `uv run python scripts/make_splits.py` first",
)

EXPECTED_METRIC_KEYS = {
    "per_fold_macro_f1",
    "mean_macro_f1",
    "std_macro_f1",
    "per_fold_kappa",
    "mean_kappa",
    "std_kappa",
    "per_fold_mcc",
    "mean_mcc",
    "std_mcc",
    "confusion_matrix",
    "model",
    "n_features",
}


def test_splits_cache_round_trip():
    if not splits.CACHE_PATH.exists():
        pytest.skip("data/cache/landmarks.parquet not present locally")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "splits.csv"
        written = splits.write_splits_cache(path=path)
        loaded = splits.load_splits_cache(path=path)
        assert written["image_path"].tolist() == loaded["image_path"].tolist()
        assert written["label"].tolist() == loaded["label"].tolist()
        assert written["split"].tolist() == loaded["split"].tolist()


def _synthetic_shapes(n: int) -> np.ndarray:
    # Same layout style as tests/test_plausibility.py's _valid_landmarks(),
    # with per-shape noise so it's a small set of distinct raw shapes.
    base = np.zeros((48, 2))
    for i, idx in enumerate(EAR):
        base[idx] = (20 + i * 5, 20)
    for i, idx in enumerate(EYE):
        base[idx] = (20 + i * 5, 60)
    for i, idx in enumerate(MUZZLE):
        base[idx] = (20 + i * 6, 140)
    rng = np.random.default_rng(0)
    return np.stack([base + rng.normal(scale=0.5, size=base.shape) for _ in range(n)])


def test_ratio_features_shape_dtype():
    out = ratio_features(_synthetic_shapes(5))
    assert out.shape == (5, 3)
    assert out.dtype == np.float64


def test_coord_features_shape_dtype():
    out = coord_features(_synthetic_shapes(5))
    assert out.shape == (5, 96)
    assert out.dtype == np.float64


@requires_splits_cache
def test_ratios_lr_run_smoke():
    metrics = ratios_lr.run()
    assert EXPECTED_METRIC_KEYS <= metrics.keys()
    assert len(metrics["per_fold_macro_f1"]) == 5


@requires_splits_cache
def test_coords_lr_run_smoke():
    metrics = coords_lr.run()
    assert EXPECTED_METRIC_KEYS <= metrics.keys()
    assert len(metrics["per_fold_macro_f1"]) == 5


@requires_splits_cache
def test_mlp_run_smoke():
    metrics = mlp.run()
    assert EXPECTED_METRIC_KEYS <= metrics.keys()
    assert len(metrics["per_fold_macro_f1"]) == 5


@requires_splits_cache
def test_random_baseline_run_smoke():
    metrics = random_baseline.run()
    assert EXPECTED_METRIC_KEYS <= metrics.keys()
    assert len(metrics["per_fold_macro_f1"]) == 5


def test_oversample_to_balance():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(size=(14, 2)), rng.normal(size=(4, 2)), rng.normal(size=(2, 2))])
    y = np.array([0] * 14 + [1] * 4 + [2] * 2)

    X_bal, y_bal = oversample_to_balance(X, y, seed=0)
    counts = np.bincount(y_bal)
    assert (counts == 14).all()

    X_bal2, y_bal2 = oversample_to_balance(X, y, seed=0)
    np.testing.assert_array_equal(X_bal, X_bal2)
    np.testing.assert_array_equal(y_bal, y_bal2)
