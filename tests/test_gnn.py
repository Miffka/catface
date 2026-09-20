import ast
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from catface.core.graph import random_edges
from catface.ml import gnn, splits

ROOT = Path(__file__).resolve().parents[1]

requires_splits_cache = pytest.mark.skipif(
    not splits.SPLITS_PATH.exists(),
    reason="data/cache/splits.csv not generated yet -- run `uv run python scripts/make_splits.py` first",
)


def test_random_edges_exact_count_no_self_loops_no_duplicates():
    edges = random_edges(48, 76, seed=0)
    assert len(edges) == 76
    assert all(a != b for a, b in edges)
    assert len(set(edges)) == len(edges)


def test_random_edges_deterministic_given_same_seed():
    assert random_edges(48, 76, seed=0) == random_edges(48, 76, seed=0)


def test_random_edges_different_for_different_seed():
    assert random_edges(48, 76, seed=0) != random_edges(48, 76, seed=1)


@pytest.mark.parametrize("readout", ["flatten", "mean"])
def test_gnn_classifier_fit_predict_shapes(readout):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 48, 4))
    y = rng.integers(0, 3, size=12)
    A_hat = np.eye(48)
    clf = gnn.GNNClassifier(A_hat, epochs=3, readout=readout)
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == (12,)
    assert set(pred.tolist()) <= {0, 1, 2}


def test_readout_default_is_flatten_and_sizes_the_head():
    flatten = gnn.GNNClassifier(np.eye(48))
    assert flatten.model.readout == "flatten"
    assert flatten.model.head.in_features == gnn.HIDDEN * 48
    assert gnn.GNNClassifier(np.eye(48), readout="mean").model.head.in_features == gnn.HIDDEN


def test_unknown_readout_rejected():
    with pytest.raises(ValueError):
        gnn.GNNClassifier(np.eye(48), readout="maxpool")


def test_mean_readout_discards_which_node_is_which_flatten_does_not():
    """The re-groom's defect 1, as a property: with A_hat = I the convs are
    per-node, so a mean over the node axis is permutation-invariant -- it
    cannot see where a landmark sits. The flatten readout can."""
    A_hat = torch.eye(48)
    x = torch.randn(1, 48, 4, generator=torch.Generator().manual_seed(0))
    shuffled = x[:, torch.randperm(48, generator=torch.Generator().manual_seed(1))]
    for readout, invariant in (("mean", True), ("flatten", False)):
        net = gnn._Net(A_hat, readout=readout).eval()
        with torch.no_grad():
            assert torch.allclose(net(x), net(shuffled), atol=1e-5) is invariant


@pytest.mark.parametrize("readout", ["flatten", "mean"])
def test_onnx_export_check_success(readout):
    pytest.importorskip("onnx")
    A_hat = np.eye(48)
    with tempfile.TemporaryDirectory() as tmp:
        result = gnn.onnx_export_check(A_hat, Path(tmp) / "test.onnx", readout=readout)
    assert result["success"] is True
    assert result["opset"] == 17
    assert result["input_shape"] == [1, 48, 4]
    assert result["readout"] == readout


def test_no_torch_geometric_import_in_gnn_module():
    tree = ast.parse((ROOT / "src" / "catface" / "ml" / "gnn.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            assert not any(n and "torch_geometric" in n for n in names)


@requires_splits_cache
def test_gnn_run_smoke():
    from catface.core.graph import build_cat_adjacency

    _edges, A_hat = build_cat_adjacency()
    metrics = gnn.run(A_hat, "gnn_anatomical")
    assert {"per_fold_macro_f1", "mean_macro_f1", "confusion_matrix", "model", "n_features"} <= metrics.keys()
    assert len(metrics["per_fold_macro_f1"]) == 5
