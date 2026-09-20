"""E3 (RSCH-3): a hand-written dense graph conv over the 48-node landmark
graph -- no PyTorch Geometric, its sparse scatter ops break ONNX export and
a fixed 48-node graph makes the dense form identical maths.
"""

import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from catface.ml import cv
from catface.ml.features import load_features

HIDDEN = 32
EPOCHS = 200
N_CLASSES = 3
N_FEATURES = 4  # per-node channel count: aligned (x,y) + offset-from-mean (x,y)
READOUTS = ("flatten", "mean")


class GraphConv(nn.Module):
    """H' = act(A_hat @ H @ W). A_hat is fixed (a buffer, not a parameter) --
    it's what lets this same module run both the anatomical and the
    random-adjacency ablation, just by passing a different A_hat in."""

    def __init__(self, in_features: int, out_features: int, A_hat: torch.Tensor):
        super().__init__()
        self.register_buffer("A_hat", A_hat)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(self.A_hat @ H))


class _Net(nn.Module):
    """`readout="flatten"` concatenates all 48 nodes' hidden vectors and is
    the default: the node features come from `generalized_procrustes`, whose
    `_center_scale` subtracts each shape's centroid, so a mean over the node
    axis is ~zero for every row and the classifier head sees nothing of the
    input. `readout="mean"` keeps that mean pool, solely so the original E3
    defect can be re-measured as an arm of its own (docs/backlog.md RSCH-3
    re-grooming, 2026-09-20)."""

    def __init__(
        self,
        A_hat: torch.Tensor,
        n_features: int = N_FEATURES,
        n_classes: int = N_CLASSES,
        hidden: int = HIDDEN,
        readout: str = "flatten",
    ):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        self.readout = readout
        self.conv1 = GraphConv(n_features, hidden, A_hat)
        self.conv2 = GraphConv(hidden, hidden, A_hat)
        self.conv3 = GraphConv(hidden, hidden, A_hat)
        self.head = nn.Linear(hidden * len(A_hat) if readout == "flatten" else hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        pooled = h.flatten(1) if self.readout == "flatten" else h.mean(dim=1)
        return self.head(pooled)


class GNNClassifier:
    """.fit(X, y)/.predict(X) wrapper, same contract as mlp.MLPClassifier."""

    def __init__(
        self,
        A_hat: np.ndarray,
        n_features: int = N_FEATURES,
        epochs: int = EPOCHS,
        readout: str = "flatten",
    ):
        self.epochs = epochs
        torch.manual_seed(0)
        self.model = _Net(torch.tensor(A_hat, dtype=torch.float32), n_features, readout=readout)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GNNClassifier":
        # No loss weighting: balancing is training-fold oversampling only
        # (`cv.cross_validate(..., oversample=True)`), identical to E2's MLP.
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.model(X_t), y_t)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
        return logits.argmax(dim=1).numpy()


def build_model(A_hat: np.ndarray, readout: str = "flatten"):
    return lambda: GNNClassifier(A_hat, readout=readout)


def run(A_hat: np.ndarray, model_name: str, readout: str = "flatten") -> dict:
    features = load_features()
    metrics = cv.cross_validate(
        features.node_X, features.y, features.split, build_model(A_hat, readout), oversample=True
    )
    metrics["labels"] = features.classes
    metrics["model"] = model_name
    metrics["n_features"] = N_FEATURES  # per-node channel count, not a flattened total like E2's n_features
    metrics["readout"] = readout
    return metrics


def onnx_export_check(A_hat: np.ndarray, out_path: Path, readout: str = "flatten") -> dict:
    """Export an untrained GNN to ONNX and record the result. Must run
    before any training loop -- a literal RSCH-3 acceptance criterion, and
    re-run per head shape: the flatten head is a different graph from the
    mean-pool one the first E3 run exported."""
    result: dict = {
        "success": False,
        "opset": 17,
        "input_shape": [1, 48, 4],
        "exporter": "torch.onnx.export(dynamo=False)",
        "readout": readout,
    }
    try:
        model = _Net(torch.tensor(A_hat, dtype=torch.float32), readout=readout)
        model.eval()
        dummy_input = torch.zeros(*result["input_shape"], dtype=torch.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.onnx.export(
                model,
                dummy_input,
                str(out_path),
                opset_version=result["opset"],
                input_names=["node_features"],
                output_names=["logits"],
                dynamo=False,
            )
        result["warnings"] = [str(w.message) for w in caught]
        result["success"] = True
    except Exception as exc:  # noqa: BLE001 -- record any export failure, don't let it crash the run
        result["error"] = str(exc)
    return result
