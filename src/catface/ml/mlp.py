"""Model (3): a small two-hidden-layer PyTorch MLP on the same
Procrustes-aligned coordinates as model (2) -- not sklearn's MLPClassifier,
which has no class_weight support (RSCH-2 grooming notes, docs/backlog.md).
"""

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from catface.ml import cv
from catface.ml.features import load_features

HIDDEN = 32
EPOCHS = 200
N_CLASSES = 3


class _Net(nn.Module):
    def __init__(self, n_features: int, n_classes: int = N_CLASSES, hidden: int = HIDDEN):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLPClassifier:
    """.fit(X, y)/.predict(X) wrapper so cv.cross_validate can treat this
    identically to the sklearn models."""

    def __init__(self, n_features: int = 96, epochs: int = EPOCHS):
        self.epochs = epochs
        self.model = _Net(n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        weight_tensor = torch.ones(N_CLASSES)
        weight_tensor[classes] = torch.tensor(weights, dtype=torch.float32)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

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


def build_model() -> MLPClassifier:
    return MLPClassifier()


def run() -> dict:
    torch.manual_seed(0)  # deterministic init + training across folds
    features = load_features()
    metrics = cv.cross_validate(features.coord_X, features.y, features.split, build_model)
    metrics["labels"] = features.classes
    metrics["model"] = "mlp"
    metrics["n_features"] = 96
    return metrics
