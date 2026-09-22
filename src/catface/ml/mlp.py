"""Model (3): a small two-hidden-layer PyTorch MLP on the same
Procrustes-aligned coordinates as model (2) -- not sklearn's MLPClassifier
(RSCH-2 grooming notes, docs/backlog.md). Class balancing is done by
training-fold oversampling (`cv.cross_validate(..., oversample=True)`), not
loss weighting.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from catface.ml import cv, detect_landmarks
from catface.ml.features import coord_features_one, load_features

ROOT = Path(__file__).resolve().parents[3]

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
        torch.manual_seed(0)  # deterministic init, independent of caller's RNG state
        self.epochs = epochs
        self.model = _Net(n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
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


def build_model() -> MLPClassifier:
    return MLPClassifier()


def run() -> dict:
    torch.manual_seed(0)  # deterministic init + training across folds
    features = load_features()
    metrics = cv.cross_validate(
        features.coord_X, features.y, features.split, build_model, oversample=True
    )
    metrics["labels"] = features.classes
    metrics["model"] = "mlp"
    metrics["n_features"] = 96
    return metrics


def save_checkpoint(
    model: MLPClassifier, mean_shape: np.ndarray, classes: list[str], path: Path, **extra
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.model.state_dict(), "mean_shape": mean_shape, "classes": list(classes), **extra},
        path,
    )


def load_checkpoint(path: Path) -> tuple[MLPClassifier, np.ndarray, list[str]]:
    ckpt = torch.load(path, weights_only=False)
    clf = MLPClassifier()
    clf.model.load_state_dict(ckpt["state_dict"])
    clf.model.eval()
    return clf, ckpt["mean_shape"], ckpt["classes"]


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="checkpoint.pt saved by scripts/baselines.py")
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_image", type=Path)
    args = parser.parse_args(argv)

    clf, mean_shape, classes = load_checkpoint(args.checkpoint)

    localizer, landmarks_model = detect_landmarks.load_models(ROOT / "models" / "manifest.json")
    image = cv2.imread(str(args.input_image))
    detect_landmarks.validate_image(image, args.input_image)
    assert image is not None  # validate_image raises above if it is

    box = detect_landmarks.detect_face_box(image, localizer)
    shape = detect_landmarks.detect_landmarks(image, box, landmarks_model)
    coord_X = coord_features_one(shape, mean_shape)[None].astype(np.float32)

    pred_class = classes[int(clf.predict(coord_X)[0])]
    print(f"predicted class: {pred_class}")

    overlay = detect_landmarks.draw_overlay(image, box, shape)
    cv2.putText(overlay, pred_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_image), overlay)
    print(f"wrote {args.output_image}")


if __name__ == "__main__":
    main(sys.argv[1:])
