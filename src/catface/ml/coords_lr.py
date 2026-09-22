"""Model (2): logistic regression on all 96 Procrustes-aligned coordinates
-- same input as model (3), the MLP.
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from catface.ml import cv, detect_landmarks
from catface.ml.features import coord_features_one, load_features

ROOT = Path(__file__).resolve().parents[3]


def build_model() -> LogisticRegression:
    return LogisticRegression()


def run() -> dict:
    features = load_features()
    metrics = cv.cross_validate(
        features.coord_X, features.y, features.split, build_model, oversample=True
    )
    metrics["labels"] = features.classes
    metrics["model"] = "coords_lr"
    metrics["n_features"] = 96
    return metrics


def save_checkpoint(
    model: LogisticRegression, mean_shape: np.ndarray, classes: list[str], path: Path, **extra
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"model": model, "mean_shape": mean_shape, "classes": list(classes), **extra}, f)


def load_checkpoint(path: Path) -> tuple[LogisticRegression, np.ndarray, list[str]]:
    with path.open("rb") as f:
        ckpt = pickle.load(f)
    return ckpt["model"], ckpt["mean_shape"], ckpt["classes"]


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="checkpoint.pkl saved by scripts/baselines.py")
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_image", type=Path)
    args = parser.parse_args(argv)

    model, mean_shape, classes = load_checkpoint(args.checkpoint)

    localizer, landmarks_model = detect_landmarks.load_models(ROOT / "models" / "manifest.json")
    image = cv2.imread(str(args.input_image))
    detect_landmarks.validate_image(image, args.input_image)
    assert image is not None  # validate_image raises above if it is

    box = detect_landmarks.detect_face_box(image, localizer)
    shape = detect_landmarks.detect_landmarks(image, box, landmarks_model)
    coord_X = coord_features_one(shape, mean_shape)[None]

    pred_class = classes[int(model.predict(coord_X)[0])]
    print(f"predicted class: {pred_class}")

    overlay = detect_landmarks.draw_overlay(image, box, shape)
    cv2.putText(overlay, pred_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_image), overlay)
    print(f"wrote {args.output_image}")


if __name__ == "__main__":
    main(sys.argv[1:])
