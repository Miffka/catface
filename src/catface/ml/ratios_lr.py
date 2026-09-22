"""Model (1): logistic regression on three geometric ratios (eye aspect
ratio, ear angle, muzzle spread) -- the Q2 baseline E2 tests a learned
model against.
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

from catface.ml import cv, detect_landmarks
from catface.ml.features import load_features, ratio_features_one

ROOT = Path(__file__).resolve().parents[3]


def build_model() -> LogisticRegression:
    return LogisticRegression()


def run() -> dict:
    features = load_features()
    metrics = cv.cross_validate(
        features.ratio_X, features.y, features.split, build_model, oversample=True
    )
    metrics["labels"] = features.classes
    metrics["model"] = "ratios_lr"
    metrics["n_features"] = 3
    return metrics


def save_checkpoint(model: LogisticRegression, classes: list[str], path: Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"model": model, "classes": list(classes), **extra}, f)


def load_checkpoint(path: Path) -> tuple[LogisticRegression, list[str]]:
    with path.open("rb") as f:
        ckpt = pickle.load(f)
    return ckpt["model"], ckpt["classes"]


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="checkpoint.pkl saved by scripts/baselines.py")
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_image", type=Path)
    args = parser.parse_args(argv)

    model, classes = load_checkpoint(args.checkpoint)

    localizer, landmarks_model = detect_landmarks.load_models(ROOT / "models" / "manifest.json")
    image = cv2.imread(str(args.input_image))
    detect_landmarks.validate_image(image, args.input_image)
    assert image is not None  # validate_image raises above if it is

    box = detect_landmarks.detect_face_box(image, localizer)
    shape = detect_landmarks.detect_landmarks(image, box, landmarks_model)
    ratio_X = ratio_features_one(shape)[None]

    pred_class = classes[int(model.predict(ratio_X)[0])]
    print(f"predicted class: {pred_class}")

    overlay = detect_landmarks.draw_overlay(image, box, shape)
    cv2.putText(overlay, pred_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_image), overlay)
    print(f"wrote {args.output_image}")


if __name__ == "__main__":
    main(sys.argv[1:])
