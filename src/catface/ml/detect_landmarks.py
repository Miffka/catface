"""Run the hugocornellier/cat-face-landmarks two-stage detector on one image
and write an annotated overlay (face box + 48 landmarks).

Usage: uv run python -m catface.ml.detect_landmarks <input_image> <output_image>
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from catface.core.geometry import (
    BoxXYXY,
    crop_and_resize,
    expand_box,
    letterbox_square,
    map_points_to_image,
    unletterbox_xyxy,
)
from catface.core.graph import Edge, build_cat_edges

ROOT = Path(__file__).resolve().parents[3]
MIN_SIDE = 32
LANDMARK_MARGIN = 0.1  # HF model card: "face box + 0.1 margin" — exact split unconfirmed


def load_models(manifest_path: Path) -> tuple[ov.CompiledModel, ov.CompiledModel]:
    manifest = json.loads(manifest_path.read_text())
    core = ov.Core()

    def compile(name: str) -> ov.CompiledModel:
        path = manifest_path.parent / manifest[name]["file"]
        return core.compile_model(core.read_model(path), "CPU")

    return compile("cat_face_localizer"), compile("cat_face_landmarks")


def validate_image(image: np.ndarray | None, path: Path) -> None:
    if image is None:
        raise ValueError(f"could not read image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected a 3-channel image, got shape {image.shape}: {path}")
    if min(image.shape[:2]) < MIN_SIDE:
        raise ValueError(f"image too small {image.shape[:2]}, need >= {MIN_SIDE}px: {path}")


def _to_model_input(bgr_image: np.ndarray) -> np.ndarray:
    # RGB assumed (not stated on the model card) — swap here if overlays look off.
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb[None]


def detect_face_box(image: np.ndarray, localizer: ov.CompiledModel) -> BoxXYXY:
    h, w = image.shape[:2]
    letterboxed, scale, pad_x, pad_y = letterbox_square(image, size=224)
    output = localizer(_to_model_input(letterboxed))[localizer.output(0)].reshape(-1)
    # Raw tensor order is [x2, y1, x1, y2], not xyxy — confirmed against CatFLW
    # ground-truth boxes; the model card doesn't document this index order.
    box_norm = (output[2], output[1], output[0], output[3])
    return unletterbox_xyxy(box_norm, 224, scale, pad_x, pad_y, orig_w=w, orig_h=h)


def detect_landmarks(image: np.ndarray, box_xyxy: BoxXYXY, landmarks_model: ov.CompiledModel) -> np.ndarray:
    h, w = image.shape[:2]
    crop_box = expand_box(box_xyxy, margin=LANDMARK_MARGIN, img_w=w, img_h=h)
    crop = crop_and_resize(image, crop_box, size=384)
    output = landmarks_model(_to_model_input(crop))[landmarks_model.output(0)]
    points_norm = output.reshape(48, 2)
    return map_points_to_image(points_norm, crop_box)


def draw_overlay(
    image: np.ndarray, box_xyxy: BoxXYXY, landmarks: np.ndarray, edges: list[Edge] | None = None
) -> np.ndarray:
    overlay = image.copy()
    if edges:
        for i, j in edges:
            p1 = tuple(round(v) for v in landmarks[i])
            p2 = tuple(round(v) for v in landmarks[j])
            cv2.line(overlay, p1, p2, (255, 255, 0), 1)
    x1, y1, x2, y2 = (round(v) for v in box_xyxy)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x, y in landmarks:
        cv2.circle(overlay, (round(x), round(y)), 2, (0, 0, 255), -1)
    return overlay


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_image", type=Path)
    parser.add_argument("output_image", type=Path)
    parser.add_argument("--edges", action="store_true", help="draw the anatomical adjacency edges (core.graph)")
    args = parser.parse_args(argv)

    localizer, landmarks_model = load_models(ROOT / "models" / "manifest.json")

    image = cv2.imread(str(args.input_image))
    validate_image(image, args.input_image)
    assert image is not None  # validate_image raises above if it is

    box = detect_face_box(image, localizer)
    landmarks = detect_landmarks(image, box, landmarks_model)
    edges = build_cat_edges() if args.edges else None
    overlay = draw_overlay(image, box, landmarks, edges)

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_image), overlay)
    print(f"box: {tuple(round(v, 1) for v in box)}")
    print(f"wrote {args.output_image}")


if __name__ == "__main__":
    main(sys.argv[1:])
