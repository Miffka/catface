"""Plausibility filter for detector output: points inside the box, eyes above the
muzzle, no degenerate configurations, no absurd aspect ratios.

Landmark index groups verified against 4 CatFLW ground-truth labels (one during
initial derivation, three more as a spot check) and cross-checked against
docs/plan-research.md's point counts (8 per eye, 5 per ear, 22 across nose and
whiskers) — see docs/DECISIONS.md. Ear indices aren't needed here and are left
undefined; re-derive them the same way if a later experiment needs them.
"""

import numpy as np

from catface.core.geometry import BoxXYXY, expand_box
from catface.ml.detect_landmarks import LANDMARK_MARGIN

LEFT_EYE = (3, 4, 5, 6, 7, 36, 37, 38)
RIGHT_EYE = (1, 8, 9, 10, 11, 39, 40, 41)
EYE = LEFT_EYE + RIGHT_EYE
_EAR = (22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
MUZZLE = tuple(i for i in range(48) if i not in EYE and i not in _EAR)

INSIDE_BOX_TOLERANCE = 0.15
MIN_SPREAD_FRACTION = 0.05
DUPLICATE_POINT_EPS_PX = 0.5
# Derived from all 2079 CatFLW ground-truth box aspect ratios (width/height):
# min 0.763, p1 0.856, p50 1.019, p99 1.291, max 1.436. Bounds below pad that
# range generously since Roboflow photos are "in the wild," not curated like
# CatFLW — see docs/DECISIONS.md.
ASPECT_RATIO_BOUNDS = (0.5, 1.8)


def fraction_inside(points: np.ndarray, box: BoxXYXY) -> float:
    x1, y1, x2, y2 = box
    inside = (points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2)
    return float(inside.mean())


def check_landmarks(box_xyxy: BoxXYXY, landmarks: np.ndarray, img_w: int, img_h: int) -> list[str]:
    reasons = []

    crop_box = expand_box(box_xyxy, margin=LANDMARK_MARGIN, img_w=img_w, img_h=img_h)
    tol_box = expand_box(crop_box, margin=INSIDE_BOX_TOLERANCE, img_w=img_w, img_h=img_h)
    if fraction_inside(landmarks, tol_box) < 1.0:
        reasons.append("out_of_box")

    eye_y = landmarks[list(EYE), 1].mean()
    muzzle_y = landmarks[list(MUZZLE), 1].mean()
    if eye_y >= muzzle_y:
        reasons.append("eyes_below_muzzle")

    x1, y1, x2, y2 = box_xyxy
    box_w, box_h = x2 - x1, y2 - y1
    spread_w = landmarks[:, 0].max() - landmarks[:, 0].min()
    spread_h = landmarks[:, 1].max() - landmarks[:, 1].min()
    degenerate = spread_w < MIN_SPREAD_FRACTION * box_w or spread_h < MIN_SPREAD_FRACTION * box_h
    if not degenerate:
        diffs = landmarks[:, None, :] - landmarks[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        degenerate = bool((dists < DUPLICATE_POINT_EPS_PX).any())
    if degenerate:
        reasons.append("degenerate")

    aspect = box_w / box_h if box_h else float("inf")
    if not (ASPECT_RATIO_BOUNDS[0] <= aspect <= ASPECT_RATIO_BOUNDS[1]):
        reasons.append("bad_aspect_ratio")

    return reasons


def detector_confidence_proxy(box_xyxy: BoxXYXY, landmarks: np.ndarray, img_w: int, img_h: int) -> float:
    """Derived geometric-plausibility score. Neither OpenVINO model emits a
    confidence value, so this is a stand-in, not a native model output.

    Fraction of landmarks inside the *tight* localizer box (not the
    margin-expanded crop the landmarks model runs on) — the crop is where the
    model's output is normalized to, so nearly every point lands inside it by
    construction, making that fraction always ~1.0 and useless as a signal.
    The tight box isn't part of the model's output space, so this fraction
    actually varies with how well the box fits the face.
    """
    return fraction_inside(landmarks, box_xyxy)
