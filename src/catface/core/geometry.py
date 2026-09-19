"""Letterbox and crop-margin math shared by both tracks, plus Procrustes
alignment (E1) and derived-geometry readouts (E2: eye aspect ratio, ear
angle, muzzle spread) used both as model features and as app-side
readouts."""

from collections.abc import Sequence

import cv2
import numpy as np

BoxXYXY = tuple[float, float, float, float]


def letterbox_square(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio, pad to size x size. Returns
    (letterboxed_image, scale, pad_x, pad_y)."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.zeros((size, size, 3), dtype=image.dtype)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def unletterbox_xyxy(
    box_norm: BoxXYXY,
    size: int,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_w: int,
    orig_h: int,
) -> BoxXYXY:
    """Map a normalized xyxy box from a size x size letterboxed square back
    to original image pixel coordinates."""
    x1n, y1n, x2n, y2n = box_norm
    x1, y1 = (x1n * size - pad_x) / scale, (y1n * size - pad_y) / scale
    x2, y2 = (x2n * size - pad_x) / scale, (y2n * size - pad_y) / scale
    return (
        max(0.0, min(x1, orig_w)),
        max(0.0, min(y1, orig_h)),
        max(0.0, min(x2, orig_w)),
        max(0.0, min(y2, orig_h)),
    )


def expand_box(box_xyxy: BoxXYXY, margin: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Grow a box by margin * (width, height) on each side, clipped to the image."""
    x1, y1, x2, y2 = box_xyxy
    pad_x, pad_y = margin * (x2 - x1), margin * (y2 - y1)
    return (
        max(0, round(x1 - pad_x)),
        max(0, round(y1 - pad_y)),
        min(img_w, round(x2 + pad_x)),
        min(img_h, round(y2 + pad_y)),
    )


def crop_and_resize(image: np.ndarray, box_xyxy: tuple[int, int, int, int], size: int) -> np.ndarray:
    """Crop to box and resize (stretch) to size x size."""
    x1, y1, x2, y2 = box_xyxy
    return cv2.resize(image[y1:y2, x1:x2], (size, size))


def map_points_to_image(points_norm: np.ndarray, box_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Map (N,2) points normalized to a crop back to image pixel coordinates,
    using the same box passed to crop_and_resize."""
    x1, y1, x2, y2 = box_xyxy
    points = points_norm.astype(float).copy()
    points[:, 0] = points[:, 0] * (x2 - x1) + x1
    points[:, 1] = points[:, 1] * (y2 - y1) + y1
    return points


def _center_scale(shape: np.ndarray) -> np.ndarray:
    """Subtract the centroid, divide by Frobenius norm."""
    centered = shape - shape.mean(axis=0)
    return centered / np.linalg.norm(centered)


def procrustes_align(shape: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotate `shape` onto `reference`'s normalized frame (Kabsch/SVD),
    after independently centering and scaling both. Rotation-only:
    reflections are forbidden by flipping the sign of U's last column when
    the fitted rotation would otherwise have a negative determinant.

    This matters because landmarks here are labeled and anatomically
    chiral (left eye vs right eye are distinct indices, not an unordered
    point cloud — see LEFT_EYE/RIGHT_EYE in catface.ml.plausibility). An
    unconstrained Procrustes fit can silently pick a mirror-image solution
    that swaps left and right.
    """
    shape_n = _center_scale(shape)
    reference_n = _center_scale(reference)
    U, _, Vt = np.linalg.svd(shape_n.T @ reference_n)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R = U @ Vt
    return shape_n @ R


def generalized_procrustes(
    shapes: np.ndarray, tol: float = 1e-6, max_iter: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative Generalized Procrustes Analysis over `shapes` (M,48,2).

    Seeds the reference with the first (centered+scaled) shape, repeatedly
    aligns every shape to the current reference and recenters/rescales the
    mean of the aligned shapes into a new reference, until the reference
    stops moving (by `tol`) or `max_iter` is reached. Returns the shapes
    given one final alignment pass to the converged reference, plus that
    reference as the mean shape.
    """
    reference = _center_scale(shapes[0])
    for _ in range(max_iter):
        aligned = np.stack([procrustes_align(shape, reference) for shape in shapes])
        new_reference = _center_scale(aligned.mean(axis=0))
        converged = np.linalg.norm(new_reference - reference) < tol
        reference = new_reference
        if converged:
            break

    aligned = np.stack([procrustes_align(shape, reference) for shape in shapes])
    return aligned, reference


def eye_aspect_ratio(shape: np.ndarray, eye_indices: Sequence[int]) -> float:
    """Eyelid opening / corner-to-corner span. The horizontal axis is the
    pair of eye points farthest apart (robust to point ordering); the
    vertical distance is the largest perpendicular offset of the remaining
    points from that line."""
    points = shape[list(eye_indices)]
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    horizontal = dists[i, j]
    axis = points[j] - points[i]
    normal = np.array([-axis[1], axis[0]]) / horizontal
    vertical = np.abs((points - points[i]) @ normal).max()
    return float(vertical / horizontal)


def ear_angle(
    shape: np.ndarray,
    ear_indices: Sequence[int],
    left_eye_center: np.ndarray,
    right_eye_center: np.ndarray,
) -> float:
    """Signed angle in degrees, base->tip vector vs the inter-ocular axis,
    via atan2 of each vector's heading (positive = counter-clockwise from
    the inter-ocular axis). Base/tip are the ear points nearest/farthest
    from the eye-center midpoint."""
    points = shape[list(ear_indices)]
    midpoint = (np.asarray(left_eye_center) + np.asarray(right_eye_center)) / 2
    dists_to_mid = np.linalg.norm(points - midpoint, axis=1)
    base, tip = points[np.argmin(dists_to_mid)], points[np.argmax(dists_to_mid)]
    ear_vec = tip - base
    ocular_axis = np.asarray(right_eye_center) - np.asarray(left_eye_center)
    angle = np.degrees(
        np.arctan2(ear_vec[1], ear_vec[0]) - np.arctan2(ocular_axis[1], ocular_axis[0])
    )
    return float((angle + 180) % 360 - 180)


def muzzle_spread_ratio(
    shape: np.ndarray,
    muzzle_indices: Sequence[int],
    left_eye_center: np.ndarray,
    right_eye_center: np.ndarray,
) -> float:
    """Geometry-only stand-in for whisker-pad spread: max pairwise distance
    among the muzzle points, divided by inter-ocular distance. Not a named
    whisker-pad index pair (no anatomical index map is published for these
    points) and NOT a validated "tension" measure — see docs/backlog.md
    RSCH-2 grooming notes."""
    points = shape[list(muzzle_indices)]
    spread = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1).max()
    ocular_dist = np.linalg.norm(np.asarray(right_eye_center) - np.asarray(left_eye_center))
    return float(spread / ocular_dist)
