"""Letterbox and crop-margin math shared by both tracks, plus Procrustes
alignment (E1), derived-geometry readouts (E2: eye aspect ratio, ear angle,
muzzle spread) and the E4 yaw estimators, used both as model features and as
app-side readouts.

Axis convention and projection model (E4, `docs/backlog.md` RSCH-4 second
grooming pass), stated once and used by every `yaw_*` function below:

    Right-handed world coordinates: `x` to the image right, `y` up, `z`
    toward the camera. Yaw is rotation by `theta` about the vertical axis
    `y`. Projection is weak perspective, orthographic plus a uniform scale,
    so a world point `(x, y, z)` lands at `u = x*cos(theta) + z*sin(theta)`,
    `v = y`.

All three estimators run on `generalized_procrustes` output, so in-plane
roll is already removed and the shapes share a frame. Pitch is not modelled
and not removed: a stated limitation of every estimator here.
"""

from collections.abc import Sequence

import cv2
import numpy as np

BoxXYXY = tuple[float, float, float, float]

# Landmark index groups for the 48-point Finka/CatFLW scheme. Canonical home
# for both tracks: `catface.ml.plausibility` re-exports these rather than
# keeping its own copy (AGENTS.md).
LEFT_EYE = (3, 4, 5, 6, 7, 36, 37, 38)
RIGHT_EYE = (1, 8, 9, 10, 11, 39, 40, 41)
EYE = LEFT_EYE + RIGHT_EYE
LEFT_EAR = (22, 23, 24, 25, 26)
RIGHT_EAR = (27, 28, 29, 30, 31)
EAR = LEFT_EAR + RIGHT_EAR
MUZZLE = tuple(i for i in range(48) if i not in EYE and i not in EAR)

# Node identities from the hand-authored legend at the head of
# `models/graph_edge_schemes/graph_edges_manual_v3.txt`. Each pair is
# (left, right) as that legend names them; 19 and 21 are named but are not a
# matched pair and are excluded.
BILATERAL_PAIRS = (
    (4, 8),  # eyelid outside
    (5, 9),  # eyelid inside
    (6, 10),  # eyelid top
    (7, 11),  # eyelid bottom
    (3, 1),  # pupil bottom
    (36, 41),  # pupil outside
    (37, 40),  # pupil inside
    (38, 39),  # pupil top
    (12, 13),  # nose top
    (14, 15),  # nostril middle
    (44, 45),  # nostril bottom
    (20, 18),  # mouth corner
    (33, 34),  # muzzle middle
    (46, 47),  # muzzle outside
    (32, 35),  # whisker pad outside
    (42, 43),  # whisker pad middle
    (22, 31),  # ear bottom-outside
    (23, 30),  # ear middle-outside
    (24, 29),  # ear top
    (25, 28),  # ear middle-inside
    (26, 27),  # ear bottom-inside
)
# Sagittal-plane points, same legend: nose philtrum, mouth top, mouth, chin.
MIDLINE_POINTS = (16, 17, 0, 2)
# The bilateral pair every yaw estimator measures against: the outer eye
# corners. Skeletally anchored, unlike the pupil points, and free of the ear
# position E1 measured PC1 tracking at r = 0.84.
OCULAR_PAIR = (4, 8)
PHILTRUM = 16
CHIN = 2


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


def _ocular_frame(
    shapes: np.ndarray, pair: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row in-plane frame built on a bilateral pair: (span, u_hat, n_hat,
    midpoint), each (N,) or (N,2). `u_hat` runs from pair[0] to pair[1]."""
    left, right = shapes[:, pair[0]], shapes[:, pair[1]]
    axis = right - left
    span = np.linalg.norm(axis, axis=1)
    u_hat = axis / span[:, None]
    n_hat = np.stack([-u_hat[:, 1], u_hat[:, 0]], axis=1)
    return span, u_hat, n_hat, (left + right) / 2


def yaw_foreshortening_ratio(
    shapes: np.ndarray, pair: tuple[int, int] = OCULAR_PAIR, vertical_index: int = CHIN
) -> np.ndarray:
    """(N,48,2) -> (N,). Family 1 of RSCH-4's second grooming pass: the
    observed bilateral span over a yaw-invariant vertical span.

    A bilateral pair at `(+/-a, y0, z0)` projects to `u = +/-a*cos(theta) +
    z0*sin(theta)`, so the observed span is `w*|cos(theta)|` and the pair's
    own depth cancels. A vertical span is untouched by rotation about the
    vertical axis. So `w_obs / v_obs = r_frontal * |cos(theta)|`, with no
    depth prior anywhere. Unsigned: `cos` is even. Use
    `yaw_midline_offset` for the sign.

    This is the rotation-invariant form of the issue's `w_obs = |u_4 - u_8|`
    and `v_obs = |v_2 - (v_4 + v_8)/2|`: those read raw coordinate
    differences and so assume each row's eye-corner axis is exactly
    horizontal, which `generalized_procrustes` does not give -- it rotates
    each shape to best-fit the *mean*, leaving the ocular axis only
    approximately horizontal. Projecting onto the pair's own axis and its
    normal measures the same two quantities in the shape's own frame,
    the idiom `ear_angle` already uses.
    """
    span, _u_hat, n_hat, midpoint = _ocular_frame(shapes, pair)
    vertical = np.abs(((shapes[:, vertical_index] - midpoint) * n_hat).sum(axis=1))
    return span / vertical


def yaw_midline_offset(
    shapes: np.ndarray, midline_index: int = PHILTRUM, pair: tuple[int, int] = OCULAR_PAIR
) -> np.ndarray:
    """(N,48,2) -> (N,). Family 2: the signed offset of a sagittal point from
    the bilateral pair's midpoint, in units of the pair's observed span.

    For a midline point `M` at `(0, y_M, z_M)` and a pair at depth `z0`, the
    numerator is `(z_M - z0)*sin(theta)` and the denominator `w*cos(theta)`,
    so `s = d_rel * tan(theta)` with `d_rel = (z_M - z0)/w`: signed,
    monotonic, no saturation and no second root. Positive `s` means the
    midline point sits toward pair[1] relative to the pair's midpoint.

    Rotation-invariant form of the issue's `s = (u_M - (u_L + u_R)/2) /
    (u_R - u_L)`, for the same reason as `yaw_foreshortening_ratio`: the
    projection is onto the pair's own axis rather than onto an assumed
    horizontal.
    """
    span, u_hat, _n_hat, midpoint = _ocular_frame(shapes, pair)
    return ((shapes[:, midline_index] - midpoint) * u_hat).sum(axis=1) / span


def yaw_centroid_proxy(shapes: np.ndarray) -> np.ndarray:
    """(N,48,2) -> (N,). E1's head-yaw confound proxy, kept as a named legacy
    quantity: muzzle-centroid distance to the right eye centroid minus the
    same to the left. Superseded as a yaw estimator by the two functions
    above (RSCH-4 second grooming pass) -- it goes as `sin(2*theta)`, so it
    turns over -- but E1's PC2 correlation of r = 0.90 is stated against
    this array, so the summation order is the one `scripts/shape_space.py`
    published and must not be rearranged."""
    muzzle_centroid = shapes[:, list(MUZZLE), :].mean(axis=1)
    left_eye_centroid = shapes[:, list(LEFT_EYE), :].mean(axis=1)
    right_eye_centroid = shapes[:, list(RIGHT_EYE), :].mean(axis=1)
    dist_right = np.linalg.norm(muzzle_centroid - right_eye_centroid, axis=1)
    dist_left = np.linalg.norm(muzzle_centroid - left_eye_centroid, axis=1)
    return dist_right - dist_left


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
