"""Letterbox and crop-margin math shared by both tracks. Procrustes lands
here too, at E1 — not needed yet.
"""

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
