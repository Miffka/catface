"""Near-duplicate image detection via a difference hash (dHash) — adjacent-pixel
brightness gradients, not raw brightness vs. the image mean. No imagehash
dependency: PIL already covers the grayscale+resize this needs.

An average hash was tried first and rejected: on these two datasets (different
backgrounds/compositions — grass vs. studio backdrops) it hashed unrelated cats
as exact matches (Hamming distance 0), because 8x8 average brightness collapses
to the same rough light/dark pattern regardless of subject. dHash compares
gradients instead of absolute brightness, which is far more content-sensitive —
verified against that same false-positive pair (0/64 with average hash, 30/64
with dHash) before adopting it. See docs/DECISIONS.md.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def dhash(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    small = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = np.asarray(small, dtype=np.float64)
    return pixels[:, 1:] > pixels[:, :-1]


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def find_near_duplicates(
    paths_a: list[Path], paths_b: list[Path], max_distance: int = 5
) -> list[tuple[Path, Path, int]]:
    hashes_a = np.stack([dhash(Image.open(p)).reshape(-1) for p in paths_a])
    hashes_b = np.stack([dhash(Image.open(p)).reshape(-1) for p in paths_b])
    # (N_a, 1, 64) != (1, N_b, 64) -> (N_a, N_b, 64) -> sum over last axis = Hamming matrix
    distances = np.count_nonzero(hashes_a[:, None, :] != hashes_b[None, :, :], axis=-1)
    matches = []
    for i, j in zip(*np.where(distances <= max_distance)):
        matches.append((paths_a[i], paths_b[j], int(distances[i, j])))
    return matches
