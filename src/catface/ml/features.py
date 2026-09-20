"""Feature builders for E2's three baseline models: ratio features (eye
aspect ratio, ear angle, muzzle spread) for model (1), Procrustes-aligned
flattened coordinates for models (2) and (3). Shared here so all three
models train on identical rows, labels, and splits rather than each
re-deriving them.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from catface.core.geometry import (
    ear_angle,
    eye_aspect_ratio,
    generalized_procrustes,
    muzzle_spread_ratio,
)
from catface.ml.plausibility import LEFT_EAR, LEFT_EYE, MUZZLE, RIGHT_EAR, RIGHT_EYE
from catface.ml.splits import CACHE_PATH, load_splits_cache


class Features(NamedTuple):
    ratio_X: np.ndarray  # (N, 3)
    coord_X: np.ndarray  # (N, 96)
    y: np.ndarray  # (N,) int-encoded label
    split: np.ndarray  # (N,) fold index 0-4
    classes: list[str]  # class name per encoded int, in encoder order


def load_raw_shapes() -> tuple[np.ndarray, pd.Series, np.ndarray]:
    """Join data/cache/splits.csv back to landmarks.parquet on image_id.
    Returns (raw_shapes (N,48,2), label, split)."""
    splits = load_splits_cache()
    splits["image_id"] = splits["image_path"].map(lambda p: Path(p).stem)

    landmarks = pd.read_parquet(CACHE_PATH, columns=["image_id", "landmarks"])
    merged = splits.merge(landmarks, on="image_id", how="left")
    missing = merged["landmarks"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} splits.csv rows have no matching landmarks.parquet row: "
            f"{merged.loc[missing, 'image_id'].tolist()[:5]}"
        )

    raw_shapes = np.stack(
        [np.asarray(row, dtype=float).reshape(48, 2) for row in merged["landmarks"]]
    )
    return raw_shapes, merged["label"], merged["split"].to_numpy()


def ratio_features(raw_shapes: np.ndarray) -> np.ndarray:
    """(N, 3): mean(eye_aspect_ratio) and mean(ear_angle) across left/right,
    plus muzzle_spread_ratio, computed directly on raw (unaligned) landmarks
    -- all three are already scale/rotation-derived ratios, not raw
    coordinates, so no Procrustes alignment is needed first."""
    out = np.zeros((len(raw_shapes), 3))
    for i, shape in enumerate(raw_shapes):
        left_eye_center = shape[list(LEFT_EYE)].mean(axis=0)
        right_eye_center = shape[list(RIGHT_EYE)].mean(axis=0)
        aspect_ratio = (
            eye_aspect_ratio(shape, LEFT_EYE) + eye_aspect_ratio(shape, RIGHT_EYE)
        ) / 2
        angle = (
            ear_angle(shape, LEFT_EAR, left_eye_center, right_eye_center)
            + ear_angle(shape, RIGHT_EAR, left_eye_center, right_eye_center)
        ) / 2
        spread = muzzle_spread_ratio(shape, MUZZLE, left_eye_center, right_eye_center)
        out[i] = (aspect_ratio, angle, spread)
    return out


def coord_features(raw_shapes: np.ndarray) -> np.ndarray:
    """(N, 96): Procrustes-align all shapes together (core.geometry), then
    flatten each aligned (48,2) shape."""
    aligned, _mean_shape = generalized_procrustes(raw_shapes)
    return aligned.reshape(len(aligned), -1)


def load_features() -> Features:
    raw_shapes, label, split = load_raw_shapes()
    encoder = LabelEncoder()
    y = encoder.fit_transform(label)
    return Features(
        ratio_X=ratio_features(raw_shapes),
        coord_X=coord_features(raw_shapes),
        y=y,
        split=split,
        classes=list(encoder.classes_),
    )
