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
    procrustes_align,
)
from catface.ml.plausibility import LEFT_EAR, LEFT_EYE, MUZZLE, RIGHT_EAR, RIGHT_EYE
from catface.ml.splits import CACHE_PATH, load_splits_cache


class Features(NamedTuple):
    ratio_X: np.ndarray  # (N, 3)
    coord_X: np.ndarray  # (N, 96)
    node_X: np.ndarray  # (N, 48, 4)
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


def ratio_features_one(shape: np.ndarray) -> np.ndarray:
    """(3,): mean(eye_aspect_ratio) and mean(ear_angle) across left/right,
    plus muzzle_spread_ratio, for a single raw (unaligned) (48,2) shape --
    all three are already scale/rotation-derived ratios, not raw
    coordinates, so no Procrustes alignment is needed first."""
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
    return np.array([aspect_ratio, angle, spread])


def ratio_features(raw_shapes: np.ndarray) -> np.ndarray:
    """(N, 3): see `ratio_features_one`, applied per row."""
    return np.stack([ratio_features_one(shape) for shape in raw_shapes])


def coord_features_one(shape: np.ndarray, mean_shape: np.ndarray) -> np.ndarray:
    """(96,): Procrustes-align a single raw (48,2) shape to an already-fitted
    `mean_shape` (single-shape Kabsch align, distinct from `coord_features`'s
    batch `generalized_procrustes` fit), then flatten. For single-image
    inference, where `mean_shape` is the training-time GPA reference."""
    return procrustes_align(shape, mean_shape).reshape(-1)


def coord_features(raw_shapes: np.ndarray) -> np.ndarray:
    """(N, 96): Procrustes-align all shapes together (core.geometry), then
    flatten each aligned (48,2) shape."""
    aligned, _mean_shape = generalized_procrustes(raw_shapes)
    return aligned.reshape(len(aligned), -1)


def node_features(raw_shapes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 48, 4): per-node Procrustes-aligned (x,y) plus offset from the
    mean shape (x,y), for E3's GNN. `mean_shape` is constant across rows but
    varies per node, and the graph conv's weight is shared across all 48
    nodes with no per-node bias -- the offset channel is what hands the
    network each node's own baseline position. Returns (node_X, mean_shape)."""
    aligned, mean_shape = generalized_procrustes(raw_shapes)
    offset = aligned - mean_shape
    return np.concatenate([aligned, offset], axis=-1), mean_shape


def load_features() -> Features:
    raw_shapes, label, split = load_raw_shapes()
    encoder = LabelEncoder()
    y = encoder.fit_transform(label)
    node_X, _mean_shape = node_features(raw_shapes)
    return Features(
        ratio_X=ratio_features(raw_shapes),
        coord_X=coord_features(raw_shapes),
        node_X=node_X,
        y=y,
        split=split,
        classes=list(encoder.classes_),
    )
