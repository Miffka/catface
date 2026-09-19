"""Dataset filter (cat-emotions-3, plausible, three usable classes) plus a
stratified 5-fold CV split, cached to data/cache/splits.csv so every E2
model and RSCH-3's GNN train on identical rows and folds instead of each
redrawing its own.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[3]
CACHE_PATH = ROOT / "data" / "cache" / "landmarks.parquet"
SPLITS_PATH = ROOT / "data" / "cache" / "splits.csv"
DATASET_DIR = ROOT / "data" / "cat-emotions-3"

# Locked in at RSCH-2 grooming (docs/backlog.md), same three classes E1's
# models/class_means.json already covers.
USABLE_CLASSES = ("attentive", "relaxed", "uncomfortable")


def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(CACHE_PATH)
    before = len(df)
    subset = df[
        (df["dataset"] == "cat-emotions-3")
        & (df["plausible"] == True)  # noqa: E712
        & (df["class"].isin(USABLE_CLASSES))
    ].reset_index(drop=True)
    subset["image_path"] = [
        str(DATASET_DIR / split / class_name / f"{image_id}.jpg")
        for image_id, class_name, split in zip(subset["image_id"], subset["class"], subset["split"])
    ]
    print(f"landmarks.parquet: {before} rows total, {len(subset)} after dataset/plausible/class filter")
    return subset


def make_splits(df: pd.DataFrame, seed: int = 0, n_splits: int = 5) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=df.index)
    for k, (_, test_idx) in enumerate(skf.split(df, df["class"])):
        fold.iloc[test_idx] = k
    return pd.DataFrame({"image_path": df["image_path"], "label": df["class"], "split": fold})


def write_splits_cache(path: Path = SPLITS_PATH, seed: int = 0, n_splits: int = 5) -> pd.DataFrame:
    df = load_dataset()
    splits = make_splits(df, seed=seed, n_splits=n_splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False)
    return splits


def load_splits_cache(path: Path = SPLITS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
