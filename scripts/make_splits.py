"""E2: write the stratified 5-fold CV split (cat-emotions-3, three usable
classes) to data/cache/splits.csv, once, so ratios_lr/coords_lr/mlp all
train on identical rows and folds.

Usage: uv run python scripts/make_splits.py
"""

import sys

from catface.ml.splits import write_splits_cache


def main(argv: list[str]) -> None:
    del argv  # no arguments: the split is deterministic (seed=0)

    splits = write_splits_cache()
    print(f"wrote {len(splits)} rows to data/cache/splits.csv")

    breakdown = splits.groupby(["split", "label"]).size().unstack(fill_value=0)
    print("fold size per class:")
    print(breakdown.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
