"""E0: run the landmark detector over both Roboflow sets once, cache every
prediction to parquet, and report pass/drop stats. Nothing downstream re-runs
the detector — every processed row is kept, plausible or not, so the filter
can be retuned later without another ~2742-image OpenVINO pass.

Usage: uv run python scripts/landmark_cache.py
"""

import argparse
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import dataset_stats
import matplotlib.pyplot as plt
import pandas as pd

from catface.ml.detect_landmarks import (
    detect_face_box,
    detect_landmarks,
    draw_overlay,
    load_models,
    validate_image,
)
from catface.ml.plausibility import check_landmarks, detector_confidence_proxy

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"

DATASETS = {
    "cat-emotions-3": (ROOT / "data" / "cat-emotions-3", ["train"]),
    "cat-emotions-7": (ROOT / "data" / "cat-emotions-7", ["train", "valid"]),
}


def iter_images(dataset_dir: Path, splits: list[str]) -> Iterator[tuple[Path, str, str]]:
    for split in splits:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                for image_path in sorted(class_dir.glob("*.jpg")):
                    yield image_path, class_dir.name, split


def _failed_row(image_path: Path, dataset_name: str, class_name: str, split: str, reason: str) -> dict:
    return {
        "image_id": image_path.stem,
        "dataset": dataset_name,
        "class": class_name,
        "split": split,
        "box_x1": None,
        "box_y1": None,
        "box_x2": None,
        "box_y2": None,
        "landmarks": None,
        "detector_confidence": None,
        "plausible": False,
        "drop_reasons": [reason],
    }


def build_cache(
    datasets: dict[str, tuple[Path, list[str]]], localizer, landmarks_model
) -> pd.DataFrame:
    rows = []
    for dataset_name, (dataset_dir, splits) in datasets.items():
        images = list(iter_images(dataset_dir, splits))
        for n, (image_path, class_name, split) in enumerate(images, 1):
            if n % 200 == 0:
                print(f"  {dataset_name}: {n}/{len(images)}")

            image = cv2.imread(str(image_path))
            try:
                validate_image(image, image_path)
            except ValueError as exc:
                print(f"  skipping unreadable image {image_path}: {exc}")
                rows.append(_failed_row(image_path, dataset_name, class_name, split, "invalid_image"))
                continue

            h, w = image.shape[:2]
            try:
                box = detect_face_box(image, localizer)
                landmarks = detect_landmarks(image, box, landmarks_model)
            except cv2.error as exc:
                # A degenerate localizer box (e.g. clipped to zero width/height
                # against the image bounds) makes the landmarks-stage crop
                # empty. Rare, but real — cache it as a failure rather than
                # crashing the whole batch run.
                print(f"  detection failed on {image_path}: {exc}")
                rows.append(_failed_row(image_path, dataset_name, class_name, split, "detection_failed"))
                continue

            reasons = check_landmarks(box, landmarks, w, h)
            confidence = detector_confidence_proxy(box, landmarks, w, h)
            rows.append(
                {
                    "image_id": image_path.stem,
                    "dataset": dataset_name,
                    "class": class_name,
                    "split": split,
                    "box_x1": box[0],
                    "box_y1": box[1],
                    "box_x2": box[2],
                    "box_y2": box[3],
                    "landmarks": landmarks.reshape(-1).tolist(),
                    "detector_confidence": confidence,
                    "plausible": not reasons,
                    "drop_reasons": reasons,
                }
            )
    return pd.DataFrame(rows)


def sample_overlays(df: pd.DataFrame, n: int, seed: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = df[df["landmarks"].notna()]
    sample = valid.sample(n=min(n, len(valid)), random_state=seed)
    for i, row in enumerate(sample.to_dict("records")):
        dataset_dir, _ = DATASETS[row["dataset"]]
        image_path = dataset_dir / row["split"] / row["class"] / f"{row['image_id']}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        box = (row["box_x1"], row["box_y1"], row["box_x2"], row["box_y2"])
        landmarks = np.array(row["landmarks"]).reshape(48, 2)
        overlay = draw_overlay(image, box, landmarks)
        name = f"{row['dataset']}_{row['split']}_{row['class']}_{i}.jpg".replace(" ", "_")
        cv2.imwrite(str(out_dir / name), overlay)


def _plot_confidence_hist(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["detector_confidence"].dropna(), bins=20)
    ax.set_title("detector_confidence (derived proxy) distribution")
    ax.set_xlabel("fraction of landmarks inside the tight localizer box")
    ax.set_ylabel("images")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_cache_readme(df: pd.DataFrame, readme_path: Path, overlay_count: int) -> None:
    total = len(df)
    plausible = int(df["plausible"].sum())
    pass_rate = plausible / total if total else 0.0
    drop_counts = Counter(reason for reasons in df.loc[~df["plausible"], "drop_reasons"] for reason in (reasons or []))
    stop_triggered = pass_rate < 0.5

    lines = [
        "# data/cache — E0 landmark cache",
        "",
        "## What this is",
        "`landmarks.parquet` holds every detector prediction over cat-emotions-3 (train) and",
        "cat-emotions-7 (train+valid) — every processed row is kept, plausible or not, so a",
        "later retune of the plausibility filter can refilter this cache instead of re-running",
        "the detector. Produced by `uv run python scripts/landmark_cache.py`.",
        "",
        "## Schema",
        "- `image_id`, `dataset`, `class`, `split` — identifiers",
        "- `box_x1, box_y1, box_x2, box_y2` — detected face box, image pixel coordinates",
        "- `landmarks` — flat list of 96 floats, `[x0, y0, x1, y1, ...]` for the 48 points",
        "- `detector_confidence` — **not a native model output**: neither OpenVINO model in",
        "  `models/manifest.json` emits a confidence score. This is a derived geometric",
        "  proxy (fraction of the 48 landmarks that land inside the tight localizer box);",
        "  see `src/catface/ml/plausibility.py:detector_confidence_proxy`.",
        "- `plausible` — result of `src/catface/ml/plausibility.py:check_landmarks`",
        "- `drop_reasons` — list of failed rule names when `plausible` is false",
        "",
        "## Detector run",
        f"Processed {total} images. Plausible: {plausible} ({pass_rate:.1%}). Dropped: {total - plausible}.",
        "Drop reasons (an image can fail more than one rule):",
    ]
    if drop_counts:
        lines += [f"  {reason}: {count}" for reason, count in drop_counts.most_common()]
    else:
        lines.append("  none")
    lines += [
        "",
        "## Stop condition",
        (
            f"{'TRIGGERED' if stop_triggered else 'NOT triggered'} — the detector fails on most images"
            f" is defined here as a plausible-rate under 50%; actual rate is {pass_rate:.1%}."
        ),
        "TRIGGERED means halt the research track and escalate to the Research PM before E1.",
        "",
        "## Plots",
        "`plots/class_distribution_cat-emotions-3.png`, `plots/class_distribution_cat-emotions-7.png`",
        "(written by `scripts/dataset_stats.py`), `plots/detector_confidence_hist.png`.",
        "",
        "## Near-duplicates",
        "See `near_duplicates.csv` and the per-dataset README.txt (written by",
        "`scripts/dataset_stats.py`).",
        "",
        f"## {min(overlay_count, total)}-overlay spot check",
        f"{min(overlay_count, total)} images sampled uniformly at random (not filtered by `plausible`) into",
        "`overlays/`. The human verdict per image is `overlays/overlay.csv`",
        "(`img_fn, annotation_ok, reason`); the summary is in `docs/MODEL_REPORT.md`.",
        "",
        "## 100-image blind relabel",
        "Waived — see `docs/DECISIONS.md` (2026-09-19).",
    ]
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("\n".join(lines))


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlay-count", type=int, default=80)
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="skip the detector; regenerate overlays, plots and READMEs from the existing parquet",
    )
    args = parser.parse_args(argv)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.from_cache:
        df = pd.read_parquet(CACHE_DIR / "landmarks.parquet")
        df["drop_reasons"] = df["drop_reasons"].map(list)  # parquet gives back numpy arrays
    else:
        localizer, landmarks_model = load_models(ROOT / "models" / "manifest.json")
        print("running detector over cat-emotions-3 and cat-emotions-7...")
        df = build_cache(DATASETS, localizer, landmarks_model)
        df.to_parquet(CACHE_DIR / "landmarks.parquet", index=False)

    sample_overlays(df, args.overlay_count, args.seed, CACHE_DIR / "overlays")
    _plot_confidence_hist(df, CACHE_DIR / "plots" / "detector_confidence_hist.png")
    _write_cache_readme(df, CACHE_DIR / "README.md", args.overlay_count)

    for dataset_name in DATASETS:
        subset = df[df["dataset"] == dataset_name]
        drop_counts = Counter(
            reason for reasons in subset.loc[~subset["plausible"], "drop_reasons"] for reason in (reasons or [])
        )
        readme_path = DATASETS[dataset_name][0] / "README.txt"
        dataset_stats.append_section(
            readme_path,
            "Detector run (E0)",
            [
                (
                    f"Processed {len(subset)} images: {int(subset['plausible'].sum())} plausible,"
                    f" {len(subset) - int(subset['plausible'].sum())} dropped."
                ),
                (
                    "detector_confidence is a derived geometric-plausibility proxy, not a native"
                    " model output — see data/cache/README.md."
                ),
            ]
            + ([f"  {reason}: {count}" for reason, count in drop_counts.most_common()] if drop_counts else []),
        )

    total = len(df)
    plausible = int(df["plausible"].sum())
    print(f"cached {total} rows to {CACHE_DIR / 'landmarks.parquet'}")
    print(f"plausible: {plausible} ({plausible / total:.1%})" if total else "no images processed")


if __name__ == "__main__":
    main(sys.argv[1:])
