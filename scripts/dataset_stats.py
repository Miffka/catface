"""E0: licence table, actual-vs-advertised class balance, and near-duplicate
check for the three RSCH-0 data sources. Writes a README.txt into each dataset
folder under data/. No detector, no model I/O — scripts/landmark_cache.py
appends its own section to the two Roboflow README.txt files afterward.

Usage: uv run python scripts/dataset_stats.py
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dedupe import find_near_duplicates

ROOT = Path(__file__).resolve().parent.parent

LICENCES = {
    "cat-emotions-3": (
        "CC BY 4.0",
        "https://universe.roboflow.com/cat-emotion-classification/cat-emotions-cgrxv",
        "confirmed in data/cat-emotions-3/README.roboflow.txt",
    ),
    "cat-emotions-7": (
        "CC BY 4.0",
        "https://universe.roboflow.com/cats-xofvm/cat-emotions",
        "confirmed in data/cat-emotions-7/README.roboflow.txt",
    ),
    "catflw": (
        "CC BY-NC 4.0",
        "https://github.com/martvelge/CatFLW",
        "no licence file ships with the download; confirmed from the dataset's GitHub page",
    ),
}

ADVERTISED = {
    "cat-emotions-3": {"images": 2071, "classes": 3},
    "cat-emotions-7": {"images": 671, "classes": 7},
    "catflw": {"images": 2016},
}


def class_counts(dataset_dir: Path, split: str) -> dict[str, int]:
    split_dir = dataset_dir / split
    counts = {}
    if not split_dir.is_dir():
        return counts
    for class_dir in sorted(split_dir.iterdir()):
        if class_dir.is_dir():
            counts[class_dir.name] = len(list(class_dir.glob("*.jpg")))
    return counts


def catflw_counts(images_dir: Path, labels_dir: Path) -> dict[str, int]:
    return {
        "images": len(list(images_dir.glob("*.png"))),
        "labels": len(list(labels_dir.glob("*.json"))),
    }


def write_readme(readme_path: Path, title: str, sections: list[tuple[str, list[str]]]) -> None:
    """Rewrite the owned sections; keep any "## heading" another script added
    (landmark_cache.py's "Detector run (E0)") so rerun order doesn't matter."""
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    existing = readme_path.read_text() if readme_path.exists() else ""
    owned = {heading for heading, _ in sections}
    foreign = [
        s.rstrip("\n")
        for s in re.split(r"(?m)^(?=## )", existing)[1:]
        if s.split("\n", 1)[0][3:] not in owned
    ]
    parts = [f"# {title}", ""]
    for heading, lines in sections:
        parts.append(f"## {heading}")
        parts.extend(lines)
        parts.append("")
    parts.extend(s + "\n" for s in foreign)
    readme_path.write_text("\n".join(parts) + ("\n" if foreign else ""))


def append_section(readme_path: Path, heading: str, lines: list[str]) -> None:
    """Append a "## heading" section, replacing any existing section with the
    same heading — idempotent under reruns, unlike a plain file append."""
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    existing = readme_path.read_text() if readme_path.exists() else ""
    chunks = re.split(r"(?m)^(?=## )", existing)
    preamble = chunks[0] if not chunks[0].startswith("## ") else ""
    sections = chunks[1:] if preamble else chunks
    sections = [s for s in sections if not s.startswith(f"## {heading}\n")]
    sections = [s.rstrip("\n") + "\n\n" for s in sections]
    if preamble:
        preamble = preamble.rstrip("\n") + "\n\n"
    sections.append(f"## {heading}\n" + "\n".join(lines) + "\n\n")
    readme_path.write_text(preamble + "".join(sections))


def _licence_lines(name: str) -> list[str]:
    licence, url, note = LICENCES[name]
    return [f"{licence} — {url}", note]


def _advertised_vs_actual_lines(name: str, actual_images: int, actual_classes: int | None = None) -> list[str]:
    advertised = ADVERTISED[name]
    lines = [f"Advertised: {advertised['images']} images" + (f", {advertised['classes']} classes" if "classes" in advertised else "")]
    lines.append(f"Actual: {actual_images} images" + (f", {actual_classes} classes/label folders" if actual_classes is not None else ""))
    return lines


def _plot_class_distribution(counts: dict[str, int], name: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_title(f"{name} class distribution")
    ax.set_ylabel("images")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_near_duplicates_csv(out_path: Path, matches: list[tuple[Path, Path, int]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cat-emotions-3_path", "cat-emotions-7_path", "hamming_distance"])
        for path_a, path_b, distance in matches:
            writer.writerow([path_a, path_b, distance])


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    ce3_dir = ROOT / "data" / "cat-emotions-3"
    ce7_dir = ROOT / "data" / "cat-emotions-7"
    catflw_dir = ROOT / "data" / "catflw" / "CatFLW dataset"
    cache_dir = ROOT / "data" / "cache"

    ce3_counts = class_counts(ce3_dir, "train")
    ce7_train = class_counts(ce7_dir, "train")
    ce7_valid = class_counts(ce7_dir, "valid")
    ce7_counts = {k: ce7_train.get(k, 0) + ce7_valid.get(k, 0) for k in set(ce7_train) | set(ce7_valid)}
    catflw_stats = catflw_counts(catflw_dir / "images", catflw_dir / "labels")

    ce3_images = sorted((ce3_dir / "train").rglob("*.jpg"))
    ce7_images = sorted(ce7_dir.rglob("*.jpg"))
    near_dups = find_near_duplicates(ce3_images, ce7_images, max_distance=5)
    _write_near_duplicates_csv(cache_dir / "near_duplicates.csv", near_dups)

    _plot_class_distribution(ce3_counts, "cat-emotions-3", cache_dir / "plots" / "class_distribution_cat-emotions-3.png")
    _plot_class_distribution(ce7_counts, "cat-emotions-7", cache_dir / "plots" / "class_distribution_cat-emotions-7.png")

    near_dup_lines = [
        f"{len(near_dups)} near-duplicate pairs found between cat-emotions-3 and cat-emotions-7 (difference hash, Hamming distance <= 5).",
        "Full list: data/cache/near_duplicates.csv" if near_dups else "None found at this threshold.",
    ]

    write_readme(
        ce3_dir / "README.txt",
        "cat-emotions-3 — dataset notes",
        [
            ("Licence", _licence_lines("cat-emotions-3")),
            (
                "Advertised vs actual",
                _advertised_vs_actual_lines("cat-emotions-3", sum(ce3_counts.values()), len(ce3_counts))
                + [
                    "Advertised as 3 classes; the actual download has 8 raw label folders, train split only, no valid/test split:",
                    *[f"  {name}: {count}" for name, count in ce3_counts.items()],
                ],
            ),
            ("Near-duplicates against cat-emotions-7", near_dup_lines),
        ],
    )

    write_readme(
        ce7_dir / "README.txt",
        "cat-emotions-7 — dataset notes",
        [
            ("Licence", _licence_lines("cat-emotions-7")),
            (
                "Advertised vs actual",
                _advertised_vs_actual_lines("cat-emotions-7", sum(ce7_counts.values()), len(ce7_counts))
                + [
                    "train + valid splits, per-class counts:",
                    *[f"  {name}: train {ce7_train.get(name, 0)}, valid {ce7_valid.get(name, 0)}" for name in sorted(ce7_counts)],
                ],
            ),
            ("Near-duplicates against cat-emotions-3", near_dup_lines),
        ],
    )

    write_readme(
        catflw_dir / "README.txt",
        "CatFLW — dataset notes",
        [
            ("Licence", _licence_lines("catflw")),
            (
                "Advertised vs actual",
                [
                    f"Advertised: {ADVERTISED['catflw']['images']} images.",
                    f"Actual: {catflw_stats['images']} images, {catflw_stats['labels']} label files (48 landmarks + a bounding box each).",
                    "No class labels — CatFLW is used as a shape prior and detector sanity check, not for classification.",
                ],
            ),
        ],
    )

    print(f"cat-emotions-3: {sum(ce3_counts.values())} images across {len(ce3_counts)} label folders (advertised 2071 images / 3 classes)")
    print(f"cat-emotions-7: {sum(ce7_counts.values())} images across {len(ce7_counts)} classes (advertised 671 images / 7 classes)")
    print(f"catflw: {catflw_stats['images']} images, {catflw_stats['labels']} labels (advertised 2016 images)")
    print(f"near-duplicates (cat-emotions-3 vs cat-emotions-7, Hamming <= 5): {len(near_dups)}")
    print("wrote data/cat-emotions-3/README.txt, data/cat-emotions-7/README.txt, data/catflw/CatFLW dataset/README.txt")


if __name__ == "__main__":
    main(sys.argv[1:])
