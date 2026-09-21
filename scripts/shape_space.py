"""E1: Procrustes-align the cat-emotions-3 landmark cache, run PCA on the
aligned shapes, and check whether any of the first six components tracks a
pose confound (ear position, head yaw) rather than expression. Also writes
models/class_means.json, the per-class Procrustes mean shapes the app's
/edit endpoint subtracts to compute an expression-change delta.

Usage: uv run python scripts/shape_space.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from catface.core.geometry import generalized_procrustes, yaw_centroid_proxy
from catface.ml.plausibility import EAR, EYE

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "cache" / "landmarks.parquet"
PLOTS_DIR = ROOT / "experiments" / "e1" / "plots"
README_PATH = ROOT / "experiments" / "e1" / "README.md"
CLASS_MEANS_PATH = ROOT / "models" / "class_means.json"

N_COMPONENTS = 6
# Documented threshold (RSCH-1): a PC "visibly tracks" a confound if the
# magnitude of its correlation with that confound's proxy exceeds this.
CONFOUND_R_THRESHOLD = 0.3
# The only cat-emotions-3 label folders treated as real classes (RSCH-4): the
# other five are single/low-double-digit counts and not all real expressions.
USABLE_CLASSES = ("attentive", "relaxed", "uncomfortable")
CLASS_COLORS = {"attentive": "tab:blue", "relaxed": "tab:green", "uncomfortable": "tab:red"}


def load_aligned_shapes() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_parquet(CACHE_PATH)
    total_ce3 = int((df["dataset"] == "cat-emotions-3").sum())
    subset = df[(df["dataset"] == "cat-emotions-3") & (df["plausible"] == True)].reset_index(drop=True)  # noqa: E712
    raw_shapes = np.stack([np.asarray(row, dtype=float).reshape(48, 2) for row in subset["landmarks"]])
    aligned, mean_shape = generalized_procrustes(raw_shapes)
    print(f"cat-emotions-3: {total_ce3} rows total, {len(subset)} plausible, aligned to {aligned.shape}")
    return subset, aligned, mean_shape


def pose_confound_proxies(aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ear_y = aligned[:, list(EAR), 1].mean(axis=1)
    eye_y = aligned[:, list(EYE), 1].mean(axis=1)
    ear_position = ear_y - eye_y
    # Moved to core.geometry at E4 (RSCH-4); the array is unchanged, so E1's
    # published r = 0.90 against PC2 still holds without re-running E1.
    return ear_position, yaw_centroid_proxy(aligned)


def plot_explained_variance(pca: PCA, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    variance_pct = pca.explained_variance_ratio_ * 100
    cumulative_pct = np.cumsum(variance_pct)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, N_COMPONENTS + 1), variance_pct, color="tab:blue", label="per-PC")
    ax2 = ax.twinx()
    ax2.plot(range(1, N_COMPONENTS + 1), cumulative_pct, color="tab:orange", marker="o", label="cumulative")
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance (%)")
    ax2.set_ylabel("cumulative explained variance (%)")
    ax.set_xticks(range(1, N_COMPONENTS + 1))
    ax.set_title("E1: PC explained variance (cat-emotions-3, aligned)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pc_scatter_grid(scores: np.ndarray, classes: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [(0, 1), (2, 3), (4, 5)]
    is_usable = classes.isin(USABLE_CLASSES)

    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(
            scores[~is_usable, i], scores[~is_usable, j], s=6, c="lightgray", label="other (5 tiny folders)"
        )
        for class_name, color in CLASS_COLORS.items():
            mask = (classes == class_name).to_numpy()
            ax.scatter(scores[mask, i], scores[mask, j], s=8, c=color, label=class_name, alpha=0.6)
        ax.set_xlabel(f"PC{i + 1}")
        ax.set_ylabel(f"PC{j + 1}")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("E1: PC pairs colored by class (usable 3 classes highlighted)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confound_scatter(scores: np.ndarray, pc_index: int, proxy: np.ndarray, proxy_name: str, r: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(scores[:, pc_index], proxy, s=6, alpha=0.4)
    ax.set_xlabel(f"PC{pc_index + 1} score")
    ax.set_ylabel(proxy_name)
    ax.set_title(f"PC{pc_index + 1} vs {proxy_name} (r={r:.2f})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def between_within_variance_ratio(scores: np.ndarray, classes: pd.Series) -> list[float]:
    """Per-PC ratio of between-class variance to within-class variance,
    across the three usable classes only. Higher means the classes separate
    more along that PC."""
    ratios = []
    is_usable = classes.isin(USABLE_CLASSES).to_numpy()
    usable_scores = scores[is_usable]
    usable_classes = classes[is_usable].to_numpy()
    grand_mean = usable_scores.mean(axis=0)
    for pc in range(N_COMPONENTS):
        between = 0.0
        within = 0.0
        for class_name in USABLE_CLASSES:
            mask = usable_classes == class_name
            class_scores = usable_scores[mask, pc]
            between += mask.sum() * (class_scores.mean() - grand_mean[pc]) ** 2
            within += ((class_scores - class_scores.mean()) ** 2).sum()
        ratios.append(between / within if within else float("inf"))
    return ratios


def write_readme(
    total_ce3: int,
    n_plausible: int,
    class_counts: dict[str, int],
    pca: PCA,
    confound_correlations: dict[str, tuple[int, float]],
    separation_ratios: list[float],
    confound_plot_written: bool,
) -> None:
    variance_pct = pca.explained_variance_ratio_ * 100
    cumulative_pct = np.cumsum(variance_pct)

    lines = [
        "# experiments/e1 — shape space",
        "",
        "## What this is",
        "Procrustes alignment + PCA over the E0 landmark cache, cat-emotions-3 only,",
        "plausible rows only. Also writes `models/class_means.json`. Produced by",
        "`uv run python scripts/shape_space.py`.",
        "",
        "## Input",
        f"cat-emotions-3 total rows: {total_ce3}. Plausible (used): {n_plausible}.",
        "Per-class plausible counts:",
        *[f"  {name}: {count}" for name, count in class_counts.items()],
        "",
        "## PCA explained variance",
        "| PC | explained variance | cumulative |",
        "|---|---|---|",
        *[
            f"| PC{i + 1} | {variance_pct[i]:.1f}% | {cumulative_pct[i]:.1f}% |"
            for i in range(N_COMPONENTS)
        ],
        "Plot: `plots/pc_explained_variance.png`.",
        "",
        "## Pose confound check",
        f"Plot threshold: |r| > {CONFOUND_R_THRESHOLD}.",
    ]

    for proxy_name, (pc_index, r) in confound_correlations.items():
        lines.append(f"- **{proxy_name}**: strongest at PC{pc_index + 1}, r = {r:.2f}.")

    lines += [
        "",
        (
            "Plot: `plots/pc_confound_scatter.png` (the strongest offending PC vs its proxy)."
            if confound_plot_written
            else "No `pc_confound_scatter.png` — skipped, since no PC cleared the threshold."
        ),
        "",
        "## Class separation",
        "Between-class / within-class variance ratio per PC, computed across the three",
        "usable classes only (attentive, relaxed, uncomfortable):",
        "",
        "| PC | ratio |",
        "|---|---|",
        *[f"| PC{i + 1} | {separation_ratios[i]:.3f} |" for i in range(N_COMPONENTS)],
        "",
        "Plot: `plots/pc_scatter_grid.png`.",
        "",
        "## models/class_means.json",
        f"Covers exactly the three usable classes: {', '.join(USABLE_CLASSES)}.",
        "The other five cat-emotions-3 label folders (`no clear emotion recognizable`,",
        "`sad`, `angry`, `Unlabeled`, `attentive uncomfortable`) are single/low-double-digit",
        "counts and, per RSCH-4's method in `docs/backlog.md`, are not treated as real",
        "expression classes. Each entry is a 48x2 mean of the shared Procrustes-aligned",
        "frame computed above (same alignment the PCA ran on, not a separate one).",
    ]

    README_PATH.parent.mkdir(parents=True, exist_ok=True)
    README_PATH.write_text("\n".join(lines) + "\n")


def write_class_means(subset: pd.DataFrame, aligned: np.ndarray) -> dict[str, int]:
    counts = {}
    class_means = {}
    for class_name in USABLE_CLASSES:
        mask = (subset["class"] == class_name).to_numpy()
        counts[class_name] = int(mask.sum())
        class_means[class_name] = aligned[mask].mean(axis=0).tolist()

    CLASS_MEANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_MEANS_PATH.write_text(json.dumps(class_means, indent=2))
    return counts


def main(argv: list[str]) -> None:
    del argv  # no arguments: GPA and PCA here are deterministic

    subset, aligned, _mean_shape = load_aligned_shapes()

    flattened = aligned.reshape(len(aligned), -1)
    pca = PCA(n_components=N_COMPONENTS)
    scores = pca.fit_transform(flattened)
    print("explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))

    ear_position, head_yaw = pose_confound_proxies(aligned)
    proxies = {"ear_position": ear_position, "head_yaw": head_yaw}

    confound_correlations: dict[str, tuple[int, float]] = {}
    strongest: tuple[str, int, float] | None = None
    for proxy_name, proxy in proxies.items():
        best_pc, best_r = 0, 0.0
        for pc in range(N_COMPONENTS):
            r = float(np.corrcoef(scores[:, pc], proxy)[0, 1])
            print(f"  corr(PC{pc + 1}, {proxy_name}) = {r:.3f}")
            if abs(r) > abs(best_r):
                best_pc, best_r = pc, r
        confound_correlations[proxy_name] = (best_pc, best_r)
        if abs(best_r) > CONFOUND_R_THRESHOLD and (strongest is None or abs(best_r) > abs(strongest[2])):
            strongest = (proxy_name, best_pc, best_r)

    plot_explained_variance(pca, PLOTS_DIR / "pc_explained_variance.png")
    plot_pc_scatter_grid(scores, subset["class"], PLOTS_DIR / "pc_scatter_grid.png")

    confound_plot_written = False
    if strongest is not None:
        proxy_name, pc_index, r = strongest
        plot_confound_scatter(scores, pc_index, proxies[proxy_name], proxy_name, r, PLOTS_DIR / "pc_confound_scatter.png")
        confound_plot_written = True
        print(f"confound plot written: PC{pc_index + 1} vs {proxy_name} (r={r:.3f})")
    else:
        print("no PC exceeds the confound threshold; skipping pc_confound_scatter.png")

    separation_ratios = between_within_variance_ratio(scores, subset["class"])
    print("between/within variance ratio per PC:", [round(r, 3) for r in separation_ratios])

    usable_counts = write_class_means(subset, aligned)
    print(f"wrote {CLASS_MEANS_PATH} for classes: {usable_counts}")

    all_class_counts = subset["class"].value_counts().to_dict()
    total_ce3 = int((pd.read_parquet(CACHE_PATH)["dataset"] == "cat-emotions-3").sum())
    write_readme(total_ce3, len(subset), all_class_counts, pca, confound_correlations, separation_ratios, confound_plot_written)
    print(f"wrote {README_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:])
