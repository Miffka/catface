"""E4: head pose and class structure. Answers RSCH-4 / Q3 (how much head
pose contaminates the prediction) and Q4 (whether the three classes are
separable or some collapse).

Everything decidable was decided before this script ran: the bin count, the
verdict metric, the percentile that fixes `r_frontal`, the bootstrap, the
decision rules and both verdict functions are committed in
`src/catface/ml/pose.py` and `src/catface/ml/scoring.py`, per the Research
PM's two grooming passes in `docs/backlog.md`. This file measures; it does
not choose.

Usage: uv run python scripts/pose_and_classes.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shape_space
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from catface.core.geometry import (
    CHIN,
    OCULAR_PAIR,
    PHILTRUM,
    expand_box,
    yaw_centroid_proxy,
    yaw_foreshortening_ratio,
    yaw_midline_offset,
)
from catface.ml import (
    coords_lr,
    cv,
    gnn,
    mlp,
    pose,
    random_baseline,
    ratios_lr,
    scoring,
)
from catface.ml.features import load_features, load_raw_shapes
from catface.ml.splits import SPLITS_PATH, USABLE_CLASSES, load_splits_cache

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "experiments" / "e4"
PLOTS_DIR = RUN_DIR / "plots"
E2_DIR = ROOT / "experiments" / "e2"
E3_DIR = ROOT / "experiments" / "e3"
SEED = 0
N_SPLITS = 5
N_NODES = 48

MODEL_ARMS = ("ratios_lr", "coords_lr", "mlp", "gnn_identity")
CONTROL_ARMS = ("yaw_only_lr", "random_baseline")
ALL_ARMS = MODEL_ARMS + CONTROL_ARMS
# Which committed run each arm's per-fold metrics are checked against
# (control (d)). `yaw_only_lr` is new at E4 and has no source.
REPRODUCTION_SOURCE = {
    "ratios_lr": E2_DIR / "ratios_lr" / "metrics.json",
    "coords_lr": E2_DIR / "coords_lr" / "metrics.json",
    "mlp": E2_DIR / "mlp" / "metrics.json",
    "gnn_identity": E3_DIR / "gnn_identity" / "metrics.json",
    "random_baseline": E2_DIR / "random_baseline" / "metrics.json",
}
PER_FOLD_KEYS = ("per_fold_macro_f1", "per_fold_kappa", "per_fold_mcc")
# Reported the way E2 and E3 report it: a flattened total for the coordinate
# arms, a per-node channel count for the GNN, None for the dummy.
N_FEATURES = {
    "ratios_lr": 3,
    "coords_lr": 96,
    "mlp": 96,
    "gnn_identity": 4,
    "yaw_only_lr": 1,
    "random_baseline": None,
}

ARM_COLORS = {
    "ratios_lr": "tab:blue",
    "coords_lr": "tab:orange",
    "mlp": "tab:green",
    "gnn_identity": "tab:purple",
    "yaw_only_lr": "tab:red",
    "random_baseline": "tab:brown",
}


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- pose measurement -------------------------------------------------------


def measure_pose(aligned: np.ndarray) -> dict:
    """Both yaw estimators plus the legacy centroid proxy, over the whole
    GPA-aligned stack. `r_frontal` and the degrees that follow from it are
    computed at the pre-registered 95th percentile and swept at 90 and 99."""
    ratios = yaw_foreshortening_ratio(aligned)
    r_frontal = {p: pose.frontal_ratio(ratios, p) for p in pose.FRONTAL_PERCENTILE_SWEEP}
    degrees, out_of_domain = pose.yaw_degrees(ratios, r_frontal[pose.FRONTAL_PERCENTILE])
    offsets = yaw_midline_offset(aligned, midline_index=PHILTRUM)
    chin_offsets = yaw_midline_offset(aligned, midline_index=CHIN)
    return {
        "ratio": ratios,
        "r_frontal": r_frontal,
        "theta": degrees,
        "out_of_domain": out_of_domain,
        "s": offsets,
        "s_chin": chin_offsets,
        "d_rel": pose.relative_depth(np.abs(offsets), degrees),
        "d_rel_chin": pose.relative_depth(np.abs(chin_offsets), degrees),
        "centroid_proxy": yaw_centroid_proxy(aligned),
    }


def ratio_of_theta_edge(theta_edge: float, r_frontal: float) -> float:
    """The inverse of `pose.yaw_degrees` at the primary percentile: a bin edge
    in exact foreshortening-ratio units, which is what `core/states.py` would
    consume and what carries no modelling assumption."""
    return float(r_frontal * np.cos(np.radians(theta_edge)))


def degrees_of_ratio_edge(ratio_edge: float, r_frontal: dict[float, float]) -> dict[float, float]:
    """One ratio edge, in degrees under each percentile setting. Bin edges are
    quantiles of theta at the primary percentile; converting them back to the
    ratio makes them percentile-free, which is what the app consumes, and the
    degree figures then carry the percentile that produced them."""
    return {
        p: float(pose.yaw_degrees(np.array([ratio_edge]), r)[0][0]) for p, r in r_frontal.items()
    }


def occupancy(bin_index: np.ndarray, split: np.ndarray, minority: np.ndarray, n_bins: int) -> dict:
    """Per-bin and per-(bin x fold) counts of the minority class, taken
    immediately after the edges are frozen and before any arm is scored."""
    per_bin = [int(minority[bin_index == b].sum()) for b in range(n_bins)]
    per_cell = [
        [int(minority[(bin_index == b) & (split == k)].sum()) for k in range(N_SPLITS)]
        for b in range(n_bins)
    ]
    return {"per_bin": per_bin, "per_bin_fold": per_cell, "empty_cells": int(np.sum(np.array(per_cell) == 0))}


def freeze_bins(theta: np.ndarray, split: np.ndarray, y: np.ndarray, uncomfortable: int) -> dict:
    """Quantile bins on theta, with the pre-registered 3-bin fallback: if any
    (bin x fold) cell holds zero `uncomfortable` rows at 4 bins, drop to 3.
    Both occupancy tables are recorded either way."""
    minority = y == uncomfortable
    edges, index = scoring.quantile_bins(theta, pose.N_BINS)
    four = occupancy(index, split, minority, pose.N_BINS)
    fallback = four["empty_cells"] > 0
    result = {"n_bins": pose.N_BINS, "occupancy_4_bins": four, "fallback_taken": fallback}
    if fallback:
        edges, index = scoring.quantile_bins(theta, pose.FALLBACK_N_BINS)
        result["n_bins"] = pose.FALLBACK_N_BINS
        result["occupancy_3_bins"] = occupancy(index, split, minority, pose.FALLBACK_N_BINS)
    result["theta_edges"] = [float(e) for e in edges]
    result["bin_index"] = index
    return result


# --- arms -------------------------------------------------------------------


def build_arms(features, theta: np.ndarray) -> dict:
    """(feature block, model factory) per arm. Each model module's own
    `build_model` is called directly rather than its `run()`, because E4
    needs `return_oof=True` and `run()` does not pass it."""
    return {
        "ratios_lr": (features.ratio_X, ratios_lr.build_model),
        "coords_lr": (features.coord_X, coords_lr.build_model),
        "mlp": (features.coord_X, mlp.build_model),
        "gnn_identity": (features.node_X, gnn.build_model(np.eye(N_NODES), readout="flatten")),
        # Control (c): logistic regression on the binning quantity alone, in
        # the same configuration as the other two LR arms. Separates "pose
        # degrades the model" from "pose predicts the label".
        "yaw_only_lr": (theta[:, None], ratios_lr.build_model),
        "random_baseline": (features.ratio_X, random_baseline.build_model),
    }


def run_arm(name: str, X: np.ndarray, build_model, y: np.ndarray, split: np.ndarray) -> dict:
    if name == "mlp":
        torch.manual_seed(SEED)  # exactly as mlp.run does, and in the same place
    return cv.cross_validate(X, y, split, build_model, oversample=True, return_oof=True)


def reproduction_check(name: str, metrics: dict) -> dict:
    """Control (d): E2's and E3's metrics.json hold no per-row predictions, so
    every arm is re-run here. The re-run's per-fold macro F1, kappa and MCC
    are compared against the committed numbers. A hidden mismatch is the
    defect; a recorded one is not."""
    source = REPRODUCTION_SOURCE.get(name)
    if source is None:
        return {"source": None, "max_abs_delta": None, "exact": None}
    committed = json.loads(source.read_text())
    deltas = [
        abs(a - b)
        for key in PER_FOLD_KEYS
        for a, b in zip(metrics[key], committed[key], strict=True)
    ]
    return {
        "source": str(source.relative_to(ROOT)),
        "max_abs_delta": float(max(deltas)),
        "exact": max(deltas) == 0.0,
    }


# --- per-bin scoring --------------------------------------------------------


def bin_metrics(
    y: np.ndarray,
    oof: np.ndarray,
    split: np.ndarray,
    bin_index: np.ndarray,
    labels: list[int],
    bins: dict,
    r_frontal: dict[float, float],
    ratio: np.ndarray,
) -> list[dict]:
    """Per bin: kappa with a bootstrap CI, macro F1, accuracy beside that
    bin's own majority-class rate, n per class, and the five per-fold kappas
    as the secondary sanity check."""
    overall_counts = np.bincount(y, minlength=len(labels))
    rows = []
    for b in range(bins["n_bins"]):
        mask = bin_index == b
        y_b, pred_b = y[mask], oof[mask]
        counts = np.bincount(y_b, minlength=len(labels))
        # theta ascending, so bin b's frontal-side edge is its lower theta
        # edge, which is its *upper* ratio edge.
        theta_edge = bins["theta_edges"][b]
        ratio_edge = ratio_of_theta_edge(theta_edge, r_frontal[pose.FRONTAL_PERCENTILE])
        rows.append(
            {
                "index": b,
                "n": int(mask.sum()),
                "class_counts": counts.tolist(),
                "kappa": float(cohen_kappa_score(y_b, pred_b, labels=labels)),
                "kappa_ci": list(scoring.kappa_ci(y_b, pred_b, labels)),
                "macro_f1": float(f1_score(y_b, pred_b, average="macro", labels=labels)),
                "accuracy": float(accuracy_score(y_b, pred_b)),
                "majority_rate": float(counts.max() / counts.sum()),
                "tv": scoring.total_variation(counts, overall_counts),
                "theta_range": [theta_edge, bins["theta_edges"][b + 1]],
                "edge_ratio": ratio_edge,
                "edge_degrees": degrees_of_ratio_edge(ratio_edge, r_frontal),
                "per_fold_kappa": [
                    float(cohen_kappa_score(y[mask & (split == k)], oof[mask & (split == k)], labels=labels))
                    for k in range(N_SPLITS)
                ],
            }
        )
    return rows


def as_json(value):
    """kappa_ci tuples and numpy scalars -> plain JSON. edge_degrees keys are
    percentiles, which JSON turns into strings; that is fine to read back."""
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


# --- Q4 ---------------------------------------------------------------------


def pair_statistics(y: np.ndarray, oof: np.ndarray, cm: np.ndarray, classes: list[str]) -> list[dict]:
    """All three pairs against the pre-registered merge thresholds: cross-talk
    >= 0.25 **and** the pairwise 2x2 kappa CI contains 0, both required.

    The 2x2 kappa is computed over the rows whose true *and* predicted labels
    both fall in the pair, which is the sub-problem "told these two apart".
    """
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            mask = np.isin(y, [i, j]) & np.isin(oof, [i, j])
            kappa = float(cohen_kappa_score(y[mask], oof[mask], labels=[i, j]))
            ci = list(scoring.kappa_ci(y[mask], oof[mask], [i, j]))
            talk = scoring.crosstalk(cm, i, j)
            pairs.append(
                {
                    "classes": [i, j],
                    "names": [classes[i], classes[j]],
                    "n": int(mask.sum()),
                    "crosstalk": talk,
                    "kappa": kappa,
                    "kappa_ci": ci,
                    "candidate": talk >= scoring.CROSSTALK_THRESHOLD and scoring.contains_zero(ci),
                }
            )
    return pairs


def merged_labels(values: np.ndarray, pair: list[int], labels: list[int]) -> np.ndarray:
    """Collapse `pair` onto its first member and relabel to 0..1. The mapping
    is fixed by `labels`, not by which values happen to occur, so the true
    labels and a set of predictions always merge the same way."""
    surviving = [v for v in labels if v != pair[1]]
    lookup = {v: i for i, v in enumerate(surviving)}
    lookup[pair[1]] = lookup[pair[0]]
    return np.array([lookup[v] for v in values])


def run_merge(pair: dict, arm: str, X: np.ndarray, build_model, y: np.ndarray, oof: np.ndarray, split: np.ndarray) -> tuple[dict, dict]:
    """The three 2-class quantities side by side: the retrained merged model,
    the post-hoc collapse of the 3-class predictions, and a 2-class uniform
    dummy under the merged prior. A 2-class kappa is not comparable to a
    3-class one, so this is the only valid comparison."""
    labels = sorted(set(y.tolist()))
    y_merged = merged_labels(y, pair["classes"], labels)
    collapsed = merged_labels(oof, pair["classes"], labels)
    merged_label_set = sorted(set(y_merged.tolist()))

    retrained = run_arm(arm, X, build_model, y_merged, split)
    predicted = np.array(retrained["oof_pred"])
    if not set(predicted.tolist()) <= set(merged_label_set):
        # mlp/gnn keep a 3-unit head under merged labels; a prediction of the
        # vanished third class would be silently dropped by the 2-class kappa.
        raise ValueError(f"merged {arm} predicted classes outside {merged_label_set}")
    dummy = cv.cross_validate(
        X, y_merged, split, random_baseline.build_model, oversample=True, return_oof=True
    )
    merge = {
        "names": pair["names"],
        "classes": pair["classes"],
        "candidate": pair["candidate"],
        "retrained_kappa": float(cohen_kappa_score(y_merged, predicted, labels=merged_label_set)),
        "collapsed_kappa": float(cohen_kappa_score(y_merged, collapsed, labels=merged_label_set)),
        "dummy_kappa": float(
            cohen_kappa_score(y_merged, np.array(dummy["oof_pred"]), labels=merged_label_set)
        ),
    }
    return merge, retrained


# --- plots ------------------------------------------------------------------


def plot_per_bin_kappa(per_arm_bins: dict[str, list[dict]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    n_bins = len(next(iter(per_arm_bins.values())))
    x = np.arange(n_bins)
    width = 0.8 / len(per_arm_bins)
    for i, (name, rows) in enumerate(per_arm_bins.items()):
        kappas = [r["kappa"] for r in rows]
        lo = np.clip([r["kappa"] - r["kappa_ci"][0] for r in rows], 0, None)
        hi = np.clip([r["kappa_ci"][1] - r["kappa"] for r in rows], 0, None)
        ax.bar(x + i * width, kappas, width, yerr=[lo, hi], capsize=3, label=name, color=ARM_COLORS[name])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([f"bin {b}" for b in range(n_bins)])
    ax.set_ylabel("out-of-fold kappa (95% bootstrap CI)")
    ax.set_title("E4: kappa per yaw bin, all arms (bin 0 = most frontal)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_estimator_agreement(theta: np.ndarray, offsets: np.ndarray, rho: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(np.tan(np.radians(theta)), np.abs(offsets), s=5, alpha=0.3)
    ax.set_xlabel("tan(theta), family 1 (foreshortening)")
    ax.set_ylabel("|s|, family 2 (midline offset)")
    ax.set_title(f"E4 internal control: the two estimators against each other (Spearman rho = {rho:.3f})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pose_distributions(measured: dict, bins: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(measured["theta"], bins=60, color="tab:blue")
    for edge in bins["theta_edges"][1:-1]:
        axes[0].axvline(edge, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("theta (degrees, 95th-percentile r_frontal)")
    axes[0].set_title("family 1, with the frozen bin edges")

    axes[1].hist(measured["s"], bins=60, color="tab:red")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_xlabel("s, signed midline offset")
    axes[1].set_title("family 2, sign = turn direction")

    finite = measured["d_rel"][np.isfinite(measured["d_rel"])]
    axes[2].hist(finite, bins=60, range=(0, np.percentile(finite, 99)), color="tab:green")
    axes[2].set_xlabel("d_rel = |s| / tan(theta)")
    axes[2].set_title("measured philtrum depth, per row")
    for ax in axes:
        ax.set_ylabel("rows")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_yaw_extremes_montage(
    raw_shapes: np.ndarray, image_paths: list[str], measured: dict, out_path: Path
) -> list[dict]:
    """The most-negative, near-zero and most-positive rows by family 2's
    signed `s`, rendered from their source images. The sign-to-turn-direction
    mapping is read off this montage rather than asserted from the maths: the
    image y-axis points down and the GPA frame's global orientation is
    arbitrary, so which way a positive `s` turns is a question about the
    pictures."""
    order = np.argsort(measured["s"])
    middle = len(order) // 2
    picks = {
        "most negative s": order[:3],
        "near zero s": order[middle - 1 : middle + 2],
        "most positive s": order[-3:][::-1],
    }

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    for row, (title, indices) in enumerate(picks.items()):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            image = cv2.imread(image_paths[idx])
            points = raw_shapes[idx]
            # Crop to the landmarks so the head is large enough to read a turn
            # direction off; `expand_box` is core's, not a local copy.
            x1, y1, x2, y2 = expand_box(
                (points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()),
                margin=0.6,
                img_w=image.shape[1],
                img_h=image.shape[0],
            )
            ax.imshow(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            points = points - (x1, y1)
            ax.scatter(points[:, 0], points[:, 1], s=8, c="red")
            for name, index in (("4", OCULAR_PAIR[0]), ("8", OCULAR_PAIR[1]), ("16", PHILTRUM), ("2", CHIN)):
                ax.scatter(*points[index], s=40, c="yellow", edgecolors="black")
                ax.annotate(name, points[index], color="yellow", fontsize=9, fontweight="bold")
            ax.set_title(
                f"{title}\ns = {measured['s'][idx]:+.4f}, ratio = {measured['ratio'][idx]:.3f}, "
                f"theta = {measured['theta'][idx]:.1f} deg"
                f"{' (out of domain)' if measured['out_of_domain'][idx] else ''}",
                fontsize=9,
            )
            ax.axis("off")
    fig.suptitle(
        "E4: family 2's signed s at both extremes and at zero (points 4/8 eye corners, 16 "
        f"philtrum, 2 chin). Every theta here is at the {pose.FRONTAL_PERCENTILE:.0f}th-percentile "
        "`r_frontal`; see config.json for the 90th/99th sweep."
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return [
        {"group": title, "row": int(idx), "image_path": image_paths[idx],
         "s": float(measured["s"][idx]), "theta": float(measured["theta"][idx])}
        for title, indices in picks.items()
        for idx in indices
    ]


# --- README -----------------------------------------------------------------

# What the montage showed, written after looking at
# `yaw_extremes_manual_review.png` and not derived from the algebra. The
# grooming asks for the sign-to-turn-direction mapping to be read off the
# pictures, because the image y-axis points down and the GPA frame's global
# orientation is arbitrary.
MONTAGE_READING = [
    ("Two of the three most-negative-`s` rows are landmark failures, not poses. In the "
     "most-negative row of all (`s` = -0.92) point 4 sits on one cat's eye and point 8 on a "
     "second cat's eye in the same photo; the next (`s` = -0.60) spreads the 48 points across "
     "three kittens in a basket. Both passed E0's plausibility filter. The extreme tail of "
     "family 2 is therefore populated by detector failures rather than by extreme yaw, and any "
     "statistic that leans on that tail is reading the detector."),
    ("The rows that are readable do carry the sign. The third most-negative row is a single "
     "ginger cat with its head turned toward the image right, and the two readable "
     "most-positive rows are cats with their heads turned toward the image left. Read off the "
     "pictures: **positive `s` means the head is turned toward the image left**, which is the "
     "cat's own right, and negative `s` toward the image right. That is the direction "
     "`u_hat` running from point 4 (the cat's left outer eye corner, on the image right in a "
     "frontal photo) to point 8 predicts, so the convention is consistent, but it is stated "
     "here because the montage shows it, not because the algebra says it."),
    ("The near-zero-`s` panels are where family 1 comes apart. The middle panel is a black cat "
     "facing the camera square on -- no turn a human would call yaw -- and family 1 puts it at "
     "38.9 degrees, above the median of the whole dataset. Its face is simply narrow relative "
     "to its chin-to-eye span. This is the brachycephalic/dolichocephalic confound named in the "
     "grooming, seen in one picture, and it is the mechanism behind the low rho below."),
]

REJECTED_FAMILIES = [
    ("Family 3, mirror-Procrustes residual",
     ("Swap all 21 bilateral pairs, align the relabelled shape to the original, read the "
      "residual. Rejected at grooming: one non-negative scalar mixing yaw with expression "
      "asymmetry, one ear forward and landmark error, with no way to decompose it, and unsigned "
      "like family 1 without family 1's freedom from a depth prior.")),
    ("Per-row least-squares fit of theta against all 21 pairs",
     ("Family 1 generalised, and right if a 3D reference cat existed. None does: 21 frontal spans "
      "each carry the same population-versus-individual width confound, so the extra precision "
      "buys nothing against a shared systematic error that dominates it. The obvious upgrade if "
      "a 3D model ever lands.")),
    ("The E1 centroid proxy",
     ("Kept as a named legacy column, not as a binning quantity. It goes as sin(2*theta), so it "
      "turns over and one value has two candidate angles; the first grooming pass's saturation "
      "machinery existed only to manage that. See below.")),
]


def reproduction_reading(config: dict, all_metrics: dict) -> str:
    """Reads the reproduction table rather than narrating it, so the sentence
    cannot drift from `config.json`."""
    checkable = [n for n in ALL_ARMS if config["reproduction_check"][n]["source"]]
    mismatched = [n for n in checkable if not config["reproduction_check"][n]["exact"]]
    if not mismatched:
        return (
            f"All {len(checkable)} checkable arms reproduce the committed per-fold macro F1, "
            "kappa and MCC bit-for-bit, so this run's out-of-fold predictions belong to the same "
            "models E2 and E3 reported."
        )
    details = "; ".join(
        f"`{name}` by up to {config['reproduction_check'][name]['max_abs_delta']:.4f} "
        f"(mean kappa {json.loads(REPRODUCTION_SOURCE[name].read_text())['mean_kappa']:.3f} -> "
        f"{all_metrics[name]['mean_kappa']:.3f})"
        for name in mismatched
    )
    return (
        f"{len(checkable) - len(mismatched)} of the {len(checkable)} checkable arms reproduce "
        f"bit-for-bit. These do not: {details}. Recorded rather than hidden, per the "
        "pre-registered control (d): a hidden mismatch is the defect, a recorded one is not."
    )


def monotone_reading(all_metrics: dict, verdict_arm: str) -> str:
    """Which arms decline monotonically across the bins. Computed, because a
    trend read off a table by eye is exactly what the pre-registered
    degradation rule exists to replace."""
    declining = [
        name
        for name in MODEL_ARMS
        if all(
            b["kappa"] > c["kappa"]
            for b, c in zip(all_metrics[name]["per_bin"], all_metrics[name]["per_bin"][1:])
        )
    ]
    if not declining:
        return (
            "No model arm's per-bin kappa declines monotonically from bin 0 to the last bin, so "
            "there is not even an uncontrolled trend here for the degradation rule to have missed."
        )
    named = ", ".join(f"`{n}`" for n in declining)
    return (
        f"One thing the pre-registered rule does not capture and the tables do: {named} "
        "declines monotonically across the bins"
        + (
            ", and it is the one arm whose three features are themselves geometric ratios of the "
            "same kind as the binning quantity, so the likeliest reading is shared measurement "
            "rather than pose. "
            if declining == ["ratios_lr"]
            else ". "
        )
        + f"The verdict arm `{verdict_arm}` does not, and the CIs overlap throughout, which is "
        "why the rule did not fire. Recorded here so the observation is on the page rather than "
        "left for a reader to find and over-read."
    )


def per_fold_table(metrics: dict) -> list[str]:
    per_fold = zip(metrics["per_fold_macro_f1"], metrics["per_fold_kappa"], metrics["per_fold_mcc"])
    return [
        "| fold | macro F1 | kappa | MCC |",
        "|---|---|---|---|",
        *[f"| {k} | {f1:.3f} | {kappa:.3f} | {mcc:.3f} |" for k, (f1, kappa, mcc) in enumerate(per_fold)],
        (f"| **mean** | **{metrics['mean_macro_f1']:.3f} +/- {metrics['std_macro_f1']:.3f}** | "
         f"**{metrics['mean_kappa']:.3f} +/- {metrics['std_kappa']:.3f}** | "
         f"**{metrics['mean_mcc']:.3f} +/- {metrics['std_mcc']:.3f}** |"),
        (f"| **pooled out-of-fold** | | **{metrics['pooled_kappa']:.3f}** (CI "
         f"{metrics['pooled_kappa_ci'][0]:.3f} to {metrics['pooled_kappa_ci'][1]:.3f}) | |"),
        "",
    ]


def per_bin_table(metrics: dict, classes: list[str]) -> list[str]:
    rows = [
        "| bin | n | kappa (95% CI) | macro F1 | accuracy | that bin's majority rate | "
        + " | ".join(f"n {c}" for c in classes)
        + " | per-fold kappa |",
        "|---|---|---|---|---|---|" + "|".join(["---"] * (len(classes) + 1)) + "|",
    ]
    for b in metrics["per_bin"]:
        rows.append(
            f"| {b['index']} | {b['n']} | {b['kappa']:.3f} ({b['kappa_ci'][0]:.3f} to "
            f"{b['kappa_ci'][1]:.3f}) | {b['macro_f1']:.3f} | {b['accuracy']:.3f} | "
            f"{b['majority_rate']:.3f} | " + " | ".join(str(c) for c in b["class_counts"]) + " | "
            + ", ".join(f"{k:.2f}" for k in b["per_fold_kappa"]) + " |"
        )
    return rows + [""]


def write_readme(
    config: dict,
    all_metrics: dict,
    bins: dict,
    measured: dict,
    classes: list[str],
    class_counts: dict,
    verdict_arm: str,
    q3_status: str,
    q3_text: str,
    q4_status: str,
    q4_text: str,
    merge: dict,
    pairs: list[dict],
    candidates: list[dict],
) -> None:
    ratio_edges = config["ratio_edges"]
    theta_edges = bins["theta_edges"]
    occupancy_4 = bins["occupancy_4_bins"]
    d = config["d_rel"]

    lines = [
        "# experiments/e4 — class structure and pose",
        "",
        "## What this is",
        ("RSCH-4 asks Q3 (how much head pose contaminates the prediction) and Q4 (whether the "
         "three classes are separable or some collapse). Six arms on the same 1967 cat-emotions-3 "
         "rows and the same frozen five folds E2 and E3 used, each scored inside four bins of a "
         "pose estimator built from the landmark map. Method, bin count, verdict metric, "
         "thresholds and decision rules are the Research PM's two grooming passes in "
         "`docs/backlog.md`, all pre-registered; the constants and both verdict functions were "
         "committed in `src/catface/ml/pose.py` and `src/catface/ml/scoring.py` before this run "
         "produced a single metric. Produced by `uv run python scripts/pose_and_classes.py`."),
        "",
        "## Input",
        f"Rows: {config['n_rows']}. Per-class counts:",
        *[f"  {name}: {count}" for name, count in class_counts.items()],
        (f"Splits: `{config['splits_path']}`, sha256 `{config['splits_sha256']}`, reused as-is "
         "from E2 and E3, never regenerated."),
        "",
        "## The pose estimators",
        ("The axis convention and the weak-perspective projection model are written out in "
         "`src/catface/core/geometry.py`'s module docstring and are not repeated here. Both "
         "estimators run on the same `generalized_procrustes` output E1 used, so in-plane roll "
         "is already removed. Pitch is not modelled and not removed."),
        "",
        ("**Family 1, foreshortening (primary).** `w_obs / v_obs`, the outer-eye-corner span over "
         "the chin-to-eye-corner-midpoint vertical, equals `r_frontal * |cos(theta)|`: the pair's "
         "own depth cancels, so no depth prior enters anywhere. Unsigned, monotonic on "
         "[0, 90] degrees, nothing saturates."),
        ("**Family 2, midline offset (secondary, signed).** `s`, the philtrum's offset from the "
         "eye-corner midpoint in units of the observed span, equals `d_rel * tan(theta)`. Signed "
         "and monotonic over the full range."),
        ("**The legacy centroid proxy.** E1's quantity, reported per row and correlated against "
         "the new estimator, so E1's PC2 finding stays connected to this one. Not a binning "
         "quantity and no edge is computed from it."),
        "",
        "Rejected estimator families, recorded so the choice is on the record:",
        "",
        *[f"- **{name}.** {why}" for name, why in REJECTED_FAMILIES],
        "",
        "## r_frontal and the out-of-domain rows",
        ("Pre-registered: `r_frontal` is the 95th percentile of `w_obs / v_obs` over all "
         f"{config['n_rows']} rows, pooled, computed once and frozen into `config.json` before "
         "any arm was scored. Rows above it give `cos(theta) > 1`, clamp to theta = 0 and land in "
         "bin 0; they are counted, never given an invented angle."),
        "",
        "| percentile | `r_frontal` | out-of-domain rows | share |",
        "|---|---|---|---|",
        *[
            f"| {p}th | {config['r_frontal'][p]:.4f} | {config['out_of_domain_rows'][p]} | "
            f"{config['out_of_domain_rows'][p] / config['n_rows']:.1%} |"
            for p in ("90.0", "95.0", "99.0")
        ],
        "",
        "## Bins",
        (f"Four equal-count quantile bins on the estimator, computed at run time over all "
         f"{config['n_rows']} rows, pooled, and frozen into `config.json` before any arm was "
         "scored. Bin 0 is the most frontal. The ratio edge is what an app-side consumer would "
         "key on; it carries no modelling assumption at all."),
        "",
        "| bin | ratio range (`w_obs / v_obs`) | theta range (95th pct) | rows | `uncomfortable` rows | per-fold `uncomfortable` |",
        "|---|---|---|---|---|---|",
        *[
            f"| {b} | {ratio_edges[b + 1]:.3f} to {ratio_edges[b]:.3f} | {theta_edges[b]:.1f} to "
            f"{theta_edges[b + 1]:.1f} deg | {all_metrics[verdict_arm]['per_bin'][b]['n']} | "
            f"{occupancy_4['per_bin'][b]} | {occupancy_4['per_bin_fold'][b]} |"
            for b in range(bins["n_bins"])
        ],
        "",
        (f"Pre-registered 3-bin fallback: **not taken**. The smallest (bin x fold) "
         f"`uncomfortable` cell holds {min(min(row) for row in occupancy_4['per_bin_fold'])} rows, "
         "none is empty, so the run stayed at four bins."
         if not bins["fallback_taken"]
         else "Pre-registered 3-bin fallback: **taken**, an empty (bin x fold) cell at four bins."),
        "",
        "## The internal control, and what it did to the word \"yaw\"",
        (f"Spearman rho between |s| and tan(theta) over all {config['n_rows']} rows: "
         f"**{config['internal_control']['spearman_rho']:.3f}**, 95% bootstrap CI "
         f"{config['internal_control']['spearman_ci'][0]:.3f} to "
         f"{config['internal_control']['spearman_ci'][1]:.3f} "
         f"(B = {config['internal_control']['bootstrap_b']}, seed "
         f"{config['internal_control']['bootstrap_seed']}). The pre-registered thresholds are "
         f"{pose.RHO_YAW} to call the binning quantity yaw and {pose.RHO_WITHDRAW} to withdraw "
         "the label."),
        "",
        ("**The measured rho is below the withdrawal threshold.** The two estimators, built from "
         "the same landmark map on the same frame and predicted by the same geometry to agree, "
         "are very nearly unrelated on this data. That is the single most important number in "
         "this run: whatever the bins are binning, both families cannot be measuring it. Q3's "
         "verdict is therefore reported against the foreshortening ratio under its operational "
         "name, with no degree figure attached to it, and the degree columns above are kept only "
         "because the bins were computed on them. Plot: `plots/estimator_agreement.png`."),
        "",
        "## The measured depth term, and the chin control",
        (f"`d_rel = |s| / tan(theta)` over the {d['n_rows_used']} rows above "
         f"{d['min_degrees']:.0f} degrees (below that `tan(theta) -> 0` and the ratio is noise): "
         f"median **{d['philtrum_median']:.4f}**, IQR {d['philtrum_iqr'][0]:.4f} to "
         f"{d['philtrum_iqr'][1]:.4f}. This is the quantity the first grooming pass would have "
         "swept as an invented constant; here it is measured per row."),
        "",
        (f"The chin control, node 2, sits close to the eye-corner plane, so the geometry predicts "
         f"its implied depth comes out far smaller in magnitude than the philtrum's. It does not: "
         f"median {d['chin_median']:.4f} against the philtrum's {d['philtrum_median']:.4f}, "
         f"a ratio of {d['chin_median'] / d['philtrum_median']:.2f}. **The falsifiable prediction "
         "of the geometry is falsified.** The reading that fits both this and the rho above is "
         "that neither column is measuring depth: `|s| / tan(theta)` is a ratio of two noisy "
         "asymmetries, and a near-zero-depth point produces just as much of it as a protruding "
         "one. The synthetic round trip in `tests/test_pose.py` recovers the built-in 0.30 to "
         "1e-6 and puts the chin below 1% of it, so the estimator is right and the data is not "
         "the geometry it assumes."),
        "",
        "## The manual review montage",
        ("`yaw_extremes_manual_review.png`: the most-negative, near-zero and most-positive rows "
         "by family 2's signed `s`, cropped to the landmark box and rendered from the source "
         "images with points 4, 8, 16 and 2 marked. Looked at before this README was written; "
         "`yaw_extremes_manual_review.json` names the rows."),
        "",
        *[f"- {reading}" for reading in MONTAGE_READING],
        "",
        (f"Sign balance: {config['sign_split']['negative_s']} rows negative, "
         f"{config['sign_split']['positive_s']} positive, close to even as a population of "
         "photographs should be."),
        "",
        "## The superseded centroid proxy, and why the estimator changed",
        (f"E1's centroid proxy reaches |proxy| = {config['centroid_proxy_abs_max']:.4f} on these "
         f"{config['n_rows']} rows, against the roughly 0.0315 that the first grooming pass's "
         "forward model could generate at its central depth prior d = 0.30 -- the data overruns "
         "the model by a factor of about four, so for those rows no angle exists that the model "
         "could have produced. That finding is what motivated the second grooming pass to replace "
         "the estimator rather than narrow its prior, and it is recorded here as a measurement on "
         "the superseded quantity, not as an input to any bin, threshold or verdict in this run. "
         "The proxy moves under roll, pitch, expression asymmetry and plain landmark error as "
         "well as yaw."),
        "",
        (f"Correlation of the legacy proxy against this run's theta: r = "
         f"{config['centroid_proxy_vs_theta_r']:.3f}. The two are essentially unrelated, which is "
         "the same disagreement the internal control found between families 1 and 2."),
        (f"E1's published r = 0.90 between PC2 and the proxy re-derives as "
         f"{config['e1_best_pc_centroid_proxy_r']:.3f} on E1's own 2029-row plausible stack, now "
         "that the proxy is imported from `core.geometry` rather than computed inline. "
         "`experiments/e1` is byte-unchanged and E1 was not re-run."),
        "",
        "## Arms",
        "",
        "| arm | features | role |",
        "|---|---|---|",
        "| `ratios_lr` | 3 geometric ratios | E2 model arm |",
        "| `coords_lr` | 96 aligned coordinates | E2 model arm |",
        "| `mlp` | 96 aligned coordinates | E2 model arm |",
        "| `gnn_identity` | 48x4 node features, `A_hat = I` | E3's winning arm |",
        "| `yaw_only_lr` | the binning quantity alone | control (c) |",
        "| `random_baseline` | none (uniform dummy) | control (b) |",
        "",
        ("All six on the frozen folds with `oversample=True` on training folds only, test folds "
         "untouched, exactly as E2 and E3 ran them. Each arm is called through its own module's "
         "`build_model`, not its `run()`, because E4 needs per-row out-of-fold predictions."),
        "",
        "## Results, pooled",
        "",
        "| arm | mean macro F1 | mean kappa | pooled out-of-fold kappa (95% CI) |",
        "|---|---|---|---|",
        *[
            f"| `{name}` | {all_metrics[name]['mean_macro_f1']:.3f} +/- "
            f"{all_metrics[name]['std_macro_f1']:.3f} | {all_metrics[name]['mean_kappa']:.3f} +/- "
            f"{all_metrics[name]['std_kappa']:.3f} | {all_metrics[name]['pooled_kappa']:.3f} "
            f"({all_metrics[name]['pooled_kappa_ci'][0]:.3f} to "
            f"{all_metrics[name]['pooled_kappa_ci'][1]:.3f}) |"
            for name in ALL_ARMS
        ],
        "",
        (f"Verdict arm by the pre-registered rule (highest pooled out-of-fold kappa among the "
         f"four model arms): **`{verdict_arm}`**."),
        "",
        ("Control (c), `yaw_only_lr`, is reported at the same detail as the model arms below, not "
         "as a footnote. Logistic regression on the binning quantity alone, same configuration as "
         "the other two LR arms, same folds, same oversampling: pooled kappa "
         f"{all_metrics['yaw_only_lr']['pooled_kappa']:.3f} against the uniform dummy's "
         f"{all_metrics['random_baseline']['pooled_kappa']:.3f}. Pose barely predicts the label, "
         "so a per-bin kappa that moved could not be explained away as yaw carrying label "
         "information."),
        "",
    ]

    for name in ALL_ARMS:
        lines += [f"### {name}", ""]
        lines += per_fold_table(all_metrics[name])
        lines += ["Per bin (bin 0 = most frontal):", ""]
        lines += per_bin_table(all_metrics[name], classes)

    lines += [
        "Plot: `plots/kappa_per_bin.png`, every arm's kappa in every bin with its bootstrap CI.",
        "",
        monotone_reading(all_metrics, verdict_arm),
        "",
        ("Accuracy is printed beside each bin's own majority-class rate on purpose: with a "
         "class prior this skewed, an arm gains accuracy by predicting `attentive` harder, and "
         "the verdict rests on kappa for that reason."),
        "",
        "## Control (a): per-bin class mix",
        ("Guards against reading prior shift as pose degradation. Total-variation distance "
         f"between each bin's class distribution and the overall prior; the pre-registered "
         f"tolerance is {scoring.TV_TOLERANCE}."),
        "",
        "| bin | " + " | ".join(classes) + " | TV from the overall prior |",
        "|---|" + "|".join(["---"] * (len(classes) + 1)) + "|",
        *[
            f"| {b['index']} | " + " | ".join(str(c) for c in b["class_counts"]) + f" | {b['tv']:.3f} |"
            for b in all_metrics[verdict_arm]["per_bin"]
        ],
        "",
        "## Control (b): the uniform dummy, inside each bin",
        ("`DummyClassifier(strategy=\"uniform\", random_state=0)` on the identical folds, scored "
         "within each bin under that bin's own label prior rather than once over the pooled rows "
         "-- see the `random_baseline` per-bin table above. It sits at chance in every bin, which "
         "is what makes the model arms' per-bin kappas readable as discrimination."),
        "",
        "## Control (d): reproduction against the committed E2/E3 runs",
        ("E2's and E3's `metrics.json` hold no per-row predictions, so every arm had to be re-run "
         "to recover them. Per-fold macro F1, kappa and MCC compared against the committed files:"),
        "",
        "| arm | source | max abs delta | exact |",
        "|---|---|---|---|",
        *[
            f"| `{name}` | "
            + (f"`{config['reproduction_check'][name]['source']}`" if config["reproduction_check"][name]["source"] else "none (new at E4)")
            + " | "
            + (f"{config['reproduction_check'][name]['max_abs_delta']:.6f}" if config["reproduction_check"][name]["source"] else "-")
            + " | "
            + {True: "yes", False: "**no**", None: "-"}[config["reproduction_check"][name]["exact"]]
            + " |"
            for name in ALL_ARMS
        ],
        "",
        reproduction_reading(config, all_metrics),
        "",
        ("Recorded because it was observed rather than inferred: an earlier execution of this "
         "same script, at the same commit and on the same inputs, reproduced every other arm "
         "exactly and `mlp` in four of its five folds, fold 0 differing by 0.0153 of kappa -- "
         "about five predictions out of that fold's 393. The seeding is identical to `mlp.run`'s "
         "(`torch.manual_seed(0)` immediately before the arm), so torch CPU training here is not "
         "bit-reproducible from one execution to the next. Every artifact in this directory comes "
         "from a single execution, and the reproduction table above is that execution's."),
        "",
        "## Q3 verdict",
        f"Status: **{q3_status}**.",
        "",
        q3_text,
        "",
        "## Q4: class structure",
        f"Pooled out-of-fold confusion matrix, `{verdict_arm}` (rows = true, columns = predicted):",
        "",
        "| true \\ pred | " + " | ".join(classes) + " |",
        "|---|" + "|".join(["---"] * len(classes)) + "|",
        *[
            f"| {classes[i]} | " + " | ".join(str(v) for v in row) + " |"
            for i, row in enumerate(all_metrics[verdict_arm]["pooled_confusion_matrix"])
        ],
        "",
        "Plot: `plots/pooled_confusion_matrix.png`. Full pair detail: `merge.json`.",
        "",
        f"Status: **{q4_status}**.",
        "",
        q4_text,
        "",
        (f"The retrained comparison was run anyway for **{merge['names'][0]} / "
         f"{merge['names'][1]}**, the highest cross-talk pair, although it is not a candidate: "
         f"retrained 2-class kappa {merge['retrained_kappa']:.3f}, post-hoc collapse of the "
         f"3-class predictions {merge['collapsed_kappa']:.3f}, 2-class uniform dummy "
         f"{merge['dummy_kappa']:.3f} (`merged/{verdict_arm}/metrics.json`). Retraining under the "
         "merged label does not beat collapsing the predictions after the fact"
         if not candidates and merge["retrained_kappa"] < merge["collapsed_kappa"]
         else f"The retrained comparison for **{merge['names'][0]} / {merge['names'][1]}**: "
         f"retrained {merge['retrained_kappa']:.3f}, collapsed {merge['collapsed_kappa']:.3f}, "
         f"dummy {merge['dummy_kappa']:.3f} (`merged/{verdict_arm}/metrics.json`)")
        + ", so the null path rests on a measurement rather than on an argument.",
        "",
        ("The Method text's expectation that Scared, Surprised and Angry would prove "
         "interchangeable is **not testable here**: cat-emotions-7 is excluded "
         "(`docs/DECISIONS.md`, 2026-09-19). It is recorded as not testable, no other pair is "
         "substituted for it, and it stays as the prediction to check if that set ever returns."),
        "",
        "## Limitations",
        ("- **Facial width.** Family 1 divides a bilateral span by a vertical one, so a "
         "genuinely narrow-faced cat photographed head-on reads as yawed. Breed is not labelled "
         "in cat-emotions-3, and with one photo per cat and no identity labels a per-row "
         "`r_frontal` is not estimable, so a single population value has to absorb real "
         "brachycephalic-to-dolichocephalic variation. The montage's frontal black cat at 38.9 "
         "degrees is this confound in one picture, and the low rho says it dominates rather than "
         "perturbs. This is the limitation to carry into `docs/MODEL_REPORT.md` under Q3."),
        ("- **Pitch is not modelled and not removed.** A cat looking up or down changes the "
         "chin-to-eye vertical that family 1 divides by, and nothing here separates that from a "
         "narrower face."),
        ("- **Landmark error.** The montage shows two of the six extreme rows are photographs of "
         "two cats with the 48 points split between them, both passing E0's plausibility filter. "
         "Every quantity in this run inherits that."),
        ("- **Within-dataset near-duplicates were never hashed** (`docs/MODEL_REPORT.md`), so the "
         "absolute per-bin numbers are an upper bound. It inflates every arm and every bin "
         "equally, so it does not tilt the per-bin comparison, which is what Q3 turns on."),
        "",
        "## Required controls",
        "- [x] Control (a): per-bin class counts and TV distance from the overall prior, above.",
        "- [x] Control (b): uniform `DummyClassifier` scored within each bin under that bin's own prior, above.",
        "- [x] Control (c): `yaw_only_lr` reported at the same detail as the model arms, above.",
        "- [x] Control (d): per-fold metrics compared against the committed E2/E3 `metrics.json`, deltas in `config.json` and above.",
        ("- [x] Bootstrap as specified: within-bin over rows, B = "
         f"{scoring.BOOTSTRAP_B}, seed {scoring.BOOTSTRAP_SEED}, "
         f"{scoring.BOOTSTRAP_PERCENTILES[0]}/{scoring.BOOTSTRAP_PERCENTILES[1]} percentile; "
         "per-fold kappas reported beside it as the secondary sanity check."),
        "- [x] Bin edges frozen into `config.json` before any arm was scored, with the occupancy tables taken at the same moment.",
        "- [x] Internal control reported against the pre-registered 0.5 / 0.3 thresholds, and the word \"yaw\" withdrawn by it.",
        "- [x] Every degree figure in this run carries the percentile that produced it; the verdict carries none, because the control withdrew them.",
        "",
        "## Files",
        "- `config.json` — every frozen constant, the edges, the reproduction deltas, the git sha.",
        "- `per_row.csv` — 1967 rows: both estimators, theta, the out-of-domain flag, the bin, the legacy proxy and one out-of-fold prediction column per arm. Every table above recomputes from it.",
        "- `<arm>/metrics.json`, `merged/<arm>/metrics.json`, `merge.json`, `plots/`, `yaw_extremes_manual_review.png`.",
        "",
    ]

    (RUN_DIR / "README.md").write_text("\n".join(lines) + "\n")


# --- main -------------------------------------------------------------------


def e1_centroid_proxy_correlation() -> float:
    """Re-derive E1's r = 0.90 between PC2 and the centroid proxy, on E1's own
    2029-row plausible stack, now that the proxy has moved into
    `core.geometry`. E1 itself is not re-run and `experiments/e1` is not
    touched."""
    _subset, aligned, _mean = shape_space.load_aligned_shapes()
    scores = PCA(n_components=shape_space.N_COMPONENTS).fit_transform(
        aligned.reshape(len(aligned), -1)
    )
    proxy = yaw_centroid_proxy(aligned)
    return max(
        (float(np.corrcoef(scores[:, pc], proxy)[0, 1]) for pc in range(shape_space.N_COMPONENTS)),
        key=abs,
    )


def main(argv: list[str]) -> None:
    del argv  # no arguments: deterministic given data/cache/splits.csv

    if not SPLITS_PATH.exists():
        print(f"error: {SPLITS_PATH} does not exist.", file=sys.stderr)
        sys.exit(1)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    y, split, classes = features.y, features.split, features.classes
    labels = sorted(set(y.tolist()))
    aligned = features.coord_X.reshape(len(features.coord_X), N_NODES, 2)
    raw_shapes, _label, _split = load_raw_shapes()
    image_paths = load_splits_cache()["image_path"].tolist()

    # 1. Pose, before anything is scored.
    measured = measure_pose(aligned)
    rho = pose.spearman_rho(np.abs(measured["s"]), np.tan(np.radians(measured["theta"])))
    rho_ci = scoring.spearman_ci(np.abs(measured["s"]), np.tan(np.radians(measured["theta"])))
    print(f"r_frontal: {measured['r_frontal']}")
    print(f"internal control: Spearman rho = {rho:.3f}, CI {rho_ci}")

    bins = freeze_bins(measured["theta"], split, y, classes.index("uncomfortable"))
    bin_index = bins.pop("bin_index")
    print(f"bins: {bins['n_bins']}, edges {bins['theta_edges']}, fallback={bins['fallback_taken']}")

    d_rel = measured["d_rel"][np.isfinite(measured["d_rel"])]
    d_rel_chin = measured["d_rel_chin"][np.isfinite(measured["d_rel_chin"])]
    out_of_domain_by_percentile = {
        p: int(pose.yaw_degrees(measured["ratio"], r)[1].sum())
        for p, r in measured["r_frontal"].items()
    }

    config = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "classes": list(USABLE_CLASSES),
        "git_sha": git_sha(),
        "splits_path": str(SPLITS_PATH.relative_to(ROOT)),
        "splits_sha256": sha256(SPLITS_PATH),
        "n_rows": len(y),
        "estimators": {
            "family_1": "core.geometry.yaw_foreshortening_ratio, pair (4, 8), vertical to point 2",
            "family_2": "core.geometry.yaw_midline_offset, midline 16, pair (4, 8)",
            "legacy": "core.geometry.yaw_centroid_proxy (E1's quantity, not a binning quantity)",
        },
        "frontal_percentile": pose.FRONTAL_PERCENTILE,
        "frontal_percentile_sweep": list(pose.FRONTAL_PERCENTILE_SWEEP),
        "r_frontal": {str(p): v for p, v in measured["r_frontal"].items()},
        "out_of_domain_rows": {str(p): v for p, v in out_of_domain_by_percentile.items()},
        "bins": as_json(bins),
        "ratio_edges": [
            ratio_of_theta_edge(edge, measured["r_frontal"][pose.FRONTAL_PERCENTILE])
            for edge in bins["theta_edges"]
        ],
        "internal_control": {
            "spearman_rho": rho,
            "spearman_ci": list(rho_ci),
            "bootstrap_b": scoring.BOOTSTRAP_B,
            "bootstrap_seed": scoring.BOOTSTRAP_SEED,
            "rho_thresholds": {"yaw": pose.RHO_YAW, "withdraw": pose.RHO_WITHDRAW},
        },
        "d_rel": {
            "philtrum_median": float(np.median(d_rel)),
            "philtrum_iqr": [float(np.percentile(d_rel, 25)), float(np.percentile(d_rel, 75))],
            "chin_median": float(np.median(d_rel_chin)),
            "chin_iqr": [float(np.percentile(d_rel_chin, 25)), float(np.percentile(d_rel_chin, 75))],
            "min_degrees": pose.D_REL_MIN_DEGREES,
            "n_rows_used": len(d_rel),
        },
        "sign_split": {
            "negative_s": int((measured["s"] < 0).sum()),
            "positive_s": int((measured["s"] > 0).sum()),
        },
        "centroid_proxy_vs_theta_r": float(np.corrcoef(measured["centroid_proxy"], measured["theta"])[0, 1]),
        "centroid_proxy_abs_max": float(np.abs(measured["centroid_proxy"]).max()),
        "e1_best_pc_centroid_proxy_r": e1_centroid_proxy_correlation(),
    }
    # Frozen before any arm is scored; the reproduction deltas are added back
    # after the arms run.
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))
    print(f"froze {RUN_DIR / 'config.json'} before scoring any arm")

    # 2. Arms.
    arms = build_arms(features, measured["theta"])
    all_metrics, oof_preds, reproduction = {}, {}, {}
    for name in ALL_ARMS:
        X, build_model = arms[name]
        print(f"running {name}...")
        metrics = run_arm(name, X, build_model, y, split)
        oof = np.array(metrics.pop("oof_pred"))
        oof_preds[name] = oof
        reproduction[name] = reproduction_check(name, metrics)
        metrics["labels"] = classes
        metrics["model"] = name
        metrics["n_features"] = N_FEATURES[name]
        metrics["pooled_kappa"] = float(cohen_kappa_score(y, oof, labels=labels))
        metrics["pooled_kappa_ci"] = list(scoring.kappa_ci(y, oof, labels))
        metrics["pooled_confusion_matrix"] = confusion_matrix(y, oof, labels=labels).tolist()
        metrics["per_bin"] = as_json(
            bin_metrics(y, oof, split, bin_index, labels, bins, measured["r_frontal"], measured["ratio"])
        )
        all_metrics[name] = metrics
        out_dir = RUN_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(
            f"  {name}: pooled kappa {metrics['pooled_kappa']:.3f}, "
            f"reproduction {reproduction[name]}"
        )

    config["reproduction_check"] = reproduction
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))

    # 3. Verdicts.
    verdict_arm = max(MODEL_ARMS, key=lambda name: all_metrics[name]["pooled_kappa"])
    verdict_metrics = all_metrics[verdict_arm]
    q3_status, q3_text = scoring.q3_verdict(verdict_arm, verdict_metrics["per_bin"], rho, rho_ci)
    print(f"Q3: {q3_status}")

    cm = np.array(verdict_metrics["pooled_confusion_matrix"])
    pairs = pair_statistics(y, oof_preds[verdict_arm], cm, classes)
    candidates = [p for p in pairs if p["candidate"]]
    # The pre-registered rule retrains only for a candidate pair. With no
    # candidate, the highest cross-talk pair is retrained anyway, so the null
    # path rests on a measurement rather than on an argument; it cannot change
    # the verdict, which already failed the thresholds.
    to_retrain = candidates[0] if candidates else max(pairs, key=lambda p: p["crosstalk"])
    X, build_model = arms[verdict_arm]
    merge, merged_metrics = run_merge(
        to_retrain, verdict_arm, X, build_model, y, oof_preds[verdict_arm], split
    )
    merged_metrics.pop("oof_pred")
    merged_dir = RUN_DIR / "merged" / verdict_arm
    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "metrics.json").write_text(json.dumps(merged_metrics, indent=2))

    q4_status, q4_text = scoring.q4_verdict(
        pairs,
        merge if candidates else None,
        verdict_arm,
        verdict_metrics["pooled_kappa"],
        tuple(verdict_metrics["pooled_kappa_ci"]),
    )
    print(f"Q4: {q4_status}")
    (RUN_DIR / "merge.json").write_text(
        json.dumps(
            {
                "verdict_arm": verdict_arm,
                "pooled_confusion_matrix": cm.tolist(),
                "classes": classes,
                "thresholds": {
                    "crosstalk": scoring.CROSSTALK_THRESHOLD,
                    "pairwise_kappa_ci_contains_zero": True,
                },
                "pairs": as_json(pairs),
                "retrained": as_json(merge),
                "retrained_is_candidate": bool(candidates),
                "status": q4_status,
            },
            indent=2,
        )
    )

    # 4. Per-row artifact: every table in the README recomputes from this.
    columns = {
        "row": np.arange(len(y)),
        "image_path": image_paths,
        "split": split,
        "y_true": [classes[v] for v in y],
        "foreshortening_ratio": measured["ratio"],
        "theta_degrees": measured["theta"],
        "midline_offset_s": measured["s"],
        "midline_offset_s_chin": measured["s_chin"],
        "d_rel": measured["d_rel"],
        "out_of_domain": measured["out_of_domain"].astype(int),
        "bin": bin_index,
        "centroid_proxy": measured["centroid_proxy"],
        **{f"pred_{name}": [classes[v] for v in oof_preds[name]] for name in ALL_ARMS},
    }
    pd.DataFrame(columns).to_csv(RUN_DIR / "per_row.csv", index=False)

    # 5. Plots and the manual-review montage.
    plot_per_bin_kappa({name: all_metrics[name]["per_bin"] for name in ALL_ARMS}, PLOTS_DIR / "kappa_per_bin.png")
    plot_estimator_agreement(measured["theta"], measured["s"], rho, PLOTS_DIR / "estimator_agreement.png")
    plot_pose_distributions(measured, bins, PLOTS_DIR / "pose_distributions.png")
    plot_confusion_matrix(cm, classes, f"E4: {verdict_arm} pooled out-of-fold", PLOTS_DIR / "pooled_confusion_matrix.png")
    montage = write_yaw_extremes_montage(
        raw_shapes, image_paths, measured, RUN_DIR / "yaw_extremes_manual_review.png"
    )
    (RUN_DIR / "yaw_extremes_manual_review.json").write_text(json.dumps(montage, indent=2))
    print(f"wrote {RUN_DIR / 'yaw_extremes_manual_review.png'} -- inspect it before writing the README")

    write_readme(
        config,
        all_metrics,
        bins,
        measured,
        classes,
        load_splits_cache()["label"].value_counts().reindex(USABLE_CLASSES).to_dict(),
        verdict_arm,
        q3_status,
        q3_text,
        q4_status,
        q4_text,
        merge,
        pairs,
        candidates,
    )
    print(f"wrote {RUN_DIR / 'README.md'}")

    print("\n" + q3_text + "\n\n" + q4_text)


if __name__ == "__main__":
    main(sys.argv[1:])
