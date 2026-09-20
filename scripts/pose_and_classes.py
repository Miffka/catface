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

    print("\n" + q3_text + "\n\n" + q4_text)


if __name__ == "__main__":
    main(sys.argv[1:])
