"""E4: pose-binned check of the two best E2/E3 arms by pooled kappa.

Loads the already-trained `coords_lr` and `gnn_identity` checkpoints
(full-dataset fits from `experiments/e2/coords_lr/checkpoint.pkl` and
`experiments/e3/gnn_identity/checkpoint.pt`) instead of retraining, predicts
over the whole dataset, and reports kappa/macro-F1/accuracy inside four
pose bins. This is an in-sample check (the checkpoints were fit on these
same rows), not a re-run of the pre-registered out-of-fold RSCH-4 numbers
in `docs/backlog.md` -- no verdict is computed here, just the numbers.

Usage: uv run python scripts/pose_and_classes.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from catface.core.geometry import yaw_foreshortening_ratio
from catface.ml import coords_lr, gnn, pose, scoring
from catface.ml.features import Features, load_features
from catface.ml.splits import SPLITS_PATH

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "experiments" / "e4"
PLOTS_DIR = RUN_DIR / "plots"
N_NODES = 48

CHECKPOINTS = {
    "coords_lr": ROOT / "experiments" / "e2" / "coords_lr" / "checkpoint.pkl",
    "gnn_identity": ROOT / "experiments" / "e3" / "gnn_identity" / "checkpoint.pt",
}
ARM_COLORS = {"coords_lr": "tab:orange", "gnn_identity": "tab:purple"}
# Second-grooming-pass prior-shift tolerance (docs/backlog.md RSCH-4):
# exceeding it means a bin's class mix has drifted enough from the pooled
# prior that its kappa can't be read as pose degradation alone.
TV_TOLERANCE = 0.05


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_arm(name: str, features: Features) -> np.ndarray:
    """Load `name`'s full-dataset checkpoint and predict over the whole
    dataset. Checked against `features.classes` because a class-order
    mismatch between the checkpoint and this run's LabelEncoder would
    silently mislabel every prediction."""
    path = CHECKPOINTS[name]
    if name == "coords_lr":
        model, _mean_shape, classes = coords_lr.load_checkpoint(path)
        X = features.coord_X
    else:
        model, _mean_shape, classes = gnn.load_checkpoint(path)
        X = features.node_X
    if classes != features.classes:
        raise ValueError(f"{name} checkpoint classes {classes} != run classes {features.classes}")
    return model.predict(X)


def bin_metrics(
    y: np.ndarray, pred: np.ndarray, bin_index: np.ndarray, labels: list[int], n_bins: int, pooled_counts: np.ndarray
) -> list[dict]:
    rows = []
    for b in range(n_bins):
        mask = bin_index == b
        y_b, pred_b = y[mask], pred[mask]
        counts = np.bincount(y_b, minlength=len(labels))
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
                # Control (a): guards against a bin's class mix drifting from
                # the pooled prior and showing up as "pose degradation" that
                # is really prior shift. Same for every arm (it only depends
                # on y and the bin, not on pred), computed once per arm for
                # simplicity.
                "tv_distance": scoring.total_variation(counts, pooled_counts),
            }
        )
    return rows


def plot_kappa_per_bin(per_arm_bins: dict[str, list[dict]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
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
    ax.set_ylabel("in-sample kappa (95% bootstrap CI)")
    ax.set_title("E4: kappa per yaw bin (bin 0 = most frontal), in-sample")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def per_bin_table(rows: list[dict], classes: list[str]) -> list[str]:
    header = (
        "| bin | n | kappa (95% CI) | macro F1 | accuracy | majority rate | TV vs pooled prior | "
        + " | ".join(f"n {c}" for c in classes)
        + " |"
    )
    out = [header, "|---|---|---|---|---|---|---|" + "|".join(["---"] * len(classes)) + "|"]
    for b in rows:
        flag = " (over 0.05 tolerance)" if b["tv_distance"] > TV_TOLERANCE else ""
        out.append(
            f"| {b['index']} | {b['n']} | {b['kappa']:.3f} ({b['kappa_ci'][0]:.3f} to "
            f"{b['kappa_ci'][1]:.3f}) | {b['macro_f1']:.3f} | {b['accuracy']:.3f} | "
            f"{b['majority_rate']:.3f} | {b['tv_distance']:.3f}{flag} | "
            + " | ".join(str(c) for c in b["class_counts"])
            + " |"
        )
    return out + [""]


def majority_rate_sentence(arm: str, rows: list[dict]) -> str:
    """Names each bin's majority-class rate in prose, as the chance floor
    that bin's kappa is being read against (kappa is chance-corrected, but
    the floor it corrects against is otherwise left as a table column no
    one is pointed at)."""
    parts = ", ".join(f"bin {b['index']} {b['majority_rate']:.3f}" for b in rows)
    return f"`{arm}`'s kappa in each bin is read against that bin's own majority-class rate as the chance floor: {parts}."


def tv_distance_sentence(rows: list[dict]) -> str:
    over = [b["index"] for b in rows if b["tv_distance"] > TV_TOLERANCE]
    if over:
        detail = ", ".join(f"bin {b} ({rows[b]['tv_distance']:.3f})" for b in over)
        return (
            f"Per-bin total-variation distance from the pooled class prior exceeds the "
            f"pre-registered {TV_TOLERANCE:.2f} tolerance in {detail}: that bin's class mix has "
            "drifted from the pooled prior enough that its kappa cannot be read as pose effect alone."
        )
    return (
        f"Per-bin total-variation distance from the pooled class prior stays within the "
        f"pre-registered {TV_TOLERANCE:.2f} tolerance in every bin."
    )


def write_readme(config: dict, all_metrics: dict, classes: list[str]) -> None:
    lines = [
        "# experiments/e4 — pose-binned check of coords_lr and gnn_identity",
        "",
        ("In-sample check of the two E2/E3 arms with the highest pooled out-of-fold kappa "
         "(`gnn_identity`, `coords_lr`, per `docs/backlog.md` RSCH-4): their committed "
         "full-dataset checkpoints are loaded and scored inside four pose bins. Not a "
         "re-run of the pre-registered out-of-fold numbers -- these checkpoints were fit "
         "on the same rows they're scored on here. No verdict is computed; the numbers "
         "below are for the Research PM to read directly. Produced by "
         "`uv run python scripts/pose_and_classes.py`."),
        "",
        ("**This is an in-sample check and its evidence is one-sided.** Both checkpoints "
         "were fit on the exact rows they are scored on here, so a flat-or-rising per-bin "
         "trend below does not establish that the model holds up on unseen high-yaw "
         "photos -- it only shows the checkpoints are not visibly underfitting their own "
         "high-yaw training rows. A falling trend would have been informative; this one "
         "is not proof of generalization."),
        "",
        f"Rows: {config['n_rows']}. Splits: `{config['splits_path']}`, sha256 `{config['splits_sha256']}`.",
        (f"Pose estimator: `core.geometry.yaw_foreshortening_ratio`. `r_frontal` (the "
         f"{config['frontal_percentile']:.0f}th percentile of the observed ratio): "
         f"{config['r_frontal']:.4f}, {config['out_of_domain_rows']} out-of-domain rows "
         "clamped to theta = 0."),
        "",
        "## Bins",
        ("Four equal-count quantile bins on theta (arccos of the ratio), bin 0 most "
         "frontal. Reported here in raw `w_obs / v_obs` ratio units rather than degrees: "
         "this project's own internal control (`docs/backlog.md` RSCH-4, 2026-09-20 "
         "comment) found the yaw label unearned for this estimator -- Spearman rho 0.106 "
         "(CI 0.062 to 0.149) against a pre-registered 0.3 threshold -- so no degree "
         "figure is attached to it here. Ratio falls as yaw rises, so bin 0's range sits "
         "highest."),
        "",
        "| bin | w_obs / v_obs range |",
        "|---|---|",
        *[
            f"| {b} | {config['ratio_bin_ranges'][b][0]:.4f} to {config['ratio_bin_ranges'][b][1]:.4f} |"
            for b in range(pose.N_BINS)
        ],
        "",
        "## Results",
        "",
        "| arm | checkpoint | pooled kappa (95% CI) | pooled macro F1 | pooled accuracy |",
        "|---|---|---|---|---|",
        *[
            f"| `{name}` | `{config['checkpoints'][name]}` | {all_metrics[name]['pooled_kappa']:.3f} "
            f"({all_metrics[name]['pooled_kappa_ci'][0]:.3f} to "
            f"{all_metrics[name]['pooled_kappa_ci'][1]:.3f}) | {all_metrics[name]['macro_f1']:.3f} | "
            f"{all_metrics[name]['accuracy']:.3f} |"
            for name in CHECKPOINTS
        ],
        "",
    ]
    for name in CHECKPOINTS:
        lines += [f"### {name}", "", "Pooled confusion matrix (rows = true, columns = predicted):", ""]
        lines += ["| true \\ pred | " + " | ".join(classes) + " |", "|---|" + "|".join(["---"] * len(classes)) + "|"]
        lines += [
            f"| {classes[i]} | " + " | ".join(str(v) for v in row) + " |"
            for i, row in enumerate(all_metrics[name]["pooled_confusion_matrix"])
        ]
        lines += ["", "Per bin:", ""]
        lines += per_bin_table(all_metrics[name]["per_bin"], classes)
        lines += [majority_rate_sentence(name, all_metrics[name]["per_bin"]), ""]

    lines += [
        tv_distance_sentence(all_metrics[next(iter(CHECKPOINTS))]["per_bin"]),
        "",
        "## Limitations",
        "",
        ("- **Facial-width confound.** `theta`/the ratio comes from dividing a bilateral "
         "span by a vertical one (family 1, foreshortening); a genuinely narrow- or "
         "wide-faced cat reads as more or less yawed at zero rotation. Breed and identity "
         "aren't labelled in cat-emotions-3 and there's one photo per cat, so a per-row "
         "correction isn't estimable at this sample size."),
        ("- **Pitch is not modelled.** The estimator only accounts for rotation about the "
         "vertical (yaw) axis; any pitch present in a photo is absorbed into the ratio "
         "uncalibrated, not removed."),
        "",
        "Plot: `plots/kappa_per_bin.png`.",
        "",
    ]
    (RUN_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> None:
    del argv  # no arguments: deterministic given data/cache/splits.csv and the committed checkpoints

    if not SPLITS_PATH.exists():
        print(f"error: {SPLITS_PATH} does not exist.", file=sys.stderr)
        sys.exit(1)
    for name, path in CHECKPOINTS.items():
        if not path.exists():
            print(f"error: {name} checkpoint {path} does not exist.", file=sys.stderr)
            sys.exit(1)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    features = load_features()
    y, classes = features.y, features.classes
    labels = sorted(set(y.tolist()))

    aligned = features.coord_X.reshape(len(features.coord_X), N_NODES, 2)
    ratios = yaw_foreshortening_ratio(aligned)
    r_frontal = pose.frontal_ratio(ratios, pose.FRONTAL_PERCENTILE)
    theta, out_of_domain = pose.yaw_degrees(ratios, r_frontal)
    theta_edges, bin_index = scoring.quantile_bins(theta, pose.N_BINS)
    print(f"r_frontal: {r_frontal:.4f}, out-of-domain rows: {int(out_of_domain.sum())}")
    print(f"bin edges (deg): {[round(float(e), 1) for e in theta_edges]}")

    pooled_counts = np.bincount(y, minlength=len(labels))
    ratio_bin_ranges = [
        (float(ratios[bin_index == b].min()), float(ratios[bin_index == b].max())) for b in range(pose.N_BINS)
    ]

    config = {
        "n_rows": len(y),
        "classes": classes,
        "git_sha": git_sha(),
        "splits_path": str(SPLITS_PATH.relative_to(ROOT)),
        "splits_sha256": sha256(SPLITS_PATH),
        "frontal_percentile": pose.FRONTAL_PERCENTILE,
        "r_frontal": r_frontal,
        "out_of_domain_rows": int(out_of_domain.sum()),
        "theta_edges": [float(e) for e in theta_edges],
        "ratio_bin_ranges": ratio_bin_ranges,
        "pooled_class_counts": pooled_counts.tolist(),
        "checkpoints": {name: str(path.relative_to(ROOT)) for name, path in CHECKPOINTS.items()},
        "evaluation": "in_sample_full_dataset_checkpoint",
    }

    all_metrics = {}
    for name in CHECKPOINTS:
        print(f"scoring {name}...")
        pred = predict_arm(name, features)
        metrics = {
            "model": name,
            "labels": classes,
            "evaluation": "in_sample_full_dataset_checkpoint",
            "pooled_kappa": float(cohen_kappa_score(y, pred, labels=labels)),
            "pooled_kappa_ci": list(scoring.kappa_ci(y, pred, labels)),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=labels)),
            "accuracy": float(accuracy_score(y, pred)),
            "pooled_confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
            "per_bin": bin_metrics(y, pred, bin_index, labels, pose.N_BINS, pooled_counts),
        }
        all_metrics[name] = metrics
        out_dir = RUN_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  {name}: pooled kappa {metrics['pooled_kappa']:.3f}, macro F1 {metrics['macro_f1']:.3f}")

    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))

    plot_kappa_per_bin({name: all_metrics[name]["per_bin"] for name in CHECKPOINTS}, PLOTS_DIR / "kappa_per_bin.png")
    write_readme(config, all_metrics, classes)
    print(f"wrote {RUN_DIR / 'README.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
