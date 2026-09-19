"""E2: run all three baseline models (ratios+LR, coords+LR, coords+MLP)
over the identical stratified 5-fold split, write metrics/config per run
directory, confusion-matrix and macro-F1-comparison plots, and the E2
README. Answers RSCH-2 / Q2: does a learned model beat two geometric
ratios fed to logistic regression?

Usage: uv run python scripts/baselines.py
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from catface.core.geometry import muzzle_spread_ratio
from catface.ml import coords_lr, mlp, ratios_lr
from catface.ml.features import load_raw_shapes
from catface.ml.plausibility import LEFT_EYE, MUZZLE, RIGHT_EYE
from catface.ml.splits import SPLITS_PATH, USABLE_CLASSES

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "experiments" / "e2"
PLOTS_DIR = RUN_DIR / "plots"
SEED = 0
N_SPLITS = 5

MODELS = {"ratios_lr": ratios_lr, "coords_lr": coords_lr, "mlp": mlp}
MODEL_COLORS = {"ratios_lr": "tab:blue", "coords_lr": "tab:orange", "mlp": "tab:green"}


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def plot_confusion_matrix(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["labels"]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{metrics['model']} confusion matrix (summed over {N_SPLITS} folds)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_macro_f1_comparison(all_metrics: dict[str, dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(all_metrics.keys())
    means = [all_metrics[n]["mean_macro_f1"] for n in names]
    stds = [all_metrics[n]["std_macro_f1"] for n in names]
    colors = [MODEL_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, means, yerr=stds, capsize=5, color=colors)
    ax.set_ylabel("macro F1 (mean +/- std across folds)")
    ax.set_ylim(0, 1)
    ax.set_title("E2: macro F1 by model, 5-fold CV")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def q2_verdict(all_metrics: dict[str, dict]) -> tuple[bool, str]:
    m1, m3 = all_metrics["ratios_lr"]["mean_macro_f1"], all_metrics["mlp"]["mean_macro_f1"]
    std3 = all_metrics["mlp"]["std_macro_f1"]
    diff = abs(m1 - m3)
    matches = diff <= std3
    verdict = "MATCH" if matches else "NO MATCH"
    comparison = "within" if matches else "beyond"
    if matches:
        reading = (
            "A match is a valid negative answer to Q2 under RSCH-2's stop condition, not a "
            "failure: the coordinate-level model finds nothing past what the two hand-built "
            "ratios already capture."
        )
    else:
        reading = (
            "The MLP clears the two-ratio baseline by more than one fold's worth of its own "
            "noise, so Q2's answer here is that the learned model does find signal past eye "
            "aperture and ear angle."
        )
    sentence = (
        f"Model (1) ratios_lr scores mean macro F1 {m1:.3f}. Model (3) mlp scores "
        f"{m3:.3f}, std {std3:.3f} across folds. The gap between them is {diff:.3f}, "
        f"{comparison} one CV std of model (3)'s macro F1: **{verdict}**. {reading}"
    )
    return matches, sentence


def model_results_section(name: str, metrics: dict) -> list[str]:
    labels = metrics["labels"]
    cm = metrics["confusion_matrix"]
    lines = [
        f"### {name}",
        f"`n_features` = {metrics['n_features']}.",
        "",
        "| fold | macro F1 |",
        "|---|---|",
        *[f"| {k} | {f1:.3f} |" for k, f1 in enumerate(metrics["per_fold_macro_f1"])],
        f"| **mean** | **{metrics['mean_macro_f1']:.3f} +/- {metrics['std_macro_f1']:.3f}** |",
        "",
        f"Confusion matrix (rows = true, columns = predicted, summed over {N_SPLITS} folds):",
        "",
        "| true \\ pred | " + " | ".join(labels) + " |",
        "|---|" + "|".join(["---"] * len(labels)) + "|",
        *[f"| {labels[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(cm)],
        "",
        f"Plot: `plots/{name}_confusion_matrix.png`.",
        "",
    ]
    return lines


def write_readme(
    all_metrics: dict[str, dict],
    config: dict,
    input_counts: dict[str, int],
    spread_stats: tuple[float, float],
) -> None:
    _matches, verdict_sentence = q2_verdict(all_metrics)
    spread_mean, spread_std = spread_stats

    lines = [
        "# experiments/e2 — baselines",
        "",
        "## What this is",
        ("RSCH-2 asks Q2: does a learned model beat two geometric ratios (eye aperture, ear "
        "angle) fed to logistic regression? This run trains three baseline models on "
        "cat-emotions-3, plausible rows only, the three usable classes (attentive, relaxed, "
        "uncomfortable), on identical stratified 5-fold splits with balanced class weights. "
        "Produced by `uv run python scripts/make_splits.py` then "
        "`uv run python scripts/baselines.py`."),
        "",
        "## Input",
        f"Rows: {sum(input_counts.values())}. Per-class counts:",
        *[f"  {name}: {count}" for name, count in input_counts.items()],
        (f"Split recipe: `StratifiedKFold(n_splits={config['n_splits']}, shuffle=True, "
        f"random_state={config['seed']})` over `class`, cached at `data/cache/splits.csv`."),
        "",
        "## Results",
        "",
    ]
    for name in ("ratios_lr", "coords_lr", "mlp"):
        lines += model_results_section(name, all_metrics[name])

    lines += [
        "## Required controls",
        ("- [x] Identical splits across all three models: all three read "
        "`data/cache/splits.csv`, written once by `scripts/make_splits.py`."),
        ("- [x] 5-fold: `n_splits=5` in `catface.ml.splits.make_splits` and "
        "`catface.ml.cv.cross_validate`."),
        "- [x] Macro F1 reported (not accuracy alone): per-fold and mean +/- std, above.",
        "- [x] Confusion matrix per model: above, summed over folds.",
        ("- [x] Balanced class weights: `class_weight=\"balanced\"` (LR models), "
        "`nn.CrossEntropyLoss(weight=...)` computed via "
        "`sklearn.utils.class_weight.compute_class_weight(\"balanced\", ...)` (MLP)."),
        "",
        "## Q2 verdict",
        verdict_sentence,
        "",
        "## Extra readout",
        (f"`muzzle_spread_ratio` (max pairwise distance among the 22 MUZZLE points, divided "
        f"by inter-ocular distance) over the {sum(input_counts.values())} input rows: mean "
        f"{spread_mean:.3f}, std {spread_std:.3f}. This is descriptive only. None of the "
        "three baselines above use it as a model input; model (1) stays exactly eye aperture "
        "plus ear angle, per Q2 as posed. It is a geometry-only stand-in for whisker-pad "
        "spread, not a validated \"tension\" measure. See `docs/backlog.md` RSCH-2 grooming "
        "notes and `src/catface/core/geometry.py:muzzle_spread_ratio`."),
        "",
        "## Citations",
        ("- CatFLW (landmark scheme, Finka et al.) and the Finka landmark scheme: see "
        "`docs/MODEL_REPORT.md` Citations."),
        ("- Feline pain-assessment descriptors, Scientific Reports, "
        "https://www.nature.com/articles/s41598-023-49031-2: 35 geometric descriptors "
        "(angles, distance ratios, area ratios by action unit) over cat facial landmarks. "
        "It reports that ear-position and orbital-tightening descriptors gave the smallest "
        "prediction error of the set, external validation that eye aperture and ear angle "
        "make a reasonable pair to baseline against here, not a requirement to add features "
        "beyond Q2's stated two."),
        "",
    ]

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> None:
    del argv  # no arguments: deterministic given data/cache/splits.csv

    if not SPLITS_PATH.exists():
        print(
            f"error: {SPLITS_PATH} does not exist. Run "
            "`uv run python scripts/make_splits.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    all_metrics: dict[str, dict] = {}
    for name, module in MODELS.items():
        print(f"running {name}...")
        metrics = module.run()
        all_metrics[name] = metrics
        out_dir = RUN_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  {name}: mean macro F1 = {metrics['mean_macro_f1']:.3f} +/- {metrics['std_macro_f1']:.3f}")
        plot_confusion_matrix(metrics, PLOTS_DIR / f"{name}_confusion_matrix.png")

    plot_macro_f1_comparison(all_metrics, PLOTS_DIR / "macro_f1_comparison.png")

    config = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "classes": list(USABLE_CLASSES),
        "git_sha": git_sha(),
    }
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))

    raw_shapes, label, _split = load_raw_shapes()
    input_counts = label.value_counts().reindex(USABLE_CLASSES).to_dict()
    spreads = np.array(
        [
            muzzle_spread_ratio(
                shape, MUZZLE, shape[list(LEFT_EYE)].mean(axis=0), shape[list(RIGHT_EYE)].mean(axis=0)
            )
            for shape in raw_shapes
        ]
    )
    print(f"muzzle_spread_ratio: mean={spreads.mean():.3f} std={spreads.std():.3f} (extra readout, not a model input)")

    write_readme(all_metrics, config, input_counts, (float(spreads.mean()), float(spreads.std())))
    print(f"wrote {RUN_DIR / 'README.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
