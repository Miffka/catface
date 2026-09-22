"""E2: run all three baseline models (ratios+LR, coords+LR, coords+MLP)
over the identical stratified 5-fold split, write metrics/config per run
directory, confusion-matrix and macro-F1-comparison plots, and the E2
README.

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

from catface.ml import coords_lr, cv, mlp, onnx_export, random_baseline, ratios_lr
from catface.ml.features import load_features, load_raw_shapes, node_features
from catface.ml.splits import SPLITS_PATH, USABLE_CLASSES

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "experiments" / "e2"
PLOTS_DIR = RUN_DIR / "plots"
SEED = 0
N_SPLITS = 5

MODELS = {"ratios_lr": ratios_lr, "coords_lr": coords_lr, "mlp": mlp}
MODEL_COLORS = {"ratios_lr": "tab:blue", "coords_lr": "tab:orange", "mlp": "tab:green"}
CHECKPOINT_EXT = {"ratios_lr": "pkl", "coords_lr": "pkl", "mlp": "pt"}


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def finalize_model(
    name: str,
    module,
    X: np.ndarray,
    features,
    out_dir: Path,
    mean_shape: np.ndarray | None,
    export_onnx: bool,
) -> dict:
    """Fit one more `module.build_model()` on the full, oversample-balanced
    dataset and checkpoint it -- the per-fold models `cv.cross_validate`
    already ran are fit-and-discarded, never kept. `mean_shape` is required
    for coords_lr/mlp (Procrustes alignment target for single-image
    inference) and omitted for ratios_lr (ratios need no alignment)."""
    model = cv.fit_final_model(X, features.y, module.build_model, oversample=True)

    ckpt_path = out_dir / f"checkpoint.{CHECKPOINT_EXT[name]}"
    if mean_shape is None:
        module.save_checkpoint(model, features.classes, ckpt_path, model_name=name, git_sha=git_sha())
    else:
        module.save_checkpoint(
            model, mean_shape, features.classes, ckpt_path, model_name=name, git_sha=git_sha()
        )

    onnx_success = None
    if export_onnx:
        onnx_result = onnx_export.export_and_verify_onnx(
            model.model, X, out_dir, input_names=["coord_features"], output_names=["logits"]
        )
        onnx_success = onnx_result["success"]

    return {"checkpoint_path": str(ckpt_path.relative_to(ROOT)), "onnx_success": onnx_success}


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


def model_results_section(name: str, metrics: dict) -> list[str]:
    labels = metrics["labels"]
    cm = metrics["confusion_matrix"]
    per_fold = zip(metrics["per_fold_macro_f1"], metrics["per_fold_kappa"], metrics["per_fold_mcc"])
    lines = [
        f"### {name}",
        f"`n_features` = {metrics['n_features']}.",
        "",
        "| fold | macro F1 | kappa | MCC |",
        "|---|---|---|---|",
        *[f"| {k} | {f1:.3f} | {kappa:.3f} | {mcc:.3f} |" for k, (f1, kappa, mcc) in enumerate(per_fold)],
        (f"| **mean** | **{metrics['mean_macro_f1']:.3f} +/- {metrics['std_macro_f1']:.3f}** | "
         f"**{metrics['mean_kappa']:.3f} +/- {metrics['std_kappa']:.3f}** | "
         f"**{metrics['mean_mcc']:.3f} +/- {metrics['std_mcc']:.3f}** |"),
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


def random_baseline_section(metrics: dict) -> list[str]:
    return [
        "## Random-classifier reference",
        ("`DummyClassifier(strategy=\"uniform\", random_state=0)`, run on the identical "
        "5-fold splits as the three models above -- a chance-level reference, excluded "
        "from the macro-F1 comparison plot."),
        "",
        "| metric | mean +/- std |",
        "|---|---|",
        f"| macro F1 | {metrics['mean_macro_f1']:.3f} +/- {metrics['std_macro_f1']:.3f} |",
        f"| kappa | {metrics['mean_kappa']:.3f} +/- {metrics['std_kappa']:.3f} |",
        f"| MCC | {metrics['mean_mcc']:.3f} +/- {metrics['std_mcc']:.3f} |",
        "",
        "Plot: `plots/random_baseline_confusion_matrix.png`.",
        "",
    ]


def checkpoint_section(finalize_results: dict[str, dict]) -> list[str]:
    def onnx_cell(result: dict) -> str:
        if result["onnx_success"] is None:
            return "n/a"
        return "PASS" if result["onnx_success"] else "FAIL"

    return [
        "## Checkpoints & ONNX export",
        "Each model above is also fit once more on the full, oversample-balanced dataset and checkpointed:",
        "",
        "| model | checkpoint | onnx export |",
        "|---|---|---|",
        *[
            f"| `{name}` | `{result['checkpoint_path']}` | {onnx_cell(result)} |"
            for name, result in finalize_results.items()
        ],
        "",
    ]


def write_readme(
    all_metrics: dict[str, dict],
    random_metrics: dict,
    config: dict,
    input_counts: dict[str, int],
    finalize_results: dict[str, dict],
) -> None:
    lines = [
        "# experiments/e2 — baselines",
        "",
        "## What this is",
        ("Three baseline models -- ratios+LR (`ratios_lr`, three geometric ratios: eye "
        "aperture, ear angle, muzzle spread), coords+LR (`coords_lr`), coords+MLP (`mlp`) -- "
        "trained on cat-emotions-3, plausible rows only, the three usable classes (attentive, "
        "relaxed, uncomfortable), on identical stratified 5-fold splits. Balancing: random "
        "oversampling of minority classes on the training folds only "
        "(`oversample_to_balance`). Each model reports macro F1, Cohen's kappa and MCC, with "
        "a uniform-random classifier run as a chance-level reference. Produced by "
        "`uv run python scripts/make_splits.py` then `uv run python scripts/baselines.py`."),
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

    lines += random_baseline_section(random_metrics)

    lines += checkpoint_section(finalize_results)

    lines += [
        "## Citations",
        ("- CatFLW (landmark scheme, Finka et al.) and the Finka landmark scheme: see "
        "`docs/MODEL_REPORT.md` Citations."),
        ("- Feline pain-assessment descriptors, Scientific Reports, "
        "https://www.nature.com/articles/s41598-023-49031-2: 35 geometric descriptors "
        "(angles, distance ratios, area ratios by action unit) over cat facial landmarks. "
        "It reports that ear-position and orbital-tightening descriptors gave the smallest "
        "prediction error of the set, external validation that eye aperture and ear angle "
        "make a reasonable pair to baseline against here."),
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

    raw_shapes, label, _split = load_raw_shapes()
    features = load_features()
    # load_features() doesn't expose mean_shape; recomputed via the same
    # public node_features() call it uses internally (scripts/gnn.py does
    # the same), so coords_lr/mlp checkpoints carry the training-time
    # Procrustes reference single-image inference needs.
    _aligned, mean_shape = node_features(raw_shapes)
    MODEL_X = {"ratios_lr": features.ratio_X, "coords_lr": features.coord_X, "mlp": features.coord_X}

    all_metrics: dict[str, dict] = {}
    finalize_results: dict[str, dict] = {}
    for name, module in MODELS.items():
        print(f"running {name}...")
        metrics = module.run()
        all_metrics[name] = metrics
        out_dir = RUN_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  {name}: mean macro F1 = {metrics['mean_macro_f1']:.3f} +/- {metrics['std_macro_f1']:.3f}")
        plot_confusion_matrix(metrics, PLOTS_DIR / f"{name}_confusion_matrix.png")

        print(f"  finalizing {name}: full-dataset checkpoint" + (" + ONNX export" if name == "mlp" else "") + "...")
        finalize_results[name] = finalize_model(
            name,
            module,
            MODEL_X[name],
            features,
            out_dir,
            mean_shape=None if name == "ratios_lr" else mean_shape,
            export_onnx=(name == "mlp"),
        )
        onnx_success = finalize_results[name]["onnx_success"]
        onnx_note = "" if onnx_success is None else f", onnx export: {'PASS' if onnx_success else 'FAIL'}"
        print(f"  checkpoint: {finalize_results[name]['checkpoint_path']}{onnx_note}")

    plot_macro_f1_comparison(all_metrics, PLOTS_DIR / "macro_f1_comparison.png")

    print("running random_baseline...")
    random_metrics = random_baseline.run()
    random_dir = RUN_DIR / "random_baseline"
    random_dir.mkdir(parents=True, exist_ok=True)
    (random_dir / "metrics.json").write_text(json.dumps(random_metrics, indent=2))
    print(
        f"  random_baseline: mean macro F1 = {random_metrics['mean_macro_f1']:.3f} "
        f"+/- {random_metrics['std_macro_f1']:.3f}"
    )
    plot_confusion_matrix(random_metrics, PLOTS_DIR / "random_baseline_confusion_matrix.png")

    config = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "classes": list(USABLE_CLASSES),
        "git_sha": git_sha(),
        "checkpoints": finalize_results,
    }
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))

    input_counts = label.value_counts().reindex(USABLE_CLASSES).to_dict()

    write_readme(all_metrics, random_metrics, config, input_counts, finalize_results)
    print(f"wrote {RUN_DIR / 'README.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
