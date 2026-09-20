"""E3: dense graph conv over the anatomical adjacency, against a random
adjacency, an identity (no message passing) control, a mean-pool arm that
keeps the first run's readout defect on the record, and a chance-level
reference. Answers RSCH-3 / Q1: does a dense graph conv over the anatomical
adjacency beat the E2 MLP?

Arms, method and acceptance criteria follow the Research PM's re-grooming of
2026-09-20 in `docs/backlog.md`, which supersedes RSCH-3's original method.

Usage: uv run python scripts/gnn.py
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from catface.core.graph import (
    DEFAULT_EDGES_PATH,
    build_cat_adjacency,
    build_random_adjacency,
)
from catface.ml import gnn, random_baseline
from catface.ml.features import load_raw_shapes
from catface.ml.splits import SPLITS_PATH, USABLE_CLASSES

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "experiments" / "e3"
PLOTS_DIR = RUN_DIR / "plots"
E2_MLP_METRICS = ROOT / "experiments" / "e2" / "mlp" / "metrics.json"
SEED = 0
N_SPLITS = 5
N_NODES = 48

GNN_ARMS = ("gnn_anatomical", "gnn_random_ablation", "gnn_identity", "gnn_anatomical_meanpool")
MODEL_COLORS = {
    "mlp": "tab:green",
    "gnn_anatomical": "tab:blue",
    "gnn_random_ablation": "tab:red",
    "gnn_identity": "tab:purple",
    "gnn_anatomical_meanpool": "tab:gray",
    "random_baseline": "tab:brown",
}

ARM_PURPOSE = {
    "gnn_anatomical": "the hypothesis: v3 hand-authored anatomical adjacency, flatten readout.",
    "gnn_random_ablation": (
        "the ablation RSCH-3 requires: a random adjacency of the same edge count, drawn once "
        "with seed 1, everything else identical."
    ),
    "gnn_identity": (
        "the control the first grooming missed: `A_hat = I`, so the three conv layers run at the "
        "same depth and parameter count with message passing switched off. Separates what the "
        "graph contributes from what the extra depth contributes, which the random-adjacency "
        "ablation cannot do on its own -- a random graph still passes messages."
    ),
    "gnn_anatomical_meanpool": (
        "the first E3 run's defect, kept on the record: same v3 adjacency, mean pool over the "
        "48-node axis instead of flatten. See the readout section below."
    ),
}


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def plot_metric_comparison(all_metrics: dict[str, dict], metric: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(all_metrics.keys())
    means = [all_metrics[n][f"mean_{metric}"] for n in names]
    stds = [all_metrics[n][f"std_{metric}"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, means, yerr=stds, capsize=5, color=[MODEL_COLORS[n] for n in names])
    ax.set_ylabel(f"{metric} (mean +/- std across folds)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.set_title(f"E3: {metric}, all arms, {N_SPLITS}-fold CV")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def q1_verdict(mlp_metrics: dict, anatomical: dict, random_ablation: dict, identity: dict) -> tuple[bool, str]:
    """Decided on kappa, per the re-grooming's defect 4: between two
    near-chance classifiers macro F1 tracks the prediction marginals rather
    than discrimination, so it cannot carry the verdict. Macro F1 is reported
    beside each kappa."""
    kappa, std = anatomical["mean_kappa"], anatomical["std_kappa"]
    mlp_kappa, random_kappa = mlp_metrics["mean_kappa"], random_ablation["mean_kappa"]
    diff_mlp, diff_random = kappa - mlp_kappa, kappa - random_kappa
    condition_a, condition_b = diff_mlp > std, diff_random > std
    verdict = condition_a and condition_b

    def held(ok: bool) -> str:
        return "HOLDS" if ok else "DOES NOT HOLD"

    lines = [
        (f"Condition A (beats the E2 MLP by more than the anatomical arm's own CV std): anatomical "
         f"GNN kappa {kappa:.3f} vs. MLP {mlp_kappa:.3f}, diff {diff_mlp:+.3f}, anatomical std "
         f"{std:.3f}: **{held(condition_a)}**. Macro F1 beside it: "
         f"{anatomical['mean_macro_f1']:.3f} vs. {mlp_metrics['mean_macro_f1']:.3f}."),
        (f"Condition B (beats the random-adjacency ablation by the same margin): anatomical GNN "
         f"kappa {kappa:.3f} vs. random-adjacency {random_kappa:.3f}, diff {diff_random:+.3f}, "
         f"anatomical std {std:.3f}: **{held(condition_b)}**. Macro F1 beside it: "
         f"{anatomical['mean_macro_f1']:.3f} vs. {random_ablation['mean_macro_f1']:.3f}."),
    ]
    if verdict:
        lines.append(
            "Both conditions hold: **Q1 verdict: YES**, the anatomical-adjacency graph conv beats "
            "the E2 MLP and beats a random graph of equal density."
        )
    else:
        lines.append(
            "At least one condition fails: **Q1 verdict: NO**. RSCH-3's stop condition names a "
            "random-adjacency tie, or an anatomical GNN that does not clear the MLP, as a valid "
            "answer that has to be reported, not a failed run."
        )

    identity_kappa = identity["mean_kappa"]
    identity_wins = identity_kappa > max(kappa, random_kappa)
    reading = (
        "above both graph arms. Switching message passing off raises the score, so nothing the "
        "anatomical arm achieves can be credited to the graph: the convolution is smoothing away "
        "signal the same layers keep when they act per-node."
        if identity_wins
        else "at or below both graph arms, so the depth on its own does not account for what "
        "they score."
    )
    lines.append(
        f"Identity control (`A_hat = I`, message passing off, same depth and parameter count): "
        f"kappa {identity_kappa:.3f} +/- {identity['std_kappa']:.3f}, macro F1 "
        f"{identity['mean_macro_f1']:.3f} +/- {identity['std_macro_f1']:.3f} -- {reading}"
    )
    return verdict, "\n".join(lines)


def model_results_section(name: str, metrics: dict) -> list[str]:
    labels = metrics["labels"]
    per_fold = zip(metrics["per_fold_macro_f1"], metrics["per_fold_kappa"], metrics["per_fold_mcc"])
    header = [f"### {name}", ARM_PURPOSE[name]] if name in ARM_PURPOSE else [f"### {name}"]
    return [
        *header,
        (f"`n_features` = {metrics['n_features']} (per-node channel count), readout "
         f"`{metrics['readout']}`, balancing: training-fold oversampling."),
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
        *[f"| {labels[i]} | " + " | ".join(str(v) for v in row) + " |" for i, row in enumerate(metrics["confusion_matrix"])],
        "",
        f"Plot: `plots/{name}_confusion_matrix.png`.",
        "",
    ]


def random_baseline_section(metrics: dict) -> list[str]:
    return [
        "### random_baseline",
        ("`DummyClassifier(strategy=\"uniform\", random_state=0)` on the identical folds through "
         "the same `cv.cross_validate`, reused from E2's `catface.ml.random_baseline`. A "
         "chance-level reference: any arm below this line is not classifying."),
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


def readout_defect_section(all_metrics: dict[str, dict]) -> list[str]:
    anatomical, meanpool = all_metrics["gnn_anatomical"], all_metrics["gnn_anatomical_meanpool"]
    return [
        "## The readout defect that caused the re-groom",
        ("The first E3 run ended `_Net.forward` in `h.mean(dim=1)`, a mean over the 48-node axis. "
         "Node features come from `generalized_procrustes`, and `_center_scale` subtracts each "
         "shape's centroid, so the node-axis mean of the GNN's input is zero for all 1967 rows in "
         "all four channels. The Research PM measured 2.58e-33 of between-sample variance "
         "surviving that readout, against 1.64e-2 for the flattened coordinates the E2 MLP reads. "
         "The head saw a near-constant vector; only ReLU asymmetry leaked anything through."),
        ("Q1 asks what the adjacency contributes, and that run varied adjacency and readout at "
         "once with the readout term dominating, so its numbers could not answer the question. "
         "`readout` is now a constructor argument on `_Net`/`GNNClassifier`, defaulting to "
         "`flatten` (head `nn.Linear(HIDDEN * 48, N_CLASSES)`). `gnn_anatomical_meanpool` keeps "
         "the old `mean` head so the defect stays measurable rather than described: "
         f"{meanpool['mean_macro_f1']:.3f} macro F1 / {meanpool['mean_kappa']:.3f} kappa against "
         f"the flatten arm's {anatomical['mean_macro_f1']:.3f} / {anatomical['mean_kappa']:.3f}, "
         "same adjacency, same splits, same epochs."),
        ("Two more things changed with it. `GNNClassifier.fit` no longer applies "
         "`compute_class_weight(\"balanced\")`, and `gnn.run` now passes `oversample=True`, so "
         "every arm here balances the way E2's MLP does and the comparison is against like. The "
         "Q1 verdict is decided on kappa, not macro F1, because macro F1 between two chance-level "
         "classifiers reflects their prediction marginals rather than how well either "
         "discriminates."),
        ("`tests/test_gnn.py` pins the readout argument: the default is `flatten`, the head is "
         "sized `HIDDEN * 48` for it, and with `A_hat = I` the mean readout is permutation-"
         "invariant over nodes while the flatten readout is not."),
        "",
    ]


def write_readme(
    all_metrics: dict[str, dict],
    random_metrics: dict,
    config: dict,
    input_counts: dict[str, int],
    export_result: dict,
    verdict_text: str,
) -> None:
    export_line = (
        f"ONNX export of an untrained GNN **{'succeeded' if export_result['success'] else 'FAILED'}** "
        f"(opset {export_result['opset']}, input shape {export_result['input_shape']}, readout "
        f"`{export_result['readout']}`, exporter `{export_result['exporter']}`)."
    )
    if not export_result["success"]:
        export_line += f" Error: `{export_result.get('error')}`"

    lines = [
        "# experiments/e3 — GNN",
        "",
        "## What this is",
        ("RSCH-3 asks Q1: does a dense graph conv over the anatomical adjacency beat the E2 MLP? "
         "This directory holds the re-run ordered by the Research PM's re-grooming of 2026-09-20 "
         "(`docs/backlog.md`), which sent the first run back on methodology. Five arms, all on the "
         "same cat-emotions-3 rows and the frozen splits in `data/cache/splits.csv`, all balanced "
         "by training-fold oversampling, and the four graph-conv arms all at the same depth and "
         "readout except where the arm exists to vary one of those. No PyTorch Geometric anywhere: "
         "the conv is hand-written dense matmuls over a fixed adjacency buffer, which is what makes "
         "it export to ONNX. Produced by `uv run python scripts/gnn.py`."),
        "",
        "## Input",
        f"Rows: {sum(input_counts.values())}. Per-class counts:",
        *[f"  {name}: {count}" for name, count in input_counts.items()],
        (f"Split recipe: `StratifiedKFold(n_splits={config['n_splits']}, shuffle=True, "
         f"random_state={config['seed']})` over `class`, cached at `data/cache/splits.csv` "
         "(reused as-is from E2, not regenerated)."),
        "Node features: `(N, 48, 4)` -- Procrustes-aligned (x,y) plus offset from the mean shape (x,y).",
        (f"Anatomical adjacency: {config['anatomical_edge_count']} unique edges, parsed from "
         f"`{config['adjacency_path']}`, sha256 `{config['adjacency_sha256']}`."),
        (f"Random-adjacency ablation: {config['random_edge_count']} edges, drawn once (not per "
         f"fold) with seed {config['random_adjacency_seed']}."),
        "Identity control: `A_hat = I`, 48 nodes, no edges.",
        "",
        "## Arms",
        "",
        "| arm | adjacency | readout | balancing |",
        "|---|---|---|---|",
        "| `gnn_anatomical` | v3, 98 edges | flatten | oversample |",
        "| `gnn_random_ablation` | random, 98 edges, seed 1 | flatten | oversample |",
        "| `gnn_identity` | `I`, no message passing | flatten | oversample |",
        "| `gnn_anatomical_meanpool` | v3, 98 edges | mean | oversample |",
        "| `random_baseline` | none | none | oversample |",
        "",
        "## ONNX export check",
        export_line,
        ("It ran on an untrained, randomly initialised model with the flatten head, before any "
         "training loop, per RSCH-3's method and the re-grooming's first acceptance criterion: the "
         "head shape changed from `Linear(32, 3)` to `Linear(1536, 3)`, so the first run's export "
         "result does not carry over. The export target went to a temp directory and was not kept. "
         "This is not RSCH-6's export verification (parity, benchmarking, manifest), which belongs "
         "to the Export Verifier once a winning model exists."),
        "",
        "## Results",
        "",
        "### mlp (E2, carried over for comparison)",
        (f"`n_features` = {all_metrics['mlp']['n_features']} (flattened total, not per-node). "
         f"Mean macro F1 **{all_metrics['mlp']['mean_macro_f1']:.3f} +/- "
         f"{all_metrics['mlp']['std_macro_f1']:.3f}**, kappa **{all_metrics['mlp']['mean_kappa']:.3f} "
         f"+/- {all_metrics['mlp']['std_kappa']:.3f}**, MCC **{all_metrics['mlp']['mean_mcc']:.3f} "
         f"+/- {all_metrics['mlp']['std_mcc']:.3f}** across {N_SPLITS} folds. Full detail: "
         "`experiments/e2/README.md`."),
        "",
    ]
    for name in GNN_ARMS:
        lines += model_results_section(name, all_metrics[name])
    lines += random_baseline_section(random_metrics)

    lines += [
        ("Comparison plots: `plots/kappa_comparison.png` (the metric the verdict uses) and "
         "`plots/macro_f1_comparison.png`."),
        "",
    ]
    lines += readout_defect_section(all_metrics)

    lines += [
        "## Required controls",
        ("- [x] Untrained-model ONNX export attempted and recorded before any training run, for "
         "the flatten head: above."),
        ("- [x] All five arms trained on the frozen splits, macro F1 + kappa + MCC + confusion "
         "matrix, 5-fold: above."),
        ("- [x] Readout identical across the four trained GNN arms except `gnn_anatomical_meanpool`, "
         "which exists to vary it; balancing identical across all arms (`oversample=True`, no "
         "in-`fit` class weights)."),
        "- [x] `gnn_identity` reported in the same detail as the other arms.",
        ("- [x] Q1 verdict computed on kappa with macro F1 beside it, both conditions checked: "
         "below."),
        "- [x] Adjacency file path and sha256 pinned in `config.json`.",
        "- [x] No `torch_geometric` import anywhere in the run (AST-walk tested, `tests/test_gnn.py`).",
        "",
        "## Q1 verdict",
        verdict_text,
        "",
        "## Citations",
        ("- CatFLW (landmark scheme, Finka et al.) and the Finka landmark scheme: see "
         "`docs/MODEL_REPORT.md` Citations."),
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

    anatomical_edges, A_hat_anatomical = build_cat_adjacency()

    # Untrained-model ONNX export check, before any training, on the flatten
    # head the experiment arms use -- required ordering, and the head shape
    # changed since the first run.
    with tempfile.TemporaryDirectory() as tmp:
        export_result = gnn.onnx_export_check(A_hat_anatomical, Path(tmp) / "gnn_untrained.onnx")
    print(f"onnx export check: {export_result}")

    random_adjacency_seed = SEED + 1
    random_edges_list, A_hat_random = build_random_adjacency(
        N_NODES, len(anatomical_edges), seed=random_adjacency_seed
    )
    # Identity: same depth, same parameter count, message passing off. This is
    # what normalize_adjacency returns for an edgeless graph anyway
    # (D^-1/2 (0+I) D^-1/2 = I), so it is the zero-edge end of the same family.
    A_hat_identity = np.eye(N_NODES)

    arms = (
        ("gnn_anatomical", A_hat_anatomical, "flatten"),
        ("gnn_random_ablation", A_hat_random, "flatten"),
        ("gnn_identity", A_hat_identity, "flatten"),
        ("gnn_anatomical_meanpool", A_hat_anatomical, "mean"),
    )

    all_metrics: dict[str, dict] = {}
    for name, A_hat, readout in arms:
        print(f"running {name} (readout={readout})...")
        metrics = gnn.run(A_hat, name, readout=readout)
        all_metrics[name] = metrics
        out_dir = RUN_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(
            f"  {name}: macro F1 = {metrics['mean_macro_f1']:.3f} +/- {metrics['std_macro_f1']:.3f}, "
            f"kappa = {metrics['mean_kappa']:.3f} +/- {metrics['std_kappa']:.3f}"
        )
        plot_confusion_matrix(metrics, PLOTS_DIR / f"{name}_confusion_matrix.png")

    print("running random_baseline...")
    random_metrics = random_baseline.run()
    random_dir = RUN_DIR / "random_baseline"
    random_dir.mkdir(parents=True, exist_ok=True)
    (random_dir / "metrics.json").write_text(json.dumps(random_metrics, indent=2))
    print(
        f"  random_baseline: macro F1 = {random_metrics['mean_macro_f1']:.3f}, "
        f"kappa = {random_metrics['mean_kappa']:.3f}"
    )
    plot_confusion_matrix(random_metrics, PLOTS_DIR / "random_baseline_confusion_matrix.png")

    mlp_metrics = json.loads(E2_MLP_METRICS.read_text())
    for_plot = {"mlp": mlp_metrics, **all_metrics, "random_baseline": random_metrics}
    plot_metric_comparison(for_plot, "kappa", PLOTS_DIR / "kappa_comparison.png")
    plot_metric_comparison(for_plot, "macro_f1", PLOTS_DIR / "macro_f1_comparison.png")

    _verdict, verdict_text = q1_verdict(
        mlp_metrics,
        all_metrics["gnn_anatomical"],
        all_metrics["gnn_random_ablation"],
        all_metrics["gnn_identity"],
    )
    print(verdict_text)

    config = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "classes": list(USABLE_CLASSES),
        "git_sha": git_sha(),
        "adjacency_path": str(DEFAULT_EDGES_PATH.relative_to(ROOT)),
        "adjacency_sha256": sha256(DEFAULT_EDGES_PATH),
        "anatomical_edge_count": len(anatomical_edges),
        "random_edge_count": len(random_edges_list),
        "random_adjacency_seed": random_adjacency_seed,
        "arms": {name: {"readout": readout} for name, _A, readout in arms},
        "balancing": "cv.cross_validate(oversample=True), no in-fit class weights",
        "onnx_export_check": export_result,
    }
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))

    _raw_shapes, label, _split = load_raw_shapes()
    input_counts = label.value_counts().reindex(USABLE_CLASSES).to_dict()

    write_readme({"mlp": mlp_metrics, **all_metrics}, random_metrics, config, input_counts, export_result, verdict_text)
    print(f"wrote {RUN_DIR / 'README.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
