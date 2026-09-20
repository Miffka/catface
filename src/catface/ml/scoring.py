"""E4 (RSCH-4): binning, uncertainty, the required controls' statistics and
the two verdict functions.

The verdicts live here rather than in `scripts/pose_and_classes.py`, which
is a deviation from the `q1_verdict`/`q2_verdict`-in-`scripts/` precedent E2
and E3 set. They are acceptance criteria of the issue, so they need tests,
and tests only import `catface.*`. The README's verdict sentences are these
functions' literal return values: every number in them comes from
`metrics.json` or `config.json`, and a hand-written number in a verdict
sentence is a defect.

Every threshold below is pre-registered in `docs/backlog.md` RSCH-4 and
committed before any E4 metric existed.
"""

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

from catface.ml.pose import FRONTAL_PERCENTILE, RHO_WITHDRAW, RHO_YAW

BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 0
BOOTSTRAP_PERCENTILES = (2.5, 97.5)
# A bin whose class mix differs from the overall prior by more than this
# total-variation distance makes the Q3 verdict PROVISIONAL: its kappa moves
# for reasons that have nothing to do with pose.
TV_TOLERANCE = 0.05
# Q4 merge thresholds, both required, not either.
CROSSTALK_THRESHOLD = 0.25


def quantile_bins(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count quantile bins over all rows, pooled, not per fold.
    Returns (edges of length n_bins+1, per-row bin index).

    Ties (the out-of-domain rows all clamp to theta = 0) make the counts
    only approximately equal; a value sitting exactly on an edge goes to the
    higher bin.
    """
    edges = np.quantile(np.asarray(values, dtype=float), np.linspace(0, 1, n_bins + 1))
    index = np.searchsorted(edges[1:-1], values, side="right")
    return edges, np.clip(index, 0, n_bins - 1)


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    percentiles: tuple[float, float] = BOOTSTRAP_PERCENTILES,
) -> tuple[float, float]:
    """Percentile interval from `b` row-resamples of `n` rows.
    `statistic` takes the resampled row indices. Resamples that leave the
    statistic undefined (a bootstrap sample of one class only) are dropped
    rather than counted as zero."""
    rng = np.random.default_rng(seed)
    values = [statistic(rng.integers(0, n, n)) for _ in range(b)]
    finite = [v for v in values if np.isfinite(v)]
    lo, hi = np.percentile(finite, percentiles)
    return float(lo), float(hi)


def kappa_ci(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int], **kwargs) -> tuple[float, float]:
    """Bootstrap CI for Cohen's kappa over the given rows."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    def statistic(idx: np.ndarray) -> float:
        return cohen_kappa_score(y_true[idx], y_pred[idx], labels=list(labels))

    return bootstrap_ci(statistic, len(y_true), **kwargs)


def spearman_ci(x: np.ndarray, y: np.ndarray, **kwargs) -> tuple[float, float]:
    from catface.ml.pose import spearman_rho

    x, y = np.asarray(x), np.asarray(y)
    return bootstrap_ci(lambda idx: spearman_rho(x[idx], y[idx]), len(x), **kwargs)


def total_variation(counts: np.ndarray, reference_counts: np.ndarray) -> float:
    """Total-variation distance between two class distributions given as
    counts. Control (a): guards against reading prior shift as pose
    degradation."""
    p = np.asarray(counts, dtype=float) / np.sum(counts)
    q = np.asarray(reference_counts, dtype=float) / np.sum(reference_counts)
    return float(0.5 * np.abs(p - q).sum())


def crosstalk(confusion: np.ndarray, i: int, j: int) -> float:
    """(C[i,j] + C[j,i]) / (n_i + n_j): the share of two classes' rows that
    the model swaps between them."""
    confusion = np.asarray(confusion, dtype=float)
    return float((confusion[i, j] + confusion[j, i]) / (confusion[i].sum() + confusion[j].sum()))


def contains_zero(ci: tuple[float, float]) -> bool:
    return ci[0] <= 0.0 <= ci[1]


def degree_phrase(edge_degrees: dict[float, float]) -> str:
    """Every degree figure in the run carries the percentile that produced it
    and the 90th/99th range. A bare degree figure is a defect."""
    sweep = [edge_degrees[p] for p in sorted(edge_degrees)]
    return (
        f"{edge_degrees[FRONTAL_PERCENTILE]:.1f} deg at the "
        f"{FRONTAL_PERCENTILE:.0f}th-percentile `r_frontal` "
        f"({min(sweep):.1f}-{max(sweep):.1f} deg across the 90th/99th sweep)"
    )


def rho_reading(rho: float, rho_ci: tuple[float, float]) -> tuple[bool, str]:
    """The internal control decides whether the run may call the binning
    quantity yaw. Returns (may_use_degrees, sentence)."""
    stated = f"Spearman rho between |s| (family 2) and tan(theta) (family 1) is {rho:.3f} (95% bootstrap CI {rho_ci[0]:.3f} to {rho_ci[1]:.3f})"
    if rho >= RHO_YAW:
        return True, (
            f"{stated}, at or above the pre-registered {RHO_YAW} threshold: the two estimators "
            "agree well enough to call the binning quantity yaw, and degree figures are reported."
        )
    if rho < RHO_WITHDRAW:
        return False, (
            f"{stated}, below the pre-registered {RHO_WITHDRAW} threshold: **the yaw label is not "
            "earned**. The verdict is reported against the binning quantity under its operational "
            "name, the foreshortening ratio, with no degree figure attached."
        )
    return True, (
        f"{stated}, between the pre-registered {RHO_WITHDRAW} and {RHO_YAW}: the yaw label is "
        "**weakly supported**. Both the ratio and the degrees are reported and the label is "
        "qualified wherever it appears."
    )


def q3_verdict(
    arm: str,
    bins: list[dict],
    rho: float,
    rho_ci: tuple[float, float],
) -> tuple[str, str]:
    """Applies the pre-registered chance gate, degradation rule and
    prior-shift tolerance by name, and states which one fired.

    Each entry of `bins` carries `index`, `n`, `kappa`, `kappa_ci`,
    `macro_f1`, `accuracy`, `majority_rate`, `tv`, `edge_ratio` (the
    foreshortening ratio at that bin's frontal-side edge) and
    `edge_degrees` (that edge in degrees, per percentile setting).
    Returns (status, verdict text).
    """
    use_degrees, rho_sentence = rho_reading(rho, rho_ci)
    quantity = "yaw" if use_degrees else "the foreshortening ratio"
    frontal, later = bins[0], bins[1:]
    shifted = [b for b in bins if b["tv"] > TV_TOLERANCE]

    lines = [
        (f"Verdict arm: `{arm}`, the highest pooled out-of-fold kappa among the four model arms, "
         "picked by the pre-registered rule rather than by which arm gives the nicer pose story. "
         "Verdict metric: kappa, with macro F1 beside it."),
        rho_sentence,
    ]

    if contains_zero(frontal["kappa_ci"]):
        status = "INCONCLUSIVE"
        lines.append(
            f"**Chance gate fired.** Bin 0's kappa CI ({frontal['kappa_ci'][0]:.3f} to "
            f"{frontal['kappa_ci'][1]:.3f}, kappa {frontal['kappa']:.3f}, n = {frontal['n']}) "
            "contains 0, so there is no performance for pose to degrade. **Q3 verdict: "
            "INCONCLUSIVE.** The app takes no pose penalty out of E4. No threshold is locatable, "
            "and reading a trend out of the later bins would be reading it out of noise."
        )
    else:
        degrading = next(
            (b for b in later if b["kappa_ci"][1] < frontal["kappa_ci"][0]), None
        )
        if degrading is None:
            status = "NEGATIVE"
            lines.append(
                f"Bin 0 clears chance (kappa {frontal['kappa']:.3f}, CI {frontal['kappa_ci'][0]:.3f} "
                f"to {frontal['kappa_ci'][1]:.3f}). **The degradation rule did not fire**: no bin "
                f"b >= 1 has a kappa upper CI below bin 0's lower CI. **Q3 verdict: NEGATIVE** -- no "
                f"measurable degradation over the {quantity} range this dataset covers, threshold "
                "not locatable at this sample size, and the app applies no pose penalty."
            )
        else:
            status = "POSITIVE"
            edge = f"foreshortening ratio {degrading['edge_ratio']:.3f}"
            if use_degrees:
                edge += f", i.e. {degree_phrase(degrading['edge_degrees'])}"
            lines.append(
                f"Bin 0 clears chance (kappa {frontal['kappa']:.3f}, CI {frontal['kappa_ci'][0]:.3f} "
                f"to {frontal['kappa_ci'][1]:.3f}). **The degradation rule fired at bin "
                f"{degrading['index']}**: its kappa upper CI {degrading['kappa_ci'][1]:.3f} falls "
                f"below bin 0's lower CI {frontal['kappa_ci'][0]:.3f}. **Q3 verdict: POSITIVE** -- "
                f"stable above {edge}, degrades past it."
            )

    if shifted:
        status = f"{status} (PROVISIONAL)"
        named = ", ".join(f"bin {b['index']} (TV {b['tv']:.3f})" for b in shifted)
        lines.append(
            f"**Prior-shift tolerance exceeded** at {named}, against the pre-registered "
            f"TV = {TV_TOLERANCE} against the overall class prior: the verdict is **PROVISIONAL** "
            "and those bins' kappas move partly for reasons that are not pose."
        )
    else:
        lines.append(
            f"Prior-shift tolerance: not exceeded. The largest total-variation distance from the "
            f"overall class prior in any bin is {max(b['tv'] for b in bins):.3f}, under the "
            f"pre-registered {TV_TOLERANCE}."
        )

    return status, "\n\n".join(lines)


def q4_verdict(
    pairs: list[dict],
    merge: dict | None,
    arm: str,
    arm_kappa: float,
    arm_kappa_ci: tuple[float, float],
) -> tuple[str, str]:
    """The merge decision by rule, off the pooled out-of-fold confusion
    matrix. Each entry of `pairs` carries `names`, `crosstalk`, `kappa`,
    `kappa_ci` and `candidate`. `merge` is the retrained comparison for the
    one candidate pair, or None if no pair cleared both thresholds.
    Returns (status, verdict text)."""
    lines = [
        (f"Pooled out-of-fold confusion matrix from `{arm}` (kappa {arm_kappa:.3f}, CI "
         f"{arm_kappa_ci[0]:.3f} to {arm_kappa_ci[1]:.3f}). Pre-registered merge thresholds, both "
         f"required and not either: cross-talk >= {CROSSTALK_THRESHOLD} **and** the pairwise 2x2 "
         "kappa CI contains 0.")
    ]
    for pair in pairs:
        lines.append(
            f"- **{pair['names'][0]} / {pair['names'][1]}**: cross-talk {pair['crosstalk']:.3f} "
            f"({'clears' if pair['crosstalk'] >= CROSSTALK_THRESHOLD else 'below'} "
            f"{CROSSTALK_THRESHOLD}), pairwise 2x2 kappa {pair['kappa']:.3f} with CI "
            f"{pair['kappa_ci'][0]:.3f} to {pair['kappa_ci'][1]:.3f} "
            f"({'contains' if contains_zero(pair['kappa_ci']) else 'excludes'} 0) -- "
            f"{'**candidate**' if pair['candidate'] else 'not a candidate'}."
        )

    if merge is None:
        status = "NEGATIVE"
        null_path = (
            " Every arm sits at chance over the three classes, so no pair is separable, and an "
            "inseparable pair cannot be shown inseparable-but-merge-worthy either: there is no "
            "separation anywhere to contrast it against."
            if contains_zero(arm_kappa_ci)
            else ""
        )
        lines.append(
            "No pair clears both thresholds. **Q4 verdict: NEGATIVE, no merge is warranted on the "
            "evidence**, and the three classes stay as they are." + null_path + " This is the "
            "pre-registered null path, written before the run so that a flat confusion matrix "
            "could not be turned into a merge out of its largest off-diagonal cell."
        )
    else:
        accepted = merge["retrained_kappa"] >= merge["collapsed_kappa"] and min(
            merge["retrained_kappa"], merge["collapsed_kappa"]
        ) > merge["dummy_kappa"]
        status = "POSITIVE" if accepted else "NEGATIVE"
        lines.append(
            f"Candidate pair **{merge['names'][0]} / {merge['names'][1]}** retrained under merged "
            f"labels on the same frozen folds: retrained 2-class kappa {merge['retrained_kappa']:.3f}, "
            f"post-hoc collapse of the 3-class predictions {merge['collapsed_kappa']:.3f}, 2-class "
            f"uniform dummy under the merged prior {merge['dummy_kappa']:.3f}. The merge earns its "
            "place only if retrained >= collapsed and both clear the dummy: "
            + (
                f"it does. **Q4 verdict: POSITIVE**, merge {merge['names'][0]} and "
                f"{merge['names'][1]}."
                if accepted
                else "it does not, so merging bought nothing. **Q4 verdict: NEGATIVE, no merge "
                "warranted on the evidence.**"
            )
        )

    return status, "\n\n".join(lines)
