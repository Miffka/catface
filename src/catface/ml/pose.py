"""E4 (RSCH-4): the degree calibration for `core.geometry`'s yaw
estimators, plus the pre-registered constants of the pose half of the
experiment.

The geometry itself lives in `catface.core.geometry`, because the app's
future `core/states.py` consumes the foreshortening ratio and AGENTS.md
forbids either track keeping a local copy. What lives here is the
research-track modelling on top of it: the population reference `r_frontal`,
the percentile rule that picks it, and the conversion to degrees. The app
consumes the ratio, which needs no modelling assumption at all; the angle is
a research-track label.

Family 1's model (see `core.geometry`'s docstring for the axis convention):

    w_obs / v_obs = r_frontal * |cos(theta)|
    theta         = arccos( (w_obs / v_obs) / r_frontal )

`r_frontal` is the population's frontal value of the ratio. Yaw only ever
shrinks the ratio, never grows it, so the frontal value sits at the top of
the observed distribution; the maximum is the noisiest order statistic and
the one landmark error reaches first, so a high percentile stands in for it.
The percentile is pre-registered at 95, swept at 90 and 99.

`cos` is even, so family 1 is unsigned, and strictly decreasing on
[0, 90] degrees, so the inverse is single-valued over the whole usable
range: nothing saturates and there is no second root. Rows whose ratio
exceeds `r_frontal` are out of the model's domain rather than saturated;
they clamp to theta = 0 and are counted, never given an invented angle.
"""

import numpy as np

# Pre-registered before any E4 metric existed (docs/backlog.md RSCH-4,
# second grooming pass). Every one of these is a choice that could otherwise
# have been made after seeing a number.
FRONTAL_PERCENTILE = 95.0
FRONTAL_PERCENTILE_SWEEP = (90.0, 95.0, 99.0)
N_BINS = 4
# d_rel = s / tan(theta) explodes as theta -> 0, so the per-row depth
# distribution is taken over rows above this angle only.
D_REL_MIN_DEGREES = 5.0


def frontal_ratio(ratios: np.ndarray, percentile: float = FRONTAL_PERCENTILE) -> float:
    """The population's frontal `w_obs / v_obs`, as a high percentile of the
    observed ratios. Computed once over all rows, pooled, and frozen into
    `experiments/e4/config.json` before any arm is scored."""
    return float(np.percentile(np.asarray(ratios, dtype=float), percentile))


def yaw_degrees(ratios: np.ndarray, r_frontal: float) -> tuple[np.ndarray, np.ndarray]:
    """(N,) ratios -> (N,) unsigned yaw in degrees, plus an (N,) bool mask of
    the out-of-domain rows.

    `ratio / r_frontal > 1` means `cos(theta) > 1`, which no yaw can produce:
    the row is wider-than-frontal for its vertical span, from facial-width
    variation or landmark error rather than pose. Those rows clamp to
    theta = 0 and land in the frontal bin. At the 95th percentile roughly 5%
    of rows do this by construction.
    """
    cos_theta = np.asarray(ratios, dtype=float) / r_frontal
    out_of_domain = cos_theta > 1.0
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))), out_of_domain


def relative_depth(
    offsets: np.ndarray, degrees: np.ndarray, min_degrees: float = D_REL_MIN_DEGREES
) -> np.ndarray:
    """`d_rel = s / tan(theta)`: family 2's depth term, measured per row
    instead of assumed, once family 1 supplies theta. Rows below
    `min_degrees` are returned as NaN -- `tan(theta) -> 0` there and the
    ratio is dominated by landmark noise, not by depth."""
    degrees = np.asarray(degrees, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_rel = np.asarray(offsets, dtype=float) / np.tan(np.radians(degrees))
    return np.where(degrees >= min_degrees, d_rel, np.nan)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks 1..n, ties sharing their average rank."""
    values = np.asarray(values, dtype=float)
    ranks = np.empty(len(values), dtype=float)
    ranks[values.argsort(kind="stable")] = np.arange(1, len(values) + 1)
    _unique, inverse = np.unique(values, return_inverse=True)
    totals = np.bincount(inverse, weights=ranks)
    counts = np.bincount(inverse)
    return (totals / counts)[inverse]


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rho: Pearson on average ranks. Hand-written on numpy
    because scipy is not a declared dependency (`pyproject.toml`; AGENTS.md
    forbids adding one without asking) -- it is present only transitively
    under scikit-learn."""
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])
