"""E4 (RSCH-4): the degree calibration, the hand-written Spearman, and the
two pre-registered verdict functions.

The calibration test is a round trip through the *real* pipeline against an
independently constructed 3D truth: a synthetic bilaterally symmetric cat
face, rotated by known angles with a rotation matrix written here, projected
orthographically, then fed to `core.geometry.procrustes_align` ->
`yaw_foreshortening_ratio` -> `pose.yaw_degrees`. Asserting
`yaw_degrees(ratio_from_yaw(theta)) == theta` would only test the inverse
against its own forward function and would pass with both of them wrong.
"""

import numpy as np
import pytest

from catface.core.geometry import (
    BILATERAL_PAIRS,
    CHIN,
    MIDLINE_POINTS,
    OCULAR_PAIR,
    PHILTRUM,
    procrustes_align,
    yaw_foreshortening_ratio,
    yaw_midline_offset,
)
from catface.ml import pose, scoring

# The philtrum's depth ahead of the eye-corner plane, in units of the
# eye-corner span: what the fixture builds in and `relative_depth` must
# measure back out.
BUILT_IN_D_REL = 0.30
# The synthetic face is exact under the weak-perspective model, so this
# tolerance covers float error in the Procrustes fit only, not modelling
# slack.
DEGREE_TOLERANCE = 0.01


def synthetic_cat_face(d_rel: float = BUILT_IN_D_REL) -> np.ndarray:
    """A 48-point bilaterally symmetric 3D face: every `BILATERAL_PAIRS`
    member mirrored about x = 0, every `MIDLINE_POINTS` member on x = 0, the
    philtrum protruding toward the camera by `d_rel` eye-corner spans and the
    chin sitting in the eye-corner plane."""
    points = np.zeros((48, 3))
    rng = np.random.default_rng(0)
    for k, (left, right) in enumerate(BILATERAL_PAIRS):
        x, y, z = 0.1 + 0.02 * k, 0.4 - 0.05 * k, rng.uniform(-0.2, 0.4)
        points[left] = (-x, y, z)
        points[right] = (x, y, z)

    # Eye corners: span exactly 1.0, in the z = 0 plane, on the y = 0 line.
    points[OCULAR_PAIR[0]] = (-0.5, 0.0, 0.0)
    points[OCULAR_PAIR[1]] = (0.5, 0.0, 0.0)
    # Midline. Chin one span below the eye-corner line -> frontal ratio 1.0.
    points[PHILTRUM] = (0.0, -0.4, d_rel)
    points[CHIN] = (0.0, -1.0, 0.0)
    points[17] = (0.0, -0.55, 0.15)  # mouth top
    points[0] = (0.0, -0.7, 0.1)  # mouth
    # 19 and 21 are named in the v3 legend but are not a matched pair; park
    # them off the midline so the face is not accidentally symmetric there.
    points[19] = (0.3, -0.5, 0.1)
    points[21] = (-0.35, -0.6, 0.05)
    return points


def rotate_yaw(points: np.ndarray, degrees: float) -> np.ndarray:
    """Rotation about the vertical axis, written here rather than imported,
    so the test's truth does not come from the code under test."""
    t = np.radians(degrees)
    R = np.array([[np.cos(t), 0.0, np.sin(t)], [0.0, 1.0, 0.0], [-np.sin(t), 0.0, np.cos(t)]])
    return points @ R.T


def project(points_3d: np.ndarray) -> np.ndarray:
    """Weak perspective: u = x, v = y, after the rotation has mixed z into x."""
    return points_3d[:, :2]


# Kept clear of D_REL_MIN_DEGREES so float error at the cutoff cannot decide
# how many rows the depth distribution covers.
_ANGLES = np.array([-55.0, -40.0, -20.0, -8.0, 0.0, 8.0, 20.0, 40.0, 55.0])


def _aligned_stack(d_rel: float = BUILT_IN_D_REL) -> np.ndarray:
    """The real preprocessing: project each rotated face, then Procrustes-
    align it to the frontal projection exactly as the run aligns to the GPA
    mean."""
    face = synthetic_cat_face(d_rel)
    reference = project(face)
    return np.stack([procrustes_align(project(rotate_yaw(face, a)), reference) for a in _ANGLES])


def test_yaw_degrees_recovers_the_angle_the_face_was_rotated_by():
    shapes = _aligned_stack()
    ratios = yaw_foreshortening_ratio(shapes)
    # The frontal reference is the face's own zero-yaw ratio, measured
    # through the same pipeline, which is what `frontal_ratio`'s percentile
    # rule estimates from a population.
    r_frontal = pose.frontal_ratio(ratios, percentile=100.0)
    degrees, out_of_domain = pose.yaw_degrees(ratios, r_frontal)

    assert np.allclose(degrees, np.abs(_ANGLES), atol=DEGREE_TOLERANCE)
    assert not out_of_domain.any()


def test_family_one_is_unsigned_and_family_two_carries_the_sign():
    shapes = _aligned_stack()
    offsets = yaw_midline_offset(shapes)

    # Family 1 cannot tell a left turn from a right turn: +/-40 degrees give
    # the same ratio. Family 2 separates them.
    ratios = yaw_foreshortening_ratio(shapes)
    minus_40, plus_40 = list(_ANGLES).index(-40.0), list(_ANGLES).index(40.0)
    assert np.isclose(ratios[minus_40], ratios[plus_40])
    assert offsets[minus_40] < 0 < offsets[plus_40]
    assert np.isclose(offsets[list(_ANGLES).index(0.0)], 0.0, atol=1e-9)
    # Monotonic in the signed angle over the whole range, no turning over.
    assert np.all(np.diff(offsets) > 0)


def test_relative_depth_measures_back_the_depth_the_fixture_built_in():
    shapes = _aligned_stack()
    degrees, _ = pose.yaw_degrees(
        yaw_foreshortening_ratio(shapes), pose.frontal_ratio(yaw_foreshortening_ratio(shapes), 100.0)
    )
    d_rel = pose.relative_depth(np.abs(yaw_midline_offset(shapes)), degrees)

    measured = d_rel[np.isfinite(d_rel)]
    assert len(measured) == (np.abs(_ANGLES) >= pose.D_REL_MIN_DEGREES).sum()
    assert np.allclose(measured, BUILT_IN_D_REL, atol=1e-6)

    # The chin control: it sits in the eye-corner plane, so its implied depth
    # is far smaller in magnitude than the philtrum's. This is a falsifiable
    # prediction of the geometry, not a restatement of it.
    chin_offsets = yaw_midline_offset(shapes, midline_index=CHIN)
    chin_d_rel = pose.relative_depth(np.abs(chin_offsets), degrees)
    assert np.nanmax(np.abs(chin_d_rel)) < 0.01 * BUILT_IN_D_REL


def test_synthetic_face_is_actually_symmetric():
    """Guards the fixture itself: a face that is not bilaterally symmetric
    would make the round trip above pass for the wrong reason."""
    face = synthetic_cat_face()
    for left, right in BILATERAL_PAIRS:
        assert np.allclose(face[left] * (-1, 1, 1), face[right])
    for index in MIDLINE_POINTS:
        assert face[index, 0] == 0.0


def test_yaw_degrees_clamps_out_of_domain_rows_to_zero():
    ratios = np.array([1.2, 1.0, 0.5])
    degrees, out_of_domain = pose.yaw_degrees(ratios, r_frontal=1.0)
    assert out_of_domain.tolist() == [True, False, False]
    assert degrees[0] == 0.0  # clamped, not an invented angle
    assert np.isclose(degrees[2], 60.0)


def test_frontal_ratio_is_the_requested_percentile():
    ratios = np.arange(101, dtype=float)
    assert pose.frontal_ratio(ratios, 95.0) == 95.0
    assert pose.frontal_ratio(ratios, 99.0) == 99.0


# --- Spearman (no scipy) ----------------------------------------------------


def test_average_ranks_handles_ties():
    assert pose.average_ranks([10.0, 20.0, 20.0, 40.0]).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_rho_against_a_hand_computed_case_with_a_tie():
    """x = [1, 2, 3, 4, 5], y = [2, 1, 4, 4, 5], with the tied pair of 4s.
    Ranks of x: [1, 2, 3, 4, 5], of y: [2, 1, 3.5, 3.5, 5]; both mean 3.
    Deviations multiply to 2 + 2 + 0 + 0.5 + 4 = 8.5, so cov = 1.7;
    var_x = 10/5 = 2, var_y = 9.5/5 = 1.9.
    rho = 1.7 / sqrt(2 * 1.9) = 0.872081599272381."""
    rho = pose.spearman_rho([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 4.0, 4.0, 5.0])
    assert rho == pytest.approx(1.7 / np.sqrt(2.0 * 1.9))


def test_spearman_rho_is_one_for_a_monotonic_nonlinear_relation():
    x = np.arange(1.0, 20.0)
    assert pose.spearman_rho(x, np.exp(x)) == pytest.approx(1.0)


# --- Binning and the controls' statistics -----------------------------------


def test_quantile_bins_are_equal_count_and_ordered():
    values = np.arange(100, dtype=float)
    edges, index = scoring.quantile_bins(values, 4)
    assert len(edges) == 5
    assert np.bincount(index).tolist() == [25, 25, 25, 25]
    assert index[0] == 0 and index[-1] == 3


def test_total_variation_and_crosstalk_known_values():
    assert scoring.total_variation([1, 1], [1, 1]) == 0.0
    assert scoring.total_variation([2, 0], [1, 1]) == pytest.approx(0.5)
    confusion = np.array([[8, 2, 0], [4, 6, 0], [0, 0, 10]])
    # (2 + 4) / (10 + 10)
    assert scoring.crosstalk(confusion, 0, 1) == pytest.approx(0.3)


def test_bootstrap_ci_brackets_a_known_mean():
    values = np.random.default_rng(1).normal(5.0, 1.0, 400)
    lo, hi = scoring.bootstrap_ci(lambda idx: float(values[idx].mean()), len(values))
    assert lo < 5.0 < hi
    assert hi - lo < 0.5


# --- The verdict functions --------------------------------------------------


def _bin(index, kappa, ci, tv=0.0, n=490):
    return {
        "index": index,
        "n": n,
        "kappa": kappa,
        "kappa_ci": ci,
        "macro_f1": 0.4,
        "accuracy": 0.5,
        "majority_rate": 0.57,
        "tv": tv,
        "edge_ratio": 1.5 - 0.1 * index,
        "edge_degrees": {90.0: 10.0 * index, 95.0: 12.0 * index, 99.0: 14.0 * index},
    }


_GOOD_RHO = (0.7, (0.6, 0.8))


def test_q3_chance_gate_fires_when_bin_zero_cannot_clear_zero():
    bins = [_bin(0, 0.02, (-0.05, 0.09)), _bin(1, 0.2, (0.1, 0.3))]
    status, text = scoring.q3_verdict("gnn_identity", bins, *_GOOD_RHO)
    assert status == "INCONCLUSIVE"
    assert "Chance gate fired" in text and "no pose penalty" in text


def test_q3_degradation_rule_fires_and_names_the_ratio_edge_and_degrees():
    bins = [
        _bin(0, 0.30, (0.20, 0.40)),
        _bin(1, 0.25, (0.15, 0.35)),
        _bin(2, 0.05, (-0.05, 0.15)),
    ]
    status, text = scoring.q3_verdict("gnn_identity", bins, *_GOOD_RHO)
    assert status == "POSITIVE"
    assert "degradation rule fired at bin 2" in text
    assert "foreshortening ratio 1.300" in text
    assert "24.0 deg at the 95th-percentile" in text and "20.0-28.0 deg" in text


def test_q3_reports_negative_when_no_bin_degrades():
    bins = [_bin(0, 0.30, (0.20, 0.40)), _bin(1, 0.25, (0.15, 0.35))]
    status, text = scoring.q3_verdict("gnn_identity", bins, *_GOOD_RHO)
    assert status == "NEGATIVE"
    assert "degradation rule did not fire" in text


def test_q3_prior_shift_makes_the_verdict_provisional_and_names_the_bin():
    bins = [_bin(0, 0.30, (0.20, 0.40)), _bin(1, 0.25, (0.15, 0.35), tv=0.09)]
    status, text = scoring.q3_verdict("gnn_identity", bins, *_GOOD_RHO)
    assert status == "NEGATIVE (PROVISIONAL)"
    assert "bin 1 (TV 0.090)" in text


def test_q3_withdraws_the_yaw_label_and_every_degree_when_rho_is_low():
    bins = [
        _bin(0, 0.30, (0.20, 0.40)),
        _bin(1, 0.25, (0.15, 0.35)),
        _bin(2, 0.05, (-0.05, 0.15)),
    ]
    _status, text = scoring.q3_verdict("gnn_identity", bins, 0.12, (0.05, 0.19))
    assert "yaw label is not earned" in text
    assert "foreshortening ratio 1.300" in text  # the ratio edge still appears
    assert "deg at the" not in text and "percentile" not in text.split("earned")[1]


def test_q4_null_path_when_no_pair_clears_both_thresholds():
    pairs = [
        {"names": ("attentive", "relaxed"), "crosstalk": 0.30, "kappa": 0.21,
         "kappa_ci": (0.12, 0.29), "candidate": False},
        {"names": ("attentive", "uncomfortable"), "crosstalk": 0.10, "kappa": 0.18,
         "kappa_ci": (-0.02, 0.35), "candidate": False},
    ]
    status, text = scoring.q4_verdict(pairs, None, "gnn_identity", 0.23, (0.18, 0.28))
    assert status == "NEGATIVE"
    assert "no merge is warranted on the evidence" in text
    assert "pre-registered null path" in text


def test_q4_accepts_a_merge_only_if_retrained_beats_collapsed_and_both_clear_the_dummy():
    pairs = [{"names": ("attentive", "relaxed"), "crosstalk": 0.40, "kappa": 0.02,
              "kappa_ci": (-0.05, 0.09), "candidate": True}]
    merge = {"names": ("attentive", "relaxed"), "retrained_kappa": 0.31,
             "collapsed_kappa": 0.28, "dummy_kappa": 0.0}
    status, _text = scoring.q4_verdict(pairs, merge, "gnn_identity", 0.23, (0.18, 0.28))
    assert status == "POSITIVE"

    merge["retrained_kappa"] = 0.20  # below the post-hoc collapse
    status, text = scoring.q4_verdict(pairs, merge, "gnn_identity", 0.23, (0.18, 0.28))
    assert status == "NEGATIVE"
    assert "merging bought nothing" in text
