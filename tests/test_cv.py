"""`cv.cross_validate`'s out-of-fold predictions (E4/RSCH-4)."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from catface.ml.cv import cross_validate

# Twelve rows, three folds, labels that a majority-class dummy reproduces
# deterministically -- the point here is the plumbing, not the model.
_X = np.arange(12).reshape(12, 1).astype(float)
_Y = np.array([0, 1, 2] * 4)
_SPLIT = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])


def _build():
    return DummyClassifier(strategy="most_frequent")


def test_default_result_keys_and_order_are_unchanged():
    """E2's and E3's committed metrics.json must stay byte-identical without
    a regeneration, so the default call's key insertion order is pinned."""
    result = cross_validate(_X, _Y, _SPLIT, _build, n_splits=3)
    assert list(result) == [
        "per_fold_macro_f1",
        "mean_macro_f1",
        "std_macro_f1",
        "per_fold_kappa",
        "mean_kappa",
        "std_kappa",
        "per_fold_mcc",
        "mean_mcc",
        "std_mcc",
        "confusion_matrix",
        "labels",
    ]


def test_oof_predictions_cover_every_row_and_match_the_confusion_matrix():
    result = cross_validate(_X, _Y, _SPLIT, _build, n_splits=3, return_oof=True)
    oof = np.array(result["oof_pred"])

    assert list(result)[-1] == "oof_pred"  # appended, not interleaved
    assert oof.shape == (len(_Y),)
    assert (oof >= 0).all()

    # The pooled out-of-fold predictions must sum to the same confusion
    # matrix cross_validate accumulated fold by fold.
    labels = result["labels"]
    pooled = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(_Y, oof):
        pooled[labels.index(true), labels.index(pred)] += 1
    assert pooled.tolist() == result["confusion_matrix"]


def test_oof_raises_when_a_row_is_in_no_test_fold():
    split = _SPLIT.copy()
    split[0] = -1  # what make_splits leaves behind if a row is never assigned
    with pytest.raises(ValueError, match="never in a test fold"):
        cross_validate(_X, _Y, split, _build, n_splits=3, return_oof=True)
