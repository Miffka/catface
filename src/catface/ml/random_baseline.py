"""Chance-level reference: a uniform-random classifier, run on the identical
folds as the three real E2 models, so their macro F1/kappa/MCC have something
to compare against. Not a competing model, excluded from the Q2 verdict.
"""

from sklearn.dummy import DummyClassifier

from catface.ml import cv
from catface.ml.features import load_features


def build_model() -> DummyClassifier:
    return DummyClassifier(strategy="uniform", random_state=0)


def run() -> dict:
    features = load_features()
    # Feature array choice doesn't matter: the uniform strategy ignores X
    # entirely and predicts each class with equal probability.
    metrics = cv.cross_validate(
        features.ratio_X, features.y, features.split, build_model, oversample=True
    )
    metrics["labels"] = features.classes
    metrics["model"] = "random_baseline"
    metrics["n_features"] = None
    return metrics
