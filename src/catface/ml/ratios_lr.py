"""Model (1): logistic regression on the two geometric ratios (eye aspect
ratio, ear angle) -- the Q2 baseline E2 tests a learned model against.
"""

from sklearn.linear_model import LogisticRegression

from catface.ml import cv
from catface.ml.features import load_features


def build_model() -> LogisticRegression:
    return LogisticRegression(class_weight="balanced")


def run() -> dict:
    features = load_features()
    metrics = cv.cross_validate(features.ratio_X, features.y, features.split, build_model)
    metrics["labels"] = features.classes
    metrics["model"] = "ratios_lr"
    metrics["n_features"] = 2
    return metrics
