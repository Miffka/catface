"""Model (2): logistic regression on all 96 Procrustes-aligned coordinates
-- same input as model (3), the MLP.
"""

from sklearn.linear_model import LogisticRegression

from catface.ml import cv
from catface.ml.features import load_features


def build_model() -> LogisticRegression:
    return LogisticRegression()


def run() -> dict:
    features = load_features()
    metrics = cv.cross_validate(
        features.coord_X, features.y, features.split, build_model, oversample=True
    )
    metrics["labels"] = features.classes
    metrics["model"] = "coords_lr"
    metrics["n_features"] = 96
    return metrics
