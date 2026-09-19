import numpy as np

from catface.ml.plausibility import EAR, EYE, MUZZLE, check_landmarks

BOX = (0.0, 0.0, 200.0, 200.0)
IMG_W = IMG_H = 300


def _valid_landmarks() -> np.ndarray:
    landmarks = np.zeros((48, 2))
    for i, idx in enumerate(EAR):
        landmarks[idx] = (20 + i * 5, 20)
    for i, idx in enumerate(EYE):
        landmarks[idx] = (20 + i * 5, 60)
    for i, idx in enumerate(MUZZLE):
        landmarks[idx] = (20 + i * 6, 140)
    return landmarks


def test_valid_landmarks_pass():
    assert check_landmarks(BOX, _valid_landmarks(), IMG_W, IMG_H) == []


def test_out_of_box():
    landmarks = _valid_landmarks()
    landmarks[EYE[0]] = (290, 60)
    assert "out_of_box" in check_landmarks(BOX, landmarks, IMG_W, IMG_H)


def test_eyes_below_muzzle():
    landmarks = _valid_landmarks()
    for idx in EYE:
        landmarks[idx, 1] = 180
    for idx in MUZZLE:
        landmarks[idx, 1] = 60
    assert "eyes_below_muzzle" in check_landmarks(BOX, landmarks, IMG_W, IMG_H)


def test_degenerate_collapsed():
    landmarks = np.full((48, 2), 100.0)
    assert "degenerate" in check_landmarks(BOX, landmarks, IMG_W, IMG_H)


def test_bad_aspect_ratio():
    wide_box = (0.0, 0.0, 200.0, 20.0)
    landmarks = _valid_landmarks()
    landmarks[:, 1] /= 10
    assert "bad_aspect_ratio" in check_landmarks(wide_box, landmarks, IMG_W, IMG_H)
