import numpy as np

from catface.core.geometry import (
    crop_and_resize,
    ear_angle,
    expand_box,
    eye_aspect_ratio,
    generalized_procrustes,
    letterbox_square,
    map_points_to_image,
    muzzle_spread_ratio,
    procrustes_align,
    unletterbox_xyxy,
)


def test_letterbox_round_trip():
    image = np.zeros((50, 100, 3), dtype=np.uint8)  # h=50, w=100
    letterboxed, scale, pad_x, pad_y = letterbox_square(image, size=224)
    assert letterboxed.shape == (224, 224, 3)

    orig_box = (10.0, 5.0, 90.0, 45.0)
    box_norm = (
        (orig_box[0] * scale + pad_x) / 224,
        (orig_box[1] * scale + pad_y) / 224,
        (orig_box[2] * scale + pad_x) / 224,
        (orig_box[3] * scale + pad_y) / 224,
    )
    recovered = unletterbox_xyxy(box_norm, 224, scale, pad_x, pad_y, orig_w=100, orig_h=50)
    assert all(abs(a - b) < 1e-6 for a, b in zip(recovered, orig_box))


def test_expand_box_grows_and_clips():
    grown = expand_box((40.0, 40.0, 60.0, 60.0), margin=0.5, img_w=1000, img_h=1000)
    assert grown == (30, 30, 70, 70)

    clipped = expand_box((0.0, 0.0, 10.0, 10.0), margin=1.0, img_w=100, img_h=100)
    assert clipped == (0, 0, 20, 20)


def test_crop_and_resize_shape():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = crop_and_resize(image, (10, 10, 110, 60), size=384)
    assert crop.shape == (384, 384, 3)


def test_map_points_to_image_round_trip():
    box = (20, 30, 120, 230)  # x1,y1,x2,y2 -> width 100, height 200
    points_norm = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    mapped = map_points_to_image(points_norm, box)
    expected = np.array([[20.0, 30.0], [70.0, 130.0], [120.0, 230.0]])
    assert np.allclose(mapped, expected)


def _center_scale(shape):
    centered = shape - shape.mean(axis=0)
    return centered / np.linalg.norm(centered)


# A scalene, non-symmetric point set (not a regular/symmetric polygon) so a
# reflection is never confusable with some rotation of the same shape.
_SCALENE_SHAPE = np.array(
    [
        [0.0, 0.0],
        [3.0, 0.2],
        [1.0, 2.0],
        [4.0, 3.0],
        [-1.0, 1.5],
        [2.0, -1.0],
    ]
)


def test_procrustes_align_self_is_noop():
    aligned = procrustes_align(_SCALENE_SHAPE, _SCALENE_SHAPE)
    assert np.allclose(aligned, _center_scale(_SCALENE_SHAPE))


def test_procrustes_align_recovers_original_after_similarity_transform():
    theta = np.radians(37)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    transformed = (_SCALENE_SHAPE @ rot.T) * 2.5 + np.array([10.0, -4.0])

    aligned = procrustes_align(transformed, _SCALENE_SHAPE)
    assert np.allclose(aligned, _center_scale(_SCALENE_SHAPE), atol=1e-8)


def test_procrustes_align_forbids_reflection():
    reflected = _SCALENE_SHAPE.copy()
    reflected[:, 0] *= -1  # mirror across the y-axis

    reference_n = _center_scale(_SCALENE_SHAPE)
    shape_n = _center_scale(reflected)

    # An unconstrained (reflection-allowed) Kabsch fit finds the exact mirror
    # transform and matches the reference perfectly -- that's the "silently
    # accept the mirror solution" trap this function must avoid.
    U, _, Vt = np.linalg.svd(shape_n.T @ reference_n)
    naive_R = U @ Vt
    assert np.linalg.det(naive_R) < 0  # this pair genuinely needs a reflection
    assert np.allclose(shape_n @ naive_R, reference_n, atol=1e-8)

    # procrustes_align forbids that: since no pure rotation can undo a
    # reflection of an asymmetric shape, it does not land on the reference,
    # proving it rejected the mirror shortcut rather than silently taking it.
    aligned = procrustes_align(reflected, _SCALENE_SHAPE)
    assert not np.allclose(aligned, reference_n, atol=1e-2)

    # And the rotation it actually applied is a proper rotation (det > 0),
    # recovered by solving aligned = shape_n @ R for R.
    recovered_R, *_ = np.linalg.lstsq(shape_n, aligned, rcond=None)
    assert np.linalg.det(recovered_R) > 0


def test_generalized_procrustes_mean_is_fixed_point():
    rng = np.random.default_rng(0)
    shapes = []
    for i in range(6):
        theta = np.radians(15 * i + 5)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        noise = rng.normal(scale=0.05, size=_SCALENE_SHAPE.shape)
        shape = (_SCALENE_SHAPE + noise) @ rot.T * (1.0 + 0.1 * i) + np.array([i, -i])
        shapes.append(shape)
    shapes = np.stack(shapes)

    aligned, mean_shape = generalized_procrustes(shapes)
    assert aligned.shape == shapes.shape
    assert mean_shape.shape == _SCALENE_SHAPE.shape

    # The mean shape is a fixed point of procrustes_align.
    self_aligned = procrustes_align(mean_shape, mean_shape)
    assert np.allclose(self_aligned, mean_shape, atol=1e-8)

    # Re-running one more alignment pass of all shapes to the returned mean
    # doesn't move the mean further (within tol).
    realigned = np.stack([procrustes_align(shape, mean_shape) for shape in shapes])
    new_mean = _center_scale(realigned.mean(axis=0))
    assert np.linalg.norm(new_mean - mean_shape) < 1e-6


# Three points: corners 4 apart on the x-axis, one point 1 above their
# midpoint -- corner-to-corner (horizontal) = 4, perpendicular (vertical) = 1.
_EYE_SHAPE = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 1.0]])
_EYE_INDICES = (0, 1, 2)


def test_eye_aspect_ratio_known_rectangle():
    ratio = eye_aspect_ratio(_EYE_SHAPE, _EYE_INDICES)
    assert np.isclose(ratio, 0.25)


def test_eye_aspect_ratio_index_order_invariant():
    # Corner-to-corner is found by max pairwise distance, not index order,
    # so shuffling the eye's point order must not change the ratio.
    shuffled = _EYE_SHAPE[[2, 0, 1]]
    ratio = eye_aspect_ratio(shuffled, (0, 1, 2))
    assert np.isclose(ratio, 0.25)


# Ear base at the eye-center midpoint, tip straight "up" (0, 2): the ear
# vector is perpendicular to a horizontal inter-ocular axis, so the angle
# is exactly 90 degrees.
_EAR_SHAPE = np.array([[0.0, 0.0], [0.0, 2.0]])
_EAR_INDICES = (0, 1)
_LEFT_EYE_CENTER = np.array([-1.0, 0.0])
_RIGHT_EYE_CENTER = np.array([1.0, 0.0])


def test_ear_angle_perpendicular_to_ocular_axis():
    angle = ear_angle(_EAR_SHAPE, _EAR_INDICES, _LEFT_EYE_CENTER, _RIGHT_EYE_CENTER)
    assert np.isclose(angle, 90.0)


def test_ear_angle_aligned_with_ocular_axis_is_zero():
    ear_shape = np.array([[0.0, 0.0], [2.0, 0.0]])  # base at midpoint, tip along +x
    angle = ear_angle(ear_shape, _EAR_INDICES, _LEFT_EYE_CENTER, _RIGHT_EYE_CENTER)
    assert np.isclose(angle, 0.0)


# Muzzle points 3 apart; inter-ocular distance 2 -> ratio 1.5.
_MUZZLE_SHAPE = np.array([[0.0, 0.0], [3.0, 0.0]])
_MUZZLE_INDICES = (0, 1)


def test_muzzle_spread_ratio_known_values():
    ratio = muzzle_spread_ratio(
        _MUZZLE_SHAPE, _MUZZLE_INDICES, _LEFT_EYE_CENTER, _RIGHT_EYE_CENTER
    )
    assert np.isclose(ratio, 1.5)
