import numpy as np

from catface.core.geometry import (
    crop_and_resize,
    expand_box,
    letterbox_square,
    map_points_to_image,
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
