import numpy as np
import pytest
from trackastra.data import apply_spatial_spacing
from trackastra.data.matching import match_points, matching

# Mark all tests in this module as core/inference tests
pytestmark = pytest.mark.core


def repeat_tile(x, repeats=(4, 4)):
    ry, rx = repeats
    x = np.repeat(x, ry, axis=0)
    x = np.repeat(x, rx, axis=1)
    return x


def test_matching():
    np.random.seed(42)

    gt = repeat_tile(np.arange(64).reshape((8, 8)), (16, 16))

    gt = np.zeros((128, 128), np.int32)
    gt[::16, ::16] = np.arange(64).reshape((8, 8))

    pred = gt + 1
    pred[pred == 1] = 0

    ds = tuple(matching(gt, np.roll(pred, r), max_distance=16) for r in range(17))

    tuple(len(d) for d in ds)


def test_match_points_uses_model_space_coordinates():
    proposals = np.array([[0.2, 0.0, 0.0], [0.0, 0.7, 0.0]], dtype=np.float32)
    gt = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    unscaled = match_points(proposals, gt, max_distance=0.75)
    scaled = match_points(
        apply_spatial_spacing(proposals, (4, 1, 1)),
        apply_spatial_spacing(gt, (4, 1, 1)),
        max_distance=0.75,
    )

    assert tuple((i, j) for i, j, _dist in unscaled) == ((0, 0),)
    assert tuple((i, j) for i, j, _dist in scaled) == ((1, 0),)


def test_match_points_validates_inputs():
    with pytest.raises(ValueError, match="same coordinate dimension"):
        match_points(np.zeros((1, 2)), np.zeros((1, 3)), max_distance=1)
    with pytest.raises(ValueError, match="non-negative"):
        match_points(np.zeros((1, 2)), np.zeros((1, 2)), max_distance=-1)
