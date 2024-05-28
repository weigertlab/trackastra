import numpy as np
from trackastra.data.matching import matching


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
