import numpy as np
from scipy.ndimage import maximum_filter
from trackastra.data.wrfeat import (
    WRAugmentationPipeline,
    WRFeatures,
    WRRandomMovement,
)


def generate_data(ndim: int = 2, ngrid=10):
    y = np.zeros((4,) + (128,) * ndim, np.uint16)

    points = np.stack(
        np.meshgrid(
            *tuple(np.linspace(10, 100, ngrid).astype(int) for _ in range(ndim)),
            indexing="ij"
        ),
        axis=-1,
    ).reshape(-1, ndim)
    ts = np.random.randint(0, 4, len(points))
    for i in range(len(y)):
        p = points[ts == i]
        y[i][tuple(p.T)] = np.arange(1, len(p) + 1) + y.max()

    size = list((0,) + (3,) * ndim)
    size[-1] *= 2
    y = maximum_filter(y, size=size)
    x = y + 0.2 + 0.2 * y.max() * np.random.uniform(0, 1, y.shape)
    return x, y, points, ts


def test_features():
    x, y, _p, _ts = generate_data(ndim=2, ngrid=10)
    WRFeatures.from_mask_img(mask=y, img=x)


if __name__ == "__main__":

    x, y, p, ts = generate_data(ndim=2, ngrid=10)
    feats = WRFeatures.from_mask_img(mask=y, img=x)

    np.random.seed(42)

    aug = WRAugmentationPipeline(
        [
            # WRRandomFlip(p=0.5),
            # WRRandomAffine(p=1, degrees=180, scale=(0.5, 2), shear=(0.1, 0.1)),
            # WRRandomAffine(p=1, degrees=180, scale=(1, 1), shear=(0, 0)),
            WRRandomMovement(p=1),
            # WRRandomBrightness(p=1, factor=(0.5, 0.2)),
            # WRRandomOffset(p=1, offset=(-3, 3))
            # WRRandomAffine(p=1, degrees=None, scale=None, shear=(.1,.1)),
        ]
    )

    feats2 = aug(feats)
    # aug = WRRandomAffine(p=1, degrees=180, scale=None, shear=None)

    # from trackastra.data.wrfeat import _rotation_matrix, _transform_affine
    # from scipy.ndimage import rotate

    # degrees = 180
    # M = _rotation_matrix(degrees / 180 * np.pi)[1:, 1:]

    # feats = WRFeatures.from_mask_img(mask=y, img=x)
    # x0 = feats["inertia_tensor"]
    # x1 = _transform_affine("inertia_tensor", feats["inertia_tensor"], M)
    # feats2 = WRFeatures.from_mask_img(
    #     mask=rotate(y, degrees, axes=(1, 2), order=0),
    #     img=rotate(x, degrees, axes=(1, 2)),
    # )
    # x2 = feats2["inertia_tensor"]

    # print(x0[0].round(2))
    # print(x1[0].round(2))
    # print(x2[0].round(2))

    # feats2 = aug(feats)

    # (x2, y2, p2), idx = pipe(x, y, p, ts)

    # for _ in range(10):
    #     (x2, y2, p2), idx = pipe(x, y, p)
    #     print(len(idx))
