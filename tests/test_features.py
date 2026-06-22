import numpy as np
import pytest
from scipy.ndimage import maximum_filter
from trackastra.data import wrfeat
from trackastra.data.wrfeat import (
    WRAugmentationPipeline,
    WRFeatures,
    WRRandomAffine,
    WRRandomCrop,
    WRRandomMovement,
)

pytestmark = pytest.mark.core


def generate_data(ndim: int = 2, ngrid=10):
    y = np.zeros((4,) + (128,) * ndim, np.uint16)

    points = np.stack(
        np.meshgrid(
            *tuple(np.linspace(10, 100, ngrid).astype(int) for _ in range(ndim)),
            indexing="ij",
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


def test_random_affine_scale_is_isotropic_and_log_symmetric():
    x, y, _p, _ts = generate_data(ndim=2, ngrid=4)
    features = WRFeatures.from_mask_img(mask=y, img=x)
    augmentation = WRRandomAffine(
        p=1, degrees=0, scale=(0.5, 2), shear=(0, 0)
    )
    state = np.random.get_state()
    try:
        np.random.seed(42)
        scales = []
        for _ in range(2000):
            augmentation(features)
            assert augmentation._M[0, 0] == pytest.approx(augmentation._M[1, 1])
            scales.append(augmentation._M[0, 0])
    finally:
        np.random.set_state(state)

    assert np.mean(np.log(scales)) == pytest.approx(0, abs=0.03)
    assert np.mean(np.asarray(scales) < 1) == pytest.approx(0.5, abs=0.03)


def test_wr_augmentation_uses_seeded_process_rng():
    features = WRFeatures(
        coords=np.array([[0.0, 0.0], [1.0, 1.0]]),
        labels=np.array([1, 2]),
        timepoints=np.array([0, 1]),
        features={},
    )
    augmentation = WRRandomMovement(offset=(-10, 10), p=1)
    state = np.random.get_state()
    try:
        np.random.seed(42)
        first = augmentation(features).coords
        np.random.seed(42)
        repeated = augmentation(features).coords
        np.random.seed(43)
        other_worker = augmentation(features).coords
    finally:
        np.random.set_state(state)

    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, other_worker)


def test_random_crop_retains_centers_in_dimensions_that_fit():
    coords = np.array([
        [y, x] for x in (0.0, 350.0, 700.0) for y in (0.0, 400.0)
    ])
    features = WRFeatures(
        coords=coords,
        labels=np.arange(len(coords)),
        timepoints=np.zeros(len(coords), dtype=int),
        features={},
    )
    crop = WRRandomCrop(crop_size=(512, 512), ndim=2)

    for _ in range(100):
        cropped, idx = crop(features)
        # The y extent fits and must never be cropped. The x extent does not.
        assert np.array_equal(cropped.coords, features.coords[idx])
        for x in np.unique(cropped.coords[:, 1]):
            assert set(cropped.coords[cropped.coords[:, 1] == x, 0]) == {0.0, 400.0}


def test_random_crop_retains_all_centers_when_extent_fits():
    features = WRFeatures(
        coords=np.array([[0.0, 0.0], [400.0, 500.0], [200.0, 250.0]]),
        labels=np.arange(3),
        timepoints=np.zeros(3, dtype=int),
        features={},
    )
    crop = WRRandomCrop(crop_size=(512, 512), ndim=2)

    for _ in range(100):
        cropped, idx = crop(features)
        assert np.array_equal(idx, np.arange(3))
        assert np.array_equal(cropped.coords, features.coords)


def test_random_crop_can_disable_center_retention():
    features = WRFeatures(
        coords=np.array([[0.0, 0.0], [400.0, 500.0], [200.0, 250.0]]),
        labels=np.arange(3),
        timepoints=np.zeros(3, dtype=int),
        features={},
    )
    crop = WRRandomCrop(
        crop_size=(512, 512), ndim=2, ensure_all_centers=False
    )

    assert any(len(crop(features)[1]) < len(features) for _ in range(100))


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("properties", ["regionprops2", "regionprops"])
def test_fast_regionprops_matches_skimage(ndim, properties, monkeypatch):
    """fast_regionprops backend must match the skimage fallback bit-for-bit."""
    if not hasattr(wrfeat, "regionprops_table_fast"):
        pytest.skip("fast_regionprops not installed")
    x, y, _p, _ts = generate_data(ndim=ndim, ngrid=6)

    # force skimage fallback
    monkeypatch.setattr(wrfeat, "FAST_REGIONPROPS_INSTALLED", False)
    ref = WRFeatures.from_mask_img(mask=y, img=x, properties=properties)

    # use fast backend
    monkeypatch.setattr(wrfeat, "FAST_REGIONPROPS_INSTALLED", True)
    fast = WRFeatures.from_mask_img(mask=y, img=x, properties=properties)

    assert np.array_equal(ref.labels, fast.labels)
    assert np.array_equal(ref.timepoints, fast.timepoints)
    assert np.allclose(ref.coords, fast.coords, atol=1e-3)
    assert ref.features.keys() == fast.features.keys()
    for k in ref.features:
        a, b = ref.features[k], fast.features[k]
        atol = 1e-3 * max(1.0, np.abs(a).max())
        assert np.allclose(a, b, atol=atol), f"feature {k} differs"


if __name__ == "__main__":
    x, y, p, ts = generate_data(ndim=2, ngrid=10)
    feats = WRFeatures.from_mask_img(mask=y, img=x)

    np.random.seed(42)

    aug = WRAugmentationPipeline([
        # WRRandomFlip(p=0.5),
        # WRRandomAffine(p=1, degrees=180, scale=(0.5, 2), shear=(0.1, 0.1)),
        # WRRandomAffine(p=1, degrees=180, scale=(1, 1), shear=(0, 0)),
        WRRandomMovement(p=1),
        # WRRandomBrightness(p=1, factor=(0.5, 0.2)),
        # WRRandomOffset(p=1, offset=(-3, 3))
        # WRRandomAffine(p=1, degrees=None, scale=None, shear=(.1,.1)),
    ])

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
