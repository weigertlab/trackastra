import numpy as np
import pytest
from scipy.ndimage import maximum_filter
from trackastra.data import wrfeat
from trackastra.data.wrfeat import (
    WRAugmentationPipeline,
    WRFeatures,
    WRRandomAffine,
    WRRandomCrop,
    WRRandomFrameJump,
    WRRandomMovement,
    _rotation_matrix,
    build_windows,
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


def _wrfeat2_example():
    return WRFeatures(
        coords=np.zeros((3, 2), dtype=np.float32),
        labels=np.arange(3),
        timepoints=np.zeros(3, dtype=int),
        features={
            "equivalent_diameter_area": np.full((3, 1), 2, dtype=np.float32),
            "intensity": np.array([[0.1], [0.2], [0.3]], dtype=np.float32),
            "inertia_tensor": np.array(
                [
                    [2, 0, 0, 2],  # isotropic
                    [3, 0, 0, 1],  # anisotropy along an axis
                    [2, 1, 1, 2],  # same anisotropy, rotated 45 degrees
                ],
                dtype=np.float32,
            ),
            "border_dist": np.array([[0], [0.5], [1]], dtype=np.float32),
        },
    )


def _wrfeat2_3d_example():
    return WRFeatures(
        coords=np.zeros((3, 3), dtype=np.float32),
        labels=np.arange(3),
        timepoints=np.zeros(3, dtype=int),
        features={
            "equivalent_diameter_area": np.full((3, 1), 2, dtype=np.float32),
            "intensity": np.array([[0.1], [0.2], [0.3]], dtype=np.float32),
            "inertia_tensor": np.array(
                [
                    np.eye(3).ravel(),
                    np.diag([4, 1, 1]).ravel(),
                    np.array([[2, 1, 0], [1, 2, 0], [0, 0, 2]]).ravel(),
                ],
                dtype=np.float32,
            ),
            "border_dist": np.array([[0], [0.5], [1]], dtype=np.float32),
        },
    )


def _features_from_covariance(covariance: np.ndarray, diameter: float = 2) -> WRFeatures:
    ndim = covariance.shape[0]
    inertia = np.trace(covariance) * np.eye(ndim) - covariance
    return WRFeatures(
        coords=np.zeros((1, ndim), dtype=np.float32),
        labels=np.ones(1, dtype=np.int32),
        timepoints=np.zeros(1, dtype=np.int32),
        features={
            "equivalent_diameter_area": np.array([[diameter]], dtype=np.float32),
            "intensity": np.array([[0.25]], dtype=np.float32),
            "inertia_tensor": inertia.reshape(1, -1).astype(np.float32),
            "border_dist": np.array([[0.5]], dtype=np.float32),
        },
    )


def test_wrfeat2_decomposes_inertia_and_bounds_orientation():
    features = _wrfeat2_example().features_stacked_for("wrfeat2")

    assert features.shape == (3, 6)
    assert features.dtype == np.float32
    assert np.allclose(features[:, 0], np.log1p(2))
    assert np.allclose(features[:, 1], [0.1, 0.2, 0.3])
    assert np.allclose(features[:, 2], np.log1p(4 / np.pi))
    assert np.allclose(features[:, 3:5], [[0, 0], [0.5, 0], [0, 0.5]])
    assert np.allclose(features[:, 5], np.log1p([0, 0.5, 1]))
    assert np.all(np.linalg.norm(features[:, 3:5], axis=1) <= 1)


def test_wrfeat3_has_one_explicit_fixed_width_schema():
    expected_names = [
        "log_equivalent_diameter",
        "mean_normalized_intensity",
        "log_normalized_second_moment",
        "xy_anisotropy",
        "xy_orientation",
        "z_elongation",
        "zy_coupling",
        "zx_coupling",
        "log_border_dist",
    ]

    assert [c.name for c in wrfeat.feature_channels("wrfeat3", 2)] == expected_names
    assert [c.name for c in wrfeat.feature_channels("wrfeat3", 3)] == expected_names
    assert wrfeat.feature_output_dim("wrfeat3", 2) == 9
    assert wrfeat.feature_output_dim("wrfeat3", 3) == 9
    assert wrfeat.feature_schema_manifest("wrfeat3") == {
        "name": "wrfeat3",
        "version": 1,
        "channels": expected_names,
    }


def test_wrfeat3_isotropic_second_moment_matches_in_2d_and_3d():
    radius = 2.0
    covariance_2d = np.eye(2, dtype=np.float32) * radius**2 / 4
    covariance_3d = np.eye(3, dtype=np.float32) * radius**2 / 5

    features_2d = _features_from_covariance(
        covariance_2d, diameter=2 * radius
    ).features_stacked_for("wrfeat3")
    features_3d = _features_from_covariance(
        covariance_3d, diameter=2 * radius
    ).features_stacked_for("wrfeat3")

    assert features_2d.shape == features_3d.shape == (1, 9)
    assert features_2d[0, 2] == pytest.approx(np.log(2), abs=1e-6)
    assert features_3d[0, 2] == pytest.approx(np.log(2), abs=1e-6)


@pytest.mark.parametrize("ndim", [2, 3])
def test_wrfeat3_regionprops_disk_and_ball_are_isotropic(ndim):
    shape = (48,) * ndim
    center = np.array(shape) // 2
    coords = np.indices(shape)
    squared_distance = sum(
        np.square(coords[axis] - center[axis]) for axis in range(ndim)
    )
    mask = (squared_distance <= 12**2).astype(np.uint16)[None]
    image = np.ones_like(mask, dtype=np.float32)

    stacked = WRFeatures.from_mask_img(mask, image).features_stacked_for("wrfeat3")

    assert stacked[0, 2] == pytest.approx(np.log(2), abs=0.02)
    np.testing.assert_allclose(stacked[0, 3:5], 0, atol=1e-6)
    if ndim == 3:
        np.testing.assert_allclose(stacked[0, 5:8], 0, atol=1e-6)


def test_wrfeat3_in_plane_shape_channels_match_2d_and_3d():
    covariance_2d = np.diag([4.0, 1.0]).astype(np.float32)
    covariance_3d = np.diag([1.0, 4.0, 1.0]).astype(np.float32)

    features_2d = _features_from_covariance(covariance_2d).features_stacked_for(
        "wrfeat3"
    )
    features_3d = _features_from_covariance(covariance_3d).features_stacked_for(
        "wrfeat3"
    )

    np.testing.assert_allclose(features_2d[:, 3:5], features_3d[:, 3:5])
    np.testing.assert_allclose(features_2d[0, 3:5], [0.6, 0])
    assert features_3d[0, 5] == pytest.approx(-3 / 7)


def test_wrfeat3_2d_masks_only_z_shape_channels():
    raw = _features_from_covariance(np.diag([4.0, 1.0]).astype(np.float32))
    stacked, mask = raw.stacked_with_mask("wrfeat3")

    assert stacked.shape == mask.shape == (1, 9)
    np.testing.assert_array_equal(
        mask[0], [True, True, True, True, True, False, False, False, True]
    )
    np.testing.assert_array_equal(stacked[0, 5:8], 0)


def test_wrfeat3_3d_exposes_all_shape_channels():
    covariance = np.array(
        [[2.0, 0.25, -0.1], [0.25, 4.0, 0.5], [-0.1, 0.5, 1.0]],
        dtype=np.float32,
    )
    raw = _features_from_covariance(covariance)
    stacked, mask = raw.stacked_with_mask("wrfeat3")

    assert stacked.shape == mask.shape == (1, 9)
    assert np.all(mask)
    assert np.all(np.isfinite(stacked))
    assert np.all(np.abs(stacked[:, 3:8]) <= 1)


def test_wrfeat2_no_intensity_removes_only_intensity_channel():
    raw = _wrfeat2_example()
    full = raw.features_stacked_for("wrfeat2")
    without_intensity = raw.features_stacked_for("wrfeat2_no_intensity")

    assert without_intensity.shape == (3, 5)
    assert np.array_equal(without_intensity, full[:, [0, 2, 3, 4, 5]])


def test_wrfeat2_is_derived_after_scale_augmentation():
    raw = _wrfeat2_example()
    augmentation = WRRandomAffine(
        p=1, degrees=0, scale=(1.5, 1.5), shear=(0, 0)
    )
    augmented = augmentation(raw)
    before = raw.features_stacked_for("wrfeat2")
    after = augmented.features_stacked_for("wrfeat2")

    assert not np.allclose(before[:, 0], after[:, 0])
    assert np.allclose(before[:, 1:], after[:, 1:], atol=1e-6)


def test_wrfeat2_3d_decomposes_inertia_and_bounds_orientation():
    features = _wrfeat2_3d_example().features_stacked_for("wrfeat2")

    assert features.shape == (3, 9)
    assert features.dtype == np.float32
    assert np.allclose(features[:, 0], np.log1p(2))
    assert np.allclose(features[:, 1], [0.1, 0.2, 0.3])
    assert np.all(np.isfinite(features))
    assert np.allclose(features[0, 3:8], 0)
    assert np.any(np.abs(features[1:, 3:8]) > 0)
    assert np.all(np.linalg.norm(features[:, 3:8], axis=1) <= 1)
    assert np.allclose(features[:, 8], np.log1p([0, 0.5, 1]))


def test_wrfeat2_3d_is_derived_after_scale_augmentation():
    raw = _wrfeat2_3d_example()
    augmentation = WRRandomAffine(
        p=1, degrees=0, scale=(1.5, 1.5), shear=(0, 0)
    )
    augmented = augmentation(raw)
    before = raw.features_stacked_for("wrfeat2")
    after = augmented.features_stacked_for("wrfeat2")

    assert after.shape == (3, 9)
    assert not np.allclose(before[:, 0], after[:, 0])
    assert np.allclose(before[:, 1:], after[:, 1:], atol=1e-6)


def test_rotation_matrix_axis_angle_matches_existing_xy_rotation():
    theta = np.pi / 2
    matrix = _rotation_matrix(theta, axis=(1, 0, 0))

    np.testing.assert_allclose(
        matrix,
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        atol=1e-7,
    )


def test_wr_random_affine_tilt_is_ignored_for_2d(monkeypatch):
    raw = _wrfeat2_example()
    draws = iter([0.0, 1.0, 0.0, 0.0])
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: next(draws))
    augmentation = WRRandomAffine(
        p=1,
        degrees=0,
        tilt_degrees=10,
        scale=(1, 1),
        shear=(0, 0),
    )

    augmented = augmentation(raw)

    np.testing.assert_allclose(augmented.coords, raw.coords)


def test_wr_random_frame_jump_shifts_exactly_one_frame(monkeypatch):
    features = WRFeatures(
        coords=np.array(
            [[0, 0], [1, 0], [0, 10], [1, 10], [0, 20], [1, 20]],
            dtype=np.float32,
        ),
        labels=np.arange(1, 7),
        timepoints=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        features={},
    )

    monkeypatch.setattr(np.random, "choice", lambda frames: 1)

    augmented = WRRandomFrameJump(offset=(2, 2), p=1)(features)
    delta = augmented.coords - features.coords

    np.testing.assert_allclose(delta[features.timepoints == 0], 0)
    np.testing.assert_allclose(delta[features.timepoints == 1], [[2, 2], [2, 2]])
    np.testing.assert_allclose(delta[features.timepoints == 2], 0)


def test_normalize_to_diameter_scales_feature_geometry_only():
    raw = _wrfeat2_example()
    raw.coords = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    scaled = wrfeat.normalize_to_diameter([raw], normalize_diameter=4)[0]

    assert scaled is not raw
    np.testing.assert_allclose(scaled.coords, raw.coords * 2)
    np.testing.assert_allclose(
        scaled.features["equivalent_diameter_area"],
        raw.features["equivalent_diameter_area"] * 2,
    )
    np.testing.assert_allclose(
        scaled.features["inertia_tensor"], raw.features["inertia_tensor"] * 4
    )
    np.testing.assert_allclose(
        scaled.features["border_dist"], raw.features["border_dist"] * 2
    )
    np.testing.assert_allclose(
        scaled.features["intensity"], raw.features["intensity"]
    )


def test_build_windows_uses_requested_wrfeat_mode():
    first = _wrfeat2_example()
    second = _wrfeat2_example()
    second.timepoints = np.ones(3, dtype=int)
    windows = build_windows(
        [first, second],
        window_size=2,
        progbar_class=lambda iterable, **_kwargs: iterable,
        feature_mode="wrfeat2",
    )

    expected = WRFeatures.concat([first, second]).features_stacked_for("wrfeat2")
    assert np.array_equal(windows[0]["features"], expected)


def test_build_windows_supports_point_only_inputs_as_torch():
    first = WRFeatures(
        coords=np.array([[0, 0], [1, 1]], dtype=np.float32),
        labels=np.array([1, 2], dtype=np.int32),
        timepoints=np.zeros(2, dtype=np.int32),
        features={},
    )
    second = WRFeatures(
        coords=np.array([[0, 1]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
        timepoints=np.ones(1, dtype=np.int32),
        features={},
    )

    windows = build_windows(
        [first, second],
        window_size=2,
        progbar_class=lambda iterable, **_kwargs: iterable,
        as_torch=True,
        feature_mode="none",
    )

    assert tuple(windows[0]["features"].shape) == (3, 0)
    assert tuple(windows[0]["coords"].shape) == (3, 2)
    assert tuple(windows[0]["labels"].tolist()) == (1, 2, 1)


def test_wrfeat2_3d_no_intensity_removes_only_intensity_channel():
    raw = _wrfeat2_3d_example()
    full = raw.features_stacked_for("wrfeat2")
    without_intensity = raw.features_stacked_for("wrfeat2_no_intensity")

    assert without_intensity.shape == (3, 8)
    assert np.array_equal(without_intensity, full[:, [0, 2, 3, 4, 5, 6, 7, 8]])


def test_stacked_with_mask_matches_features_stacked_when_complete():
    raw = _wrfeat2_example()
    stacked, mask = raw.stacked_with_mask("wrfeat2")

    assert stacked.shape == (3, 6)
    assert mask.shape == (3, 6)
    assert mask.dtype == bool
    assert np.all(mask)
    np.testing.assert_array_equal(stacked, raw.features_stacked_for("wrfeat2"))


def test_stacked_with_mask_masks_missing_intensity():
    full = _wrfeat2_example()
    without = WRFeatures(
        coords=full.coords,
        labels=full.labels,
        timepoints=full.timepoints,
        features={k: v for k, v in full.features.items() if k != "intensity"},
    )
    stacked, mask = without.stacked_with_mask("wrfeat2")

    # intensity is column 1 for wrfeat2; only it is masked and zeroed.
    assert stacked.shape == (3, 6)
    expected_mask = np.ones((3, 6), dtype=bool)
    expected_mask[:, 1] = False
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(stacked[:, 1], 0)
    # every other column is identical to the fully-featured stack.
    reference = full.features_stacked_for("wrfeat2")
    keep = [0, 2, 3, 4, 5]
    np.testing.assert_array_equal(stacked[:, keep], reference[:, keep])


def test_stacked_with_mask_masks_all_shape_when_only_coords():
    coords_only = WRFeatures(
        coords=np.zeros((3, 2), dtype=np.float32),
        labels=np.arange(3),
        timepoints=np.zeros(3, dtype=int),
        features={},
    )
    stacked, mask = coords_only.stacked_with_mask("wrfeat2")

    assert stacked.shape == (3, 6)
    assert not np.any(mask)
    np.testing.assert_array_equal(stacked, 0)


def test_stacked_with_mask_masks_dependent_columns():
    full = _wrfeat2_example()
    # diameter present, inertia absent: log1p_diameter stays available, but
    # compactness (needs both) and the inertia-derived q channels are masked.
    partial = WRFeatures(
        coords=full.coords,
        labels=full.labels,
        timepoints=full.timepoints,
        features={
            k: v for k, v in full.features.items() if k != "inertia_tensor"
        },
    )
    _stacked, mask = partial.stacked_with_mask("wrfeat2")

    # columns: 0 diam, 1 intensity, 2 compactness, 3-4 q, 5 border_dist
    expected = np.array([True, True, False, False, False, True])
    assert np.array_equal(mask[0], expected)


def test_stacked_with_mask_3d_shape_and_masking():
    full = _wrfeat2_3d_example()
    stacked, mask = full.stacked_with_mask("wrfeat2")
    assert stacked.shape == (3, 9)
    assert np.all(mask)
    np.testing.assert_array_equal(stacked, full.features_stacked_for("wrfeat2"))


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
    if properties == "regionprops2":
        assert np.allclose(
            ref.features_stacked_for("wrfeat3"),
            fast.features_stacked_for("wrfeat3"),
            atol=1e-3,
        )


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
