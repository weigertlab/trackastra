from inspect import signature
from pathlib import Path

import numpy as np
import pytest
import torch
from trackastra.data import apply_spatial_spacing, validate_spatial_spacing
from trackastra.data.dataset import (
    TrackingDataset,
    _sample_detection_keep_indices,
    _sample_neighborhood_indices,
    association_distances,
    collate_sequence_padding,
    densify_assoc,
    warn_association_distances,
)
from trackastra.data.io import DetectionSet, TrackingSequence


def test_tracking_dataset_default_window_and_features():
    params = signature(TrackingDataset).parameters

    assert params["window_size"].default == 4
    assert params["features"].default == "wrfeat2"


def test_tracking_dataset_feature_mode_error_lists_available_options():
    seg = DetectionSet(
        name="points",
        n_frames=2,
        coords=np.array([[0, 0], [1, 0]], dtype=np.float32),
        labels=np.array([1, 1]),
        timepoints=np.array([0, 1], dtype=np.int64),
        features={"intensity": np.array([[0.25], [0.75]], dtype=np.float32)},
        lineage_index=np.array([0, 0], dtype=np.int64),
    )
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        lineage_relation=np.eye(1, dtype=bool),
        lineage_parents=np.full(1, -1),
    )

    with pytest.raises(ValueError) as exc_info:
        TrackingDataset(sequence, window_size=2, features="wrfeat")

    message = str(exc_info.value)
    assert "Feature mode 'wrfeat' requires feature properties" in message
    assert "border_dist" in message
    assert "equivalent_diameter_area" in message
    assert "inertia_tensor" in message
    assert "tracking sequence only has ['intensity']" in message
    assert "Compatible feature modes: ['none', 'intensity']" in message


def test_apply_spatial_spacing_scales_model_space_distances():
    coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)

    scaled = apply_spatial_spacing(coords, (4, 1, 1))

    np.testing.assert_allclose(scaled, [[0, 0, 0], [4, 1, 1]])
    assert np.linalg.norm(scaled[1] - scaled[0]) == pytest.approx(np.sqrt(18))
    assert np.linalg.norm(coords[1] - coords[0]) == pytest.approx(np.sqrt(3))


def test_validate_spatial_spacing_defaults_and_rejects_bad_values():
    assert validate_spatial_spacing(None, 2) == (1.0, 1.0)

    with pytest.raises(ValueError, match="length 3"):
        validate_spatial_spacing((1, 1), 3)
    with pytest.raises(ValueError, match="positive"):
        validate_spatial_spacing((1, 0), 2)


def test_detection_dropout_drops_whole_lineages():
    # two 3-frame lineages; dropping a first-frame seed removes its entire lineage,
    # so the kept set is always a complete lineage (never a severed partner).
    assoc = np.zeros((6, 6), dtype=bool)
    assoc[0, 1] = assoc[1, 2] = True  # lineage A: 0(t0) -> 1(t1) -> 2(t2)
    assoc[3, 4] = assoc[4, 5] = True  # lineage B: 3(t0) -> 4(t1) -> 5(t2)
    timepoints = np.array([0, 1, 2, 0, 1, 2])

    state = np.random.get_state()
    try:
        np.random.seed(0)
        keep = _sample_detection_keep_indices(assoc, timepoints, drop_fraction=0.5)
    finally:
        np.random.set_state(state)

    # exactly one of the two first-frame lineages survives, kept whole
    assert set(keep.tolist()) in ({0, 1, 2}, {3, 4, 5})


def test_detection_dropout_always_keeps_one_lineage():
    assoc = np.zeros((6, 6), dtype=bool)
    assoc[0, 1] = assoc[1, 2] = True
    assoc[3, 4] = assoc[4, 5] = True
    timepoints = np.array([0, 1, 2, 0, 1, 2])

    state = np.random.get_state()
    try:
        np.random.seed(0)
        keep = _sample_detection_keep_indices(assoc, timepoints, drop_fraction=1.0)
    finally:
        np.random.set_state(state)

    assert set(keep.tolist()) in ({0, 1, 2}, {3, 4, 5})


def test_neighborhood_sampling_keeps_whole_lineages(monkeypatch):
    # 3 first-frame cells, budget 2: keep the 2 closest seeds' full lineages only.
    coords = np.array(
        [[0, 0], [1, 0], [100, 0], [0, 0], [1, 0], [100, 0]], dtype=np.float32
    )
    timepoints = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([1, 2, 3, 1, 2, 3])
    assoc = labels[:, None] == labels[None, :]
    monkeypatch.setattr(np.random, "randint", lambda _n: 0)  # anchor = first seed

    keep = _sample_neighborhood_indices(coords, timepoints, assoc, max_detections=2)

    assert len(keep) == 4
    assert 2 not in keep and 5 not in keep  # far lineage dropped whole
    # no association of a kept detection points outside the kept set (no severed link)
    keep_set = set(keep.tolist())
    for i in keep:
        assert set(np.flatnonzero(assoc[i]).tolist()) <= keep_set


def test_neighborhood_sampling_leaves_small_first_frame_unchanged():
    coords = np.zeros((3, 2), dtype=np.float32)
    timepoints = np.arange(3)  # one detection in the first frame
    assoc = np.eye(3, dtype=bool)

    keep = _sample_neighborhood_indices(coords, timepoints, assoc, max_detections=3)

    assert np.array_equal(keep, np.arange(3))


def test_collate_assoc_coo_densifies_to_dense_padding():
    torch.manual_seed(0)

    def sample(n):
        return {
            "coords": torch.zeros((n, 3)),
            "features": torch.zeros((n, 1)),
            "labels": torch.arange(n),
            "timepoints": torch.zeros(n, dtype=torch.long),
            "assoc_matrix": (torch.rand(n, n) < 0.25).float(),
        }

    samples = [sample(3), sample(5), sample(2)]
    batch = collate_sequence_padding(samples)

    bsz = len(samples)
    n_max = max(len(s["coords"]) for s in samples)
    dense = densify_assoc(batch["assoc_coo"], bsz, n_max)

    # densified COO must match the old dense block-padding exactly
    ref = torch.zeros(bsz, n_max, n_max)
    for i, s in enumerate(samples):
        n = s["assoc_matrix"].shape[0]
        ref[i, :n, :n] = s["assoc_matrix"]
    assert torch.equal(dense, ref)
    # the dense (B, N, N) buffer is no longer shipped
    assert "assoc_matrix" not in batch
    assert batch["assoc_coo"].shape[1] == 3


def test_dataset_and_collate_preserve_matched_gt_vector():
    seg = DetectionSet(
        name="points",
        n_frames=2,
        coords=np.array([[0, 0], [10, 0], [1, 0]], dtype=np.float32),
        labels=np.array([1, 1, 2]),
        timepoints=np.array([0, 1, 1], dtype=np.int64),
        features={"v": np.zeros((3, 1), dtype=np.float32)},
        lineage_index=np.array([0, -1, 0], dtype=np.int64),
        matched_gt=np.array([True, False, True]),
        gt_predecessor_set_available=np.array([False, False, True]),
        gt_successor_set_available=np.array([True, False, False]),
    )
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        lineage_relation=np.eye(1, dtype=bool),
        lineage_parents=np.full(1, -1),
    )
    sample = TrackingDataset(sequence, window_size=2, features="none")[0]

    assert sample["matched_gt"].tolist() == [True, False, True]
    assert sample["gt_predecessor_set_available"].tolist() == [False, False, True]
    assert sample["gt_successor_set_available"].tolist() == [True, False, False]
    assert sample["assoc_matrix"].bool().tolist() == [
        [True, False, True],
        [False, False, False],
        [True, False, True],
    ]

    short = {
        **sample,
        "coords": sample["coords"][:2],
        "coords0": sample["coords0"][:2],
        "features": sample["features"][:2],
        "labels": sample["labels"][:2],
        "timepoints": sample["timepoints"][:2],
        "matched_gt": sample["matched_gt"][:2],
        "gt_predecessor_set_available": sample["gt_predecessor_set_available"][:2],
        "gt_successor_set_available": sample["gt_successor_set_available"][:2],
        "assoc_matrix": sample["assoc_matrix"][:2, :2],
    }
    batch = collate_sequence_padding([sample, short])

    assert batch["matched_gt"].tolist() == [
        [True, False, True],
        [True, False, False],
    ]
    assert batch["gt_predecessor_set_available"].tolist() == [
        [False, False, True],
        [False, False, False],
    ]
    assert batch["gt_successor_set_available"].tolist() == [
        [True, False, False],
        [True, False, False],
    ]


def test_tracking_dataset_supports_none_and_intensity_features():
    seg = DetectionSet(
        name="points",
        n_frames=2,
        coords=np.array([[0, 0], [1, 0]], dtype=np.float32),
        labels=np.array([1, 1]),
        timepoints=np.array([0, 1], dtype=np.int64),
        features={"intensity": np.array([[0.25], [0.75]], dtype=np.float32)},
        lineage_index=np.array([0, 0], dtype=np.int64),
    )
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        lineage_relation=np.eye(1, dtype=bool),
        lineage_parents=np.full(1, -1),
    )

    none_sample = TrackingDataset(sequence, window_size=2, features="none")[0]
    intensity_sample = TrackingDataset(sequence, window_size=2, features="intensity")[0]

    assert tuple(none_sample["features"].shape) == (2, 0)
    assert none_sample["gt_predecessor_set_available"].tolist() == [True, True]
    assert none_sample["gt_successor_set_available"].tolist() == [True, True]
    np.testing.assert_allclose(
        intensity_sample["features"].numpy(),
        [[0.25], [0.75]],
    )


def test_tracking_dataset_resolves_canonical_intensity_alias():
    seg = DetectionSet(
        name="points",
        n_frames=2,
        coords=np.array([[0, 0], [1, 0]], dtype=np.float32),
        labels=np.array([1, 1]),
        timepoints=np.array([0, 1], dtype=np.int64),
        features={"intensity_mean": np.array([[0.25], [0.75]], dtype=np.float32)},
        lineage_index=np.array([0, 0], dtype=np.int64),
    )
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        lineage_relation=np.eye(1, dtype=bool),
        lineage_parents=np.full(1, -1),
    )

    sample = TrackingDataset(sequence, window_size=2, features="intensity")[0]

    assert "intensity" in seg.features
    assert "intensity_mean" not in seg.features
    np.testing.assert_allclose(sample["features"].numpy(), [[0.25], [0.75]])


def _single_lineage_sequence(xs: tuple[float, ...]) -> TrackingSequence:
    """One detection per frame, all the same tracklet, placed at the given x."""
    n_frames = len(xs)
    seg = DetectionSet(
        name="TRA",
        n_frames=n_frames,
        coords=np.array([[x, 0.0] for x in xs], dtype=np.float32),
        labels=np.array([10 + t for t in range(n_frames)]),
        timepoints=np.arange(n_frames, dtype=np.int64),
        features={
            "equivalent_diameter_area": np.full((n_frames, 1), 2, dtype=np.float32),
            "v": np.zeros((n_frames, 1), dtype=np.float32),
        },
        lineage_index=np.zeros(n_frames, dtype=np.int64),
    )
    return TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        lineage_relation=np.eye(1, dtype=bool),
        lineage_parents=np.full(1, -1),
    )


def test_association_distances_deduplicate_overlapping_windows():
    # 4 frames, window_size 3 -> windows [0,1,2] and [1,2,3] share the t1->t2 edge,
    # which must be counted once.
    sequence = _single_lineage_sequence((0.0, 3.0, 13.0, 30.0))
    data = TrackingDataset(sequence, window_size=3, features="none")

    distances = association_distances(data, delta_cutoff=1)

    assert sorted(distances.tolist()) == [3.0, 10.0, 17.0]


def test_association_distances_use_runtime_model_units():
    sequence = _single_lineage_sequence((0.0, 3.0, 13.0, 30.0))
    data = TrackingDataset(
        sequence,
        window_size=3,
        features="none",
        normalize_diameter=4,
    )

    distances = association_distances(data, delta_cutoff=1)

    assert sorted(distances.tolist()) == [6.0, 20.0, 34.0]


def test_association_distance_warning_ignores_values_at_cutoff(caplog):
    warn_association_distances(
        np.array([3.0, 10.0]),
        max_distance=10,
        delta_cutoff=1,
        cutoff_name="max_distance",
        dataset_name="test dataset",
    )
    assert caplog.text == ""


def test_association_distance_warning_reports_exceedances(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        warn_association_distances(
            np.array([3.0, 10.0]),
            max_distance=5,
            delta_cutoff=1,
            cutoff_name="max_distance",
            dataset_name="test dataset",
        )
    assert "1/2 (50.00%)" in caplog.text
    assert "max=10.00" in caplog.text
    assert "cannot be recovered" in caplog.text
