from inspect import signature
from pathlib import Path

import numpy as np
import pytest
import torch
from trackastra.data import apply_spatial_spacing, validate_spatial_spacing
from trackastra.data.dataset import (
    AugmentationConfig,
    TrackingDataset,
    _sample_detection_keep_indices,
    _sample_neighborhood_indices,
    association_distances,
    collate_sequence_padding,
    densify_assoc,
    warn_association_distances,
)
from trackastra.data.io import (
    DetectionSequence,
    DetectionSupervision,
    LineageGraph,
    TrackingSequence,
)


def _lineage_graph(relation: np.ndarray, parents: np.ndarray) -> LineageGraph:
    return LineageGraph(
        coords=np.zeros((0, 2), dtype=np.float32),
        timepoints=np.zeros(0, dtype=np.int64),
        node_ids=np.zeros(0, dtype=object),
        lineage_relation=relation,
        lineage_parents=parents,
    )


def _sequence(
    seg: DetectionSequence, lineage_index: np.ndarray, relation: np.ndarray
) -> TrackingSequence:
    return TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        gt=_lineage_graph(relation, np.full(len(relation), -1)),
        supervision=(DetectionSupervision(lineage_index=lineage_index),),
    )


def test_tracking_dataset_default_window_and_features():
    params = signature(TrackingDataset).parameters

    assert params["window_size"].default == 4
    assert params["features"].default == "wrfeat2"


def test_tracking_dataset_counts_max_detections_per_frame():
    timepoints = np.array([0, 0, 0, 0, 1, 3, 3, 3], dtype=np.int64)
    seg = DetectionSequence(
        name="points",
        n_frames=4,
        coords=np.zeros((len(timepoints), 2), dtype=np.float32),
        labels=np.arange(1, len(timepoints) + 1),
        timepoints=timepoints,
        features={},
    )
    relation = np.eye(len(timepoints), dtype=bool)
    sequence = _sequence(seg, np.arange(len(timepoints)), relation)

    uncapped = TrackingDataset(sequence, window_size=4, features="none")
    capped = TrackingDataset(
        sequence,
        window_size=4,
        features="none",
        max_detections=2,
    )

    assert uncapped.n_objects == (8,)
    assert capped.n_objects == (5,)  # min(4, 2) + min(1, 2) + 0 + min(3, 2)


def test_tracking_dataset_augment_details_override_jitter_and_drift():
    lineage_index = np.array([0, 0], dtype=np.int64)
    seg = DetectionSequence(
        name="points",
        n_frames=2,
        coords=np.array([[0, 0], [1, 0]], dtype=np.float32),
        labels=np.array([1, 1]),
        timepoints=np.array([0, 1], dtype=np.int64),
        features={},
    )
    sequence = _sequence(seg, lineage_index, np.eye(1, dtype=bool))

    dataset = TrackingDataset(
        sequence,
        window_size=2,
        features="none",
        augment=3,
        augment_details={"jitter": 2.5, "drift": 7.0, "tilt": 5.0},
    )

    assert dataset.augment_config == AugmentationConfig(
        preset=3,
        jitter=2.5,
        drift=7.0,
        tilt=5.0,
    )


def test_tracking_dataset_augment_details_reject_unknown_keys():
    lineage_index = np.array([0, 0], dtype=np.int64)
    seg = DetectionSequence(
        name="points",
        n_frames=2,
        coords=np.array([[0, 0], [1, 0]], dtype=np.float32),
        labels=np.array([1, 1]),
        timepoints=np.array([0, 1], dtype=np.int64),
        features={},
    )
    sequence = _sequence(seg, lineage_index, np.eye(1, dtype=bool))

    with pytest.raises(ValueError, match="Unknown augment_details keys"):
        TrackingDataset(
            sequence,
            window_size=2,
            features="none",
            augment=3,
            augment_details={"rotate": 1.0},
        )


def test_tracking_dataset_masks_missing_feature_properties():
    # A sequence with only intensity is no longer rejected for a richer recipe:
    # the absent shape columns are masked.
    lineage_index = np.array([0, 0], dtype=np.int64)
    seg = DetectionSequence(
        name="points",
        n_frames=2,
        coords=np.array([[0, 0], [1, 0]], dtype=np.float32),
        labels=np.array([1, 1]),
        timepoints=np.array([0, 1], dtype=np.int64),
        features={"intensity": np.array([[0.25], [0.75]], dtype=np.float32)},
    )
    sequence = _sequence(seg, lineage_index, np.eye(1, dtype=bool))

    dataset = TrackingDataset(sequence, window_size=2, features="wrfeat2")
    sample = dataset[0]

    assert tuple(sample["features"].shape) == (2, 6)
    mask = sample["feature_mask"]
    assert mask.dtype == torch.bool
    expected = torch.tensor([False, True, False, False, False, False])
    assert torch.equal(mask[0], expected)
    # masked columns are zero-filled; the present intensity column carries values.
    assert torch.all(sample["features"][:, mask[0] == 0] == 0)
    assert torch.allclose(sample["features"][:, 1], torch.tensor([0.25, 0.75]))


def _wrfeat2_seg(name, with_shape=True, with_intensity=True):
    features = {}
    if with_shape:
        features["equivalent_diameter_area"] = np.full((2, 1), 2, dtype=np.float32)
        features["inertia_tensor"] = np.array(
            [[2, 0, 0, 2], [3, 0, 0, 1]], dtype=np.float32
        )
        features["border_dist"] = np.array([[0.0], [0.5]], dtype=np.float32)
    if with_intensity:
        features["intensity"] = np.array([[0.25], [0.75]], dtype=np.float32)
    return DetectionSequence(
        name=name,
        n_frames=2,
        coords=np.array([[0, 0], [1, 0]], dtype=np.float32),
        labels=np.array([1, 1]),
        timepoints=np.array([0, 1], dtype=np.int64),
        features=features,
    )


def test_mixed_feature_availability_batch_runs_through_model():
    from trackastra.model import TrackingTransformer

    lineage_index = np.array([0, 0], dtype=np.int64)
    relation = np.eye(1, dtype=bool)
    full = _sequence(_wrfeat2_seg("full"), lineage_index, relation)
    intensity_only = _sequence(
        _wrfeat2_seg("intensity_only", with_shape=False), lineage_index, relation
    )

    full_sample = TrackingDataset(full, window_size=2, features="wrfeat2")[0]
    partial_sample = TrackingDataset(intensity_only, window_size=2, features="wrfeat2")[
        0
    ]

    # both stacks share the fixed wrfeat2 width; only the masks differ.
    assert full_sample["features"].shape[1] == 6
    assert partial_sample["features"].shape[1] == 6
    assert bool(full_sample["feature_mask"].all())
    # intensity is column 1; the shape columns are masked for the intensity-only seq.
    assert bool(partial_sample["feature_mask"][:, 1].all())
    assert not bool(partial_sample["feature_mask"][:, [0, 2, 3, 4, 5]].any())

    batch = collate_sequence_padding([full_sample, partial_sample])

    model = TrackingTransformer(
        coord_dim=2,
        feat_dim=6,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    model.eval()
    with torch.no_grad():
        A, _ = model(
            batch["coords"].float(),
            features=batch["features"].float(),
            feature_mask=batch["feature_mask"],
            padding_mask=batch["padding_mask"],
        )
    assert A.shape == (2, 2, 2)
    assert torch.isfinite(A).all()


def test_feature_group_drop_masks_intensity_every_window():
    lineage_index = np.array([0, 0], dtype=np.int64)
    sequence = _sequence(_wrfeat2_seg("full"), lineage_index, np.eye(1, dtype=bool))

    dataset = TrackingDataset(
        sequence,
        window_size=2,
        features="wrfeat2",
        feature_group_drop={"intensity": 1.0},
    )
    sample = dataset[0]

    # intensity is wrfeat2 column 1; it is always dropped, the shape columns remain.
    mask = sample["feature_mask"]
    assert not bool(mask[:, 1].any())
    assert torch.all(sample["features"][:, 1] == 0)
    assert bool(mask[:, [0, 2, 3, 4, 5]].all())


def test_wrfeat3_dataset_keeps_fixed_width_and_2d_z_mask():
    lineage_index = np.array([0, 0], dtype=np.int64)
    sequence = _sequence(_wrfeat2_seg("full"), lineage_index, np.eye(1, dtype=bool))

    sample = TrackingDataset(sequence, window_size=2, features="wrfeat3")[0]

    assert tuple(sample["features"].shape) == (2, 9)
    expected = torch.tensor([True, True, True, True, True, False, False, False, True])
    assert torch.equal(sample["feature_mask"][0], expected)
    assert torch.all(sample["features"][:, 5:8] == 0)


def test_mixed_2d_3d_wrfeat3_batch_runs_through_3d_model():
    lineage_index = np.array([0, 0], dtype=np.int64)
    relation = np.eye(1, dtype=bool)
    sequence_2d = _sequence(_wrfeat2_seg("2d"), lineage_index, relation)

    seg_3d = DetectionSequence(
        name="3d",
        n_frames=2,
        coords=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        labels=np.array([1, 1]),
        timepoints=np.array([0, 1], dtype=np.int64),
        features={
            "equivalent_diameter_area": np.full((2, 1), 2, dtype=np.float32),
            "intensity": np.array([[0.25], [0.75]], dtype=np.float32),
            "inertia_tensor": np.tile(np.eye(3, dtype=np.float32).ravel(), (2, 1)),
            "border_dist": np.zeros((2, 1), dtype=np.float32),
        },
    )
    sequence_3d = TrackingSequence(
        root=Path("synthetic_3d"),
        ndim=3,
        detections=(seg_3d,),
        gt=LineageGraph(
            coords=np.zeros((0, 3), dtype=np.float32),
            timepoints=np.zeros(0, dtype=np.int64),
            node_ids=np.zeros(0, dtype=object),
            lineage_relation=relation,
            lineage_parents=np.array([-1], dtype=np.int64),
        ),
        supervision=(DetectionSupervision(lineage_index=lineage_index),),
    )

    sample_2d = TrackingDataset(
        sequence_2d, window_size=2, features="wrfeat3", model_coord_dim=3
    )[0]
    sample_3d = TrackingDataset(
        sequence_3d, window_size=2, features="wrfeat3", model_coord_dim=3
    )[0]
    batch = collate_sequence_padding([sample_2d, sample_3d])

    assert batch["coords"].shape == (2, 2, 4)
    assert torch.all(batch["coords"][0, :, 1] == 0)
    assert not bool(batch["feature_mask"][0, :, 5:8].any())
    assert bool(batch["feature_mask"][1].all())

    from trackastra.model import TrackingTransformer

    model = TrackingTransformer(
        coord_dim=3,
        feat_dim=9,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
    ).eval()
    with torch.no_grad():
        association, _ = model(
            batch["coords"].float(),
            features=batch["features"].float(),
            feature_mask=batch["feature_mask"],
            padding_mask=batch["padding_mask"],
        )
    assert association.shape == (2, 2, 2)
    assert torch.isfinite(association).all()


def test_2d_position_noise_precedes_coordinate_lifting():
    sequence = _sequence(_wrfeat2_seg("2d"), np.array([0, 0]), np.eye(1, dtype=bool))
    sample = TrackingDataset(
        sequence,
        window_size=2,
        features="wrfeat3",
        augment=1,
        position_noise=2.0,
        model_coord_dim=3,
    )[0]

    assert torch.all(sample["coords0"][:, 1] == 0)
    assert torch.all(sample["coords"][:, 1] == 0)


def test_tracking_dataset_rejects_3d_source_for_2d_model():
    seg = DetectionSequence(
        name="3d",
        n_frames=2,
        coords=np.zeros((2, 3), dtype=np.float32),
        labels=np.ones(2, dtype=np.int64),
        timepoints=np.arange(2, dtype=np.int64),
        features={},
    )
    sequence = TrackingSequence(
        root=Path("synthetic_3d"),
        ndim=3,
        detections=(seg,),
        gt=LineageGraph(
            coords=np.zeros((0, 3), dtype=np.float32),
            timepoints=np.zeros(0, dtype=np.int64),
            node_ids=np.zeros(0, dtype=object),
            lineage_relation=np.eye(1, dtype=bool),
            lineage_parents=np.array([-1], dtype=np.int64),
        ),
        supervision=(DetectionSupervision(lineage_index=np.array([0, 0])),),
    )

    with pytest.raises(ValueError, match="Cannot lift 3D source coordinates to 2D"):
        TrackingDataset(sequence, window_size=2, model_coord_dim=2)


def test_feature_group_drop_validates_probability():
    lineage_index = np.array([0, 0], dtype=np.int64)
    sequence = _sequence(_wrfeat2_seg("full"), lineage_index, np.eye(1, dtype=bool))

    with pytest.raises(ValueError, match="must be in"):
        TrackingDataset(
            sequence,
            window_size=2,
            features="wrfeat2",
            feature_group_drop={"intensity": 1.5},
        )


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


def test_detection_sequence_warns_on_implicit_3d_unit_spacing(caplog):
    with caplog.at_level("WARNING", logger="trackastra.data.io"):
        detections = DetectionSequence.from_points(
            [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]
        )

    assert "3D detections without spacing" in caplog.text
    assert detections.spacing == (1.0, 1.0, 1.0)

    caplog.clear()
    with caplog.at_level("WARNING", logger="trackastra.data.io"):
        DetectionSequence.from_points(
            [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)],
            spacing=(1, 1, 1),
        )
    assert caplog.text == ""


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


def _no_severed_links(keep, assoc):
    keep_set = set(keep.tolist())
    return all(set(np.flatnonzero(assoc[i]).tolist()) <= keep_set for i in keep)


def test_neighborhood_sampling_keeps_whole_lineages(monkeypatch):
    # 3 lineages, budget 2: the crop around the seed keeps the 2 nearest per frame,
    # then lineage closure keeps those whole and drops the far lineage entirely.
    coords = np.array(
        [[0, 0], [1, 0], [100, 0], [0, 0], [1, 0], [100, 0]], dtype=np.float32
    )
    timepoints = np.array([0, 0, 0, 1, 1, 1])
    labels = np.array([1, 2, 3, 1, 2, 3])
    assoc = labels[:, None] == labels[None, :]
    monkeypatch.setattr(np.random, "randint", lambda _n: 0)  # seed = detection 0

    keep = _sample_neighborhood_indices(coords, timepoints, assoc, max_detections=2)

    assert len(keep) == 4
    assert 2 not in keep and 5 not in keep  # far lineage dropped whole
    assert _no_severed_links(keep, assoc)


def test_neighborhood_sampling_leaves_small_first_frame_unchanged():
    coords = np.zeros((3, 2), dtype=np.float32)
    timepoints = np.arange(3)  # one detection per frame, all within budget
    assoc = np.eye(3, dtype=bool)

    keep = _sample_neighborhood_indices(coords, timepoints, assoc, max_detections=3)

    assert np.array_equal(keep, np.arange(3))


def test_neighborhood_sampling_keeps_both_division_daughters(monkeypatch):
    # Parent (frame 0) divides into two daughters (frame 1); budget 1 per frame.
    # The crop can only catch one daughter, but lineage closure restores the other.
    coords = np.array([[0, 0], [10, 0], [0, 1], [0, 2], [0, 1.5]], dtype=np.float32)
    timepoints = np.array([0, 0, 1, 1, 1])
    matched_gt = np.array([True, False, True, True, False])  # 1 and 4 are junk
    assoc = np.zeros((5, 5), dtype=bool)
    for i, j in [(0, 2), (0, 3)]:  # parent -> both daughters
        assoc[i, j] = assoc[j, i] = True
    monkeypatch.setattr(np.random, "randint", lambda _n: 0)  # seed = parent (pool[0])

    keep = _sample_neighborhood_indices(
        coords, timepoints, assoc, max_detections=1, matched_gt=matched_gt
    )

    assert set(keep.tolist()) == {0, 2, 3}  # both daughters kept, junk dropped
    assert _no_severed_links(keep, assoc)


def test_neighborhood_sampling_completes_a_distractor_lineage(monkeypatch):
    # Seed lineage A; a node of lineage B enters the crop only as a distractor.
    # Closure must pull in the rest of B so no partial lineage survives.
    coords = np.array([[0, 0], [1, 0], [0, 0], [5, 0], [0.1, 0]], dtype=np.float32)
    timepoints = np.array([0, 0, 1, 1, 1])
    matched_gt = np.array([True, True, True, True, False])  # 4 is junk
    assoc = np.zeros((5, 5), dtype=bool)
    for i, j in [(0, 2), (1, 3)]:  # A: 0-2, B: 1-3
        assoc[i, j] = assoc[j, i] = True
    monkeypatch.setattr(np.random, "randint", lambda _n: 0)  # seed = A (pool[0] == 0)

    keep = _sample_neighborhood_indices(
        coords, timepoints, assoc, max_detections=2, matched_gt=matched_gt
    )

    # frame 0 (<=budget) keeps both A0 and the distractor B0; closure restores B1.
    assert {0, 2}.issubset(keep) and {1, 3}.issubset(keep)
    assert _no_severed_links(keep, assoc)


def test_neighborhood_sampling_seeds_only_from_gt():
    # A single GT detection amid junk must always survive: it is the only seed,
    # so it is nearest to itself and always kept.
    coords = np.array([[0, 0], [50, 0], [0, 0], [50, 0]], dtype=np.float32)
    timepoints = np.array([0, 0, 1, 1])
    matched_gt = np.array([False, True, False, False])
    assoc = np.eye(4, dtype=bool)

    for _ in range(10):
        keep = _sample_neighborhood_indices(
            coords, timepoints, assoc, max_detections=1, matched_gt=matched_gt
        )
        assert 1 in keep


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
    seg = DetectionSequence(
        name="points",
        n_frames=2,
        coords=np.array([[0, 0], [10, 0], [1, 0]], dtype=np.float32),
        labels=np.array([1, 1, 2]),
        timepoints=np.array([0, 1, 1], dtype=np.int64),
        features={"v": np.zeros((3, 1), dtype=np.float32)},
    )
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        gt=_lineage_graph(np.eye(1, dtype=bool), np.full(1, -1)),
        supervision=(
            DetectionSupervision(
                lineage_index=np.array([0, -1, 0], dtype=np.int64),
                matched_gt=np.array([True, False, True]),
                gt_predecessor_set_available=np.array([False, False, True]),
                gt_successor_set_available=np.array([True, False, False]),
            ),
        ),
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


def test_tracking_dataset_counts_node_degrees_per_detection_stream():
    seg = DetectionSequence(
        name="points",
        n_frames=4,
        coords=np.array([[0, 0], [1, 0], [2, 0], [3, 0], [30, 0]], dtype=np.float32),
        labels=np.array([1, 1, 1, 1, 2]),
        timepoints=np.array([0, 1, 2, 3, 3], dtype=np.int64),
        features={},
    )
    gt = LineageGraph(
        coords=np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32),
        timepoints=np.array([0, 1, 2, 3], dtype=np.int64),
        node_ids=np.arange(4),
        lineage_relation=np.eye(1, dtype=bool),
        lineage_parents=np.array([-1], dtype=np.int64),
        node_in_degree=np.array([0, 1, 2, 1], dtype=np.int64),
        node_out_degree=np.array([1, 0, 3, 2], dtype=np.int64),
    )
    supervision = DetectionSupervision(
        lineage_index=np.array([0, 0, 0, 0, -1], dtype=np.int64),
        gt_node_index=np.array([0, 1, 2, 3, -1], dtype=np.int64),
        gt_predecessor_set_available=np.array([False, True, True, False, False]),
        gt_successor_set_available=np.array([True, False, True, True, False]),
    )
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg, seg),
        gt=gt,
        supervision=(supervision, supervision),
    )

    dataset = TrackingDataset(sequence, window_size=2, features="none")

    assert dataset.node_in_degree_counts.tolist() == [0, 2, 2]
    assert dataset.node_out_degree_counts.tolist() == [0, 2, 2, 2]


def test_tracking_dataset_emits_per_window_node_degrees():
    seg = DetectionSequence(
        name="points",
        n_frames=4,
        coords=np.array([[0, 0], [1, 0], [2, 0], [3, 0], [30, 0]], dtype=np.float32),
        labels=np.array([1, 1, 1, 1, 2]),
        timepoints=np.array([0, 1, 2, 3, 3], dtype=np.int64),
        features={},
    )
    gt = LineageGraph(
        coords=np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32),
        timepoints=np.array([0, 1, 2, 3], dtype=np.int64),
        node_ids=np.arange(4),
        lineage_relation=np.eye(1, dtype=bool),
        lineage_parents=np.array([-1], dtype=np.int64),
        node_in_degree=np.array([0, 1, 1, 1], dtype=np.int64),
        node_out_degree=np.array([1, 1, 1, 0], dtype=np.int64),
    )
    supervision = DetectionSupervision(
        lineage_index=np.array([0, 0, 0, 0, -1], dtype=np.int64),
        gt_node_index=np.array([0, 1, 2, 3, -1], dtype=np.int64),
    )
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        gt=gt,
        supervision=(supervision,),
    )
    dataset = TrackingDataset(sequence, window_size=2, features="none")

    # window starting at t=2 covers detections at t in {2, 3}: the two matched nodes
    # (gt degrees) plus the unmatched t=3 detection, which must read -1.
    index = next(i for i, w in enumerate(dataset.windows) if w == (0, 2))
    sample = dataset.__getitem__(index)
    assert sample["node_out_degree"].tolist() == [1, 0, -1]
    assert sample["node_in_degree"].tolist() == [1, 1, -1]
    lineage_index = np.array([0, 0], dtype=np.int64)
    seg = DetectionSequence(
        name="points",
        n_frames=2,
        coords=np.array([[0, 0], [1, 0]], dtype=np.float32),
        labels=np.array([1, 1]),
        timepoints=np.array([0, 1], dtype=np.int64),
        features={"intensity": np.array([[0.25], [0.75]], dtype=np.float32)},
    )
    sequence = _sequence(seg, lineage_index, np.eye(1, dtype=bool))

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
    lineage_index = np.array([0, 0], dtype=np.int64)
    seg = DetectionSequence(
        name="points",
        n_frames=2,
        coords=np.array([[0, 0], [1, 0]], dtype=np.float32),
        labels=np.array([1, 1]),
        timepoints=np.array([0, 1], dtype=np.int64),
        features={"intensity_mean": np.array([[0.25], [0.75]], dtype=np.float32)},
    )
    sequence = _sequence(seg, lineage_index, np.eye(1, dtype=bool))

    sample = TrackingDataset(sequence, window_size=2, features="intensity")[0]

    assert "intensity" in seg.features
    assert "intensity_mean" not in seg.features
    np.testing.assert_allclose(sample["features"].numpy(), [[0.25], [0.75]])


def _single_lineage_sequence(xs: tuple[float, ...]) -> TrackingSequence:
    """One detection per frame, all the same tracklet, placed at the given x."""
    n_frames = len(xs)
    seg = DetectionSequence(
        name="TRA",
        n_frames=n_frames,
        coords=np.array([[x, 0.0] for x in xs], dtype=np.float32),
        labels=np.array([10 + t for t in range(n_frames)]),
        timepoints=np.arange(n_frames, dtype=np.int64),
        features={
            "equivalent_diameter_area": np.full((n_frames, 1), 2, dtype=np.float32),
            "v": np.zeros((n_frames, 1), dtype=np.float32),
        },
    )
    return _sequence(seg, np.zeros(n_frames, dtype=np.int64), np.eye(1, dtype=bool))


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
        spatial_cutoff=10,
        delta_cutoff=1,
        cutoff_name="spatial_cutoff",
        dataset_name="test dataset",
    )
    assert caplog.text == ""


def test_association_distance_warning_reports_exceedances(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        warn_association_distances(
            np.array([3.0, 10.0]),
            spatial_cutoff=5,
            delta_cutoff=1,
            cutoff_name="spatial_cutoff",
            dataset_name="test dataset",
        )
    assert "1/2 (50.00%)" in caplog.text
    assert "max=10.00" in caplog.text
    assert "cannot be recovered" in caplog.text
