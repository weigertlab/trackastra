import pickle
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from tifffile import imwrite
from trackastra.data.datanew import (
    DetectionFrame,
    DetectionSeries,
    TrackingData,
    TrackingSequence,
)


def _frame(timepoint=0, track_indices=(0, 1)):
    n = len(track_indices)
    return DetectionFrame(
        timepoint=timepoint,
        coords=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        labels=np.arange(1, n + 1, dtype=np.int32),
        features={
            "equivalent_diameter_area": np.full((n, 1), 2, np.float32),
            "intensity_mean": np.full((n, 1), 0.5, np.float32),
            "inertia_tensor": np.tile(np.eye(2, dtype=np.float32).ravel(), (n, 1)),
            "border_dist": np.zeros((n, 1), np.float32),
        },
        track_indices=np.asarray(track_indices),
    )


def test_detection_frame_is_immutable_and_validates_alignment():
    frame = _frame()

    with pytest.raises(ValueError):
        frame.coords[0, 0] = 10
    for values in frame.features.values():
        with pytest.raises(ValueError):
            values[0] = 10
    with pytest.raises(ValueError, match="aligned"):
        DetectionFrame(
            0,
            np.zeros((2, 2)),
            np.ones(1),
            {},
            np.zeros(2),
        )


def test_tracking_sequence_pickle_roundtrip_remains_immutable():
    frame = _frame()
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detection_series=(DetectionSeries("TRA", (frame,)),),
        lineage_relation=np.eye(2, dtype=bool),
        lineage_parents=np.full(2, -1),
    )

    restored = pickle.loads(pickle.dumps(sequence))

    with pytest.raises(ValueError):
        restored.detection_series[0].frames[0].coords[0, 0] = 10
    for values in restored.detection_series[0].frames[0].features.values():
        with pytest.raises(ValueError):
            values[0] = 10


def test_tracking_data_uses_lineage_relation_and_excludes_unmatched():
    frames = (_frame(0, (0, -1)), _frame(1, (1, 2)))
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detection_series=(DetectionSeries("TRA", frames),),
        lineage_relation=np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=bool),
        lineage_parents=np.array([-1, 0, 0]),
    )

    for mode, width in (("wrfeat", 7), ("wrfeat2", 6), ("wrfeat2_no_intensity", 5)):
        data = TrackingData(sequence, window_size=2, features=mode)
        sample = data[0]
        association = sample["assoc_matrix"].bool().numpy()

        assert len(data) == 1
        assert data.n_objects == (4,)
        assert sample["features"].shape == (4, width)
        assert association[0, 2] and association[0, 3]
        assert not association[2, 3]
        assert not association[1].any()
        assert not association[:, 1].any()


def _write_ctc_fixture(base: Path, layout: str) -> Path:
    if layout == "standard":
        root = base / "Fluo-C2DL-Huh7" / "01"
        images = root
        tra = base / "Fluo-C2DL-Huh7" / "01_GT" / "TRA"
    else:
        root = base / "151031-03"
        images = root / "img"
        tra = root / "TRA"
    images.mkdir(parents=True)
    tra.mkdir(parents=True)
    masks = []
    for t in range(3):
        image = np.arange(64, dtype=np.uint8).reshape(8, 8) + t
        mask = np.zeros((8, 8), dtype=np.uint16)
        if t == 0:
            mask[2:4, 2:4] = 1
        else:
            mask[1:3, 1:3] = 2
            mask[5:7, 5:7] = 3
        imwrite(images / f"t{t:03d}.tif", image)
        imwrite(tra / f"man_track{t:03d}.tif", mask)
        masks.append(mask)
    (tra / "man_track.txt").write_text("1 0 0 0\n2 1 2 1\n3 1 2 1\n")
    return root


@pytest.mark.parametrize("layout", ("standard", "simple"))
def test_tracking_sequence_from_ctc_supports_reference_layouts(tmp_path, layout):
    root = _write_ctc_fixture(tmp_path, layout)

    sequence = TrackingSequence.from_ctc(root, n_workers=1)
    data = TrackingData(sequence, window_size=2)
    sample = data[0]

    assert sequence.root == root
    assert len(sequence.detection_series) == 1
    assert tuple(len(frame) for frame in sequence.detection_series[0].frames) == (
        1,
        2,
        2,
    )
    assert np.array_equal(
        sequence.lineage_relation,
        np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=bool),
    )
    assert sample["assoc_matrix"][0, 1]
    assert sample["assoc_matrix"][0, 2]
    assert not sample["assoc_matrix"][1, 2]


def test_tracking_sequence_without_gt_uses_isolated_detections(tmp_path):
    root = tmp_path / "detections-only"
    images = root / "img"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    for t in range(2):
        image = np.arange(64, dtype=np.uint8).reshape(8, 8) + t
        mask = np.zeros((8, 8), dtype=np.uint16)
        mask[2:4, 2:4] = 1
        imwrite(images / f"t{t:03d}.tif", image)
        imwrite(masks / f"mask{t:03d}.tif", mask)

    sequence = TrackingSequence.from_ctc(
        root, use_gt=False, detection_folders=("masks",), n_workers=1
    )
    sample = TrackingData(sequence, window_size=2)[0]

    assert sequence.lineage_relation.shape == (2, 2)
    assert np.array_equal(sequence.lineage_relation, np.eye(2, dtype=bool))
    assert torch.equal(sample["assoc_matrix"], torch.eye(2))


def test_tracking_sequence_joblib_cache_ignores_workers(tmp_path):
    root = _write_ctc_fixture(tmp_path / "data", "simple")
    memory = joblib.Memory(tmp_path / "cache", verbose=0)
    loader = memory.cache(TrackingSequence.from_ctc, ignore=["n_workers"])

    sequence = loader(root=root, n_workers=1)

    assert loader.check_call_in_cache(root=root, n_workers=4)
    cached = loader(root=root, n_workers=4)
    assert not cached.lineage_relation.flags.writeable
    assert not cached.detection_series[0].frames[0].coords.flags.writeable
    assert np.array_equal(sequence.lineage_relation, cached.lineage_relation)


def test_tracking_data_runtime_selection_keeps_complete_lineages():
    frames = (_frame(0, (0, 1, 2)), _frame(1, (0, 1, 2)))
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detection_series=(DetectionSeries("TRA", frames),),
        lineage_relation=np.eye(3, dtype=bool),
        lineage_parents=np.full(3, -1),
    )

    np.random.seed(0)
    capped = TrackingData(sequence, window_size=2, max_detections=2)[0]
    np.random.seed(0)
    dropped = TrackingData(
        sequence,
        window_size=2,
        detect_drop=1,
        detect_drop_fraction=0.34,
    )[0]

    assert len(capped["coords"]) == 4
    assert len(dropped["coords"]) == 4
    for sample in (capped, dropped):
        labels, counts = torch.unique(sample["labels"], return_counts=True)
        assert len(labels) == 2
        assert torch.equal(counts, torch.full((2,), 2))


@pytest.mark.parametrize("level", (1, 2, 3, 4))
def test_tracking_data_augmentation_does_not_mutate_sequence(level):
    frames = (_frame(0), _frame(1))
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detection_series=(DetectionSeries("TRA", frames),),
        lineage_relation=np.eye(2, dtype=bool),
        lineage_parents=np.full(2, -1),
    )
    original = frames[0].features["inertia_tensor"].copy()

    np.random.seed(4)
    torch.manual_seed(4)
    sample = TrackingData(sequence, window_size=2, augment=level)[0]

    assert sample["features"].shape == (4, 7)
    np.testing.assert_array_equal(frames[0].features["inertia_tensor"], original)
