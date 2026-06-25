import pickle
from pathlib import Path

import joblib
import numpy as np
import pytest
import torch
from tifffile import imwrite
from trackastra.data.dataset import TrackingDataset
from trackastra.data.io import Segmentation, TrackingSequence, load_ctc_images_masks
from trackastra.utils import normalize


def _segmentation(frames, name="TRA"):
    """Build a flat Segmentation from ``(timepoint, track_indices)`` frame specs."""
    coords, labels, timepoints, track_indices = [], [], [], []
    feats = {
        "equivalent_diameter_area": [],
        "intensity_mean": [],
        "inertia_tensor": [],
        "border_dist": [],
    }
    for t, ti in frames:
        n = len(ti)
        coords.append(np.arange(n * 2, dtype=np.float32).reshape(n, 2))
        labels.append(np.arange(1, n + 1, dtype=np.int32))
        timepoints.append(np.full(n, t, dtype=np.int64))
        track_indices.append(np.asarray(ti, dtype=np.int64))
        feats["equivalent_diameter_area"].append(np.full((n, 1), 2, np.float32))
        feats["intensity_mean"].append(np.full((n, 1), 0.5, np.float32))
        feats["inertia_tensor"].append(
            np.tile(np.eye(2, dtype=np.float32).ravel(), (n, 1))
        )
        feats["border_dist"].append(np.zeros((n, 1), np.float32))
    return Segmentation(
        name=name,
        n_frames=max(t for t, _ in frames) + 1,
        coords=np.concatenate(coords),
        labels=np.concatenate(labels),
        timepoints=np.concatenate(timepoints),
        features={k: np.concatenate(v) for k, v in feats.items()},
        track_indices=np.concatenate(track_indices),
    )


def test_segmentation_is_immutable_and_validates_alignment():
    seg = _segmentation([(0, (0, 1))])

    with pytest.raises(ValueError):
        seg.coords[0, 0] = 10
    for values in seg.features.values():
        with pytest.raises(ValueError):
            values[0] = 10
    with pytest.raises(ValueError, match="aligned"):
        Segmentation(
            name="TRA",
            n_frames=1,
            coords=np.zeros((2, 2)),
            labels=np.ones(1),
            timepoints=np.zeros(2),
            features={},
            track_indices=np.zeros(2),
        )


def test_tracking_sequence_pickle_roundtrip_remains_immutable():
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        segmentations=(_segmentation([(0, (0, 1))]),),
        lineage_relation=np.eye(2, dtype=bool),
        lineage_parents=np.full(2, -1),
    )

    restored = pickle.loads(pickle.dumps(sequence))

    with pytest.raises(ValueError):
        restored.segmentations[0].coords[0, 0] = 10
    for values in restored.segmentations[0].features.values():
        with pytest.raises(ValueError):
            values[0] = 10


def test_tracking_data_uses_lineage_relation_and_excludes_unmatched():
    seg = _segmentation([(0, (0, -1)), (1, (1, 2))])
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        segmentations=(seg,),
        lineage_relation=np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=bool),
        lineage_parents=np.array([-1, 0, 0]),
    )

    for mode, width in (("wrfeat", 7), ("wrfeat2", 6), ("wrfeat2_no_intensity", 5)):
        data = TrackingDataset(sequence, window_size=2, features=mode)
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
    data = TrackingDataset(sequence, window_size=2)
    sample = data[0]

    assert sequence.root == root
    assert len(sequence.segmentations) == 1
    seg = sequence.segmentations[0]
    assert seg.n_frames == 3
    counts = tuple(int((seg.timepoints == t).sum()) for t in range(seg.n_frames))
    assert counts == (1, 2, 2)
    assert np.array_equal(
        sequence.lineage_relation,
        np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=bool),
    )
    assert sample["assoc_matrix"][0, 1]
    assert sample["assoc_matrix"][0, 2]
    assert not sample["assoc_matrix"][1, 2]


def test_load_ctc_images_masks_refines_tra_with_st(tmp_path):
    sequence = tmp_path / "dataset" / "01"
    gt = tmp_path / "dataset" / "01_GT" / "TRA"
    st = tmp_path / "dataset" / "01_ST" / "SEG"
    sequence.mkdir(parents=True)
    gt.mkdir(parents=True)
    st.mkdir(parents=True)

    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    tra = np.zeros((10, 10), dtype=np.uint16)
    tra[4, 4] = 1
    silver = np.zeros_like(tra)
    silver[3:6, 3:6] = 1
    imwrite(sequence / "t000.tif", image)
    imwrite(gt / "man_track000.tif", tra)
    (gt / "man_track.txt").write_text("1 0 0 0\n")
    imwrite(st / "man_seg000.tif", silver)

    imgs, masks, image_path, gt_path = load_ctc_images_masks(sequence, "TRA")

    np.testing.assert_allclose(imgs[0], normalize(image))
    np.testing.assert_array_equal(masks[0], np.maximum(tra, silver))
    assert image_path == sequence
    assert gt_path == gt

    _, seg_masks, seg_image_path, seg_gt_path = load_ctc_images_masks(sequence, "SEG")
    np.testing.assert_array_equal(seg_masks[0], silver)
    assert seg_image_path == sequence
    assert seg_gt_path == gt

    loaded = TrackingSequence.from_ctc(
        sequence, detection_folders=("SEG",), n_workers=1
    )
    assert loaded.segmentations[0].name == "SEG"
    assert loaded.segmentations[0].labels.tolist() == [1]


def test_load_images_flag_attaches_arrays_and_survives_pickle(tmp_path):
    root = _write_ctc_fixture(tmp_path, "simple")

    bare = TrackingSequence.from_ctc(root, n_workers=1)
    assert bare.images is None
    assert bare.segmentations[0].masks is None

    seq = TrackingSequence.from_ctc(root, n_workers=1, load_images=True)
    assert seq.images is not None and len(seq.images) == 3
    assert seq.segmentations[0].masks is not None
    assert len(seq.segmentations[0].masks) == 3
    assert not seq.images.flags.writeable

    restored = pickle.loads(pickle.dumps(seq))
    np.testing.assert_array_equal(restored.images, seq.images)
    np.testing.assert_array_equal(
        restored.segmentations[0].masks, seq.segmentations[0].masks
    )


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
    sample = TrackingDataset(sequence, window_size=2)[0]

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
    assert not cached.segmentations[0].coords.flags.writeable
    assert np.array_equal(sequence.lineage_relation, cached.lineage_relation)


def test_tracking_data_runtime_selection_keeps_complete_lineages():
    seg = _segmentation([(0, (0, 1, 2)), (1, (0, 1, 2))])
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        segmentations=(seg,),
        lineage_relation=np.eye(3, dtype=bool),
        lineage_parents=np.full(3, -1),
    )

    np.random.seed(0)
    capped = TrackingDataset(sequence, window_size=2, max_detections=2)[0]
    np.random.seed(0)
    dropped = TrackingDataset(
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
    seg = _segmentation([(0, (0, 1)), (1, (0, 1))])
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        segmentations=(seg,),
        lineage_relation=np.eye(2, dtype=bool),
        lineage_parents=np.full(2, -1),
    )
    original = seg.features["inertia_tensor"].copy()

    np.random.seed(4)
    torch.manual_seed(4)
    sample = TrackingDataset(sequence, window_size=2, augment=level)[0]

    assert sample["features"].shape == (4, 7)
    np.testing.assert_array_equal(seg.features["inertia_tensor"], original)
