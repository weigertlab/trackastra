from pathlib import Path

import numpy as np
import psutil
import pytest
import torch
from tifffile import imwrite
from trackastra.data import CTCData, collate_sequence_padding
from trackastra.data.data import (
    _sample_detection_keep_indices,
    _sample_neighborhood_indices,
    association_distances,
    warn_association_distances,
)
from trackastra.data.wrfeat import WRFeatures
from trackastra.utils import normalize

# Mark all tests in this module as requiring training dependencies
# (most tests are skipped as outdated, but they use CTCData which requires training pipeline)
pytestmark = pytest.mark.train


def test_max_detections_rejects_ambiguous_or_impossible_budgets():
    with pytest.raises(ValueError, match="Specify only max_detections"):
        CTCData(max_tokens=4, max_detections=4)
    with pytest.raises(ValueError, match="at least window_size"):
        CTCData(window_size=4, max_detections=3)


def test_load_for_inference_refines_tra_with_st(tmp_path):
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
    imwrite(st / "man_seg000.tif", silver)

    imgs, masks, image_path, gt_path = CTCData.load_for_inference(sequence, "TRA")

    np.testing.assert_allclose(imgs[0], normalize(image))
    np.testing.assert_array_equal(masks[0], np.maximum(tra, silver))
    assert image_path == sequence
    assert gt_path == gt


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


def test_detection_dropout_keeps_whole_lineages_in_wrfeat_getitem():
    features = WRFeatures(
        coords=np.arange(12, dtype=np.float32).reshape(6, 2),
        labels=np.array([1, 2, 3, 1, 2, 3]),
        timepoints=np.array([0, 0, 0, 1, 1, 1]),
        features={"value": np.arange(6, dtype=np.float32)[:, None]},
    )
    assoc = features.labels[:, None] == features.labels[None, :]
    data = CTCData.__new__(CTCData)
    data.features = "wrfeat"
    data.return_dense = False
    data.cropper = None
    data.augmenter = None
    data.detect_drop = 1.0
    data.detect_drop_fraction = 0.34  # drop 1 of the 3 first-frame lineages
    data.max_tokens = None
    data.ndim = 2
    data.windows = [{
        "coords": features.coords,
        "assoc_matrix": assoc,
        "labels": features.labels,
        "img": np.zeros((2, 1, 1), dtype=np.float32),
        "mask": np.zeros((2, 1, 1), dtype=np.int32),
        "timepoints": features.timepoints,
        "t1": 0,
        "wrfeat": features,
    }]

    state = np.random.get_state()
    try:
        np.random.seed(42)
        sample = data[0]
    finally:
        np.random.set_state(state)

    # one whole 2-frame lineage dropped -> 2 lineages (4 detections) remain
    assert sample["features"].shape == (4, 1)
    assert sample["assoc_matrix"].shape == (4, 4)
    assert len(set(sample["labels"].tolist())) == 2
    # whole lineages kept -> no censoring needed -> no loss mask emitted
    assert "loss_mask" not in sample


def test_max_detections_keeps_whole_lineages_in_wrfeat_getitem(monkeypatch):
    features = WRFeatures(
        coords=np.array([[0, 0], [50, 50], [100, 100]] * 2, dtype=np.float32),
        labels=np.array([1, 2, 3, 1, 2, 3]),
        timepoints=np.array([0, 0, 0, 1, 1, 1]),
        features={"value": np.arange(6, dtype=np.float32)[:, None]},
    )
    assoc = features.labels[:, None] == features.labels[None, :]
    data = CTCData.__new__(CTCData)
    data.features = "wrfeat"
    data.return_dense = False
    data.cropper = None
    data.augmenter = None
    data.detect_drop = 0.0
    data.max_detections = 2  # first frame has 3 -> cap triggers
    data.max_tokens = None
    data.ndim = 2
    data.windows = [{
        "coords": features.coords,
        "assoc_matrix": assoc,
        "labels": features.labels,
        "img": np.zeros((2, 1, 1), dtype=np.float32),
        "mask": np.zeros((2, 1, 1), dtype=np.int32),
        "timepoints": features.timepoints,
        "t1": 0,
        "wrfeat": features,
    }]
    monkeypatch.setattr(np.random, "randint", lambda _n: 0)  # anchor = closest seed

    sample = data[0]

    # 2 first-frame seeds (the spatially-closest) -> their 2 full lineages = 4 dets
    assert len(sample["coords"]) == 4
    assert set(sample["timepoints"].tolist()) == {0, 1}
    assert set(sample["labels"].tolist()) == {1, 2}  # the far lineage (3) dropped
    assert "loss_mask" not in sample


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


def test_collate_pads_optional_neighborhood_loss_masks():
    def sample(n):
        return {
            "coords": torch.zeros((n, 3)),
            "features": torch.zeros((n, 1)),
            "labels": torch.arange(n),
            "timepoints": torch.arange(n),
            "assoc_matrix": torch.eye(n),
        }

    first = sample(2)
    first["loss_mask"] = torch.tensor([[True, False], [True, True]])
    second = sample(1)
    batch = collate_sequence_padding([first, second])

    assert torch.equal(batch["loss_mask"][0], first["loss_mask"])
    assert batch["loss_mask"][1, 0, 0]
    assert not batch["loss_mask"][1, 1].any()


def test_collate_assoc_coo_densifies_to_dense_padding():
    from trackastra.data.data import densify_assoc

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


def test_association_distance_warning_deduplicates_overlapping_windows(caplog):
    windows = [
        {
            "t1": 0,
            "coords": np.array([[0.0, 0.0], [3.0, 0.0], [13.0, 0.0]]),
            "labels": np.array([10, 11, 12]),
            "timepoints": np.array([0, 1, 2]),
            "assoc_matrix": np.array(
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool
            ),
        },
        {
            "t1": 1,
            "coords": np.array([[3.0, 0.0], [13.0, 0.0]]),
            "labels": np.array([11, 12]),
            "timepoints": np.array([1, 2]),
            "assoc_matrix": np.ones((2, 2), dtype=bool),
        },
    ]

    distances = association_distances(windows, delta_cutoff=1)
    assert sorted(distances.tolist()) == [3.0, 10.0]

    warn_association_distances(
        distances,
        max_distance=5,
        delta_cutoff=1,
        cutoff_name="spatial_pos_cutoff",
        dataset_name="test dataset",
    )
    assert "1/2 (50.00%)" in caplog.text
    assert "max=10.00" in caplog.text
    assert "cannot be recovered" in caplog.text


def test_association_distance_warning_ignores_values_at_cutoff(caplog):
    warn_association_distances(
        np.array([3.0, 10.0]),
        max_distance=10,
        delta_cutoff=1,
        cutoff_name="spatial_pos_cutoff",
        dataset_name="test dataset",
    )
    assert caplog.text == ""


def example_dataset():
    img_dir = Path("test_data/img")
    img_dir.mkdir(exist_ok=True, parents=True)
    img = np.array([
        [0, 1, 1],  # t=0
        [0, 1, 0],  # t=1
        [1, 1, 0],  # t=2
    ])
    img = np.expand_dims(img, -1)

    for i in range(img.shape[0]):
        imwrite(img_dir / f"emp_{i}.tif", img[i])

    tra_dir = Path("test_data/TRA")
    tra_dir.mkdir(exist_ok=True, parents=True)

    man_track = np.array(
        [
            [10, 0, 1, 0],
            [11, 0, 0, 0],
            [20, 2, 2, 10],
            [22, 2, 2, 10],
        ],
        dtype=int,
    )
    np.savetxt(tra_dir / "man_track.txt", man_track, fmt="%i")

    y = np.array([
        [00, 10, 11],  # t=0
        [00, 10, 00],  # t=1
        [20, 22, 00],  # t=2
    ])
    y = np.expand_dims(y, -1)
    for i in range(y.shape[0]):
        imwrite(tra_dir / f"track_{i}.tif", y[i])

    pred_dir = Path("test_data/RES")
    pred_dir.mkdir(exist_ok=True, parents=True)
    pred = np.array([
        [50, 41, 00],  # t=0
        [41, 50, 00],  # t=1
        [43, 43, 60],  # t=2
    ])
    pred = np.expand_dims(pred, -1)

    for i in range(pred.shape[0]):
        imwrite(pred_dir / f"pred_{i}.tif", pred[i])

    emp_dir = Path("test_data/EMPTY")
    emp_dir.mkdir(exist_ok=True, parents=True)
    emp = np.array([
        [0, 0, 0],  # t=0
        [0, 0, 0],  # t=1
        [0, 0, 0],  # t=2
    ])
    emp = np.expand_dims(emp, -1)

    for i in range(emp.shape[0]):
        imwrite(emp_dir / f"emp_{i}.tif", emp[i])

    one_dir = Path("test_data/ONE_DET")
    one_dir.mkdir(exist_ok=True, parents=True)
    one = np.array([
        [1, 0, 0],  # t=0
        [0, 0, 0],  # t=1
        [0, 0, 0],  # t=2
    ])
    one = np.expand_dims(one, -1)

    for i in range(one.shape[0]):
        imwrite(one_dir / f"one_{i}.tif", one[i])


@pytest.mark.skip(reason="outdated")
def test_ctc_assoc_matrix_emtpy():
    example_dataset()
    data = CTCData(
        root="test_data",
        detection_folders=[
            # batch 0
            "TRA",
            "EMPTY",
            # batch 1
            "EMPTY",
            "EMPTY",
        ],
        window_size=3,
    )

    assert_dims(data[1])
    assert torch.all(data[1]["assoc_matrix"] == 0)

    loader_train = torch.utils.data.DataLoader(
        data,
        batch_size=2,
        collate_fn=collate_sequence_padding,
    )
    for batch in loader_train:
        pass

    return data


@pytest.mark.skip(reason="outdated")
def test_ctc_assoc_matrix_single_detection():
    example_dataset()
    data = CTCData(
        root="test_data",
        detection_folders=[
            # batch 0
            "TRA",
            "ONE_DET",
            # batch 1
            "EMPTY",
            "ONE_DET",
            # batch 2
            "ONE_DET",
            "ONE_DET",
        ],
        window_size=3,
    )

    assert_dims(data[1])
    assert torch.all(data[1]["assoc_matrix"] == 0)

    loader_train = torch.utils.data.DataLoader(
        data,
        batch_size=2,
        collate_fn=collate_sequence_padding,
    )
    for batch in loader_train:
        pass

    return data


@pytest.mark.skip(reason="outdated")
def test_ctc_assoc_matrix_toy_example():
    example_dataset()
    data = CTCData(
        root="test_data",
        detection_folders=["TRA", "RES"],
        window_size=3,
    )

    tra = data[0]
    assert_dims(tra)
    assert torch.all(
        tra["assoc_matrix"]
        == torch.tensor([
            [1.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 1.0],
        ])
    )

    res = data[1]
    assert_dims(res)
    assert torch.all(
        res["assoc_matrix"]
        == torch.tensor([
            # t: 0, 0, 1, 1, 2, 2
            # id: 41, 50, 41, 50, 43, 60
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
    )

    return data


@pytest.mark.skip(reason="outdated")
def test_ctc_assoc_matrix_toy_example_slice():
    example_dataset()
    data = CTCData(
        root="test_data",
        detection_folders=["TRA", "RES"],
        slice_pct=(0.4, 1.0),
        window_size=2,
    )

    return data


def assert_dims(x):
    assert x["assoc_matrix"].ndim == 2
    assert x["coords"].ndim == 2
    assert x["coords"].shape[1] == 3
    assert x["features"].ndim == 2
    assert x["labels"].ndim == 1
    assert x["masks"].ndim == 3
    assert x["timepoints"].ndim == 1
    if "img" in x:
        assert x["img"].ndim == 3


@pytest.mark.skip(reason="outdated")
def test_custom_data_mw(detection_folder="RES", features="regionprops2"):
    data = CTCData(
        # "../scripts/data/isbi_tracking/train/microtubule/MICROTUBULE_snr_7_density_mid/",
        "../scripts/data/deepcell/test/08",
        window_size=6,
        augment=True,
        features=features,
        detection_folders=[detection_folder],
        slice_pct=(0.4, 0.5),
        # crop_size=(128, 128),
        # features="none",
        return_dense=True,
    )

    batch = data[0]
    A = batch["assoc_matrix"]
    labels = batch["labels"]
    idx1 = batch["timepoints"] == 0
    idx2 = batch["timepoints"] == 1
    labels[idx1]
    labels[idx2]
    c1 = batch["coords"][idx1][:, 1:]
    c2 = batch["coords"][idx2][:, 1:]
    A_sub = A[idx1][:, idx2]

    print(A_sub.shape)
    i, j = np.stack(np.where(A_sub))

    return data, batch, c1[i].numpy(), c2[j].numpy()


@pytest.mark.skip(reason="outdated")
def test_memory(features="none"):
    process = psutil.Process()
    mem1 = process.memory_info().rss
    # data = ConcatDataset(tuple(CTCData(f"../scripts/data/deepcell/test/{i:02d}") for i in range(4)))
    data = CTCData(
        "../scripts/data/ker_phasecontrast/dataset1/sub_5/exp1_F0008/",
        # "../scripts/data/Fluo-N2DL-HeLa/train/02_GT",
        # augment=3,
        # crop_size=(320,320),
        window_size=4,
        compress=True,
        features=features,
    )
    mem2 = process.memory_info().rss
    # print(f"memory used by raw data: {data.imgs.nbytes / 1024 ** 3:.2} GB")
    print(f"memory used by dataset:  {(mem2 - mem1) / 1024**3:.2} GB")
    return data


@pytest.mark.skip(reason="outdated")
def test_compress(features=None):
    if features is None:
        features = ("none", "regionprops2", "patch_regionprops")
    else:
        features = (features,)
    for features in features:
        print(features)
        kwargs = dict(
            # root="../scripts/data/vanvliet/cib/140408-01/",
            root="../scripts/data/vanvliet/recA/151027-10/",
            features=features,
            augment=0,
            window_size=4,
            # slice_pct=(0.5, 0.9),
        )

        data1 = CTCData(compress=False, **kwargs)
        data2 = CTCData(compress=True, **kwargs)

        b1 = data1[0]
        b2 = data2[0]

        for k in b1.keys():
            v1, v2 = b1[k], b2[k]
            close = torch.allclose(v1, v2)
            if not close:
                print("ERROR")
                return data1, data2
            else:
                print(f"{k:20} OK")
    return data1, data2
