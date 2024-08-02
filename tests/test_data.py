from pathlib import Path

import numpy as np
import pytest
import torch
from tifffile import imwrite
from trackastra.data import CTCData, collate_sequence_padding


def example_dataset():
    img_dir = Path("test_data/img")
    img_dir.mkdir(exist_ok=True, parents=True)
    img = np.array(
        [
            [0, 1, 1],  # t=0
            [0, 1, 0],  # t=1
            [1, 1, 0],  # t=2
        ]
    )
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

    y = np.array(
        [
            [00, 10, 11],  # t=0
            [00, 10, 00],  # t=1
            [20, 22, 00],  # t=2
        ]
    )
    y = np.expand_dims(y, -1)
    for i in range(y.shape[0]):
        imwrite(tra_dir / f"track_{i}.tif", y[i])

    pred_dir = Path("test_data/RES")
    pred_dir.mkdir(exist_ok=True, parents=True)
    pred = np.array(
        [
            [50, 41, 00],  # t=0
            [41, 50, 00],  # t=1
            [43, 43, 60],  # t=2
        ]
    )
    pred = np.expand_dims(pred, -1)

    for i in range(pred.shape[0]):
        imwrite(pred_dir / f"pred_{i}.tif", pred[i])

    emp_dir = Path("test_data/EMPTY")
    emp_dir.mkdir(exist_ok=True, parents=True)
    emp = np.array(
        [
            [0, 0, 0],  # t=0
            [0, 0, 0],  # t=1
            [0, 0, 0],  # t=2
        ]
    )
    emp = np.expand_dims(emp, -1)

    for i in range(emp.shape[0]):
        imwrite(emp_dir / f"emp_{i}.tif", emp[i])

    one_dir = Path("test_data/ONE_DET")
    one_dir.mkdir(exist_ok=True, parents=True)
    one = np.array(
        [
            [1, 0, 0],  # t=0
            [0, 0, 0],  # t=1
            [0, 0, 0],  # t=2
        ]
    )
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
        == torch.tensor(
            [
                [1.0, 0.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )
    )

    res = data[1]
    assert_dims(res)
    assert torch.all(
        res["assoc_matrix"]
        == torch.tensor(
            [
                # t: 0, 0, 1, 1, 2, 2
                # id: 41, 50, 41, 50, 43, 60
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
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
    import psutil

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
    print(f"memory used by dataset:  {(mem2 - mem1) / 1024 ** 3:.2} GB")
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

