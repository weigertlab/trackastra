import os
import urllib
import zipfile
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]


def download_gt_data(url: str, root_dir: str):
    data_dir = Path(root_dir) / "scripts" / "data" / "ctc"

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    filename = url.split("/")[-1]
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)

        # Unzip the data
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)


@pytest.fixture(scope="module")
def download_gt_hela():
    url = "http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip"
    download_gt_data(url, ROOT_DIR)


def test_train_dry_run():
    os.chdir(ROOT_DIR / "scripts")
    cmd = (
        "python train.py --config example_config.yaml"
        " --device cpu --dry --epochs 1"
        " --train_samples 2 --batch_size 2"
        " --num_decoder_layers 1 --num_decoder_layers 1"
        " --d_model 128 --num_workers 2"
    )
    print(cmd)
    result = os.system(cmd)

    assert result == 0
