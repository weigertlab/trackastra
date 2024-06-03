import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_MODELS = {
    "ctc": (
        "https://github.com/weigertlab/trackastra-models/releases/download/v0.1/ctc.zip"
    ),
    "general_2d": "https://github.com/weigertlab/trackastra-models/releases/download/v0.1.1/general_2d.zip",
}


def download_and_unzip(url: str, dst: Path):
    # TODO make safe and use tempfile lib
    if dst.exists():
        print(f"{dst} already downloaded, skipping.")
        return

    # get the name of the zipfile
    zip_base = Path(url.split("/")[-1])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_file = tmp / zip_base
        # Download the zip file
        download(url, zip_file)

        # Unzip the file
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(tmp)

        shutil.move(tmp / zip_base.stem, dst)


def download(url: str, fname: Path):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(str(fname), "wb") as file, tqdm(
        desc=str(fname),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_pretrained(name: str, download_dir: Path | None = None):
    # TODO make safe, introduce versioning
    if download_dir is None:
        download_dir = Path("~/.trackastra/.models").expanduser()
    else:
        download_dir = Path(download_dir)

    download_dir.mkdir(exist_ok=True, parents=True)
    try:
        url = _MODELS[name]
    except KeyError:
        raise ValueError(
            "Pretrained model `name` is not available. Choose from"
            f" {list(_MODELS.keys())}"
        )
    folder = download_dir / name
    download_and_unzip(url=url, dst=folder)
    return folder
