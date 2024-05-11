import logging
import os
import zipfile
from pathlib import Path

import requests
from pydantic import validate_call
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_and_unzip(url: str, new_folder: Path):
    # TODO make safe
    if new_folder.exists():
        print(f"{new_folder} already downloaded, skipping.")
        return

    # Download the zip file
    download(url, "downloaded_file.zip")

    # Unzip the file
    with zipfile.ZipFile("downloaded_file.zip", "r") as zip_ref:
        zip_ref.extractall("temp_folder")

    # Remove the zip folder
    os.remove("downloaded_file.zip")

    # Rename the top-level folder
    new_folder.parent.mkdir(parents=True, exist_ok=True)
    os.rename("temp_folder/" + os.listdir("temp_folder")[0], new_folder)

    # Remove the temporary folder
    os.rmdir("temp_folder")


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


@validate_call
def download_pretrained(name: str, download_dir: Path = Path("./.models")):
    # TODO make safe, introduce versioning
    models = {
        "ctc": "https://github.com/weigertlab/trackastra-models/releases/download/v0.1/ctc.zip",
    }
    download_dir.mkdir(exist_ok=True, parents=True)
    try:
        url = models[name]
    except KeyError:
        raise ValueError(
            f"Pretrained model `name` is not available. Choose from {list(models.keys())}"
        )
    download_and_unzip(url=url, new_folder=download_dir / name)
