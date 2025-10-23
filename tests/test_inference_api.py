import os
import tempfile

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import numpy as np
import pytest
from trackastra.data import example_data_fluo_3d, example_data_hela
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks, write_to_geff

# Mark all tests in this module as core/inference tests
pytestmark = pytest.mark.core

try:
    import motile  # noqa: F401

    ILP_TESTS = True
except ModuleNotFoundError:
    ILP_TESTS = False


@pytest.mark.parametrize(
    "example_data",
    [
        example_data_hela,
        example_data_fluo_3d,
    ],
    ids=["2d", "3d"],
)
def test_api(example_data):
    imgs, masks = example_data()
    model = Trackastra.from_pretrained(
        name="ctc",
        device="cpu",
    )

    predictions = model._predict(imgs, masks)

    track_graph = model._track_from_predictions(predictions)

    track_graph, masks_tracked = model.track(
        imgs,
        masks,
        mode="greedy",
        ilp_config="gt",
    )

    # track_graph = model.track_from_disk(
    # imgs_path=...,
    # masks_path=...,
    # )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _, _masks_ctc = graph_to_ctc(
            track_graph,
            masks_tracked,
            outdir=tmp,
        )

    _napari_tracks, _napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)


@pytest.mark.parametrize(
    "example_data",
    [
        example_data_hela,
        example_data_fluo_3d,
    ],
    ids=["2d", "3d"],
)
def test_write_to_geff(example_data):
    imgs, masks = example_data()
    model = Trackastra.from_pretrained(
        name="ctc",
        device="cpu",
    )

    track_graph, masks_tracked = model.track(
        imgs,
        masks,
        mode="greedy",
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_to_geff(
            track_graph,
            masks_tracked,
            outdir=tmp / "tracked_geff.zarr",
        )


def test_empty_frame():
    """Minimal test case with an intermediate empty mask."""

    imgs = np.zeros((3, 100, 100), dtype=np.uint16)
    masks = np.zeros((3, 100, 100), dtype=np.uint16)

    masks[0, 5:10, 5:10] = 1  # Detection in frame 0
    # frame 1 is empty
    masks[2, 80:85, 80:85] = 2  # Detection in frame 2

    model = Trackastra.from_pretrained("general_2d", device="cpu")

    model.track(
        imgs,
        masks,
        mode="greedy",
    )


def test_empty_window_greedy():
    imgs, masks = example_data_hela()

    model = Trackastra.from_pretrained("general_2d", device="cpu")
    window_size = model.transformer.config["window"]
    masks[:window_size] = 0

    model.track(
        imgs,
        masks,
        mode="greedy",
    )


@pytest.mark.skipif(not ILP_TESTS, reason="Package for ILP tracking not installed")
def test_empty_window_ilp():
    imgs, masks = example_data_hela()

    model = Trackastra.from_pretrained("general_2d", device="cpu")
    window_size = model.transformer.config["window"]
    masks[:window_size] = 0

    model.track(
        imgs,
        masks,
        mode="ilp",
    )


def test_empty_sequence_greedy():
    imgs = np.zeros((10, 100, 100), dtype=float)
    masks = np.zeros((10, 100, 100), dtype=int)

    model = Trackastra.from_pretrained("general_2d", device="cpu")

    model.track(
        imgs,
        masks,
        mode="greedy",
    )


@pytest.mark.skipif(not ILP_TESTS, reason="Package for ILP tracking not installed")
def test_empty_sequence_ilp():
    imgs = np.zeros((10, 100, 100), dtype=float)
    masks = np.zeros((10, 100, 100), dtype=int)

    model = Trackastra.from_pretrained("general_2d", device="cpu")

    model.track(
        imgs,
        masks,
        mode="ilp",
    )
