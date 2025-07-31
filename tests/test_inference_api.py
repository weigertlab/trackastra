import os
import tempfile

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import numpy as np
import pytest
from trackastra.data import example_data_fluo_3d, example_data_hela
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks


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

    # TODO store predictions already on trackastra.TrackGraph
    predictions = model._predict(imgs, masks)

    track_graph = model._track_from_predictions(predictions)

    track_graph = model.track(
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
        _, _masks_tracked = graph_to_ctc(
            track_graph,
            masks,
            outdir=tmp,
        )

    _napari_tracks, _napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)


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
