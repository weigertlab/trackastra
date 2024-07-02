import os
import tempfile

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

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
