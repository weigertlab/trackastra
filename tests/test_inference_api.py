import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import pytest
import torch
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks


@pytest.mark.parametrize(
    "data_path",
    [
        "data/ctc_2024/DIC-C2DH-HeLa/train/01",
        "data/ctc_2024/Fluo-C3DL-MDA231/train/01",
    ],
    ids=["2d", "3d"],
)
def test_api(data_path: str):
    # data_path = "data/ctc_2024/DIC-C2DH-HeLa/train/01"
    data_path = Path(data_path)

    # imgs = load_tiff_timeseries(data_path, dtype=float)
    # imgs = np.stack([normalize(x) for x in imgs])
    # masks = load_tiff_timeseries(f"{data_path}_ST/SEG", dtype=int)

    from trackastra.data import example_data_hela

    imgs, masks = example_data_hela()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Trackastra.from_pretrained(
        name="general_2d",
        device=device,
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

    _, masks_tracked = graph_to_ctc(
        track_graph,
        masks,
        outdir=Path(__file__).parent.resolve() / "tmp" / data_path.name,
    )

    napari_tracks, napari_tracks_graph, _ = graph_to_napari_tracks(track_graph)
