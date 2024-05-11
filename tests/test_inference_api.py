import os
from pathlib import Path

import napari
import numpy as np
import pytest
from trackastra.data import load_tiff_timeseries
from trackastra.model import Trackastra
from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
from trackastra.utils import normalize


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "data_path",
    [
        "data/ctc_2024/DIC-C2DH-HeLa/train/01",
        "data/ctc_2024/Fluo-C3DL-MDA231/train/01",
    ],
    ids=["2d", "3d"],
)
def test_api(
    data_path: str,
    device: str,
):
    # if __name__ == "__main__":
    # data_path = "data/ctc_2024/DIC-C2DH-HeLa/train/01"
    # device = "cuda"

    # TODO download/commit datasets

    data_path = Path(data_path)

    imgs = load_tiff_timeseries(data_path, dtype=float)
    imgs = np.stack([normalize(x) for x in imgs])
    masks = load_tiff_timeseries(f"{data_path}_ST/SEG", dtype=int)

    model = Trackastra.load_pretrained(
        name="ctc",
        device=device,
    )
    # model = Trackastra.load_from_folder(
    #     # Path(__file__).parent.parent.resolve() / "scripts/runs/ctc_3d_new_3",
    #     device=device,
    # )

    # Steps
    # TODO it would probably make sense to already store the prediction as a trackastra.TrackGraph
    predictions = model._predict(imgs, masks)

    track_graph = model._track_from_predictions(predictions)

    # TODO: TrackGraph class that wraps a networkx graph
    track_graph = model.track(imgs, masks, mode="ilp", ilp_config="deepcell_gt")

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

    if "DISPLAY" in os.environ:
        if napari.current_viewer() is None:
            v = napari.Viewer()

        v.add_image(imgs)
        v.add_labels(masks_tracked)
        v.add_tracks(data=napari_tracks, graph=napari_tracks_graph)
    else:
        print("No display available.")
