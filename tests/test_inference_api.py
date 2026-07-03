import os
import tempfile

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from trackastra.data import example_data_fluo_3d, example_data_hela
from trackastra.model import Trackastra, TrackingTransformer
from trackastra.tracking import (
    graph_to_ctc,
    graph_to_napari_tracks,
    track_greedy,
    write_to_geff,
)

# Mark all tests in this module as core/inference tests
pytestmark = pytest.mark.core

try:
    import motile  # noqa: F401

    ILP_TESTS = True
except ModuleNotFoundError:
    ILP_TESTS = False


def test_resolve_inference_max_distance_defaults_warns_and_allows_lower(caplog):
    import logging

    from trackastra.model.model_api import _resolve_inference_max_distance

    # default to the model's trained radius
    assert _resolve_inference_max_distance(256, None) == 256

    # a lower request is honoured silently (tighter linking)
    with caplog.at_level(logging.WARNING):
        assert _resolve_inference_max_distance(256, 100) == 100
    assert caplog.text == ""

    # a higher request is honoured but warns
    with caplog.at_level(logging.WARNING):
        assert _resolve_inference_max_distance(256, 400) == 400
    assert "exceeds" in caplog.text


def test_track_points_reuses_prediction_and_tracking_path(monkeypatch):
    transformer = TrackingTransformer(
        coord_dim=2,
        feat_dim=0,
        d_model=16,
        nhead=2,
        num_encoder_layers=0,
        num_decoder_layers=0,
        pos_embed_per_dim=2,
        window=2,
    )
    model = Trackastra(
        transformer=transformer,
        inference_config={"features": "none"},
        device="cpu",
        batch_size=1,
    )
    seen = {}

    def fake_predict(features, windows, spatial_dim, **_kwargs):
        seen["features"] = features
        seen["windows"] = windows
        seen["spatial_dim"] = spatial_dim
        return {
            "nodes": [
                {"id": 0, "coords": (0.0, 0.0), "time": 0, "label": 1},
                {"id": 1, "coords": (1.0, 1.0), "time": 1, "label": 1},
            ],
            "weights": tuple(),
        }

    def fake_track(predictions, mode="greedy", return_candidate=False, **_kwargs):
        graph = nx.DiGraph()
        graph.add_nodes_from((node["id"], node) for node in predictions["nodes"])
        candidate = nx.DiGraph()
        if return_candidate:
            return graph, candidate
        return graph

    monkeypatch.setattr(model, "_predict_from_windows", fake_predict)
    monkeypatch.setattr(model, "_track_from_predictions", fake_track)

    result = model.track_points(
        coords=[
            np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32),
            np.array([[1.0, 1.0]], dtype=np.float32),
        ],
        return_details=True,
        progbar_class=lambda iterable, **_kwargs: iterable,
    )

    assert set(result.graph.nodes) == {0, 1}
    assert seen["spatial_dim"] == 2
    assert len(seen["features"]) == 2
    assert tuple(seen["windows"][0]["features"].shape) == (3, 0)
    assert tuple(seen["windows"][0]["coords"].shape) == (3, 2)
    assert tuple(seen["windows"][0]["labels"].tolist()) == (1, 2, 1)
    assert result.candidate_graph is not None
    assert result.predictions is not None


def test_track_points_accepts_explicit_feature_matrix(monkeypatch):
    transformer = TrackingTransformer(
        coord_dim=2,
        feat_dim=1,
        d_model=16,
        nhead=2,
        num_encoder_layers=0,
        num_decoder_layers=0,
        pos_embed_per_dim=2,
        window=2,
    )
    model = Trackastra(
        transformer=transformer,
        inference_config={"features": "_custom"},
        device="cpu",
        batch_size=1,
    )
    seen = {}

    def fake_predict(_features, windows, spatial_dim, **_kwargs):
        seen["window_features"] = windows[0]["features"]
        seen["spatial_dim"] = spatial_dim
        return {"nodes": [], "weights": tuple()}

    monkeypatch.setattr(model, "_predict_from_windows", fake_predict)
    monkeypatch.setattr(model, "_track_from_predictions", lambda *_args, **_kwargs: nx.DiGraph())

    result = model.track_points(
        coords=[
            np.array([[0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 1.0]], dtype=np.float32),
        ],
        features=[
            np.array([[0.25]], dtype=np.float32),
            np.array([[0.75]], dtype=np.float32),
        ],
        progbar_class=lambda iterable, **_kwargs: iterable,
    )

    assert len(result.graph) == 0
    assert seen["spatial_dim"] == 2
    assert tuple(seen["window_features"].shape) == (2, 1)
    np.testing.assert_allclose(seen["window_features"].numpy(), [[0.25], [0.75]])


def test_track_points_runs_tiny_point_only_model():
    transformer = TrackingTransformer(
        coord_dim=2,
        feat_dim=0,
        d_model=16,
        nhead=2,
        num_encoder_layers=0,
        num_decoder_layers=0,
        pos_embed_per_dim=2,
        window=2,
        causal_norm="none",
    )
    model = Trackastra(
        transformer=transformer,
        inference_config={"features": "none"},
        device="cpu",
        batch_size=1,
    )

    result = model.track_points(
        coords=[
            np.array([[0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 1.0]], dtype=np.float32),
        ],
        progbar_class=lambda iterable, **_kwargs: iterable,
    )

    assert set(result.graph.nodes) == {0, 1}
    assert result.graph.nodes[0]["coords"] == (0.0, 0.0)
    assert result.graph.nodes[1]["coords"] == (1.0, 1.0)


def test_track_points_requires_matching_feature_width():
    transformer = TrackingTransformer(
        coord_dim=2,
        feat_dim=1,
        d_model=16,
        nhead=2,
        num_encoder_layers=0,
        num_decoder_layers=0,
        pos_embed_per_dim=2,
        window=2,
    )
    model = Trackastra(
        transformer=transformer,
        inference_config={"features": "_custom"},
        device="cpu",
        batch_size=1,
    )

    with pytest.raises(ValueError, match="feature width"):
        model.track_points(
            coords=[
                np.array([[0.0, 0.0]], dtype=np.float32),
                np.array([[1.0, 1.0]], dtype=np.float32),
            ],
            progbar_class=lambda iterable, **_kwargs: iterable,
        )


def test_greedy_retains_isolated_detections():
    candidate_graph = nx.DiGraph()
    candidate_graph.add_node(0, time=0, label=1)
    candidate_graph.add_node(1, time=1, label=1)
    candidate_graph.add_node(2, time=0, label=2)
    candidate_graph.add_edge(0, 1, weight=0.9)

    result = track_greedy(candidate_graph)

    assert set(result.nodes) == {0, 1, 2}
    assert set(result.edges) == {(0, 1)}
    assert result.nodes[2] == {"time": 0, "label": 2}


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

    result = model.track_masks(
        imgs,
        masks,
        mode="greedy",
        ilp_config="gt",
    )
    track_graph, masks_tracked = result.graph, result.masks

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

    result = model.track_masks(
        imgs,
        masks,
        mode="greedy",
    )
    track_graph, masks_tracked = result.graph, result.masks
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

    model.track_masks(
        imgs,
        masks,
        mode="greedy",
    )


def test_empty_window_greedy():
    imgs, masks = example_data_hela()

    model = Trackastra.from_pretrained("general_2d", device="cpu")
    window_size = model.transformer.config["window"]
    masks[:window_size] = 0

    model.track_masks(
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

    model.track_masks(
        imgs,
        masks,
        mode="ilp",
    )


def test_empty_sequence_greedy():
    imgs = np.zeros((10, 100, 100), dtype=float)
    masks = np.zeros((10, 100, 100), dtype=int)

    model = Trackastra.from_pretrained("general_2d", device="cpu")

    model.track_masks(
        imgs,
        masks,
        mode="greedy",
    )


@pytest.mark.skipif(not ILP_TESTS, reason="Package for ILP tracking not installed")
def test_empty_sequence_ilp():
    imgs = np.zeros((10, 100, 100), dtype=float)
    masks = np.zeros((10, 100, 100), dtype=int)

    model = Trackastra.from_pretrained("general_2d", device="cpu")

    model.track_masks(
        imgs,
        masks,
        mode="ilp",
    )
