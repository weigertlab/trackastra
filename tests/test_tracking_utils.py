import dask.array as da
import networkx as nx
import numpy as np
import pytest
import trackastra.tracking.tracking as tracking_module
from dask import config, delayed
from trackastra.tracking import (
    apply_solution_graph_to_masks,
    graph_to_ctc,
    track_greedy,
)


def test_greedy_progress_excludes_below_threshold_edges(monkeypatch):
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=0.9)
    graph.add_edge(2, 3, weight=0.5)
    graph.add_edge(4, 5, weight=0.49)
    displayed_edges = []

    def record_edges(edges, **kwargs):
        displayed_edges.extend(edges)
        return edges

    monkeypatch.setattr(tracking_module, "tqdm", record_edges)
    result = track_greedy(graph, threshold=0.5)

    assert len(displayed_edges) == 2
    assert set(result.edges) == {(0, 1), (2, 3)}


def test_apply_solution_graph_to_numpy_masks():
    masks = np.zeros((2, 8, 8), dtype=np.int32)
    masks[0, 1:3, 1:3] = 100
    masks[0, 5:7, 5:7] = 2
    masks[1, 2:4, 2:4] = 7

    graph = nx.DiGraph()
    graph.add_node(0, time=0, label=100)
    graph.add_node(1, time=1, label=7)
    graph.add_edge(0, 1)

    expected = masks.copy()
    expected[expected == 2] = 0
    result = apply_solution_graph_to_masks(graph, masks)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)


def test_apply_solution_graph_to_dask_masks_is_lazy_and_framewise():
    loads = [0, 0, 0]

    def load_frame(t):
        loads[t] += 1
        frame = np.zeros((12, 12), dtype=np.int32)
        frame[1:3, 1:3] = 1
        frame[5:7, 5:7] = 2
        return frame

    frames = [
        da.from_delayed(
            delayed(load_frame)(t),
            shape=(12, 12),
            dtype=np.int32,
        ).rechunk((6, 6))
        for t in range(3)
    ]
    masks = da.stack(frames).rechunk((2, 6, 6))

    graph = nx.DiGraph()
    for t in range(3):
        graph.add_node((t, 1), time=t, label=1)

    result = apply_solution_graph_to_masks(graph, masks)

    assert isinstance(result, da.Array)
    assert result.chunks == masks.chunks
    assert loads == [0, 0, 0]

    expected = np.zeros((3, 12, 12), dtype=np.int32)
    expected[:, 1:3, 1:3] = 1
    with config.set(scheduler="synchronous"):
        np.testing.assert_array_equal(result.compute(), expected)

    assert loads == [1, 1, 1]


def test_graph_to_ctc_accepts_tracked_dask_masks():
    masks = np.zeros((2, 8, 8), dtype=np.int32)
    masks[0, 1:3, 1:3] = 7
    masks[1, 2:4, 2:4] = 7
    masks[:, 5:7, 5:7] = 9
    masks = da.from_array(masks, chunks=(1, 4, 4))

    graph = nx.DiGraph()
    graph.add_node(0, time=0, label=7)
    graph.add_node(1, time=1, label=7)
    graph.add_edge(0, 1)

    tracked = apply_solution_graph_to_masks(graph, masks)
    tracks, ctc_masks = graph_to_ctc(graph, tracked)

    assert tracks.to_dict("records") == [{"label": 1, "t1": 0, "t2": 1, "parent": 0}]
    expected = np.zeros((2, 8, 8), dtype=np.int32)
    expected[0, 1:3, 1:3] = 1
    expected[1, 2:4, 2:4] = 1
    np.testing.assert_array_equal(ctc_masks.compute(), expected)


def test_graph_to_ctc_rejects_missing_label_in_last_frame():
    masks = np.zeros((2, 8, 8), dtype=np.int32)
    masks[0, 1:3, 1:3] = 7
    masks = da.from_array(masks, chunks=(1, 8, 8))

    graph = nx.DiGraph()
    graph.add_node(0, time=0, label=7)
    graph.add_node(1, time=1, label=7)
    graph.add_edge(0, 1)

    with pytest.raises(
        RuntimeError,
        match="CTC track labels are missing from the output masks",
    ):
        graph_to_ctc(graph, masks)
