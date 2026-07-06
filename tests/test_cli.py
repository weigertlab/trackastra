import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from tifffile import imwrite

# Mark all tests in this module as core/inference tests
pytestmark = pytest.mark.core


def example_dataset():
    img_dir = Path("test_data/img")
    img_dir.mkdir(exist_ok=True, parents=True)
    img = np.array([
        [0, 1, 1],  # t=0
        [0, 1, 0],  # t=1
        [1, 1, 0],  # t=2
    ])
    img = np.expand_dims(img, -1)
    for i in range(img.shape[0]):
        imwrite(img_dir / f"emp_{i}.tif", img[i])

    tra_dir = Path("test_data/TRA")
    tra_dir.mkdir(exist_ok=True, parents=True)
    man_track = np.array(
        [
            [10, 0, 1, 0],
            [11, 0, 0, 0],
            [20, 2, 2, 10],
            [22, 2, 2, 10],
        ],
        dtype=int,
    )
    np.savetxt(tra_dir / "man_track.txt", man_track, fmt="%i")
    y = np.array([
        [00, 10, 11],  # t=0
        [00, 10, 00],  # t=1
        [20, 22, 00],  # t=2
    ])
    y = np.expand_dims(y, -1)
    for i in range(y.shape[0]):
        imwrite(tra_dir / f"track_{i}.tif", y[i])


@pytest.fixture
def predict_script():
    pytest.importorskip("configargparse")
    spec = importlib.util.spec_from_file_location(
        "predict_script", Path(__file__).parents[1] / "scripts" / "predict.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def train_script():
    pytest.importorskip("configargparse")
    spec = importlib.util.spec_from_file_location(
        "train_script", Path(__file__).parents[1] / "scripts" / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_parser():
    result = os.system("trackastra")
    assert result == 0


def test_cli_tracking_from_folder():
    example_dataset()
    cmd = "trackastra track -i test_data/img -m test_data/TRA --output-ctc test_data/tracked --output-edge-table test_data/tracked.csv --model-pretrained general_2d --device cpu"  # noqa: RUF100
    print(cmd)
    result = os.system(cmd)
    assert Path("test_data/tracked").exists()
    assert Path("test_data/tracked.csv").exists()
    assert result == 0


def test_cli_tracking_from_file():
    root = Path(__file__).parent.parent / "trackastra" / "data" / "resources"
    print(root)
    assert root.exists()
    output_ctc = Path("test_data") / "tracked_bacteria"
    output_edge_table = Path("test_data") / "tracked_bacteria.csv"

    cmd = f"trackastra track -i {root / 'trpL_150310-11_img.tif'} -m {root / 'trpL_150310-11_mask.tif'} --output-ctc {output_ctc} --output-edge-table {output_edge_table} --model-pretrained general_2d --device cpu"  # noqa: RUF100
    print(cmd)
    result = os.system(cmd)

    assert output_ctc.exists()
    assert output_edge_table.exists()
    assert result == 0


def test_predict_parser_reads_test_movies_from_config(tmp_path, predict_script):
    config = tmp_path / "config.yaml"
    config.write_text(
        "input_test:\n- movie_a\n- movie_b\ndetection_folders:\n- SEG\nepochs: 10\n"
    )

    args = predict_script.parse_args(["-m", "model", "-c", str(config)])

    assert args.input_test == [Path("movie_a"), Path("movie_b")]
    assert args.detection_folders == ["SEG"]
    assert args.spatial_cutoff is None
    assert predict_script.parse_args(
        ["-m", "model", "-c", str(config), "-f"]
    ).overwrite


def test_tracking_frequency_uses_completed_epoch_numbers(train_script):
    assert not train_script._is_tracking_epoch(0, 10)
    assert train_script._is_tracking_epoch(9, 10)
    assert train_script._is_tracking_epoch(19, 10)
    assert not train_script._is_tracking_epoch(9, 0)


def test_resolve_ctc_paths(tmp_path, predict_script):
    simple = tmp_path / "simple"
    (simple / "img").mkdir(parents=True)
    (simple / "TRA").mkdir()
    assert predict_script.resolve_ctc_paths(simple, "TRA") == (
        simple / "img",
        simple / "TRA",
        simple / "TRA",
    )

    sequence = tmp_path / "dataset" / "01"
    sequence.mkdir(parents=True)
    (tmp_path / "dataset" / "01_GT" / "TRA").mkdir(parents=True)
    (tmp_path / "dataset" / "01_ST" / "SEG").mkdir(parents=True)
    assert predict_script.resolve_ctc_paths(sequence, "SEG") == (
        sequence,
        tmp_path / "dataset" / "01_ST" / "SEG",
        tmp_path / "dataset" / "01_GT" / "TRA",
    )


def test_evaluate_ctc_uses_ctc_metrics(monkeypatch, predict_script):
    calls = {}
    traccuracy = ModuleType("traccuracy")
    loaders = ModuleType("traccuracy.loaders")
    matchers = ModuleType("traccuracy.matchers")
    metrics = ModuleType("traccuracy.metrics")

    loaders.load_ctc_data = lambda path, run_checks: (path, run_checks)
    matchers.CTCMatcher = type("CTCMatcher", (), {})
    metrics.CTCMetrics = type("CTCMetrics", (), {})

    def fake_run_metrics(**kwargs):
        calls.update(kwargs)
        return [
            {
                "results": {
                    "TRA": 0.75,
                    "AOGM": 12,
                    "DET": 0.8,
                    "LNK": 0.7,
                    "fp_nodes": 1,
                    "fn_nodes": 2,
                    "ns_nodes": 3,
                    "fp_edges": 4,
                    "fn_edges": 5,
                    "ws_edges": 6,
                }
            }
        ], None

    traccuracy.run_metrics = fake_run_metrics
    monkeypatch.setitem(sys.modules, "traccuracy", traccuracy)
    monkeypatch.setitem(sys.modules, "traccuracy.loaders", loaders)
    monkeypatch.setitem(sys.modules, "traccuracy.matchers", matchers)
    monkeypatch.setitem(sys.modules, "traccuracy.metrics", metrics)

    result = predict_script.evaluate_ctc(Path("gt"), Path("prediction"))

    assert result["TRA"] == 0.75
    assert result["AOGM"] == 12.0
    assert result["DET"] == 0.8
    assert result["fp_edges"] == 4.0
    assert calls["gt_data"] == ("gt", False)
    assert calls["pred_data"] == ("prediction", False)
    assert isinstance(calls["matcher"], matchers.CTCMatcher)
    assert isinstance(calls["metrics"][0], metrics.CTCMetrics)


def test_error_viz_renders_wrong_semantic_once():
    spec = importlib.util.spec_from_file_location(
        "viz_utils", Path(__file__).parents[1] / "scripts" / "utils.py"
    )
    assert spec is not None and spec.loader is not None
    viz_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz_utils)

    class Graph:
        def __init__(self, graph):
            self.graph = graph
            self.nodes = graph.nodes

    class Flags:
        CTC_FALSE_NEG = "fn"
        CTC_FALSE_POS = "fp"
        WRONG_SEMANTIC = "ws"

    gt = nx.DiGraph()
    gt.add_node("g0", x=1, y=2, t=0)
    gt.add_node("g1", x=3, y=4, t=1)
    gt.add_edge("g0", "g1")
    pred = nx.DiGraph()
    pred.add_node("p0", x=1, y=2, t=0)
    pred.add_node("p1", x=3, y=4, t=1)
    pred.add_edge("p0", "p1", ws=True)

    edges = viz_utils._collect_edges(
        Graph(gt),
        Graph(pred),
        Flags,
        {"p0": "g0", "p1": "g1"},
        scale=1,
    )

    assert edges == [{"t": 1, "p0": (1, 2), "p1": (3, 4), "cls": "ws"}]


def test_link_type_breakdown_division_rates(predict_script):
    pytest.importorskip("traccuracy")
    from traccuracy._tracking_graph import EdgeFlag

    # gt: a->b (regular), c->{d,e} (division); the missed daughter c->e is FN
    gt = nx.DiGraph()
    gt.add_edges_from([("a", "b"), ("c", "d"), ("c", "e")])
    # pred: a->b, c->d (regular), f->{g,h} (false division); the extra f->h is FP
    pred = nx.DiGraph()
    pred.add_edges_from([("a", "b"), ("c", "d"), ("f", "g"), ("f", "h")])

    class G:
        def __init__(self, digraph, flagged):
            self.graph = digraph
            self.edges = list(digraph.edges)
            self._flagged = flagged

        def get_edges_with_flag(self, flag):
            return self._flagged.get(flag, [])

    matched = SimpleNamespace(
        gt_graph=G(gt, {EdgeFlag.CTC_FALSE_NEG: [("c", "e")]}),
        pred_graph=G(pred, {EdgeFlag.CTC_FALSE_POS: [("f", "h")]}),
    )

    assert predict_script.link_type_breakdown(matched) == {
        "fn_div": 0.5,  # one of two GT division edges missed
        "fp_div": 0.5,  # one of two predicted division edges is spurious
        "f1_div": 0.5,  # precision 0.5, recall 0.5 -> F1 0.5
    }


def test_predict_run_writes_and_evaluates_ctc_output(
    tmp_path, monkeypatch, capsys, predict_script
):
    movie = tmp_path / "movie"
    (movie / "img").mkdir(parents=True)
    (movie / "TRA").mkdir()
    calls = {}

    class FakeTrackastra:
        @classmethod
        def from_pretrained(cls, name, device):
            calls["model"] = (name, device)
            return cls()

        def track(self, detections, **kwargs):
            calls["track"] = (detections, kwargs)
            return SimpleNamespace(
                graph="graph",
                masks="masks",
                candidate_graph=None,
                predictions=None,
            )

    class FakeDetectionSequence:
        @classmethod
        def from_masks(cls, images, masks, **kwargs):
            calls["detections"] = (images, masks, kwargs)
            return "detections"

    def fake_graph_to_ctc(graph, masks, outdir):
        calls["ctc"] = (graph, masks, outdir)
        (outdir / "man_track.txt").write_text("1 0 0 0\n")

    monkeypatch.setattr("trackastra.model.Trackastra", FakeTrackastra)
    monkeypatch.setattr("trackastra.data.DetectionSequence", FakeDetectionSequence)
    monkeypatch.setattr(
        "trackastra.data.load_ctc_images_masks",
        lambda root, detection_folder, ndim: (
            "images",
            "refined masks",
            movie / "img",
            movie / "TRA",
        ),
    )
    monkeypatch.setattr("trackastra.tracking.graph_to_ctc", fake_graph_to_ctc)
    monkeypatch.setattr(
        predict_script,
        "evaluate_ctc",
        lambda gt, pred, return_matched=False: {
            "TRA": 0.5,
            "AOGM": 4.0,
            "DET": 0.6,
            "LNK": 0.7,
        },
    )
    args = SimpleNamespace(
        model="trained_model",
        device="cpu",
        input_test=[movie],
        detection_folders=["TRA"],
        outdir=tmp_path / "results",
        overwrite=False,
        mode="greedy",
        spatial_cutoff=42,
        spacing=None,
        normalize_diameter=None,
        errormovie=False,
    )

    result = predict_script.run(args)
    assert result["movie"].tolist() == ["movie", "Mean"]
    assert result["model"].tolist() == ["trained_model", "trained_model"]
    assert result["mode"].tolist() == ["greedy", "greedy"]
    assert result["TRA"].tolist() == [0.5, 0.5]
    assert result["AOGM"].tolist() == [4.0, 4.0]
    assert calls["model"] == ("trained_model", "cpu")
    assert calls["detections"][0:2] == ("images", "refined masks")
    assert calls["detections"][2] == {
        "name": "movie",
        "ndim": 2,
        "spacing": None,
        "normalize_imgs": False,
        "keep_masks": True,
        "keep_images": True,
    }
    assert calls["track"] == (
        "detections",
        {
            "mode": "greedy",
            "spatial_cutoff": 42,
            "normalize_diameter": None,
        },
    )
    model_output = tmp_path / "results" / "trained_model"
    assert calls["ctc"][2] == model_output / "movie"
    per_movie = predict_script.pd.read_csv(model_output / "movie" / "metrics.csv")
    assert per_movie["movie"].tolist() == ["movie"]
    saved = predict_script.pd.read_csv(model_output / "metrics.csv")
    assert saved[["movie", "model", "mode"]].to_dict("records") == [
        {"movie": "movie", "model": "trained_model", "mode": "greedy"},
        {"movie": "Mean", "model": "trained_model", "mode": "greedy"},
    ]
    output = capsys.readouterr().out
    assert "movie" in output
    assert "Mean" in output
    assert "0.500000" in output
