import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import trackastra.training as training_api
import trackastra.training.runtime as runtime_api
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from trackastra.data import distributed
from trackastra.data.dataset import (
    TrackingDataset,
    collate_sequence_padding,
)
from trackastra.data.distributed import (
    BalancedBatchSampler,
    BalancedDataModule,
    BalancedDistributedSampler,
    SequenceInputSpec,
    normalize_sequence_input_specs,
)
from trackastra.data.io import (
    DetectionSequence,
    DetectionSupervision,
    LineageGraph,
    TrackingSequence,
)
from trackastra.model import ModelConfig, TrackingTransformer
from trackastra.training import (
    DataSplitConfig,
    LightningTrainerRuntime,
    SequenceLoadingError,
    SourceSpecError,
    TrackastraTrainer,
    TrackingDatasetBundle,
    TrainConfig,
    _feature_dim,
    build_dataset,
    build_lightning_runtime,
    build_model,
    build_or_resume_lightning_module,
    build_trainer,
    configure_lightning_module_runtime_paths,
    create_train_parser,
    load_model_from_path,
    load_sequences,
    node_degree_class_weights,
    normalize_source_specs,
    parse_train_args,
    parse_training_config,
    resolve_model_checkpoint_reference,
    resume_checkpoint_path,
    tracking_inputs_from_sources,
)
from trackastra.training.callbacks import EpochMetricsCSV, TrackastraModelCheckpoint
from trackastra.training.lightning import TrackingLightningModule
from trackastra.training.losses import (
    apply_focal_weight as _apply_focal_weight,
)
from trackastra.training.losses import (
    child_ce_loss_matrix as _child_ce_loss_matrix,
)
from trackastra.training.losses import (
    edge_error_counts,
    metrics_from_counts,
)
from trackastra.training.losses import (
    quiet_softmax_child_log_null as _quiet_softmax_child_log_null,
)
from trackastra.training.losses import (
    reduce_decision_loss as _reduce_decision_loss,
)
from trackastra.training.losses import (
    reduce_matrix_loss as _reduce_matrix_loss,
)
from trackastra.training.tracking_validation import (
    summarize_lam_split_sweep,
)
from trackastra.training.tracking_validation import (
    summarize_tracking_metrics as _summarize_tracking_metrics,
)

# Mark all tests in this module as requiring training dependencies
pytestmark = pytest.mark.train

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_wrfeat2_feature_dim_supports_2d_and_3d():
    assert _feature_dim(2, "none") == 0
    assert _feature_dim(2, "intensity") == 1
    assert _feature_dim(2, "wrfeat2") == 6
    assert _feature_dim(2, "wrfeat2_no_intensity") == 5
    assert _feature_dim(3, "wrfeat2") == 9
    assert _feature_dim(3, "wrfeat2_no_intensity") == 8

    with pytest.raises(ValueError, match="Unknown feature mode"):
        _feature_dim(2, "wrfeat")


@pytest.mark.parametrize(
    ("features", "width"),
    (("wrfeat2", 6), ("wrfeat2_no_intensity", 5)),
)
def test_tracking_data_cpu_training_smoke(features, width):
    raw_features = {
        "equivalent_diameter_area": np.ones((2, 1), np.float32),
        "intensity": np.ones((2, 1), np.float32),
        "inertia_tensor": np.tile(np.eye(2, dtype=np.float32).ravel(), (2, 1)),
        "border_dist": np.zeros((2, 1), np.float32),
    }
    seg = DetectionSequence(
        name="TRA",
        n_frames=2,
        coords=np.tile(np.array([[0, 0], [4, 4]], dtype=np.float32), (2, 1)),
        labels=np.array([1, 2, 1, 2]),
        timepoints=np.array([0, 0, 1, 1], dtype=np.int64),
        features={k: np.tile(v, (2, 1)) for k, v in raw_features.items()},
    )
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        detections=(seg,),
        gt=LineageGraph(
            coords=np.zeros((0, 2), dtype=np.float32),
            timepoints=np.zeros(0, dtype=np.int64),
            node_ids=np.zeros(0, dtype=object),
            lineage_relation=np.eye(2, dtype=bool),
            lineage_parents=np.full(2, -1),
        ),
        supervision=(DetectionSupervision(lineage_index=np.array([0, 1, 0, 1])),),
    )
    batch = collate_sequence_padding([TrackingDataset(sequence, 2, features)[0]])
    model = TrackingTransformer(
        coord_dim=2,
        feat_dim=width,
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        pos_embed_per_dim=4,
        dropout=0,
    )

    loss = TrackingLightningModule(model, causal_norm="none")._common_step(batch)[
        "loss"
    ]
    loss.backward()

    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("features", "width"),
    (("wrfeat2", 9), ("wrfeat2_no_intensity", 8)),
)
def test_tracking_data_3d_wrfeat2_cpu_training_smoke(features, width):
    raw_features = {
        "equivalent_diameter_area": np.ones((2, 1), np.float32),
        "intensity": np.ones((2, 1), np.float32),
        "inertia_tensor": np.tile(np.eye(3, dtype=np.float32).ravel(), (2, 1)),
        "border_dist": np.zeros((2, 1), np.float32),
    }
    seg = DetectionSequence(
        name="TRA",
        n_frames=2,
        coords=np.tile(np.array([[0, 0, 0], [4, 4, 4]], dtype=np.float32), (2, 1)),
        labels=np.array([1, 2, 1, 2]),
        timepoints=np.array([0, 0, 1, 1], dtype=np.int64),
        features={k: np.tile(v, (2, 1)) for k, v in raw_features.items()},
    )
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=3,
        detections=(seg,),
        gt=LineageGraph(
            coords=np.zeros((0, 3), dtype=np.float32),
            timepoints=np.zeros(0, dtype=np.int64),
            node_ids=np.zeros(0, dtype=object),
            lineage_relation=np.eye(2, dtype=bool),
            lineage_parents=np.full(2, -1),
        ),
        supervision=(DetectionSupervision(lineage_index=np.array([0, 1, 0, 1])),),
    )
    batch = collate_sequence_padding([TrackingDataset(sequence, 2, features)[0]])
    model = TrackingTransformer(
        coord_dim=3,
        feat_dim=width,
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        pos_embed_per_dim=4,
        dropout=0,
    )

    loss = TrackingLightningModule(model, causal_norm="none")._common_step(batch)[
        "loss"
    ]
    loss.backward()

    assert torch.isfinite(loss)


@pytest.mark.parametrize("version", [1, 2])
def test_parse_architecture_version(monkeypatch, version):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "-c",
            str(ROOT_DIR / "scripts/configs/vanvliet.yaml"),
            "--architecture_version",
            str(version),
        ],
    )

    args = parse_train_args()

    assert args.architecture_version == version


def test_parse_max_neighbors_defaults_to_16(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "-c",
            str(ROOT_DIR / "scripts/configs/vanvliet.yaml"),
        ],
    )

    args = parse_train_args()

    assert args.max_neighbors == [16]


def test_parse_tracking_dataset_defaults(monkeypatch, tmp_path):
    config = tmp_path / "empty.yaml"
    config.write_text("{}\n")
    monkeypatch.setattr("sys.argv", ["train.py", "-c", str(config)])

    args = parse_train_args()

    assert args.window == 4
    assert args.features == "wrfeat2"


def test_parse_config_defaults_are_overridden_by_config_and_cli(monkeypatch, tmp_path):
    (tmp_path / "root.yaml").write_text("d_model: 111\nepochs: 111\nnum_workers: 111\n")
    (tmp_path / "mid.yaml").write_text(
        "defaults: root.yaml\nd_model: 222\nepochs: 222\n"
    )
    (tmp_path / "other.yaml").write_text("epochs: 333\n")
    config = tmp_path / "exp.yaml"
    config.write_text("defaults:\n  - mid.yaml\n  - other.yaml\nwindow: 8\n")
    monkeypatch.setattr(
        "sys.argv", ["train.py", "-c", str(config), "--num_workers", "9"]
    )

    args = parse_train_args()

    assert args.d_model == 222  # mid over root
    assert args.epochs == 333  # other over mid
    assert args.window == 8  # config over defaults
    assert args.num_workers == 9  # cli over everything
    assert args.batch_size == 8  # untouched parser default


def test_parse_config_defaults_missing_file(monkeypatch, tmp_path):
    config = tmp_path / "exp.yaml"
    config.write_text("defaults: nope.yaml\n")
    monkeypatch.setattr("sys.argv", ["train.py", "-c", str(config)])

    with pytest.raises(FileNotFoundError):
        parse_train_args()


def test_parse_config_defaults_cycle(monkeypatch, tmp_path):
    (tmp_path / "a.yaml").write_text("defaults: b.yaml\n")
    (tmp_path / "b.yaml").write_text("defaults: a.yaml\n")
    monkeypatch.setattr("sys.argv", ["train.py", "-c", str(tmp_path / "a.yaml")])

    with pytest.raises(ValueError, match="cycle"):
        parse_train_args()


def test_parse_training_config_accepts_extended_parser(monkeypatch, tmp_path):
    config = tmp_path / "empty.yaml"
    config.write_text("{}\n")
    parser = create_train_parser()
    parser.add_argument("--runtime_tag", default="unset")
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "-c", str(config), "--runtime_tag", "experiment-a"],
    )

    _model_config, _train_data_config, _val_data_config, train_config = (
        parse_training_config(parser)
    )

    assert train_config.runtime_kwargs["training_args"]["runtime_tag"] == "experiment-a"


def test_parse_grouped_input_config_preserves_mappings(monkeypatch, tmp_path):
    config = tmp_path / "grouped.yaml"
    config.write_text(
        """
ndim: 3
input_train:
- format: geff
  sparse_gt: true
  spacing: [4, 1, 1]
  match_max_distance: 5
  paths:
  - train_a
  - train_b
input_val:
- format: geff
  sparse_gt: true
  spacing: [4, 1, 1]
  match_max_distance: 5
  path: val_a
""",
    )
    monkeypatch.setattr("sys.argv", ["train.py", "-c", str(config)])

    args = parse_train_args()

    assert args.input_train == [
        {
            "format": "geff",
            "sparse_gt": True,
            "spacing": [4, 1, 1],
            "match_max_distance": 5,
            "paths": ["train_a", "train_b"],
        }
    ]
    assert args.input_val == [
        {
            "format": "geff",
            "sparse_gt": True,
            "spacing": [4, 1, 1],
            "match_max_distance": 5,
            "path": "val_a",
        }
    ]


def test_parse_disable_abs_pos(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "-c",
            str(ROOT_DIR / "scripts/configs/vanvliet.yaml"),
            "--disable_abs_pos",
        ],
    )

    args = parse_train_args()

    assert args.disable_abs_pos is True


def test_parse_disable_input_norm(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "-c",
            str(ROOT_DIR / "scripts/configs/vanvliet.yaml"),
            "--disable_input_norm",
        ],
    )

    args = parse_train_args()

    assert args.disable_input_norm is True


def test_parse_assoc_loss(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "-c",
            str(ROOT_DIR / "scripts/configs/vanvliet.yaml"),
            "--assoc_loss",
            "child_ce",
        ],
    )

    args = parse_train_args()

    assert args.assoc_loss == "child_ce"


def test_summarize_tracking_metrics_includes_linking_and_detection():
    metrics = pd.DataFrame(
        {
            "movie": ["01", "02", "Mean"],
            "TRA": [0.8, 1.0, 0.9],
            "AOGM": [4.0, 2.0, 3.0],
            "LNK": [0.7, 0.9, 0.8],
            "DET": [0.6, 1.0, 0.8],
            # 02 has no divisions -> NaN div rate, nanmean skips it
            "fn_div": [0.5, np.nan, np.nan],
            "fp_div": [0.2, np.nan, np.nan],
            "f1_div": [0.64, np.nan, np.nan],
        }
    )

    summary = _summarize_tracking_metrics(metrics)
    assert summary["val_TRA"] == pytest.approx(0.9)
    assert summary["val_AOGM"] == pytest.approx(3.0)
    assert summary["val_DET"] == pytest.approx(0.8)
    # LNK is logged as its error (1 - LNK), not the saturating score
    assert "val_LNK" not in summary
    assert summary["val_LNK_ERR"] == pytest.approx(0.2)
    # only division links are logged, under the post-solver "track" prefix
    assert summary["val_track_fn_div"] == pytest.approx(0.5)
    assert summary["val_track_fp_div"] == pytest.approx(0.2)
    assert summary["val_track_f1_div"] == pytest.approx(0.64)


def test_summarize_lam_split_sweep_selects_lowest_mean_aogm():
    metrics = pd.DataFrame(
        {
            "movie": ["01", "02"] * 3,
            "lam_split": [0.0, 0.0, 0.25, 0.25, 0.5, 0.5],
            "AOGM": [4.0, 6.0, 2.0, 4.0, 3.0, 5.0],
            "LNK": [0.7, 0.5, 0.9, 0.7, 0.8, 0.6],
            "TRA": [0.8, 0.6, 0.9, 0.8, 0.85, 0.75],
        }
    )

    summary = summarize_lam_split_sweep(metrics)

    assert summary["val_best_lam_split"] == pytest.approx(0.25)
    assert summary["val_best_AOGM"] == pytest.approx(3.0)
    assert summary["val_best_LNK_ERR"] == pytest.approx(0.2)
    assert summary["val_AOGM_lam_split_0"] == pytest.approx(5.0)
    assert summary["val_AOGM_lam_split_0p25"] == pytest.approx(3.0)


def test_tracking_eval_reuses_and_clears_cached_gt(monkeypatch):
    class CachedGT:
        def __init__(self):
            self.annotated = False
            self.clear_calls = 0

        def clear_annotations(self):
            self.annotated = False
            self.clear_calls += 1

    gt_data = CachedGT()
    pred_loads = []
    loaders = ModuleType("traccuracy.loaders")

    def load_ctc_data(path, run_checks):
        pred_loads.append((path, run_checks))
        return SimpleNamespace(path=path)

    loaders.load_ctc_data = load_ctc_data
    traccuracy = ModuleType("traccuracy")
    monkeypatch.setitem(sys.modules, "traccuracy", traccuracy)
    monkeypatch.setitem(sys.modules, "traccuracy.loaders", loaders)
    monkeypatch.setattr(
        "trackastra.tracking.graph_to_ctc", lambda *_args, **_kwargs: None
    )

    def ctc_metrics_from_data(gt, pred, return_matched):
        assert gt is gt_data
        assert not gt.annotated
        assert return_matched
        gt.annotated = True
        values = {
            "TRA": 0.8,
            "AOGM": 2.0,
            "DET": 1.0,
            "LNK": 0.75,
            "fp_nodes": 0.0,
            "fn_nodes": 0.0,
            "ns_nodes": 0.0,
            "fp_edges": 1.0,
            "fn_edges": 1.0,
            "ws_edges": 0.0,
        }
        return values, SimpleNamespace(gt_graph=gt, pred_graph=pred)

    def link_type_breakdown(matched):
        assert matched.gt_graph.annotated
        return {"fn_div": 0.0, "fp_div": 0.0, "f1_div": 1.0}

    predict_script = ModuleType("predict")
    predict_script.ctc_metrics_from_data = ctc_metrics_from_data
    predict_script.link_type_breakdown = link_type_breakdown
    monkeypatch.setitem(sys.modules, "predict", predict_script)

    model = TrackingTransformer(
        coord_dim=2,
        feat_dim=0,
        d_model=8,
        nhead=1,
        num_encoder_layers=0,
        num_decoder_layers=0,
        pos_embed_per_dim=2,
        spatial_cutoff=10,
    )
    module = TrackingLightningModule(model, tracking_mode="greedy")
    module._tracking_model = SimpleNamespace(
        _predict_from_windows=lambda *_args, **_kwargs: "predictions"
    )
    module._tracking_cache = [
        {
            "name": "movie",
            "features": [],
            "windows": [],
            "masks": np.zeros((1, 2, 2), dtype=np.uint16),
            "spatial_dim": 2,
            "gt_data": gt_data,
        }
    ]
    module._track_lam_split_sweep = lambda *_args: iter(
        ((0.0, "graph0"), (0.25, "graph025"))
    )

    metrics = module._run_tracking_eval()

    assert metrics["lam_split"].tolist() == [0.0, 0.25]
    assert gt_data.clear_calls == 4
    assert not gt_data.annotated
    assert len(pred_loads) == 2
    assert all(not run_checks for _path, run_checks in pred_loads)


def test_tracking_lam_split_sweep_reuses_candidate_graph(monkeypatch):
    model = TrackingTransformer(
        coord_dim=2,
        feat_dim=0,
        d_model=8,
        nhead=1,
        num_encoder_layers=0,
        num_decoder_layers=0,
        pos_embed_per_dim=2,
        spatial_cutoff=10,
        node_head=True,
    )
    module = TrackingLightningModule(model, tracking_mode="greedy")
    calls = []
    candidate_graph = object()
    module._tracking_model = SimpleNamespace(
        _track_from_predictions=lambda *args, **kwargs: (
            calls.append((args, kwargs)) or ("baseline", candidate_graph)
        )
    )

    greedy_calls = []

    def fake_track_greedy(candidate, config):
        greedy_calls.append((candidate, config.lam_split))
        return f"graph-{config.lam_split:g}"

    monkeypatch.setattr("trackastra.tracking.track_greedy", fake_track_greedy)

    results = list(module._track_lam_split_sweep("predictions", spatial_cutoff=10))

    assert [lam for lam, _graph in results] == [0.0, 0.1, 0.25, 0.5, 1.0]
    assert results[0][1] == "baseline"
    assert len(calls) == 1
    assert calls[0][1]["return_candidate"] is True
    assert greedy_calls == [
        (candidate_graph, 0.1),
        (candidate_graph, 0.25),
        (candidate_graph, 0.5),
        (candidate_graph, 1.0),
    ]


def test_metrics_from_counts():
    # TP=6, FP=2, FN=2: rates and scores from one shared triple
    m = metrics_from_counts(tp=6, fp=2, fn=2)
    assert m["fn"] == pytest.approx(2 / 8)  # FN / GT_pos = 1 - recall
    assert m["fp"] == pytest.approx(2 / 8)  # FP / pred_pos = 1 - precision
    assert m["f1"] == pytest.approx(12 / 16)
    assert m["jaccard"] == pytest.approx(6 / 10)
    # perfect: no FP, no FN
    perfect = metrics_from_counts(tp=5, fp=0, fn=0)
    assert perfect["f1"] == pytest.approx(1.0)
    assert perfect["jaccard"] == pytest.approx(1.0)
    assert perfect["fn"] == pytest.approx(0.0)
    assert perfect["fp"] == pytest.approx(0.0)
    # no edges at all -> every metric NaN
    empty = metrics_from_counts(tp=0, fp=0, fn=0)
    assert all(value != value for value in empty.values())
    # F1 and Jaccard stay consistent: J == F1 / (2 - F1)
    m2 = metrics_from_counts(tp=7, fp=3, fn=4)
    assert m2["jaccard"] == pytest.approx(m2["f1"] / (2 - m2["f1"]))


def test_edge_error_counts_for_division_links():
    from trackastra.utils import blockwise_sum_batched

    # t0: nodes 0,1 ; t1: nodes 2,3,4,5. GT: 0->2 (continuation), 1->{3,4} (division)
    tp = torch.tensor([[0, 0, 1, 1, 1, 1]])
    A = torch.zeros((1, 6, 6))
    A[0, 0, 2] = A[0, 1, 3] = A[0, 1, 4] = 1.0
    # prediction at t=0.5: keep 0->2 and 1->3, MISS 1->4 (FN div), add spurious
    # 0->5 (FP); 0->{2,5} makes node 0 a predicted division, so the FP lands in fp_div
    prob = torch.zeros((1, 6, 6))
    prob[0, 0, 2] = prob[0, 1, 3] = prob[0, 0, 5] = 0.9
    prob[0, 1, 4] = 0.4

    dt = tp.unsqueeze(1) - tp.unsqueeze(2)
    mask = ((dt > 0) & (dt <= 2)).float()
    b1 = blockwise_sum_batched(A, tp, dim=-1, reduce="sum")
    b2 = blockwise_sum_batched(A, tp, dim=-2, reduce="sum")
    block_sum = A * (b1 + b2)

    counts = edge_error_counts(A, prob, tp, mask, block_sum > 2)
    counts = {k: float(v) for k, v in counts.items()}

    assert counts == {
        "fn_num": 0.0,
        "fn_den": 1.0,  # continuation 0->2
        "fp_num": 0.0,  # spurious 0->5 is a predicted division edge
        "fp_den": 1.0,  # predicted regular edge 1->3
        "fn_div_num": 1.0,  # missed daughter 1->4
        "fn_div_den": 2.0,  # two GT division edges
        "fp_div_num": 1.0,  # spurious 0->5
        "fp_div_den": 2.0,  # two predicted division edges (0->2, 0->5)
    }

    threshold_counts = edge_error_counts(
        A, prob, tp, mask, block_sum > 2, thresholds=(0.2, 0.5, 0.8)
    )
    threshold_counts = {k: float(v) for k, v in threshold_counts.items()}

    for key, value in counts.items():
        assert threshold_counts[key] == value
        name, part = key.rsplit("_", 1)
        assert threshold_counts[f"{name}_t1_{part}"] == value
    assert threshold_counts["fn_div_t0_num"] == 0.0
    assert threshold_counts["fn_div_t0_den"] == 2.0
    assert threshold_counts["fp_div_t0_num"] == 1.0
    assert threshold_counts["fp_div_t0_den"] == 4.0


def test_assoc_time_counts_group_by_target_time():
    timepoints = torch.tensor([[0, 1, 1, 2]])
    dataset_index = torch.tensor([7])
    seg_index = torch.tensor([3])
    A = torch.zeros((1, 4, 4))
    A[0, 0, 1] = 1.0
    A[0, 0, 2] = 1.0
    A[0, 1, 3] = 1.0
    prob = torch.zeros((1, 4, 4))
    prob[0, 0, 1] = 0.9
    prob[0, 0, 2] = 0.1
    prob[0, 1, 3] = 0.9
    prob[0, 2, 3] = 0.9

    dt = timepoints.unsqueeze(1) - timepoints.unsqueeze(2)
    mask = ((dt > 0) & (dt <= 2)).float()

    counts = TrackingLightningModule._assoc_time_counts_for_batch(
        A, prob, timepoints, mask, dataset_index, seg_index
    )

    assert counts.keys() == {(7, 3, 1), (7, 3, 2)}
    np.testing.assert_array_equal(counts[(7, 3, 1)], np.array([1, 0, 1]))
    np.testing.assert_array_equal(counts[(7, 3, 2)], np.array([1, 1, 0]))


def test_assoc_time_error_matrix_pools_normalized_stream_bins():
    counts = {
        (0, 0, 10): np.array([1, 1, 0]),
        (0, 0, 20): np.array([2, 0, 0]),
        (0, 1, 5): np.array([0, 1, 1]),
    }

    matrix, labels = TrackingLightningModule._assoc_time_error_matrix(
        counts,
        {0: "/data/Fluo-N3DH-CE/01"},
        n_bins=3,
    )

    assert labels == ["Fluo-N3DH-CE/01"]
    assert matrix.shape == (1, 3)
    assert matrix[0, 0] == pytest.approx(3 / 4)
    # bin 1 has no timepoints (n_bins=3 exceeds the stream length); the interior
    # gap is nearest-filled from the closest measured bin (tie -> earlier bin 0).
    assert matrix[0, 1] == pytest.approx(3 / 4)
    assert matrix[0, 2] == pytest.approx(0.0)


def test_nearest_fill_interior_only_fills_gaps_between_measurements():
    fill = TrackingLightningModule._nearest_fill_interior
    nan = np.nan
    # interior gaps filled from nearest measured bin; ties -> earlier bin
    out = fill(np.array([nan, 1.0, nan, nan, 5.0, nan], dtype=np.float32))
    assert np.isnan(out[0]) and np.isnan(out[5])  # leading/trailing preserved
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(1.0)  # equidistant -> earlier bin
    assert out[3] == pytest.approx(5.0)
    # all-NaN row is left untouched
    assert np.all(np.isnan(fill(np.full(4, nan, dtype=np.float32))))


def test_assoc_time_dataset_labels_are_compact():
    assert (
        TrackingLightningModule._short_dataset_label("/data/Fluo-N3DH-CE/01", 0)
        == "Fluo-N3DH-CE/01"
    )
    assert (
        TrackingLightningModule._short_dataset_label("/data/biohub/44b6_0113de3b", 0)
        == "44b6_0113de3b"
    )
    assert TrackingLightningModule._short_dataset_label("", 4) == "4"


def test_dataset_metric_rows_and_csv_upsert(tmp_path):
    module = SimpleNamespace(
        current_epoch=2,
        global_step=17,
        dataset_metrics_path=tmp_path / "debug" / "dataset_metrics.csv",
        trainer=SimpleNamespace(is_global_zero=True),
        _dataset_root_map_for_stage=lambda stage: {
            3: f"data/{stage}/three",
            4: f"data/{stage}/four",
        },
    )
    counts = {
        "train": {3: np.array([12, 100, 8, 1, 2, 3, 2, 1])},
        "val": {4: np.array([4, 20, 2, 0, 0, 0, 0, 0])},
    }
    module._gather_dataset_assoc_counts = lambda: counts
    module._dataset_metric_rows = lambda values: (
        TrackingLightningModule._dataset_metric_rows(module, values)
    )

    TrackingLightningModule._write_dataset_metrics(module)
    TrackingLightningModule._write_dataset_metrics(module)

    metrics = pd.read_csv(module.dataset_metrics_path)
    assert len(metrics) == 2
    train = metrics[metrics["split"] == "train"].iloc[0]
    assert train["dataset_root"] == "data/train/three"
    assert train["sampled_windows"] == 12
    assert train["regular_f1"] == pytest.approx(16 / 19)
    assert train["division_jaccard"] == pytest.approx(3 / 6)


class _SamplerDataset(Dataset):
    def __init__(self, n=7):
        self.n = n
        self.n_objects = tuple(range(1, n + 1))
        self.n_divs = tuple(i % 3 for i in range(n))

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return index


def _decision_mask(timepoints):
    dt = timepoints.unsqueeze(1) - timepoints.unsqueeze(2)
    valid = timepoints >= 0
    mask = (dt == 1) & valid.unsqueeze(1) & valid.unsqueeze(2)
    return dt, mask


def test_decision_loss_does_not_dilute_positive_with_more_candidates():
    timepoints = torch.tensor([[0, 1, -1, -1], [0, 0, 0, 1]])
    dt, mask = _decision_mask(timepoints)
    pair_loss = torch.zeros((2, 4, 4))
    pair_loss[0, 0, 1] = 4
    pair_loss[1, 0, 3] = 4

    loss = _reduce_decision_loss(pair_loss, mask, dt, delta_cutoff=1)

    assert loss.item() == pytest.approx(4)


def test_decision_loss_averages_samples_equally():
    timepoints = torch.tensor([[0, 1, -1], [0, 1, 1]])
    dt, mask = _decision_mask(timepoints)
    pair_loss = torch.zeros((2, 3, 3))
    pair_loss[0, 0, 1] = 2
    pair_loss[1, 0, 1:] = 4

    loss = _reduce_decision_loss(pair_loss, mask, dt, delta_cutoff=1)

    assert loss.item() == pytest.approx(3)


def test_matrix_loss_averages_valid_samples_equally():
    timepoints = torch.tensor([[0, 1, -1], [0, 1, 1]])
    _, mask = _decision_mask(timepoints)
    pair_loss = torch.zeros((2, 3, 3))
    pair_loss[0, 0, 1] = 2
    pair_loss[1, 0, 1:] = 4
    eps = torch.finfo(torch.float16).eps
    counts = mask.sum(dim=(1, 2))
    per_sample = pair_loss.sum(dim=(1, 2)) / (counts + eps)
    sample_valid = counts > 0
    expected = (per_sample * sample_valid).sum() / sample_valid.sum().clamp_min(1)

    loss = _reduce_matrix_loss(pair_loss, mask)

    assert loss.item() == pytest.approx(expected.item())


def test_child_ce_loss_uses_true_parent_or_null_decision():
    timepoints = torch.tensor([[0, 0, 1]])
    dt, mask = _decision_mask(timepoints)
    log_p = torch.full((1, 3, 3), -20.0)
    log_p[0, 0, 2] = torch.log(torch.tensor(0.8))
    log_p[0, 1, 2] = torch.log(torch.tensor(0.1))
    log_p_null = torch.full_like(log_p, -20.0)
    log_p_null[0, 0, 2] = torch.log(torch.tensor(0.1))
    log_p_null[0, 1, 2] = torch.log(torch.tensor(0.1))

    target = torch.zeros((1, 3, 3))
    target[0, 0, 2] = 1
    positive_loss = _child_ce_loss_matrix(
        log_p, log_p_null, target, mask, dt, delta_cutoff=1
    )
    null_loss = _child_ce_loss_matrix(
        log_p, log_p_null, torch.zeros_like(target), mask, dt, delta_cutoff=1
    )

    assert positive_loss.sum().item() == pytest.approx(-np.log(0.8))
    assert null_loss.sum().item() == pytest.approx(-np.log(0.1))


def test_quiet_softmax_child_null_log_prob_uses_denominator_directly():
    logits = torch.full((1, 3, 3), -20.0)
    logits[0, 0, 2] = 0.0
    logits[0, 1, 2] = np.log(2.0)
    timepoints = torch.tensor([[0, 0, 1]])

    log_p_null = _quiet_softmax_child_log_null(logits, timepoints)

    assert log_p_null[0, 0, 2].item() == pytest.approx(-np.log(4.0), rel=1e-5)
    assert log_p_null[0, 1, 2].item() == pytest.approx(-np.log(4.0), rel=1e-5)


def test_child_ce_requires_quiet_softmax_and_decision_norm():
    model = torch.nn.Identity()

    with pytest.raises(ValueError, match="causal_norm='quiet_softmax'"):
        TrackingLightningModule(model, assoc_loss="child_ce", causal_norm="none")
    with pytest.raises(ValueError, match="loss_norm='decision'"):
        TrackingLightningModule(
            model,
            assoc_loss="child_ce",
            causal_norm="quiet_softmax",
            loss_norm="matrix",
        )


def test_tracking_lightning_module_saves_hyperparameters():
    model = torch.nn.Identity()
    module = TrackingLightningModule(
        model,
        warmup_epochs=2,
        max_epochs=5,
        learning_rate=0.002,
        causal_norm="none",
        loss_norm="matrix",
        div_upweight=3.0,
    )

    assert module.hparams["warmup_epochs"] == 2
    assert module.hparams["max_epochs"] == 5
    assert module.hparams["learning_rate"] == 0.002
    assert module.hparams["div_upweight"] == 3.0
    assert "model" not in module.hparams


def test_focal_weight_preserves_gamma_zero_and_focuses_hard_examples():
    bce = torch.tensor([0.1, 2.0])

    assert torch.equal(_apply_focal_weight(bce, gamma=0), bce)
    focal_ratio = _apply_focal_weight(bce, gamma=2) / bce
    assert focal_ratio[0] < focal_ratio[1]


def test_quiet_softmax_loss_keeps_bf16_gradients_finite():
    class FixedBF16Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            logits = torch.zeros((3, 3), dtype=torch.bfloat16)
            logits[0, 2] = 6
            logits[1, 2] = -6
            self.logits = torch.nn.Parameter(logits)

        def forward(self, coords, features, padding_mask=None):
            return self.logits.unsqueeze(0) + 0, None

    model = FixedBF16Model()
    module = TrackingLightningModule(
        model,
        causal_norm="quiet_softmax",
        loss_norm="decision",
        delta_cutoff=1,
    )
    batch = {
        "features": torch.zeros((1, 3, 1)),
        "coords": torch.zeros((1, 3, 2)),
        "assoc_coo": torch.zeros((0, 3), dtype=torch.int32),
        "timepoints": torch.tensor([[0, 0, 1]]),
        "padding_mask": torch.zeros((1, 3), dtype=torch.bool),
        "gt_predecessor_set_available": torch.ones((1, 3), dtype=torch.bool),
        "gt_successor_set_available": torch.ones((1, 3), dtype=torch.bool),
    }

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = module._common_step(batch)["loss"]
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.isfinite(model.logits.grad).all()
    assert model.logits.grad[0, 2] > 0.1


def test_common_step_masks_pairs_with_available_gt_link_sets():
    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros((4, 4)))

        def forward(self, coords, features, padding_mask=None):
            return self.logits.unsqueeze(0) + 0, None

    module = TrackingLightningModule(FixedModel(), causal_norm="none")
    batch = {
        "features": torch.zeros((1, 4, 1)),
        "coords": torch.zeros((1, 4, 2)),
        "assoc_coo": torch.zeros((0, 3), dtype=torch.int32),
        "timepoints": torch.tensor([[0, 0, 1, 1]]),
        "padding_mask": torch.zeros((1, 4), dtype=torch.bool),
        "matched_gt": torch.tensor([[True, True, True, True]]),
        "gt_predecessor_set_available": torch.tensor([[False, False, False, True]]),
        "gt_successor_set_available": torch.tensor([[False, True, False, False]]),
    }

    out = module._common_step(batch)

    assert out["mask"].bool().tolist() == [
        [
            [False, False, False, True],
            [False, False, True, True],
            [False, False, False, False],
            [False, False, False, False],
        ]
    ]


def test_common_step_emits_node_div_counts():
    # node 0 (t=0) divides -> out-degree 2; node 1 (t=0) has one successor. Both are
    # observable; nodes 2,3 (t=1) have no successor so are not scored. The model
    # predicts division for both node 0 (TP) and node 1 (FP), giving TP=1/FP=1/FN=0.
    class FixedNodeModel(torch.nn.Module):
        node_head = True

        def forward(
            self, coords, features, padding_mask=None, return_node_logits=False
        ):
            A = torch.zeros((1, 4, 4))
            out_logits = torch.zeros((1, 4, 3))
            out_logits[0, 0, 2] = 10.0  # node 0 -> predicted division (GT div: TP)
            out_logits[0, 1, 2] = 10.0  # node 1 -> predicted division (GT non-div: FP)
            in_logits = torch.zeros((1, 4, 2))
            if return_node_logits:
                return A, None, out_logits, in_logits
            return A, None

    module = TrackingLightningModule(
        FixedNodeModel(), causal_norm="none", node_loss=1.0, delta_cutoff=2
    )
    batch = {
        "features": torch.zeros((1, 4, 1)),
        "coords": torch.zeros((1, 4, 2)),
        "assoc_coo": torch.zeros((0, 3), dtype=torch.int32),
        "timepoints": torch.tensor([[0, 0, 1, 1]]),
        "padding_mask": torch.zeros((1, 4), dtype=torch.bool),
        "gt_predecessor_set_available": torch.tensor([[False, False, True, True]]),
        "gt_successor_set_available": torch.tensor([[True, True, False, False]]),
        "node_out_degree": torch.tensor([[2, 1, 0, 0]]),
        "node_in_degree": torch.tensor([[0, 0, 1, 1]]),
    }

    counts = module._common_step(batch)["edge_counts"]
    assert counts["node_div_fn_num"].item() == 0  # GT division node 0 is detected
    assert counts["node_div_fn_den"].item() == 1  # one GT division node
    assert counts["node_div_fp_num"].item() == 1  # node 1 wrongly called a division
    assert counts["node_div_fp_den"].item() == 2  # two predicted divisions

    from trackastra.training.losses import metrics_from_counts

    m = metrics_from_counts(1, 1, 0)  # TP, FP, FN
    assert abs(m["f1"] - 2 / 3) < 1e-6
    assert abs(m["jaccard"] - 0.5) < 1e-6


def test_degree_consistency_loss_teaches_edges_from_node_head():
    model = TrackingTransformer(
        coord_dim=2,
        feat_dim=4,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        node_head=True,
        attn_positional_bias="none",
    )
    lm = TrackingLightningModule(
        model, causal_norm="none", node_loss=1.0, consistency_weight=1.0, delta_cutoff=2
    )
    # source 0 (t=0) has two candidate successors (t=1); edges each carry prob 0.3, so
    # the edge-implied out-degree is 0.6, while the node head confidently says 2.
    A_pred = torch.zeros(1, 3, 3, requires_grad=True)
    with torch.no_grad():
        A_pred[0, 0, 1] = torch.logit(torch.tensor(0.3))
        A_pred[0, 0, 2] = torch.logit(torch.tensor(0.3))
    out_logits = torch.zeros(1, 3, 3, requires_grad=True)
    with torch.no_grad():
        out_logits[0, 0, 2] = 10.0  # node 0 -> confident division (out-degree 2)
    timepoints = torch.tensor([[0, 1, 1]])
    mask = torch.zeros(1, 3, 3)
    mask[0, 0, 1] = 1
    mask[0, 0, 2] = 1
    mask_invalid = torch.zeros(1, 3, 3, dtype=torch.bool)
    succ_avail = torch.ones(1, 3, dtype=torch.bool)
    out_tgt = torch.tensor([[2, 0, 0]])  # node 0 is a GT division

    loss = lm._degree_consistency_loss(
        A_pred, out_logits, out_tgt, succ_avail, timepoints, mask, mask_invalid
    )
    # Smooth L1 is linear beyond beta=1: abs(0.6 - 2.0) - 0.5 = 0.9.
    assert abs(loss.detach().item() - 0.9) < 1e-3

    loss.backward()
    # mutual consistency: gradients flow into both the edge logits and the node head
    assert A_pred.grad.abs().sum() > 0
    assert out_logits.grad is not None and out_logits.grad.abs().sum() > 0
    # and the edge side is pushed up toward out-degree 2
    assert A_pred.grad[0, 0, 1] < 0

    # a source whose GT successor set is unavailable (e.g. sparse-GT boundary) has an
    # untrained out-degree target and must be excluded, even though its edges are
    # masked in -> loss collapses to zero, no gradient leaks to those edges.
    A_pred2 = A_pred.detach().clone().requires_grad_(True)
    censored = lm._degree_consistency_loss(
        A_pred2,
        out_logits.detach(),
        out_tgt,
        torch.zeros(1, 3, dtype=torch.bool),  # succ set unavailable everywhere
        timepoints,
        mask,
        mask_invalid,
    )
    assert float(censored) == 0.0


def test_degree_consistency_loss_weights_by_out_degree_class():
    # two equal-error sources, one a GT division (out=2), one a normal continuation
    # (out=1). With class weights favouring divisions, the weighted loss must be pulled
    # toward the division node's error rather than the unweighted average of the two.
    model = TrackingTransformer(
        coord_dim=2,
        feat_dim=4,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        node_head=True,
        attn_positional_bias="none",
    )
    lm = TrackingLightningModule(
        model,
        causal_norm="none",
        node_loss=1.0,
        consistency_weight=1.0,
        delta_cutoff=2,
        node_out_weights=[1.0, 1.0, 9.0],  # heavily upweight the division class
    )
    # nodes 0,1 at t=0 each have one candidate successor (node 2 at t=1)
    A_pred = torch.full((1, 3, 3), torch.logit(torch.tensor(0.5)))
    out_logits = torch.zeros(1, 3, 3)
    out_logits[0, 0, 2] = 10.0  # node 0 -> division, edge_out 0.5 vs 2.0 -> err 1.0
    out_logits[0, 1, 1] = (
        10.0  # node 1 -> continuation, edge_out 0.5 vs 1.0 -> err 0.125
    )
    timepoints = torch.tensor([[0, 0, 1]])
    mask = torch.zeros(1, 3, 3)
    mask[0, 0, 2] = 1
    mask[0, 1, 2] = 1
    mask_invalid = torch.zeros(1, 3, 3, dtype=torch.bool)
    succ_avail = torch.ones(1, 3, dtype=torch.bool)
    out_tgt = torch.tensor([[2, 1, 0]])

    loss = lm._degree_consistency_loss(
        A_pred, out_logits, out_tgt, succ_avail, timepoints, mask, mask_invalid
    )
    # weighted mean = (9*1.0 + 1*0.125) / (9 + 1) = 0.9125.
    assert abs(float(loss) - 0.9125) < 1e-3


def test_consistency_weight_is_inactive_without_node_loss():
    model = TrackingTransformer(
        coord_dim=2, feat_dim=4, d_model=16, num_encoder_layers=1, num_decoder_layers=1
    )
    module = TrackingLightningModule(model, consistency_weight=1.0, node_loss=0.0)

    assert module.consistency_weight == 0.0


def test_balanced_batch_sampler_partial_batch():
    dataset = ConcatDataset([_SamplerDataset(7), _SamplerDataset(6)])
    sampler = BalancedBatchSampler(dataset, batch_size=4, n_pool=2, num_samples=10)

    batches = list(sampler)

    assert len(sampler) == len(batches) == 3
    assert sorted(len(batch) for batch in batches) == [2, 4, 4]


def test_oversample_density_preserves_dataset_mixture():
    # _SamplerDataset(n) has n_objects = 1..n. Oversampling dense windows must
    # only redistribute within a dataset (toward larger n_objects), not shift the
    # total sampling mass between the two datasets.
    dataset = ConcatDataset([_SamplerDataset(6), _SamplerDataset(6)])
    sampler = BalancedBatchSampler(
        dataset, batch_size=4, n_pool=2, oversample_density=1.0
    )
    idx = np.arange(len(sampler.n_objects))
    probs = sampler.get_probs(idx)

    ds0 = sampler.dataset_ids == 0
    # each dataset keeps half the mass (mean-1 per-dataset renormalization)
    assert probs[ds0].sum() == pytest.approx(0.5)
    assert probs[~ds0].sum() == pytest.approx(0.5)
    # within a dataset, denser windows are strictly more likely
    within = probs[ds0]
    assert np.all(np.diff(within) > 0)


def test_sample_factors_default_uniform():
    dataset = ConcatDataset([_SamplerDataset(5), _SamplerDataset(6)])
    sampler = BalancedBatchSampler(dataset, batch_size=4, n_pool=2)
    assert np.allclose(sampler.sample_weight, 1.0)


def test_write_sampler_prob_debug_csv(tmp_path):
    class RootDataset(_SamplerDataset):
        def __init__(self, n, root):
            super().__init__(n)
            self.root = root

    dataset = ConcatDataset(
        [
            RootDataset(4, Path("data/a")),
            RootDataset(4, Path("data/b")),
        ]
    )
    sampler = BalancedBatchSampler(
        dataset,
        batch_size=2,
        n_pool=2,
        oversample_density=1.0,
        weight_by_dataset=True,
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
    path = tmp_path / "debug" / "sampling_probs.csv"

    training_api._write_sampler_prob_debug(loader, path)

    df = pd.read_csv(path)
    assert list(df["global_index"]) == list(range(len(dataset)))
    assert sorted(df["dataset_root"].unique()) == ["data/a", "data/b"]
    assert df["sample_prob"].sum() == pytest.approx(1.0)
    assert set(df["oversample_density"]) == {1.0}
    assert set(df["weight_by_dataset"]) == {True}
    for _dataset_index, group in df.groupby("dataset_index"):
        assert group["sample_prob_within_dataset"].sum() == pytest.approx(1.0)
        assert group["dataset_prob_mass"].nunique() == 1


def test_write_dataset_summary_debug_csv(tmp_path):
    segment = DetectionSequence(
        name="points",
        n_frames=2,
        coords=np.array([[0, 0], [5, 0], [0.5, 0], [5.5, 0]], dtype=np.float32),
        labels=np.array([1, 2, 1, 2]),
        timepoints=np.array([0, 0, 1, 1], dtype=np.int64),
        features={"intensity": np.ones((4, 1), dtype=np.float32)},
    )
    sequence = TrackingSequence(
        root=Path("toy.geff"),
        ndim=2,
        detections=(segment,),
        gt=LineageGraph(
            coords=np.array([[0, 0], [0.5, 0]], dtype=np.float32),
            timepoints=np.array([0, 1]),
            node_ids=np.array(["a", "b"]),
            lineage_relation=np.eye(1, dtype=bool),
            lineage_parents=np.array([-1]),
            node_in_degree=np.array([0, 1]),
            node_out_degree=np.array([1, 0]),
            node_predecessor_set_available=np.array([False, True]),
            node_successor_set_available=np.array([True, False]),
        ),
        supervision=(
            DetectionSupervision(
                lineage_index=np.array([0, -1, 0, -1]),
                gt_node_index=np.array([0, -1, 1, -1]),
                matched_gt=np.array([True, False, True, False]),
                gt_predecessor_set_available=np.array([False, False, True, False]),
                gt_successor_set_available=np.array([True, False, False, False]),
            ),
        ),
    )
    dataset = ConcatDataset(
        [
            TrackingDataset(
                sequence,
                window_size=2,
                features="intensity",
                max_detections=1,
            )
        ]
    )
    sampler = BalancedBatchSampler(dataset, batch_size=1, num_samples=5)
    loader = DataLoader(dataset, batch_sampler=sampler)
    path = tmp_path / "debug" / "dataset_summary.csv"

    training_api._write_dataset_summary_debug(
        loader,
        path,
        candidate_k=1,
        candidate_mode="per_frame",
        spatial_cutoff=1.0,
        delta_cutoff=1,
    )

    row = pd.read_csv(path).iloc[0]
    assert row["dataset_root"] == "toy.geff"
    assert row["source_format"] == "geff"
    assert row["expected_samples_per_epoch"] == pytest.approx(5.0)
    assert row["matched_detection_fraction"] == pytest.approx(0.5)
    assert row["matched_gt_node_fraction"] == pytest.approx(1.0)
    assert row["candidate_recall_at_k"] == pytest.approx(1.0)
    assert row["cutoff_recall"] == pytest.approx(1.0)
    assert bool(row["intensity_available"])
    assert not bool(row["shape_available"])


def test_dataset_assoc_counts_group_by_dataset_and_link_type():
    timepoints = torch.tensor([[0, 1, 1], [0, 1, 1]])
    dataset_index = torch.tensor([3, 4])
    A = torch.zeros((2, 3, 3))
    A[0, 0, 1] = 1
    A[1, 0, 1:] = 1
    prob = torch.zeros_like(A)
    prob[0, 0, 1:] = 0.9
    prob[1, 0, 1] = 0.9
    mask = torch.zeros_like(A)
    mask[:, 0, 1:] = 1
    gt_division = torch.zeros_like(A, dtype=torch.bool)
    gt_division[1, 0, 1:] = True

    counts = TrackingLightningModule._dataset_assoc_counts_for_batch(
        A,
        prob,
        timepoints,
        mask,
        gt_division,
        dataset_index,
    )

    np.testing.assert_array_equal(counts[3], np.array([1, 2, 1, 0, 0, 0, 1, 0]))
    np.testing.assert_array_equal(counts[4], np.array([1, 2, 0, 0, 0, 1, 0, 1]))


def test_balanced_batch_sampler_iter_len_match_variable_batch():
    # with balance_batch_objects the realized batch count fluctuates below the
    # __len__ estimate; __iter__ must still yield exactly len(sampler) batches so
    # Lightning's epoch-end validation modulo fires (otherwise validation is
    # silently skipped every epoch).
    dataset = ConcatDataset([_SamplerDataset(40), _SamplerDataset(35)])
    sampler = BalancedBatchSampler(
        dataset,
        batch_size=8,
        n_pool=2,
        num_samples=64,
        balance_batch_objects=True,
    )
    for _ in range(5):
        assert len(list(sampler)) == len(sampler)


def test_balanced_distributed_sampler_supports_all_samples():
    dataset = ConcatDataset([_SamplerDataset(7), _SamplerDataset(6)])
    sampler = BalancedDistributedSampler(
        dataset,
        batch_size=4,
        n_pool=2,
        num_samples=None,
        num_replicas=2,
        rank=0,
    )

    assert len(list(sampler)) == len(sampler)


def test_sequence_input_specs_expand_supported_forms():
    specs = normalize_sequence_input_specs(
        [
            "ctc_single/01",
            {
                "format": "ctc",
                "spacing": [2, 1],
                "detection_folders": ["TRA", "SEG"],
                "paths": ["ctc/01", {"path": "ctc/02", "spacing": [1, 1]}],
            },
            {
                "path": "movie",
                "format": "geff",
                "sparse_gt": True,
                "spacing": "auto",
                "match_max_distance": 16,
            },
        ],
        ndim=2,
    )

    assert specs[0] == SequenceInputSpec(path=Path("ctc_single/01"), format="ctc")
    assert specs[1].path == Path("ctc/01")
    assert specs[1].format == "ctc"
    assert specs[1].spacing == (2.0, 1.0)
    assert specs[1].loader_kwargs == {"detection_folders": ["TRA", "SEG"]}
    assert specs[2].path == Path("ctc/02")
    assert specs[2].spacing == (1.0, 1.0)
    assert specs[2].loader_kwargs == {"detection_folders": ["TRA", "SEG"]}
    assert specs[3].path == Path("movie")
    assert specs[3].format == "geff"
    assert specs[3].sparse_gt is True
    assert specs[3].spacing == "auto"
    assert specs[3].loader_kwargs == {"match_max_distance": 16}


def test_sequence_input_specs_validate_ambiguous_or_incomplete_items():
    with pytest.raises(ValueError, match="exactly one of path/paths"):
        normalize_sequence_input_specs([{"format": "ctc"}])
    with pytest.raises(ValueError, match="GEFF input specs must set sparse_gt"):
        normalize_sequence_input_specs([{"path": "movie", "format": "geff"}])
    with pytest.raises(ValueError, match="length 3"):
        normalize_sequence_input_specs(
            [{"path": "ctc/01", "format": "ctc", "spacing": [1, 1]}],
            ndim=3,
        )
    with pytest.raises(ValueError, match='spacing="auto"'):
        normalize_sequence_input_specs(
            [{"path": "ctc/01", "format": "ctc", "spacing": "auto"}],
            ndim=2,
        )


def test_normalize_source_specs_returns_direct_loader_kwargs(tmp_path):
    ctc_root = tmp_path / "ctc"
    (ctc_root / "img").mkdir(parents=True)
    (ctc_root / "TRA").mkdir()
    geff_path = tmp_path / "track.geff"
    geff_path.mkdir()

    sources = normalize_source_specs(
        [
            {"path": ctc_root, "format": "auto", "detection_folders": ["TRA"]},
            {
                "path": geff_path,
                "format": "auto",
                "sparse_gt": True,
                "spacing": "auto",
                "match_max_distance": 16,
            },
        ],
        ndim=2,
        sequence_kwargs={
            "ndim": 2,
            "detection_folders": ["SEG"],
            "downscale_temporal": 1,
            "downscale_spatial": 1,
        },
        split_sequence_kwargs={"slice_pct": (0.0, 0.5)},
    )

    assert sources[0] == {
        "format": "ctc",
        "kwargs": {
            "root": ctc_root,
            "ndim": 2,
            "detection_folders": ["TRA"],
            "downscale_temporal": 1,
            "downscale_spatial": 1,
            "slice_pct": (0.0, 0.5),
        },
    }
    assert sources[1] == {
        "format": "geff",
        "kwargs": {
            "root_or_geff": geff_path,
            "sparse_gt": True,
            "match_max_distance": 16,
            "spacing": "auto",
        },
    }


def test_load_sequences_reports_all_bad_sources():
    def fail_a(root):
        raise ValueError(f"bad {root}")

    def fail_b(root_or_geff, sparse_gt):
        raise FileNotFoundError(f"missing {root_or_geff}, sparse_gt={sparse_gt}")

    with pytest.raises(SequenceLoadingError) as excinfo:
        load_sequences(
            [
                {"format": "ctc", "kwargs": {"root": "a"}},
                {"format": "geff", "kwargs": {"root_or_geff": "b", "sparse_gt": True}},
            ],
            loaders={"ctc": fail_a, "geff": fail_b},
        )

    message = str(excinfo.value)
    assert "2 source" in message
    assert "a\n  - bad a" in message
    assert "b\n  - missing b, sparse_gt=True" in message


def test_load_sequences_accepts_custom_loader_registry():
    loaded = load_sequences(
        [{"format": "custom", "kwargs": {"root": "movie", "channel": 1}}],
        loaders={"custom": lambda root, channel: (root, channel)},
    )

    assert loaded == (("movie", 1),)


def test_load_sequences_uses_joblib_cache(monkeypatch, tmp_path):
    cache_calls = []
    load_calls = []

    class CachedCallable:
        def __init__(self, func, ignore):
            self.func = func
            self.ignore = ignore

        def check_call_in_cache(self, **kwargs):
            cache_calls.append(("check", self.func.__name__, kwargs))
            return False

        def __call__(self, **kwargs):
            load_calls.append((self.func.__name__, kwargs))
            return self.func(**kwargs)

    class RecordingMemory:
        def __init__(self, cachedir, verbose):
            assert Path(cachedir) == tmp_path / "cache"
            assert verbose == 0

        def cache(self, func, ignore=None):
            cache_calls.append(("wrap", func.__name__, ignore))
            return CachedCallable(func, ignore)

    def ctc_loader(root, n_workers):
        return ("ctc", root, n_workers)

    def geff_loader(root_or_geff, sparse_gt):
        return ("geff", root_or_geff, sparse_gt)

    monkeypatch.setattr(training_api.joblib, "Memory", RecordingMemory)

    loaded = load_sequences(
        [
            {"format": "ctc", "kwargs": {"root": "ctc/01", "n_workers": 4}},
            {"format": "geff", "kwargs": {"root_or_geff": "movie", "sparse_gt": True}},
        ],
        loaders={"ctc": ctc_loader, "geff": geff_loader},
        cache_dir=tmp_path / "cache",
    )

    assert loaded == (
        ("ctc", "ctc/01", 4),
        ("geff", "movie", True),
    )
    assert cache_calls == [
        ("wrap", "ctc_loader", ["n_workers"]),
        ("check", "ctc_loader", {"root": "ctc/01", "n_workers": 4}),
        ("wrap", "geff_loader", None),
        ("check", "geff_loader", {"root_or_geff": "movie", "sparse_gt": True}),
    ]
    assert load_calls == [
        ("ctc_loader", {"root": "ctc/01", "n_workers": 4}),
        ("geff_loader", {"root_or_geff": "movie", "sparse_gt": True}),
    ]


def test_normalize_source_specs_reports_all_bad_sources(tmp_path):
    missing_auto = tmp_path / "missing_auto"

    with pytest.raises(SourceSpecError) as excinfo:
        normalize_source_specs(
            [
                {"path": tmp_path / "movie", "format": "geff"},
                {"path": tmp_path / "ctc", "format": "ctc", "spacing": "auto"},
                {"path": missing_auto, "format": "auto"},
            ],
            ndim=2,
            sequence_kwargs={"ndim": 2, "detection_folders": ["TRA"]},
        )

    message = str(excinfo.value)
    assert "3 source" in message
    assert "GEFF input specs must set sparse_gt" in message
    assert 'spacing="auto" is only valid for GEFF inputs' in message
    assert f"Could not auto-detect sequence format for {missing_auto}" in message


def test_build_dataset_uses_split_dataset_kwargs(monkeypatch):
    calls = []

    class RecordingDataset(_SamplerDataset):
        def __init__(self, sequence, dataset_index=0, **kwargs):
            super().__init__()
            self.root = sequence
            calls.append((sequence, dataset_index, kwargs))

    monkeypatch.setattr(training_api, "TrackingDataset", RecordingDataset)

    bundle = build_dataset(
        ["train_a", "train_b"],
        DataSplitConfig(
            split="train",
            sources=(),
            dataset_kwargs={
                "features": "wrfeat2",
                "augment": 3,
                "detect_drop": 0.25,
                "position_noise": 12.0,
            },
        ),
    )

    assert isinstance(bundle, TrackingDatasetBundle)
    assert isinstance(bundle.dataset, ConcatDataset)
    assert len(bundle) == 14
    assert calls == [
        (
            "train_a",
            0,
            {
                "features": "wrfeat2",
                "augment": 3,
                "detect_drop": 0.25,
                "position_noise": 12.0,
            },
        ),
        (
            "train_b",
            1,
            {
                "features": "wrfeat2",
                "augment": 3,
                "detect_drop": 0.25,
                "position_noise": 12.0,
            },
        ),
    ]


def test_dataset_bundle_builds_balanced_train_loader(monkeypatch):
    class RecordingDataset(_SamplerDataset):
        def __init__(self, sequence, dataset_index=0, **kwargs):
            super().__init__(n=sequence)
            self.root = sequence

    monkeypatch.setattr(training_api, "TrackingDataset", RecordingDataset)

    bundle = build_dataset(
        [5, 6],
        DataSplitConfig(
            split="train",
            sources=(),
            dataset_kwargs={"features": "wrfeat2"},
            sampler_kwargs={"batch_size": 3, "n_pool": 2, "num_samples": 8},
            loader_kwargs={"batch_size": 3, "num_workers": 0},
        ),
    )

    loader = bundle.dataloader()

    assert isinstance(loader.batch_sampler, BalancedBatchSampler)
    assert loader.batch_sampler.batch_size == 3


def test_build_model_validates_dataset_feature_dim(monkeypatch):
    class FeatureDataset(_SamplerDataset):
        def __init__(self, feat_dim):
            super().__init__()
            self.feat_dim = feat_dim

    monkeypatch.setattr(
        training_api,
        "TrackingTransformer",
        lambda **kwargs: kwargs,
    )
    dataset = ConcatDataset([FeatureDataset(6), FeatureDataset(6)])

    model = build_model(ModelConfig(feat_dim=6, d_model=32), dataset)

    assert model["feat_dim"] == 6
    assert model["d_model"] == 32
    with pytest.raises(ValueError, match="does not match"):
        build_model(ModelConfig(feat_dim=5), dataset)


def test_resolve_model_checkpoint_reference_handles_absolute_checkpoint(tmp_path):
    run = tmp_path / "runs" / "exp"
    checkpoint = run / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    (run / "config.yaml").write_text("{}\n")
    checkpoint.write_bytes(b"checkpoint")

    folder, checkpoint_path = resolve_model_checkpoint_reference(checkpoint)

    assert folder == run
    assert checkpoint_path == Path("checkpoints") / "last.ckpt"
    assert resolve_model_checkpoint_reference(run) == (run, None)


def test_load_model_from_path_delegates_folder_and_checkpoint(tmp_path):
    calls = []
    run = tmp_path / "runs" / "exp"
    checkpoint = run / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    (run / "config.yaml").write_text("{}\n")
    checkpoint.write_bytes(b"checkpoint")

    class FakeModel:
        @classmethod
        def from_folder(cls, folder, **kwargs):
            calls.append((folder, kwargs))
            return "loaded"

    loaded = load_model_from_path(
        checkpoint,
        args="args",
        map_location="cpu",
        model_cls=FakeModel,
    )

    assert loaded == "loaded"
    assert calls == [
        (
            run,
            {
                "args": "args",
                "map_location": "cpu",
                "checkpoint_path": Path("checkpoints") / "last.ckpt",
            },
        )
    ]


def test_trackastra_model_checkpoint_writes_configs_and_saves_best(tmp_path):
    saved_paths = []

    class FakeTrainer:
        is_global_zero = True
        sanity_checking = False

        def __init__(self, val_loss):
            self.logged_metrics = {"val_loss": val_loss}

    class FakeModel:
        def save(self, path):
            saved_paths.append(Path(path))

    class FakeModule:
        def __init__(self):
            self.inference_config = {"features": "wrfeat2"}
            self.model = FakeModel()

    callback = TrackastraModelCheckpoint(
        tmp_path,
        training_args={"features": "wrfeat2", "spatial_cutoff": 12},
        monitor="val_loss",
    )
    module = FakeModule()

    callback.on_fit_start(FakeTrainer(1.0), module)
    callback.on_validation_end(FakeTrainer(1.0), module)
    callback.on_validation_end(FakeTrainer(2.0), module)
    callback.on_validation_end(FakeTrainer(0.5), module)

    assert yaml.safe_load((tmp_path / "train_config.yaml").read_text()) == {
        "features": "wrfeat2",
        "spatial_cutoff": 12,
    }
    assert yaml.safe_load((tmp_path / "inference_config.yaml").read_text()) == {
        "features": "wrfeat2",
    }
    assert saved_paths == [tmp_path, tmp_path]


def test_build_trainer_facade_fits_domain_datasets(monkeypatch, tmp_path):
    created_modules = []
    created_trainers = []

    class FakeLightningModule:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_modules.append(self)

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_args = None
            created_trainers.append(self)

        def fit(self, *args, **kwargs):
            self.fit_args = (args, kwargs)
            return "fit-result"

    class FakeBundle:
        def __init__(self, name):
            self.name = name
            self.calls = []

        def dataloader(self, **kwargs):
            self.calls.append(kwargs)
            return f"{self.name}-loader"

    monkeypatch.setattr(
        training_api,
        "_tracking_lightning_module_class",
        lambda: FakeLightningModule,
    )
    monkeypatch.setattr(training_api, "_lightning_trainer_class", lambda: FakeTrainer)

    facade = build_trainer(
        TrainConfig(
            epochs=3,
            warmup_epochs=1,
            learning_rate=0.01,
            distributed=True,
            logger="none",
            outdir=tmp_path,
            mixed_precision=False,
            loss_kwargs={"delta_cutoff": 1, "causal_norm": "none"},
            tracking_kwargs={"tracking_frequency": 0},
            runtime_kwargs={
                "lightning_runtime": LightningTrainerRuntime(
                    logdir=tmp_path,
                    logger=False,
                    callbacks=["callback"],
                    profiler="profiler",
                    run_name="run",
                ),
                "trainer_kwargs": {"gradient_clip_val": 0.5},
            },
        )
    )
    train_bundle = FakeBundle("train")
    val_bundle = FakeBundle("val")

    run = facade.fit("model", train_bundle, val_bundle, ckpt_path="last.ckpt")

    assert isinstance(facade, TrackastraTrainer)
    assert run.result == "fit-result"
    assert created_modules[0].kwargs == {
        "model": "model",
        "warmup_epochs": 1,
        "max_epochs": 3,
        "learning_rate": 0.01,
        "compile": False,
        "delta_cutoff": 1,
        "causal_norm": "none",
        "tracking_frequency": 0,
    }
    assert created_trainers[0].kwargs["max_epochs"] == 3
    assert created_trainers[0].kwargs["logger"] is False
    assert created_trainers[0].kwargs["gradient_clip_val"] == 0.5
    assert created_trainers[0].kwargs["callbacks"] == ["callback"]
    assert created_trainers[0].kwargs["profiler"] == "profiler"
    assert created_trainers[0].kwargs["default_root_dir"] == tmp_path
    assert train_bundle.calls == [{"distributed": True}]
    assert val_bundle.calls == [{}]
    assert created_trainers[0].fit_args == (
        (created_modules[0],),
        {
            "train_dataloaders": "train-loader",
            "val_dataloaders": "val-loader",
            "ckpt_path": "last.ckpt",
        },
    )


def test_fit_injects_node_degree_weights_when_node_loss_positive(monkeypatch, tmp_path):
    created_modules = []

    class FakeLightningModule:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_modules.append(self)

    class FakeTrainer:
        def __init__(self, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return "fit-result"

    class FakeSubDataset:
        node_in_degree_counts = np.array([4, 400], dtype=np.int64)
        node_out_degree_counts = np.array([8, 400, 4], dtype=np.int64)

    class FakeBundle:
        def __init__(self):
            self.datasets = [FakeSubDataset()]

        def dataloader(self, **kwargs):
            return "loader"

    monkeypatch.setattr(
        training_api, "_tracking_lightning_module_class", lambda: FakeLightningModule
    )
    monkeypatch.setattr(training_api, "_lightning_trainer_class", lambda: FakeTrainer)

    facade = build_trainer(
        TrainConfig(
            epochs=1,
            warmup_epochs=1,
            logger="none",
            outdir=tmp_path,
            mixed_precision=False,
            loss_kwargs={"delta_cutoff": 1, "causal_norm": "none", "node_loss": 0.5},
            tracking_kwargs={"tracking_frequency": 0},
            runtime_kwargs={
                "lightning_runtime": LightningTrainerRuntime(
                    logdir=tmp_path,
                    logger=False,
                    callbacks=[],
                    profiler=None,
                    run_name="run",
                )
            },
        )
    )
    facade.fit("model", FakeBundle(), FakeBundle())

    kwargs = created_modules[0].kwargs
    assert kwargs["node_loss"] == 0.5
    assert kwargs["node_in_weights"].shape == (2,)
    assert kwargs["node_out_weights"].shape == (3,)
    # rarer classes get more weight; each vector is mean-normalized to 1
    assert float(kwargs["node_in_weights"][0]) > float(kwargs["node_in_weights"][1])
    assert abs(float(kwargs["node_out_weights"].mean()) - 1.0) < 1e-5


def test_trainer_fit_returns_without_lightning_when_epochs_zero(monkeypatch):
    created_modules = []
    created_trainers = []

    class FakeBundle:
        def dataloader(self, **kwargs):
            raise AssertionError("dataloader should not be built for epochs=0")

    monkeypatch.setattr(
        training_api,
        "_tracking_lightning_module_class",
        lambda: created_modules.append,
    )
    monkeypatch.setattr(
        training_api, "_lightning_trainer_class", lambda: created_trainers.append
    )

    facade = build_trainer(TrainConfig(epochs=0, logger="none"))

    run = facade.fit("model", FakeBundle(), FakeBundle())

    assert run.trainer is None
    assert run.lightning_module is None
    assert run.result is None
    assert created_modules == []
    assert created_trainers == []


def test_lightning_runtime_builds_callbacks(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_api, "git_commit", lambda: "abc123")

    runtime = build_lightning_runtime(
        dry=False,
        timestamp=False,
        name="run",
        outdir=tmp_path,
        resume=True,
        logger_name="none",
        wandb_project="proj",
        profile=False,
        training_args={"features": "wrfeat2"},
    )

    assert isinstance(runtime, LightningTrainerRuntime)
    assert runtime.logdir == tmp_path / "run"
    assert runtime.logger is False
    assert runtime.profiler is None
    assert runtime.run_name == "run"
    assert {type(callback).__name__ for callback in runtime.callbacks} >= {
        "ModelCheckpoint",
        "TrackastraModelCheckpoint",
        "EpochMetricsCSV",
        "Timer",
        "PreciseProgressBar",
    }


def test_epoch_metrics_csv_writes_one_upserted_row_per_epoch(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    trainer = SimpleNamespace(
        is_global_zero=True,
        sanity_checking=False,
        current_epoch=0,
        global_step=4,
        callback_metrics={
            "train_loss": torch.tensor(9.0),
            "train_loss_step": torch.tensor(8.0),
            "train_loss_epoch": torch.tensor(0.5),
            "val_loss": torch.tensor(0.4),
            "not_scalar": torch.ones(2),
        },
        optimizers=[optimizer],
    )
    callback = EpochMetricsCSV(tmp_path)

    callback.on_train_epoch_end(trainer, None)
    trainer.global_step = 5
    trainer.callback_metrics["val_loss"] = torch.tensor(0.3)
    callback.on_train_epoch_end(trainer, None)
    trainer.current_epoch = 1
    trainer.global_step = 9
    callback.on_train_epoch_end(trainer, None)

    metrics = pd.read_csv(tmp_path / "metrics" / "metrics.csv")
    assert list(metrics["epoch"]) == [0, 1]
    assert list(metrics["step"]) == [5, 9]
    assert list(metrics["val_loss"]) == pytest.approx([0.3, 0.3])
    assert list(metrics["train_loss_epoch"]) == pytest.approx([0.5, 0.5])
    assert list(metrics["lr-AdamW"]) == pytest.approx([0.01, 0.01])
    assert "train_loss" not in metrics
    assert "train_loss_step" not in metrics
    assert "not_scalar" not in metrics


def test_configure_lightning_module_runtime_paths(tmp_path):
    module = SimpleNamespace()

    configure_lightning_module_runtime_paths(
        module,
        logdir=tmp_path,
        debug=True,
    )

    assert module.loss_spike_debug_dir == tmp_path / "debug" / "debug_loss_spikes"
    assert module.batch_provenance_path == tmp_path / "debug"
    assert module.viz_debug_dir == tmp_path / "debug" / "viz"
    assert module.dataset_metrics_path == tmp_path / "debug" / "dataset_metrics.csv"
    assert module.tracking_metrics_path == tmp_path / "metrics"

    configure_lightning_module_runtime_paths(
        module,
        logdir=None,
        debug=False,
    )

    assert module.loss_spike_debug_dir is None
    assert module.batch_provenance_path is None
    assert module.viz_debug_dir is None
    assert module.dataset_metrics_path is None
    assert module.tracking_metrics_path is None


def test_build_or_resume_lightning_module_uses_last_checkpoint(tmp_path):
    calls = []
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    class FakeModule:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        @classmethod
        def load_from_checkpoint(cls, path, **kwargs):
            calls.append(("load", path, kwargs))
            module = cls(**kwargs)
            module.loaded_path = path
            return module

    module = build_or_resume_lightning_module(
        FakeModule,
        {"model": "m", "learning_rate": 0.1},
        logdir=run,
        resume=True,
    )

    assert module.loaded_path == checkpoint
    assert calls == [
        ("init", {"model": "m", "learning_rate": 0.1}),
        ("load", checkpoint, {"model": "m", "learning_rate": 0.1}),
        ("init", {"model": "m", "learning_rate": 0.1}),
    ]


def test_resume_checkpoint_path_returns_standard_last_checkpoint(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert resume_checkpoint_path(logdir=run, resume=True) == checkpoint
    assert resume_checkpoint_path(logdir=run, resume=False) is None
    assert resume_checkpoint_path(logdir=None, resume=True) is None
    assert resume_checkpoint_path(logdir=tmp_path / "missing", resume=True) is None


def test_parse_training_config_splits_public_configs(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
ndim: 3
input_train:
- format: geff
  sparse_gt: true
  spacing: [4, 1, 1]
  match_max_distance: 5
  paths:
  - train_a
input_val:
- format: geff
  sparse_gt: true
  spacing: [4, 1, 1]
  match_max_distance: 5
  path: val_a
features: wrfeat2_no_intensity
batch_size: 3
detect_drop: 0.25
augment: 4
augment_details:
  jitter: 2.5
  drift: 7
  tilt: 5
  frame_jump: 4
  frame_jump_p: 0.25
tracking_frequency: 2
model: saved-model
normalize_diameter: 14
cache: true
cachedir: sequence-cache
""",
    )
    monkeypatch.setattr("sys.argv", ["train.py", "-c", str(config)])

    model_config, train_data_config, val_data_config, train_config = (
        parse_training_config()
    )

    assert model_config.coord_dim == 3
    assert model_config.feat_dim == 8
    assert model_config.model_path == Path("saved-model")
    assert train_data_config.sources[0]["kwargs"]["root_or_geff"] == Path("train_a")
    assert train_data_config.sources[0]["kwargs"]["spacing"] == (4.0, 1.0, 1.0)
    assert train_data_config.dataset_kwargs["detect_drop"] == 0.25
    assert train_data_config.dataset_kwargs["augment"] == 4
    assert train_data_config.dataset_kwargs["augment_details"] == {
        "jitter": 2.5,
        "drift": 7.0,
        "tilt": 5.0,
        "frame_jump": 4.0,
        "frame_jump_p": 0.25,
    }
    assert val_data_config.sources[0]["kwargs"]["root_or_geff"] == Path("val_a")
    assert val_data_config.dataset_kwargs["detect_drop"] == 0.0
    assert val_data_config.dataset_kwargs["augment_details"] is None
    assert val_data_config.dataset_kwargs["position_noise"] == 0.0
    assert train_config.batch_size == 3
    assert train_config.resume is False
    assert train_config.cache_dir == Path("sequence-cache")
    assert train_config.loss_kwargs["consistency_weight"] == 0.1
    assert train_config.tracking_kwargs["tracking_frequency"] == 2
    assert train_config.tracking_kwargs["tracking_inputs"] == []
    assert train_config.tracking_kwargs["inference_config"] == {
        "features": "wrfeat2_no_intensity",
        "normalize_diameter": 14,
        "pretrained_feats_model": None,
        "pretrained_feats_mode": None,
        "pretrained_feats_additional_props": None,
    }
    assert "feature_embed_mode" not in train_config.runtime_kwargs["training_args"]
    assert "feat_embed_per_dim" not in train_config.runtime_kwargs["training_args"]


def test_parse_training_config_rejects_model_with_resume(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "-c",
            str(config),
            "--model",
            "pretrained-model",
            "--resume",
            "t",
        ],
    )

    with pytest.raises(ValueError, match="--model/-m and --resume cannot") as error:
        parse_training_config()

    message = str(error.value)
    assert "--timestamp f" in message
    assert "checkpoints/last.ckpt" in message
    assert "architecture settings unchanged" in message


def test_balanced_datamodule_uses_split_kwargs(monkeypatch):
    sequence_calls = []
    dataset_calls = []

    class RecordingSequence:
        @classmethod
        def from_ctc(cls, root, **kwargs):
            sequence_calls.append((Path(root), kwargs))
            return Path(root)

    class RecordingTrackingData(_SamplerDataset):
        def __init__(self, sequence, **kwargs):
            super().__init__()
            self.root = sequence
            dataset_calls.append((sequence, kwargs))

    monkeypatch.setattr(distributed, "TrackingSequence", RecordingSequence)
    monkeypatch.setattr(distributed, "TrackingDataset", RecordingTrackingData)
    module = BalancedDataModule(
        input_train=["train"],
        input_val=["val"],
        cachedir=None,
        distributed=False,
        sequence_kwargs={"ndim": 2},
        tracking_data_kwargs={"features": "wrfeat2"},
        train_sequence_kwargs={"slice_pct": (0.0, 0.8)},
        val_sequence_kwargs={"slice_pct": (0.8, 1.0)},
        train_tracking_data_kwargs={
            "detect_drop": 0.5,
            "augment": 3,
        },
        val_tracking_data_kwargs={
            "detect_drop": 0.0,
            "augment": 0,
        },
        sampler_kwargs={"batch_size": 2, "n_pool": 2, "num_samples": 4},
        loader_kwargs={"batch_size": 2, "num_workers": 0},
    )

    module.prepare_data()
    assert sequence_calls == []

    module.setup("fit")

    assert sequence_calls == [
        (Path("train"), {"ndim": 2, "slice_pct": (0.0, 0.8)}),
        (Path("val"), {"ndim": 2, "slice_pct": (0.8, 1.0)}),
    ]

    assert dataset_calls[0] == (
        Path("train"),
        {
            "dataset_index": 0,
            "features": "wrfeat2",
            "detect_drop": 0.5,
            "augment": 3,
        },
    )
    assert dataset_calls[1] == (
        Path("val"),
        {
            "dataset_index": 0,
            "features": "wrfeat2",
            "detect_drop": 0.0,
            "augment": 0,
        },
    )


def test_balanced_datamodule_dispatches_ctc_and_geff(monkeypatch):
    sequence_calls = []

    class RecordingSequence:
        @classmethod
        def from_ctc(cls, root, **kwargs):
            sequence_calls.append(("ctc", Path(root), kwargs))
            return Path(root)

        @classmethod
        def from_geff(cls, root_or_geff, **kwargs):
            sequence_calls.append(("geff", Path(root_or_geff), kwargs))
            return Path(root_or_geff)

    class RecordingTrackingData(_SamplerDataset):
        def __init__(self, sequence, **kwargs):
            super().__init__()
            self.root = sequence

    monkeypatch.setattr(distributed, "TrackingSequence", RecordingSequence)
    monkeypatch.setattr(distributed, "TrackingDataset", RecordingTrackingData)
    module = BalancedDataModule(
        input_train=[
            {
                "path": "ctc/01",
                "format": "ctc",
                "spacing": [2, 1],
                "detection_folders": ["SEG"],
            }
        ],
        input_val=[
            {
                "path": "movie/track.geff",
                "format": "geff",
                "sparse_gt": True,
                "spacing": "auto",
                "match_max_distance": 16,
            }
        ],
        cachedir=None,
        distributed=False,
        sequence_kwargs={"ndim": 2},
        tracking_data_kwargs={"features": "wrfeat2"},
        sampler_kwargs={"batch_size": 2, "n_pool": 2, "num_samples": 4},
        loader_kwargs={"batch_size": 2, "num_workers": 0},
    )

    module.setup("fit")

    assert sequence_calls == [
        (
            "ctc",
            Path("ctc/01"),
            {"ndim": 2, "spacing": (2.0, 1.0), "detection_folders": ["SEG"]},
        ),
        (
            "geff",
            Path("movie/track.geff"),
            {"sparse_gt": True, "match_max_distance": 16, "spacing": "auto"},
        ),
    ]


def test_balanced_datamodule_auto_detects_geff_directory(monkeypatch, tmp_path):
    geff_root = tmp_path / "movie"
    (geff_root / "track.geff").mkdir(parents=True)
    sequence_calls = []

    class RecordingSequence:
        @classmethod
        def from_geff(cls, root_or_geff, **kwargs):
            sequence_calls.append((Path(root_or_geff), kwargs))
            return Path(root_or_geff)

    class RecordingTrackingData(_SamplerDataset):
        def __init__(self, sequence, **kwargs):
            super().__init__()
            self.root = sequence

    monkeypatch.setattr(distributed, "TrackingSequence", RecordingSequence)
    monkeypatch.setattr(distributed, "TrackingDataset", RecordingTrackingData)
    module = BalancedDataModule(
        input_train=[
            {
                "path": geff_root,
                "sparse_gt": True,
                "spacing": "auto",
                "match_max_distance": 16,
            }
        ],
        input_val=[
            {
                "path": geff_root,
                "sparse_gt": True,
                "spacing": "auto",
                "match_max_distance": 16,
            }
        ],
        cachedir=None,
        distributed=False,
        sequence_kwargs={"ndim": 2},
        tracking_data_kwargs={"features": "wrfeat2"},
        sampler_kwargs={"batch_size": 2, "n_pool": 2, "num_samples": 4},
        loader_kwargs={"batch_size": 2, "num_workers": 0},
    )

    module.setup("fit")

    assert sequence_calls == [
        (geff_root, {"sparse_gt": True, "match_max_distance": 16, "spacing": "auto"}),
        (geff_root, {"sparse_gt": True, "match_max_distance": 16, "spacing": "auto"}),
    ]


def test_balanced_datamodule_cache_wraps_concrete_loaders(monkeypatch, tmp_path):
    cache_calls = []
    load_calls = []

    class CachedCallable:
        def __init__(self, func, ignore):
            self.func = func
            self.ignore = ignore

        def check_call_in_cache(self, **kwargs):
            cache_calls.append(("check", self.func.__name__, kwargs))
            return False

        def __call__(self, **kwargs):
            load_calls.append((self.func.__name__, kwargs))
            return self.func(**kwargs)

    class RecordingMemory:
        def __init__(self, cachedir, verbose):
            assert Path(cachedir) == tmp_path / "cache"
            assert verbose == 0

        def cache(self, func, ignore=None):
            cache_calls.append(("wrap", func.__name__, ignore))
            return CachedCallable(func, ignore)

    class RecordingSequence:
        @classmethod
        def from_ctc(cls, root, **kwargs):
            return Path(root)

        @classmethod
        def from_geff(cls, root_or_geff, **kwargs):
            return Path(root_or_geff)

    monkeypatch.setattr(distributed.joblib, "Memory", RecordingMemory)
    monkeypatch.setattr(distributed, "TrackingSequence", RecordingSequence)
    module = BalancedDataModule(
        input_train=[{"path": "ctc/01", "format": "ctc"}],
        input_val=[
            {
                "path": "movie/track.geff",
                "format": "geff",
                "sparse_gt": True,
                "match_max_distance": 16,
            }
        ],
        cachedir=str(tmp_path / "cache"),
        distributed=False,
        sequence_kwargs={"ndim": 2, "n_workers": 4},
        tracking_data_kwargs={"features": "wrfeat2"},
        sampler_kwargs={"batch_size": 2, "n_pool": 2, "num_samples": 4},
        loader_kwargs={"batch_size": 2, "num_workers": 0},
    )

    module.prepare_data()

    assert cache_calls == [
        ("wrap", "from_ctc", ["n_workers"]),
        ("check", "from_ctc", {"root": Path("ctc/01"), "ndim": 2, "n_workers": 4}),
        ("wrap", "from_geff", None),
        (
            "check",
            "from_geff",
            {
                "root_or_geff": Path("movie/track.geff"),
                "sparse_gt": True,
                "match_max_distance": 16,
            },
        ),
    ]
    assert load_calls == [
        ("from_ctc", {"root": Path("ctc/01"), "ndim": 2, "n_workers": 4}),
        (
            "from_geff",
            {
                "root_or_geff": Path("movie/track.geff"),
                "sparse_gt": True,
                "match_max_distance": 16,
            },
        ),
    ]


def test_tracking_inputs_filter_geff_sources(tmp_path):
    ctc_root = tmp_path / "ctc"
    geff_root = tmp_path / "movie"

    assert tracking_inputs_from_sources(
        [
            {"format": "ctc", "kwargs": {"root": ctc_root}},
            {"format": "geff", "kwargs": {"root_or_geff": geff_root}},
        ],
        tracking_frequency=1,
    ) == [{"root": ctc_root, "spacing": None}]


def test_tracking_inputs_carry_spacing(tmp_path):
    ctc_root = tmp_path / "ctc"

    assert tracking_inputs_from_sources(
        [
            {
                "format": "ctc",
                "kwargs": {"root": ctc_root, "spacing": (1.625, 0.40625, 0.40625)},
            }
        ],
        tracking_frequency=1,
    ) == [{"root": ctc_root, "spacing": [1.625, 0.40625, 0.40625]}]


def download_gt_data(url: str, data_dir: str | Path):
    data_dir = Path(data_dir)

    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    filename = url.split("/")[-1]
    file_path = data_dir / filename

    if not file_path.exists():
        urllib.request.urlretrieve(url, file_path)

        # Unzip the data
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)


@pytest.fixture(scope="module")
def download_gt_example_ctc():
    url = "https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip"
    download_gt_data(url, ROOT_DIR / "scripts" / "data" / "ctc")


def test_train_dry_run(download_gt_example_ctc):
    os.chdir(ROOT_DIR / "scripts")
    cmd = (
        "python train.py"
        " --input_train data/ctc/Fluo-N2DH-GOWT1/01"
        " --input_val data/ctc/Fluo-N2DH-GOWT1/02"
        " --device cpu --dry --epochs 1"
        " --train_samples 2 --batch_size 2"
        " --num_decoder_layers 1 --num_decoder_layers 1"
        " --d_model 128 --num_workers 2"
        " --cachedir None"
        " --distributed False"
    )
    print(cmd)
    result = os.system(cmd)

    assert result == 0


def test_node_degree_class_weights_properties():
    # class 0 much rarer than class 1 -> weight[0] > weight[1]; mean normalized to 1.
    weights = node_degree_class_weights(np.array([4, 400], dtype=np.int64), 2)
    assert torch.isfinite(weights).all()
    assert weights.shape == (2,)
    assert abs(float(weights.mean()) - 1.0) < 1e-5
    assert float(weights[0]) > float(weights[1])
    # zero count stays finite (the 1 + sqrt(n) guard) and gets the largest weight.
    zero = node_degree_class_weights(np.array([0, 100, 25], dtype=np.int64), 3)
    assert torch.isfinite(zero).all()
    assert float(zero.argmax()) == 0


def test_node_degree_loss_masks_unobservable_and_censored():
    model = TrackingTransformer(
        coord_dim=2,
        feat_dim=4,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        node_head=True,
        attn_positional_bias="none",
    )
    lm = TrackingLightningModule(model, node_loss=1.0, delta_cutoff=2)

    torch.manual_seed(0)
    out_logits = torch.randn(1, 3, 3, requires_grad=True)
    in_logits = torch.randn(1, 3, 2, requires_grad=True)
    batch = {
        "node_out_degree": torch.tensor([[1, 1, 0]]),
        "node_in_degree": torch.tensor([[0, 1, 1]]),
    }
    succ = torch.ones(1, 3, dtype=torch.bool)
    pred = torch.ones(1, 3, dtype=torch.bool)
    # timepoints 0,1,2 -> mask_time[i,j] = 0 < t_j - t_i <= 2
    t = torch.tensor([[0, 1, 2]])
    dt = t.unsqueeze(1) - t.unsqueeze(2)
    mask_time = (dt > 0) & (dt <= 2)
    mask_invalid = torch.zeros(1, 3, 3, dtype=torch.bool)

    out_ce, in_ce, _ = lm._node_degree_loss(
        out_logits, in_logits, batch, succ, pred, mask_time, mask_invalid
    )
    # node 2 has no later node -> out-degree unobservable; node 0 has no earlier node
    # -> in-degree unobservable. Perturbing those masked logits must not move the loss.
    with torch.no_grad():
        out_logits[0, 2] += 100.0  # masked out node
        in_logits[0, 0] += 100.0  # masked in node
    out_ce2, in_ce2, _ = lm._node_degree_loss(
        out_logits, in_logits, batch, succ, pred, mask_time, mask_invalid
    )
    assert torch.allclose(out_ce, out_ce2)
    assert torch.allclose(in_ce, in_ce2)

    # censoring a node via the availability flag also removes it from the loss.
    pred_censored = pred.clone()
    pred_censored[0, 1] = False
    _out_ce, in_ce_censored, _ = lm._node_degree_loss(
        out_logits, in_logits, batch, succ, pred_censored, mask_time, mask_invalid
    )
    with torch.no_grad():
        in_logits[0, 1] += 100.0  # now-censored node
    _out_ce, in_ce_censored2, _ = lm._node_degree_loss(
        out_logits, in_logits, batch, succ, pred_censored, mask_time, mask_invalid
    )
    assert torch.allclose(in_ce_censored, in_ce_censored2)
