import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scripts.train import (
    WrappedLightningModule,
    _apply_focal_weight,
    _child_ce_loss_matrix,
    _feature_dim,
    _quiet_softmax_child_log_null,
    _reduce_decision_loss,
    _reduce_matrix_loss,
    _resolve_feature_embed_mode,
    _summarize_tracking_metrics,
    parse_train_args,
)
from torch.utils.data import ConcatDataset, Dataset
from trackastra.data import distributed
from trackastra.data.dataset import (
    TrackingDataset,
    collate_sequence_padding,
)
from trackastra.data.distributed import (
    BalancedBatchSampler,
    BalancedDataModule,
    BalancedDistributedSampler,
)
from trackastra.data.io import Segmentation, TrackingSequence
from trackastra.model import TrackingTransformer

# Mark all tests in this module as requiring training dependencies
pytestmark = pytest.mark.train

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_wrfeat2_feature_dim_is_2d_only():
    assert _feature_dim(2, "wrfeat2") == 6
    assert _feature_dim(2, "wrfeat2_no_intensity") == 5
    with pytest.raises(ValueError, match="only 2D"):
        _feature_dim(3, "wrfeat2")
    with pytest.raises(ValueError, match="only 2D"):
        _feature_dim(3, "wrfeat2_no_intensity")


def test_wrfeat2_defaults_to_mlp_feature_embedding():
    assert _resolve_feature_embed_mode("wrfeat2", None) == "mlp"
    assert _resolve_feature_embed_mode("wrfeat2_no_intensity", None) == "mlp"
    assert _resolve_feature_embed_mode("wrfeat", None) == "fourier"
    assert _resolve_feature_embed_mode("wrfeat2", "fourier") == "fourier"


@pytest.mark.parametrize(
    ("features", "width"),
    (("wrfeat", 7), ("wrfeat2", 6), ("wrfeat2_no_intensity", 5)),
)
def test_tracking_data_cpu_training_smoke(features, width):
    raw_features = {
        "equivalent_diameter_area": np.ones((2, 1), np.float32),
        "intensity_mean": np.ones((2, 1), np.float32),
        "inertia_tensor": np.tile(np.eye(2, dtype=np.float32).ravel(), (2, 1)),
        "border_dist": np.zeros((2, 1), np.float32),
    }
    seg = Segmentation(
        name="TRA",
        n_frames=2,
        coords=np.tile(np.array([[0, 0], [4, 4]], dtype=np.float32), (2, 1)),
        labels=np.array([1, 2, 1, 2]),
        timepoints=np.array([0, 0, 1, 1], dtype=np.int64),
        features={k: np.tile(v, (2, 1)) for k, v in raw_features.items()},
        track_indices=np.array([0, 1, 0, 1]),
    )
    sequence = TrackingSequence(
        root=Path("synthetic"),
        ndim=2,
        segmentations=(seg,),
        lineage_relation=np.eye(2, dtype=bool),
        lineage_parents=np.full(2, -1),
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
        feat_embed_per_dim=2,
        feature_embed_mode=_resolve_feature_embed_mode(features, None),
        dropout=0,
    )

    loss = WrappedLightningModule(model, causal_norm="none")._common_step(batch)["loss"]
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


def test_error_rate_f1():
    # recall = 1 - fn = 0.5, precision = 1 - fp = 0.8 -> F1 = 2*0.4/1.3
    assert WrappedLightningModule._error_rate_f1(0.5, 0.2) == pytest.approx(
        2 * 0.8 * 0.5 / (0.8 + 0.5)
    )
    # perfect: no FN, no FP -> F1 = 1
    assert WrappedLightningModule._error_rate_f1(0.0, 0.0) == pytest.approx(1.0)
    # total failure: all missed and all spurious -> precision=recall=0 -> F1 = 0
    assert WrappedLightningModule._error_rate_f1(1.0, 1.0) == 0.0
    # undefined component (NaN rate) -> NaN
    assert WrappedLightningModule._error_rate_f1(float("nan"), 0.2) != WrappedLightningModule._error_rate_f1(
        float("nan"), 0.2
    )


def test_edge_error_counts_for_division_links():
    from trackastra.utils import blockwise_sum_batched

    # t0: nodes 0,1 ; t1: nodes 2,3,4,5. GT: 0->2 (continuation), 1->{3,4} (division)
    tp = torch.tensor([[0, 0, 1, 1, 1, 1]])
    A = torch.zeros((1, 6, 6))
    A[0, 0, 2] = A[0, 1, 3] = A[0, 1, 4] = 1.0
    # prediction: keep 0->2 and 1->3, MISS 1->4 (FN div), add spurious 0->5 (FP);
    # 0->{2,5} makes node 0 a predicted division, so the FP lands in fp_div
    prob = torch.zeros((1, 6, 6))
    prob[0, 0, 2] = prob[0, 1, 3] = prob[0, 0, 5] = 0.9
    prob[0, 1, 4] = 0.1

    dt = tp.unsqueeze(1) - tp.unsqueeze(2)
    mask = ((dt > 0) & (dt <= 2)).float()
    b1 = blockwise_sum_batched(A, tp, dim=-1, reduce="sum")
    b2 = blockwise_sum_batched(A, tp, dim=-2, reduce="sum")
    block_sum = A * (b1 + b2)

    counts = WrappedLightningModule._edge_error_counts(
        A, prob, tp, mask, block_sum > 2
    )
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
        WrappedLightningModule(model, assoc_loss="child_ce", causal_norm="none")
    with pytest.raises(ValueError, match="loss_norm='decision'"):
        WrappedLightningModule(
            model,
            assoc_loss="child_ce",
            causal_norm="quiet_softmax",
            loss_norm="matrix",
        )


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
    module = WrappedLightningModule(
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
    }

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = module._common_step(batch)["loss"]
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.isfinite(model.logits.grad).all()
    assert model.logits.grad[0, 2] > 0.1


def test_balanced_batch_sampler_partial_batch():
    dataset = ConcatDataset([_SamplerDataset(7), _SamplerDataset(6)])
    sampler = BalancedBatchSampler(dataset, batch_size=4, n_pool=2, num_samples=10)

    batches = list(sampler)

    assert len(sampler) == len(batches) == 3
    assert sorted(len(batch) for batch in batches) == [2, 4, 4]


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
        tracking_data_kwargs={"features": "wrfeat"},
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
            "features": "wrfeat",
            "detect_drop": 0.5,
            "augment": 3,
        },
    )
    assert dataset_calls[1] == (
        Path("val"),
        {
            "features": "wrfeat",
            "detect_drop": 0.0,
            "augment": 0,
        },
    )


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
