import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from scripts.train import (
    WrappedLightningModule,
    _apply_focal_weight,
    _feature_dim,
    _reduce_decision_loss,
    _reduce_matrix_loss,
    _resolve_feature_embed_mode,
    _summarize_tracking_metrics,
    parse_train_args,
)
from torch.utils.data import ConcatDataset, Dataset
from trackastra.data import distributed
from trackastra.data.distributed import (
    BalancedBatchSampler,
    BalancedDataModule,
    BalancedDistributedSampler,
)

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


@pytest.mark.parametrize("version", [1, 2])
def test_parse_architecture_version(monkeypatch, version):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "-c",
            str(ROOT_DIR / "scripts/configs/vanvliet.yaml"),
            "--crop_size",
            "128",
            "128",
            "--architecture_version",
            str(version),
        ],
    )

    args = parse_train_args()

    assert args.architecture_version == version


def test_summarize_tracking_metrics_includes_linking_and_detection():
    metrics = pd.DataFrame({
        "movie": ["01", "02", "Mean"],
        "TRA": [0.8, 1.0, 0.9],
        "AOGM": [4.0, 2.0, 3.0],
        "LNK": [0.7, 0.9, 0.8],
        "DET": [0.6, 1.0, 0.8],
    })

    assert _summarize_tracking_metrics(metrics) == {
        "val_TRA": pytest.approx(0.9),
        "val_AOGM": pytest.approx(3.0),
        "val_LNK": pytest.approx(0.8),
        "val_DET": pytest.approx(0.8),
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


def test_matrix_loss_preserves_original_reduction():
    timepoints = torch.tensor([[0, 1, -1], [0, 1, 1]])
    _, mask = _decision_mask(timepoints)
    pair_loss = torch.zeros((2, 3, 3))
    pair_loss[0, 0, 1] = 2
    pair_loss[1, 0, 1:] = 4
    eps = torch.finfo(torch.float16).eps
    counts = mask.sum(dim=(1, 2))
    per_sample = pair_loss.sum(dim=(1, 2)) / (counts + eps)
    weights = counts.pow(0.2)
    expected = (per_sample * weights / (weights.sum() + eps)).sum()

    loss = _reduce_matrix_loss(pair_loss, mask)

    assert loss.item() == pytest.approx(expected.item())


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
            return self.logits.unsqueeze(0) + 0

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
        "assoc_matrix": torch.zeros((1, 3, 3)),
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


def test_common_step_excludes_neighborhood_censored_associations():
    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, coords, features, padding_mask=None):
            n = coords.shape[1]
            return torch.zeros((len(coords), n, n)) + self.bias

    module = WrappedLightningModule(
        ZeroModel(), causal_norm="none", delta_cutoff=1
    )
    batch = {
        "features": torch.zeros((1, 3, 1)),
        "coords": torch.zeros((1, 3, 2)),
        "assoc_matrix": torch.zeros((1, 3, 3)),
        "timepoints": torch.tensor([[0, 1, 1]]),
        "padding_mask": torch.zeros((1, 3), dtype=torch.bool),
        "loss_mask": torch.tensor([[
            [True, False, True],
            [True, True, True],
            [True, True, True],
        ]]),
    }

    mask = module._common_step(batch)["mask"].bool()

    assert not mask[0, 0, 1]
    assert mask[0, 0, 2]


def test_balanced_batch_sampler_partial_batch():
    dataset = ConcatDataset([_SamplerDataset(7), _SamplerDataset(6)])
    sampler = BalancedBatchSampler(dataset, batch_size=4, n_pool=2, num_samples=10)

    batches = list(sampler)

    assert len(sampler) == len(batches) == 3
    assert sorted(len(batch) for batch in batches) == [2, 4, 4]


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
    calls = []

    class RecordingDataset(_SamplerDataset):
        def __init__(self, root, **kwargs):
            super().__init__()
            calls.append((Path(root), kwargs))

    monkeypatch.setattr(distributed, "CTCData", RecordingDataset)
    module = BalancedDataModule(
        input_train=["train"],
        input_val=["val"],
        cachedir=None,
        augment=3,
        distributed=False,
        dataset_kwargs={"features": "wrfeat"},
        train_dataset_kwargs={
            "slice_pct": (0.0, 0.8),
            "crop_size": (64, 64),
            "detect_drop": 0.5,
        },
        val_dataset_kwargs={
            "slice_pct": (0.8, 1.0),
            "crop_size": None,
            "detect_drop": 0.0,
        },
        sampler_kwargs={"batch_size": 2, "n_pool": 2, "num_samples": 4},
        loader_kwargs={"batch_size": 2, "num_workers": 0},
    )

    module.prepare_data()
    assert calls == []

    module.setup("fit")

    assert calls[0][1] == {
        "features": "wrfeat",
        "slice_pct": (0.0, 0.8),
        "crop_size": (64, 64),
        "detect_drop": 0.5,
        "augment": 3,
    }
    assert calls[1][1] == {
        "features": "wrfeat",
        "slice_pct": (0.8, 1.0),
        "crop_size": None,
        "detect_drop": 0.0,
        "augment": 0,
    }


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
