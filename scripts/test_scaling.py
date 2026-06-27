"""Probe a trained model's robustness to global spatial scale.

For each scale factor ``s`` the windowed region features of every dataset item
are scaled by ``s`` (so ``s=1`` reproduces the model's nominal input), the model
is run, and the association loss plus the pre-solver edge FN/FP/F1 rates are
reported. A model that is scale robust should keep a low loss and high F1 across
a wide range of ``s``.

The geometry scaling reuses the dataset's own ``scale_factor`` hook (the same
``wrfeat.scale_feature_geometry`` applied in ``TrackingDataset.__getitem__``), so
no augmenter wiring is needed: we just overwrite ``dataset.scale_factor``.

Example:
    python scripts/test_scaling.py --model runs_good/my_model \\
        -i data/vanvliet2/01 --scale 0.5 1 2 5 10
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import WrappedLightningModule

from trackastra.data import TrackingDataset, collate_sequence_padding
from trackastra.data.io import TrackingSequence
from trackastra.model import Trackastra

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model", type=Path, required=True, help="Trained model folder."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more CTC-like input movie folders (metrics pooled over all).",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Explicit scale factors to probe (relative to nominal input).",
    )
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def _to_device(batch: dict, device: str) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _run_loader(
    module: WrappedLightningModule,
    loader: DataLoader,
    device: str,
    loss_sum: float,
    n_samples: int,
    counts: dict[str, float],
) -> tuple[float, int]:
    """Accumulate loss + edge error counts from one loader into the running totals."""
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.no_grad():
            out = module._common_step(batch)
        bsz = batch["coords"].shape[0]
        loss_sum += float(out["loss"]) * bsz
        n_samples += bsz
        for key, val in (out["edge_counts"] or {}).items():
            counts[key] = counts.get(key, 0.0) + float(val)
    return loss_sum, n_samples


def evaluate_scale(
    module: WrappedLightningModule,
    loaders: list[tuple[DataLoader, float]],
    scale: float,
    device: str,
) -> dict[str, float]:
    """Set each dataset's scale, run all loaders, and pool loss + edge error counts."""
    loss_sum, n_samples = 0.0, 0
    counts: dict[str, float] = {}
    for loader, base_factor in loaders:
        loader.dataset.scale_factor = base_factor * scale
        loss_sum, n_samples = _run_loader(
            module, loader, device, loss_sum, n_samples, counts
        )

    def rate(num, den):
        return counts.get(num, 0.0) / counts[den] if counts.get(den, 0.0) else float("nan")

    fn_rate = rate("fn_num", "fn_den")
    fp_rate = rate("fp_num", "fp_den")
    recall, precision = 1.0 - fn_rate, 1.0 - fp_rate
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "loss": loss_sum / max(n_samples, 1),
        "fn": fn_rate,
        "fp": fp_rate,
        "f1": f1,
        # Raw windowed counts (edges are counted once per overlapping window, so
        # these are NOT unique-edge totals and overcount vs a post-solver metric).
        "fn_n": int(counts.get("fn_num", 0)),
        "fp_n": int(counts.get("fp_num", 0)),
        "fn_div_n": int(counts.get("fn_div_num", 0)),
        "fp_div_n": int(counts.get("fp_div_num", 0)),
    }


def run(args: argparse.Namespace) -> None:
    device = args.device
    model = Trackastra.from_folder(args.model, device=device)
    # Training-time hyperparameters live in the provenance dump, not the model.
    ta = yaml.load(open(Path(args.model) / "train_config.yaml"), Loader=yaml.FullLoader)
    transformer = model.transformer
    transformer.eval()

    # Reuse the training loss/metric logic without any training state.
    module = WrappedLightningModule(
        model=transformer,
        delta_cutoff=ta.get("delta_cutoff", ta.get("window", 4)),
        causal_norm=ta.get("causal_norm", "none"),
        loss_norm=ta.get("loss_norm", "matrix"),
        focal_loss_gamma=ta.get("focal_loss_gamma", 0.0),
        div_upweight=ta.get("div_upweight", 20),
        log_edge_rates=True,
    ).to(device)
    module.eval()

    # One loader per input movie. Each remembers the factor the model nominally
    # sees (per-movie diameter normalization, or 1.0); every probed scale
    # multiplies it so that s=1 reproduces the nominal input.
    loaders: list[tuple[DataLoader, float]] = []
    for path in args.input:
        sequence = TrackingSequence.from_ctc(
            path,
            ndim=ta.get("ndim", 2),
            detection_folders=ta.get("detection_folders", ("TRA",)),
        )
        dataset = TrackingDataset(
            sequence,
            window_size=ta["window"],
            features=ta["features"],
            augment=0,  # deterministic: no random augmentation
            max_detections=ta.get("max_detections"),
            normalize_diameter=ta.get("normalize_diameter"),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            collate_fn=collate_sequence_padding,
        )
        loaders.append((loader, dataset.scale_factor))
        logger.info(
            "Loaded %d windows from %s (base scale factor %.4g)",
            len(dataset),
            path,
            dataset.scale_factor,
        )

    rows = []
    for s in tqdm(args.scale, desc="scales"):
        rows.append({"scale": float(s), **evaluate_scale(module, loaders, float(s), device)})

    df = pd.DataFrame(rows).round(
        {"scale": 3, "loss": 6, "fn": 6, "fp": 6, "f1": 6}
    )
    print("\n" + df.to_markdown(index=False))


if __name__ == "__main__":
    run(parse_args())
