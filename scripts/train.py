import json
import os

import pandas as pd

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import torch.multiprocessing

# torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("medium")

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from timeit import default_timer

import configargparse
import git
import lightning as pl
import numpy as np
import psutil
import trackastra
import wandb
import yaml
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from skimage.morphology import binary_dilation, disk
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from trackastra.data import (
    # load_ctc_data_from_subfolders,
    collate_sequence_padding,
    densify_assoc,
)
from trackastra.data.distributed import BalancedDataModule
from trackastra.model import INFERENCE_CONFIG_KEYS, TrackingTransformer
from trackastra.utils import (
    blockwise_causal_log_prob_batched,
    blockwise_causal_norm,
    blockwise_causal_norm_batched,
    blockwise_sum,
    blockwise_sum_batched,
    normalize,
    random_label_cmap,
    render_label,
    seed,
    str2bool,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=FutureWarning)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
np.seterr(all="ignore")


def _process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def _apply_focal_weight(loss: torch.Tensor, gamma: float) -> torch.Tensor:
    """Apply binary focal modulation to an unreduced BCE loss."""
    if gamma == 0:
        return loss
    return (1 - torch.exp(-loss)).pow(gamma) * loss


_LOSS_SPIKE_DEBUG_THRESHOLD = 1.0
_LOSS_SPIKE_DEBUG_MIN_EPOCH = 10
_LOSS_SPIKE_DEBUG_TOPK = 64
_LOSS_SPIKE_DEBUG_MAX_PER_EPOCH = 8
_LOSS_SPIKE_DEBUG_MAX_TOTAL = 64
_GRAD_SPIKE_DEBUG_THRESHOLD = 100.0


def _debug_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x


def _gather_edge_values(
    x: torch.Tensor, b: torch.Tensor, i: torch.Tensor, j: torch.Tensor
):
    return x[b, i, j].detach().cpu()


def _loss_spike_debug_payload(
    batch: dict,
    out: dict,
    *,
    loss_value: float,
    epoch: int,
    global_step: int,
    batch_idx: int,
    stage: str,
    rank: int,
    causal_norm: str,
    delta_cutoff: int,
    max_distance: float,
    grad_norm_value: float | None = None,
    trigger: str = "loss",
) -> dict:
    with torch.no_grad():
        coords = batch["coords"]
        timepoints = batch["timepoints"]
        loss_matrix = out["loss_before_reduce"].detach().float()
        A_pred = out["A_pred"].detach().float()
        mask_valid = out["mask_valid"].bool()
        bsz, n, _ = loss_matrix.shape
        A = densify_assoc(batch["assoc_coo"], bsz, n, device=loss_matrix.device)

        if causal_norm != "none":
            prob = blockwise_causal_norm_batched(
                A_pred, timepoints, mode=causal_norm, mask_invalid=~mask_valid
            )
        else:
            prob = torch.sigmoid(A_pred)

        dt = timepoints.unsqueeze(1) - timepoints.unsqueeze(2)
        spatial_dist = torch.cdist(coords[:, :, 1:].float(), coords[:, :, 1:].float())
        positive_forward = (A > 0.5) & (dt > 0) & (dt <= delta_cutoff) & mask_valid
        impossible = positive_forward & (spatial_dist > max_distance)

        entries_per_sample = out["mask"].sum(dim=(1, 2)).float()
        sample_loss = loss_matrix.sum(dim=(1, 2)) / entries_per_sample.clamp_min(1)

        flat = loss_matrix.reshape(-1)
        k = min(_LOSS_SPIKE_DEBUG_TOPK, flat.numel())
        top_values, top_indices = torch.topk(flat, k=k)
        keep = top_values > 0
        top_values = top_values[keep]
        top_indices = top_indices[keep]
        top_b = top_indices // (n * n)
        rem = top_indices % (n * n)
        top_i = rem // n
        top_j = rem % n

        def edge_table(b, i, j):
            table = {
                "batch": b.detach().cpu(),
                "row": i.detach().cpu(),
                "col": j.detach().cpu(),
                "loss": _gather_edge_values(loss_matrix, b, i, j),
                "target": _gather_edge_values(A, b, i, j),
                "logit": _gather_edge_values(A_pred, b, i, j),
                "prob": _gather_edge_values(prob, b, i, j),
                "distance": _gather_edge_values(spatial_dist, b, i, j),
                "dt": _gather_edge_values(dt, b, i, j),
                "time_source": timepoints[b, i].detach().cpu(),
                "time_target": timepoints[b, j].detach().cpu(),
                "label_source": batch["labels"][b, i].detach().cpu(),
                "label_target": batch["labels"][b, j].detach().cpu(),
            }
            for key in ("window_index", "seg_index", "window_start"):
                if key in batch:
                    table[key] = batch[key][b].detach().cpu()
            return table

        impossible_idx = torch.nonzero(impossible, as_tuple=False)
        impossible_idx = impossible_idx[:_LOSS_SPIKE_DEBUG_TOPK]

        payload = {
            "meta": {
                "stage": stage,
                "epoch": epoch,
                "global_step": global_step,
                "batch_idx": batch_idx,
                "rank": rank,
                "loss": loss_value,
                "threshold": _LOSS_SPIKE_DEBUG_THRESHOLD,
                "grad_norm": grad_norm_value,
                "grad_threshold": _GRAD_SPIKE_DEBUG_THRESHOLD,
                "trigger": trigger,
                "max_distance": max_distance,
                "delta_cutoff": delta_cutoff,
                "causal_norm": causal_norm,
            },
            "summary": {
                "batch_size": int(bsz),
                "seq_len": int(n),
                "valid_edges": int(out["mask"].sum().item()),
                "positive_forward_edges": int(positive_forward.sum().item()),
                "impossible_positive_edges": int(impossible.sum().item()),
                "max_edge_loss": float(flat.max().item()),
                "min_prob": float(prob[mask_valid].min().item()),
                "max_prob": float(prob[mask_valid].max().item()),
                "positive_prob_lt_1e-6": int(
                    (positive_forward & (prob < 1e-6)).sum().item()
                ),
                "negative_prob_gt_1m1e-6": int(
                    ((A <= 0.5) & mask_valid & (prob > 1 - 1e-6)).sum().item()
                ),
            },
            "sample_loss": sample_loss.detach().cpu(),
            "batch": {k: _debug_cpu(v) for k, v in batch.items()},
            "top_edges": edge_table(top_b, top_i, top_j),
            "impossible_edges": edge_table(
                impossible_idx[:, 0], impossible_idx[:, 1], impossible_idx[:, 2]
            )
            if len(impossible_idx)
            else {},
        }
    return payload


def _reduce_decision_loss(
    loss: torch.Tensor,
    mask: torch.Tensor,
    dt: torch.Tensor,
    delta_cutoff: int,
) -> torch.Tensor:
    """Average candidate losses per association decision, then per sample."""
    if delta_cutoff < 1:
        return loss.sum() * 0

    decision_losses = []
    decision_valid = []
    mask = mask.bool()
    for delta in range(1, delta_cutoff + 1):
        candidate_mask = mask & (dt == delta)
        # For a fixed child and delta, rows are the candidate parents in one frame.
        decision_losses.append((loss * candidate_mask).sum(dim=1))
        decision_valid.append(candidate_mask.any(dim=1))

    decision_losses = torch.stack(decision_losses, dim=1)
    decision_valid = torch.stack(decision_valid, dim=1)
    decisions_per_sample = decision_valid.sum(dim=(1, 2))
    loss_per_sample = decision_losses.sum(dim=(1, 2)) / decisions_per_sample.clamp_min(
        1
    )

    sample_valid = decisions_per_sample > 0
    return (loss_per_sample * sample_valid).sum() / sample_valid.sum().clamp_min(1)


def _reduce_matrix_loss(
    loss: torch.Tensor,
    mask: torch.Tensor,
    eps: float = torch.finfo(torch.float16).eps,
) -> torch.Tensor:
    """Original reduction over all valid association-matrix entries."""
    entries_per_sample = mask.sum(dim=(1, 2))
    loss_per_sample = loss.sum(dim=(1, 2)) / (entries_per_sample + eps)
    prefactor = torch.pow(entries_per_sample, 0.2)
    return (loss_per_sample * prefactor / (prefactor.sum() + eps)).sum()


def _git_commit():
    """Returns the git commit hash of the current repository if it exists, otherwise None (for debugging purposes)."""
    logging.debug(f"Trackastra path: {Path(trackastra.__path__[0]).resolve()}")
    try:
        commit = str(git.Repo(Path(trackastra.__path__[0]).resolve().parent).commit())
    except:  # noqa: E722
        commit = "none"
    return commit


class WarmupCosineLRScheduler(LRScheduler):
    """A linear warmup + cosine lr scheduler."""

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        cosine_final: float = 0.001,
        last_epoch=-1,
    ):
        """Use cosine_final to switch on/off the cosine annealing.

        cosine_final=0 -> reduce to 0 at the end of training
        cosine_final=1 -> dont reduce at all.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.cosine_final = cosine_final
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        if self.last_epoch < self.warmup_epochs:
            # linear ramp
            initial = 1e-2
            factor = initial + (1 - initial) * self.last_epoch / self.warmup_epochs
        else:
            # cosine annealing
            epoch_rel = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs + 1
            )
            factor = (
                0.5 * (1 + np.cos(np.pi * epoch_rel)) * (1 - self.cosine_final)
                + self.cosine_final
            )

        logging.info(f"LRScheduler: relative lr factor {factor:.03f}")
        return [factor * base_lr for base_lr in self.base_lrs]


def log_tracking_metrics(
    model,
    input_paths: list[str],
    detection_folder: str,
    features: str,
    mode: str,
    max_distance: int,
    normalize_diameter: float | None = None,
):
    """Run full-movie CTC validation with the current transformer weights."""
    try:
        from scripts.predict import predict_and_evaluate
    except ImportError:
        from predict import predict_and_evaluate

    from trackastra.model import Trackastra

    device = next(model.parameters()).device.type
    tracking_model = Trackastra(
        transformer=model,
        inference_config={
            "features": features,
            "normalize_diameter": normalize_diameter,
        },
        device=device,
    )
    with TemporaryDirectory() as tmpdir:
        return predict_and_evaluate(
            model=tracking_model,
            input_paths=input_paths,
            detection_folder=detection_folder,
            outdir=Path(tmpdir),
            model_name="validation",
            mode=mode,
            max_distance=max_distance,
            normalize_diameter=normalize_diameter,
            overwrite=True,
            print_results=False,
            link_breakdown=True,
        )


def _is_tracking_epoch(current_epoch: int, frequency: int) -> bool:
    """Use human-readable epoch numbers: frequency 10 runs at 10, 20, ..."""
    return frequency > 0 and (current_epoch + 1) % frequency == 0


def _summarize_tracking_metrics(metrics) -> dict[str, float]:
    movies = metrics[metrics["movie"] != "Mean"]
    summary = {
        f"val_{name}": float(movies[name].mean()) for name in ("TRA", "AOGM", "DET")
    }
    # LNK saturates near 1; log the error (1 - LNK) so late-training gains stay visible
    summary["val_LNK_ERR"] = float(1.0 - movies["LNK"].mean())
    # post-solver division-link FP/FN/F1; nanmean so a division-free movie omits it
    for name in ("fn_div", "fp_div", "f1_div"):
        if name in movies and movies[name].notna().any():
            summary[f"val_track_{name}"] = float(np.nanmean(movies[name]))
    return summary


# define the LightningModule that contains the TrackingTransformer (to separate torch and lightning)
# this contains all the training/loss logic
class WrappedLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        learning_rate: float = 1e-5,
        causal_norm: str = "none",
        loss_norm: str = "matrix",
        focal_loss_gamma: float = 0.0,
        delta_cutoff: int = 2,
        tracking_frequency: int = -1,  # log TRA metrics every that epochs
        tracking_input_paths: list[str] | None = None,
        tracking_detection_folder: str = "TRA",
        tracking_features: str = "wrfeat",
        tracking_mode: str = "greedy",
        tracking_normalize_diameter: float | None = None,
        batch_val_tb_idx: int = 0,  # the batch index to visualize in tensorboard
        div_upweight: float = 20,
        grad_log_every_n_epochs: int = 10,
        log_edge_rates: bool = True,
    ):
        super().__init__()
        self.grad_log_every_n_epochs = grad_log_every_n_epochs
        # pre-solver association FP/FN by link type (regular vs division); accumulated
        # per epoch and pooled across DDP ranks at epoch end (see _log_edge_error_rates)
        self.log_edge_rates = log_edge_rates
        self._edge_counts: dict[str, dict] = {}

        self.model = model
        self.causal_norm = causal_norm
        if loss_norm not in ("matrix", "decision"):
            raise ValueError(f"Unknown loss_norm {loss_norm!r}")
        if focal_loss_gamma < 0:
            raise ValueError("focal_loss_gamma must be non-negative")
        self.loss_norm = loss_norm
        self.focal_loss_gamma = focal_loss_gamma
        self.delta_cutoff = delta_cutoff
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        # self.train_loss = []
        # self.val_loss = []
        self.batch_val_tb_idx = batch_val_tb_idx
        self.batch_val_tb = None
        # per-step (batch_size, seq_len) pairs, flushed to a 2d histogram each epoch
        self._bn_log: list[tuple[int, int]] = []

        self.lr = learning_rate
        self.tracking_frequency = tracking_frequency
        self.tracking_input_paths = tracking_input_paths or []
        self.tracking_detection_folder = tracking_detection_folder
        self.tracking_features = tracking_features
        self.tracking_mode = tracking_mode
        self.tracking_normalize_diameter = tracking_normalize_diameter
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.div_upweight = div_upweight
        self.loss_spike_debug_dir: Path | None = None
        self._loss_spike_debug_counts: dict[tuple[str, str, int], int] = {}
        self._loss_spike_debug_total = 0
        self._last_train_debug_context = None
        # Per-batch provenance log (which dataset each sample came from). Set by
        # the trainer to a directory; one human-readable CSV is written per epoch
        # and the index->root map is dumped once on first use.
        self.batch_provenance_path: Path | None = None
        self._batch_provenance_map: dict[int, str] = {}
        self._batch_provenance_map_written = False
        self._batch_provenance_epochs_started: set[int] = set()

    _EDGE_KEYS = ("fn", "fp", "fn_div", "fp_div")

    @staticmethod
    def _edge_error_counts(A, prob, timepoints, mask, gt_division):
        """Pre-solver association FN/FP counts on regular and division links.

        Proxy for tracking quality: thresholds the predicted association ``prob``
        against the GT assoc ``A`` on the in-window forward edges (``mask``), before
        candidate pruning and greedy/ILP solving. Returns raw numerator/denominator
        counts (not rates) so they pool cleanly across steps and DDP ranks.
        """
        m = mask.bool()
        gt_pos = (A > 0.5) & m
        pred_pos = (prob >= 0.5) & m
        # mark predicted edges leaving a predicted division (source out-degree >= 2)
        pred = pred_pos.to(A.dtype)
        row = blockwise_sum_batched(pred, timepoints, dim=-1, reduce="sum")
        col = blockwise_sum_batched(pred, timepoints, dim=-2, reduce="sum")
        pred_division = pred * (row + col) > 2

        fn = gt_pos & ~pred_pos
        fp = pred_pos & ~gt_pos
        gt_regular = gt_pos & ~gt_division
        pred_regular = pred_pos & ~pred_division
        return {
            "fn_num": (fn & gt_regular).sum().float(),
            "fn_den": gt_regular.sum().float(),
            "fp_num": (fp & pred_regular).sum().float(),
            "fp_den": pred_regular.sum().float(),
            "fn_div_num": (fn & gt_division).sum().float(),
            "fn_div_den": (gt_pos & gt_division).sum().float(),
            "fp_div_num": (fp & pred_division).sum().float(),
            "fp_div_den": (pred_pos & pred_division).sum().float(),
        }

    def _common_step(self, batch):
        feats = batch["features"]
        coords = batch["coords"]
        # association targets arrive as sparse COO; densify on-device (see collate)
        A = densify_assoc(
            batch["assoc_coo"], coords.shape[0], coords.shape[1], device=coords.device
        )
        timepoints = batch["timepoints"]
        padding_mask = batch["padding_mask"]
        padding_mask = padding_mask.bool()

        A_pred, scored_mask = self.model(coords, feats, padding_mask=padding_mask)
        # A_pred = output["assoc_matrix"]
        # remove inf values that might happen due to float16 numerics
        A_pred.clamp_(torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)

        mask_invalid = torch.logical_or(
            padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
        )

        A_pred[mask_invalid] = 0
        loss = _apply_focal_weight(self.criterion(A_pred, A), self.focal_loss_gamma)

        if self.causal_norm != "none":
            # BF16 rounds confident probabilities to exactly 0 or 1 before BCE,
            # which can produce incorrect or non-finite gradients. Evaluate the
            # probability-space loss in log-space and entirely in float32: the
            # log-softmax / log(1 - p) are computed without ever materializing p,
            # so a saturated edge can no longer underflow to log(0) and spike.
            with torch.autocast(device_type=A_pred.device.type, enabled=False):
                log_p, log_1mp = blockwise_causal_log_prob_batched(
                    A_pred.float(),
                    timepoints,
                    mode=self.causal_norm,
                    mask_invalid=mask_invalid,
                )
                At = A.float()
                soft_loss = _apply_focal_weight(
                    -(At * log_p + (1.0 - At) * log_1mp),
                    self.focal_loss_gamma,
                )
                loss = 0.01 * loss + soft_loss

        # Reweighting does not need gradients
        with torch.no_grad():
            block_sum1 = blockwise_sum_batched(A, timepoints, dim=-1, reduce="sum")
            block_sum2 = blockwise_sum_batched(A, timepoints, dim=-2, reduce="sum")
            block_sum = A * (block_sum1 + block_sum2)

            normal_tracks = block_sum == 2
            division_tracks = block_sum > 2

            # upweight normal (not starting or ending) tracks and division tracks
            loss_weight = 1 + 1.0 * normal_tracks + self.div_upweight * division_tracks

        loss = loss * loss_weight

        mask_valid = ~mask_invalid
        if scored_mask is not None:
            # Sparse head: only the kNN pairs carry a real logit; every other pair
            # is pinned to NO_EDGE_LOGIT and is structurally unpredictable. Drop
            # those from the loss (numerator and the normalisation count) so an
            # unrecoverable positive outside the neighbourhood cannot spike it.
            mask_valid = mask_valid & scored_mask
        dt = timepoints.unsqueeze(1) - timepoints.unsqueeze(2)
        mask_time = torch.logical_and(dt > 0, dt <= self.delta_cutoff)

        mask = mask_time * mask_valid
        mask = mask.float()

        loss_before_reduce = loss * mask
        if self.loss_norm == "decision":
            loss = _reduce_decision_loss(
                loss_before_reduce,
                mask,
                dt,
                delta_cutoff=self.delta_cutoff,
            )
        else:
            loss = _reduce_matrix_loss(loss_before_reduce, mask)

        edge_counts = None
        if self.log_edge_rates:
            with torch.no_grad():
                if self.causal_norm != "none":
                    prob = blockwise_causal_norm_batched(
                        A_pred.float(),
                        timepoints,
                        mode=self.causal_norm,
                        mask_invalid=mask_invalid,
                    )
                else:
                    prob = torch.sigmoid(A_pred.float())
                edge_counts = self._edge_error_counts(
                    A, prob, timepoints, mask, division_tracks
                )

        # print(padding_mask.float().mean())
        return dict(
            loss=loss,
            padding_fraction=padding_mask.float().mean(),
            loss_before_reduce=loss_before_reduce,
            A_pred=A_pred,
            mask=mask,
            mask_time=mask_time,
            mask_valid=mask_valid,
            edge_counts=edge_counts,
        )

    def _accumulate_edge_counts(self, stage, counts):
        if counts is None:
            return
        acc = self._edge_counts.setdefault(stage, {})
        for key, value in counts.items():
            acc[key] = acc[key] + value if key in acc else value

    def _log_edge_error_rates(self, stage):
        """Pool per-epoch num/den across DDP ranks and log {stage}_assoc_* rates.

        all_gather is called for every key on every rank (fixed order, zeros when a
        rank saw no such edge) so an epoch with no divisions on one rank cannot
        deadlock the collective; the rank-0 log skips keys whose pooled den is 0.
        """
        if not self.log_edge_rates:
            return
        acc = self._edge_counts.get(stage, {})
        zero = torch.zeros((), device=self.device)

        def _emit(name, value):
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                rank_zero_only=True,
                batch_size=1,
            )

        rates = {}
        for key in self._EDGE_KEYS:
            num = self.all_gather(acc.get(f"{key}_num", zero)).sum()
            den = self.all_gather(acc.get(f"{key}_den", zero)).sum()
            if den.item() == 0:
                continue
            rates[key] = float(num / den)
            _emit(f"{stage}_assoc_{key}", rates[key])
        # F1 from error rates: precision = 1 - fp rate, recall = 1 - fn rate.
        if "fn" in rates and "fp" in rates:
            _emit(f"{stage}_assoc_f1", self._error_rate_f1(rates["fn"], rates["fp"]))
        if "fn_div" in rates and "fp_div" in rates:
            _emit(f"{stage}_assoc_f1_div", self._error_rate_f1(rates["fn_div"], rates["fp_div"]))

    @staticmethod
    def _error_rate_f1(fn_rate, fp_rate):
        """F1 from the FN rate (1 - recall) and FP rate (1 - precision)."""
        recall, precision = 1.0 - fn_rate, 1.0 - fp_rate
        total = precision + recall
        if total > 0:
            return 2.0 * precision * recall / total
        if recall == recall and precision == precision:  # both finite (both 0)
            return 0.0
        return float("nan")

    def _spike_debug_dir(self) -> Path:
        if self.loss_spike_debug_dir is not None:
            return Path(self.loss_spike_debug_dir)
        root = getattr(self.trainer, "default_root_dir", None) or "."
        return Path(root) / "debug_loss_spikes"

    @staticmethod
    def _loss_tag(value: float) -> str:
        if not np.isfinite(value):
            return "nonfinite"
        return f"{value:.3g}".replace("+", "").replace("-", "m").replace(".", "p")

    def _save_spike_debug_batch(
        self,
        *,
        stage: str,
        batch: dict,
        batch_idx: int,
        out: dict,
        loss_value: float,
        trigger: str,
        grad_norm_value: float | None = None,
    ) -> None:
        key = (trigger, stage, int(self.current_epoch))
        count = self._loss_spike_debug_counts.get(key, 0)
        if (
            count >= _LOSS_SPIKE_DEBUG_MAX_PER_EPOCH
            or self._loss_spike_debug_total >= _LOSS_SPIKE_DEBUG_MAX_TOTAL
        ):
            return

        self._loss_spike_debug_counts[key] = count + 1
        self._loss_spike_debug_total += 1

        rank = int(getattr(self.trainer, "global_rank", 0))
        spike_dir = self._spike_debug_dir()
        spike_dir.mkdir(parents=True, exist_ok=True)
        path = spike_dir / (
            f"{trigger}_{stage}_ep{self.current_epoch:04d}_"
            f"gs{self.global_step:08d}_batch{batch_idx:05d}_rank{rank}_"
            f"loss{self._loss_tag(loss_value)}.pt"
        )
        payload = _loss_spike_debug_payload(
            batch,
            out,
            loss_value=loss_value,
            epoch=int(self.current_epoch),
            global_step=int(self.global_step),
            batch_idx=int(batch_idx),
            stage=stage,
            rank=rank,
            causal_norm=self.causal_norm,
            delta_cutoff=self.delta_cutoff,
            max_distance=float(self.model.config["max_distance"]),
            grad_norm_value=grad_norm_value,
            trigger=trigger,
        )
        torch.save(payload, path)
        msg = (
            f"Saved {trigger} spike debug batch to {path} "
            f"(loss={loss_value:.6g}"
        )
        if grad_norm_value is not None:
            msg += f", grad_norm={grad_norm_value:.6g}"
        msg += ")"
        logger.warning(msg)
        print(msg, flush=True)

    def _maybe_save_loss_spike_debug(
        self, stage: str, batch: dict, batch_idx: int, out: dict
    ) -> float:
        loss_value = float(out["loss"].detach().float().cpu())
        if self.loss_spike_debug_dir is None:
            return loss_value
        if np.isfinite(loss_value):
            if self.current_epoch < _LOSS_SPIKE_DEBUG_MIN_EPOCH:
                return loss_value
            if loss_value < _LOSS_SPIKE_DEBUG_THRESHOLD:
                return loss_value
        self._save_spike_debug_batch(
            stage=stage,
            batch=batch,
            batch_idx=batch_idx,
            out=out,
            loss_value=loss_value,
            trigger="loss",
        )
        return loss_value

    def _maybe_save_grad_spike_debug(self, grad_norm_value: float) -> None:
        if self.loss_spike_debug_dir is None:
            return
        if self.current_epoch < _LOSS_SPIKE_DEBUG_MIN_EPOCH:
            return
        if np.isfinite(grad_norm_value) and grad_norm_value < _GRAD_SPIKE_DEBUG_THRESHOLD:
            return
        if self._last_train_debug_context is None:
            return
        batch, batch_idx, out, loss_value = self._last_train_debug_context
        self._save_spike_debug_batch(
            stage="train",
            batch=batch,
            batch_idx=batch_idx,
            out=out,
            loss_value=loss_value,
            trigger="grad_norm",
            grad_norm_value=grad_norm_value,
        )

    def on_train_epoch_start(self):
        self._edge_counts["train"] = {}

    def on_validation_epoch_start(self):
        self._edge_counts["val"] = {}

    def checkpoint_path(self, logdir):
        path = Path(logdir) / "checkpoints" / "last.ckpt"
        if path.exists():
            return path
        else:
            return None

    def on_before_optimizer_step(self, optimizer):
        # Log the (pre-clip) gradient norm here, where gradients actually exist:
        # the backward pass has run by this hook. Compute it every step while the
        # temporary spike debugger is active so grad_norm_max > 100 can save the
        # offending batch before clipping; throttle only the W&B scalar logging.
        _norm = grad_norm(self.model, 2).get("grad_2.0_norm_total", 0)
        _norm_value = (
            float(_norm.detach().float().cpu())
            if torch.is_tensor(_norm)
            else float(_norm)
        )
        self._maybe_save_grad_spike_debug(_norm_value)
        if (
            self.grad_log_every_n_epochs > 0
            and self.current_epoch % self.grad_log_every_n_epochs == 0
        ):
            # Reduce per-epoch instead of logging every step: the raw per-step
            # (pre-clip) norm is very jagged. "grad_norm" is the epoch mean
            # (typical magnitude / trend), "grad_norm_max" the epoch max (spike
            # envelope) - two clean lines instead of a noisy per-step burst.
            # batch_size=1: weight each step equally in the epoch reduction (grad
            # norm is a model-level scalar, not per-sample) and avoid Lightning's
            # ambiguous batch-size inference under variable batch sizes
            self.log(
                "grad_norm",
                _norm,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=1,
            )
            self.log(
                "grad_norm_max",
                _norm,
                on_step=False,
                on_epoch=True,
                reduce_fx="max",
                sync_dist=True,
                batch_size=1,
            )
        self._last_train_debug_context = None

    def configure_optimizers(self):
        # eps=1e-6 (vs the 1e-8 default) damps AdamW's adaptive step lr/(sqrt(v)+eps)
        # only once gradients collapse near convergence, taming the late-training
        # kicks without the 1e-4 floor that throttled normal-regime learning.
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=1e-5, eps=1e-6
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=WarmupCosineLRScheduler(
                optimizer, self.warmup_epochs, self.max_epochs
            ),
        )

    def _write_batch_provenance_map(self) -> None:
        """Build the dataset_index -> source folder map and dump it once."""
        if self._batch_provenance_map_written or self.batch_provenance_path is None:
            return
        self._batch_provenance_map_written = True
        if int(getattr(self.trainer, "global_rank", 0)) != 0:
            return
        mapping: dict[int, str] = {}
        dm = getattr(self.trainer, "datamodule", None)
        train_ds = getattr(dm, "datasets", {}).get("train") if dm is not None else None
        for sub in getattr(train_ds, "datasets", []) or []:
            mapping[int(getattr(sub, "dataset_index", -1))] = str(
                getattr(sub, "root", "")
            )
        self._batch_provenance_map = mapping
        path = self.batch_provenance_path / "batch_provenance_index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping, indent=2))

    def _log_batch_provenance(self, batch: dict, batch_idx: int) -> None:
        """Append one human-readable CSV row per sample to the epoch's file."""
        if self.batch_provenance_path is None or "dataset_index" not in batch:
            return
        if int(getattr(self.trainer, "global_rank", 0)) != 0:
            return
        self._write_batch_provenance_map()
        epoch = int(self.current_epoch)
        dataset_index = batch["dataset_index"].detach().cpu().tolist()
        seg_index = batch["seg_index"].detach().cpu().tolist()
        window_index = batch["window_index"].detach().cpu().tolist()
        window_start = batch["window_start"].detach().cpu().tolist()

        df = pd.DataFrame(
            {
                "global_step": int(self.global_step),
                "batch_idx": int(batch_idx),
                "sample": range(len(dataset_index)),
                "dataset_index": dataset_index,
                "dataset_name": [
                    self._batch_provenance_map.get(i, "") for i in dataset_index
                ],
                "window_index": window_index,
                "seg_index": seg_index,
                "window_start": window_start,
            }
        )

        path = self.batch_provenance_path / f"batch_provenance_epoch{epoch}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = epoch not in self._batch_provenance_epochs_started
        self._batch_provenance_epochs_started.add(epoch)
        df.to_csv(path, mode="a", header=write_header, index=False)

    def training_step(self, batch, batch_idx):
        self._log_batch_provenance(batch, batch_idx)
        out = self._common_step(batch)
        loss = out["loss"]
        loss_value = self._maybe_save_loss_spike_debug("train", batch, batch_idx, out)
        self._last_train_debug_context = (batch, batch_idx, out, loss_value)
        if torch.isnan(loss):
            print("NaN loss, skipping")
            return None

        self._accumulate_edge_counts("train", out["edge_counts"])

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["coords"].shape[0],
        )

        # self.train_loss.append(loss)

        # accumulate (B, N) for the per-epoch 2d histogram (replaces the separate
        # batch_size / detections_per_sequence scalar logs -- same info, joint view)
        self._bn_log.append(
            (int(batch["coords"].shape[0]), int(batch["coords"].shape[1]))
        )
        self.log_dict(
            {
                # real (non-pad) detections summed over the batch
                "detections_per_batch": float((~batch["padding_mask"].bool()).sum()),
                "padding_fraction": out["padding_fraction"],
            },
            on_step=True,
            on_epoch=False,
            batch_size=batch["coords"].shape[0],
        )

        return loss

    def on_train_epoch_end(self):
        self._log_edge_error_rates("train")

        # flush the accumulated (B, N) pairs as a single wandb scatter plot
        bn = self._bn_log
        self._bn_log = []
        if (
            not bn
            or not isinstance(self.logger, WandbLogger)
            or not self.trainer.is_global_zero
            or self.trainer.sanity_checking
        ):
            return
        import wandb as _wandb

        table = _wandb.Table(
            data=[[n, b] for b, n in bn], columns=["seq_len_N", "batch_size_B"]
        )
        self.logger.experiment.log(
            {
                "train/batch_BN": _wandb.plot.scatter(
                    table,
                    "seq_len_N",
                    "batch_size_B",
                    title=f"B vs N (epoch {self.current_epoch})",
                )
            }
        )

    def validation_step(self, batch, batch_idx):
        out = self._common_step(batch)
        loss = out["loss"]
        self._maybe_save_loss_spike_debug("val", batch, batch_idx, out)
        if torch.isnan(loss):
            print("NaN loss, skipping")
            return None

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["coords"].shape[0],
        )

        self._accumulate_edge_counts("val", out["edge_counts"])

        # self.val_loss.append(loss)
        if batch_idx == self.batch_val_tb_idx:
            self.batch_val_tb = dict(batch=batch, out=out)

        return loss

    def on_validation_epoch_end(self):
        # skip if sanity checking
        if self.trainer.sanity_checking:
            return

        self._log_edge_error_rates("val")

        # val_loss = torch.stack(self.val_loss).mean()

        # if isinstance(self.logger, TensorBoardLogger):
        #     self.logger.experiment.add_scalars(
        #         "loss", {"val": val_loss}, self.current_epoch
        #     )
        # self.val_loss.clear()

        # Hack to make lightning progress bars with loss values persistent
        print(" ")

        if (
            _is_tracking_epoch(self.current_epoch, self.tracking_frequency)
            and self.trainer.is_global_zero
        ):
            try:
                metrics = log_tracking_metrics(
                    model=self.model,
                    input_paths=self.tracking_input_paths,
                    detection_folder=self.tracking_detection_folder,
                    features=self.tracking_features,
                    mode=self.tracking_mode,
                    max_distance=self.model.config["max_distance"],
                    normalize_diameter=self.tracking_normalize_diameter,
                )
                values = _summarize_tracking_metrics(metrics)
                msg = (
                    f"[epoch {self.current_epoch}] "
                    f"val_TRA={values['val_TRA']:.4f} "
                    f"val_AOGM={values['val_AOGM']:.4f} "
                    f"val_LNK_ERR={values['val_LNK_ERR']:.4f} "
                    f"val_DET={values['val_DET']:.4f}"
                )
                for name in ("fn_div", "fp_div", "f1_div"):
                    key = f"val_track_{name}"
                    if key in values:
                        msg += f" {key}={values[key]:.4f}"
                print(msg)
                self.log_dict(
                    values,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                    batch_size=1,
                )
            except Exception as e:
                logging.exception(f"Error logging tracking metrics: {e}")

        if self.batch_val_tb is not None:
            batch = self.batch_val_tb["batch"]
            out = self.batch_val_tb["out"]

            # First sample of the batch
            sample = 0
            bsz, n = batch["timepoints"].shape
            A_gt = densify_assoc(
                batch["assoc_coo"], bsz, n, device=batch["assoc_coo"].device
            )[sample]
            timepoints = batch["timepoints"][sample]
            A_pred = out["A_pred"][sample]
            loss_before_reduce = out["loss_before_reduce"][sample]

            if self.causal_norm != "none":
                A_pred = blockwise_causal_norm(
                    A_pred, timepoints, mode=self.causal_norm
                )
            else:
                A_pred = torch.sigmoid(A_pred)

            # create grid of timepoints for visualization
            time_grid = torch.diff(timepoints, append=timepoints[-1:]) != 0
            time_grid = time_grid.unsqueeze(0) + time_grid.unsqueeze(1)

            over = torch.stack((A_pred, A_gt, A_pred), 0)
            # add grid as blue background
            over[2, time_grid] += 0.2
            # over = over.unsqueeze(0)

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_image(
                    "assoc matrix", over, self.current_epoch, dataformats="CHW"
                )

                self.logger.experiment.add_image(
                    "loss",
                    loss_before_reduce.unsqueeze(0),
                    self.current_epoch,
                    dataformats="CHW",
                )
                self.logger.experiment.add_image(
                    "loss_mask",
                    out["mask"][sample].unsqueeze(0),
                    self.current_epoch,
                    dataformats="CHW",
                )

                # Log losses for different delta t's (in loss_val writer)
                dt = timepoints[None, :] - timepoints[:, None]
                for _, delta in enumerate(range(1, self.delta_cutoff + 1)):
                    loss_before_reduce[dt == delta].mean()
                    # if isinstance(self.logger, TensorBoardLogger):
                    #     self.logger.experiment.add_scalar(
                    #         f"val_loss/delta_t={delta}", lt, self.current_epoch
                    #     )

            elif isinstance(self.logger, WandbLogger):
                pass
                # wandb.log(
                #     {
                #         "images/assoc_matrix": wandb.Image(
                #             np.moveaxis(over.detach().cpu().numpy(), 0, -1), mode="RGB"
                #         ),
                #         "images/loss": wandb.Image(
                #             loss_before_reduce.unsqueeze(2).detach().cpu().numpy()
                #         ),
                #         "images/loss_mask": wandb.Image(
                #             out["mask"][sample].unsqueeze(2).detach().cpu().numpy()
                #         ),
                #     },
                #     step=self.current_epoch,
                # )
            elif self.logger is None:
                pass
            else:
                raise ValueError(f"Unknown logger {self.logger}")


class PreciseProgressBar(pl.pytorch.callbacks.TQDMProgressBar):
    """Progress bar that shows loss metrics with higher precision.

    Lightning passes float metrics to tqdm, which formats them with ``.3g``.
    By pre-formatting numeric values as strings here, tqdm leaves them as-is.
    """

    def __init__(self, precision: int = 8, **kwargs):
        super().__init__(**kwargs)
        self._precision = precision

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        for key, value in items.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                items[key] = f"{value:.{self._precision}f}"
        return items


class ExampleImages(pl.pytorch.callbacks.Callback):
    def __init__(self, n_samples=2, mode="overlay"):
        """Log example images to tensorboard.

        Args:
            n_samples (int, optional): _description_. Defaults to 5.
            mode (str, optional): grid for speed, overlay for beauty.
        """
        self._n_samples = n_samples
        self.mode = mode
        self.cmap = random_label_cmap()

    def on_train_start(self, trainer, pl_module):
        start = default_timer()

        if isinstance(trainer.train_dataloader.dataset, torch.utils.data.ConcatDataset):
            dataset = trainer.train_dataloader.dataset.datasets[0]
        else:
            dataset = trainer.train_dataloader.dataset

        for n in range(min(self._n_samples, len(dataset))):
            sample = dataset.__getitem__(n, return_dense=True)
            sample_img_normalized = np.clip(normalize(sample["img"].numpy()), 0, 1)
            assert sample_img_normalized.ndim == 3
            for i, (img, mask) in tqdm(
                enumerate(zip(sample_img_normalized, sample["mask"])),
                desc="Logging example images",
                leave=False,
            ):
                coords = sample["coords"][
                    sample["timepoints"] == sample["timepoints"].min() + i
                ]
                coords = coords[:, 1:].numpy().astype(int)
                points = np.zeros_like(img)
                points[coords[:, 0], coords[:, 1]] = 1

                points = binary_dilation(points, footprint=disk(3)).astype(float)

                if self.mode == "overlay":
                    # Overlay is pretty slow
                    overlay = render_label(
                        lbl=mask, img=img, cmap=self.cmap, normalize_img=False
                    )
                    overlay = torch.from_numpy(overlay[..., :3])

                    overlay = torch.maximum(
                        overlay, torch.as_tensor(points).unsqueeze(-1).expand(-1, -1, 3)
                    )

                    if isinstance(pl_module.logger, TensorBoardLogger):
                        pl_module.logger.experiment.add_image(
                            f"example_images/{n}_img",
                            overlay,
                            i,
                            dataformats="HWC",
                        )
                    elif isinstance(pl_module.logger, WandbLogger):
                        wandb.log(
                            {
                                f"example_images/{n}_img": wandb.Image(
                                    overlay.numpy(), mode="RGB"
                                )
                            },
                            # step=0,
                        )
                    else:
                        raise ValueError(f"Unknown logger {pl_module.logger}")

                elif self.mode == "grid":
                    img = torch.from_numpy(img).unsqueeze(0).expand(3, -1, -1)
                    mask = torch.from_numpy(self.cmap(mask)).moveaxis(-1, 0)[:3]
                    grid = torch.stack([img, mask], dim=0)
                    if isinstance(pl_module.logger, TensorBoardLogger):
                        pl_module.logger.experiment.add_images(
                            f"example_images/{n}_img",
                            grid,
                            i,
                            dataformats="NCHW",
                        )
                    if isinstance(pl_module.logger, WandbLogger):
                        raise NotImplementedError()
                else:
                    raise ValueError(f"Unknown mode {self.mode}")

        print(f"Logged example images in {(default_timer() - start):.1f} s")


# a modelcheckpoint that uses TrackingTransformer.save() to save the model
class MyModelCheckpoint(pl.pytorch.callbacks.Callback):
    def __init__(self, logdir, training_args: dict, monitor: str = "val_loss"):
        self._logdir = Path(logdir)
        self._monitor = monitor
        self._best = np.inf
        self._training_args = training_args

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            logging.info(f"using logdir {self._logdir}")
            self._logdir.mkdir(parents=True, exist_ok=True)
            with open(self._logdir / "train_config.yaml", "w") as f:
                yaml.safe_dump(self._training_args, f)
            inference_config = {
                k: self._training_args.get(k) for k in INFERENCE_CONFIG_KEYS
            }
            with open(self._logdir / "inference_config.yaml", "w") as f:
                yaml.safe_dump(inference_config, f)

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero and not trainer.sanity_checking:
            value = trainer.logged_metrics[self._monitor]
            if value < self._best:
                self._best = value
                logging.info(f"saved best model with {self._monitor}={value:.5f}")
                pl_module.model.save(self._logdir)


# def weight_matrix(coords: torch.Tensor, scale: float = 100):
#     D = torch.linalg.norm(coords.unsqueeze(1) - coords.unsqueeze(2), dim=-1)
#     weight = 1 + 10 * torch.exp(-(D**2) / 2 / scale**2).to(coords.device)
#     return weight


def create_run_name(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # name = f"{timestamp}_{args.name}_feats_{args.features}_pos_{args.attn_positional_bias}_causal_norm_{args.causal_norm}"
    if args.timestamp:
        name = f"{timestamp}_{args.name}"
    else:
        name = args.name
    return name


def _feature_dim(ndim: int, features: str) -> int:
    """Return the feature width produced by TrackingDataset."""
    if features == "wrfeat":
        return 7 if ndim == 2 else 12
    if features in ("wrfeat2", "wrfeat2_no_intensity"):
        if ndim != 2:
            raise ValueError(f"{features} currently supports only 2D data")
        return 6 if features == "wrfeat2" else 5
    raise ValueError(f"Unsupported feature mode {features!r} for {ndim}D data")


def _resolve_feature_embed_mode(features: str, requested: str | None) -> str:
    if requested is not None:
        return requested
    return "mlp" if features in ("wrfeat2", "wrfeat2_no_intensity") else "fourier"


def _skip_missing_input_folders(split: str, input_paths: list[str]) -> list[str]:
    """Warn about missing input folders and return existing paths."""
    existing = []
    missing = []
    for p in input_paths:
        if Path(p).exists():
            existing.append(p)
        else:
            missing.append(p)

    if missing:
        logger.warning(
            "could not find %d %s folders: %s",
            len(missing),
            split,
            ", ".join(missing),
        )

    if not existing:
        raise FileNotFoundError(
            f"No existing {split} input folders found after skipping missing inputs"
        )

    return existing


def find_val_batch(loader_val, n_gpus):
    # find the val batch with most divisions for vizualisation, which runs on GPU 0
    batches_val = tuple(
        tqdm(loader_val, desc="Scanning val batches for max divs", leave=False)
    )
    n_divs = []
    n_dets = 0
    for batch in batches_val[: len(batches_val) // n_gpus]:
        # works for any val batch size: sum divisions over every sample in the batch
        _n_divs = 0
        bsz, n = batch["timepoints"].shape
        assoc = densify_assoc(batch["assoc_coo"], bsz, n)
        for am, tp in zip(assoc, batch["timepoints"]):
            _n_divs += int((blockwise_sum(am, tp).max(dim=0)[0] == 2).sum())
        n_divs.append(_n_divs)
        _n_dets = int((batch["timepoints"] >= 0).sum())
        n_dets += _n_dets
        logger.debug(f"{_n_divs=}, {_n_dets=}")

    logger.info(
        f"Validation set division/detection ratio: {np.array(n_divs).sum() / n_dets}"
    )
    batch_val_tb_idx = np.argsort(n_divs)[-1]
    return batch_val_tb_idx


@rank_zero_only
def _init_wandb(project, name, config, save_dir):
    _ = wandb.init(project=project, name=name, config=config, dir=save_dir)


def train(args):
    args.seed = seed(args.seed)
    args.feature_embed_mode = _resolve_feature_embed_mode(
        args.features, getattr(args, "feature_embed_mode", None)
    )
    if args.model is None:
        logger.warning("Training from scratch, this is slow!\n")

    args.warmup_epochs = min(args.warmup_epochs, args.epochs)

    if args.delta_cutoff is None:
        args.delta_cutoff = args.window

    memory = _process_memory()

    if args.features in ("wrfeat", "wrfeat2", "wrfeat2_no_intensity") and (
        args.feat_embed_per_dim <= 1
    ):
        raise ValueError("For wrfeat modes, feat_embed_per_dim must be > 1 (e.g. 8)")

    callbacks = []
    if not args.dry:
        run_name = create_run_name(args)
        logdir = Path(args.outdir) / run_name
        # saving checkpoints in case training gets restarted
        callbacks.append(
            pl.pytorch.callbacks.ModelCheckpoint(
                dirpath=logdir / "checkpoints",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        )

        training_args = vars(args)
        training_args["git_commit"] = _git_commit()
        callbacks.append(MyModelCheckpoint(logdir, training_args, monitor="val_loss"))

        if args.logger == "tensorboard":
            train_logger = TensorBoardLogger(logdir, name="tb")
        elif args.logger == "wandb":
            train_logger = WandbLogger(
                name=run_name, project=args.wandb_project, save_dir=logdir
            )

            # init here to get an alert on job failure even before training
            _init_wandb(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                save_dir=logdir,
            )

        elif args.logger == "none":
            train_logger = False
        else:
            raise ValueError(f"Unknown logger {args.logger}")
    else:
        logdir = None
        train_logger = False

    if logdir is not None and logdir.exists() and not args.resume:
        raise ValueError(
            f'Logdir {logdir} exists, set "--resume t"  if you want to overwrite'
        )

    accelerator = (
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    n_gpus = (
        torch.cuda.device_count() if args.distributed and accelerator == "cuda" else 1
    )

    args.input_train = _skip_missing_input_folders("training", args.input_train)
    args.input_val = _skip_missing_input_folders("validation", args.input_val)

    sequence_kwargs = dict(
        ndim=args.ndim,
        detection_folders=args.detection_folders,
        downscale_temporal=args.downscale_temporal,
        downscale_spatial=args.downscale_spatial,
    )
    tracking_data_kwargs = dict(
        window_size=args.window,
        max_detections=args.max_detections,
        features=args.features,
        detect_drop_fraction=args.detect_drop_fraction,
        normalize_diameter=args.normalize_diameter,
    )
    sampler_kwargs = dict(
        batch_size=args.batch_size,
        n_pool=args.n_pool_sampler,
        num_samples=args.train_samples if args.train_samples > 0 else None,
        weight_by_ndivs=args.weight_by_ndivs,
        weight_by_dataset=args.weight_by_dataset,
        balance_batch_objects=args.balance_batch_objects,
        balance_pct=args.balance_pct,
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_padding,
    )
    datamodule = BalancedDataModule(
        input_train=args.input_train,
        input_val=args.input_val,
        cachedir=args.cachedir if args.cache else None,
        distributed=args.distributed,
        sequence_kwargs=sequence_kwargs,
        tracking_data_kwargs=tracking_data_kwargs,
        train_sequence_kwargs={"slice_pct": args.slice_pct_train},
        val_sequence_kwargs={"slice_pct": args.slice_pct_val},
        train_tracking_data_kwargs={
            "detect_drop": args.detect_drop,
            "augment": args.augment,
            "position_noise": args.max_distance,
        },
        val_tracking_data_kwargs={
            "detect_drop": 0.0,
            "augment": 0,
            "position_noise": 0.0,
        },
        sampler_kwargs=sampler_kwargs,
        loader_kwargs=loader_kwargs,
        association_distance_cutoffs={
            "max_distance": args.max_distance,
        },
        association_delta_cutoff=args.delta_cutoff,
    )
    if args.epochs == 0:
        datamodule.prepare_data()

    batch_val_tb_idx = 0

    if train_logger:
        callbacks.append(
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")
        )

    callbacks.append(pl.pytorch.callbacks.Timer(interval="epoch"))

    callbacks.append(PreciseProgressBar(precision=8))

    if args.example_images:
        callbacks.append(ExampleImages())

    # load the model if it was given
    if args.model is not None:
        fpath = Path(args.model)

        # allow for checkpoints to be loaded too
        if fpath.is_file():
            model = TrackingTransformer.from_folder(
                Path(*fpath.parts[:2]),
                args=args,
                checkpoint_path=Path(*fpath.parts[2:]),
            )
        else:
            model = TrackingTransformer.from_folder(fpath, args=args)
    else:
        model = TrackingTransformer(
            coord_dim=args.ndim,
            feat_dim=_feature_dim(args.ndim, args.features),
            d_model=args.d_model,
            pos_embed_per_dim=args.pos_embed_per_dim,
            feat_embed_per_dim=args.feat_embed_per_dim,
            feature_embed_mode=args.feature_embed_mode,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dropout=args.dropout,
            window=args.window,
            max_distance=args.max_distance,
            attn_positional_bias=args.attn_positional_bias,
            attn_positional_bias_n_spatial=args.attn_positional_bias_n_spatial,
            attn_dist_mode=args.attn_dist_mode,
            attn_mode=args.attn_mode,
            max_neighbors=args.max_neighbors,
            logit_norm=args.logit_norm,
            head_mode=args.head_mode,
            causal_norm=args.causal_norm,
            architecture_version=args.architecture_version,
            disable_abs_pos=args.disable_abs_pos,
            disable_input_norm=args.disable_input_norm,
        )

    model_lightning = WrappedLightningModule(
        model=model,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        delta_cutoff=args.delta_cutoff,
        causal_norm=args.causal_norm,
        loss_norm=args.loss_norm,
        focal_loss_gamma=args.focal_loss_gamma,
        tracking_frequency=args.tracking_frequency,
        tracking_input_paths=args.input_val,
        tracking_detection_folder=args.detection_folders[0],
        tracking_features=args.features,
        tracking_mode=args.tracking_mode,
        tracking_normalize_diameter=args.normalize_diameter,
        batch_val_tb_idx=batch_val_tb_idx,
        div_upweight=args.div_upweight,
        grad_log_every_n_epochs=args.grad_log_every_n_epochs,
    )

    # if logdir already exists and --resume option is set, load the last checkpoint (eg when continuing training after crash)
    if logdir is not None and logdir.exists() and args.resume:
        logging.info("logdir exists, loading last state of model")
        fpath = model_lightning.checkpoint_path(logdir)
        if fpath is not None:
            model_lightning = WrappedLightningModule.load_from_checkpoint(
                fpath,
                model=model,
                warmup_epochs=args.warmup_epochs,
                max_epochs=args.epochs,
                learning_rate=args.lr,
                delta_cutoff=args.delta_cutoff,
                causal_norm=args.causal_norm,
                loss_norm=args.loss_norm,
                focal_loss_gamma=args.focal_loss_gamma,
                tracking_frequency=args.tracking_frequency,
                tracking_input_paths=args.input_val,
                tracking_detection_folder=args.detection_folders[0],
                tracking_features=args.features,
                tracking_mode=args.tracking_mode,
                tracking_normalize_diameter=args.normalize_diameter,
                batch_val_tb_idx=batch_val_tb_idx,
                div_upweight=args.div_upweight,
                grad_log_every_n_epochs=args.grad_log_every_n_epochs,
            )
        else:
            logging.warning(f"No checkpoint found in {logdir}")

    if args.debug:
        # Gather spike dumps and per-batch provenance under <logdir>/debug.
        debug_root = (Path(logdir) if logdir is not None else Path(".")) / "debug"
        model_lightning.loss_spike_debug_dir = debug_root / "debug_loss_spikes"
        model_lightning.batch_provenance_path = debug_root
    else:
        model_lightning.loss_spike_debug_dir = None
        model_lightning.batch_provenance_path = None

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {num_params / 1e6:.1f}M parameters")

    if args.distributed:
        # strategy = "ddp"
        strategy = "ddp_find_unused_parameters_true"
    else:
        strategy = "auto"

    if args.profile:
        # profiler = AdvancedProfiler(dirpath=".", filename="profile")
        profiler = PyTorchProfiler(dirpath=".", filename="profile", skip_first=16)
    else:
        profiler = None

    trainer = pl.Trainer(
        accelerator=accelerator,
        strategy=strategy,
        devices=n_gpus,
        gradient_clip_val=1.0,
        precision="bf16-mixed" if args.mixedp else 32,
        logger=train_logger,
        default_root_dir=logdir if not args.dry else None,
        num_nodes=1,
        max_epochs=args.epochs,
        callbacks=callbacks,
        profiler=profiler,
    )

    t = default_timer()

    if logdir is not None and args.resume:
        resume_path = model_lightning.checkpoint_path(logdir)
        logger.info(f"Resuming from {resume_path}")
    else:
        resume_path = None

    if args.epochs > 0:
        trainer.fit(model_lightning, datamodule=datamodule, ckpt_path=resume_path)

    print(f"Time elapsed:     {(default_timer() - t) / 60:.02f} min")
    print(f"CPU Memory used:  {(_process_memory() - memory) / 1e9:.2f} GB")
    print(f"GPU Memory used : {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    return locals()


def parse_train_args():
    parser = configargparse.ArgumentParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        is_config_file=True,
        help="config file path",
        default="configs/vanvliet.yaml",
    )
    parser.add_argument("-o", "--outdir", type=str, default="runs")
    parser.add_argument("--name", type=str, help="Name to append to timestamp")
    parser.add_argument("--timestamp", type=str2bool, default=True)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="load this model at start (e.g. to continue training)",
    )
    parser.add_argument(
        "--ndim", type=int, default=2, help="number of spatial dimensions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="accelerator to train on ('auto' picks cuda when available)",
    )
    parser.add_argument("-d", "--d_model", type=int, default=256)
    parser.add_argument("-w", "--window", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument(
        "--detection_folders",
        type=str,
        nargs="+",
        default=["TRA"],
        help=(
            "Subfolders to search for detections. Defaults to `TRA`, which corresponds"
            " to using only the GT."
        ),
    )
    parser.add_argument("--input_train", type=str, nargs="+")
    parser.add_argument("--input_val", type=str, nargs="+")
    parser.add_argument("--slice_pct_train", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--slice_pct_val", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--downscale_temporal", type=int, default=1)
    parser.add_argument("--downscale_spatial", type=int, default=1)
    parser.add_argument("--max_distance", type=int, default=256)
    parser.add_argument("--normalize_diameter", type=float, default=None)
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--num_encoder_layers", type=int, default=6)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    parser.add_argument("--pos_embed_per_dim", type=int, default=32)
    parser.add_argument("--feat_embed_per_dim", type=int, default=8)
    parser.add_argument(
        "--feature_embed_mode",
        choices=("fourier", "mlp"),
        default=None,
        help="feature encoder; defaults to mlp for wrfeat2 modes and fourier otherwise",
    )
    parser.add_argument("--dropout", type=float, default=0.00)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_detections", type=int, default=None)
    parser.add_argument("--delta_cutoff", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--attn_positional_bias",
        type=str,
        choices=["rope", "none"],
        default="rope",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable loss/grad spike dumps and per-batch dataset provenance "
        "logging under <logdir>/debug.",
    )
    parser.add_argument("--attn_positional_bias_n_spatial", type=int, default=16)
    parser.add_argument("--attn_dist_mode", type=str, default="v1")
    parser.add_argument(
        "--attn_mode", type=str, choices=["dense", "sparse"], default="dense"
    )
    parser.add_argument("--max_neighbors", type=int, nargs="+", default=[16])
    parser.add_argument("--logit_norm", type=str2bool, default=True)
    parser.add_argument(
        "--head_mode",
        choices=["bilinear", "sparse_bilinear"],
        default=None,
        help="Association head; None auto-selects from --attn_mode.",
    )
    parser.add_argument(
        "--architecture_version",
        type=int,
        choices=(1, 2),
        default=2,
        help="model forward semantics; use 1 only for legacy-compatible training",
    )
    parser.add_argument(
        "--disable_abs_pos",
        action="store_true",
        help="omit input coordinate Fourier embeddings; attention RoPE is unchanged",
    )
    parser.add_argument(
        "--disable_input_norm",
        action="store_true",
        help="bypass the initial LayerNorm after input projection",
    )
    parser.add_argument("--mixedp", type=str2bool, default=True)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--features",
        type=str,
        choices=["wrfeat", "wrfeat2", "wrfeat2_no_intensity"],
        default="wrfeat",
    )
    parser.add_argument(
        "--causal_norm",
        type=str,
        choices=["none", "linear", "softmax", "quiet_softmax"],
        default="quiet_softmax",
    )
    parser.add_argument(
        "--loss_norm",
        choices=["matrix", "decision"],
        default="matrix",
        help="loss reduction: original matrix-entry mean or per-decision mean",
    )
    parser.add_argument(
        "--focal_loss_gamma",
        type=float,
        default=0.0,
        help="binary focal-loss exponent; 0 disables focal weighting",
    )
    parser.add_argument("--div_upweight", type=float, default=2)
    parser.add_argument(
        "--grad_log_every_n_epochs",
        type=int,
        default=10,
        help="compute/log the (expensive) full-model grad norm only every N epochs",
    )

    parser.add_argument("--augment", type=int, default=3)
    parser.add_argument(
        "--detect_drop",
        type=float,
        default=0.0,
        help="probability of applying detection dropout to a training window",
    )
    parser.add_argument(
        "--detect_drop_fraction",
        type=float,
        default=0.1,
        help="fraction of detections dropped when detection dropout is applied",
    )
    parser.add_argument(
        "--tracking_frequency",
        type=int,
        default=5,
        help="run full-movie validation every N epochs; <=0 disables it",
    )
    parser.add_argument(
        "--tracking_mode",
        choices=("greedy", "greedy_nodiv", "ilp"),
        default="greedy",
        help="linking mode used for full-movie validation",
    )
    parser.add_argument(
        "--cache",
        type=str2bool,
        default=False,
        help="cache CTCData to disk use (useful for large datasets)",
    )

    parser.add_argument(
        "--cachedir",
        type=str,
        default=".cache",
        help="cache dir for CTCData if --cache is set",
    )
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument(
        "--n_pool_sampler",
        type=int,
        default=8,
        help="pool size for balanced sampler (set to 1 to disable balancing)",
    )

    parser.add_argument(
        "--distributed",
        type=str2bool,
        default=False,
        help="use distributed DDP training",
    )
    parser.add_argument(
        "--example_images",
        type=str2bool,
        default=False,
        help="Log example images. Slow.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "none"],
    )
    parser.add_argument("--wandb_project", type=str, default="trackastra-new")
    parser.add_argument(
        "--weight_by_ndivs",
        type=str2bool,
        default=True,
        help="Oversample windows that contain divisions",
    )
    parser.add_argument(
        "--weight_by_dataset",
        type=str2bool,
        default=False,
        help=(
            "Inversely weight datasets by number of samples (to counter dataset size"
            " imbalance)"
        ),
    )
    parser.add_argument(
        "--balance_batch_objects",
        type=str2bool,
        default=False,
        help=(
            "Use a variable batch size so the total detections per batch stay"
            " roughly constant (batch_size becomes an upper cap). Equalizes GPU"
            " memory across differently sized data. Ignored under DDP."
        ),
    )
    parser.add_argument(
        "--balance_pct",
        type=float,
        default=50.0,
        help=(
            "Percentile of per-window detection counts used as the reference for"
            " --balance_batch_objects (budget = batch_size * n_ref)"
        ),
    )

    args, unknown_args = parser.parse_known_args()

    # Hack to allow for --input_test
    allowed_unknown = ["input_test"]
    if not set(a.split("=")[0].strip("-") for a in unknown_args).issubset(
        set(allowed_unknown)
    ):
        raise ValueError(f"Unknown args: {unknown_args}")

    # pprint(vars(args))

    # for backward compatibility
    # if args.attn_positional_bias == "True":
    #     args.attn_positional_bias = "bias"
    # elif args.attn_positional_bias == "False":
    #     args.attn_positional_bias = False

    if args.distributed and hasattr(sys, "ps1"):
        raise ValueError(
            "Distributed training does not work in interactive mode. Run as `python"
            " train.py`."
        )

    return args


if __name__ == "__main__":
    args = parse_train_args()

    # from torch.profiler import profile, record_function, ProfilerActivity
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     vars = train(args)
    vars = train(args)
