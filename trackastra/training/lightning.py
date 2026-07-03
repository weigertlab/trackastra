"""Lightning integration for Trackastra training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import grad_norm

from trackastra.data import association_supervision_mask, densify_assoc
from trackastra.training.debug import (
    GRAD_SPIKE_DEBUG_THRESHOLD as _GRAD_SPIKE_DEBUG_THRESHOLD,
)
from trackastra.training.debug import (
    LOSS_SPIKE_DEBUG_MAX_PER_EPOCH as _LOSS_SPIKE_DEBUG_MAX_PER_EPOCH,
)
from trackastra.training.debug import (
    LOSS_SPIKE_DEBUG_MAX_TOTAL as _LOSS_SPIKE_DEBUG_MAX_TOTAL,
)
from trackastra.training.debug import (
    LOSS_SPIKE_DEBUG_MIN_EPOCH as _LOSS_SPIKE_DEBUG_MIN_EPOCH,
)
from trackastra.training.debug import (
    LOSS_SPIKE_DEBUG_THRESHOLD as _LOSS_SPIKE_DEBUG_THRESHOLD,
)
from trackastra.training.debug import (
    loss_spike_debug_payload as _loss_spike_debug_payload,
)
from trackastra.training.losses import (
    apply_focal_weight as _apply_focal_weight,
)
from trackastra.training.losses import (
    child_ce_loss_matrix as _child_ce_loss_matrix,
)
from trackastra.training.losses import (
    edge_count_key,
    edge_error_counts,
    error_rate_f1,
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
from trackastra.training.schedulers import WarmupCosineLRScheduler
from trackastra.training.tracking_validation import (
    is_tracking_epoch as _is_tracking_epoch,
)
from trackastra.training.tracking_validation import (
    summarize_tracking_metrics as _summarize_tracking_metrics,
)
from trackastra.utils import (
    blockwise_causal_log_prob_batched,
    blockwise_causal_norm,
    blockwise_causal_norm_batched,
    blockwise_sum_batched,
)

logger = logging.getLogger(__name__)

# define the LightningModule that contains the TrackingTransformer (to separate torch and lightning)
# this contains all the training/loss logic
class TrackingLightningModule(LightningModule):
    _EDGE_PLOT_THRESHOLDS = tuple(
        sorted({0.5, *[float(x) for x in np.linspace(0.2, 0.8, 10)]})
    )

    def __init__(
        self,
        model,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        learning_rate: float = 1e-5,
        causal_norm: str = "none",
        assoc_loss: str = "bce",
        loss_norm: str = "matrix",
        focal_loss_gamma: float = 0.0,
        delta_cutoff: int = 2,
        tracking_frequency: int = -1,  # log TRA metrics every that epochs
        tracking_input_paths: list[str] | None = None,
        tracking_detection_folder: str = "TRA",
        tracking_mode: str = "greedy",
        # The inference contract (features, normalize_diameter, pretrained_feats_*)
        # used for the in-training tracking eval. Same dict written to
        # inference_config.yaml, so validation matches the shipped model exactly.
        inference_config: dict | None = None,
        batch_val_tb_idx: int = 0,  # the batch index to visualize in tensorboard
        div_upweight: float = 20,
        grad_log_every_n_epochs: int = 10,
        log_edge_rates: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.grad_log_every_n_epochs = grad_log_every_n_epochs
        # pre-solver association FP/FN by link type (regular vs division); accumulated
        # per epoch and pooled across DDP ranks at epoch end (see _log_edge_error_rates)
        self.log_edge_rates = log_edge_rates
        self._edge_counts: dict[str, dict] = {}

        self.model = model
        self.causal_norm = causal_norm
        if assoc_loss not in ("bce", "child_ce"):
            raise ValueError(f"Unknown assoc_loss {assoc_loss!r}")
        if assoc_loss == "child_ce" and causal_norm != "quiet_softmax":
            raise ValueError("assoc_loss='child_ce' requires causal_norm='quiet_softmax'")
        if loss_norm not in ("matrix", "decision"):
            raise ValueError(f"Unknown loss_norm {loss_norm!r}")
        if assoc_loss == "child_ce" and loss_norm != "decision":
            raise ValueError("assoc_loss='child_ce' requires loss_norm='decision'")
        self.assoc_loss = assoc_loss
        if focal_loss_gamma < 0:
            raise ValueError("focal_loss_gamma must be non-negative")
        self.loss_norm = loss_norm
        self.focal_loss_gamma = focal_loss_gamma
        self.delta_cutoff = delta_cutoff
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.batch_val_tb_idx = batch_val_tb_idx
        self.batch_val_tb = None
        # per-step (batch_size, seq_len) pairs, flushed to a 2d histogram each epoch
        self._bn_log: list[tuple[int, int]] = []

        self.lr = learning_rate
        self.tracking_frequency = tracking_frequency
        self.tracking_input_paths = tracking_input_paths or []
        self.tracking_detection_folder = tracking_detection_folder
        self.tracking_mode = tracking_mode
        self.inference_config = inference_config or {}
        # Lazily built (rank 0, first tracking epoch) cache of weight-independent
        # per-movie validation data: features, windows, masks, and the GT graph.
        # Reused every tracking epoch so only prediction + tracking + scoring re-run.
        self._tracking_model = None
        self._tracking_cache: list[dict] | None = None
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.div_upweight = div_upweight
        self.loss_spike_debug_dir: Path | None = None
        self._loss_spike_debug_counts: dict[tuple[str, str, int], int] = {}
        self._loss_spike_debug_total = 0
        self._last_train_debug_context = None
        # Per-epoch feature-space viz of a few sampled windows (set by the trainer
        # to a directory under <logdir>/debug when --debug is on).
        self.viz_debug_dir: Path | None = None
        self._viz_n_per_epoch = 4
        self._viz_epoch_count = 0
        # Per-batch provenance log (which dataset each sample came from). Set by
        # the trainer to a directory; one human-readable CSV is written per epoch
        # and the index->root map is dumped once on first use.
        self.batch_provenance_path: Path | None = None
        self.tracking_metrics_path: Path | None = None
        self._batch_provenance_map: dict[int, str] = {}
        self._batch_provenance_map_written = False
        self._batch_provenance_epochs_started: set[int] = set()

    def _build_tracking_cache(self):
        """Build the per-movie validation-tracking cache once (rank 0).

        Everything here is weight-independent (disk reads, feature extraction,
        window construction, GT graph), so it is computed a single time and reused
        every tracking epoch; only prediction + tracking + scoring re-run.
        """
        from trackastra.data import load_ctc_images_masks
        from trackastra.model import Trackastra

        device = next(self.model.parameters()).device.type
        # Built lazily at first eval, so the model is already on its device. Uses
        # the full inference contract (same dict as inference_config.yaml), so the
        # eval matches the shipped model - including any pretrained-feature keys.
        self._tracking_model = Trackastra(
            transformer=self.model,
            inference_config=self.inference_config,
            device=device,
        )
        ndim = int(self.model.config.get("coord_dim", 2))
        cache: list[dict] = []
        used_names: set[str] = set()
        for index, root in enumerate(self.tracking_input_paths, start=1):
            root = Path(root)
            imgs, masks, _image_path, gt_path = load_ctc_images_masks(
                root, detection_folder=self.tracking_detection_folder, ndim=ndim
            )
            # normalize_imgs=False mirrors the original predict_and_evaluate call;
            # normalize_diameter falls through to inference_config inside _predict.
            features, windows = self._tracking_model._extract_features_windows(
                imgs,
                masks,
                normalize_imgs=False,
            )
            seq_dir = root.parent if root.name == "TRA" else root
            seq_name = seq_dir.name.removesuffix("_GT")
            dataset = seq_dir.parent.name.removesuffix("_GT")
            name = f"{dataset}_{seq_name}" if seq_name.isdigit() and dataset else seq_name
            if name in used_names:
                name = f"{name}_{index}"
            used_names.add(name)
            cache.append({
                "name": name,
                "features": features,
                "windows": windows,
                "masks": masks,
                "spatial_dim": masks.ndim - 1,
                # Store the path, not the loaded graph: traccuracy annotates the GT
                # graph in place during CTC scoring (ctc_node_errors/ctc_edge_errors),
                # so a cached graph is poisoned after the first eval. Reload per epoch
                # (cheap relative to feature extraction).
                "gt_path": gt_path,
            })
        self._tracking_cache = cache

    def _run_tracking_eval(self):
        """Per-epoch CTC validation: re-run only the weight-dependent prediction,
        tracking, and scoring against the cached movies (see _build_tracking_cache).
        """
        try:
            from scripts.predict import ctc_metrics_from_data, link_type_breakdown
        except ImportError:
            from predict import ctc_metrics_from_data, link_type_breakdown

        from traccuracy.loaders import load_ctc_data

        from trackastra.tracking import graph_to_ctc

        if self._tracking_cache is None:
            self._build_tracking_cache()

        tm = self._tracking_model
        max_distance = self.model.config["max_distance"]
        rows = []
        with TemporaryDirectory() as tmpdir:
            for movie in self._tracking_cache:
                predictions = tm._predict_from_windows(
                    movie["features"],
                    movie["windows"],
                    spatial_dim=movie["spatial_dim"],
                    batch_size=1,
                )
                graph = tm._track_from_predictions(
                    predictions, mode=self.tracking_mode, max_distance=max_distance
                )
                out = Path(tmpdir) / movie["name"]
                graph_to_ctc(graph, movie["masks"], outdir=out)
                # Both graphs loaded fresh each epoch: traccuracy mutates them
                # in place during scoring, so neither may be cached/reused.
                gt_data = load_ctc_data(str(movie["gt_path"]), run_checks=False)
                pred_data = load_ctc_data(str(out), run_checks=False)
                values, matched = ctc_metrics_from_data(
                    gt_data, pred_data, return_matched=True
                )
                rows.append({
                    "movie": movie["name"], **values, **link_type_breakdown(matched)
                })
        return pd.DataFrame(rows)

    def _write_tracking_metrics(self, metrics: pd.DataFrame, epoch: int) -> None:
        """Write per-movie tracking metrics for one validation epoch."""
        if self.tracking_metrics_path is None:
            return
        path = self.tracking_metrics_path / f"tracking_metrics_epoch{epoch:04d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out = metrics.copy()
        out.insert(0, "epoch", int(epoch))
        out.insert(1, "global_step", int(self.global_step))
        out.to_csv(path, index=False)

    _EDGE_KEYS = ("fn", "fp", "fn_div", "fp_div")

    @staticmethod
    def _edge_count_key(key: str, threshold_idx: int, part: str) -> str:
        return edge_count_key(key, threshold_idx, part)

    @staticmethod
    def _edge_error_counts(A, prob, timepoints, mask, gt_division, thresholds=None):
        """Pre-solver association FN/FP counts on regular and division links.

        Proxy for tracking quality: thresholds the predicted association ``prob``
        against the GT assoc ``A`` on the in-window forward edges (``mask``), before
        candidate pruning and greedy/ILP solving. Returns raw numerator/denominator
        counts (not rates) so they pool cleanly across steps and DDP ranks.
        """
        return edge_error_counts(A, prob, timepoints, mask, gt_division, thresholds)

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

        A_pred, neighbor_mask = self.model(coords, feats, padding_mask=padding_mask)
        # A_pred = output["assoc_matrix"]
        # remove inf values that might happen due to float16 numerics
        A_pred.clamp_(torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)

        mask_invalid = torch.logical_or(
            padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
        )

        A_pred[mask_invalid] = 0
        mask_valid = ~mask_invalid
        gt_predecessor_set_available = batch["gt_predecessor_set_available"].bool()
        gt_successor_set_available = batch["gt_successor_set_available"].bool()
        pair_gt_available = gt_successor_set_available.unsqueeze(
            2
        ) | gt_predecessor_set_available.unsqueeze(1)
        mask_valid = mask_valid & pair_gt_available
        loss_mask = association_supervision_mask(
            timepoints,
            gt_predecessor_set_available,
            gt_successor_set_available,
            delta_cutoff=self.delta_cutoff,
            padding_mask=padding_mask,
        )
        if neighbor_mask is not None:
            # Sparse head: only the kNN pairs carry a real logit; every other pair
            # is pinned to NO_EDGE_LOGIT and is structurally unpredictable. Drop
            # those from the loss (numerator and the normalisation count) so an
            # unrecoverable positive outside the neighbourhood cannot spike it.
            mask_valid = mask_valid & neighbor_mask
            loss_mask = loss_mask & neighbor_mask
        dt = timepoints.unsqueeze(1) - timepoints.unsqueeze(2)
        mask_time = torch.logical_and(dt > 0, dt <= self.delta_cutoff)
        mask = loss_mask.float()

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
                if self.assoc_loss == "child_ce":
                    log_p_null = _quiet_softmax_child_log_null(
                        A_pred.float(),
                        timepoints,
                        mask_invalid=mask_invalid,
                    )
                    soft_loss = _child_ce_loss_matrix(
                        log_p,
                        log_p_null,
                        At,
                        mask,
                        dt,
                        delta_cutoff=self.delta_cutoff,
                        focal_loss_gamma=self.focal_loss_gamma,
                    )
                else:
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
                    A,
                    prob,
                    timepoints,
                    mask,
                    division_tracks,
                    thresholds=self._EDGE_PLOT_THRESHOLDS,
                )

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

    def _pooled_edge_threshold_rates(self, stage: str) -> list[dict[str, float | str]]:
        """Return pooled threshold-sweep association error rates for plotting."""
        acc = self._edge_counts.get(stage, {})
        zero = torch.zeros((), device=self.device)
        rows = []

        def _rate(key: str, threshold_idx: int) -> float | None:
            num_key = self._edge_count_key(key, threshold_idx, "num")
            den_key = self._edge_count_key(key, threshold_idx, "den")
            num = self.all_gather(acc.get(num_key, zero)).sum()
            den = self.all_gather(acc.get(den_key, zero)).sum()
            if den.item() == 0:
                return None
            return float(num / den)

        for threshold_idx, threshold in enumerate(self._EDGE_PLOT_THRESHOLDS):
            for edge_type, fn_key, fp_key in (
                ("regular", "fn", "fp"),
                ("division", "fn_div", "fp_div"),
            ):
                fn_rate = _rate(fn_key, threshold_idx)
                fp_rate = _rate(fp_key, threshold_idx)
                if fn_rate is None or fp_rate is None:
                    continue
                rows.append(
                    {
                        "stage": stage,
                        "edge_type": edge_type,
                        "threshold": float(threshold),
                        "fn": fn_rate,
                        "fp": fp_rate,
                        "f1": self._error_rate_f1(fn_rate, fp_rate),
                    }
                )
        return rows

    def _log_assoc_error_plot(self) -> None:
        """Log a compact FN/FP threshold-sweep plot for train and validation."""
        if (
            not self.log_edge_rates
            or not isinstance(self.logger, WandbLogger)
            or self.trainer.sanity_checking
        ):
            return

        rows = []
        for stage in ("train", "val"):
            rows.extend(self._pooled_edge_threshold_rates(stage))
        if not self.trainer.is_global_zero or not rows:
            return

        import matplotlib.pyplot as plt
        import wandb as _wandb

        fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=150)
        colors = {"regular": "tab:blue", "division": "tab:red"}
        linestyles = {"train": "-", "val": "--"}
        markers = {"train": "o", "val": "s"}
        labels = {
            ("train", "regular"): "train non-div",
            ("train", "division"): "train div",
            ("val", "regular"): "val non-div",
            ("val", "division"): "val div",
        }

        for stage in ("train", "val"):
            for edge_type in ("regular", "division"):
                group = [
                    row
                    for row in rows
                    if row["stage"] == stage and row["edge_type"] == edge_type
                ]
                if not group:
                    continue
                group = sorted(group, key=lambda row: row["threshold"])
                xs = [row["fn"] for row in group]
                ys = [row["fp"] for row in group]
                current = [
                    row for row in group if abs(float(row["threshold"]) - 0.5) < 1e-12
                ]
                ax.plot(
                    xs,
                    ys,
                    color=colors[edge_type],
                    linestyle=linestyles[stage],
                    linewidth=1.0,
                    alpha=0.35,
                    label=None if current else labels[(stage, edge_type)],
                    zorder=1,
                )
                if current:
                    row = current[0]
                    ax.scatter(
                        row["fn"],
                        row["fp"],
                        color=colors[edge_type],
                        marker=markers[stage],
                        edgecolor="black",
                        linewidth=0.5,
                        s=42,
                        label=labels[(stage, edge_type)],
                        zorder=2,
                    )

        ax.set_xlabel("missed links (FN / GT links)")
        ax.set_ylabel("wrong proposed links (FP / predicted links)")
        ax.set_title(f"Association errors, epoch {self.current_epoch}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(alpha=0.2, linewidth=0.5)
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, frameon=False, fontsize=8)
        fig.tight_layout()
        self.logger.experiment.log({"assoc_errors": _wandb.Image(fig)})
        plt.close(fig)

    @staticmethod
    def _error_rate_f1(fn_rate, fp_rate):
        """F1 from the FN rate (1 - recall) and FP rate (1 - precision)."""
        return error_rate_f1(fn_rate, fp_rate)

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
        self._viz_epoch_count = 0

    def _maybe_save_window_viz(self, batch: dict) -> None:
        """Render a few sampled windows as feature-space ellipse images (--debug)."""
        if (
            self.viz_debug_dir is None
            or not self.trainer.is_global_zero
            or self._viz_epoch_count >= self._viz_n_per_epoch
        ):
            return
        try:
            from scripts.utils import save_window_debug_viz
        except ImportError:
            from utils import save_window_debug_viz

        bsz = batch["coords"].shape[0]
        take = min(self._viz_n_per_epoch - self._viz_epoch_count, bsz)
        idxs = list(range(take))
        names = None
        if "dataset_index" in batch:
            self._write_batch_provenance_map()
            ds_idx = batch["dataset_index"].detach().cpu().tolist()
            names = [
                Path(self._batch_provenance_map.get(ds_idx[i], "")).name for i in idxs
            ]
        try:
            save_window_debug_viz(
                batch,
                idxs,
                self.viz_debug_dir,
                mode=self.inference_config.get("features", "wrfeat2"),
                delta_cutoff=self.delta_cutoff,
                epoch=int(self.current_epoch),
                names=names,
            )
            self._viz_epoch_count += take
        except Exception as e:
            logging.exception(f"Error saving window debug viz: {e}")

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
        self._maybe_save_window_viz(batch)
        out = self._common_step(batch)
        loss = out["loss"]
        loss_value = self._maybe_save_loss_spike_debug("train", batch, batch_idx, out)
        self._last_train_debug_context = (batch, batch_idx, out, loss_value)
        if torch.isnan(loss):
            logger.warning("NaN train loss at epoch=%s batch_idx=%s, skipping", self.current_epoch, batch_idx)
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
            logger.warning("NaN validation loss at epoch=%s batch_idx=%s, skipping", self.current_epoch, batch_idx)
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

        if batch_idx == self.batch_val_tb_idx:
            self.batch_val_tb = dict(batch=batch, out=out)

        return loss

    def on_validation_epoch_end(self):
        # skip if sanity checking
        if self.trainer.sanity_checking:
            return

        self._log_edge_error_rates("val")
        self._log_assoc_error_plot()

        # Hack to make lightning progress bars with loss values persistent
        print(" ")

        if (
            _is_tracking_epoch(self.current_epoch, self.tracking_frequency)
            and self.trainer.is_global_zero
            and self.tracking_input_paths
        ):
            try:
                metrics = self._run_tracking_eval()
                self._write_tracking_metrics(metrics, epoch=int(self.current_epoch))
                logger.info(
                    "[epoch %s] per-movie tracking metrics:\n%s",
                    self.current_epoch,
                    metrics.to_markdown(index=False),
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
                logger.info(msg)
                self.log_dict(
                    values,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                    rank_zero_only=True,
                    batch_size=1,
                )
            except Exception:
                logger.exception("Error logging tracking metrics")

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

            elif isinstance(self.logger, WandbLogger):
                pass
            elif self.logger is None:
                pass
            else:
                raise ValueError(f"Unknown logger {self.logger}")

__all__ = ["TrackingLightningModule"]
