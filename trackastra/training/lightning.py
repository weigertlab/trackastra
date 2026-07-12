"""Lightning integration for Trackastra training."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

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
from trackastra.training.schedulers import WarmupCosineLRScheduler
from trackastra.training.tracking_validation import (
    is_tracking_epoch as _is_tracking_epoch,
)
from trackastra.training.tracking_validation import (
    summarize_lam_split_sweep as _summarize_lam_split_sweep,
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
    _TRACKING_LAM_SPLIT_SWEEP = (0.0, 0.1, 0.25, 0.5, 1.0)
    _EDGE_PLOT_THRESHOLDS = tuple(
        sorted({0.5, *[float(x) for x in np.linspace(0.2, 0.8, 10)]})
    )
    _ASSOC_TIME_BINS = 20

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
        tracking_inputs: list[Mapping[str, Any]] | None = None,
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
        node_loss: float = 0.0,
        consistency_weight: float = 0.1,
        node_in_weights=None,
        node_out_weights=None,
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["model", "node_in_weights", "node_out_weights"]
        )
        self.grad_log_every_n_epochs = grad_log_every_n_epochs
        # pre-solver association FP/FN by link type (regular vs division); accumulated
        # per epoch and pooled across DDP ranks at epoch end (see _log_edge_error_rates)
        self.log_edge_rates = log_edge_rates
        self._edge_counts: dict[str, dict] = {}
        self._assoc_time_counts: dict[str, dict[tuple[int, int, int], np.ndarray]] = {}
        self._dataset_assoc_counts: dict[str, dict[int, np.ndarray]] = {}

        self.model = model
        # Compile only the training-step forward path. Store the wrapper WITHOUT
        # registering it as a submodule (object.__setattr__ bypasses nn.Module's
        # registration): torch.compile's OptimizedModule shares parameters with
        # `model`, so registering it would duplicate every parameter into
        # state_dict (with an _orig_mod. prefix) and double-count them in
        # self.parameters()/the optimizer. self.model stays the sole owner of the
        # (shared) parameters, so checkpointing and optimization are unchanged.
        # Eval/tracking keeps using the raw self.model to avoid recompiles on the
        # varying shapes seen during inference.
        object.__setattr__(
            self, "_forward_model", torch.compile(model) if compile else model
        )
        self.causal_norm = causal_norm
        if assoc_loss not in ("bce", "child_ce"):
            raise ValueError(f"Unknown assoc_loss {assoc_loss!r}")
        if assoc_loss == "child_ce" and causal_norm != "quiet_softmax":
            raise ValueError(
                "assoc_loss='child_ce' requires causal_norm='quiet_softmax'"
            )
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
        self.tracking_inputs = list(tracking_inputs or [])
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
        # Auxiliary node in/out-degree loss. node_loss==0 disables it entirely (and
        # requires no node head on the model). The class-weight vectors are registered
        # as buffers so they follow the module to its device; None -> unweighted CE.
        self.node_loss = node_loss
        # Degree-consistency: pull the edge-implied out-degree toward the node head's
        # prediction (node head teaches edges). It is inactive without the node head.
        self.consistency_weight = consistency_weight if node_loss > 0 else 0.0
        if node_loss > 0 and not getattr(self.model, "node_head", False):
            raise ValueError("node_loss > 0 requires a model built with node_head=True")
        self.register_buffer(
            "node_in_weights",
            None
            if node_in_weights is None
            else torch.as_tensor(node_in_weights).float(),
        )
        self.register_buffer(
            "node_out_weights",
            None
            if node_out_weights is None
            else torch.as_tensor(node_out_weights).float(),
        )
        self._node_violation_warned = False
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
        self.dataset_metrics_path: Path | None = None
        self.tracking_metrics_path: Path | None = None
        self._batch_provenance_map: dict[int, str] = {}
        self._batch_provenance_map_written = False
        self._batch_provenance_epochs_started: set[int] = set()
        self._assoc_time_dataset_maps: dict[str, dict[int, str]] = {}

    def _build_tracking_cache(self):
        """Build the per-movie validation-tracking cache once (rank 0).

        Everything here is weight-independent (disk reads, feature extraction,
        window construction, GT graph), so it is computed a single time and reused
        every tracking epoch; only prediction + tracking + scoring re-run.
        """
        from traccuracy.loaders import load_ctc_data

        from trackastra.data import DetectionSequence, load_ctc_images_masks
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
        for index, inp in enumerate(self.tracking_inputs, start=1):
            root = Path(inp["root"])
            spacing = inp.get("spacing")
            imgs, masks, _image_path, gt_path = load_ctc_images_masks(
                root, detection_folder=self.tracking_detection_folder, ndim=ndim
            )
            # Go through the canonical detection path so spacing scales coordinates
            # and geometry exactly as TrackingSequence.from_ctc does for the windowed
            # dataset; otherwise the eval would run on voxel-index coordinates while
            # the model was trained in physical units. load_ctc_images_masks already
            # normalizes the images, hence normalize_imgs=False.
            detections = DetectionSequence.from_masks(
                imgs,
                masks,
                name=str(root),
                ndim=ndim,
                spacing=spacing,
                normalize_imgs=False,
                keep_masks=True,
                keep_images=True,
            )
            features, windows, _dim, _mode = (
                self._tracking_model._extract_detection_windows(detections)
            )
            seq_dir = root.parent if root.name == "TRA" else root
            seq_name = seq_dir.name.removesuffix("_GT")
            dataset = seq_dir.parent.name.removesuffix("_GT")
            name = (
                f"{dataset}_{seq_name}" if seq_name.isdigit() and dataset else seq_name
            )
            if name in used_names:
                name = f"{name}_{index}"
            used_names.add(name)
            cache.append(
                {
                    "name": name,
                    "features": features,
                    "windows": windows,
                    "masks": masks,
                    "spatial_dim": masks.ndim - 1,
                    # traccuracy annotates this graph in place. clear_annotations() makes
                    # the cached graph pristine again between predictions and epochs.
                    "gt_data": load_ctc_data(str(gt_path), run_checks=False),
                }
            )
        self._tracking_cache = cache

    def _tracking_lam_split_values(self) -> tuple[float, ...]:
        """Return the validation sweep supported by the current model and solver."""
        if bool(getattr(self.model, "node_head", False)) and self.tracking_mode in (
            "greedy",
            "ilp",
        ):
            return self._TRACKING_LAM_SPLIT_SWEEP
        return (0.0,)

    def _track_lam_split_sweep(self, predictions, spatial_cutoff):
        """Decode one candidate graph at every configured split-prior weight."""
        from trackastra.tracking import GreedyConfig, track_greedy

        values = self._tracking_lam_split_values()
        baseline, candidate_graph = self._tracking_model._track_from_predictions(
            predictions,
            mode=self.tracking_mode,
            spatial_cutoff=spatial_cutoff,
            return_candidate=True,
        )
        yield values[0], baseline
        for lam_split in values[1:]:
            if self.tracking_mode == "greedy":
                graph = track_greedy(
                    candidate_graph,
                    config=GreedyConfig(lam_split=lam_split),
                )
            elif self.tracking_mode == "ilp":
                from trackastra.tracking.ilp import ILPConfig, track_ilp

                graph = track_ilp(
                    candidate_graph,
                    ilp_config=ILPConfig(lam_split=lam_split),
                )
            else:
                raise RuntimeError(
                    f"Unexpected lam_split sweep mode {self.tracking_mode!r}"
                )
            yield lam_split, graph

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
        spatial_cutoff = self.model.config["spatial_cutoff"]
        rows = []
        with TemporaryDirectory() as tmpdir:
            for movie in self._tracking_cache:
                predictions = tm._predict_from_windows(
                    movie["features"],
                    movie["windows"],
                    spatial_dim=movie["spatial_dim"],
                    batch_size=1,
                )
                for lam_split, graph in self._track_lam_split_sweep(
                    predictions, spatial_cutoff
                ):
                    suffix = f"{lam_split:g}".replace(".", "p")
                    out = Path(tmpdir) / movie["name"] / f"lam_split_{suffix}"
                    graph_to_ctc(graph, movie["masks"], outdir=out)
                    pred_data = load_ctc_data(str(out), run_checks=False)
                    gt_data = movie["gt_data"]
                    gt_data.clear_annotations()
                    try:
                        values, matched = ctc_metrics_from_data(
                            gt_data, pred_data, return_matched=True
                        )
                        rows.append(
                            {
                                "movie": movie["name"],
                                "lam_split": lam_split,
                                **values,
                                **link_type_breakdown(matched),
                            }
                        )
                    finally:
                        gt_data.clear_annotations()
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

    @staticmethod
    def _masked_ce(logits, target, valid, weight):
        """Weighted cross-entropy over the valid nodes; 0 if none are valid."""
        if not bool(valid.any()):
            return logits.new_zeros(())
        return torch.nn.functional.cross_entropy(
            logits[valid].float(), target[valid], weight=weight
        )

    @staticmethod
    def _node_div_counts(out_logits, out_tgt, valid, thresholds):
        """Thresholded node-division (out-degree>=2, i.e. a split) FN/FP counts.

        Mirrors ``edge_error_counts``: a division/split is the positive, thresholded on
        ``P(out-degree>=2)`` (the summed prob of every split class, so it generalizes
        to >2-way splits; for the default max_out_degree=2 it is exactly class 2).
        Returns raw num/den counts (not rates) keyed like the edge counts so they pool
        across steps/ranks and feed the same F1 and FN/FP sweep plot. FN denominator =
        GT div nodes, FP denominator = predicted div.
        """
        p_div = out_logits.float().softmax(-1)[..., 2:].sum(-1)
        v = valid.bool()
        gt = (out_tgt >= 2) & v
        gt_den = gt.sum().float()
        base_only = tuple(thresholds) == (0.5,)
        counts = {}
        for ti, t in enumerate(thresholds):
            pred = (p_div >= t) & v
            values = {
                "node_div_fn_num": (gt & ~pred).sum().float(),
                "node_div_fn_den": gt_den,
                "node_div_fp_num": (pred & ~gt).sum().float(),
                "node_div_fp_den": pred.sum().float(),
            }
            if abs(t - 0.5) < 1e-12:
                counts.update(values)
            if not base_only:
                for key, value in values.items():
                    name, part = key.rsplit("_", 1)
                    counts[edge_count_key(name, ti, part)] = value
        return counts

    def _node_degree_loss(
        self,
        out_logits,
        in_logits,
        batch,
        succ_avail,
        pred_avail,
        mask_time,
        mask_invalid,
    ):
        """Weighted CE for out-degree (source rep) and in-degree (target rep).

        Valid = GT link-set available AND target in the head's class range AND the
        node is observable in-window (has a candidate successor/predecessor within
        delta_cutoff). Observability reuses ``mask_time & ~mask_invalid`` so it is
        automatically sparsity- and delta-consistent with the edge supervision.
        """
        if "node_out_degree" not in batch or "node_in_degree" not in batch:
            raise KeyError(
                "node_loss > 0 but the batch has no node_in/out_degree targets; the "
                "training GT (LineageGraph) does not carry node degrees"
            )
        out_tgt = batch["node_out_degree"]
        in_tgt = batch["node_in_degree"]
        succ_avail = succ_avail.bool()
        pred_avail = pred_avail.bool()

        valid_pair = mask_time & ~mask_invalid
        obs_out = valid_pair.any(dim=-1)  # source has a candidate successor in-window
        obs_in = valid_pair.any(dim=-2)  # target has a candidate predecessor in-window

        # class range follows the head widths (max_out_degree/max_in_degree), so a
        # target beyond a head's range (e.g. a merge, or a >2-way split on a head not
        # built for it) is dropped rather than indexing out of the logits
        max_out = out_logits.shape[-1] - 1
        max_in = in_logits.shape[-1] - 1
        valid_out = succ_avail & (out_tgt >= 0) & (out_tgt <= max_out) & obs_out
        valid_in = pred_avail & (in_tgt >= 0) & (in_tgt <= max_in) & obs_in

        if not self._node_violation_warned:
            n_bad = int(
                ((out_tgt > max_out) & succ_avail).sum()
                + ((in_tgt > max_in) & pred_avail).sum()
            )
            if n_bad:
                logger.warning(
                    "Dropping %d node(s) with out-degree>%d or in-degree>%d "
                    "(merge / over-division) from the node loss",
                    n_bad,
                    max_out,
                    max_in,
                )
                self._node_violation_warned = True

        out_ce = self._masked_ce(out_logits, out_tgt, valid_out, self.node_out_weights)
        in_ce = self._masked_ce(in_logits, in_tgt, valid_in, self.node_in_weights)
        node_div_counts = None
        if self.log_edge_rates:
            with torch.no_grad():
                node_div_counts = self._node_div_counts(
                    out_logits, out_tgt, valid_out, self._EDGE_PLOT_THRESHOLDS
                )
        return out_ce, in_ce, node_div_counts

    def _degree_consistency_loss(
        self, A_pred, out_logits, out_tgt, succ_avail, timepoints, mask, mask_invalid
    ):
        """Pull the edge-implied out-degree toward the node head's expected out-degree.

        The edge-implied out-degree of a source is the sum of its predicted forward
        edge probabilities over candidate successors (row block-sum of ``prob``); the
        node-head expected out-degree is ``E[deg] = p1 + 2*p2`` under the out-degree
        softmax. Smooth L1 between them sends gradients into both heads (mutual
        consistency): the edge head is pulled toward the node head's out-degree and
        the node head toward the edge-implied one; both remain anchored by their own
        GT losses (association BCE and node CE). Its linear regime prevents large
        initial degree disagreements from dominating the other loss terms.

        Each source's error is weighted by ``node_out_weights[out_tgt]``, the
        same class weights the node CE uses, so rare divisions are not drowned out by
        abundant continuations (an unweighted mean would spend almost all its gradient
        on out-degree-1 nodes the edge head already handles). Falls back to a plain
        mean when no weights are set.

        Scored only on sources whose GT successor set is *available* (complete) and
        whose out-degree is in the head's class range, matching ``valid_out`` in the
        node loss. This matters on sparse GT: a censored source (e.g. a sparse track
        start/boundary) has an untrained node out-degree, and the supervision mask --
        being a per-pair ``OR`` of endpoint availability -- can still leak some of its
        edges in, so observability alone would pull edges toward an unreliable target.
        Only the out-degree is constrained: under causal_norm the in-degree (column)
        is already ~normalized, so it carries no division signal.
        """
        if self.causal_norm != "none":
            prob = blockwise_causal_norm_batched(
                A_pred.float(),
                timepoints,
                mode=self.causal_norm,
                mask_invalid=mask_invalid,
            )
        else:
            prob = torch.sigmoid(A_pred.float())
        prob = prob * mask  # forward, in-window, supervised candidate pairs only

        edge_out = prob.sum(dim=-1)  # (B, N) expected number of children
        p_out = out_logits.float().softmax(-1)
        # expected out-degree = sum_k k * p_k, general over however many out-degree
        # classes the head has (3 today: 0/1/2; e.g. 4 if 3-way splits are added)
        n_out_classes = p_out.shape[-1]
        degrees = torch.arange(n_out_classes, device=p_out.device, dtype=p_out.dtype)
        node_out = (p_out * degrees).sum(dim=-1)

        # complete (available) GT successor set, in the head's class range (drop
        # merges), and a supervised candidate successor -> out-degree target is usable
        valid = (
            succ_avail.bool()
            & (out_tgt >= 0)
            & (out_tgt < n_out_classes)
            & mask.bool().any(dim=-1)
        )
        if not bool(valid.any()):
            return A_pred.new_zeros(())
        err = torch.nn.functional.smooth_l1_loss(
            edge_out, node_out, reduction="none", beta=1.0
        )[valid]
        if self.node_out_weights is not None:
            w = self.node_out_weights[out_tgt[valid].long()]
            return (w * err).sum() / w.sum()
        return err.mean()

    def _log_node_losses(self, stage: str, out: dict, batch_size: int) -> None:
        """Log node in/out-degree losses and their sum (losses only)."""
        if self.node_loss <= 0 or out.get("node_loss_out") is None:
            return
        out_ce = out["node_loss_out"]
        in_ce = out["node_loss_in"]
        losses = {
            f"{stage}_node_loss": out_ce + in_ce,
            f"{stage}_node_loss_out": out_ce,
            f"{stage}_node_loss_in": in_ce,
        }
        if out.get("consistency_loss") is not None:
            losses[f"{stage}_consistency_loss"] = out["consistency_loss"]
        self.log_dict(
            losses,
            on_step=stage == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )

    def _common_step(self, batch):
        feats = batch["features"]
        feat_mask = batch.get("feature_mask")
        coords = batch["coords"]
        # association targets arrive as sparse COO; densify on-device (see collate)
        A = densify_assoc(
            batch["assoc_coo"], coords.shape[0], coords.shape[1], device=coords.device
        )
        timepoints = batch["timepoints"]
        padding_mask = batch["padding_mask"]
        padding_mask = padding_mask.bool()

        out_degree_logits = None
        in_degree_logits = None
        # Only forward feature_mask when the batch carries one, so simplified model
        # stubs and mask-free legacy batches keep working (the model treats an absent
        # mask as all-present anyway).
        mask_kw = {} if feat_mask is None else {"feature_mask": feat_mask}
        if self.node_loss > 0:
            A_pred, neighbor_mask, out_degree_logits, in_degree_logits = (
                self._forward_model(
                    coords,
                    feats,
                    padding_mask=padding_mask,
                    return_node_logits=True,
                    **mask_kw,
                )
            )
        else:
            A_pred, neighbor_mask = self._forward_model(
                coords, feats, padding_mask=padding_mask, **mask_kw
            )
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

        node_loss_out = None
        node_loss_in = None
        node_div_counts = None
        consistency_loss = None
        if self.node_loss > 0:
            node_loss_out, node_loss_in, node_div_counts = self._node_degree_loss(
                out_degree_logits,
                in_degree_logits,
                batch,
                gt_successor_set_available,
                gt_predecessor_set_available,
                mask_time,
                mask_invalid,
            )
            loss = loss + self.node_loss * (node_loss_out + node_loss_in)

        if out_degree_logits is not None and self.consistency_weight > 0:
            consistency_loss = self._degree_consistency_loss(
                A_pred,
                out_degree_logits,
                batch["node_out_degree"],
                gt_successor_set_available,
                timepoints,
                mask,
                mask_invalid,
            )
            loss = loss + self.consistency_weight * consistency_loss

        edge_counts = None
        assoc_time_counts = None
        dataset_assoc_counts = None
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
                assoc_time_counts = self._assoc_time_counts_for_batch(
                    A,
                    prob,
                    timepoints,
                    mask,
                    batch.get("dataset_index"),
                    batch.get("seg_index"),
                )
                dataset_assoc_counts = self._dataset_assoc_counts_for_batch(
                    A,
                    prob,
                    timepoints,
                    mask,
                    division_tracks,
                    batch.get("dataset_index"),
                )
                if node_div_counts is not None:
                    edge_counts.update(node_div_counts)

        return dict(
            loss=loss,
            padding_fraction=padding_mask.float().mean(),
            loss_before_reduce=loss_before_reduce,
            A_pred=A_pred,
            mask=mask,
            mask_time=mask_time,
            mask_valid=mask_valid,
            edge_counts=edge_counts,
            assoc_time_counts=assoc_time_counts,
            dataset_assoc_counts=dataset_assoc_counts,
            node_loss_out=node_loss_out,
            node_loss_in=node_loss_in,
            consistency_loss=consistency_loss,
        )

    def _accumulate_edge_counts(self, stage, counts):
        if counts is None:
            return
        acc = self._edge_counts.setdefault(stage, {})
        for key, value in counts.items():
            acc[key] = acc[key] + value if key in acc else value

    @staticmethod
    def _assoc_time_counts_for_batch(
        A: torch.Tensor,
        prob: torch.Tensor,
        timepoints: torch.Tensor,
        mask: torch.Tensor,
        dataset_index: torch.Tensor | None,
        seg_index: torch.Tensor | None,
        threshold: float = 0.5,
    ) -> dict[tuple[int, int, int], np.ndarray]:
        """Association TP/FP/FN counts grouped by target timepoint."""
        if dataset_index is None or seg_index is None:
            return {}
        m = mask.bool()
        gt_pos = (A > 0.5) & m
        pred_pos = (prob >= threshold) & m
        per_target = torch.stack(
            (
                (gt_pos & pred_pos).sum(dim=-2),
                (pred_pos & ~gt_pos).sum(dim=-2),
                (gt_pos & ~pred_pos).sum(dim=-2),
            ),
            dim=-1,
        )
        counts = per_target.detach().to("cpu", dtype=torch.int64).numpy()
        times = timepoints.detach().to("cpu").numpy()
        dataset_ids = dataset_index.detach().to("cpu").numpy()
        seg_ids = seg_index.detach().to("cpu").numpy()

        out: dict[tuple[int, int, int], np.ndarray] = {}
        for b in range(counts.shape[0]):
            ds = int(dataset_ids[b])
            seg = int(seg_ids[b])
            for node in range(counts.shape[1]):
                t = int(times[b, node])
                if t < 0:
                    continue
                value = counts[b, node]
                if int(value.sum()) == 0:
                    continue
                key = (ds, seg, t)
                out[key] = out[key] + value if key in out else value.copy()
        return out

    def _accumulate_assoc_time_counts(
        self, stage: str, counts: dict[tuple[int, int, int], np.ndarray] | None
    ) -> None:
        if not counts:
            return
        acc = self._assoc_time_counts.setdefault(stage, {})
        for key, value in counts.items():
            acc[key] = acc[key] + value if key in acc else value.copy()

    @staticmethod
    def _dataset_assoc_counts_for_batch(
        A: torch.Tensor,
        prob: torch.Tensor,
        timepoints: torch.Tensor,
        mask: torch.Tensor,
        gt_division: torch.Tensor,
        dataset_index: torch.Tensor | None,
        threshold: float = 0.5,
    ) -> dict[int, np.ndarray]:
        """Return regular and division TP/FP/FN counts grouped by dataset."""
        if dataset_index is None:
            return {}
        supervised = mask.bool()
        gt_pos = (A > 0.5) & supervised
        gt_regular = gt_pos & ~gt_division
        gt_div = gt_pos & gt_division
        pred_pos = (prob >= threshold) & supervised
        pred = pred_pos.to(A.dtype)
        row = blockwise_sum_batched(pred, timepoints, dim=-1, reduce="sum")
        col = blockwise_sum_batched(pred, timepoints, dim=-2, reduce="sum")
        pred_division = pred * (row + col) > 2
        pred_regular = pred_pos & ~pred_division
        false_pos = pred_pos & ~gt_pos

        dataset_ids = dataset_index.detach().to("cpu").numpy()
        out = {}
        for current in np.unique(dataset_ids):
            selected = torch.as_tensor(
                dataset_ids == current, device=A.device, dtype=torch.bool
            )[:, None, None]
            values = (
                int(np.sum(dataset_ids == current)),
                int((supervised & selected).sum()),
                int((gt_regular & pred_pos & selected).sum()),
                int((false_pos & pred_regular & selected).sum()),
                int((gt_regular & ~pred_pos & selected).sum()),
                int((gt_div & pred_pos & selected).sum()),
                int((false_pos & pred_division & selected).sum()),
                int((gt_div & ~pred_pos & selected).sum()),
            )
            out[int(current)] = np.asarray(values, dtype=np.int64)
        return out

    def _accumulate_dataset_assoc_counts(
        self, stage: str, counts: dict[int, np.ndarray] | None
    ) -> None:
        if not counts:
            return
        acc = self._dataset_assoc_counts.setdefault(stage, {})
        for dataset_index, value in counts.items():
            acc[dataset_index] = (
                acc[dataset_index] + value if dataset_index in acc else value.copy()
            )

    @staticmethod
    def _merge_dataset_assoc_counts(
        items: list[dict[str, dict[int, np.ndarray]]],
    ) -> dict[str, dict[int, np.ndarray]]:
        merged: dict[str, dict[int, np.ndarray]] = {}
        for item in items:
            for stage, counts in item.items():
                acc = merged.setdefault(stage, {})
                for dataset_index, value in counts.items():
                    array = np.asarray(value, dtype=np.int64)
                    acc[dataset_index] = (
                        acc[dataset_index] + array
                        if dataset_index in acc
                        else array.copy()
                    )
        return merged

    def _gather_dataset_assoc_counts(self) -> dict[str, dict[int, np.ndarray]]:
        local = {
            stage: {
                dataset_index: value.copy() for dataset_index, value in counts.items()
            }
            for stage, counts in self._dataset_assoc_counts.items()
        }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered, local)
            return self._merge_dataset_assoc_counts(gathered)
        return self._merge_dataset_assoc_counts([local])

    def _dataset_metric_rows(
        self, counts: dict[str, dict[int, np.ndarray]]
    ) -> list[dict[str, Any]]:
        rows = []
        for stage in ("train", "val"):
            roots = self._dataset_root_map_for_stage(stage)
            for dataset_index, value in sorted(counts.get(stage, {}).items()):
                samples, supervised, reg_tp, reg_fp, reg_fn, div_tp, div_fp, div_fn = (
                    int(x) for x in value
                )
                regular = metrics_from_counts(reg_tp, reg_fp, reg_fn)
                division = metrics_from_counts(div_tp, div_fp, div_fn)
                rows.append(
                    {
                        "epoch": int(self.current_epoch),
                        "global_step": int(self.global_step),
                        "split": stage,
                        "dataset_index": dataset_index,
                        "dataset_root": roots.get(dataset_index, ""),
                        "sampled_windows": samples,
                        "supervised_pairs": supervised,
                        "regular_gt_edges": reg_tp + reg_fn,
                        "regular_pred_edges": reg_tp + reg_fp,
                        "regular_tp": reg_tp,
                        "regular_fp": reg_fp,
                        "regular_fn": reg_fn,
                        "regular_f1": regular["f1"],
                        "regular_jaccard": regular["jaccard"],
                        "division_gt_edges": div_tp + div_fn,
                        "division_pred_edges": div_tp + div_fp,
                        "division_tp": div_tp,
                        "division_fp": div_fp,
                        "division_fn": div_fn,
                        "division_f1": division["f1"],
                        "division_jaccard": division["jaccard"],
                    }
                )
        return rows

    def _write_dataset_metrics(self) -> None:
        """Upsert compact per-epoch, per-dataset association metrics."""
        counts = self._gather_dataset_assoc_counts()
        if not self.trainer.is_global_zero or self.dataset_metrics_path is None:
            return
        rows = self._dataset_metric_rows(counts)
        if not rows:
            return
        path = self.dataset_metrics_path
        path.parent.mkdir(parents=True, exist_ok=True)
        current = pd.DataFrame(rows)
        if path.exists():
            previous = pd.read_csv(path)
            current = pd.concat((previous, current), ignore_index=True)
            current = current.drop_duplicates(
                subset=("epoch", "split", "dataset_index"), keep="last"
            )
        current = current.sort_values(["epoch", "split", "dataset_index"])
        current.to_csv(path, index=False)

    @staticmethod
    def _merge_assoc_time_counts(
        items: list[dict[str, dict[tuple[int, int, int], np.ndarray]]],
    ) -> dict[str, dict[tuple[int, int, int], np.ndarray]]:
        merged: dict[str, dict[tuple[int, int, int], np.ndarray]] = {}
        for item in items:
            for stage, counts in item.items():
                acc = merged.setdefault(stage, {})
                for key, value in counts.items():
                    arr = np.asarray(value, dtype=np.int64)
                    acc[key] = acc[key] + arr if key in acc else arr.copy()
        return merged

    def _gather_assoc_time_counts(
        self,
    ) -> dict[str, dict[tuple[int, int, int], np.ndarray]]:
        local = {
            stage: {key: value.copy() for key, value in counts.items()}
            for stage, counts in self._assoc_time_counts.items()
        }
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered, local)
            return self._merge_assoc_time_counts(gathered)
        return self._merge_assoc_time_counts([local])

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

        pooled = {}
        for key in self._EDGE_KEYS:
            num = self.all_gather(acc.get(f"{key}_num", zero)).sum()
            den = self.all_gather(acc.get(f"{key}_den", zero)).sum()
            pooled[key] = (float(num), float(den))

        # All association metrics from one consistent (TP, FP, FN) triple per
        # group: TP = GT_pos - FN (GT_pos is the fn denominator), FN/FP the fn/fp
        # numerators. Metrics agree by construction (e.g. jaccard == f1 / (2 - f1));
        # NaN (empty group) skipped.
        def _emit_group(suffix, fn_key, fp_key):
            fn_num, gt_pos = pooled[fn_key]
            fp_num = pooled[fp_key][0]
            metrics = metrics_from_counts(gt_pos - fn_num, fp_num, fn_num)
            for name, value in metrics.items():
                if value == value:  # skip NaN
                    _emit(f"{stage}_assoc_{name}{suffix}", value)

        _emit_group("", "fn", "fp")
        _emit_group("_div", "fn_div", "fp_div")

        # Node-division (out-degree==2) quality, pooled like the edge counts. Emitted
        # as {stage}_node_div_{f1,jaccard,fn,fp} to parallel the edge _div metrics.
        if self.node_loss > 0:
            fn_num = float(self.all_gather(acc.get("node_div_fn_num", zero)).sum())
            gt_pos = float(self.all_gather(acc.get("node_div_fn_den", zero)).sum())
            fp_num = float(self.all_gather(acc.get("node_div_fp_num", zero)).sum())
            if gt_pos > 0:
                metrics = metrics_from_counts(gt_pos - fn_num, fp_num, fn_num)
                for name, value in metrics.items():
                    if value == value:  # skip NaN
                        _emit(f"{stage}_node_div_{name}", value)

    def _pooled_class_threshold_rates(
        self, stage: str, classes: list[tuple[str, str, str]]
    ) -> list[dict[str, float | str]]:
        """Pooled threshold-sweep FN/FP rates for the given classes.

        ``classes`` is a list of ``(label, fn_key, fp_key)``; each pooled row carries
        ``fn`` (= 1 - recall) and ``fp`` (= 1 - precision) for one class at one
        threshold. all_gather runs for every (threshold, class) in fixed order so the
        collective cannot deadlock across DDP ranks.
        """
        acc = self._edge_counts.get(stage, {})
        zero = torch.zeros((), device=self.device)
        rows = []

        def _counts(key: str, threshold_idx: int) -> tuple[float, float]:
            num_key = self._edge_count_key(key, threshold_idx, "num")
            den_key = self._edge_count_key(key, threshold_idx, "den")
            num = self.all_gather(acc.get(num_key, zero)).sum()
            den = self.all_gather(acc.get(den_key, zero)).sum()
            return float(num), float(den)

        for threshold_idx, threshold in enumerate(self._EDGE_PLOT_THRESHOLDS):
            for label, fn_key, fp_key in classes:
                fn_num, gt_pos = _counts(fn_key, threshold_idx)
                fp_num, pred_pos = _counts(fp_key, threshold_idx)
                if gt_pos == 0 or pred_pos == 0:
                    continue
                metrics = metrics_from_counts(gt_pos - fn_num, fp_num, fn_num)
                rows.append(
                    {
                        "stage": stage,
                        "class": label,
                        "threshold": float(threshold),
                        **metrics,
                    }
                )
        return rows

    def _log_fnfp_plot(self, classes, colors, class_labels, axis, title, log_key):
        """Log a compact FN/FP threshold-sweep plot for train and validation.

        Shared by the association (edge) and node error plots. ``classes`` selects the
        pooled counts; ``class_labels`` maps a class label to its human name; ``axis``
        is ``(xlabel, ylabel)``. A faint swept curve plus a marker at threshold 0.5.
        """
        if (
            not self.log_edge_rates
            or not isinstance(self.logger, WandbLogger)
            or self.trainer.sanity_checking
        ):
            return

        rows = []
        for stage in ("train", "val"):
            rows.extend(self._pooled_class_threshold_rates(stage, classes))
        if not self.trainer.is_global_zero or not rows:
            return

        import matplotlib.pyplot as plt
        import wandb as _wandb

        fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=150)
        linestyles = {"train": "-", "val": "--"}
        markers = {"train": "o", "val": "s"}

        for stage in ("train", "val"):
            for label, _fn_key, _fp_key in classes:
                group = [
                    row
                    for row in rows
                    if row["stage"] == stage and row["class"] == label
                ]
                if not group:
                    continue
                group = sorted(group, key=lambda row: row["threshold"])
                xs = [row["fn"] for row in group]
                ys = [row["fp"] for row in group]
                current = [
                    row for row in group if abs(float(row["threshold"]) - 0.5) < 1e-12
                ]
                legend = f"{stage} {class_labels[label]}"
                ax.plot(
                    xs,
                    ys,
                    color=colors[label],
                    linestyle=linestyles[stage],
                    linewidth=1.0,
                    alpha=0.35,
                    label=None if current else legend,
                    zorder=1,
                )
                if current:
                    row = current[0]
                    ax.scatter(
                        row["fn"],
                        row["fp"],
                        color=colors[label],
                        marker=markers[stage],
                        edgecolor="black",
                        linewidth=0.5,
                        s=42,
                        label=legend,
                        zorder=2,
                    )

        ax.set_xlabel(axis[0])
        ax.set_ylabel(axis[1])
        ax.set_title(f"{title}, epoch {self.current_epoch}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(alpha=0.2, linewidth=0.5)
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, frameon=False, fontsize=8)
        fig.tight_layout()
        self.logger.experiment.log({log_key: _wandb.Image(fig)})
        plt.close(fig)

    def _log_assoc_error_plot(self) -> None:
        """Log the edge (association) FN/FP threshold-sweep plot."""
        self._log_fnfp_plot(
            classes=[("regular", "fn", "fp"), ("division", "fn_div", "fp_div")],
            colors={"regular": "tab:blue", "division": "tab:red"},
            class_labels={"regular": "non-div", "division": "div"},
            axis=(
                "missed links (FN / GT links)",
                "wrong proposed links (FP / predicted links)",
            ),
            title="Association errors",
            log_key="assoc_errors",
        )

    def _log_node_error_plot(self) -> None:
        """Log the node-degree FN/FP threshold-sweep plot (division class for now)."""
        if self.node_loss <= 0:
            return
        # One class today (division = out-degree 2); the list generalizes to the full
        # (in, out) degree combos if we add their counts later.
        self._log_fnfp_plot(
            classes=[("div", "node_div_fn", "node_div_fp")],
            colors={"div": "tab:red"},
            class_labels={"div": "div (out=2)"},
            axis=(
                "missed nodes (FN / GT nodes)",
                "wrong proposed nodes (FP / predicted nodes)",
            ),
            title="Node degree errors",
            log_key="node_errors",
        )

    @staticmethod
    def _short_dataset_label(root: str, dataset_index: int, max_len: int = 28) -> str:
        if not root:
            return str(dataset_index)
        path = Path(root)
        parts = [part for part in path.parts if part not in ("", path.anchor)]
        if parts:
            label = (
                "/".join(parts[-2:])
                if len(parts[-1]) < 5 and len(parts) >= 2
                else parts[-1]
            )
        else:
            label = path.name
        if not label:
            label = str(dataset_index)
        if len(label) > max_len:
            label = f"...{label[-(max_len - 3) :]}"
        return label

    @staticmethod
    def _loader_datasets(loader) -> list:
        """Return the map-style dataset(s) behind a trainer loader object.

        Handles a bare ``DataLoader``, a list/tuple of loaders, or a Lightning
        ``CombinedLoader`` (which exposes its child loaders via ``flattened``).
        """
        if loader is None:
            return []
        flattened = getattr(loader, "flattened", None)
        items = flattened if flattened is not None else loader
        if not isinstance(items, (list, tuple)):
            items = [items]
        return [getattr(item, "dataset", item) for item in items]

    def _stage_subdatasets(self, stage: str) -> list:
        """Per-source sub-datasets for a split, however the trainer was fed.

        Supports both a ``LightningDataModule`` (``dm.datasets[stage]``) and the
        plain-dataloader path used by ``scripts/train.py`` (bundles handed to
        ``trainer.fit``), where the sources live on the loader's ``ConcatDataset``.
        """
        dm = getattr(self.trainer, "datamodule", None)
        datasets = getattr(dm, "datasets", None) if dm is not None else None
        if isinstance(datasets, dict):
            subs = getattr(datasets.get(stage), "datasets", None)
            if subs:
                return list(subs)
        loader = getattr(
            self.trainer,
            "train_dataloader" if stage == "train" else "val_dataloaders",
            None,
        )
        for ds in self._loader_datasets(loader):
            subs = getattr(ds, "datasets", None)
            if subs:
                return list(subs)
            if hasattr(ds, "dataset_index"):
                return [ds]
        return []

    def _dataset_root_map_for_stage(self, stage: str) -> dict[int, str]:
        cached = self._assoc_time_dataset_maps.get(stage)
        if cached:
            return cached
        mapping: dict[int, str] = {
            int(getattr(sub, "dataset_index", -1)): str(getattr(sub, "root", ""))
            for sub in self._stage_subdatasets(stage)
        }
        if mapping:
            self._assoc_time_dataset_maps[stage] = mapping
        return mapping

    @classmethod
    def _assoc_time_error_matrix(
        cls,
        counts: dict[tuple[int, int, int], np.ndarray],
        dataset_roots: dict[int, str],
        n_bins: int | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Pool per-time association counters into dataset rows and normalized bins."""
        n_bins = cls._ASSOC_TIME_BINS if n_bins is None else int(n_bins)
        if not counts:
            return np.full((0, n_bins), np.nan), []

        stream_ranges: dict[tuple[int, int], list[int]] = {}
        for (dataset_index, seg_index, timepoint), value in counts.items():
            if int(np.asarray(value).sum()) == 0:
                continue
            key = (dataset_index, seg_index)
            if key in stream_ranges:
                stream_ranges[key][0] = min(stream_ranges[key][0], timepoint)
                stream_ranges[key][1] = max(stream_ranges[key][1], timepoint)
            else:
                stream_ranges[key] = [timepoint, timepoint]

        pooled: dict[tuple[int, int], np.ndarray] = {}
        for (dataset_index, seg_index, timepoint), value in counts.items():
            stream_key = (dataset_index, seg_index)
            if stream_key not in stream_ranges:
                continue
            lo, hi = stream_ranges[stream_key]
            frac = 0.0 if hi == lo else (timepoint - lo) / (hi - lo)
            bin_index = min(int(frac * n_bins), n_bins - 1)
            key = (dataset_index, bin_index)
            arr = np.asarray(value, dtype=np.int64)
            pooled[key] = pooled[key] + arr if key in pooled else arr.copy()

        dataset_indices = sorted({dataset_index for dataset_index, _bin in pooled})
        labels = [
            cls._short_dataset_label(
                dataset_roots.get(dataset_index, ""), dataset_index
            )
            for dataset_index in dataset_indices
        ]
        label_counts = {label: labels.count(label) for label in labels}
        for i, label in enumerate(labels):
            if label_counts[label] > 1:
                labels[i] = f"{label} [{dataset_indices[i]}]"

        matrix = np.full((len(dataset_indices), n_bins), np.nan, dtype=np.float32)
        row_for_dataset = {
            dataset_index: row for row, dataset_index in enumerate(dataset_indices)
        }
        for (dataset_index, bin_index), value in pooled.items():
            tp, fp, fn = np.asarray(value, dtype=np.float64)
            den = tp + fp + fn
            if den > 0:
                matrix[row_for_dataset[dataset_index], bin_index] = (fp + fn) / den
        for row in range(matrix.shape[0]):
            matrix[row] = cls._nearest_fill_interior(matrix[row])
        return matrix, labels

    @staticmethod
    def _nearest_fill_interior(row: np.ndarray) -> np.ndarray:
        """Fill NaN gaps between the first and last measured bin with the nearest
        measured value (ties resolve to the earlier bin).

        Short movies have fewer timepoints than time bins, so ``int(frac * n_bins)``
        leaves interior columns empty; this paints those holes from the closest real
        measurement. Leading/trailing NaNs are left untouched so the plot never
        extrapolates past a movie's actual time span.
        """
        finite = np.flatnonzero(np.isfinite(row))
        if finite.size == 0:
            return row
        filled = row.copy()
        for i in range(int(finite[0]) + 1, int(finite[-1])):
            if np.isfinite(filled[i]):
                continue
            filled[i] = row[finite[np.argmin(np.abs(finite - i))]]
        return filled

    @staticmethod
    def _draw_assoc_time_panel(ax, matrix, labels, title, *, cmap, vmax):
        n_bins = matrix.shape[1]
        if matrix.shape[0] == 0:
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_axis_off()
            ax.set_box_aspect(1 / n_bins)
            return None
        image = ax.matshow(
            matrix,
            interpolation="nearest",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("normalized target time")
        ax.set_xticks([0, n_bins // 2, n_bins - 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_label_position("bottom")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.tick_params(axis="both", length=0)
        ax.set_box_aspect(matrix.shape[0] / n_bins)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return image

    def _log_assoc_time_plot(self) -> None:
        """Log train/val association error heatmaps over normalized target time."""
        if (
            not self.log_edge_rates
            or not isinstance(self.logger, WandbLogger)
            or self.trainer.sanity_checking
        ):
            return
        gathered = self._gather_assoc_time_counts()
        if not self.trainer.is_global_zero:
            return

        train_matrix, train_labels = self._assoc_time_error_matrix(
            gathered.get("train", {}),
            self._dataset_root_map_for_stage("train"),
        )
        val_matrix, val_labels = self._assoc_time_error_matrix(
            gathered.get("val", {}),
            self._dataset_root_map_for_stage("val"),
        )
        finite_parts = [x[np.isfinite(x)] for x in (train_matrix, val_matrix) if x.size]
        if not finite_parts:
            return
        finite = np.concatenate(finite_parts)
        if finite.size == 0:
            return

        import matplotlib.pyplot as plt
        import wandb as _wandb
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.ticker import PercentFormatter

        vmax = float(np.nanpercentile(finite, 95))
        vmax = min(1.0, max(vmax, 0.05))
        row_counts = [max(train_matrix.shape[0], 1), max(val_matrix.shape[0], 1)]
        height = max(3.0, 1.8 + 0.28 * sum(row_counts))
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.0, height),
            dpi=150,
            constrained_layout=True,
            gridspec_kw={"height_ratios": row_counts},
        )
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("#e6e6e6")
        self._draw_assoc_time_panel(
            axes[0],
            train_matrix,
            train_labels,
            f"train, epoch {self.current_epoch}",
            cmap=cmap,
            vmax=vmax,
        )
        self._draw_assoc_time_panel(
            axes[1],
            val_matrix,
            val_labels,
            f"val, epoch {self.current_epoch}",
            cmap=cmap,
            vmax=vmax,
        )
        sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=vmax), cmap=cmap)
        cbar = fig.colorbar(sm, ax=axes, shrink=0.8)
        cbar.set_label("association error (1 - Jaccard)")
        cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        self.logger.experiment.log({"assoc_errors_time": _wandb.Image(fig)})
        plt.close(fig)

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
            spatial_cutoff=float(self.model.config["spatial_cutoff"]),
            grad_norm_value=grad_norm_value,
            trigger=trigger,
        )
        torch.save(payload, path)
        msg = f"Saved {trigger} spike debug batch to {path} (loss={loss_value:.6g}"
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
        if (
            np.isfinite(grad_norm_value)
            and grad_norm_value < _GRAD_SPIKE_DEBUG_THRESHOLD
        ):
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
        self._assoc_time_counts["train"] = {}
        self._dataset_assoc_counts["train"] = {}
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
        self._assoc_time_counts["val"] = {}
        self._dataset_assoc_counts["val"] = {}

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
        if int(getattr(self.trainer, "global_rank", 0)) != 0:
            self._batch_provenance_map_written = True
            return
        mapping = {
            int(getattr(sub, "dataset_index", -1)): str(getattr(sub, "root", ""))
            for sub in self._stage_subdatasets("train")
        }
        if not mapping:
            return
        self._batch_provenance_map_written = True
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
            logger.warning(
                "NaN train loss at epoch=%s batch_idx=%s, skipping",
                self.current_epoch,
                batch_idx,
            )
            return None

        self._accumulate_edge_counts("train", out["edge_counts"])
        self._accumulate_assoc_time_counts("train", out["assoc_time_counts"])
        self._accumulate_dataset_assoc_counts("train", out["dataset_assoc_counts"])

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["coords"].shape[0],
        )
        self._log_node_losses("train", out, batch["coords"].shape[0])

        # self.train_loss.append(loss)

        # accumulate (B, N) for the per-epoch 2d histogram (replaces the separate
        # batch_size / detections_per_sequence scalar logs -- same info, joint view)
        self._bn_log.append(
            (int(batch["coords"].shape[0]), int(batch["coords"].shape[1]))
        )
        n_windows = batch["coords"].shape[0]
        # total frames in the batch = windows * frames-per-window
        n_frames = n_windows * self.model.config["window"]
        detections_per_batch = float((~batch["padding_mask"].bool()).sum())
        supervised_per_batch = float(out["mask"].sum())
        self.log_dict(
            {
                # real (non-pad) detections summed over the batch
                "detections_per_batch": detections_per_batch,
                # ... averaged per frame in the batch
                "detections_per_frame": detections_per_batch / n_frames,
                # supervised directed association decisions (loss_mask) over the batch
                "supervised_per_batch": supervised_per_batch,
                # ... averaged per frame in the batch
                "supervised_per_frame": supervised_per_batch / n_frames,
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
            logger.warning(
                "NaN validation loss at epoch=%s batch_idx=%s, skipping",
                self.current_epoch,
                batch_idx,
            )
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
        self._log_node_losses("val", out, batch["coords"].shape[0])

        self._accumulate_edge_counts("val", out["edge_counts"])
        self._accumulate_assoc_time_counts("val", out["assoc_time_counts"])
        self._accumulate_dataset_assoc_counts("val", out["dataset_assoc_counts"])

        if batch_idx == self.batch_val_tb_idx:
            self.batch_val_tb = dict(batch=batch, out=out)

        return loss

    def on_validation_epoch_end(self):
        # skip if sanity checking
        if self.trainer.sanity_checking:
            return

        self._log_edge_error_rates("val")
        self._log_assoc_error_plot()
        self._log_assoc_time_plot()
        self._log_node_error_plot()
        self._write_dataset_metrics()

        # Hack to make lightning progress bars with loss values persistent
        print(" ")

        if (
            _is_tracking_epoch(self.current_epoch, self.tracking_frequency)
            and self.trainer.is_global_zero
            and self.tracking_inputs
        ):
            try:
                metrics = self._run_tracking_eval()
                self._write_tracking_metrics(metrics, epoch=int(self.current_epoch))
                logger.info(
                    "[epoch %s] per-movie tracking metrics:\n%s",
                    self.current_epoch,
                    metrics.to_markdown(index=False),
                )
                baseline = metrics
                if "lam_split" in metrics:
                    baseline = metrics[np.isclose(metrics["lam_split"], 0.0)]
                values = _summarize_tracking_metrics(baseline)
                values.update(_summarize_lam_split_sweep(metrics))
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
                if "val_best_lam_split" in values:
                    msg += (
                        f" val_best_lam_split={values['val_best_lam_split']:.4g}"
                        f" val_best_AOGM={values['val_best_AOGM']:.4f}"
                    )
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
