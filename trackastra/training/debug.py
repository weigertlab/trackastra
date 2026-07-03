"""Debug payload helpers for Trackastra training."""

from __future__ import annotations

import torch

from trackastra.data import densify_assoc
from trackastra.utils import blockwise_causal_norm_batched

LOSS_SPIKE_DEBUG_THRESHOLD = 1.0
LOSS_SPIKE_DEBUG_MIN_EPOCH = 10
LOSS_SPIKE_DEBUG_TOPK = 64
LOSS_SPIKE_DEBUG_MAX_PER_EPOCH = 8
LOSS_SPIKE_DEBUG_MAX_TOTAL = 64
GRAD_SPIKE_DEBUG_THRESHOLD = 100.0


def _debug_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x


def _gather_edge_values(
    x: torch.Tensor, b: torch.Tensor, i: torch.Tensor, j: torch.Tensor
):
    return x[b, i, j].detach().cpu()


def loss_spike_debug_payload(
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
    """Build the serialized payload for loss and gradient spike debugging."""
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
        k = min(LOSS_SPIKE_DEBUG_TOPK, flat.numel())
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
        impossible_idx = impossible_idx[:LOSS_SPIKE_DEBUG_TOPK]

        payload = {
            "meta": {
                "stage": stage,
                "epoch": epoch,
                "global_step": global_step,
                "batch_idx": batch_idx,
                "rank": rank,
                "loss": loss_value,
                "threshold": LOSS_SPIKE_DEBUG_THRESHOLD,
                "grad_norm": grad_norm_value,
                "grad_threshold": GRAD_SPIKE_DEBUG_THRESHOLD,
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
