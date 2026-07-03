"""Pure tensor loss helpers for Trackastra training."""

from __future__ import annotations

import torch

from trackastra.utils import blockwise_sum_batched


def apply_focal_weight(loss: torch.Tensor, gamma: float) -> torch.Tensor:
    """Apply binary focal modulation to an unreduced BCE loss."""
    if gamma == 0:
        return loss
    return (1 - torch.exp(-loss)).pow(gamma) * loss


def reduce_decision_loss(
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


def reduce_matrix_loss(
    loss: torch.Tensor,
    mask: torch.Tensor,
    eps: float = torch.finfo(torch.float16).eps,
) -> torch.Tensor:
    """Reduction over all valid association-matrix entries.

    Each sample is normalised by its number of valid pairs, then samples with at
    least one valid pair are averaged with equal weight.
    """
    entries_per_sample = mask.sum(dim=(1, 2))
    sample_valid = entries_per_sample > 0
    loss_per_sample = loss.sum(dim=(1, 2)) / (entries_per_sample + eps)
    return (loss_per_sample * sample_valid).sum() / sample_valid.sum().clamp_min(1)


def child_ce_loss_matrix(
    log_p: torch.Tensor,
    log_p_null: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dt: torch.Tensor,
    delta_cutoff: int,
    focal_loss_gamma: float = 0.0,
) -> torch.Tensor:
    """Matrix-shaped child parent-or-null CE loss."""
    mask = mask.bool()
    target = (target > 0.5) & mask
    loss = torch.zeros_like(log_p)

    for delta in range(1, delta_cutoff + 1):
        candidate_mask = mask & (dt == delta)
        target_delta = target & (dt == delta)

        true_edge_loss = (-log_p).masked_fill(~target_delta, 0.0)
        loss = loss + apply_focal_weight(true_edge_loss, focal_loss_gamma)

        candidate_count = candidate_mask.sum(dim=1)
        target_count = target_delta.sum(dim=1)
        null_decision = (candidate_count > 0) & (target_count == 0)
        null_logp = log_p_null.masked_fill(~candidate_mask, -torch.inf)
        null_loss = torch.where(
            candidate_count > 0,
            -null_logp.amax(dim=1),
            torch.zeros_like(candidate_count, dtype=log_p.dtype),
        )
        null_loss = apply_focal_weight(null_loss, focal_loss_gamma)
        null_per_candidate = null_loss / candidate_count.clamp_min(1)
        loss = loss + (
            candidate_mask.float()
            * null_decision.unsqueeze(1)
            * null_per_candidate.unsqueeze(1)
        )

    return loss


def quiet_softmax_child_log_null(
    logits: torch.Tensor,
    timepoints: torch.Tensor,
    mask_invalid: torch.BoolTensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Log-probability of the quiet-softmax child null class."""
    logits = logits.clone()
    if mask_invalid is not None:
        logits[mask_invalid] = -torch.inf

    with torch.no_grad():
        max_parent = blockwise_sum_batched(logits, timepoints, dim=0, reduce="amax")
    exp_parent = torch.exp(logits - max_parent)
    denom = blockwise_sum_batched(exp_parent, timepoints, dim=0, reduce="sum") + eps
    denom = denom + torch.exp(-max_parent)
    log_p_null = -max_parent - torch.log(denom)
    return torch.where(
        torch.isfinite(log_p_null),
        log_p_null,
        torch.zeros_like(log_p_null),
    )


def edge_count_key(key: str, threshold_idx: int, part: str) -> str:
    """Build the stable metric key for a thresholded edge count."""
    return f"{key}_t{threshold_idx}_{part}"


def edge_error_counts(
    A: torch.Tensor,
    prob: torch.Tensor,
    timepoints: torch.Tensor,
    mask: torch.Tensor,
    gt_division: torch.Tensor,
    thresholds: tuple[float, ...] | None = None,
) -> dict[str, torch.Tensor]:
    """Pre-solver association FN/FP counts on regular and division links."""
    m = mask.bool()
    gt_pos = (A > 0.5) & m
    gt_regular = gt_pos & ~gt_division
    gt_div = gt_pos & gt_division
    thresholds = (0.5,) if thresholds is None else tuple(float(t) for t in thresholds)
    include_threshold_counts = thresholds != (0.5,)
    counts = {}

    for threshold_idx, threshold in enumerate(thresholds):
        pred_pos = (prob >= threshold) & m
        pred = pred_pos.to(A.dtype)
        row = blockwise_sum_batched(pred, timepoints, dim=-1, reduce="sum")
        col = blockwise_sum_batched(pred, timepoints, dim=-2, reduce="sum")
        pred_division = pred * (row + col) > 2

        fn = gt_pos & ~pred_pos
        fp = pred_pos & ~gt_pos
        pred_regular = pred_pos & ~pred_division
        values = {
            "fn_num": (fn & gt_regular).sum().float(),
            "fn_den": gt_regular.sum().float(),
            "fp_num": (fp & pred_regular).sum().float(),
            "fp_den": pred_regular.sum().float(),
            "fn_div_num": (fn & gt_div).sum().float(),
            "fn_div_den": gt_div.sum().float(),
            "fp_div_num": (fp & pred_division).sum().float(),
            "fp_div_den": (pred_pos & pred_division).sum().float(),
        }
        if abs(threshold - 0.5) < 1e-12:
            counts.update(values)
        if include_threshold_counts:
            for key, value in values.items():
                name, part = key.rsplit("_", 1)
                counts[edge_count_key(name, threshold_idx, part)] = value
    return counts


def error_rate_f1(fn_rate: float, fp_rate: float) -> float:
    """F1 from the FN rate (1 - recall) and FP rate (1 - precision)."""
    recall, precision = 1.0 - fn_rate, 1.0 - fp_rate
    total = precision + recall
    if total > 0:
        return 2.0 * precision * recall / total
    if recall == recall and precision == precision:
        return 0.0
    return float("nan")
