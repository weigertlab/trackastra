"""Batched blockwise ops must match the per-sample loop exactly."""

import torch

from trackastra.utils.utils import (
    blockwise_causal_log_prob_batched,
    blockwise_causal_norm,
    blockwise_causal_norm_batched,
    blockwise_sum,
    blockwise_sum_batched,
)


def _make_batch(B=4, seed=0):
    """Random batch with window-style timepoints, variable size, and padding (-1)."""
    g = torch.Generator().manual_seed(seed)
    N = 18
    window = 4
    A = torch.randn(B, N, N, generator=g)
    timepoints = torch.empty(B, N, dtype=torch.long)
    for b in range(B):
        # valid tokens carry one of `window` consecutive (offset) timepoints
        n_valid = int(torch.randint(window, N, (1,), generator=g))
        base = int(torch.randint(0, 50, (1,), generator=g))
        tp = base + torch.randint(0, window, (n_valid,), generator=g)
        tp, _ = torch.sort(tp)  # blocks are contiguous, as in real windows
        timepoints[b, :n_valid] = tp
        timepoints[b, n_valid:] = -1  # padding
    return A, timepoints


def _loop_sum(A, timepoints, dim, reduce):
    return torch.stack(
        [blockwise_sum(a, t, dim=dim, reduce=reduce) for a, t in zip(A, timepoints)], 0
    )


def test_blockwise_sum_batched_matches_loop():
    A, tp = _make_batch()
    for dim in (-2, -1, 0, 1):
        ref = _loop_sum(A, tp, dim, "sum")
        bat = blockwise_sum_batched(A, tp, dim=dim, reduce="sum")
        assert torch.allclose(ref, bat, atol=1e-5), (dim, (ref - bat).abs().max())


def test_blockwise_sum_batched_amax_matches_loop():
    A, tp = _make_batch(seed=2)
    for dim in (0, 1):
        ref = _loop_sum(A, tp, dim, "amax")
        bat = blockwise_sum_batched(A, tp, dim=dim, reduce="amax")
        assert torch.allclose(ref, bat, atol=1e-5), (dim, (ref - bat).abs().max())


def test_blockwise_sum_integer_counts_exact():
    # the GT association matrix is 0/1; block sums must stay exact integers so the
    # downstream `block_sum == 2` / `> 2` track-type tests are correct.
    A, tp = _make_batch(seed=5)
    A = (A > 0).float()
    bat = blockwise_sum_batched(A, tp, dim=-1, reduce="sum")
    assert torch.equal(bat, bat.round())


def test_blockwise_causal_norm_batched_matches_loop():
    A, tp = _make_batch(seed=3)
    # batched mask_invalid (padded query or key), as built in _common_step
    pad = tp < 0
    mask_invalid = pad.unsqueeze(1) | pad.unsqueeze(2)

    for mode in ("quiet_softmax", "softmax", "linear"):
        ref = torch.stack([
            blockwise_causal_norm(a, t, mode=mode, mask_invalid=m)
            for a, t, m in zip(A, tp, mask_invalid)
        ])
        bat = blockwise_causal_norm_batched(
            A, tp, mode=mode, mask_invalid=mask_invalid
        )
        assert torch.allclose(ref, bat, atol=1e-5, equal_nan=True), (
            mode,
            (ref - bat).abs().nan_to_num().max(),
        )


def test_log_prob_matches_probability_bce_on_benign_inputs():
    # on well-conditioned logits the log-space loss must equal the old
    # `BCELoss(blockwise_causal_norm_batched(...))` path elementwise.
    A, tp = _make_batch(seed=7)
    pad = tp < 0
    mask_invalid = pad.unsqueeze(1) | pad.unsqueeze(2)
    target = (torch.rand_like(A) > 0.5).float()

    for mode in ("quiet_softmax", "softmax", "linear"):
        prob = blockwise_causal_norm_batched(A, tp, mode=mode, mask_invalid=mask_invalid)
        ref = torch.nn.functional.binary_cross_entropy(
            prob.clamp(1e-7, 1 - 1e-7), target, reduction="none"
        )
        log_p, log_1mp = blockwise_causal_log_prob_batched(
            A, tp, mode=mode, mask_invalid=mask_invalid
        )
        bce = -(target * log_p + (1 - target) * log_1mp)
        # compare valid, non-saturated edges; the log-space path deliberately
        # caps the negative term in the saturated tail (the stability fix), so
        # only assert agreement where p is away from 0/1.
        well = (~mask_invalid) & (prob > 1e-3) & (prob < 1 - 1e-3)
        assert torch.allclose(ref[well], bce[well], atol=1e-4), (
            mode,
            (ref[well] - bce[well]).abs().max(),
        )


def test_log_prob_finite_and_bounded_grad_on_saturated_logits():
    # the failure mode that produced the loss spikes: confident-wrong saturated
    # logits. The log-space loss must stay finite with a bounded gradient where
    # the materialized-probability BCE underflowed to log(0).
    A, tp = _make_batch(seed=11)
    A = (A * 200.0).requires_grad_(True)  # drive softmax to hard 0/1 in fp32
    pad = tp < 0
    mask_invalid = pad.unsqueeze(1) | pad.unsqueeze(2)
    target = (torch.rand(A.shape) > 0.5).float()

    log_p, log_1mp = blockwise_causal_log_prob_batched(
        A, tp, mode="quiet_softmax", mask_invalid=mask_invalid
    )
    valid = ~mask_invalid
    bce = -(target * log_p + (1 - target) * log_1mp)
    loss = bce[valid].mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(A.grad).all()
