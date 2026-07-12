"""Transformer class."""

import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .rope import RotaryPositionalEncoding

logger = logging.getLogger(__name__)

# Dimensionless period dynamic range of the spatial Fourier / RoPE spectra: the
# shortest resolved period is `cutoff_spatial / _POS_PERIOD_RANGE` and the longest
# is `cutoff_spatial` (= spatial_cutoff). Anchoring the short end to spatial_cutoff
# (instead of a hardcoded 1) makes both spectrum ends scale together with the
# voxel spacing, so a uniformly rescaled dataset sees an identical `coord * freq`
# signal. 256 recovers the historical init exactly at spatial_cutoff == 256.
_POS_PERIOD_RANGE = 256.0


def _pos_embed_fourier1d_init(
    cutoff: float = 256, n: int = 32, cutoff_start: float = 1
):
    return (
        torch.exp(torch.linspace(-math.log(cutoff_start), -math.log(cutoff), n))
        .unsqueeze(0)
        .unsqueeze(0)
    )


class FeedForward(nn.Module):
    def __init__(
        self, d_model, expand: float = 2, bias: bool = True, dropout: float = 0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, int(d_model * expand))
        self.fc2 = nn.Linear(int(d_model * expand), d_model, bias=bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class MaskedRunningNorm(nn.Module):
    """Per-channel standardization using mask-aware running statistics.

    Normalizes each feature channel to zero mean / unit variance from running buffers
    that are updated (in training) only over unmasked entries. The estimator is the
    count-weighted cumulative average over every detection seen so far, so it converges
    to the dataset statistic rather than tracking the current batch. ``momentum_floor``
    bounds the update weight from below, which caps the memory at ``1 / momentum_floor``
    batches; without it the buffers would freeze and a fine-tune on data with different
    object sizes could never re-adapt.

    Weighting by ``count`` (not per batch) matters here because detections per window,
    and hence valid samples per channel, vary widely across movies; equal-per-batch
    weighting would bias the mean toward sparse sequences.

    Normalization always uses the running buffers (not per-batch stats), so the
    transform is identical in train and eval. Masked entries are re-zeroed on output so
    they never participate.
    """

    _version = 2

    def __init__(self, dim: int, momentum_floor: float = 1e-3, eps: float = 1e-5):
        super().__init__()
        self.momentum_floor = momentum_floor
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("sq_mean", torch.ones(dim))
        # float64: float32 stops incrementing past its 24-bit mantissa (~1.7e7 samples).
        self.register_buffer("count_seen", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("initialized", torch.zeros(dim, dtype=torch.bool))
        self._dbg_steps = 0  # debug: throttle running-stat logging

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "count_seen"
        if key not in state_dict:
            # Checkpoints predating count_seen resume as a floored EMA instead of
            # snapping to a fresh cumulative average on the first batch.
            state_dict[key] = torch.full_like(self.count_seen, 1e12)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def update(self, x, mask):
        dims = tuple(range(x.ndim - 1))
        mask = mask.bool()
        w = mask.to(x.dtype)
        x_masked = torch.where(mask, x, torch.zeros((), dtype=x.dtype, device=x.device))

        count = w.sum(dim=dims)
        sum_x = x_masked.sum(dim=dims)
        sum_sq = x_masked.square().sum(dim=dims)

        # Aggregate the sufficient statistics across DDP ranks so every process
        # updates from the full global batch (no-op when not distributed).
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            packed = torch.stack((count, sum_x, sum_sq))
            torch.distributed.all_reduce(packed)
            count, sum_x, sum_sq = packed.unbind(0)

        valid = count > 0
        denom = count.clamp_min(1)
        batch_mean = sum_x / denom
        batch_sq_mean = sum_sq / denom

        self.count_seen += count.double()
        # Count-weighted cumulative average, i.e. the exact sample mean over all
        # detections seen. Clamped to [floor, 1]: the floor keeps the estimator from
        # freezing, the upper bound guards the lerp against an extrapolating weight
        # when count_seen was seeded by the legacy-checkpoint path above.
        alpha = (
            (count.double() / self.count_seen.clamp_min(1.0))
            .clamp(self.momentum_floor, 1.0)
            .to(self.mean.dtype)
        )

        first = valid & ~self.initialized
        update = valid & self.initialized

        self.mean[first] = batch_mean[first].to(self.mean.dtype)
        self.sq_mean[first] = batch_sq_mean[first].to(self.sq_mean.dtype)

        # Indexed assignment (not in-place lerp_ on a boolean-indexed copy, which
        # would silently no-op) so the EMA actually advances.
        self.mean[update] = torch.lerp(
            self.mean[update], batch_mean[update].to(self.mean.dtype), alpha[update]
        )
        self.sq_mean[update] = torch.lerp(
            self.sq_mean[update],
            batch_sq_mean[update].to(self.sq_mean.dtype),
            alpha[update],
        )

        self.initialized |= valid

        # debug: log running stats every 100 updates (remove once tuned)
        self._dbg_steps += 1
        if self._dbg_steps % 1000 == 0:
            std = (self.sq_mean - self.mean.square()).clamp_min(0).sqrt()
            logger.info(
                "MaskedRunningNorm step=%d mean=%s std=%s",
                self._dbg_steps,
                [round(v, 3) for v in self.mean.tolist()],
                [round(v, 3) for v in std.tolist()],
            )

    def forward(self, x, mask):
        mask = mask.bool()
        if self.training:
            self.update(x.detach(), mask)

        mean = self.mean.to(x.dtype)
        var = (self.sq_mean - self.mean.square()).clamp_min(0).to(x.dtype)

        y = (x - mean) * torch.rsqrt(var + self.eps)
        return y.masked_fill(~mask, 0)


class FeatureEmbedding(nn.Module):
    """Embed masked scalar object features.

    ``features`` is a real ``(B, N, feat_dim)`` tensor; ``feature_mask`` is either the
    matching boolean tensor or ``None`` (all present). Features are per-channel
    standardized with mask-aware running statistics (``MaskedRunningNorm``), which also
    re-zeros masked columns so their values never participate; an all-False row fires
    the model's learned null embedding. ``features=None`` is resolved by the caller
    (see ``TrackingTransformer.forward``) and never reaches here.
    """

    def __init__(self, feat_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        if feat_dim <= 0 or output_dim <= 0:
            raise ValueError("FeatureEmbedding dimensions must be positive")
        self.feat_dim = feat_dim
        self.norm = MaskedRunningNorm(feat_dim)
        self.fc1 = nn.Linear(2 * feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()

    def forward(self, features, feature_mask=None):
        if features.ndim != 3 or features.shape[-1] != self.feat_dim:
            raise ValueError(
                f"features must have shape (B, N, {self.feat_dim}), "
                f"got {tuple(features.shape)}"
            )
        if feature_mask is None:
            feature_mask = torch.ones_like(features, dtype=torch.bool)
        elif feature_mask.shape != features.shape:
            raise ValueError(
                "feature_mask must match features shape, got "
                f"{tuple(feature_mask.shape)} and {tuple(features.shape)}"
            )
        # Standardizes present channels and re-zeros masked ones.
        features = self.norm(features, feature_mask)

        x = torch.cat((features, feature_mask.to(features.dtype)), dim=-1)
        return self.fc2(self.act(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        cutoffs: tuple[float] = (256,),
        n_pos: tuple[int] = (32,),
        cutoffs_start=None,
    ):
        """Positional encoding with given cutoff and number of frequencies for each dimension.
        number of dimension is inferred from the length of cutoffs and n_pos.
        """
        super().__init__()
        if cutoffs_start is None:
            cutoffs_start = (1,) * len(cutoffs)

        assert len(cutoffs) == len(n_pos)
        self.freqs = nn.ParameterList([
            nn.Parameter(_pos_embed_fourier1d_init(cutoff, n // 2, cutoff_start))
            for cutoff, n, cutoff_start in zip(cutoffs, n_pos, cutoffs_start)
        ])

    def forward(self, coords: torch.Tensor):
        _B, _N, D = coords.shape
        assert D == len(self.freqs)
        embed = torch.cat(
            tuple(
                torch.cat(
                    (
                        torch.sin(0.5 * math.pi * x.unsqueeze(-1) * freq),
                        torch.cos(0.5 * math.pi * x.unsqueeze(-1) * freq),
                    ),
                    axis=-1,
                )
                / math.sqrt(len(freq))
                for x, freq in zip(coords.moveaxis(-1, 0), self.freqs)
            ),
            axis=-1,
        )

        return embed


class NoPositionalEncoding(nn.Module):
    def __init__(self, d):
        """One learnable input token that ignores positional information."""
        super().__init__()
        self.d = d
        # self.token = nn.Parameter(torch.randn(d))

    def forward(self, coords: torch.Tensor):
        B, N, _ = coords.shape
        return (
            # torch.ones((B, N, self.d), device=coords.device) * 0.1
            # torch.randn((1, 1, self.d), device=coords.device).expand(B, N, -1) * 0.01
            torch.randn((B, N, self.d), device=coords.device) * 0.01
            + torch.randn((1, 1, self.d), device=coords.device).expand(B, N, -1) * 0.1
        )
        # return self.token.view(1, 1, -1).expand(B, N, -1)


class RelativePositionalAttention(nn.Module):
    def __init__(
        self,
        coord_dim: int,
        embed_dim: int,
        n_head: int,
        cutoff_spatial: float = 256,
        cutoff_temporal: float = 16,
        n_spatial: int = 32,
        n_temporal: int = 16,
        dropout: float = 0.0,
        mode: Literal["rope", "none"] = "rope",
        attn_dist_mode: str = "v0",
    ):
        super().__init__()

        if not embed_dim % (2 * n_head) == 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by 2 times n_head {2 * n_head}"
            )

        # qkv projection
        self.q_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_pro = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_pro = nn.Linear(embed_dim, embed_dim, bias=True)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        # regularization
        self.dropout = dropout
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.cutoff_spatial = cutoff_spatial
        self.attn_dist_mode = attn_dist_mode

        if mode == "rope":
            # each part needs to be divisible by 2
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))

            # Time keeps the absolute short-period anchor (1 frame); spatial dims
            # scale their short period with cutoff_spatial for scale invariance.
            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
                cutoffs_start=(1.0,)
                + (cutoff_spatial / _POS_PERIOD_RANGE,) * coord_dim,
            )
        elif mode == "none":
            pass
        elif mode is None or mode is False:
            logger.warning(
                "attn_positional_bias is not set (None or False), no positional bias."
            )
            pass
        else:
            raise ValueError(f"Unknown mode {mode}")

        self._mode = mode

    def build_attn_mask(
        self,
        coords: torch.Tensor,
        padding_mask: torch.Tensor,
        B: int,
        N: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Additive (B, 1, N, N) attention mask: spatial cutoff + distance bias
        + padding.

        None of the terms are head-dependent (the only per-head term was the
        learned positional bias, now removed), so the mask is built with a
        singleton head dim and broadcast over heads by scaled_dot_product_attention
        - this is n_head x smaller than a full (B, n_head, N, N) mask. It depends
        only on coords/padding (not the layer features), so it is identical across
        layers and precomputed once and shared (see TrackingTransformer.forward).
        The rope rotation of q/k is layer-local and handled in forward, not here.
        """
        yx = coords[..., 1:]
        spatial_dist = torch.cdist(yx, yx)  # (B, N, N)
        too_far = spatial_dist > self.cutoff_spatial

        if self.attn_dist_mode == "none":
            # Boolean mask (True = attend): only the hard exclusions (spatial
            # cutoff + padded keys), no soft distance bias. A bool mask is
            # n_bytes smaller than the float additive mask and lets SDPA use the
            # memory-efficient kernel. Always allow self-attention so no query
            # row is fully masked (a fully-masked row would softmax to NaN).
            allowed = ~too_far
            if padding_mask is not None:
                allowed = allowed & ~padding_mask.unsqueeze(1)  # padded keys
            eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
            allowed = allowed | eye
            return allowed.unsqueeze(1)  # (B, 1, N, N) bool

        # Float additive mask: spatial cutoff + soft distance bias + padding.
        attn_mask = torch.zeros((B, 1, N, N), device=device, dtype=dtype)
        # add negative value but not too large to keep mixed precision loss from becoming nan
        attn_ignore_val = -1e3
        attn_mask.masked_fill_(too_far.unsqueeze(1), attn_ignore_val)

        if self.attn_dist_mode == "v0":
            dist = torch.cdist(coords, coords, p=2)
            attn_mask += torch.exp(-0.1 * dist.unsqueeze(1))
        elif self.attn_dist_mode == "v1":
            attn_mask += torch.exp(-5 * spatial_dist.unsqueeze(1) / self.cutoff_spatial)
        else:
            raise ValueError(f"Unknown attn_dist_mode {self.attn_dist_mode}")

        # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
        if padding_mask is not None:
            ignore_mask = torch.logical_or(
                padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
            ).unsqueeze(1)
            attn_mask.masked_fill_(ignore_mask, attn_ignore_val)

        return attn_mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
    ):
        B, N, D = query.size()
        q = self.q_pro(query)  # (B, N, D)
        k = self.k_pro(key)  # (B, N, D)
        v = self.v_pro(value)  # (B, N, D)
        # (B, nh, N, hs)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

        # Build the mask unless a precomputed one is passed in (shared across
        # layers since it is layer-independent). Only float (additive) masks are
        # cast to q's dtype; a boolean mask (attn_dist_mode="none") is passed
        # through unchanged for SDPA.
        if attn_mask is None:
            attn_mask = self.build_attn_mask(coords, padding_mask, B, N, q.dtype, query.device)
        elif attn_mask.dtype.is_floating_point and attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(q.dtype)

        # rope rotation is layer-local (applied to this layer's q/k)
        if coords is not None and self._mode == "rope":
            q, k = self.rot_pos_enc(q, k, coords)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0
        )

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        # output projection
        y = self.proj(y)

        return y
