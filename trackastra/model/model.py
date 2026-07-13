"""Transformer class."""

import logging
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

# from torch_geometric.nn import GATv2Conv
import yaml
from torch import nn

# NoPositionalEncoding,
from trackastra.utils import blockwise_causal_norm

from .heads import HeadBilinear, HeadEdgeMLP, HeadEdgeStar, HeadSparseBilinear
from .model_parts import (
    _POS_PERIOD_RANGE,
    FeatureEmbedding,
    FeedForward,
    PositionalEncoding,
    RelativePositionalAttention,
)
from .sparse_attn import (
    SparseRelativePositionalAttention,
    build_knn_index,
    build_knn_index_next_frame,
    build_knn_index_per_frame,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for constructing a TrackingTransformer."""

    coord_dim: int = 2
    feat_dim: int = 6
    d_model: int = 256
    pos_embed_per_dim: int = 32
    num_encoder_layers: int = 6
    # None mirrors num_encoder_layers; encoder_only forces it to 0. Resolved to a
    # concrete count inside TrackingTransformer and stored in its saved config.
    num_decoder_layers: int | None = None
    dropout: float = 0.0
    window: int = 4
    spatial_cutoff: float | None = 256
    attn_positional_bias: Literal["rope", "none"] = "rope"
    attn_positional_bias_n_spatial: int = 16
    attn_dist_mode: str = "v1"
    attn_mode: Literal["dense", "sparse"] = "dense"
    max_neighbors: tuple[int, ...] = (16,)
    sparse_knn_mode: Literal["global", "per_frame", "next_frame"] = "per_frame"
    logit_norm: bool = True
    head_mode: Literal["bilinear", "sparse_bilinear", "edge_star", "edge_mlp"] | None = None
    edge_mlp_dim: int | None = None
    causal_norm: Literal["none", "linear", "softmax", "quiet_softmax"] = "quiet_softmax"
    architecture_version: Literal[1, 2] = 2
    data_dim_embed: bool = False
    disable_abs_pos: bool = False
    disable_input_norm: bool = False
    node_head: bool = False
    encoder_only: bool = False
    # width of the auxiliary node-degree heads: out-degree in 0..max_out_degree,
    # in-degree in 0..max_in_degree. Defaults (2, 1) = the biological baseline
    # (up to a 2-way division; a single parent). Raise max_out_degree for >2-way
    # splits. Only used when node_head=True.
    max_in_degree: int = 1
    max_out_degree: int = 2
    model_path: Path | None = None

    def transformer_kwargs(self) -> dict[str, Any]:
        """Return resolved kwargs accepted and saved by TrackingTransformer."""
        num_decoder_layers = self.num_decoder_layers
        if self.encoder_only:
            num_decoder_layers = 0
        elif num_decoder_layers is None:
            num_decoder_layers = self.num_encoder_layers

        max_neighbors = self.max_neighbors
        if len(max_neighbors) == 1:
            max_neighbors = (max_neighbors[0], max_neighbors[0])

        return {
            "coord_dim": self.coord_dim,
            "feat_dim": self.feat_dim,
            "d_model": self.d_model,
            "pos_embed_per_dim": self.pos_embed_per_dim,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "dropout": self.dropout,
            "window": self.window,
            "spatial_cutoff": self.spatial_cutoff,
            "attn_positional_bias": self.attn_positional_bias,
            "attn_positional_bias_n_spatial": self.attn_positional_bias_n_spatial,
            "attn_dist_mode": self.attn_dist_mode,
            "attn_mode": self.attn_mode,
            "max_neighbors": max_neighbors,
            "sparse_knn_mode": self.sparse_knn_mode,
            "logit_norm": self.logit_norm,
            "head_mode": self.head_mode,
            "edge_mlp_dim": self.edge_mlp_dim,
            "causal_norm": self.causal_norm,
            "architecture_version": self.architecture_version,
            "data_dim_embed": self.data_dim_embed,
            "disable_abs_pos": self.disable_abs_pos,
            "disable_input_norm": self.disable_input_norm,
            "node_head": self.node_head,
            "encoder_only": self.encoder_only,
            "max_in_degree": self.max_in_degree,
            "max_out_degree": self.max_out_degree,
        }


class EncoderLayer(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        d_model=256,
        num_heads=4,
        dropout=0.1,
        cutoff_spatial: int = 256,
        window: int = 16,
        positional_bias: Literal["rope", "none"] = "rope",
        positional_bias_n_spatial: int = 32,
        attn_dist_mode: str = "v0",
        attn_mode: Literal["dense", "sparse"] = "dense",
        architecture_version: Literal[1, 2] = 2,
    ):
        super().__init__()
        self.positional_bias = positional_bias
        self.attn_mode = attn_mode
        self.architecture_version = architecture_version
        attn_cls = (
            SparseRelativePositionalAttention
            if attn_mode == "sparse"
            else RelativePositionalAttention
        )
        self.attn = attn_cls(
            coord_dim,
            d_model,
            num_heads,
            cutoff_spatial=cutoff_spatial,
            n_spatial=positional_bias_n_spatial,
            cutoff_temporal=window,
            n_temporal=window,
            dropout=dropout,
            mode=positional_bias,
            attn_dist_mode=attn_dist_mode,
        )
        self.mlp = FeedForward(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        nbr_idx: torch.Tensor = None,
    ):
        # Version 1 checkpoints were trained with the normalized input replacing
        # the residual stream. Version 2 uses the standard pre-norm residual.
        h = self.norm1(x)
        if self.architecture_version == 1:
            x = h
        # setting coords to None disables positional bias
        coords_in = coords if self.positional_bias else None
        if self.attn_mode == "sparse":
            a = self.attn(h, h, h, coords=coords_in, nbr_idx=nbr_idx)
        else:
            a = self.attn(
                h, h, h, coords=coords_in, padding_mask=padding_mask, attn_mask=attn_mask
            )

        x = x + a
        x = x + self.mlp(self.norm2(x))

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        d_model=256,
        num_heads=4,
        dropout=0.1,
        window: int = 16,
        cutoff_spatial: int = 256,
        positional_bias: Literal["rope", "none"] = "rope",
        positional_bias_n_spatial: int = 32,
        attn_dist_mode: str = "v0",
        attn_mode: Literal["dense", "sparse"] = "dense",
        architecture_version: Literal[1, 2] = 2,
    ):
        super().__init__()
        self.positional_bias = positional_bias
        self.attn_mode = attn_mode
        self.architecture_version = architecture_version
        attn_cls = (
            SparseRelativePositionalAttention
            if attn_mode == "sparse"
            else RelativePositionalAttention
        )
        self.attn = attn_cls(
            coord_dim,
            d_model,
            num_heads,
            cutoff_spatial=cutoff_spatial,
            n_spatial=positional_bias_n_spatial,
            cutoff_temporal=window,
            n_temporal=window,
            dropout=dropout,
            mode=positional_bias,
            attn_dist_mode=attn_dist_mode,
        )

        self.mlp = FeedForward(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        nbr_idx: torch.Tensor = None,
    ):
        # Version 1 checkpoints used the normalized query as the residual.
        h = self.norm1(x)
        if self.architecture_version == 1:
            x = h
        y = self.norm2(y)
        # cross attention
        # setting coords to None disables positional bias
        coords_in = coords if self.positional_bias else None
        if self.attn_mode == "sparse":
            a = self.attn(h, y, y, coords=coords_in, nbr_idx=nbr_idx)
        else:
            a = self.attn(
                h, y, y, coords=coords_in, padding_mask=padding_mask, attn_mask=attn_mask
            )

        x = x + a
        x = x + self.mlp(self.norm3(x))

        return x


# class BidirectionalRelativePositionalAttention(RelativePositionalAttention):
#     def forward(
#         self,
#         query1: torch.Tensor,
#         query2: torch.Tensor,
#         coords: torch.Tensor,
#         padding_mask: torch.Tensor = None,
#     ):
#         B, N, D = query1.size()
#         q1 = self.q_pro(query1)  # (B, N, D)
#         q2 = self.q_pro(query2)  # (B, N, D)
#         v1 = self.v_pro(query1)  # (B, N, D)
#         v2 = self.v_pro(query2)  # (B, N, D)

#         # (B, nh, N, hs)
#         q1 = q1.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
#         v1 = v1.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
#         q2 = q2.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
#         v2 = v2.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

#         attn_mask = torch.zeros(
#             (B, self.n_head, N, N), device=query1.device, dtype=q1.dtype
#         )

#         # add negative value but not too large to keep mixed precision loss from becoming nan
#         attn_ignore_val = -1e3

#         # spatial cutoff
#         yx = coords[..., 1:]
#         spatial_dist = torch.cdist(yx, yx)
#         spatial_mask = (spatial_dist > self.cutoff_spatial).unsqueeze(1)
#         attn_mask.masked_fill_(spatial_mask, attn_ignore_val)

#         # dont add positional bias to self-attention if coords is None
#         if coords is not None:
#             if self._mode == "bias":
#                 attn_mask = attn_mask + self.pos_bias(coords)
#             elif self._mode == "rope":
#                 q1, q2 = self.rot_pos_enc(q1, q2, coords)
#             else:
#                 pass

#             dist = torch.cdist(coords, coords, p=2)
#             attn_mask += torch.exp(-0.1 * dist.unsqueeze(1))

#         # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
#         if padding_mask is not None:
#             ignore_mask = torch.logical_or(
#                 padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
#             ).unsqueeze(1)
#             attn_mask.masked_fill_(ignore_mask, attn_ignore_val)

#         self.attn_mask = attn_mask.clone()

#         y1 = nn.functional.scaled_dot_product_attention(
#             q1,
#             q2,
#             v1,
#             attn_mask=attn_mask,
#             dropout_p=self.dropout if self.training else 0,
#         )
#         y2 = nn.functional.scaled_dot_product_attention(
#             q2,
#             q1,
#             v2,
#             attn_mask=attn_mask,
#             dropout_p=self.dropout if self.training else 0,
#         )

#         y1 = y1.transpose(1, 2).contiguous().view(B, N, D)
#         y1 = self.proj(y1)
#         y2 = y2.transpose(1, 2).contiguous().view(B, N, D)
#         y2 = self.proj(y2)
#         return y1, y2


# class BidirectionalCrossAttention(nn.Module):
#     def __init__(
#         self,
#         coord_dim: int = 2,
#         d_model=256,
#         num_heads=4,
#         dropout=0.1,
#         window: int = 16,
#         cutoff_spatial: int = 256,
#         positional_bias: Literal["rope", "none"] = "rope",
#         positional_bias_n_spatial: int = 32,
#     ):
#         super().__init__()
#         self.positional_bias = positional_bias
#         self.attn = BidirectionalRelativePositionalAttention(
#             coord_dim,
#             d_model,
#             num_heads,
#             cutoff_spatial=cutoff_spatial,
#             n_spatial=positional_bias_n_spatial,
#             cutoff_temporal=window,
#             n_temporal=window,
#             dropout=dropout,
#             mode=positional_bias,
#         )

#         self.mlp = FeedForward(d_model)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)

#     def forward(
#         self,
#         x: torch.Tensor,
#         y: torch.Tensor,
#         coords: torch.Tensor,
#         padding_mask: torch.Tensor = None,
#     ):
#         x = self.norm1(x)
#         y = self.norm1(y)

#         # cross attention
#         # setting coords to None disables positional bias
#         x2, y2 = self.attn(
#             x,
#             y,
#             coords=coords if self.positional_bias else None,
#             padding_mask=padding_mask,
#         )
#         # print(torch.norm(x2).item()/torch.norm(x).item())
#         x = x + x2
#         x = x + self.mlp(self.norm2(x))
#         y = y + y2
#         y = y + self.mlp(self.norm2(y))

#         return x, y


# ---------------------------------------------------------------------------
# LEGACY checkpoint shims. Self-contained and key-based (not gated on
# architecture_version) so they stay no-ops for current checkpoints and can be
# deleted wholesale once all published/saved weights are re-exported.
# ---------------------------------------------------------------------------
def _migrate_legacy_state_dict(state: OrderedDict) -> OrderedDict:
    """Adapt pre-``head_mode`` checkpoints to the current module layout.

    The bilinear head used to live at the top level (``head_x.*``, ``head_y.*``);
    it now sits under the ``head`` submodule (``head.head_x.*``). Detection is
    key-based, so this is a no-op for checkpoints already saved with the new
    layout.
    """
    if any(k.startswith(("head_x.", "head_y.")) for k in state):
        logger.info("Migrating legacy top-level head_* state_dict keys under `head.`")
        state = OrderedDict(
            (f"head.{k}" if k.startswith(("head_x.", "head_y.")) else k, v)
            for k, v in state.items()
        )
    return state


def _make_node_head(d_model: int, num_classes: int) -> nn.Module:
    """Small per-node classifier: LayerNorm -> Linear -> GELU -> Linear."""
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Linear(d_model, num_classes),
    )


class TrackingTransformer(torch.nn.Module):
    def __init__(
        self,
        coord_dim: int = 3,
        feat_dim: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int | None = None,
        dropout: float = 0.1,
        pos_embed_per_dim: int = 32,
        window: int = 6,
        spatial_cutoff: int = 256,
        attn_positional_bias: Literal["rope", "none"] = "rope",
        attn_positional_bias_n_spatial: int = 16,
        causal_norm: Literal[
            "none", "linear", "softmax", "quiet_softmax"
        ] = "quiet_softmax",
        attn_dist_mode: str = "v0",
        attn_mode: Literal["dense", "sparse"] = "dense",
        max_neighbors: int | Sequence[int] = 16,
        sparse_knn_mode: Literal["global", "per_frame", "next_frame"] = "per_frame",
        logit_norm: bool = True,
        head_mode: (
            Literal["bilinear", "sparse_bilinear", "edge_star", "edge_mlp"] | None
        ) = None,
        edge_star_dim: int = 128,
        edge_star_n_heads: int = 4,
        edge_star_n_blocks: int = 1,
        edge_mlp_dim: int | None = None,
        architecture_version: Literal[1, 2] = 2,
        data_dim_embed: bool = False,
        disable_abs_pos: bool = False,
        disable_input_norm: bool = False,
        node_head: bool = False,
        encoder_only: bool = False,
        max_in_degree: int = 1,
        max_out_degree: int = 2,
        max_distance: int | None = None,
    ):
        super().__init__()

        if max_distance is not None:
            logger.warning(
                "TrackingTransformer(max_distance=...) is deprecated; use "
                "spatial_cutoff=... instead."
            )
            spatial_cutoff = max_distance

        if architecture_version not in (1, 2):
            raise ValueError(
                f"Unsupported architecture_version={architecture_version}; expected 1 or 2"
            )
        if sparse_knn_mode not in ("global", "per_frame", "next_frame"):
            raise ValueError(
                "sparse_knn_mode must be 'global', 'per_frame' or 'next_frame', "
                f"got {sparse_knn_mode!r}"
            )

        # Resolve num_decoder_layers to a concrete count that is stored in
        # self.config (so saved configs stay unambiguous and reload strict):
        #   encoder_only -> 0 (no decoder; forward feeds y = x to the head),
        #   None         -> num_encoder_layers (symmetric default),
        #   int          -> used as given.
        if encoder_only:
            if num_decoder_layers not in (None, 0):
                logger.warning(
                    "encoder_only=True ignores num_decoder_layers=%s "
                    "(no decoder is built)",
                    num_decoder_layers,
                )
            num_decoder_layers = 0
        elif num_decoder_layers is None:
            num_decoder_layers = num_encoder_layers

        # Normalize max_neighbors to a (lo, hi) pair. A single k becomes (k, k)
        # (fixed K); a (k1, k2) pair samples K~Uniform[k1, k2] per forward during
        # sparse training, while eval/inference always uses the larger hi.
        mn = (max_neighbors,) if isinstance(max_neighbors, int) else tuple(max_neighbors)
        if len(mn) == 1:
            mn = (mn[0], mn[0])
        if len(mn) != 2 or not (1 <= mn[0] <= mn[1]):
            raise ValueError(
                f"max_neighbors must be k or (k1, k2) with 1<=k1<=k2, got {max_neighbors}"
            )
        max_neighbors = mn

        # head_mode selects the association readout. None auto-follows attn_mode
        # ("sparse_bilinear" for sparse attention, else "bilinear"). The sparse
        # readouts ("sparse_bilinear", "edge_star", "edge_mlp") score only against
        # the kNN neighbour list that sparse attention builds, so they are rejected
        # under dense attention. "edge_star"/"edge_mlp" are opt-in (never auto-selected).
        _sparse_heads = ("sparse_bilinear", "edge_star", "edge_mlp")
        if head_mode is None:
            head_mode = "sparse_bilinear" if attn_mode == "sparse" else "bilinear"
        if head_mode not in ("bilinear", *_sparse_heads):
            raise ValueError(
                "head_mode must be None, 'bilinear', 'sparse_bilinear', 'edge_star' "
                f"or 'edge_mlp', got {head_mode!r}"
            )
        if head_mode in _sparse_heads and attn_mode != "sparse":
            raise ValueError(
                f"head_mode={head_mode!r} requires attn_mode='sparse' "
                "(the sparse heads need the kNN neighbour list)"
            )

        self.config = dict(
            coord_dim=coord_dim,
            feat_dim=feat_dim,
            pos_embed_per_dim=pos_embed_per_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            window=window,
            dropout=dropout,
            attn_positional_bias=attn_positional_bias,
            attn_positional_bias_n_spatial=attn_positional_bias_n_spatial,
            spatial_cutoff=spatial_cutoff,
            causal_norm=causal_norm,
            attn_dist_mode=attn_dist_mode,
            attn_mode=attn_mode,
            max_neighbors=max_neighbors,
            sparse_knn_mode=sparse_knn_mode,
            logit_norm=logit_norm,
            head_mode=head_mode,
            edge_star_dim=edge_star_dim,
            edge_star_n_heads=edge_star_n_heads,
            edge_star_n_blocks=edge_star_n_blocks,
            edge_mlp_dim=edge_mlp_dim,
            architecture_version=architecture_version,
            data_dim_embed=data_dim_embed,
            disable_abs_pos=disable_abs_pos,
            disable_input_norm=disable_input_norm,
            node_head=node_head,
            encoder_only=encoder_only,
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
        )
        self.encoder_only = encoder_only
        self.architecture_version = architecture_version
        self._missing_source_ndim_warned = False
        self.attn_mode = attn_mode
        self.max_neighbors = max_neighbors
        self.sparse_knn_mode = sparse_knn_mode
        self.spatial_cutoff = spatial_cutoff
        self.max_distance = spatial_cutoff
        self.attn_dist_mode = attn_dist_mode
        self.disable_abs_pos = disable_abs_pos
        self.disable_input_norm = disable_input_norm

        # TODO remove, alredy present in self.config
        # self.window = window
        # self.feat_dim = feat_dim
        # self.coord_dim = coord_dim

        pos_dim = 0 if disable_abs_pos else (1 + coord_dim) * pos_embed_per_dim
        self.d_model = d_model
        self.feat_dim = feat_dim
        # Coordinates (always present) and the shallow object features are embedded to
        # d_model separately and summed. Features arrive as a fixed-width (F) stack plus
        # a per-column availability mask; a 2-layer MLP over concat(features, mask)
        # embeds the present values and learns a null response for columns a dataset
        # does not provide (missing shape/intensity), so datasets with different feature
        # sets can train one model.
        self.coord_proj = (
            nn.Identity() if disable_abs_pos else nn.Linear(pos_dim, d_model)
        )
        self.feat_embed = (
            FeatureEmbedding(feat_dim, d_model//2, d_model) if feat_dim > 0 else None
        )
        self.data_dim_embed = nn.Embedding(2, d_model) if data_dim_embed else None
        if self.data_dim_embed is not None:
            nn.init.zeros_(self.data_dim_embed.weight)
        self.norm = nn.Identity() if disable_input_norm else nn.LayerNorm(d_model)

        self.encoder = nn.ModuleList([
            EncoderLayer(
                coord_dim,
                d_model,
                nhead,
                dropout,
                window=window,
                cutoff_spatial=spatial_cutoff,
                positional_bias=attn_positional_bias,
                positional_bias_n_spatial=attn_positional_bias_n_spatial,
                attn_dist_mode=attn_dist_mode,
                attn_mode=attn_mode,
                architecture_version=architecture_version,
            )
            for _ in range(num_encoder_layers)
        ])
        # num_decoder_layers is 0 under encoder_only (resolved above), so no
        # decoder parameters are allocated; forward() then feeds the encoder
        # output x into the head as both sides (y = x).
        self.decoder = nn.ModuleList([
            DecoderLayer(
                coord_dim,
                d_model,
                nhead,
                dropout,
                window=window,
                cutoff_spatial=spatial_cutoff,
                positional_bias=attn_positional_bias,
                positional_bias_n_spatial=attn_positional_bias_n_spatial,
                attn_dist_mode=attn_dist_mode,
                attn_mode=attn_mode,
                architecture_version=architecture_version,
            )
            for _ in range(num_decoder_layers)
        ])

        # association readout. "bilinear" (currently the only option) is kept as
        # a Literal so further heads can be added later. In sparse mode the same
        # head is computed over the kNN neighbour list (identical parameters, so a
        # dense checkpoint runs sparse and vice versa).
        self.head_mode = head_mode
        if head_mode == "edge_star":
            self.head = HeadEdgeStar(
                d_model,
                edge_star_dim=edge_star_dim,
                edge_star_n_heads=edge_star_n_heads,
                edge_star_n_blocks=edge_star_n_blocks,
                dropout=dropout,
            )
        elif head_mode == "edge_mlp":
            # edge_mlp_dim defaults to d_model // 2 so the head width scales with the
            # model rather than a fixed constant; reuses the model dropout.
            self.head = HeadEdgeMLP(
                d_model,
                edge_mlp_dim=edge_mlp_dim if edge_mlp_dim is not None else d_model // 2,
                logit_norm=logit_norm,
                dropout=dropout,
            )
        else:
            head_cls = (
                HeadSparseBilinear if head_mode == "sparse_bilinear" else HeadBilinear
            )
            self.head = head_cls(d_model, logit_norm=logit_norm, dropout=dropout)

        # Optional auxiliary node-event heads. Built only when enabled, so a disabled
        # model has no extra parameters and existing checkpoints load strict. The
        # out-degree head reads the source (encoder) representation x, the in-degree
        # head the target (decoder) representation y; see forward().
        self.node_head = node_head
        if node_head:
            if max_out_degree < 1 or max_in_degree < 1:
                raise ValueError("max_in_degree and max_out_degree must be >= 1")
            self.out_degree_head = _make_node_head(d_model, max_out_degree + 1)
            self.in_degree_head = _make_node_head(d_model, max_in_degree + 1)

        self.pos_embed = None
        if not disable_abs_pos:
            # Time keeps the absolute short-period anchor (1 frame); spatial dims
            # scale their short period with spatial_cutoff for scale invariance.
            self.pos_embed = PositionalEncoding(
                cutoffs=(window,) + (spatial_cutoff,) * coord_dim,
                n_pos=(pos_embed_per_dim,) * (1 + coord_dim),
                cutoffs_start=(1.0,)
                + (spatial_cutoff / _POS_PERIOD_RANGE,) * coord_dim,
            )

        # self.pos_embed = NoPositionalEncoding(d=pos_embed_per_dim * (1 + coord_dim))

    def forward(
        self,
        coords,
        features=None,
        feature_mask=None,
        source_ndim=None,
        padding_mask=None,
        return_node_logits=False,
    ):
        assert coords.ndim == 3 and coords.shape[-1] in (3, 4)
        _B, _N, _D = coords.shape

        # disable padded coords (such that it doesnt affect minimum)
        if padding_mask is not None:
            coords = coords.clone()
            coords[padding_mask] = coords.max()

        min_time = coords[:, :, :1].min(dim=1, keepdims=True).values
        if self.architecture_version == 1:
            # Preserve the broadcast subtraction used to train released models.
            coords = coords - min_time
        else:
            # Version 2 removes the temporal offset from the time column only.
            coords = torch.cat(
                [coords[:, :, :1] - min_time, coords[:, :, 1:]], dim=-1
            )

        if self.disable_abs_pos:
            token = coords.new_zeros((_B, _N, self.d_model))
        else:
            token = self.coord_proj(self.pos_embed(coords))

        expected_feature_shape = (_B, _N, self.feat_dim)
        if features is not None and tuple(features.shape) != expected_feature_shape:
            raise ValueError(
                f"features must have shape {expected_feature_shape}, "
                f"got {tuple(features.shape)}"
            )
        if feature_mask is not None and features is None:
            raise ValueError("feature_mask requires features")
        if feature_mask is not None and feature_mask.shape != features.shape:
            raise ValueError(
                "feature_mask must match features shape, got "
                f"{tuple(feature_mask.shape)} and {tuple(features.shape)}"
            )

        if self.feat_embed is not None:
            # Only forward knows _B/_N and the device, so it materializes absent
            # features here: no features and no mask -> all-False mask over zero
            # features, which fires the learned null embedding; features absent with a
            # mask given is invalid. A present features tensor with feature_mask=None is
            # left as-is (FeatureEmbedding treats None as all-present).
            if features is None:
                features = coords.new_zeros((_B, _N, self.feat_dim))
                feature_mask = torch.zeros_like(features, dtype=torch.bool)
            token = token + self.feat_embed(features, feature_mask)
        if self.data_dim_embed is not None:
            if source_ndim is None:
                if not self._missing_source_ndim_warned:
                    warnings.warn(
                        "data_dim_embed=True but source_ndim was omitted; skipping "
                        "the data-dimension embedding",
                        stacklevel=2,
                    )
                    self._missing_source_ndim_warned = True
            else:
                source_ndim = torch.as_tensor(source_ndim, device=coords.device)
                if tuple(source_ndim.shape) != (_B,):
                    raise ValueError(
                        f"source_ndim must have shape ({_B},), got "
                        f"{tuple(source_ndim.shape)}"
                    )
                valid_source_ndim = (source_ndim == 2) | (source_ndim == 3)
                if not bool(valid_source_ndim.all()):
                    raise ValueError(
                        "source_ndim values must be 2 or 3, got "
                        f"{source_ndim.detach().cpu().tolist()}"
                    )
                dim_token = self.data_dim_embed(source_ndim.to(torch.long) - 2)
                token = token + dim_token[:, None, :]
        features = self.norm(token)

        x = features

        # Precompute the layer-independent attention context once and share it
        # across all encoder/decoder layers (depends only on coords/padding).
        # dense: a single additive (B, nH, N, N) mask. sparse: a fixed kNN
        # neighbour list (-1 padded), shared across all layers. K is sampled in
        # [lo, hi] per forward during training (augmentation), hi at eval.
        attn_mask = None
        nbr_idx = None
        if self.attn_mode == "sparse":
            lo, hi = self.max_neighbors
            K = int(torch.randint(lo, hi + 1, (1,)).item()) if self.training else hi
            build = {
                "per_frame": build_knn_index_per_frame,
                "next_frame": build_knn_index_next_frame,
            }.get(self.sparse_knn_mode, build_knn_index)
            nbr_idx = build(
                coords,
                padding_mask,
                self.spatial_cutoff,
                K,
            )
        elif self.encoder:
            a0 = self.encoder[0].attn
            attn_mask = a0.build_attn_mask(
                coords, padding_mask, coords.shape[0], coords.shape[1],
                features.dtype, coords.device,
            )

        # encoder
        for enc in self.encoder:
            x = enc(
                x, coords=coords, padding_mask=padding_mask,
                attn_mask=attn_mask, nbr_idx=nbr_idx,
            )

        if self.encoder_only:
            # No decoder: the head sees the encoder output on both sides (y = x).
            y = x
        else:
            y = features
            # decoder w cross attention
            for dec in self.decoder:
                y = dec(
                    y, x, coords=coords, padding_mask=padding_mask,
                    attn_mask=attn_mask, nbr_idx=nbr_idx,
                )
                # y = dec(y, y, coords=coords, padding_mask=padding_mask)

        # outer product is the association matrix (logits), (B, N, N)
        A = (
            self.head(x, y, nbr_idx)
            if self.head_mode in ("sparse_bilinear", "edge_star", "edge_mlp")
            else self.head(x, y)
        )

        # Always return (A, neighbor_mask). neighbor_mask is None in dense mode; in
        # sparse mode it is a (B, N, N) bool that is True only at kNN pairs in
        # nbr_idx. Training may use it to restrict loss to the sparse neighbourhood.
        neighbor_mask = None
        if nbr_idx is not None:
            b, n, _k = nbr_idx.shape
            neighbor_mask = nbr_idx.new_zeros((b, n, A.shape[-1]), dtype=torch.bool)
            v = nbr_idx >= 0
            bi = torch.arange(b, device=nbr_idx.device).view(b, 1, 1).expand_as(nbr_idx)
            ni = torch.arange(n, device=nbr_idx.device).view(1, n, 1).expand_as(nbr_idx)
            neighbor_mask[bi[v], ni[v], nbr_idx[v]] = True

        if return_node_logits:
            if not self.node_head:
                raise RuntimeError(
                    "return_node_logits=True requires the model to be built with "
                    "node_head=True"
                )
            # out-degree from the source rep x, in-degree from the target rep y
            out_degree_logits = self.out_degree_head(x)  # (B, N, 3)
            in_degree_logits = self.in_degree_head(y)  # (B, N, 2)
            return A, neighbor_mask, out_degree_logits, in_degree_logits

        return A, neighbor_mask

    def normalize_output(
        self,
        A: torch.FloatTensor,
        timepoints: torch.LongTensor,
        coords: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply (parental) softmax, or elementwise sigmoid.

        Args:
            A: Tensor of shape B, N, N
            timepoints: Tensor of shape B, N
            coords: Tensor of shape B, N, (time + n_spatial)
        """
        assert A.ndim == 3
        assert timepoints.ndim == 2
        assert coords.ndim == 3
        assert coords.shape[2] == 1 + self.config["coord_dim"]

        # spatial distances
        dist = torch.cdist(coords[:, :, 1:], coords[:, :, 1:])
        invalid = dist > self.config["spatial_cutoff"]
        invalid = (
            invalid | (timepoints.unsqueeze(1) == -1) | (timepoints.unsqueeze(2) == -1)
        )

        if self.config["causal_norm"] == "none":
            # Spatially distant entries are set to zero
            A = torch.sigmoid(A)
            A[invalid] = 0
        else:
            return torch.stack([
                blockwise_causal_norm(
                    _A, _t, mode=self.config["causal_norm"], mask_invalid=_m
                )
                for _A, _t, _m in zip(A, timepoints, invalid)
            ])
        return A

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(self.config, open(folder / "config.yaml", "w"))
        torch.save(self.state_dict(), folder / "model.pt")

    @staticmethod
    def create(config):
        config = dict(config)
        # The model's single spatial radius was historically named `n`, then
        # `spatial_pos_cutoff`, then `max_distance`. Map the legacy keys so
        # released/pretrained checkpoints still load.
        for legacy in ("max_distance", "spatial_pos_cutoff", "n"):
            if legacy in config:
                config.setdefault("spatial_cutoff", config.pop(legacy))

        model_classes = {
            "default": TrackingTransformer,
        }
        try:
            from trackastra_pretrained_feats import TrackingTransformerwPretrainedFeats

            PRETRAINED_FEATS_INSTALLED = True
        except ImportError:
            PRETRAINED_FEATS_INSTALLED = False
        if PRETRAINED_FEATS_INSTALLED:
            model_classes["pretrained_feats"] = TrackingTransformerwPretrainedFeats

        model_type = (
            "pretrained_feats" if "pretrained_feat_dim" in config else "default"
        )
        # TODO instead we could add explicit field in config to dispatch to different model classes rather than a train arg

        if model_type == "pretrained_feats" and not PRETRAINED_FEATS_INSTALLED:
            raise ImportError(
                "Model was trained with pretrained features, but trackastra_pretrained_feats is not installed. "
                "Please install it with `pip install trackastra[etultra]`."
            )

        return model_classes[model_type](**config)

    @classmethod
    def from_folder(
        cls, folder, map_location=None, args=None, checkpoint_path: str = "model.pt"
    ):
        folder = Path(folder)

        config = yaml.load(open(folder / "config.yaml"), Loader=yaml.FullLoader)
        if "architecture_version" not in config:
            # Released models predate both fields. Current unversioned models
            # include logit_norm in their saved constructor configuration.
            config["architecture_version"] = 1 if "logit_norm" not in config else 2
        # Back-compat: configs published before a param existed must reconstruct
        # the original architecture, not inherit a newer __init__ default that
        # would add/remove parameters and break strict state_dict loading.
        # The training CLI defaults logit_norm to False. Legacy checkpoints also
        # lack its `logit_scale` parameter (e.g. general_2d v0.3.0).
        if "logit_norm" not in config:
            config["logit_norm"] = False
        if args:
            args = vars(args)
            for k, v in config.items():
                errors = []
                if k in args:
                    if config[k] != args[k]:
                        errors.append(
                            f"Loaded model config {k}={config[k]}, but current argument"
                            f" {k}={args[k]}."
                        )
            if errors:
                raise ValueError("\n".join(errors))
        model = cls.create(config)

        # try:
        #     # Try to load from lightning checkpoint first
        #     v_folder = sorted((folder / "tb").glob("version_*"))[version]
        #     checkpoint = sorted((v_folder / "checkpoints").glob("*epoch*.ckpt"))[0]
        #     pl_state_dict = torch.load(checkpoint, map_location=map_location)[
        #         "state_dict"
        #     ]
        #     state_dict = OrderedDict()

        #     # Hack
        #     for k, v in pl_state_dict.items():
        #         if k.startswith("model."):
        #             state_dict[k[6:]] = v
        #         else:
        #             raise ValueError(f"Unexpected key {k} in state_dict")

        #     model.load_state_dict(state_dict)
        #     logger.info(f"Loaded model from {checkpoint}")
        # except:
        #     # Default: Load manually saved model (legacy)

        fpath = folder / checkpoint_path
        logger.info(f"Loading model state from {fpath}")

        state = torch.load(fpath, map_location=map_location, weights_only=True)
        # if state is a checkpoint, we have to extract state_dict
        if "state_dict" in state:
            state = state["state_dict"]
            state = OrderedDict(
                (k[6:], v) for k, v in state.items() if k.startswith("model.")
            )
        state = _migrate_legacy_state_dict(state)  # LEGACY: drop with the shim above
        model.load_state_dict(state)

        return model
