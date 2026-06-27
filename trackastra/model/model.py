"""Transformer class."""

import logging
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch

# from torch_geometric.nn import GATv2Conv
import yaml
from torch import nn

# NoPositionalEncoding,
from trackastra.utils import blockwise_causal_norm

from .heads import HeadBilinear, HeadSparseBilinear
from .model_parts import (
    FeatureMLP,
    FeedForward,
    PositionalEncoding,
    RelativePositionalAttention,
)
from .sparse_attn import SparseRelativePositionalAttention, build_knn_index

logger = logging.getLogger(__name__)


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


class TrackingTransformer(torch.nn.Module):
    def __init__(
        self,
        coord_dim: int = 3,
        feat_dim: int = 0,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        pos_embed_per_dim: int = 32,
        feat_embed_per_dim: int = 1,
        window: int = 6,
        max_distance: int = 256,
        attn_positional_bias: Literal["rope", "none"] = "rope",
        attn_positional_bias_n_spatial: int = 16,
        causal_norm: Literal[
            "none", "linear", "softmax", "quiet_softmax"
        ] = "quiet_softmax",
        attn_dist_mode: str = "v0",
        attn_mode: Literal["dense", "sparse"] = "dense",
        max_neighbors: int | Sequence[int] = 16,
        logit_norm: bool = True,
        assoc_head: Literal["bilinear"] = "bilinear",
        feature_embed_mode: Literal["fourier", "mlp"] = "fourier",
        architecture_version: Literal[1, 2] = 2,
        disable_abs_pos: bool = False,
        disable_input_norm: bool = False,
    ):
        super().__init__()

        if architecture_version not in (1, 2):
            raise ValueError(
                f"Unsupported architecture_version={architecture_version}; expected 1 or 2"
            )

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
            max_distance=max_distance,
            feat_embed_per_dim=feat_embed_per_dim,
            feature_embed_mode=feature_embed_mode,
            causal_norm=causal_norm,
            attn_dist_mode=attn_dist_mode,
            attn_mode=attn_mode,
            max_neighbors=max_neighbors,
            logit_norm=logit_norm,
            assoc_head=assoc_head,
            architecture_version=architecture_version,
            disable_abs_pos=disable_abs_pos,
            disable_input_norm=disable_input_norm,
        )
        self.architecture_version = architecture_version
        self.attn_mode = attn_mode
        self.max_neighbors = max_neighbors
        self.max_distance = max_distance
        self.attn_dist_mode = attn_dist_mode
        self.disable_abs_pos = disable_abs_pos
        self.disable_input_norm = disable_input_norm

        # TODO remove, alredy present in self.config
        # self.window = window
        # self.feat_dim = feat_dim
        # self.coord_dim = coord_dim

        pos_dim = 0 if disable_abs_pos else (1 + coord_dim) * pos_embed_per_dim
        self.proj = nn.Linear(
            pos_dim + feat_dim * feat_embed_per_dim, d_model
        )
        self.norm = nn.Identity() if disable_input_norm else nn.LayerNorm(d_model)

        self.encoder = nn.ModuleList([
            EncoderLayer(
                coord_dim,
                d_model,
                nhead,
                dropout,
                window=window,
                cutoff_spatial=max_distance,
                positional_bias=attn_positional_bias,
                positional_bias_n_spatial=attn_positional_bias_n_spatial,
                attn_dist_mode=attn_dist_mode,
                attn_mode=attn_mode,
                architecture_version=architecture_version,
            )
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(
                coord_dim,
                d_model,
                nhead,
                dropout,
                window=window,
                cutoff_spatial=max_distance,
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
        self.assoc_head = assoc_head
        if assoc_head != "bilinear":
            raise ValueError(f"unknown assoc_head: {assoc_head!r}")
        head_cls = HeadSparseBilinear if attn_mode == "sparse" else HeadBilinear
        self.head = head_cls(d_model, logit_norm=logit_norm, dropout=dropout)

        if feature_embed_mode == "fourier":
            if feat_embed_per_dim > 1:
                self.feat_embed = PositionalEncoding(
                    cutoffs=(1000,) * feat_dim,
                    n_pos=(feat_embed_per_dim,) * feat_dim,
                    cutoffs_start=(0.01,) * feat_dim,
                )
            else:
                self.feat_embed = nn.Identity()
        elif feature_embed_mode == "mlp":
            self.feat_embed = FeatureMLP(
                input_dim=feat_dim,
                output_dim=feat_dim * feat_embed_per_dim,
            )
        else:
            raise ValueError(f"Unknown feature_embed_mode {feature_embed_mode!r}")

        self.pos_embed = None
        if not disable_abs_pos:
            self.pos_embed = PositionalEncoding(
                cutoffs=(window,) + (max_distance,) * coord_dim,
                n_pos=(pos_embed_per_dim,) * (1 + coord_dim),
            )

        # self.pos_embed = NoPositionalEncoding(d=pos_embed_per_dim * (1 + coord_dim))

    def forward(self, coords, features=None, padding_mask=None):
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
            if features is None or features.numel() == 0:
                features = coords.new_empty((_B, _N, 0))
            else:
                features = self.feat_embed(features)
        else:
            pos = self.pos_embed(coords)
            if features is None or features.numel() == 0:
                features = pos
            else:
                features = self.feat_embed(features)
                features = torch.cat((pos, features), axis=-1)

        features = self.proj(features)
        features = self.norm(features)

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
            nbr_idx = build_knn_index(
                coords,
                padding_mask,
                self.max_distance,
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

        y = features
        # decoder w cross attention
        for dec in self.decoder:
            y = dec(
                y, x, coords=coords, padding_mask=padding_mask,
                attn_mask=attn_mask, nbr_idx=nbr_idx,
            )
            # y = dec(y, y, coords=coords, padding_mask=padding_mask)

        # outer product is the association matrix (logits), (B, N, N)
        A = self.head(x, y, nbr_idx) if self.attn_mode == "sparse" else self.head(x, y)

        return A

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
        invalid = dist > self.config["max_distance"]
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
        # `spatial_pos_cutoff`; it is now `max_distance`. Map the legacy keys so
        # released/pretrained checkpoints still load.
        for legacy in ("spatial_pos_cutoff", "n"):
            if legacy in config:
                config.setdefault("max_distance", config.pop(legacy))

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
        model.load_state_dict(state)

        return model
