"""Transformer class."""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Literal

import torch

# from torch_geometric.nn import GATv2Conv
import yaml
from torch import nn

# NoPositionalEncoding,
from trackastra.utils import blockwise_causal_norm

from .model_parts import (
    FeedForward,
    PositionalEncoding,
    RelativePositionalAttention,
)

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
        positional_bias: Literal["bias", "rope", "none"] = "bias",
        positional_bias_n_spatial: int = 32,
        attn_dist_mode: str = "v0",
    ):
        super().__init__()
        self.positional_bias = positional_bias
        self.attn = RelativePositionalAttention(
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
        self.mlp = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        x = self.norm1(x)

        # setting coords to None disables positional bias
        a = self.attn(
            x,
            x,
            x,
            coords=coords if self.positional_bias else None,
            padding_mask=padding_mask,
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
        positional_bias: Literal["bias", "rope", "none"] = "bias",
        positional_bias_n_spatial: int = 32,
        attn_dist_mode: str = "v0",
    ):
        super().__init__()
        self.positional_bias = positional_bias
        self.attn = RelativePositionalAttention(
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

        self.mlp = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        x = self.norm1(x)
        y = self.norm2(y)
        # cross attention
        # setting coords to None disables positional bias
        a = self.attn(
            x,
            y,
            y,
            coords=coords if self.positional_bias else None,
            padding_mask=padding_mask,
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
#         positional_bias: Literal["bias", "rope", "none"] = "bias",
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
        spatial_pos_cutoff: int = 256,
        attn_positional_bias: Literal["bias", "rope", "none"] = "rope",
        attn_positional_bias_n_spatial: int = 16,
        causal_norm: Literal[
            "none", "linear", "softmax", "quiet_softmax"
        ] = "quiet_softmax",
        attn_dist_mode: str = "v0",
    ):
        super().__init__()

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
            spatial_pos_cutoff=spatial_pos_cutoff,
            feat_embed_per_dim=feat_embed_per_dim,
            causal_norm=causal_norm,
            attn_dist_mode=attn_dist_mode,
        )

        # TODO remove, alredy present in self.config
        # self.window = window
        # self.feat_dim = feat_dim
        # self.coord_dim = coord_dim

        self.proj = nn.Linear(
            (1 + coord_dim) * pos_embed_per_dim + feat_dim * feat_embed_per_dim, d_model
        )
        self.norm = nn.LayerNorm(d_model)

        self.encoder = nn.ModuleList(
            [
                EncoderLayer(
                    coord_dim,
                    d_model,
                    nhead,
                    dropout,
                    window=window,
                    cutoff_spatial=spatial_pos_cutoff,
                    positional_bias=attn_positional_bias,
                    positional_bias_n_spatial=attn_positional_bias_n_spatial,
                    attn_dist_mode=attn_dist_mode,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    coord_dim,
                    d_model,
                    nhead,
                    dropout,
                    window=window,
                    cutoff_spatial=spatial_pos_cutoff,
                    positional_bias=attn_positional_bias,
                    positional_bias_n_spatial=attn_positional_bias_n_spatial,
                    attn_dist_mode=attn_dist_mode,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.head_x = FeedForward(d_model)
        self.head_y = FeedForward(d_model)

        if feat_embed_per_dim > 1:
            self.feat_embed = PositionalEncoding(
                cutoffs=(1000,) * feat_dim,
                n_pos=(feat_embed_per_dim,) * feat_dim,
                cutoffs_start=(0.01,) * feat_dim,
            )
        else:
            self.feat_embed = nn.Identity()

        self.pos_embed = PositionalEncoding(
            cutoffs=(window,) + (spatial_pos_cutoff,) * coord_dim,
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

        # remove temporal offset
        min_time = coords[:, :, :1].min(dim=1, keepdims=True).values
        coords = coords - min_time

        pos = self.pos_embed(coords)

        if features is None or features.numel() == 0:
            features = pos
        else:
            features = self.feat_embed(features)
            features = torch.cat((pos, features), axis=-1)

        features = self.proj(features)
        features = self.norm(features)

        x = features

        # encoder
        for enc in self.encoder:
            x = enc(x, coords=coords, padding_mask=padding_mask)

        y = features
        # decoder w cross attention
        for dec in self.decoder:
            y = dec(y, x, coords=coords, padding_mask=padding_mask)
            # y = dec(y, y, coords=coords, padding_mask=padding_mask)

        x = self.head_x(x)
        y = self.head_y(y)

        # outer product is the association matrix (logits)
        A = torch.einsum("bnd,bmd->bnm", x, y)

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
        invalid = dist > self.config["spatial_pos_cutoff"]

        if self.config["causal_norm"] == "none":
            # Spatially distant entries are set to zero
            A = torch.sigmoid(A)
            A[invalid] = 0
        else:
            return torch.stack(
                [
                    blockwise_causal_norm(
                        _A, _t, mode=self.config["causal_norm"], mask_invalid=_m
                    )
                    for _A, _t, _m in zip(A, timepoints, invalid)
                ]
            )
        return A

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        yaml.safe_dump(self.config, open(folder / "config.yaml", "w"))
        torch.save(self.state_dict(), folder / "model.pt")

    @classmethod
    def from_folder(
        cls, folder, map_location=None, args=None, checkpoint_path: str = "model.pt"
    ):
        folder = Path(folder)

        config = yaml.load(open(folder / "config.yaml"), Loader=yaml.FullLoader)
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

        model = cls(**config)

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
