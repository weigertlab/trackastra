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
        positional_bias: Literal["bias", "rope", "none"] = "rope",
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


class LearnedRoPERotation(nn.Module):
    def __init__(self, coord_dim, feature_dim, rope_dim=None):
        super().__init__()
        self.rope_dim = rope_dim or feature_dim
        # MLP to predict rotation angle(s) from coords
        self.angle_mlp = nn.Sequential(
            nn.Linear(coord_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.rope_dim // 2)  # one angle per feature pair
        )

    def forward(self, features, coords):
        """features: (B, N, D)
        coords: (B, N, coord_dim).
        """
        B, N, D = features.shape
        assert D % 2 == 0, "Feature dim must be even for RoPE."
        rope_dim = self.rope_dim

        # Predict angles (B, N, rope_dim//2)
        angles = self.angle_mlp(coords[..., :])  # you can select which coords to use
        # Expand to (B, N, rope_dim)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        # Prepare features for rotation
        f = features[..., :rope_dim].reshape(B, N, -1, 2)
        x, y = f[..., 0], f[..., 1]
        # Apply rotation
        x_rot = x * cos - y * sin
        y_rot = x * sin + y * cos
        rotated = torch.stack([x_rot, y_rot], dim=-1).reshape(B, N, rope_dim)
        # Concatenate with the rest of the features if needed
        if rope_dim < D:
            rotated = torch.cat([rotated, features[..., rope_dim:]], dim=-1)
        return rotated


class TrackingTransformer(torch.nn.Module):
    def __init__(
        self,
        coord_dim: int = 3,
        feat_dim: int = 0,
        pretrained_feat_dim: int = 0,
        reduced_pretrained_feat_dim: int = 128,
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
        disable_xy_coords: bool = False,
        disable_all_coords: bool = False,
    ):
        super().__init__()
        
        self.config = dict(
            coord_dim=coord_dim,
            feat_dim=feat_dim,
            pretrained_feat_dim=pretrained_feat_dim,
            reduced_pretrained_feat_dim=reduced_pretrained_feat_dim,
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
            disable_xy_coords=disable_xy_coords,
            disable_all_coords=disable_all_coords,
        )
        
        # TODO temp attr, add as train config arg
        if pretrained_feat_dim > 0:
            self.reduced_pretrained_feat_dim = reduced_pretrained_feat_dim
        else:
            self.reduced_pretrained_feat_dim = 0
        self._return_norms = True
        self.norms = {}

        self._disable_xy_coords = disable_xy_coords
        self._disable_all_coords = disable_all_coords
        
        if self._disable_all_coords:
            coords_proj_dims = 0
        elif self._disable_xy_coords:
            coords_proj_dims = pos_embed_per_dim
        else:
            coords_proj_dims = (1 + coord_dim) * pos_embed_per_dim
        
        feats_proj_dims = feat_dim * feat_embed_per_dim
        
        self.proj = nn.Linear(
            coords_proj_dims + feats_proj_dims + self.reduced_pretrained_feat_dim,
            d_model
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
        
        if pretrained_feat_dim > 0:
            self.ptfeat_proj = nn.Sequential(
                nn.Linear(pretrained_feat_dim, self.reduced_pretrained_feat_dim),
            ) 
            self.ptfeat_norm = nn.LayerNorm(self.reduced_pretrained_feat_dim)
        else:
            self.ptfeat_proj = nn.Identity()
            self.ptfeat_norm = nn.Identity()

        if self._disable_all_coords:
            self.pos_embed = nn.Identity()
            
        elif self._disable_xy_coords:
            self.pos_embed = PositionalEncoding(
                cutoffs=(window,),
                n_pos=(pos_embed_per_dim,),
            )
        else:
            self.pos_embed = PositionalEncoding(
                cutoffs=(window,) + (spatial_pos_cutoff,) * coord_dim,
                n_pos=(pos_embed_per_dim,) * (1 + coord_dim),
            )

        # self.pos_embed = NoPositionalEncoding(d=pos_embed_per_dim * (1 + coord_dim))

    def forward(self, coords, features=None, pretrained_features=None, padding_mask=None):
        assert coords.ndim == 3 and coords.shape[-1] in (3, 4)
        _B, _N, _D = coords.shape
        device = coords.device.type

        # disable padded coords (such that it doesnt affect minimum)
        if padding_mask is not None:
            coords = coords.clone()
            coords[padding_mask] = coords.max()

        # remove temporal offset
        min_time = coords[:, :, :1].min(dim=1, keepdims=True).values
        coords = coords - min_time

        if self._disable_xy_coords:
            coords_feat = coords[:, :, :1].clone()
        else:
            coords_feat = coords.clone()

        if not self._disable_all_coords:
            pos = self.pos_embed(coords_feat)
        else:
            pos = None
            
        if self._return_norms:
            self.norms = {}
            if not self._disable_all_coords:
                self.norms["pos_embed"] = pos.norm(dim=-1).detach().cpu().mean().item()
                self.norms["coords"] = coords_feat.norm(dim=-1).detach().cpu().mean().item()
        
        with torch.amp.autocast(enabled=False, device_type=device):
            # Determine if we have any features to use
            has_features = features is not None and features.numel() > 0
            has_pretrained = pretrained_features is not None and pretrained_features.numel() > 0 and self.config["pretrained_feat_dim"] > 0
            
            if self._return_norms:
                if has_features:
                    self.norms["features"] = features.norm(dim=-1).detach().cpu().mean().item()
                if has_pretrained:
                    self.norms["pretrained_features"] = pretrained_features.norm(dim=-1).detach().cpu().mean().item()

            if not has_features and not has_pretrained:
                if self._disable_all_coords:
                    raise ValueError("features is None and all coords are disabled. Please enable at least one of the two.")
                features_out = pos
            else:
                # Start with features if present, else None
                features_out = self.feat_embed(features) if has_features else None
                if self._return_norms and has_features:
                    self.norms["features_out"] = features_out.norm(dim=-1).detach().cpu().mean().item()

                # Add pretrained features if configured
                if self.config["pretrained_feat_dim"] > 0 and has_pretrained:
                    pt_features = self.ptfeat_proj(pretrained_features)
                    pt_features = self.ptfeat_norm(pt_features)
                    if self._return_norms:
                        self.norms["pt_features_out"] = pt_features.norm(dim=-1).detach().cpu().mean().item()
                    if features_out is not None:
                        features_out = torch.cat((features_out, pt_features), dim=-1)
                    else:
                        features_out = pt_features

                # Add encoded coords if not disabled
                if not self._disable_all_coords:
                    if features_out is not None:
                        features_out = torch.cat((pos, features_out), axis=-1)
                    else:
                        features_out = pos

            features = self.proj(features_out)
            if self._return_norms:
                self.norms["features_cat"] = features_out.norm(dim=-1).detach().cpu().mean().item()
                self.norms["features_proj"] = features.norm(dim=-1).detach().cpu().mean().item()
        # Clamp input when returning to mixed precision
        features = features.clamp(torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)
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
        A = torch.einsum("bnd,bmd->bnm", x, y)  # /math.sqrt(_D)
        
        if torch.any(torch.isnan(A)):
            logger.error("NaN in A")

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
