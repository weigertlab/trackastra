"""Transformer class."""

import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    _FLASH_ATTN = True and torch.cuda.is_available()
except ImportError:
    flash_attn_varlen_qkvpacked_func = None
    _FLASH_ATTN = False

# if not _FLASH_ATTN:
#     warnings.warn("flash_attn not found or not available for device, falling back to normal attention.")
#     warnings.warn("Install with\n\npip install flash-attn --no-build-isolation\n\n")
    
from .rope import RotaryPositionalEncoding

logger = logging.getLogger(__name__)


def _pos_embed_fourier1d_init(
    cutoff: float = 256, n: int = 32, cutoff_start: float = 1
):
    return (
        torch.exp(torch.linspace(-math.log(cutoff_start), -math.log(cutoff), n))
        .unsqueeze(0)
        .unsqueeze(0)
    )


class FeedForward(nn.Module):
    def __init__(self, d_model, expand: float = 2, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, int(d_model * expand))
        self.fc2 = nn.Linear(int(d_model * expand), d_model, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
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
        self.freqs = nn.ParameterList(
            [
                nn.Parameter(_pos_embed_fourier1d_init(cutoff, n // 2))
                for cutoff, n, cutoff_start in zip(cutoffs, n_pos, cutoffs_start)
            ]
        )

    def forward(self, coords: torch.Tensor):
        _B, _N, D = coords.shape
        assert D == len(self.freqs), f"coords dim {D} must be equal to number of frequencies {len(self.freqs)}"
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


def _bin_init_exp(cutoff: float, n: int):
    return torch.exp(torch.linspace(0, math.log(cutoff + 1), n))


def _bin_init_linear(cutoff: float, n: int):
    return torch.linspace(-cutoff, cutoff, n)


class RelativePositionalBias(nn.Module):
    def __init__(
        self,
        n_head: int,
        cutoff_spatial: float,
        cutoff_temporal: float,
        n_spatial: int = 32,
        n_temporal: int = 16,
    ):
        """Learnt relative positional bias to add to self-attention matrix.

        Spatial bins are exponentially spaced, temporal bins are linearly spaced.

        Args:
            n_head (int): Number of pos bias heads. Equal to number of attention heads
            cutoff_spatial (float): Maximum distance in space.
            cutoff_temporal (float): Maxium distance in time. Equal to window size of transformer.
            n_spatial (int, optional): Number of spatial bins.
            n_temporal (int, optional): Number of temporal bins in each direction. Should be equal to window size. Total = 2 * n_temporal + 1. Defaults to 16.
        """
        super().__init__()
        self._spatial_bins = _bin_init_exp(cutoff_spatial, n_spatial)
        self._temporal_bins = _bin_init_linear(cutoff_temporal, 2 * n_temporal + 1)
        self.register_buffer("spatial_bins", self._spatial_bins)
        self.register_buffer("temporal_bins", self._temporal_bins)
        self.n_spatial = n_spatial
        self.n_head = n_head
        self.bias = nn.Parameter(
            -0.5 + torch.rand((2 * n_temporal + 1) * n_spatial, n_head)
        )

    def forward(self, coords: torch.Tensor):
        _B, _N, _D = coords.shape
        t = coords[..., 0]
        yx = coords[..., 1:]
        temporal_dist = t.unsqueeze(-1) - t.unsqueeze(-2)
        spatial_dist = torch.cdist(yx, yx)

        spatial_idx = torch.bucketize(spatial_dist, self.spatial_bins)
        torch.clamp_(spatial_idx, max=len(self.spatial_bins) - 1)
        temporal_idx = torch.bucketize(temporal_dist, self.temporal_bins)
        torch.clamp_(temporal_idx, max=len(self.temporal_bins) - 1)

        # do some index gymnastics such that backward is not super slow
        # https://discuss.pytorch.org/t/how-to-select-multiple-indexes-over-multiple-dimensions-at-the-same-time/98532/2
        idx = spatial_idx.flatten() + temporal_idx.flatten() * self.n_spatial
        bias = self.bias.index_select(0, idx).view((*spatial_idx.shape, self.n_head))
        # -> B, nH, N, N
        bias = bias.transpose(-1, 1)
        return bias


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
        mode: Literal["bias", "rope", "none"] = "bias",
        attn_dist_mode: str = 'v0'
    ):
        """ 
        
        attn_dist_mode: str
            v0: exponential decay
            v1: exponential decay with cutoff_spatial
            v2: no masking (except padding_mask).
        """ 
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

        if mode == "bias" or mode is True:
            self.pos_bias = RelativePositionalBias(
                n_head=n_head,
                cutoff_spatial=cutoff_spatial,
                cutoff_temporal=cutoff_temporal,
                n_spatial=n_spatial,
                n_temporal=n_temporal,
            )
        elif mode == "rope":
            # each part needs to be divisible by 2
            n_split = 2 * (embed_dim // (2 * (coord_dim + 1) * n_head))

            self.rot_pos_enc = RotaryPositionalEncoding(
                cutoffs=((cutoff_temporal,) + (cutoff_spatial,) * coord_dim),
                n_pos=(embed_dim // n_head - coord_dim * n_split,)
                + (n_split,) * coord_dim,
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

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            coords: torch.Tensor,
            padding_mask: torch.Tensor = None,
        ):
        B, N, D = query.size()
        q = self.q_pro(query)  # (B, N, D)
        k = self.k_pro(key)  # (B, N, D)
        v = self.v_pro(value)  # (B, N, D)
        # (B, nh, N, hs)
        k = k.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        q = q.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, N, self.n_head, D // self.n_head).transpose(1, 2)

        attn_mask = torch.zeros(
            (B, self.n_head, N, N), device=query.device, dtype=q.dtype
        )

        # add negative value but not too large to keep mixed precision loss from becoming nan
        attn_ignore_val = -1e3

        if self.attn_dist_mode != 'v2':
            # spatial cutoff
            yx = coords[..., 1:]
            spatial_dist = torch.cdist(yx, yx)
            spatial_mask = (spatial_dist > self.cutoff_spatial).unsqueeze(1)
            attn_mask.masked_fill_(spatial_mask, attn_ignore_val)

        # dont add positional bias to self-attention if coords is None
        if coords is not None:
            if self._mode == "bias":
                attn_mask = attn_mask + self.pos_bias(coords)
            elif self._mode == "rope":
                q, k = self.rot_pos_enc(q, k, coords)
            else:
                pass

            if self.attn_dist_mode == 'v0':
                dist = torch.cdist(coords, coords, p=2)
                attn_mask += torch.exp(-0.1 * dist.unsqueeze(1))
            elif self.attn_dist_mode == 'v1':
                attn_mask += torch.exp(-5 * spatial_dist.unsqueeze(1) / self.cutoff_spatial)            
            elif self.attn_dist_mode == 'v2':
                pass
            else: 
                raise ValueError(f"Unknown attn_dist_mode {self.attn_dist_mode}")
            
        # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
        if padding_mask is not None:
            ignore_mask = torch.logical_or(
                padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
            ).unsqueeze(1)
            if self.attn_dist_mode == 'v2':
                attn_mask = ~ignore_mask
            else:
                attn_mask.masked_fill_(ignore_mask, attn_ignore_val)

        self.attn_mask = attn_mask.clone()

        if _FLASH_ATTN and self.attn_dist_mode == 'v2' and False:  # Disable for now
            y = compute_attention_with_unpadding(q, k, v, padding_mask)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0
            )

        y = y.transpose(1, 2).contiguous().view(B, N, D)
        # output projection
        y = self.proj(y)
        return y


def compute_attention_with_unpadding(q, k, v, padding_mask):
    """Compute self-attention using flash_attn_varlen_qkvpacked_func with unpadding.

    Args:
        q, k, v: Tensors of shape (B, H, N, D)
        padding_mask: Tensor of shape (B, N), where True means the element should be ignored.

    Returns:
        output: Tensor of shape (B, H, N, D), the result of the attention computation.
    """
    B, H, N, D = q.shape
    assert q.shape == k.shape == v.shape
    assert padding_mask.shape == (B, N)

    # Extract sequence lengths and create cumulative sequence lengths
    valid_tokens_mask = ~padding_mask  # Flip the mask so True means valid
    lens = valid_tokens_mask.sum(dim=-1).tolist()  # Length of each sequence
    cu_seqlens = torch.tensor([0, *torch.cumsum(torch.tensor(lens), dim=0).tolist()], dtype=torch.int32, device=q.device)
    
    # Unpad Q, K, V
    q_unpadded = q.transpose(1, 2)[valid_tokens_mask]  # Shape: (total_tokens, H, D)
    k_unpadded = k.transpose(1, 2)[valid_tokens_mask]  # Shape: (total_tokens, H, D)
    v_unpadded = v.transpose(1, 2)[valid_tokens_mask]  # Shape: (total_tokens, H, D)

    # Stack Q, K, V into a single tensor for FlashAttention
    qkv_unpadded = torch.stack([q_unpadded, k_unpadded, v_unpadded], dim=1)  # Shape: (total_tokens, 3, H, D)

    qkv_unpadded = qkv_unpadded.bfloat16()
    # FlashAttention
    max_seqlen = max(lens)  # Maximum sequence length in the batch
    output_unpadded = flash_attn_varlen_qkvpacked_func(
        qkv_unpadded,  # (total_tokens, 3, H, D)
        cu_seqlens,  # (B + 1,)
        max_seqlen=max_seqlen,
        dropout_p=0.0,  # Set to 0.0 for evaluation
        causal=False,
    )  # Output: (total_tokens, H, D)

    output_unpadded = output_unpadded.to(q.dtype)
    # Re-pad to original dimensions
    output_padded = torch.zeros((B, N, H, D), dtype=output_unpadded.dtype, device=output_unpadded.device)
    output_padded[valid_tokens_mask] = output_unpadded
    output_padded = output_padded.transpose(1, 2)  # Shape: (B, H, N, D)

    return output_padded

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