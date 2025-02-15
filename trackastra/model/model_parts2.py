"""Transformer class."""

import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

from x_transformers import Attention
from x_transformers.x_transformers import RotaryEmbedding

class RotaryEmbeddingTrainable(RotaryEmbedding):
    """ 1D RotaryEmbedding with trainable frequencies"""
    def __init__(self, dim:int, base:int=512):
        super().__init__(dim=dim, base=base)
        self.inv_freq = nn.Parameter(self.inv_freq)
        
        
class RotaryEmbeddingND(nn.Module):
    """ ndimensional RotaryEmbedding with trainable frequencies"""
    def __init__(self, ndim:int, dims:tuple[int], bases:tuple[int]):
        super().__init__()
        if not len(dims) == len(bases) == ndim:
            raise ValueError(f"dims and bases must be of length {ndim}")
        if not min(dims) >= 8:
            raise ValueError(f"min of dims must be at least 8")
        self.rotary_pos_embs = nn.ModuleList([
            RotaryEmbeddingTrainable(dim=d, base=b) for d,b in zip(dims, bases)])
   
    def forward(self, coords:torch.Tensor):
        B, N, D = coords.shape
        assert len(self.rotary_pos_embs) == D
        
        freqs, scales = zip(*[emb(coords[..., i]) for i, emb in enumerate(self.rotary_pos_embs)])
        freqs = torch.cat(freqs, dim=-1)
        scales = scales[0]
        return freqs, scales
        
        
        
class FeedForward(nn.Module):
    def __init__(self, d_model, expand: float = 2, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_model, int(d_model * expand), bias=True)
        self.fc2 = nn.Linear(int(d_model * expand), d_model, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class RelativePositionalAttention2(nn.Module):
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
    ):
        super().__init__()

        if not embed_dim % n_head == 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by n_head {n_head}"
            )

        self.n_head = n_head
        self.embed_dim = embed_dim
        self.cutoff_spatial = cutoff_spatial
        
        self.attn = Attention(
            embed_dim,
            dim_head=embed_dim // n_head,
            heads=n_head,
            dropout=dropout,
            flash=True
            )

        bases = (cutoff_temporal,) + (cutoff_spatial,) * coord_dim 
        n_split = 2*(embed_dim // n_head// (2* coord_dim + 1))
        rot_dims = (embed_dim // n_head - coord_dim * n_split,) + (n_split,) * coord_dim
        self.rot_pos_enc = RotaryEmbeddingND(ndim=coord_dim+1, dims=rot_dims, bases=bases)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        y: torch.Tensor = None,
        coords_y: torch.Tensor = None,
        padding_mask: torch.Tensor = None,
    ):
        B, N, D = x.size()
        
        rotary_pos_emb = self.rot_pos_enc(coords)
        
        if y is not None:
            context = y 
            context_rotary_pos_emb = self.rot_pos_enc(coords_y)            
        else:
            context, context_rotary_pos_emb = None, None
            
        
        mask = None
        # if given key_padding_mask = (B,N) then ignore those tokens (e.g. padding tokens)
        if padding_mask is not None:
            mask = torch.logical_and(
                ~padding_mask.unsqueeze(1), ~padding_mask.unsqueeze(2)
            ).unsqueeze(1)
        
        return self.attn(x, rotary_pos_emb=rotary_pos_emb, 
                         context=context, context_rotary_pos_emb=context_rotary_pos_emb, 
                         attn_mask=mask)



if __name__ == "__main__":
    
    ndim=2 
    embed_dim=320
    B,N=1,100 
    
    torch.manual_seed(0)
    coords = torch.randint(0, 400, (B, N, ndim+1)).float()

    model = RelativePositionalAttention2(
        coord_dim=ndim,
        embed_dim=embed_dim,
        n_head=4,
        cutoff_spatial=256,
        cutoff_temporal=16,
    )
    
    x = torch.randn(B, N, embed_dim)
    
    u = model(x, coords)
    
    # print(u.shape)
    
    # model = RotaryEmbeddingND(ndim=ndim+1, dims=(16,32,32), bases=(10,256,256))
    # freqs, scales = model(coords)

    # padding_mask = torch.zeros(1, 100).bool()
    # padding_mask[:, -10:] = True
    # coords[padding_mask] += 100
    # A = model(coords, padding_mask=padding_mask)
    # M = torch.logical_or(padding_mask.unsqueeze(1), padding_mask.unsqueeze(2))
    # A[M] = 0

    # print(A.sum())