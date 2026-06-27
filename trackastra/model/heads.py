"""Association readout heads."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .model_parts import FeedForward


class HeadBilinear(nn.Module):
    """Bilinear association readout: per-side FeedForward then a dot product.

    With ``logit_norm`` the two sides are L2-normalized and the cosine
    similarities scaled by a learned temperature (CLIP-style), decoupling logit
    magnitude from ``d_model`` for better-calibrated, more stable associations.
    """

    def __init__(self, d_model: int, logit_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.head_x = FeedForward(d_model, dropout=dropout)
        self.head_y = FeedForward(d_model, dropout=dropout)
        self.logit_norm = logit_norm
        if logit_norm:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, x, y):
        # (B, N, D), (B, M, D) -> (B, N, M)
        x = self.head_x(x)
        y = self.head_y(y)
        if self.logit_norm:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
            # clamp the temperature just above its init (1/0.07 = 14.3): BCE
            # always rewards sharper logits, so if left free it runs away until
            # the softmax becomes a cliff and training NaNs.
            scale = self.logit_scale.exp().clamp(max=50.0)
            return scale * torch.einsum("bnd,bmd->bnm", x, y)
        return torch.einsum("bnd,bmd->bnm", x, y)
