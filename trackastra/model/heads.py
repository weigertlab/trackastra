"""Association readout heads."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .model_parts import FeedForward
from .sparse_attn import sparse_bilinear_scores

# logit for non-neighbour pairs in the scattered dense matrix: large-negative so
# they vanish under the downstream sigmoid / blockwise-softmax, yet finite (well
# inside fp16 range) to avoid NaNs in the loss.
NO_EDGE_LOGIT = -1e4


def _scatter_neighbours_to_dense(scores, nbr_idx, m):
    """Scatter ``(B, N, K)`` neighbour scores into a dense ``(B, N, M)`` matrix.

    Non-neighbour pairs are pinned to ``NO_EDGE_LOGIT``; ``-1`` neighbour slots are
    skipped. topk neighbour indices are unique per row, so there are no collisions.
    """
    b, n, _k = nbr_idx.shape
    A = scores.new_full((b, n, m), NO_EDGE_LOGIT)
    valid = nbr_idx >= 0
    bi = torch.arange(b, device=scores.device).view(b, 1, 1).expand_as(nbr_idx)
    ni = torch.arange(n, device=scores.device).view(1, n, 1).expand_as(nbr_idx)
    A.index_put_((bi[valid], ni[valid], nbr_idx[valid]), scores[valid])
    return A


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


class HeadSparseBilinear(nn.Module):
    """Sparse drop-in for :class:`HeadBilinear` used with ``attn_mode='sparse'``.

    Same parameters as ``HeadBilinear`` (a dense checkpoint loads unchanged), but
    the dot product runs only against each node's K nearest neighbours via
    ``sparse_einsum``, then the ``(B, N, K)`` scores are scattered into a dense
    ``(B, N, N)`` matrix (non-neighbours = ``NO_EDGE_LOGIT``) so nothing
    downstream changes. A match outside the K-neighbourhood cannot be predicted.
    """

    # non-neighbour pair logit; see module-level NO_EDGE_LOGIT.
    NO_EDGE_LOGIT = NO_EDGE_LOGIT

    def __init__(self, d_model: int, logit_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.head_x = FeedForward(d_model, dropout=dropout)
        self.head_y = FeedForward(d_model, dropout=dropout)
        self.logit_norm = logit_norm
        if logit_norm:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, x, y, nbr_idx):
        # x, y: (B, N, D); nbr_idx: (B, N, K) into y -> A: (B, N, N)
        x = self.head_x(x)
        y = self.head_y(y)
        if self.logit_norm:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
            scale = self.logit_scale.exp().clamp(max=50.0)  # see HeadBilinear
            scores = scale * sparse_bilinear_scores(x, y, nbr_idx)
        else:
            scores = sparse_bilinear_scores(x, y, nbr_idx)
        return _scatter_neighbours_to_dense(scores, nbr_idx, y.shape[1])


class HeadEdgeStar(nn.Module):
    """Bidirectional edge-star readout for ``head_mode='edge_star'`` (sparse only).

    Each candidate edge is contextualised against its outgoing star (edges sharing the
    source) and incoming star (edges sharing the target) via
    :class:`sparse_attn.BidirectionalEdgeStar`, so competing successors/predecessors
    inform the decision. ``c_out=1``, scattered to dense like the other sparse heads, so
    the loss/tracking are unchanged. Adds params absent from the bilinear heads (no
    warm-start from a bilinear checkpoint).
    """

    def __init__(
        self,
        d_model: int,
        edge_star_dim: int = 128,
        edge_star_n_heads: int = 4,
        edge_star_n_blocks: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # lazy import: edge_star ships with the optional 'sparse_attn' package.
        from .sparse_attn import _IMPORT_ERROR_MSG

        try:
            from sparse_attn import BidirectionalEdgeStar, build_incoming
        except ImportError as e:
            raise ImportError(_IMPORT_ERROR_MSG) from e
        self._build_incoming = build_incoming
        self.block = BidirectionalEdgeStar(
            dim_in=d_model,
            dim_edge=edge_star_dim,
            n_heads=edge_star_n_heads,
            n_blocks=edge_star_n_blocks,
            c_out=1,
        )

    def forward(self, x, y, nbr_idx):
        # x, y: (B, N, D); nbr_idx: (B, N, K) into y -> A: (B, N, N)
        n = y.shape[1]
        inc_idx, inc_mask = self._build_incoming(nbr_idx, n)
        logits = self.block(x, y, nbr_idx, inc_idx, inc_mask)  # (B, N, K, 1)
        return _scatter_neighbours_to_dense(logits[..., 0], nbr_idx, n)


class HeadEdgeMLP(nn.Module):
    """Context-free per-edge MLP readout for ``head_mode='edge_mlp'`` (sparse only).

    Scores each candidate edge independently via :class:`sparse_attn.EdgeMLPClassifier`
    (asymmetric edge embedding, no star context); the cheap middle rung between
    ``sparse_bilinear`` and ``edge_star``. ``c_out=1``, scattered to dense like the
    other sparse heads, so the loss/tracking are unchanged.

    Uses the ``"affine"`` edge feature ``[x_i, y_j, y_j - x_i]`` (no bilinear ``x*y``
    term), whose first MLP layer is separable and so hoisted to the node level: unlike
    ``sparse_bilinear`` (whose two FeedForwards run over ``N`` nodes), the per-edge MLP
    otherwise pays its first matmul over ``N*K`` edges. ``"affine"`` keeps the matmul at
    node level, closing most of that throughput gap.
    """

    def __init__(self, d_model: int, edge_mlp_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        # lazy import: edge_mlp ships with the optional 'sparse_attn' package.
        from .sparse_attn import _IMPORT_ERROR_MSG

        try:
            from sparse_attn import EdgeMLPClassifier
        except ImportError as e:
            raise ImportError(_IMPORT_ERROR_MSG) from e
        self.block = EdgeMLPClassifier(
            dim_in=d_model, dim_edge=edge_mlp_dim, c_out=1, edge_feats="affine"
        )
        if dropout > 0:
            # EdgeMLPClassifier has no dropout slot; reuse the model dropout by
            # inserting it after the GELU of its (Linear, GELU, Linear) edge MLP,
            # so edge_mlp adds no new hyperparameter of its own.
            mlp = self.block.edge_mlp
            self.block.edge_mlp = nn.Sequential(
                mlp[0], mlp[1], nn.Dropout(dropout), mlp[2]
            )

    def forward(self, x, y, nbr_idx):
        # x, y: (B, N, D); nbr_idx: (B, N, K) into y -> A: (B, N, N)
        logits = self.block(x, y, nbr_idx)  # (B, N, K, 1)
        return _scatter_neighbours_to_dense(logits[..., 0], nbr_idx, y.shape[1])
