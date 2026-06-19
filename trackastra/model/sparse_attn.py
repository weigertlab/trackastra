"""Sparse kNN attention.

The dense additive attention mask is effectively local: each node only attends
to keys within ``cutoff_spatial`` (and the temporal window), every other entry is
killed by a large negative bias. This module exploits that by attending only over
a fixed list of the K nearest neighbours per node, turning the O(N^2) attention
into O(N*K).

The neighbour list is built once per forward (it depends only on coords/padding,
not on the layer features) and shared across all encoder/decoder layers, mirroring
the shared additive mask in the dense path.
"""

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model_parts import RelativePositionalAttention

# large negative additive value for masked-out slots (matches dense build_attn_mask)
_NEG = -1e3


def _sparse_attend(q, k, v, nbr_idx, nbr_bias, dropout, training):
    """Core kNN attention: gather neighbours, softmax over K, weight values.

    Kept as a free function so it can be gradient-checkpointed during training
    (the gathered (B, H, N, K, hd) tensors are then recomputed in backward
    instead of being stored, which is what keeps sparse competitive on memory).
    """
    B, H, N, hd = q.shape
    K = nbr_idx.shape[-1]

    # gather the K neighbours of each query along the key dim -> (B, H, N, K, hd)
    idx_flat = nbr_idx.clamp(min=0).view(B, 1, N * K, 1).expand(B, H, N * K, hd)
    k_g = torch.gather(k, 2, idx_flat).view(B, H, N, K, hd)
    v_g = torch.gather(v, 2, idx_flat).view(B, H, N, K, hd)

    # scores (B, H, N, K) + additive distance bias (broadcast over heads)
    scores = (q.unsqueeze(3) * k_g).sum(-1) / math.sqrt(hd)
    scores = scores + nbr_bias.unsqueeze(1)

    attn = torch.softmax(scores, dim=-1)
    if training and dropout > 0:
        attn = F.dropout(attn, p=dropout)

    return (attn.unsqueeze(-1) * v_g).sum(3)  # (B, H, N, hd)


def build_knn_index(
    coords: torch.Tensor,
    padding_mask: torch.Tensor | None,
    cutoff_spatial: float,
    max_neighbors: int,
    attn_dist_mode: str = "v0",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the fixed kNN neighbour list and its additive distance bias.

    Args:
        coords: (B, N, 1 + coord_dim), column 0 is time, the rest are spatial.
        padding_mask: (B, N) bool, True for padded tokens (ignored as keys).
        cutoff_spatial: maximum spatial distance for a valid neighbour (= max_dist).
        max_neighbors: K, the number of neighbour slots per node.
        attn_dist_mode: "v0" or "v1", matching RelativePositionalAttention.

    Returns:
        nbr_idx: (B, N, K) int64 neighbour indices; slots beyond ``cutoff_spatial``
            or pointing at padded/absent keys are set to -1 (sentinel).
        nbr_bias: (B, N, K) additive distance bias for each slot; sentinel slots
            are set to a large negative value so softmax ignores them.
    """
    B, N, _ = coords.shape
    K = min(max_neighbors, N)

    yx = coords[..., 1:]
    spatial_dist = torch.cdist(yx, yx)  # (B, N, N)

    invalid = spatial_dist > cutoff_spatial
    if padding_mask is not None:
        invalid = invalid | padding_mask.unsqueeze(1)  # padded keys

    # K nearest by spatial distance, invalid pairs pushed to +inf
    dist_sel = spatial_dist.masked_fill(invalid, float("inf"))
    vals, idx = torch.topk(dist_sel, K, dim=-1, largest=False)  # (B, N, K)

    valid = ~torch.isinf(vals)
    # set sentinel index -1 wherever the slot exceeds the cutoff / is padded
    nbr_idx = idx.masked_fill(~valid, -1)

    # additive distance bias, identical to the dense path for the kept entries
    if attn_dist_mode == "v0":
        full_dist = torch.cdist(coords, coords)  # (B, N, N), includes time
        bias = torch.exp(-0.1 * torch.gather(full_dist, 2, idx))
    elif attn_dist_mode == "v1":
        bias = torch.exp(-5 * vals / cutoff_spatial)
    else:
        raise ValueError(f"Unknown attn_dist_mode {attn_dist_mode}")

    nbr_bias = bias.masked_fill(~valid, _NEG)
    return nbr_idx, nbr_bias


class SparseRelativePositionalAttention(RelativePositionalAttention):
    """kNN-sparse variant of RelativePositionalAttention.

    Reuses the same q/k/v projections and rope (no extra parameters - a model
    trained dense can run sparse and vice versa). Instead of a full (N, N) score
    matrix it gathers the K neighbours of each query and softmaxes over them.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor,
        nbr_idx: torch.Tensor,
        nbr_bias: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        B, N, D = query.size()
        H = self.n_head
        hd = D // H

        q = self.q_pro(query).view(B, N, H, hd).transpose(1, 2)  # (B, H, N, hd)
        k = self.k_pro(key).view(B, N, H, hd).transpose(1, 2)
        v = self.v_pro(value).view(B, N, H, hd).transpose(1, 2)

        # rope rotation is layer-local (applied to this layer's q/k)
        if coords is not None and self._mode == "rope":
            q, k = self.rot_pos_enc(q, k, coords)

        if self.training:
            # recompute the gathered neighbours in backward instead of storing them
            out = checkpoint(
                _sparse_attend,
                q, k, v, nbr_idx, nbr_bias, self.dropout, True,
                use_reentrant=False,
            )
        else:
            out = _sparse_attend(q, k, v, nbr_idx, nbr_bias, self.dropout, False)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.proj(out)
