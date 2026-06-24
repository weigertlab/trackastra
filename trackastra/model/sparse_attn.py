"""Sparse kNN attention backed by the ``sparse_attn`` Triton package.

The dense additive attention mask is effectively local: each node only attends
to keys within ``cutoff_spatial`` (and the temporal window), every other entry is
killed by a large negative bias. This module exploits that by attending only over
a fixed list of the K nearest neighbours per node, turning the O(N^2) attention
into O(N*K).

The neighbour list is built once per forward (it depends only on coords/padding,
not on the layer features) and shared across all encoder/decoder layers, mirroring
the shared additive mask in the dense path. The attention itself is delegated to
the ``sparse_attn`` package: a Triton FlashAttention-style kernel on CUDA and a
pure-PyTorch gather fallback on CPU/MPS. Both skip ``-1`` padded neighbour slots,
so no additive bias mask is needed.
"""

import torch

from .model_parts import RelativePositionalAttention

_IMPORT_ERROR_MSG = (
    "attn_mode='sparse' requires the 'sparse_attn' package, which is not installed. "
    "Install it with `pip install trackastra[sparse]` (or `pip install "
    "'sparse_attn @ git+https://github.com/maweigert/sparse_attn.git'`)."
)


def _load_sparse_attn():
    """Lazily import the sparse_attn kernels with an actionable error.

    Keeping the import lazy means dense inference stays usable on machines without
    Triton; only requesting ``attn_mode='sparse'`` pulls the dependency in.
    """
    try:
        from sparse_attn import sparse_attention, sparse_attention_gather
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e
    return sparse_attention, sparse_attention_gather


def build_knn_index(
    coords: torch.Tensor,
    padding_mask: torch.Tensor | None,
    cutoff_spatial: float,
    max_neighbors: int,
) -> torch.Tensor:
    """Build the fixed kNN neighbour-index list for sparse attention.

    Unlike the dense path, no soft distance bias is applied: the kNN neighbourhood
    already encodes spatial locality and rope encodes relative position, so the
    hand-tuned ``exp(-dist)`` weighting is a redundant crutch that would just bias
    attention toward the spatially-closest neighbour (not necessarily the match).

    Args:
        coords: (B, N, 1 + coord_dim), column 0 is time, the rest are spatial.
        padding_mask: (B, N) bool, True for padded tokens (ignored as keys).
        cutoff_spatial: maximum spatial distance for a valid neighbour (= max_dist).
        max_neighbors: K, the number of neighbour slots per node.

    Returns:
        nbr_idx: (B, N, K) int64 neighbour indices; slots beyond ``cutoff_spatial``
            or pointing at padded/absent keys are encoded as -1 (sentinel), which
            the ``sparse_attn`` kernels skip.
    """
    N = coords.shape[1]
    K = min(max_neighbors, N)

    yx = coords[..., 1:]
    spatial_dist = torch.cdist(yx, yx)  # (B, N, N)

    invalid = spatial_dist > cutoff_spatial
    if padding_mask is not None:
        invalid = invalid | padding_mask.unsqueeze(1)  # padded keys

    # K nearest by spatial distance, invalid pairs pushed to +inf
    dist_sel = spatial_dist.masked_fill(invalid, float("inf"))
    vals, idx = torch.topk(dist_sel, K, dim=-1, largest=False)  # (B, N, K)

    # set sentinel index -1 wherever the slot exceeds the cutoff / is padded
    valid = ~torch.isinf(vals)
    return idx.masked_fill(~valid, -1)


class SparseRelativePositionalAttention(RelativePositionalAttention):
    """kNN-sparse variant of RelativePositionalAttention.

    Reuses the same q/k/v projections and rope (no extra parameters - a model
    trained dense can run sparse and vice versa). Instead of a full (N, N) score
    matrix it attends only over the K nearest neighbours of each query via the
    ``sparse_attn`` kernel, which runs an online (FlashAttention-style) softmax
    and recomputes the attention probabilities in backward, so no (B, H, N, K, hd)
    gathered tensor is ever stored.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        coords: torch.Tensor,
        nbr_idx: torch.Tensor,
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

        sparse_attention, sparse_attention_gather = _load_sparse_attn()
        if q.is_cuda:
            # Triton kernel: scale is 1/sqrt(hd), -1 neighbours skipped, no dropout.
            out = sparse_attention(q, k, v, nbr_idx, bwd="atomics")  # (B, H, N, hd)
        else:
            # portable O(N*K) gather fallback for CPU/MPS
            out = sparse_attention_gather(q, k, v, nbr_idx)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.proj(out)
