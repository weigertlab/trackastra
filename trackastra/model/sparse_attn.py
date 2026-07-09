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


def sparse_bilinear_scores(
    x: torch.Tensor, y: torch.Tensor, nbr_idx: torch.Tensor
) -> torch.Tensor:
    """Neighbour-gathered bilinear scores ``x[n] . y[nbr_idx[n, k]]`` -> (B, N, K).

    Sparse counterpart of ``einsum("bnd,bmd->bnm")`` that never forms the dense
    ``(N, N)`` matrix: Triton kernel on CUDA, gather fallback on CPU/MPS, ``-1``
    slots yield ``0``.
    """
    try:
        from sparse_attn import sparse_einsum, sparse_einsum_gather
    except ImportError as e:
        raise ImportError(_IMPORT_ERROR_MSG) from e
    if x.is_cuda:
        return sparse_einsum(x, y, nbr_idx, bwd="atomics")
    return sparse_einsum_gather(x, y, nbr_idx)


@torch.compiler.disable
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


@torch.compiler.disable
def build_knn_index_per_frame(
    coords: torch.Tensor,
    padding_mask: torch.Tensor | None,
    cutoff_spatial: float,
    max_neighbors: int,
) -> torch.Tensor:
    """Time-aware kNN: K nearest spatial neighbours *within each frame*.

    Unlike :func:`build_knn_index` (a single global budget of K nearest over the
    whole window), this allocates K neighbour slots *per frame*, so a query always
    sees its K spatially-closest detections in every frame of the window instead of
    spending the budget on whichever frame happens to be densest. The query's own
    frame contributes self (spatial distance 0) plus its K-1 nearest same-frame
    detections, so the diagonal is always present. ``K=1`` therefore means "self in
    the own frame, nearest neighbour in every other frame".

    Args:
        coords: (B, N, 1 + coord_dim), column 0 is the (integer) frame time.
        padding_mask: (B, N) bool, True for padded tokens (ignored as keys).
        cutoff_spatial: maximum spatial distance for a valid neighbour (= max_dist).
        max_neighbors: K, the number of neighbour slots *per frame*.

    Returns:
        nbr_idx: (B, N, F * K) int64 neighbour indices, F = number of frames in the
            window. Slots beyond ``cutoff_spatial``, pointing at padded keys, or in
            a frame with fewer than K valid keys are encoded as -1 (sentinel), which
            the ``sparse_attn`` kernels skip.
    """
    N = coords.shape[1]
    t = coords[..., 0]  # (B, N) frame time
    yx = coords[..., 1:]
    spatial_dist = torch.cdist(yx, yx)  # (B, N, N)

    invalid = spatial_dist > cutoff_spatial
    if padding_mask is not None:
        invalid = invalid | padding_mask.unsqueeze(1)  # padded keys

    # Relative integer frame index in [0, F-1] per batch element. Padded tokens are
    # pushed out of the min/max so they cannot shift the frame origin or count.
    if padding_mask is not None:
        t_lo = t.masked_fill(padding_mask, float("inf")).amin(dim=1, keepdim=True)
        t_hi = t.masked_fill(padding_mask, float("-inf")).amax(dim=1, keepdim=True)
    else:
        t_lo = t.amin(dim=1, keepdim=True)
        t_hi = t.amax(dim=1, keepdim=True)
    frame_idx = (t - t_lo).round().long()  # (B, N); garbage for padded (unused)
    n_frames = int((t_hi - t_lo).amax().item()) + 1

    K = min(max_neighbors, N)
    key_frame = frame_idx.unsqueeze(1)  # (B, 1, N): frame of each key
    out = []
    for f in range(n_frames):
        # restrict candidate keys to frame f, then K nearest by spatial distance
        dist_f = spatial_dist.masked_fill(invalid | (key_frame != f), float("inf"))
        vals, idx = torch.topk(dist_f, K, dim=-1, largest=False)  # (B, N, K)
        out.append(idx.masked_fill(torch.isinf(vals), -1))
    return torch.cat(out, dim=-1)  # (B, N, F * K)


@torch.compiler.disable
def build_knn_index_next_frame(
    coords: torch.Tensor,
    padding_mask: torch.Tensor | None,
    cutoff_spatial: float,
    max_neighbors: int,
) -> torch.Tensor:
    """Time-local kNN: K nearest in the same frame (dt=0) and the next frame (dt=1).

    A causal-style restriction of :func:`build_knn_index_per_frame`: instead of all
    frames, a query only sees its K spatially-closest detections in its own frame
    (``dt=0``, which always includes self at distance 0, so the diagonal is kept) and
    in the immediately following frame (``dt=1``). This matches the forward
    associations the loss actually scores (``0 < dt <= delta_cutoff``) when
    ``delta_cutoff=1`` and keeps the slot count fixed at ``2 * K`` regardless of
    window length.

    Args:
        coords: (B, N, 1 + coord_dim), column 0 is the (integer) frame time.
        padding_mask: (B, N) bool, True for padded tokens (ignored as keys).
        cutoff_spatial: maximum spatial distance for a valid neighbour (= max_dist).
        max_neighbors: K, the number of neighbour slots *per frame offset*.

    Returns:
        nbr_idx: (B, N, 2 * K) int64 neighbour indices (same-frame block then
            next-frame block); empty/under-filled slots are -1 sentinels.
    """
    N = coords.shape[1]
    t = coords[..., 0]  # (B, N) frame time
    yx = coords[..., 1:]
    spatial_dist = torch.cdist(yx, yx)  # (B, N, N)

    invalid = spatial_dist > cutoff_spatial
    if padding_mask is not None:
        invalid = invalid | padding_mask.unsqueeze(1)  # padded keys

    # relative frame offset of each key w.r.t. each query: dt[b, i, j] = t_j - t_i
    dt = (t.unsqueeze(1) - t.unsqueeze(2)).round()  # (B, N, N)

    K = min(max_neighbors, N)
    out = []
    for offset in (0, 1):
        dist_o = spatial_dist.masked_fill(invalid | (dt != offset), float("inf"))
        vals, idx = torch.topk(dist_o, K, dim=-1, largest=False)  # (B, N, K)
        out.append(idx.masked_fill(torch.isinf(vals), -1))
    return torch.cat(out, dim=-1)  # (B, N, 2 * K)


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
