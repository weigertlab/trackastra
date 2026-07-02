"""Sparse kNN attention: mechanism correctness and sentinel handling.

The sparse path delegates the attention itself to the ``sparse_attn`` package (a
Triton kernel on CUDA, a portable gather fallback on CPU; both covered by that
package's own tests). It intentionally drops the dense soft distance bias, so it no
longer matches the dense model numerically. These tests validate the trackastra-side
kNN graph construction and that the model runs end to end in sparse mode.
"""

import pytest
import torch
from trackastra.model.model import TrackingTransformer
from trackastra.model.sparse_attn import (
    build_knn_index,
    build_knn_index_next_frame,
    build_knn_index_per_frame,
)


def _sparse_inputs(B=2, N=40, coord_dim=2, seed=0):
    torch.manual_seed(seed)
    t = torch.randint(0, 4, (B, N, 1)).float()
    yx = 50 * torch.rand((B, N, coord_dim))
    coords = torch.cat([t, yx], dim=-1)
    padding_mask = torch.zeros(B, N, dtype=torch.bool)
    padding_mask[:, -3:] = True
    return coords, padding_mask


def test_build_knn_index_sentinels():
    # node 2 is far from 0/1 (beyond the cutoff); node 3 is padded.
    coords = torch.tensor([[
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 100.0, 0.0],
        [0.0, 2.0, 0.0],
    ]])
    padding_mask = torch.tensor([[False, False, False, True]])
    nbr_idx = build_knn_index(coords, padding_mask, cutoff_spatial=10.0, max_neighbors=4)

    # node 0 must reach nodes {0, 1}; node 2 (too far) and node 3 (padded) excluded.
    row = nbr_idx[0, 0]
    valid = row >= 0
    assert set(row[valid].tolist()) <= {0, 1}
    assert 2 not in row[valid].tolist()  # beyond cutoff
    assert 3 not in row[valid].tolist()  # padded key
    # invalid slots are encoded as the -1 sentinel
    assert torch.all(row[~valid] == -1)


def test_per_frame_knn_keeps_self_and_covers_every_frame():
    # 3 frames x 3 detections; spatial layout irrelevant beyond ordering.
    t = torch.repeat_interleave(torch.arange(3), 3).float().view(1, 9, 1)
    yx = 5 * torch.rand(1, 9, 2)
    coords = torch.cat([t, yx], dim=-1)

    nbr_idx = build_knn_index_per_frame(
        coords, None, cutoff_spatial=256.0, max_neighbors=1
    )
    # K=1 over 3 frames -> exactly F*K = 3 slots per query
    assert nbr_idx.shape == (1, 9, 3)
    # query 0 lives in frame 0: it must include itself (self distance 0) and land
    # exactly one neighbour in each of the three frames.
    row = nbr_idx[0, 0].tolist()
    assert 0 in row  # diagonal preserved
    frames = t[0, :, 0].long()
    assert {int(frames[j]) for j in row} == {0, 1, 2}


def test_per_frame_knn_sentinels_for_empty_frame():
    # frame 1 has no detections -> its bucket must be all -1 sentinels.
    coords = torch.tensor([[
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
    ]])
    nbr_idx = build_knn_index_per_frame(
        coords, None, cutoff_spatial=10.0, max_neighbors=1
    )
    # 3 frame buckets (t in {0,1,2}); the t=1 bucket is empty for every query.
    assert nbr_idx.shape == (1, 3, 3)
    assert torch.all(nbr_idx[:, :, 1] == -1)


def test_next_frame_knn_covers_only_same_and_next_frame():
    # 3 frames x 3 detections.
    t = torch.repeat_interleave(torch.arange(3), 3).float().view(1, 9, 1)
    yx = 5 * torch.rand(1, 9, 2)
    coords = torch.cat([t, yx], dim=-1)

    nbr_idx = build_knn_index_next_frame(
        coords, None, cutoff_spatial=256.0, max_neighbors=1
    )
    # K=1 over 2 offsets -> 2*K = 2 slots per query
    assert nbr_idx.shape == (1, 9, 2)
    frames = t[0, :, 0].long()

    # query 0 (frame 0): same-frame slot keeps self (dt=0), next slot lands in frame 1
    same, nxt = nbr_idx[0, 0].tolist()
    assert same == 0  # diagonal preserved in the dt=0 block
    assert int(frames[nxt]) == 1

    # a query in the last frame has no next frame -> next-frame slot is a sentinel
    same_last, nxt_last = nbr_idx[0, 8].tolist()
    assert int(frames[same_last]) == 2
    assert nxt_last == -1


def test_sparse_model_runs_finite_per_frame():
    coords, padding_mask = _sparse_inputs()
    model = TrackingTransformer(
        coord_dim=2,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
        window=6,
        max_distance=40,
        attn_mode="sparse",
        max_neighbors=1,
        sparse_knn_mode="per_frame",
    ).eval()
    with torch.no_grad():
        a, _ = model(coords, padding_mask=padding_mask)
    assert torch.isfinite(a).all()


def test_sparse_model_runs_finite():
    torch.manual_seed(0)
    B, N, coord_dim = 2, 40, 2
    t = torch.randint(0, 4, (B, N, 1)).float()
    yx = 50 * torch.rand((B, N, coord_dim))
    coords = torch.cat([t, yx], dim=-1)
    padding_mask = torch.zeros(B, N, dtype=torch.bool)
    padding_mask[:, -3:] = True

    model = TrackingTransformer(
        coord_dim=coord_dim,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
        window=6,
        max_distance=40,
        attn_mode="sparse",
        max_neighbors=8,
    ).eval()
    with torch.no_grad():
        a, _ = model(coords, padding_mask=padding_mask)
    assert torch.isfinite(a).all()


@pytest.mark.parametrize("head_mode", ["edge_star", "edge_mlp"])
def test_edge_head_runs_and_requires_sparse(head_mode):
    coords, padding_mask = _sparse_inputs()
    model = TrackingTransformer(
        coord_dim=2,
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.0,
        window=6,
        max_distance=40,
        attn_mode="sparse",
        max_neighbors=8,
        head_mode=head_mode,
        edge_star_dim=32,
    ).eval()
    with torch.no_grad():
        a, neighbor_mask = model(coords, padding_mask=padding_mask)
    assert a.shape == (coords.shape[0], coords.shape[1], coords.shape[1])
    assert torch.isfinite(a).all()
    # sparse head: only kNN pairs carry a real logit (the rest are pinned)
    assert neighbor_mask is not None and neighbor_mask.any()

    # the edge heads need the kNN neighbour list, so dense attention is rejected
    with pytest.raises(ValueError):
        TrackingTransformer(coord_dim=2, attn_mode="dense", head_mode=head_mode)


def test_max_neighbors_normalized_to_pair():
    # a scalar k becomes the fixed pair (k, k)
    m = TrackingTransformer(coord_dim=2, attn_mode="sparse", max_neighbors=8)
    assert m.max_neighbors == (8, 8)
    assert m.config["max_neighbors"] == (8, 8)
    # a range is preserved as (lo, hi)
    m = TrackingTransformer(coord_dim=2, attn_mode="sparse", max_neighbors=(3, 16))
    assert m.max_neighbors == (3, 16)
    # invalid pairs are rejected early
    for bad in [(0, 4), (16, 3), (1, 2, 3)]:
        with pytest.raises(ValueError):
            TrackingTransformer(coord_dim=2, attn_mode="sparse", max_neighbors=bad)


def test_sparse_knn_mode_rejects_typos():
    with pytest.raises(ValueError, match="sparse_knn_mode"):
        TrackingTransformer(coord_dim=2, attn_mode="sparse", sparse_knn_mode="perframe")


def test_sparse_model_runs_with_sampled_k():
    coords, padding_mask = _sparse_inputs()
    model = TrackingTransformer(
        coord_dim=2,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
        window=6,
        max_distance=40,
        attn_mode="sparse",
        max_neighbors=(3, 16),
    )
    # training samples K~[lo, hi] per forward; eval uses hi. Both must stay finite.
    model.train()
    a_train, _ = model(coords, padding_mask=padding_mask)
    assert torch.isfinite(a_train).all()
    model.eval()
    with torch.no_grad():
        a_eval, _ = model(coords, padding_mask=padding_mask)
    assert torch.isfinite(a_eval).all()


def test_max_neighbors_roundtrips_through_folder(tmp_path):
    model = TrackingTransformer(
        coord_dim=2, attn_mode="sparse", max_neighbors=(3, 16)
    ).eval()
    model.save(tmp_path)
    # yaml serializes the pair as a list; reload must normalize it back to (3, 16).
    loaded = TrackingTransformer.from_folder(tmp_path).eval()
    assert loaded.max_neighbors == (3, 16)
    coords, padding_mask = _sparse_inputs()
    with torch.no_grad():
        a, _ = loaded(coords, padding_mask=padding_mask)
    assert torch.isfinite(a).all()
