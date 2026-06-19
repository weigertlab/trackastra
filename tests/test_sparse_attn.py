"""Sparse kNN attention: mechanism correctness and sentinel handling.

The sparse path intentionally drops the dense soft distance bias, so it no longer
matches the dense model numerically. These tests instead validate the gather +
masked-softmax mechanism directly, and that the model runs end to end in sparse mode.
"""

import math

import torch

from trackastra.model.model import TrackingTransformer
from trackastra.model.sparse_attn import _sparse_attend, build_knn_index


def test_sparse_attend_matches_full_attention():
    # With K = N and an all-keys neighbour list, sparse attention must reduce to
    # plain scaled-dot-product attention over all keys.
    torch.manual_seed(0)
    B, H, N, hd = 2, 4, 16, 8
    q = torch.randn(B, H, N, hd)
    k = torch.randn(B, H, N, hd)
    v = torch.randn(B, H, N, hd)

    nbr_idx = torch.arange(N).view(1, 1, N).expand(B, N, N).contiguous()
    nbr_bias = torch.zeros(B, N, N)
    out_sparse = _sparse_attend(q, k, v, nbr_idx, nbr_bias, dropout=0.0, training=False)

    scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    out_full = torch.softmax(scores, dim=-1) @ v

    assert torch.allclose(out_sparse, out_full, atol=1e-5), (
        (out_sparse - out_full).abs().max().item()
    )


def test_build_knn_index_sentinels():
    # node 2 is far from 0/1 (beyond the cutoff); node 3 is padded.
    coords = torch.tensor([[
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 100.0, 0.0],
        [0.0, 2.0, 0.0],
    ]])
    padding_mask = torch.tensor([[False, False, False, True]])
    nbr_idx, nbr_bias = build_knn_index(
        coords, padding_mask, cutoff_spatial=10.0, max_neighbors=4
    )

    # node 0 must reach nodes {0, 1, 3-is-padded->excluded, 2-too-far->excluded}
    row = nbr_idx[0, 0]
    valid = row >= 0
    assert set(row[valid].tolist()) <= {0, 1}
    assert 2 not in row[valid].tolist()  # beyond cutoff
    assert 3 not in row[valid].tolist()  # padded key
    # bias is 0 on valid slots, large-negative on sentinels
    assert torch.all(nbr_bias[0, 0][valid] == 0)
    assert torch.all(nbr_bias[0, 0][~valid] <= -1e3 + 1)


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
        spatial_pos_cutoff=40,
        attn_mode="sparse",
        max_neighbors=8,
    ).eval()
    with torch.no_grad():
        a = model(coords, padding_mask=padding_mask)
    assert torch.isfinite(a).all()
