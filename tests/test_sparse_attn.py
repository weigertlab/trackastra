"""Sparse kNN attention: mechanism correctness and sentinel handling.

The sparse path delegates the attention itself to the ``sparse_attn`` package (a
Triton kernel on CUDA, a portable gather fallback on CPU; both covered by that
package's own tests). It intentionally drops the dense soft distance bias, so it no
longer matches the dense model numerically. These tests validate the trackastra-side
kNN graph construction and that the model runs end to end in sparse mode.
"""

import torch
from trackastra.model.model import TrackingTransformer
from trackastra.model.sparse_attn import build_knn_index


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
