"""Sparse kNN attention parity with the dense path."""

import torch

from trackastra.model.model import TrackingTransformer


def _make_inputs(B=2, N=24, coord_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    # time in [0, window), spatial coords spread so some pairs exceed the cutoff
    t = torch.randint(0, 4, (B, N, 1), generator=g).float()
    yx = 50 * torch.rand((B, N, coord_dim), generator=g)
    coords = torch.cat([t, yx], dim=-1)
    padding_mask = torch.zeros(B, N, dtype=torch.bool)
    padding_mask[:, -3:] = True  # a few padded tokens
    return coords, padding_mask


def _build_pair(coord_dim=2, max_neighbors=128, **kw):
    cfg = dict(
        coord_dim=coord_dim,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
        window=6,
        spatial_pos_cutoff=40,
        attn_positional_bias="rope",
        **kw,
    )
    dense = TrackingTransformer(attn_mode="dense", **cfg).eval()
    sparse = TrackingTransformer(
        attn_mode="sparse", max_neighbors=max_neighbors, **cfg
    ).eval()
    # sparse adds no parameters: it must accept the dense state dict verbatim
    sparse.load_state_dict(dense.state_dict())
    return dense, sparse


def _valid_mask(padding_mask):
    # (B, N, N) entries where both query and key are real tokens; padded
    # rows/cols are masked downstream in the loss and need not match.
    valid = ~padding_mask
    return valid.unsqueeze(2) & valid.unsqueeze(1)


def test_sparse_equals_dense_when_k_ge_n():
    coords, padding_mask = _make_inputs(N=24)
    dense, sparse = _build_pair(max_neighbors=128)  # K >= N
    with torch.no_grad():
        a = dense(coords, padding_mask=padding_mask)
        b = sparse(coords, padding_mask=padding_mask)
    vv = _valid_mask(padding_mask)
    assert torch.allclose(a[vv], b[vv], atol=1e-4), (a - b).abs()[vv].max().item()


def test_sparse_distance_modes_parity():
    for mode in ("v0", "v1"):
        coords, padding_mask = _make_inputs(N=20, seed=1)
        dense, sparse = _build_pair(max_neighbors=64, attn_dist_mode=mode)
        with torch.no_grad():
            a = dense(coords, padding_mask=padding_mask)
            b = sparse(coords, padding_mask=padding_mask)
        vv = _valid_mask(padding_mask)
        assert torch.allclose(a[vv], b[vv], atol=1e-4), (
            mode,
            (a - b).abs()[vv].max().item(),
        )


def test_sparse_runs_with_small_k():
    # K < N: should run and produce finite output (no parity expected)
    coords, padding_mask = _make_inputs(N=40)
    _, sparse = _build_pair(max_neighbors=8)
    with torch.no_grad():
        b = sparse(coords, padding_mask=padding_mask)
    assert torch.isfinite(b).all()
