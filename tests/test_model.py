import pytest
import torch
from trackastra.model import TrackingTransformer
from trackastra.model.model_parts import PositionalEncoding

# Mark all tests in this module as core/inference tests
pytestmark = pytest.mark.core


def test_positional_encoding_cutoffs_start():
    """cutoffs_start must control the highest init frequency (= 1/cutoff_start)."""
    cutoff = 1000.0
    default = PositionalEncoding(cutoffs=(cutoff,), n_pos=(8,))
    small = PositionalEncoding(cutoffs=(cutoff,), n_pos=(8,), cutoffs_start=(0.01,))

    f_default = default.freqs[0].flatten()
    f_small = small.freqs[0].flatten()

    # highest frequency = 1 / cutoff_start
    assert torch.isclose(f_default.max(), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(f_small.max(), torch.tensor(100.0), rtol=1e-3)
    # cutoffs_start must actually change the init (regression for the dropped arg)
    assert not torch.allclose(f_default, f_small)


def test_model():
    torch.manual_seed(0)
    coords = torch.randint(0, 400, (1, 100, 3)).float()

    model = TrackingTransformer(coord_dim=2, attn_positional_bias="rope")

    padding_mask = torch.zeros(1, 100).bool()
    padding_mask[:, -10:] = True
    coords[padding_mask] += 100
    A = model(coords, padding_mask=padding_mask)
    M = torch.logical_or(padding_mask.unsqueeze(1), padding_mask.unsqueeze(2))
    A[M] = 0

    print(A.sum())


def test_model_multichannel_head():
    torch.manual_seed(0)
    coords = torch.randint(0, 400, (2, 60, 3)).float()
    padding_mask = torch.zeros(2, 60).bool()
    padding_mask[:, -10:] = True
    coords[padding_mask] += 100

    model = TrackingTransformer(
        coord_dim=2, assoc_head="multichannel", assoc_channels=8
    )
    A = model(coords, padding_mask=padding_mask)
    assert A.shape == (2, 60, 60)
    assert torch.isfinite(A).all()
