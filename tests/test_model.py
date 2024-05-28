import torch
from trackastra.model import TrackingTransformer


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
