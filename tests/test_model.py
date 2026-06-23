import pytest
import torch
from trackastra.model import TrackingTransformer
from trackastra.model.model_parts import FeatureMLP, PositionalEncoding

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


def test_mlp_feature_embedding_runs_without_fourier_features():
    torch.manual_seed(0)
    model = TrackingTransformer(
        coord_dim=2,
        feat_dim=6,
        feat_embed_per_dim=8,
        feature_embed_mode="mlp",
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0,
    )
    coords = torch.rand((2, 12, 3))
    features = torch.rand((2, 12, 6))
    output = model(coords, features)

    assert isinstance(model.feat_embed, FeatureMLP)
    assert model.feat_embed.fc1.in_features == 6
    assert model.feat_embed.fc1.out_features == 48
    assert model.feat_embed.fc2.out_features == 48
    assert model.config["feature_embed_mode"] == "mlp"
    assert output.shape == (2, 12, 12)
    assert torch.isfinite(output).all()


def test_legacy_model_config_defaults_to_fourier_features():
    model = TrackingTransformer(coord_dim=2, feat_dim=6, feat_embed_per_dim=8)
    legacy_config = model.config.copy()
    legacy_config.pop("feature_embed_mode")

    restored = TrackingTransformer.create(legacy_config)

    assert isinstance(restored.feat_embed, PositionalEncoding)
    assert restored.config["feature_embed_mode"] == "fourier"


def test_mlp_feature_embedding_survives_save_load(tmp_path):
    model = TrackingTransformer(
        coord_dim=2,
        feat_dim=6,
        feat_embed_per_dim=8,
        feature_embed_mode="mlp",
        d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    model.save(tmp_path)

    restored = TrackingTransformer.from_folder(tmp_path)

    assert isinstance(restored.feat_embed, FeatureMLP)
    assert restored.config["feature_embed_mode"] == "mlp"
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


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


@pytest.mark.parametrize("assoc_head", ["bilinear", "multichannel"])
def test_dropout_is_applied_to_attention_and_mlp(assoc_head):
    model = TrackingTransformer(
        coord_dim=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.2,
        assoc_head=assoc_head,
    )

    assert model.encoder[0].attn.dropout == pytest.approx(0.2)
    assert model.encoder[0].mlp.dropout.p == pytest.approx(0.2)
    assert model.decoder[0].mlp.dropout.p == pytest.approx(0.2)
    if assoc_head == "bilinear":
        assert model.head_x.dropout.p == pytest.approx(0.2)
        assert model.head_y.dropout.p == pytest.approx(0.2)
    else:
        assert model.pair_head.mlp[2].p == pytest.approx(0.2)
