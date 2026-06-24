import pytest
import torch
import yaml
from trackastra.model import TrackingTransformer
from trackastra.model.model import DecoderLayer, EncoderLayer
from trackastra.model.model_parts import FeatureMLP, PositionalEncoding

# Mark all tests in this module as core/inference tests
pytestmark = pytest.mark.core


class _ZeroModule(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return torch.zeros_like(x)


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


@pytest.mark.parametrize("layer_cls", [EncoderLayer, DecoderLayer])
def test_architecture_version_controls_residual_semantics(layer_cls):
    x = torch.tensor([[[1.0, 2.0, 4.0, 8.0], [3.0, 5.0, 7.0, 9.0]]])
    y = x + 1
    coords = torch.zeros(1, 2, 3)

    def run(version):
        layer = layer_cls(
            coord_dim=2,
            d_model=4,
            num_heads=2,
            dropout=0,
            positional_bias="none",
            architecture_version=version,
        )
        layer.attn = _ZeroModule()
        layer.mlp = _ZeroModule()
        if layer_cls is DecoderLayer:
            output = layer(x, y, coords)
        else:
            output = layer(x, coords)
        return layer, output

    legacy_layer, legacy_output = run(1)
    _, current_output = run(2)

    assert torch.allclose(legacy_output, legacy_layer.norm1(x))
    assert torch.equal(current_output, x)


@pytest.mark.parametrize("version", [1, 2])
def test_architecture_version_controls_coordinate_normalization(version):
    model = TrackingTransformer(
        coord_dim=2,
        d_model=8,
        nhead=2,
        num_encoder_layers=0,
        num_decoder_layers=0,
        pos_embed_per_dim=2,
        dropout=0,
        logit_norm=False,
        architecture_version=version,
    )
    coords = torch.tensor([[[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]]])
    captured = []
    handle = model.pos_embed.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().clone())
    )

    model(coords)
    handle.remove()

    expected = coords - coords[:, :, :1].min(dim=1, keepdim=True).values
    if version == 2:
        expected[..., 1:] = coords[..., 1:]
    assert torch.equal(captured[0], expected)


@pytest.mark.parametrize("legacy", [True, False])
def test_unversioned_config_infers_architecture_version(tmp_path, legacy):
    version = 1 if legacy else 2
    model = TrackingTransformer(
        coord_dim=2,
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        pos_embed_per_dim=2,
        logit_norm=not legacy,
        architecture_version=version,
    )
    model.save(tmp_path)

    config_path = tmp_path / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.pop("architecture_version")
    if legacy:
        config.pop("logit_norm")
    config_path.write_text(yaml.safe_dump(config))

    restored = TrackingTransformer.from_folder(tmp_path)

    assert restored.architecture_version == version
    assert restored.config["architecture_version"] == version


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
