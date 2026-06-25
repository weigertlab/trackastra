import pytest
import torch
import yaml
from trackastra.model import TrackingTransformer
from trackastra.model.model import DecoderLayer, EncoderLayer
from trackastra.model.model_parts import FeatureMLP, PositionalEncoding
from trackastra.model.sparse_attn import build_knn_index

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


def test_max_distance_drives_attention_pos_enc_and_knn():
    """The single max_distance reaches the attention cutoff, pos-enc, and kNN radius."""
    R = 123
    model = TrackingTransformer(
        coord_dim=2,
        attn_positional_bias="rope",
        attn_mode="sparse",
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_distance=R,
    )

    assert model.config["max_distance"] == R
    assert model.max_distance == R  # feeds build_knn_index in forward
    # reaches the attention layers (spatial mask + rope cutoff)
    assert model.encoder[0].attn.cutoff_spatial == R
    assert model.decoder[0].attn.cutoff_spatial == R

    # reaches the additive positional encoding: the spatial frequency groups depend on
    # max_distance, the temporal group depends on `window` and must not change.
    other = TrackingTransformer(
        coord_dim=2,
        attn_positional_bias="rope",
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_distance=2 * R,
    )
    assert torch.equal(model.pos_embed.freqs[0], other.pos_embed.freqs[0])  # time
    assert not torch.allclose(model.pos_embed.freqs[1], other.pos_embed.freqs[1])  # space


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


def test_sparse_attention_defaults_to_16_neighbors():
    model = TrackingTransformer(coord_dim=2, attn_mode="sparse")

    assert model.max_neighbors == 16
    assert model.config["max_neighbors"] == 16


def test_explicit_max_neighbors_overrides_default():
    """A saved config with max_neighbors=64 must keep 64, not the new default."""
    model = TrackingTransformer(coord_dim=2, attn_mode="sparse", max_neighbors=64)

    assert model.max_neighbors == 64
    assert model.config["max_neighbors"] == 64


def test_build_knn_index_excludes_cutoff_and_padding_with_minus_one():
    # 4 points on a line (time, y, x); the last token is a padded key.
    coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 5.0, 0.0], [0.0, 6.0, 0.0]]]
    )
    padding_mask = torch.tensor([[False, False, False, True]])

    nbr_idx = build_knn_index(coords, padding_mask, cutoff_spatial=2.0, max_neighbors=4)

    assert nbr_idx.shape == (1, 4, 4)
    assert nbr_idx.dtype == torch.int64
    # the padded key (index 3) must never appear as anyone's neighbour
    assert (nbr_idx != 3).all()
    # query 0 (y=0): within dist 2 -> {0, 1}; remaining slots are -1 sentinels
    assert set(nbr_idx[0, 0].tolist()) == {0, 1, -1}
    # query 2 (y=5): only self within dist 2 (key 3 at y=6 is padded out)
    assert set(nbr_idx[0, 2].tolist()) == {2, -1}


def test_build_knn_index_is_batch_specific():
    coords = torch.zeros(2, 3, 3)
    coords[0, :, 1] = torch.tensor([0.0, 1.0, 2.0])  # tightly packed
    coords[1, :, 1] = torch.tensor([0.0, 10.0, 20.0])  # spread out

    nbr_idx = build_knn_index(coords, None, cutoff_spatial=1.5, max_neighbors=3)

    assert set(nbr_idx[0, 0].tolist()) == {0, 1, -1}
    assert set(nbr_idx[1, 0].tolist()) == {0, -1}
    assert not torch.equal(nbr_idx[0], nbr_idx[1])


def test_sparse_model_forward_backward_cpu_fallback():
    torch.manual_seed(0)
    model = TrackingTransformer(
        coord_dim=2,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        attn_mode="sparse",
        max_neighbors=8,
        dropout=0,
    )
    coords = torch.randint(0, 100, (2, 40, 3)).float()

    A = model(coords)
    assert A.shape == (2, 40, 40)
    assert torch.isfinite(A).all()

    A.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_dense_and_sparse_share_state_dict():
    common = dict(
        coord_dim=2,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0,
    )
    dense = TrackingTransformer(attn_mode="dense", **common)
    sparse = TrackingTransformer(attn_mode="sparse", max_neighbors=8, **common)

    # identical parameter names/shapes -> a dense checkpoint loads into sparse
    sparse.load_state_dict(dense.state_dict())
    for key, value in dense.state_dict().items():
        assert torch.equal(value, sparse.state_dict()[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
def test_sparse_model_forward_backward_cuda():
    torch.manual_seed(0)
    model = TrackingTransformer(
        coord_dim=2,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        attn_mode="sparse",
        max_neighbors=16,
        dropout=0,
    ).cuda()
    coords = torch.randint(0, 100, (2, 64, 3)).float().cuda()

    A = model(coords)
    assert A.shape == (2, 64, 64)
    assert torch.isfinite(A).all()

    A.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


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
