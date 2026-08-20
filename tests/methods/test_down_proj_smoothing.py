"""Tests for down_proj-only SmoothQuant scale migration."""

import torch
from torch import nn

import neuropress.t2n.qtensors  # noqa: F401
from neuropress.methods.smooth_quant import smooth_down_proj


class _MLP(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.gate_proj = nn.Linear(d, dff, bias=False)
        self.up_proj = nn.Linear(d, dff, bias=False)
        self.down_proj = nn.Linear(dff, d, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class _Layer(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.mlp = _MLP(d, dff)

    def forward(self, x):
        return x + self.mlp(x)


class _Model(nn.Module):
    """Minimal stand-in exposing the Qwen3.5 mlp naming."""

    def __init__(self, vocab=16, d=32, dff=64, n=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_Layer(d, dff) for _ in range(n)])

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return x


def _toy():
    torch.manual_seed(0)
    return _Model().double()


# ---------------------------------------------------------------------------
# down_proj is the only MLP layer with no absorbable predecessor: its input is
# silu(gate) * up.  But `up` enters that product *linearly*, so scaling
# up_proj's output rows by s exactly multiplies down_proj's input by s, making
# the migration exact without touching any norm.
# ---------------------------------------------------------------------------


def test_smoothing_preserves_model_output():
    model = _toy()
    ids = torch.tensor([[1, 5, 9, 2]])
    before = model(ids).clone()

    smooth_down_proj(model, [{"ids": ids}], alpha=0.5)

    assert torch.allclose(before, model(ids), atol=1e-9)


def test_smoothing_actually_rescales_down_proj():
    model = _toy()
    ids = torch.tensor([[1, 5, 9, 2]])
    before = model.layers[0].mlp.down_proj.weight.clone()

    smooth_down_proj(model, [{"ids": ids}], alpha=0.5)

    assert not torch.allclose(before, model.layers[0].mlp.down_proj.weight)


def test_smoothing_touches_every_down_proj():
    model = _toy()
    ids = torch.tensor([[1, 5, 9, 2]])
    scales = smooth_down_proj(model, [{"ids": ids}], alpha=0.5)
    assert len(scales) == 2
    assert all(k.endswith("mlp.down_proj") for k in scales)


def test_alpha_zero_and_one_both_preserve_output():
    """The migration must be exact at both extremes of the balance parameter."""
    for alpha in (0.0, 1.0):
        model = _toy()
        ids = torch.tensor([[1, 5, 9, 2]])
        before = model(ids).clone()
        smooth_down_proj(model, [{"ids": ids}], alpha=alpha)
        assert torch.allclose(before, model(ids), atol=1e-9), f"alpha={alpha}"


# ---------------------------------------------------------------------------
# Geometric-mean normalization.  With grouping="all" a single codebook is fitted
# to vectors pooled across every layer, so any per-layer scale shift moves that
# layer off the pooled distribution.  Unnormalized smoothing shrinks down_proj's
# VQ-vector RMS well below the pool, which costs real quality.  Normalizing s to
# geometric mean 1 keeps the layer commensurate while preserving the *relative*
# per-channel structure, which is the part that carries the exploitable
# anisotropy.
# ---------------------------------------------------------------------------


def _biased_toy():
    """Toy whose activations vary enough per channel to produce lopsided scales."""
    torch.manual_seed(3)
    m = _Model(vocab=16, d=32, dff=64, n=2).double()
    for layer in m.layers:
        w = layer.mlp.up_proj.weight.data
        w *= torch.logspace(-1, 1, w.shape[0], dtype=torch.float64).unsqueeze(1)
    return m


def test_normalized_smoothing_preserves_model_output():
    model = _biased_toy()
    ids = torch.tensor([[1, 5, 9, 2]])
    before = model(ids).clone()

    smooth_down_proj(model, [{"ids": ids}], alpha=0.5, normalize_geomean=True)

    assert torch.allclose(before, model(ids), atol=1e-9)


def test_normalization_keeps_layer_scale_commensurate():
    ids = torch.tensor([[1, 5, 9, 2]])

    plain = _biased_toy()
    before = plain.layers[0].mlp.down_proj.weight.data.abs().median().item()
    smooth_down_proj(plain, [{"ids": ids}], alpha=0.5, normalize_geomean=False)
    plain_ratio = (
        plain.layers[0].mlp.down_proj.weight.data.abs().median().item() / before
    )

    norm = _biased_toy()
    smooth_down_proj(norm, [{"ids": ids}], alpha=0.5, normalize_geomean=True)
    norm_ratio = norm.layers[0].mlp.down_proj.weight.data.abs().median().item() / before

    # normalized must stay near the original magnitude; unnormalized need not
    assert abs(norm_ratio - 1.0) < 0.35, f"normalized shifted scale {norm_ratio:.3f}x"
    assert abs(norm_ratio - 1.0) < abs(plain_ratio - 1.0)


def test_normalization_keeps_relative_channel_structure():
    """Normalization may only remove a global factor, not the per-channel shape."""
    ids = torch.tensor([[1, 5, 9, 2]])
    a = _biased_toy()
    b = _biased_toy()
    sa = smooth_down_proj(a, [{"ids": ids}], alpha=0.5, normalize_geomean=False)
    sb = smooth_down_proj(b, [{"ids": ids}], alpha=0.5, normalize_geomean=True)
    k = "layers.0.mlp.down_proj"
    ratio = sa[k] / sb[k]
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-9)
