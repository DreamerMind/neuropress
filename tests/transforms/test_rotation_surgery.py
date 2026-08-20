"""Tests for whole-model Hadamard rotation surgery."""

import pytest
import torch
from torch import nn

import neuropress.t2n.qtensors  # noqa: F401
from neuropress.transforms.rotation_surgery import apply_hadamard_rotation


class _RMSNorm(nn.Module):
    """Qwen3.5 semantics: scale is ``1 + weight``."""

    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, dtype=torch.float64) * 0.3)

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * (1.0 + self.weight)


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


class _LinearAttn(nn.Module):
    """Stands in for Qwen3_5GatedDeltaNet: four in-projections, ends in out_proj."""

    def __init__(self, d):
        super().__init__()
        self.in_proj_qkv = nn.Linear(d, d, bias=False)
        self.in_proj_z = nn.Linear(d, d, bias=False)
        self.in_proj_b = nn.Linear(d, 4, bias=False)
        self.in_proj_a = nn.Linear(d, 4, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        mixed = torch.tanh(self.in_proj_qkv(x)) * torch.sigmoid(self.in_proj_z(x))
        gate = self.in_proj_a(x).sum(-1, keepdim=True) + self.in_proj_b(x).sum(
            -1, keepdim=True
        )
        return self.out_proj(mixed * torch.sigmoid(gate))


class _SelfAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        w = torch.softmax(self.q_proj(x) @ self.k_proj(x).transpose(-1, -2), dim=-1)
        return self.o_proj(w @ self.v_proj(x))


class _Layer(nn.Module):
    def __init__(self, d, dff, linear_attn: bool):
        super().__init__()
        if linear_attn:
            self.linear_attn = _LinearAttn(d)
        else:
            self.self_attn = _SelfAttn(d)
        self.mlp = _MLP(d, dff)
        self.input_layernorm = _RMSNorm(d)
        self.post_attention_layernorm = _RMSNorm(d)

    def forward(self, x):
        mixer = getattr(self, "linear_attn", None) or self.self_attn
        x = x + mixer(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class _Inner(nn.Module):
    def __init__(self, vocab, d, dff, n):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(
            [_Layer(d, dff, linear_attn=(i % 2 == 0)) for i in range(n)]
        )
        self.norm = _RMSNorm(d)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _Model(nn.Module):
    """Mirrors Qwen3_5ForCausalLM: tied lm_head, mixed block types."""

    def __init__(self, vocab=16, d=8, dff=16, n=4):
        super().__init__()
        self.model = _Inner(vocab, d, dff, n)

    def forward(self, ids):
        return self.model(ids) @ self.model.embed_tokens.weight.T


def _toy(seed=0):
    torch.manual_seed(seed)
    return _Model().double()


def test_rotation_surgery_preserves_model_output():
    model = _toy()
    ids = torch.tensor([[1, 5, 9, 2]])
    before = model(ids)

    apply_hadamard_rotation(model, seed=17, rotate_down_proj=False)

    assert torch.allclose(before, model(ids), atol=1e-9)


def test_rotation_surgery_with_online_down_proj_preserves_output():
    model = _toy()
    ids = torch.tensor([[1, 5, 9, 2]])
    before = model(ids)

    apply_hadamard_rotation(model, seed=17, rotate_down_proj=True)

    assert torch.allclose(before, model(ids), atol=1e-9)


def test_rotation_surgery_reports_what_it_touched():
    model = _toy()
    summary = apply_hadamard_rotation(model, seed=17, rotate_down_proj=True)
    assert summary["layers_rotated"] == 4
    assert summary["down_proj_wrapped"] == 4
    assert summary["hidden_size"] == 8


def test_rotation_surgery_rejects_unknown_topology():
    with pytest.raises(ValueError, match="topology"):
        apply_hadamard_rotation(nn.Linear(4, 4), seed=1)


# ---------------------------------------------------------------------------
# Low-precision fidelity.  Orthogonal transforms only cancel if forward and
# inverse are computed with enough precision.  The weight fold runs in float64,
# so the runtime ops must not silently degrade to the model's storage dtype: in
# bf16 the residual relative logit error is orders of magnitude larger than in
# float32, which is enough to destroy model quality outright.
# Width matters here - error accumulates with the transform size, so a d=8 toy
# cannot surface it.
# ---------------------------------------------------------------------------


def _wide_toy(dtype):
    torch.manual_seed(0)
    return _Model(vocab=32, d=1024, dff=2048, n=2).to(dtype)


def test_rotation_is_exact_in_float32():
    """In float32 the fold and its runtime inverse must cancel essentially exactly."""
    model = _wide_toy(torch.float32)
    ids = torch.tensor([[1, 5, 9, 2]])
    before = model(ids)

    apply_hadamard_rotation(model, seed=17, rotate_down_proj=True)

    rel = ((model(ids) - before).abs().max() / before.abs().max()).item()
    assert rel < 1e-5, f"float32 rotation should be near-exact, got {rel:.3e}"


def test_rotation_error_in_bfloat16_stays_far_below_quantization_error():
    """bf16 caps fidelity, but the residue must be negligible beside VQ error.

    Storing rotated weights in bf16 is irreversible - the rotation spreads each
    value across many entries and ~8 mantissa bits cannot hold the result - so a
    few-times-1e-3 residue is inherent, not a defect.  The bound that matters is
    that it stays well under the quantizer's own error: VQ at 3.5 bpw is ~9%
    RMS, so 1e-2 leaves an order of magnitude of headroom.  Regressions past
    this would mean the transform itself, not storage, has broken.
    """
    model = _wide_toy(torch.bfloat16)
    ids = torch.tensor([[1, 5, 9, 2]])
    before = model(ids).float()

    apply_hadamard_rotation(model, seed=17, rotate_down_proj=True)
    after = model(ids).float()

    rel = ((after - before).abs().max() / before.abs().max()).item()
    assert rel < 1e-2, f"bf16 rotation error too large: {rel:.3e}"


def test_rotation_survives_final_norm_module_replacement():
    """Kernel lowering swaps every RMSNorm instance, including ``model.norm``.

    Hooks bind to a module instance, so a hook-based un-rotate is silently
    destroyed by that swap - which leaves the residual stream stuck in the
    rotated basis and destroys model quality.  The bracketing must therefore
    survive its target being replaced.
    """
    torch.manual_seed(0)
    model = _Model(vocab=32, d=64, dff=128, n=2).double()
    ids = torch.tensor([[1, 5, 9, 2]])
    before = model(ids)

    apply_hadamard_rotation(model, seed=17, rotate_down_proj=True)

    # Simulate kernel lowering: replace the inner norm with an equivalent one.
    inner_name, inner = _find_final_norm(model)
    replacement = _RMSNorm(64).double()
    replacement.weight.data = inner.weight.data.clone()
    parent_name, _, attr = inner_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attr, replacement)

    assert torch.allclose(before, model(ids), atol=1e-9)


def _find_final_norm(model):
    """Locate the actual RMSNorm behind the model's final norm slot."""
    for name, mod in model.named_modules():
        if isinstance(mod, _RMSNorm) and name.startswith("model.norm"):
            return name, mod
    raise AssertionError("final norm not found")
