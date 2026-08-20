"""Tests for Hadamard incoherence processing (rotation transforms)."""

import pytest
import torch
from torch import nn

# neuropress.t2n.qtensors must be imported before neuropress.transforms submodules
# to break a circular import cycle: transforms/__init__ → t2n → transforms.
import neuropress.t2n.qtensors  # noqa: F401
from neuropress.transforms.rotation import (
    fast_hadamard_transform,
    fold_rms_norm_into_linears,
    hadamard_matrix,
    random_hadamard_matrix,
    rotate_attention_vo,
    rotate_residual_stream,
    wrap_linear_with_online_hadamard,
)


def _dense_hadamard(n: int) -> torch.Tensor:
    """Reference orthonormal Hadamard matrix built by explicit Kronecker doubling."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / (n**0.5)


# ---------------------------------------------------------------------------
# fast_hadamard_transform
# ---------------------------------------------------------------------------


def test_fwht_matches_dense_matmul_n4():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    expected = x @ _dense_hadamard(4).T
    assert torch.allclose(fast_hadamard_transform(x), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# hadamard_matrix — order 12 is required for Gemma-4-E2B (1536 = 12 x 128,
# 6144 = 12 x 512); neither is a power of two.
# ---------------------------------------------------------------------------


def test_hadamard_matrix_order_12_is_hadamard():
    h = hadamard_matrix(12)
    assert h.shape == (12, 12)
    assert set(h.unique().tolist()) == {-1.0, 1.0}
    assert torch.allclose(h @ h.T, 12 * torch.eye(12, dtype=h.dtype), atol=1e-10)


# ---------------------------------------------------------------------------
# Non-power-of-two transform sizes (12 * 2^k)
# ---------------------------------------------------------------------------


def test_fwht_matches_dense_matmul_n24():
    """24 = 12 x 2, the smallest size needing the Kronecker path."""
    dense = torch.kron(hadamard_matrix(12), hadamard_matrix(2)) / (24**0.5)
    x = torch.arange(1, 25, dtype=torch.float64).reshape(1, 24)
    assert torch.allclose(fast_hadamard_transform(x), x @ dense.T, atol=1e-10)


def test_fwht_preserves_norm_at_gemma_hidden_size():
    """1536 = 12 x 128; the transform must be orthonormal."""
    x = torch.randn(4, 1536, dtype=torch.float64)
    y = fast_hadamard_transform(x)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-10)


def test_fwht_rejects_unsupported_size():
    with pytest.raises(ValueError):
        fast_hadamard_transform(torch.randn(1, 7))


# ---------------------------------------------------------------------------
# random_hadamard_matrix — the H·D transform actually used for rotation
# ---------------------------------------------------------------------------


def test_random_hadamard_is_orthonormal():
    r = random_hadamard_matrix(256, seed=0)
    assert torch.allclose(r @ r.T, torch.eye(256, dtype=r.dtype), atol=1e-10)


def test_random_hadamard_is_reproducible_from_seed():
    assert torch.equal(
        random_hadamard_matrix(64, seed=7), random_hadamard_matrix(64, seed=7)
    )


def test_random_hadamard_differs_across_seeds():
    assert not torch.equal(
        random_hadamard_matrix(64, seed=1), random_hadamard_matrix(64, seed=2)
    )


# ---------------------------------------------------------------------------
# rotate_attention_vo — the only fully-foldable rotation in a sandwich-norm
# architecture like Gemma-4.  Only sequence-axis mixing sits between v_proj and
# o_proj, so a per-head rotation cancels exactly and costs nothing at runtime.
# ---------------------------------------------------------------------------


def _run_attention(v, o, x, mix, n_kv, n_q, head_dim):
    """Toy GQA attention: mixing acts on the token axis only, never head_dim."""
    n_tok = x.shape[0]
    val = v(x).view(n_tok, n_kv, head_dim)
    val = val.repeat_interleave(n_q // n_kv, dim=1)
    out = torch.einsum("hij,jhd->ihd", mix, val)
    return o(out.reshape(n_tok, n_q * head_dim))


def test_vo_rotation_leaves_attention_output_unchanged():
    torch.manual_seed(0)
    head_dim, n_kv, n_q, hidden, n_tok = 4, 2, 4, 8, 3
    v = nn.Linear(hidden, n_kv * head_dim, bias=False).double()
    o = nn.Linear(n_q * head_dim, hidden, bias=False).double()
    x = torch.randn(n_tok, hidden, dtype=torch.float64)
    mix = torch.randn(n_q, n_tok, n_tok, dtype=torch.float64)

    before = _run_attention(v, o, x, mix, n_kv, n_q, head_dim)
    rotate_attention_vo(v, o, head_dim=head_dim, seed=3)
    after = _run_attention(v, o, x, mix, n_kv, n_q, head_dim)

    assert torch.allclose(before, after, atol=1e-10)


def test_vo_rotation_actually_changes_the_weights():
    """Guards against a no-op implementation trivially passing the invariance test."""
    v = nn.Linear(8, 8, bias=False).double()
    o = nn.Linear(8, 8, bias=False).double()
    v_before = v.weight.clone()
    rotate_attention_vo(v, o, head_dim=4, seed=3)
    assert not torch.allclose(v_before, v.weight)


# ---------------------------------------------------------------------------
# fold_rms_norm_into_linears — prerequisite for any residual-stream rotation.
# A norm's elementwise weight must move into its consumers so the remaining
# pure RMS normalization commutes with an orthogonal R (since ||Rx|| == ||x||).
# Qwen3.5 uses the (1 + weight) convention, so identity means weight == 0.
# ---------------------------------------------------------------------------


class _PlusOneRMSNorm(nn.Module):
    """RMSNorm with Qwen3.5/Gemma semantics: scale is ``1 + weight``."""

    def __init__(self, n: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n, dtype=torch.float64))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * (1.0 + self.weight)


def test_norm_folding_preserves_output():
    torch.manual_seed(0)
    norm = _PlusOneRMSNorm(8)
    norm.weight.data = torch.randn(8, dtype=torch.float64) * 0.3
    lin = nn.Linear(8, 5, bias=False).double()
    x = torch.randn(4, 8, dtype=torch.float64)

    before = lin(norm(x))
    fold_rms_norm_into_linears(norm, [lin], plus_one=True)
    after = lin(norm(x))

    assert torch.allclose(before, after, atol=1e-10)


def test_norm_folding_leaves_identity_norm():
    norm = _PlusOneRMSNorm(8)
    norm.weight.data = torch.randn(8, dtype=torch.float64) * 0.3
    fold_rms_norm_into_linears(
        norm, [nn.Linear(8, 5, bias=False).double()], plus_one=True
    )
    # Under the +1 convention an identity scale is stored as zero.
    assert torch.allclose(
        norm.weight.data, torch.zeros(8, dtype=torch.float64), atol=1e-12
    )


# ---------------------------------------------------------------------------
# rotate_residual_stream — the full-value rotation, available on pre-norm
# architectures like Qwen3.5 (unlike Gemma's sandwich norms).
# ---------------------------------------------------------------------------


class _ToyLayer(nn.Module):
    """Mirrors Qwen3_5DecoderLayer: pre-norm, sublayer ends in a plain Linear."""

    def __init__(self, d, dff):
        super().__init__()
        self.input_layernorm = _PlusOneRMSNorm(d)
        self.qkv = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.post_attention_layernorm = _PlusOneRMSNorm(d)
        self.gate_proj = nn.Linear(d, dff, bias=False)
        self.up_proj = nn.Linear(d, dff, bias=False)
        self.down_proj = nn.Linear(dff, d, bias=False)

    def forward(self, x):
        residual = x
        h = self.input_layernorm(x)
        # tanh stands in for the attention/DeltaNet mixer: whatever happens
        # here, the sublayer ends in out_proj, which is all the fold needs.
        h = self.out_proj(torch.tanh(self.qkv(h)))
        x = residual + h

        residual = x
        h = self.post_attention_layernorm(x)
        h = self.down_proj(
            torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h)
        )
        return residual + h


class _ToyModel(nn.Module):
    def __init__(self, vocab, d, dff, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_ToyLayer(d, dff) for _ in range(n_layers)])
        self.final_norm = _PlusOneRMSNorm(d)
        self.rot = None  # set once the residual stream is rotated

    def forward(self, ids):
        x = self.embed(ids)
        if self.rot is not None:
            x = x @ self.rot.T  # enter the rotated basis
        for layer in self.layers:
            x = layer(x)
        if self.rot is not None:
            x = x @ self.rot  # leave it again, before the un-foldable final norm
        # tied lm_head: the embedding table is reused as the output projection
        return self.final_norm(x) @ self.embed.weight.T


def _make_toy(seed=0):
    torch.manual_seed(seed)
    m = _ToyModel(vocab=16, d=8, dff=16, n_layers=2).double()
    for mod in m.modules():
        if isinstance(mod, _PlusOneRMSNorm):
            mod.weight.data = torch.randn_like(mod.weight.data) * 0.3
    return m


def test_residual_rotation_leaves_model_output_unchanged():
    model = _make_toy()
    ids = torch.tensor([[1, 5, 9, 2]])
    before = model(ids)

    rot = rotate_residual_stream(
        hidden_size=8,
        layers=[
            {
                "input_norm": layer.input_layernorm,
                "input_consumers": [layer.qkv],
                "attn_output": layer.out_proj,
                "post_norm": layer.post_attention_layernorm,
                "post_consumers": [layer.gate_proj, layer.up_proj],
                "mlp_output": layer.down_proj,
            }
            for layer in model.layers
        ],
        seed=11,
        plus_one_norms=True,
    )
    model.rot = rot

    assert torch.allclose(before, model(ids), atol=1e-9)


def test_residual_rotation_actually_rotates_weights():
    model = _make_toy()
    before = model.layers[0].down_proj.weight.clone()
    rotate_residual_stream(
        hidden_size=8,
        layers=[
            {
                "input_norm": model.layers[0].input_layernorm,
                "input_consumers": [model.layers[0].qkv],
                "attn_output": model.layers[0].out_proj,
                "post_norm": model.layers[0].post_attention_layernorm,
                "post_consumers": [model.layers[0].gate_proj, model.layers[0].up_proj],
                "mlp_output": model.layers[0].down_proj,
            }
        ],
        seed=11,
    )
    assert not torch.allclose(before, model.layers[0].down_proj.weight)


# ---------------------------------------------------------------------------
# wrap_linear_with_online_hadamard — the down_proj input rotation.
# The MLP activation (silu(gate) * up) is elementwise nonlinear, so a rotation
# of the intermediate cannot be folded backwards into gate/up_proj.  It must be
# applied online, with R^T folded forward into down_proj.
# ---------------------------------------------------------------------------


def test_online_hadamard_preserves_linear_output():
    torch.manual_seed(0)
    lin = nn.Linear(16, 4, bias=False).double()
    z = torch.randn(3, 16, dtype=torch.float64)

    before = lin(z)
    wrapped = wrap_linear_with_online_hadamard(lin, seed=5)
    after = wrapped(z)

    assert torch.allclose(before, after, atol=1e-10)


def test_online_hadamard_preserves_output_at_qwen_intermediate_size():
    """6144 = 12 x 512 exercises the Kronecker path with a non-symmetric H_12."""
    torch.manual_seed(0)
    lin = nn.Linear(6144, 8, bias=False).double()
    z = torch.randn(2, 6144, dtype=torch.float64)

    before = lin(z)
    after = wrap_linear_with_online_hadamard(lin, seed=5)(z)

    assert torch.allclose(before, after, atol=1e-8)


def test_online_hadamard_rotates_the_weight():
    lin = nn.Linear(16, 4, bias=False).double()
    before = lin.weight.clone()
    wrap_linear_with_online_hadamard(lin, seed=5)
    assert not torch.allclose(before, lin.weight)
