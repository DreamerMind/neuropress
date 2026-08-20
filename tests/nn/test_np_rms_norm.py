"""Tests for NPRMSNorm and linearize_rms_norms."""

import pytest
import torch
import torch.nn as nn

from neuropress.nn.np_rms_norm import (
    NPRMSNorm,
    _is_unfused_rms_norm,
    _uses_plus_one_convention,
    linearize_rms_norms,
)

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Numerically equivalent unfused reference (LLaMA/Mistral-style)."""
    x_f = x.float()
    rms = x_f.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x_f * rms * weight.float()).to(x.dtype)


def _ref_rms_norm_plus_one(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Gemma/Qwen3.5-style: out = x_normed * (1 + weight), weight near zero."""
    x_f = x.float()
    rms = x_f.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x_f * rms * (1.0 + weight.float())).to(x.dtype)


# ---------------------------------------------------------------------------
# Fake unfused RMSNorm (simulates HuggingFace pattern)
# ---------------------------------------------------------------------------


class FakeRMSNorm(nn.Module):
    """LLaMA/Mistral-style: out = x_normed * weight, weight starts at one."""

    def __init__(self, H: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H))
        self.eps = eps

    def forward(self, x):
        return _ref_rms_norm(x, self.weight, self.eps)


class FakeGemmaRMSNorm(nn.Module):
    """Gemma/Qwen3.5-style: out = x_normed * (1 + weight), weight starts at
    zero -- see HF PR #29402. Regression fixture: the fused kernel swap
    (neuropress/nn/np_rms_norm.py) silently dropped the "+1" for this
    convention, collapsing every normalization layer toward zero."""

    def __init__(self, H: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(H))
        self.eps = eps

    def forward(self, x):
        return _ref_rms_norm_plus_one(x, self.weight, self.eps)


# ---------------------------------------------------------------------------
# CPU correctness tests (no CUDA required)
# ---------------------------------------------------------------------------


class TestIsUnfusedRmsNorm:
    def test_detects_fake_rms_norm(self):
        m = FakeRMSNorm(64)
        assert _is_unfused_rms_norm(m)

    def test_rejects_linear(self):
        m = nn.Linear(64, 64)
        assert not _is_unfused_rms_norm(m)

    def test_rejects_layer_norm(self):
        m = nn.LayerNorm(64)
        assert not _is_unfused_rms_norm(m)

    def test_rejects_np_rms_norm(self):
        np_norm = NPRMSNorm(torch.ones(64))
        assert not _is_unfused_rms_norm(np_norm)

    def test_class_name_must_contain_rmsnorm(self):
        # Module with weight+eps but wrong name is NOT replaced.
        class MyNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(16))
                self.eps = 1e-6

        assert not _is_unfused_rms_norm(MyNorm())


class TestUsesPlusOneConvention:
    """Regression coverage: distinguishing the two RMSNorm calling
    conventions is what the original fused-kernel bug got wrong."""

    def test_detects_plain_weight_convention(self):
        m = FakeRMSNorm(64)
        weight = next(m.parameters()).data
        orig = weight.clone()
        assert not _uses_plus_one_convention(m, weight)
        # Detection must restore the original weight afterward.
        assert torch.equal(weight, orig)

    def test_detects_plus_one_convention(self):
        m = FakeGemmaRMSNorm(64)
        weight = next(m.parameters()).data
        orig = weight.clone()
        assert _uses_plus_one_convention(m, weight)
        assert torch.equal(weight, orig)


class TestLinearizeRmsNormsNoCuda:
    """Replacement is skipped when CUDA JIT is unavailable."""

    def test_returns_zero_without_jit(self, monkeypatch):
        monkeypatch.setattr(
            "neuropress.nn.backends.is_cuda_jit_available", lambda: False
        )
        model = nn.Sequential(FakeRMSNorm(32), nn.Linear(32, 32))
        n = linearize_rms_norms(model)
        assert n == 0
        assert isinstance(model[0], FakeRMSNorm)


# ---------------------------------------------------------------------------
# CUDA correctness tests
# ---------------------------------------------------------------------------

cuda_available = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

try:
    from neuropress.kernels import _load_rms_norm_extension

    _load_rms_norm_extension()
    _cuda_jit_ok = True
except Exception:
    _cuda_jit_ok = False

cuda_jit = pytest.mark.skipif(not _cuda_jit_ok, reason="CUDA JIT not available")


@cuda_jit
class TestNPRMSNormCuda:
    @pytest.mark.parametrize("H", [64, 256, 640, 1152])
    @pytest.mark.parametrize("M", [1, 4, 16])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_numerical_correctness(self, H, M, dtype):
        torch.manual_seed(0)
        x = torch.randn(M, H, dtype=dtype, device="cuda")
        w = torch.randn(H, dtype=dtype, device="cuda")
        eps = 1e-6

        np_norm = NPRMSNorm(w, eps=eps).cuda().to(dtype)
        out_fused = np_norm(x)
        out_ref = _ref_rms_norm(x, w, eps)

        # Tolerance: bf16 accumulation is less precise
        atol = 2e-2 if dtype == torch.bfloat16 else 5e-3
        assert torch.allclose(out_fused.float(), out_ref.float(), atol=atol), (
            f"Max diff = {(out_fused.float() - out_ref.float()).abs().max():.4f}"
        )

    def test_leading_dims_preserved(self):
        # NPRMSNorm accepts [*, H] via reshape
        x = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
        w = torch.ones(64, dtype=torch.float16, device="cuda")
        np_norm = NPRMSNorm(w).cuda().to(torch.float16)
        out = np_norm(x)
        assert out.shape == x.shape

    def test_linearize_replaces_modules(self, monkeypatch):
        monkeypatch.setattr(
            "neuropress.nn.backends.is_cuda_jit_available", lambda: True
        )

        class ModelWithNorms(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = FakeRMSNorm(32)
                self.norm2 = FakeRMSNorm(32)
                self.linear = nn.Linear(32, 32)

        model = ModelWithNorms().cuda().to(torch.float16)
        n = linearize_rms_norms(model)
        assert n == 2
        assert isinstance(model.norm1, NPRMSNorm)
        assert isinstance(model.norm2, NPRMSNorm)
        assert isinstance(model.linear, nn.Linear)

    def test_linearize_idempotent(self, monkeypatch):
        monkeypatch.setattr(
            "neuropress.nn.backends.is_cuda_jit_available", lambda: True
        )
        model = nn.Sequential(FakeRMSNorm(32)).cuda().to(torch.float16)
        n1 = linearize_rms_norms(model)
        n2 = linearize_rms_norms(model)
        assert n1 == 1
        assert n2 == 0  # already NPRMSNorm, not replaced again

    def test_linearize_preserves_plus_one_convention_output(self, monkeypatch):
        """Regression test for the exact bug found bisecting the Gemma3-270M
        and Qwen3.5-2B VPTQ collapse: lowering a "+1"-convention RMSNorm
        (Gemma/Qwen3.5-style, weight near zero) must reproduce its original
        forward() numerically, not silently drop the "+1" and attenuate the
        signal toward zero."""
        monkeypatch.setattr(
            "neuropress.nn.backends.is_cuda_jit_available", lambda: True
        )
        torch.manual_seed(0)
        H = 64
        orig = FakeGemmaRMSNorm(H)
        # Non-zero weight so a missing "+1" is numerically obvious, not
        # masked by weight already being near zero.
        orig.weight.data = torch.randn(H) * 0.1
        x = torch.randn(4, H)

        expected = orig(x).clone()

        model = nn.Sequential(orig).cuda().to(torch.float16)
        x_cuda = x.cuda().to(torch.float16)
        n = linearize_rms_norms(model)
        assert n == 1
        assert isinstance(model[0], NPRMSNorm)

        lowered_out = model(x_cuda)
        assert torch.allclose(lowered_out.float().cpu(), expected.float(), atol=5e-2), (
            f"Lowered NPRMSNorm diverges from the original '+1'-convention "
            f"forward() -- max diff = "
            f"{(lowered_out.float().cpu() - expected.float()).abs().max():.4f}"
        )
