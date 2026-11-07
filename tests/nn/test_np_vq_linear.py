"""Tests for NPVQLinear and linearize_vq_weights."""

import pytest
import torch
import torch.nn as nn

# Importing neuropress.t2n.qtensors first breaks a circular import that arises
# when neuropress.transforms submodules are imported standalone.
import neuropress.t2n.qtensors  # noqa: F401
from neuropress.nn.np_linear import Backend
from neuropress.nn.np_vq_linear import NPVQLinear, VQProxy, linearize_vq_weights
from neuropress.transforms.vq import ProxySelector, QTensorTractVQ, VQScheme

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear_with_vq(
    out_features: int,
    in_features: int,
    num_centroids: int = 8,
    vector_len: int = 4,
    bias: bool = False,
    centroids: torch.Tensor | None = None,
) -> nn.Linear:
    """Create nn.Linear with a QTensorTractVQ weight."""
    if centroids is None:
        centroids = torch.randn(num_centroids, vector_len, dtype=torch.float16)
    shape = torch.Size([out_features, in_features])
    scheme = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
    )
    weight = torch.randn(out_features, in_features)
    packed = scheme.quantize_as_u8(weight)
    fp_approx = scheme.dequantize(packed, torch.float32)
    qt = QTensorTractVQ(fp_approx, scheme)

    linear = nn.Linear(in_features, out_features, bias=bias)
    # Bypass nn.Parameter validation by assigning directly to __dict__.
    linear.__dict__["weight"] = qt
    if bias:
        linear.bias = nn.Parameter(torch.zeros(out_features))
    return linear


# ---------------------------------------------------------------------------
# VQProxy
# ---------------------------------------------------------------------------


def test_vq_proxy_stores_fp16_centroids():
    centroids = torch.randn(16, 8)  # float32 input
    proxy = VQProxy(centroids)
    assert proxy.centroids.dtype == torch.float16
    assert proxy.centroids.shape == (16, 8)


def test_vq_proxy_forward_raises():
    proxy = VQProxy(torch.randn(8, 4, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="should not be called"):
        proxy(torch.zeros(1))


# ---------------------------------------------------------------------------
# from_qtensor_linear factory
# ---------------------------------------------------------------------------


def test_from_qtensor_linear_basic():
    linear = _make_linear_with_vq(out_features=16, in_features=32)
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)
    assert layer.out_features == 16
    assert layer.in_features == 32


def test_from_qtensor_linear_wrong_type_raises():
    linear = nn.Linear(32, 16)
    proxy = VQProxy(torch.randn(8, 4, dtype=torch.float16))
    with pytest.raises(TypeError, match="QTensorTractVQ"):
        NPVQLinear.from_qtensor_linear(linear, proxy)


def test_from_qtensor_linear_preserves_bias():
    linear = _make_linear_with_vq(16, 32, bias=True)
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)
    assert layer.bias is not None
    assert layer.bias.shape == (16,)


# ---------------------------------------------------------------------------
# Forward: DEFAULT backend (CPU)
# ---------------------------------------------------------------------------


def test_forward_default_shape():
    linear = _make_linear_with_vq(out_features=16, in_features=32)
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)

    x = torch.randn(4, 32)
    y = layer(x)
    assert y.shape == (4, 16)


def test_forward_default_dtype_preserved_float32():
    linear = _make_linear_with_vq(16, 32)
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)

    x = torch.randn(2, 32, dtype=torch.float32)
    y = layer(x)
    assert y.dtype == torch.float32


def test_forward_default_3d_input():
    linear = _make_linear_with_vq(16, 32)
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)

    x = torch.randn(2, 5, 32)
    y = layer(x)
    assert y.shape == (2, 5, 16)


def test_forward_default_matches_dequant():
    """NPVQLinear(DEFAULT) output must match reference F.linear(dequantized W)."""
    linear = _make_linear_with_vq(16, 32, num_centroids=8, vector_len=4)
    qscheme = linear.weight.qscheme
    proxy = VQProxy(qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)

    # Reference: dequantize → float → F.linear
    packed = linear.weight.u8_blob
    w_ref = qscheme.dequantize(packed, torch.float32)

    x = torch.randn(3, 32)
    y_layer = layer(x)
    y_ref = torch.nn.functional.linear(x, w_ref.to(x.dtype))

    assert torch.allclose(y_layer, y_ref, atol=1e-5), (
        f"max diff = {(y_layer - y_ref).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# linearize_vq_weights — model-level transform
# ---------------------------------------------------------------------------


class _SharedProxyModel(nn.Module):
    def __init__(self, centroids: torch.Tensor) -> None:
        super().__init__()
        self.fc1 = _make_linear_with_vq(16, 32, centroids=centroids)
        self.fc2 = _make_linear_with_vq(8, 32, centroids=centroids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(x) + self.fc1(x)[:, :8]


def test_linearize_replaces_linear_modules():
    centroids = torch.randn(8, 4, dtype=torch.float16)
    model = _SharedProxyModel(centroids)
    linearize_vq_weights(model, backend=Backend.DEFAULT)
    assert isinstance(model.fc1, NPVQLinear)
    assert isinstance(model.fc2, NPVQLinear)


def test_linearize_proxy_shared_across_layers():
    """Layers sharing the same centroids object must receive the same VQProxy."""
    centroids = torch.randn(8, 4, dtype=torch.float16)
    model = _SharedProxyModel(centroids)
    linearize_vq_weights(model, backend=Backend.DEFAULT)
    # Same proxy Python object for both layers.
    assert model.fc1.proxy is model.fc2.proxy


def test_linearize_proxy_registered_on_model():
    """VQProxy must be a registered sub-module of the top-level model."""
    centroids = torch.randn(8, 4, dtype=torch.float16)
    model = _SharedProxyModel(centroids)
    linearize_vq_weights(model, backend=Backend.DEFAULT)
    proxy_modules = [
        name for name, m in model.named_modules() if isinstance(m, VQProxy)
    ]
    assert len(proxy_modules) == 1, (
        f"Expected exactly 1 VQProxy in model, found {proxy_modules}"
    )


def test_linearize_forward_runs():
    centroids = torch.randn(8, 4, dtype=torch.float16)
    model = _SharedProxyModel(centroids)
    linearize_vq_weights(model, backend=Backend.DEFAULT)
    x = torch.randn(2, 32)
    y = model(x)
    assert y.shape == (2, 8)


def test_linearize_different_centroids_different_proxies():
    """Two distinct centroids tensors must produce two distinct VQProxy instances."""

    class _TwoCodebooks(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c1 = torch.randn(8, 4, dtype=torch.float16)
            c2 = torch.randn(8, 4, dtype=torch.float16)
            self.fc1 = _make_linear_with_vq(16, 32, centroids=c1)
            self.fc2 = _make_linear_with_vq(8, 32, centroids=c2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(x) + self.fc1(x)[:, :8]

    model = _TwoCodebooks()
    linearize_vq_weights(model, backend=Backend.DEFAULT)
    assert model.fc1.proxy is not model.fc2.proxy
    proxy_modules = [
        name for name, m in model.named_modules() if isinstance(m, VQProxy)
    ]
    assert len(proxy_modules) == 2


# ---------------------------------------------------------------------------
# backend resolution
# ---------------------------------------------------------------------------


def test_gemlite_backend_raises():
    from neuropress.nn.np_vq_linear import _resolve_vq_backend

    with pytest.raises(ValueError, match=r"(?i)gemlite"):
        _resolve_vq_backend(Backend.GEMLITE)


def test_auto_backend_resolves_to_compile_on_cpu():
    from neuropress.nn.np_vq_linear import _resolve_vq_backend

    # On CPU-only test environment AUTO should resolve to COMPILE.
    if torch.cuda.is_available():
        pytest.skip("CUDA available — AUTO resolves to CUDA, not COMPILE")
    assert _resolve_vq_backend(Backend.AUTO) == Backend.COMPILE


# ---------------------------------------------------------------------------
# NPVQLinear — tiered (use_tiered=True) backend
# ---------------------------------------------------------------------------


def _make_tiered_linear(
    out_features: int = 16,
    in_features: int = 32,
    num_centroids: int = 64,
    vector_len: int = 4,
    bias: bool = False,
) -> nn.Linear:
    """Create nn.Linear with a QTensorTractVQ weight using use_tiered=True."""
    centroids = torch.randn(num_centroids, vector_len, dtype=torch.float16)
    shape = torch.Size([out_features, in_features])
    scheme = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
        use_tiered=True,
    )
    weight = torch.randn(out_features, in_features)
    packed = scheme.quantize_as_u8(weight)
    fp_approx = scheme.dequantize(packed, torch.float32)
    qt = QTensorTractVQ(fp_approx, scheme)

    linear = nn.Linear(in_features, out_features, bias=bias)
    linear.__dict__["weight"] = qt
    if bias:
        linear.bias = nn.Parameter(torch.zeros(out_features))
    return linear


def test_tiered_from_qtensor_linear_sets_use_tiered():
    linear = _make_tiered_linear()
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)
    assert layer.use_tiered is True


def test_tiered_from_qtensor_linear_computes_row_offsets():
    """from_qtensor_linear must precompute non-None row offsets for tiered layers."""
    linear = _make_tiered_linear(out_features=16, in_features=32, vector_len=4)
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)
    assert layer.hot_row_offsets is not None
    assert layer.cold_row_offsets is not None
    assert layer.hot_row_offsets.shape == (17,)  # out_features + 1
    assert layer.cold_row_offsets.shape == (17,)


def test_tiered_row_offsets_last_equals_total_hot_cold():
    """hot_row_offsets[-1] + cold_row_offsets[-1] == total index count."""
    linear = _make_tiered_linear(out_features=8, in_features=32, vector_len=4)
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)

    n_groups = linear.in_features // linear.weight.qscheme.vector_len
    n_total = linear.out_features * n_groups
    assert (
        int(layer.hot_row_offsets[-1].item()) + int(layer.cold_row_offsets[-1].item())
        == n_total
    )


def test_tiered_forward_default_shape():
    linear = _make_tiered_linear(out_features=16, in_features=32)
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)
    x = torch.randn(4, 32)
    y = layer(x)
    assert y.shape == (4, 16)


def test_tiered_forward_default_matches_dequant():
    """NPVQLinear(DEFAULT, tiered) output must match VQScheme.dequantize reference."""
    linear = _make_tiered_linear(
        out_features=8, in_features=32, num_centroids=64, vector_len=4
    )
    qscheme = linear.weight.qscheme
    proxy = VQProxy(qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)

    packed = linear.weight.u8_blob
    w_ref = qscheme.dequantize(packed, torch.float32)

    x = torch.randn(3, 32)
    y_layer = layer(x)
    y_ref = torch.nn.functional.linear(x, w_ref.to(x.dtype))

    assert torch.allclose(y_layer, y_ref, atol=1e-4), (
        f"max diff = {(y_layer - y_ref).abs().max().item()}"
    )


def test_tiered_forward_matches_non_tiered():
    """Tiered and standard NPVQLinear must produce identical outputs for same weight."""
    nc, vl, out_f, in_f = 64, 4, 8, 32
    centroids = torch.randn(nc, vl, dtype=torch.float16)
    weight = torch.randn(out_f, in_f)

    # Standard (non-tiered) layer.
    scheme_std = VQScheme(
        centroids=centroids.clone(),
        proxy_selector=ProxySelector.L2,
        original_shape=torch.Size([out_f, in_f]),
        use_tiered=False,
    )
    packed_std = scheme_std.quantize_as_u8(weight)
    fp_std = scheme_std.dequantize(packed_std, torch.float32)
    qt_std = QTensorTractVQ(fp_std, scheme_std)
    linear_std = nn.Linear(in_f, out_f, bias=False)
    linear_std.__dict__["weight"] = qt_std
    proxy_std = VQProxy(scheme_std.centroids)
    layer_std = NPVQLinear.from_qtensor_linear(
        linear_std, proxy_std, backend=Backend.DEFAULT
    )

    # Tiered layer with same centroids and weight.
    scheme_tiered = VQScheme(
        centroids=centroids.clone(),
        proxy_selector=ProxySelector.L2,
        original_shape=torch.Size([out_f, in_f]),
        use_tiered=True,
    )
    packed_tiered = scheme_tiered.quantize_as_u8(weight)
    fp_tiered = scheme_tiered.dequantize(packed_tiered, torch.float32)
    qt_tiered = QTensorTractVQ(fp_tiered, scheme_tiered)
    linear_tiered = nn.Linear(in_f, out_f, bias=False)
    linear_tiered.__dict__["weight"] = qt_tiered
    proxy_tiered = VQProxy(scheme_tiered.centroids)
    layer_tiered = NPVQLinear.from_qtensor_linear(
        linear_tiered, proxy_tiered, backend=Backend.DEFAULT
    )

    x = torch.randn(5, in_f)
    y_std = layer_std(x)
    y_tiered = layer_tiered(x)
    assert torch.allclose(y_std, y_tiered, atol=1e-4), (
        f"max diff = {(y_std - y_tiered).abs().max().item()}"
    )


def test_tiered_extra_repr_contains_tiered():
    linear = _make_tiered_linear()
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)
    assert "tiered=True" in repr(layer)


def test_cuda_tiered_backend_resolves():
    """Backend.CUDA_TIERED must resolve without error."""
    from neuropress.nn.np_vq_linear import _resolve_vq_backend

    assert _resolve_vq_backend(Backend.CUDA_TIERED) == Backend.CUDA_TIERED


def test_tiered_forward_cuda_tiered_skipped_without_cuda():
    """CUDA_TIERED forward must raise RuntimeError when CUDA is unavailable."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is present — kernel would build.")
    linear = _make_tiered_linear(out_features=8, in_features=32)
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.CUDA_TIERED)
    x = torch.randn(2, 32)
    with pytest.raises(RuntimeError, match="CUDA"):
        layer(x)


# ---------------------------------------------------------------------------
# Large-M switch: fused kernels for decode, dequant + GEMM for prefill
# ---------------------------------------------------------------------------


def _fused_layer_with_spies(
    monkeypatch, backend, lut_max_tokens=16, triton_max_tokens=0
):
    import neuropress.nn.np_vq_linear as mod

    monkeypatch.setattr(mod, "_FUSED_MAX_TOKENS", 16)
    monkeypatch.setattr(mod, "_LUT_MAX_TOKENS", lut_max_tokens)
    monkeypatch.setattr(mod, "_TRITON_MAX_TOKENS", triton_max_tokens)
    linear = _make_linear_with_vq(out_features=16, in_features=32)
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=backend)
    calls = []
    fused = {
        Backend.CUDA_WMMA: "_forward_cuda_wmma",
        Backend.CUDA_LUT: "_forward_cuda_lut",
        Backend.CUDA: "_forward_cuda",
    }[backend]
    monkeypatch.setattr(layer, fused, lambda x: calls.append("fused") or x[..., :16])
    monkeypatch.setattr(
        layer, "_forward_dequant_gemm", lambda x: calls.append("dense") or x[..., :16]
    )
    monkeypatch.setattr(
        layer, "_forward_triton", lambda x: calls.append("triton") or x[..., :16]
    )
    return layer, calls


@pytest.mark.parametrize("backend", [Backend.CUDA_WMMA, Backend.CUDA])
def test_mid_sized_batches_use_triton_band(monkeypatch, backend):
    layer, calls = _fused_layer_with_spies(monkeypatch, backend, triton_max_tokens=64)
    layer(torch.randn(16, 32))  # fused
    layer(torch.randn(17, 32))  # triton
    layer(torch.randn(64, 32))  # triton, at its threshold
    layer(torch.randn(65, 32))  # decode + GEMM
    assert calls == ["fused", "triton", "triton", "dense"]


def test_lut_layers_skip_triton_band(monkeypatch):
    layer, calls = _fused_layer_with_spies(
        monkeypatch, Backend.CUDA_LUT, lut_max_tokens=0, triton_max_tokens=64
    )
    layer(torch.randn(8, 32))
    assert calls == ["dense"]


def test_triton_failure_falls_back_once(monkeypatch):
    import sys
    import types

    import neuropress.nn.np_vq_linear as mod

    monkeypatch.setattr(mod, "_triton_failed", False)
    linear = _make_linear_with_vq(out_features=16, in_features=32)
    layer = NPVQLinear.from_qtensor_linear(
        linear, VQProxy(linear.weight.qscheme.centroids), backend=Backend.CUDA_WMMA
    )
    calls = []

    def _boom(*args, **kwargs):
        calls.append("triton")
        raise RuntimeError("no triton here")

    # A stand-in module: CPU-only installs have no Triton to import.
    fake = types.ModuleType("neuropress.kernels.vq_triton")
    fake.vq_gemm_triton = _boom
    monkeypatch.setitem(sys.modules, "neuropress.kernels.vq_triton", fake)
    monkeypatch.setattr(
        layer, "_forward_dequant_gemm", lambda x: calls.append("dense") or x[..., :16]
    )
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    layer._forward_triton(x)
    layer._forward_triton(x)
    assert calls == ["triton", "dense", "dense"]


@pytest.mark.parametrize("backend", [Backend.CUDA_WMMA, Backend.CUDA_LUT, Backend.CUDA])
def test_fused_backend_keeps_kernel_up_to_threshold(monkeypatch, backend):
    layer, calls = _fused_layer_with_spies(monkeypatch, backend)
    layer(torch.randn(1, 32))
    layer(torch.randn(2, 8, 32))  # 16 tokens, at the threshold
    assert calls == ["fused", "fused"]


@pytest.mark.parametrize("backend", [Backend.CUDA_WMMA, Backend.CUDA_LUT, Backend.CUDA])
def test_fused_backend_switches_to_dequant_gemm_above_threshold(monkeypatch, backend):
    layer, calls = _fused_layer_with_spies(monkeypatch, backend)
    layer(torch.randn(17, 32))
    layer(torch.randn(2, 9, 32))  # leading dims count as tokens
    assert calls == ["dense", "dense"]


def test_lut_backend_uses_dequant_gemm_by_default(monkeypatch):
    """cuda_lut loses to decode + GEMM at every token count, even M=1."""
    import neuropress.nn.np_vq_linear as mod

    layer, calls = _fused_layer_with_spies(
        monkeypatch, Backend.CUDA_LUT, lut_max_tokens=mod._LUT_MAX_TOKENS
    )
    layer(torch.randn(1, 32))
    assert calls == ["dense"]


def test_large_m_switch_matches_dequant():
    """The rerouted path computes the same result as the dequantized weight."""
    import neuropress.nn.np_vq_linear as mod

    linear = _make_linear_with_vq(out_features=16, in_features=32)
    proxy = VQProxy(linear.weight.qscheme.centroids)
    layer = NPVQLinear.from_qtensor_linear(linear, proxy, backend=Backend.DEFAULT)
    x = torch.randn(40, 32)
    ref = layer(x)
    out = mod._vq_dequant_linear(
        x,
        layer.packed_indices,
        layer.proxy.centroids,
        layer.bits,
        layer.n_vectors,
        layer.out_features,
        layer.in_features,
        layer.bias,
    )
    torch.testing.assert_close(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("bits", [4, 8, 10, 12, 14, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_vq_dequant_kernel_matches_python_decode(bits, dtype):
    from neuropress.kernels import vq_dequant
    from neuropress.transforms.vq_packing import IndexPacking

    out_features, in_features, vl = 37, 64, 4  # odd N: last index ends mid-byte
    nc = min(1 << bits, 4096)
    n_vectors = out_features * in_features // vl
    idx = torch.randint(0, nc, (n_vectors,), device="cuda")
    centroids = torch.randn(nc, vl, device="cuda", dtype=torch.float16)
    packed = IndexPacking.pack(idx, bits)
    ref = centroids[idx].reshape(out_features, in_features).to(dtype)
    out = vq_dequant(packed, centroids, bits, out_features, in_features, dtype)
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dequant_gemm_forward_matches_default():
    linear = _make_linear_with_vq(out_features=48, in_features=64, num_centroids=16)
    centroids = linear.weight.qscheme.centroids
    ref_layer = NPVQLinear.from_qtensor_linear(
        linear, VQProxy(centroids), backend=Backend.DEFAULT
    )
    # The proxy is shared across layers, not a child module: move it separately.
    layer = NPVQLinear.from_qtensor_linear(
        linear, VQProxy(centroids).cuda(), backend=Backend.DEFAULT
    ).cuda()
    x = torch.randn(100, 64)
    ref = ref_layer(x)
    out = layer._forward_dequant_gemm(x.cuda())
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("num_centroids", [16, 256, 1024, 4096, 16384])
def test_cuda_backend_matches_default_for_all_index_widths(num_centroids):
    import neuropress.nn.np_vq_linear as mod

    linear = _make_linear_with_vq(
        out_features=24, in_features=64, num_centroids=num_centroids
    )
    centroids = linear.weight.qscheme.centroids
    ref_layer = NPVQLinear.from_qtensor_linear(
        linear, VQProxy(centroids), backend=Backend.DEFAULT
    )
    layer = NPVQLinear.from_qtensor_linear(
        linear, VQProxy(centroids).cuda(), backend=Backend.CUDA
    ).cuda()
    tokens = mod._fused_max_tokens(layer.backend)
    assert tokens >= 1  # the kernel itself runs
    x = torch.randn(tokens, 64)
    ref = ref_layer(x)
    out = layer(x.cuda().half()).float().cpu()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_forward_matches_dequant_gemm():
    """Same bf16 input through both paths: only accumulation order differs."""
    pytest.importorskip("triton")
    linear = _make_linear_with_vq(out_features=48, in_features=64, num_centroids=16)
    layer = NPVQLinear.from_qtensor_linear(
        linear,
        VQProxy(linear.weight.qscheme.centroids).cuda(),
        backend=Backend.DEFAULT,
    ).cuda()
    x = torch.randn(2, 50, 64, device="cuda", dtype=torch.bfloat16)
    ref = layer._forward_dequant_gemm(x).float()
    out = layer._forward_triton(x).float()
    assert out.shape == (2, 50, 48)
    # bf16 has an 8-bit mantissa: allow two units in the last place.
    torch.testing.assert_close(out, ref, rtol=2 * 2**-8, atol=1e-2)
