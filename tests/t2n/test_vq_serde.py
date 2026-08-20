"""Tests for QTensorTractVQ safetensors serialization round-trip."""

import torch

# Importing neuropress.t2n.qtensors first breaks a circular import that arises
# when neuropress.transforms submodules are imported standalone.
import neuropress.t2n.qtensors  # noqa: F401
from neuropress.t2n.qtensors import load_qtensor, save_qtensor
from neuropress.transforms.vq import ProxySelector, QTensorTractVQ, VQScheme

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vq_tensor(
    rows: int = 32,
    cols: int = 64,
    num_centroids: int = 16,
    vector_len: int = 8,
) -> QTensorTractVQ:
    """Create a QTensorTractVQ via the production quantize_as_u8 path."""
    centroids = torch.randn(num_centroids, vector_len).to(torch.float16)
    shape = torch.Size([rows, cols])
    scheme = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
    )
    weight = torch.randn(rows, cols)
    packed = scheme.quantize_as_u8(weight)
    # QTensorTract.__init__ takes a float tensor and calls quantize_as_u8 internally.
    # Dequantize first so each vector lies on a centroid — round-trips losslessly.
    fp_approx = scheme.dequantize(packed, torch.float32)
    return QTensorTractVQ(fp_approx, scheme)


# ---------------------------------------------------------------------------
# Round-trip: packed indices
# ---------------------------------------------------------------------------


def test_roundtrip_u8_blob(tmp_path):
    q = _make_vq_tensor()
    path = tmp_path / "vq_layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert torch.equal(q.u8_blob, q2.u8_blob)


def test_roundtrip_centroids(tmp_path):
    q = _make_vq_tensor()
    path = tmp_path / "vq_layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert torch.equal(q.qscheme.centroids, q2.qscheme.centroids)
    assert q2.qscheme.centroids.dtype == torch.float16


def test_roundtrip_dequant_output(tmp_path):
    q = _make_vq_tensor()
    path = tmp_path / "vq_layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    # Both tensors must dequantize to identical float values.
    w1 = q.qscheme.dequantize(q.u8_blob, torch.float32)
    w2 = q2.qscheme.dequantize(q2.u8_blob, torch.float32)
    assert torch.allclose(w1, w2)


def test_roundtrip_type(tmp_path):
    q = _make_vq_tensor()
    path = tmp_path / "vq_layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert isinstance(q2, QTensorTractVQ)


def test_roundtrip_original_shape(tmp_path):
    q = _make_vq_tensor(rows=16, cols=128)
    path = tmp_path / "vq_layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert q2.qscheme.original_shape == torch.Size([16, 128])


# ---------------------------------------------------------------------------
# Round-trip: proxy_selector
# ---------------------------------------------------------------------------


def test_roundtrip_proxy_selector_l2(tmp_path):
    q = _make_vq_tensor()
    path = tmp_path / "vq_layer.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert q2.qscheme.proxy_selector == ProxySelector.L2


def test_roundtrip_proxy_selector_cosine(tmp_path):
    centroids = torch.randn(16, 8).to(torch.float16)
    scheme = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.COSINE,
        original_shape=torch.Size([32, 64]),
    )
    weight = torch.randn(32, 64)
    packed = scheme.quantize_as_u8(weight)
    fp_approx = scheme.dequantize(packed, torch.float32)
    q = QTensorTractVQ(fp_approx, scheme)

    path = tmp_path / "cosine.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert q2.qscheme.proxy_selector == ProxySelector.COSINE


# ---------------------------------------------------------------------------
# Round-trip: 8-bit indices (num_centroids=256)
# ---------------------------------------------------------------------------


def test_roundtrip_8bit_indices(tmp_path):
    q = _make_vq_tensor(rows=64, cols=128, num_centroids=256, vector_len=4)
    path = tmp_path / "vq_8bit.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert q2.qscheme.is_packed_indices == 8
    assert torch.equal(q.u8_blob, q2.u8_blob)


# ---------------------------------------------------------------------------
# Shared centroids: two tensors referencing the same codebook
# ---------------------------------------------------------------------------


def test_shared_centroids_round_trip(tmp_path):
    """Layers with the same centroid tensor save/load independently."""
    centroids = torch.randn(16, 8).to(torch.float16)
    scheme1 = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.L2,
        original_shape=torch.Size([32, 64]),
    )
    scheme2 = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.L2,
        original_shape=torch.Size([16, 64]),
    )

    w1 = torch.randn(32, 64)
    w2 = torch.randn(16, 64)
    p1_packed = scheme1.quantize_as_u8(w1)
    p2_packed = scheme2.quantize_as_u8(w2)
    q1 = QTensorTractVQ(scheme1.dequantize(p1_packed, torch.float32), scheme1)
    q2 = QTensorTractVQ(scheme2.dequantize(p2_packed, torch.float32), scheme2)

    p1 = tmp_path / "l1.safetensors"
    p2 = tmp_path / "l2.safetensors"
    save_qtensor(q1, p1)
    save_qtensor(q2, p2)
    r1 = load_qtensor(p1)
    r2 = load_qtensor(p2)

    # Each loaded tensor must have correct shape and matching u8_blob.
    assert r1.qscheme.original_shape == torch.Size([32, 64])
    assert r2.qscheme.original_shape == torch.Size([16, 64])
    assert torch.equal(q1.u8_blob, r1.u8_blob)
    assert torch.equal(q2.u8_blob, r2.u8_blob)
    # Loaded centroids are equal in value (may be different objects).
    assert torch.equal(r1.qscheme.centroids, r2.qscheme.centroids)
