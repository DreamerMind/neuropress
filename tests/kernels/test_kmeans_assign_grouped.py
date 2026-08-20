"""Unit tests for kmeans_assign_grouped CUDA kernel.

Tests verify correctness against the per-group loop implementation of
kmeans_assign, shape/dtype contracts, and large-NC smoke tests.
"""

import pytest
import torch


def _require_cuda():
    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _load_kernels():
    from neuropress.kernels import kmeans_assign, kmeans_assign_grouped

    return kmeans_assign, kmeans_assign_grouped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensors(R, G, NC, D, dev, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(R, G, D, dtype=torch.float32, device=dev).contiguous()
    C = torch.randn(NC, D, dtype=torch.float32, device=dev).contiguous()
    # imp is sqrt-importance: positive, [G, D]
    imp = torch.rand(G, D, dtype=torch.float32, device=dev).add(0.1).contiguous()
    return x, C, imp


def _loop_reference(x, C, imp, vector_len, bias, kmeans_assign_fn):
    """Reference: call kmeans_assign G times in a loop with per-group scaling."""
    _R, G, _D = x.shape
    results = []
    for g in range(G):
        # Scale query by imp[g]
        xg = (x[:, g, :] * imp[g].unsqueeze(0)).contiguous()
        # Scale centroids by imp[g]
        cg = (C * imp[g].unsqueeze(0)).contiguous()
        labels_g = kmeans_assign_fn(xg, cg, vector_len, bias)  # [R] int64
        results.append(labels_g)
    return torch.stack(results, dim=1)  # [R, G]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_grouped_matches_per_group_loop():
    """Grouped kernel must produce identical results to per-group loop."""
    _require_cuda()
    kmeans_assign, kmeans_assign_grouped = _load_kernels()
    dev = torch.device("cuda")

    R, G, NC, D = 64, 16, 256, 4
    x, C, imp = _make_tensors(R, G, NC, D, dev, seed=42)

    ref = _loop_reference(x, C, imp, D, None, kmeans_assign)  # [R, G]
    got = kmeans_assign_grouped(x, C, imp, D)  # [R, G]

    assert got.shape == (R, G), f"expected [{R}, {G}], got {list(got.shape)}"
    assert got.dtype == torch.int64, f"expected int64, got {got.dtype}"
    assert torch.equal(ref, got), (
        f"mismatch: {(ref != got).sum().item()} / {R * G} entries differ"
    )


def test_grouped_shape():
    """Output shape is [R, G] int64, values in [0, NC)."""
    _require_cuda()
    _, kmeans_assign_grouped = _load_kernels()
    dev = torch.device("cuda")

    R, G, NC, D = 128, 8, 512, 4
    x, C, imp = _make_tensors(R, G, NC, D, dev, seed=7)

    out = kmeans_assign_grouped(x, C, imp, D)

    assert out.shape == (R, G), f"expected ({R}, {G}), got {tuple(out.shape)}"
    assert out.dtype == torch.int64
    assert out.min().item() >= 0, "label below 0"
    assert out.max().item() < NC, f"label >= NC={NC}"


def test_grouped_single_group():
    """G=1 grouped result must match a direct kmeans_assign call."""
    _require_cuda()
    kmeans_assign, kmeans_assign_grouped = _load_kernels()
    dev = torch.device("cuda")

    R, G, NC, D = 256, 1, 128, 4
    torch.manual_seed(99)
    x = torch.randn(R, G, D, dtype=torch.float32, device=dev).contiguous()
    C = torch.randn(NC, D, dtype=torch.float32, device=dev).contiguous()
    imp = torch.rand(G, D, dtype=torch.float32, device=dev).add(0.1).contiguous()

    # Grouped kernel
    got = kmeans_assign_grouped(x, C, imp, D)  # [R, 1]

    # Direct kmeans_assign with scaled inputs
    x_flat = (x[:, 0, :] * imp[0].unsqueeze(0)).contiguous()
    c_flat = (C * imp[0].unsqueeze(0)).contiguous()
    ref = kmeans_assign(x_flat, c_flat, D).unsqueeze(1)  # [R, 1]

    assert torch.equal(ref, got), (
        f"G=1 mismatch: {(ref != got).sum().item()} / {R} entries differ"
    )


def test_grouped_with_bias():
    """All-zeros bias must produce the same result as no bias."""
    _require_cuda()
    _, kmeans_assign_grouped = _load_kernels()
    dev = torch.device("cuda")

    R, G, NC, D = 64, 4, 256, 4
    x, C, imp = _make_tensors(R, G, NC, D, dev, seed=13)

    zero_bias = torch.zeros(NC, dtype=torch.float32, device=dev)

    out_no_bias = kmeans_assign_grouped(x, C, imp, D, None)
    out_zero_bias = kmeans_assign_grouped(x, C, imp, D, zero_bias)

    assert torch.equal(out_no_bias, out_zero_bias), "all-zeros bias changed results"


def test_grouped_large_nc():
    """R=1024, G=8, NC=32768 — verify no crash and valid indices."""
    _require_cuda()
    _, kmeans_assign_grouped = _load_kernels()
    dev = torch.device("cuda")

    R, G, NC, D = 1024, 8, 32768, 4
    x, C, imp = _make_tensors(R, G, NC, D, dev, seed=5)

    out = kmeans_assign_grouped(x, C, imp, D)

    assert out.shape == (R, G)
    assert out.dtype == torch.int64
    assert out.min().item() >= 0
    assert out.max().item() < NC
