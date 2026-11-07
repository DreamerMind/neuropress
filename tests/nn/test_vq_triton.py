"""Triton VQ GEMM must match decode + dense matmul."""

import pytest
import torch
import torch.nn.functional as F

triton = pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("bits", [4, 8, 10, 12, 14, 16])
@pytest.mark.parametrize("m", [1, 17, 130])
def test_vq_gemm_triton_matches_dequant(bits, m):
    from neuropress.kernels import vq_dequant
    from neuropress.kernels.vq_triton import vq_gemm_triton
    from neuropress.transforms.vq_packing import IndexPacking

    torch.manual_seed(0)
    n, k = 72, 96  # not multiples of the block sizes
    nc = min(1 << bits, 4096)
    idx = torch.randint(0, nc, (n * k // 4,), device="cuda")
    packed = IndexPacking.pack(idx, bits)
    centroids = torch.randn(nc, 4, device="cuda", dtype=torch.float16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = vq_dequant(packed, centroids, bits, n, k, torch.bfloat16)
    ref = F.linear(x.float(), w.float(), bias.float())
    out = vq_gemm_triton(x, packed, centroids, bits, n, bias)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=5e-2)
