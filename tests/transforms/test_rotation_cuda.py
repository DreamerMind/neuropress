"""CUDA-only regression tests for rotation device handling.

These cover a bug class the rest of the suite structurally cannot: the rotation
matrices and sign diagonals are constructed on CPU in float64, while model
weights live on the accelerator.  A CPU-only run passes regardless of whether
the code moves them, so this file is the only guard.  Sweep failure on
2026-08-02 was exactly this: ``mat2 is on cpu, different from other tensors on
cuda:0``.
"""

import pytest
import torch
from torch import nn

import neuropress.t2n.qtensors  # noqa: F401
from neuropress.transforms.rotation import (
    rotate_attention_vo,
    rotate_residual_stream,
    wrap_linear_with_online_hadamard,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class _RMSNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n) * 0.3)

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * (1.0 + self.weight)


def test_residual_rotation_runs_on_cuda_weights():
    d, dff = 8, 16
    spec = {
        "input_norm": _RMSNorm(d).cuda(),
        "input_consumers": [nn.Linear(d, d, bias=False).cuda()],
        "attn_output": nn.Linear(d, d, bias=False).cuda(),
        "post_norm": _RMSNorm(d).cuda(),
        "post_consumers": [nn.Linear(d, dff, bias=False).cuda()],
        "mlp_output": nn.Linear(dff, d, bias=False).cuda(),
    }
    rotate_residual_stream(layers=[spec], hidden_size=d, seed=3)

    assert spec["mlp_output"].weight.device.type == "cuda"
    assert spec["input_consumers"][0].weight.device.type == "cuda"


def test_online_hadamard_runs_on_cuda_weights():
    linear = nn.Linear(16, 4, bias=False).cuda()
    wrapped = wrap_linear_with_online_hadamard(linear, seed=5)

    x = torch.randn(3, 16, device="cuda")
    assert wrapped(x).device.type == "cuda"
    assert wrapped.signs.device.type == "cuda"


def test_online_hadamard_preserves_output_on_cuda():
    torch.manual_seed(0)
    linear = nn.Linear(16, 4, bias=False).cuda().double()
    x = torch.randn(3, 16, device="cuda", dtype=torch.float64)

    before = linear(x)
    after = wrap_linear_with_online_hadamard(linear, seed=5)(x)

    assert torch.allclose(before, after, atol=1e-10)


def test_online_hadamard_kronecker_size_on_cuda():
    """12 * 2^k sizes take the Kronecker path, which builds H_12 internally.

    The 2026-08-03 sweep failure was exactly here: ``fast_hadamard_transform``
    created H_12 on CPU while the activation was on cuda.  The power-of-two
    tests above cannot catch it, since that path never constructs H_12.
    """
    linear = nn.Linear(6144, 8, bias=False).cuda()
    wrapped = wrap_linear_with_online_hadamard(linear, seed=5)

    x = torch.randn(2, 6144, device="cuda")
    assert wrapped(x).device.type == "cuda"


def test_fwht_kronecker_path_preserves_output_on_cuda():
    torch.manual_seed(0)
    linear = nn.Linear(24, 4, bias=False).cuda().double()
    x = torch.randn(3, 24, device="cuda", dtype=torch.float64)

    before = linear(x)
    after = wrap_linear_with_online_hadamard(linear, seed=5)(x)

    assert torch.allclose(before, after, atol=1e-10)


def test_vo_rotation_runs_on_cuda_weights():
    v = nn.Linear(8, 8, bias=False).cuda()
    o = nn.Linear(8, 8, bias=False).cuda()
    rotate_attention_vo(v, o, head_dim=4, seed=3)

    assert v.weight.device.type == "cuda"
    assert o.weight.device.type == "cuda"
