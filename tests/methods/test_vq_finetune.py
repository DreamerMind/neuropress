"""Codebook fine-tuning: exact forward, codebook-only gradients, write-back."""

import torch
import torch.nn.functional as F
from torch import nn

import neuropress.t2n.qtensors  # noqa: F401
from neuropress.methods.vq_finetune import (
    TrainableVQLinear,
    finetune_codebooks,
    kd_loss,
    prepare_vq_finetune,
    write_back_codebooks,
)
from neuropress.transforms.vq import ProxySelector, QTensorTractVQ, VQScheme


def _vq_weight(
    weight: torch.Tensor, centroids: torch.Tensor, perm: torch.Tensor | None
):
    scheme = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.L2,
        original_shape=weight.shape,
        column_permutation=perm,
    )
    packed = scheme.quantize_as_u8(weight)
    return QTensorTractVQ(scheme.dequantize(packed, torch.float32), scheme)


class _Toy(nn.Module):
    """Three linears: fc1 and fc2 share a codebook, fc3 has its own."""

    def __init__(self, dense: nn.Module | None = None, quantize: bool = True) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(16, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=False)
        self.fc3 = nn.Linear(32, 8, bias=False)
        if dense is not None:
            self.load_state_dict(dense.state_dict())
        if quantize:
            shared = torch.randn(16, 4, dtype=torch.float16) * 0.2
            own = torch.randn(16, 4, dtype=torch.float16) * 0.2
            perm = torch.randperm(32)
            for lin, cent, p in (
                (self.fc1, shared, None),
                (self.fc2, shared, perm),
                (self.fc3, own, None),
            ):
                # As WeightAssignmentAdapter does it, so the saver sees it.
                lin._parameters["weight"] = _vq_weight(lin.weight.data, cent, p)

    def forward(self, x):
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))


def _dequantized(model: _Toy, x: torch.Tensor) -> torch.Tensor:
    def lin(m, h):
        w = m.weight.qscheme.dequantize(m.weight.u8_blob, torch.float32)
        return F.linear(h, w, m.bias)

    return lin(model.fc3, torch.tanh(lin(model.fc2, torch.tanh(lin(model.fc1, x)))))


def test_prepare_groups_shared_codebooks_and_matches_dequantized_forward():
    model = _Toy()
    x = torch.randn(5, 16)
    ref = _dequantized(model, x)
    groups = prepare_vq_finetune(model)
    assert len(groups) == 2
    assert sorted(len(g.names) for g in groups) == [1, 2]
    assert all(
        isinstance(m, TrainableVQLinear) for m in (model.fc1, model.fc2, model.fc3)
    )
    torch.testing.assert_close(model(x), ref, rtol=1e-4, atol=1e-5)


def test_gradient_reaches_only_codebooks():
    model = _Toy()
    groups = prepare_vq_finetune(model)
    for g in groups:
        g.codebook.requires_grad_(True)
    model(torch.randn(4, 16)).pow(2).sum().backward()
    assert all(g.codebook.grad is not None for g in groups)
    others = [p for n, p in model.named_parameters() if "codebook" not in n]
    assert others and all(p.grad is None and not p.requires_grad for p in others)


def test_finetune_lowers_kd_loss():
    dense = _Toy(quantize=False)
    student = _Toy(dense=dense)
    groups = prepare_vq_finetune(student)
    batches = [torch.randn(64, 16) for _ in range(8)]
    with torch.no_grad():
        before = sum(kd_loss(student(b), dense(b)).item() for b in batches)
    finetune_codebooks(
        groups, student, dense, batches, lr=1e-2, epochs=20, log_every=1000
    )
    with torch.no_grad():
        after = sum(kd_loss(student(b), dense(b)).item() for b in batches)
    assert after < 0.7 * before


def test_write_back_updates_schemes_restores_layers_and_round_trips(tmp_path):
    from neuropress.t2n import load_quantized_model, save_quantized_model

    model = _Toy()
    x = torch.randn(3, 16)
    groups = prepare_vq_finetune(model)
    with torch.no_grad():
        for g in groups:
            g.codebook.add_(0.05)
        trained = model(x)
    write_back_codebooks(model, groups)
    assert all(type(m) is nn.Linear for m in (model.fc1, model.fc2, model.fc3))
    assert model.fc1.weight.qscheme.centroids is model.fc2.weight.qscheme.centroids
    torch.testing.assert_close(_dequantized(model, x), trained, rtol=1e-3, atol=1e-3)

    path = tmp_path / "ft.safetensors"
    save_quantized_model(model, path)
    fresh = _Toy(quantize=False)
    load_quantized_model(fresh, path, kernel_lowering=None)
    torch.testing.assert_close(_dequantized(fresh, x), trained, rtol=1e-3, atol=1e-3)
