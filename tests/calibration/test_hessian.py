"""Tests for HessianAccumulator."""

import torch
from torch import nn

from neuropress.calibration.probe import HessianAccumulator


def _make_linear(in_features=16, out_features=8) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=False)


def _fill_accumulator(
    acc: HessianAccumulator, module: nn.Module, n_batches: int = 4, batch_size: int = 8
) -> None:
    w = module.weight
    assert isinstance(w, torch.Tensor)
    acc.weight_tensor = w
    acc.module_class = type(module)
    acc.module = module
    acc.name = "test_layer"
    in_features = int(w.shape[1])
    for _ in range(n_batches):
        x = torch.randn(batch_size, in_features)
        acc.update(x)


# ---------------------------------------------------------------------------
# HessianAccumulatorBasic
# ---------------------------------------------------------------------------


def test_hessian_accumulator_process_batch_updates_hessian():
    module = _make_linear()
    acc = HessianAccumulator()
    _fill_accumulator(acc, module, n_batches=2)
    assert acc.hessian is not None
    assert acc.hessian.shape == (module.weight.shape[1], module.weight.shape[1])


def test_hessian_accumulator_nsamples_increments():
    # For 2-D input [batch, features], process_batch unsqueezes to
    # [1, batch, features]
    # so batch_size = 1 per call — nsamples counts calls, not individual rows.
    module = _make_linear()
    acc = HessianAccumulator()
    _fill_accumulator(acc, module, n_batches=3, batch_size=4)
    assert acc.nsamples == 3


def test_hessian_accumulator_observer_n_call_increments():
    module = _make_linear()
    acc = HessianAccumulator()
    _fill_accumulator(acc, module, n_batches=5, batch_size=2)
    assert acc.observer_n_call == 5


def test_hessian_accumulator_rows_columns_cached():
    module = _make_linear(in_features=16, out_features=8)
    acc = HessianAccumulator()
    _fill_accumulator(acc, module)
    assert acc.rows == 8
    assert acc.columns == 16


def test_hessian_accumulator_reset_clears_state():
    module = _make_linear()
    acc = HessianAccumulator()
    _fill_accumulator(acc, module, n_batches=2)
    # Trigger cached_property computation.
    _ = acc.rows
    _ = acc.columns
    acc.reset()
    assert acc.hessian is None
    assert acc.nsamples == 0
    assert acc.observer_n_call == 0
    # cached_property slots should be cleared.
    assert "rows" not in acc.__dict__
    assert "columns" not in acc.__dict__


def test_hessian_accumulator_reset_then_reuse():
    module = _make_linear()
    acc = HessianAccumulator()
    _fill_accumulator(acc, module, n_batches=2)
    acc.reset()
    _fill_accumulator(acc, module, n_batches=3, batch_size=4)
    assert acc.nsamples == 3  # 3 calls, nsamples counts calls for 2-D input
    assert acc.hessian is not None


def test_hessian_accumulator_hessian_is_positive_semidefinite():
    module = _make_linear(in_features=8, out_features=4)
    acc = HessianAccumulator()
    _fill_accumulator(acc, module, n_batches=8, batch_size=16)
    # All eigenvalues should be >= 0 for a valid empirical Hessian.
    eigvals = torch.linalg.eigvalsh(acc.hessian)
    assert (eigvals >= -1e-5).all(), (
        f"Negative eigenvalue found: {eigvals.min().item()}"
    )


def test_hessian_accumulator_clone_weights_returns_float():
    module = _make_linear()
    acc = HessianAccumulator()
    acc.weight_tensor = module.weight
    acc.module_class = type(module)
    cloned = acc._clone_weights()
    assert cloned.dtype == torch.float32
    assert cloned.shape == module.weight.shape


# ---------------------------------------------------------------------------
# HessianAccumulatorConv2d
# ---------------------------------------------------------------------------


def test_hessian_accumulator_conv2d_accumulation():
    module = nn.Conv2d(4, 8, kernel_size=3, bias=False)
    acc = HessianAccumulator()
    acc.weight_tensor = module.weight
    acc.module_class = type(module)
    acc.module = module
    acc.name = "conv"
    x = torch.randn(2, 4, 8, 8)
    acc.update(x)
    assert acc.hessian is not None
    assert acc.nsamples > 0


def test_hessian_accumulator_clone_weights_flattens_conv2d():
    module = nn.Conv2d(4, 8, kernel_size=3, bias=False)
    acc = HessianAccumulator()
    acc.weight_tensor = module.weight
    acc.module_class = nn.Conv2d
    cloned = acc._clone_weights()
    # Conv2d weight is [out_ch, in_ch, kH, kW] → flattened to [out_ch, in_ch*kH*kW]
    assert cloned.ndim == 2
    assert cloned.shape[0] == 8
    assert cloned.shape[1] == 4 * 3 * 3
