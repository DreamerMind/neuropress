"""Tests for HessianCalibration and GPTQCalibration."""

import torch
from torch import nn

from neuropress.methods.gptq import (
    GPTQCalibration,
    _fix_dead_diag,
    _reorder_by_hessian_diag,
)
from neuropress.methods.zp_scale_calib import NormZpScaleCalibration
from neuropress.qscheme import Granularity
from neuropress.qscheme.zp_scale import ZpScaleScheme

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear(in_features=64, out_features=16):
    return nn.Linear(in_features, out_features, bias=False)


def _make_calib(group_size=32, hessian_order=True, static=False):
    gran = Granularity.per_group([group_size])
    return GPTQCalibration(
        granularity=gran,
        quant_col_in_hessian_diag_magnitude_order=hessian_order,
        static_groups=static,
        damp_percent=0.01,
        damp_auto_increment=0.01,
        possible_quantizers=[
            NormZpScaleCalibration(
                nbits=4,
                symmetric=True,
                granularity=gran,
                grid=5,  # fast
            )
        ],
    )


def _calibrate_gptq(
    calib: GPTQCalibration, module: nn.Module, n_batches=4, batch_size=8
):
    """Feed synthetic activations into a GPTQCalibration instance."""
    w = module.weight
    assert isinstance(w, torch.Tensor)
    calib.weight_tensor = w
    calib.module_class = type(module)
    calib.module = module
    calib.name = "test_layer"
    calib.hessian = None
    calib.nsamples = 0

    in_features = int(w.shape[1])
    calib.init()
    for _ in range(n_batches):
        x = torch.randn(batch_size, in_features)
        calib.update_statistics(x)  # activations go into Hessian
    return calib


# ---------------------------------------------------------------------------
# Free function tests
# ---------------------------------------------------------------------------


def test_gptq_helpers_fix_dead_diag():
    w = torch.ones(4, 8)
    h = torch.eye(8)
    h[3, 3] = 0.0
    w_out, h_out = _fix_dead_diag(w.clone(), h.clone())
    assert h_out[3, 3] == 1.0
    assert w_out[:, 3].sum() == 0.0


def test_gptq_helpers_reorder_by_hessian_diag():
    w = torch.randn(4, 8)
    h = torch.diag(torch.arange(8, dtype=torch.float))
    perm, w2, h2 = _reorder_by_hessian_diag(w.clone(), h.clone())
    # Highest diagonal first → perm[0] should be 7
    assert perm[0].item() == 7
    assert w2.shape == w.shape
    assert h2.shape == h.shape


# ---------------------------------------------------------------------------
# GPTQCalibration
# ---------------------------------------------------------------------------


def test_gptq_compute_quantization_returns_valid_scheme():
    module = _make_linear(64, 16)
    calib = _make_calib(group_size=32)
    _calibrate_gptq(calib, module)
    weights, qscheme, duration, avg_loss, _damp = calib.compute_quantization()
    assert isinstance(qscheme, ZpScaleScheme)
    assert weights.shape == module.weight.shape
    assert duration > 0
    assert avg_loss >= 0


def test_gptq_compute_qscheme_updates_weight_in_place():
    module = _make_linear(64, 16)
    original_weight = module.weight.data.clone()
    calib = _make_calib(group_size=32)
    _calibrate_gptq(calib, module)
    qscheme = calib.compute_qscheme()
    # After GPTQ, the weight should have been updated.
    assert qscheme is not None
    # The weight changes (quantized-dequantized is not equal to original)
    # With a small model + random weights there's essentially always a difference.
    assert not torch.allclose(module.weight.data, original_weight, atol=0)


def test_gptq_static_groups():
    module = _make_linear(64, 16)
    calib = _make_calib(group_size=32, static=True, hessian_order=False)
    _calibrate_gptq(calib, module)
    _weights, qscheme, *_ = calib.compute_quantization()
    assert isinstance(qscheme, ZpScaleScheme)


def test_gptq_dynamic_groups_no_perm():
    module = _make_linear(64, 16)
    calib = _make_calib(group_size=32, hessian_order=False, static=False)
    _calibrate_gptq(calib, module)
    _weights, qscheme, *_ = calib.compute_quantization()
    assert isinstance(qscheme, ZpScaleScheme)


def test_gptq_group_size_property():
    gran = Granularity.per_group([32])
    calib = GPTQCalibration(
        granularity=gran,
        possible_quantizers=[
            NormZpScaleCalibration(nbits=4, symmetric=True, granularity=gran)
        ],
    )
    assert calib.group_size == 32


def test_gptq_per_channel_property():
    gran = Granularity.per_group([32])
    calib = GPTQCalibration(
        granularity=gran,
        possible_quantizers=[
            NormZpScaleCalibration(nbits=4, symmetric=True, granularity=gran)
        ],
    )
    assert calib.per_channel is True
