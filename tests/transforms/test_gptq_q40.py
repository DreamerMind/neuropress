import numpy as np
import pytest
import torch
from torch import nn
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.nnef_io.tensor import DatBinHeader
from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

import neuropress.transforms.weight_only  # noqa: F401
from neuropress.adapter import WeightAssignmentAdapter
from neuropress.estimator_configs import GptqEstimatorConfig
from neuropress.estimators import GPTQEstimator
from neuropress.methods.gptq import GPTQCalibration
from neuropress.methods.zp_scale_calib import NormZpScaleCalibration
from neuropress.pipeline import QuantizationPipeline
from neuropress.plan import QuantizedWeight
from neuropress.qscheme import Granularity
from neuropress.recipe import QuantizationRecipe
from neuropress.transforms.weight_only import _q4_0_from_assignment


@pytest.mark.parametrize(
    "format_name", ["q4_0", "q4_0_unpacked", "q4_0_packed", "q4_0_packed_no_lm_head"]
)
@pytest.mark.parametrize("group_size", [32, 64])
def test_pipeline_preserves_gptq_grid_in_nnef(format_name, group_size, tmp_path):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(64, 16, bias=False)).eval()
    original = model[0].weight.detach().clone()
    x = torch.randn(8, 64, 64)
    gran = Granularity.per_group([group_size])
    reference = nn.Linear(64, 16, bias=False)
    reference.weight.data.copy_(original)
    calib = GPTQCalibration(
        granularity=gran,
        possible_quantizers=[
            NormZpScaleCalibration(nbits=4, symmetric=True, granularity=gran)
        ],
    )
    calib.weight_tensor = reference.weight
    calib.module_class = nn.Linear
    calib.module = reference
    calib.name = "reference"
    calib.init()
    calib.update_statistics(x)
    _weights, scheme, *_ = calib.compute_quantization()
    expected_codes = calib.quantized_codes.reshape(-1, 32)
    expected_scales = scheme.scale.flatten().repeat_interleave(group_size // 32).half()
    estimator = GptqEstimatorConfig(
        group_size=group_size, weight_transform=format_name
    ).build()
    recipe = QuantizationRecipe(
        param_estimator=estimator, adapter=WeightAssignmentAdapter()
    )
    QuantizationPipeline(recipe).run(model, [x])
    q = model[0].weight
    torch.testing.assert_close(
        q.decompress_to_u8().reshape(-1, 32), expected_codes, rtol=0, atol=0
    )
    torch.testing.assert_close(
        q.qscheme.scale.flatten(), expected_scales, rtol=0, atol=0
    )
    q.write_in_file(tmp_path, "weights", TractNNEF.latest())
    with (tmp_path / "weights.dat").open("rb") as stream:
        DatBinHeader.from_dat_file(stream)
        payload = np.frombuffer(stream.read(), dtype=np.uint8).reshape(-1, 18)
    actual_scales = payload[:, :2].copy().view("<f2").reshape(-1)
    nibbles = payload[:, 2:]
    actual_codes = np.concatenate([nibbles & 15, nibbles >> 4], axis=1)
    np.testing.assert_array_equal(actual_scales, expected_scales.numpy())
    np.testing.assert_array_equal(actual_codes, expected_codes.numpy())


@pytest.mark.parametrize("group_size, symmetric", [(16, True), (32, False)])
def test_q40_rejects_incompatible_gptq_scheme(group_size, symmetric):
    gran = Granularity.per_group([group_size])
    calib = NormZpScaleCalibration(nbits=4, symmetric=symmetric, granularity=gran)
    weight = torch.linspace(-1, 2, 128).reshape(2, 64)
    calib.init()
    calib.update(weight)
    scheme = calib.compute_qscheme()
    with pytest.raises(ValueError, match="symmetric 4-bit quantization"):
        _q4_0_from_assignment(
            QuantizedWeight(scheme.quantize_as_u8(weight), scheme, weight.dtype),
            "weight",
        )


def _run_q40_gptq(scale_first_block: float):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(64, 16, bias=False)).eval()
    with torch.no_grad():
        model[0].weight[:, :32] *= scale_first_block
    original = model[0].weight.detach().clone()
    estimator = GptqEstimatorConfig(
        group_size=32, weight_transform="q4_0_packed"
    ).build()
    recipe = QuantizationRecipe(
        param_estimator=estimator, adapter=WeightAssignmentAdapter()
    )
    QuantizationPipeline(recipe).run(model, [torch.randn(8, 64, 64)])
    return model, original


def test_q40_keeps_groups_whose_scale_underflows_f16():
    model, original = _run_q40_gptq(1e-9)
    q = model[0].weight
    assert isinstance(q, QTensorTractScaleOnly)
    assert torch.all(q.qscheme.scale > 0)
    decoded = q.decompress()
    # Stored as exact zeros: the error stays bounded by the real weights.
    assert torch.all(decoded[:, :32] == 0)
    assert (decoded[:, :32] - original[:, :32]).abs().max() < 1e-6
    assert (decoded[:, 32:] - original[:, 32:]).abs().max() < 0.2


def test_q40_skips_layer_whose_scale_overflows_f16():
    model, _ = _run_q40_gptq(1e7)
    weight = model[0].weight
    assert not isinstance(weight, QTensorTractScaleOnly)
    assert torch.isfinite(weight).all()


def test_q40_skips_layer_that_would_decode_outside_f16():
    # A scale near 2e4 is representable, but code 0 decodes to about -1.6e5.
    model, _ = _run_q40_gptq(1e6)
    weight = model[0].weight
    assert not isinstance(weight, QTensorTractScaleOnly)
    assert torch.isfinite(weight).all()


def _run_conv2d(format_name, kernel):
    torch.manual_seed(3)
    model = nn.Sequential(nn.Conv2d(64, 16, kernel, bias=False)).eval()
    original = model[0].weight.detach().clone()
    estimator = GptqEstimatorConfig(
        target="conv2d", group_size=32, weight_transform=format_name
    ).build()
    recipe = QuantizationRecipe(
        param_estimator=estimator, adapter=WeightAssignmentAdapter()
    )
    QuantizationPipeline(recipe).run(model, [torch.randn(8, 64, 8, 8)])
    return model[0].weight, original


@pytest.mark.parametrize(
    "format_name, kernel",
    [("q4_0", 1), ("q4_0", 3), ("q4_0_packed", 2)],
)
def test_q40_pipeline_handles_conv2d(format_name, kernel):
    q, original = _run_conv2d(format_name, kernel)
    assert isinstance(q, QTensorTractScaleOnly)
    decoded = q.decompress()
    assert decoded.shape == original.shape
    assert (decoded - original).abs().max() < 0.2


@pytest.mark.parametrize("kernel", [1, 3])
def test_q40_packed_conv2d_with_odd_kernel_stays_float(kernel):
    # Packing cannot restore this shape, so it must not decode flat.
    weight, original = _run_conv2d("q4_0_packed", kernel)
    assert not isinstance(weight, QTensorTractScaleOnly)
    assert weight.shape == original.shape
    assert (weight - original).abs().max() < 0.2


def test_q40_rejection_keeps_other_layers():
    torch.manual_seed(5)

    class Branches(nn.Module):
        def __init__(self):
            super().__init__()
            self.compatible = nn.Linear(96, 8, bias=False)
            self.rejected = nn.Linear(64, 8, bias=False)

        def forward(self, x):
            return self.compatible(x[..., :96]) + self.rejected(x[..., :64])

    model = Branches().eval()
    # Configs reject this group size at build(), so bypass them to reach the
    # per-layer path: 96 keeps groups of 48, 64 falls back to per-channel.
    estimator = GPTQEstimator(group_size=48, weight_transform="q4_0")
    recipe = QuantizationRecipe(
        param_estimator=estimator, adapter=WeightAssignmentAdapter()
    )
    QuantizationPipeline(recipe).run(model, [torch.randn(8, 96)])
    assert not isinstance(model.compatible.weight, QTensorTractScaleOnly)
    assert isinstance(model.rejected.weight, QTensorTractScaleOnly)
