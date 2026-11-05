"""Regression tests for GPTQ defects found while reviewing ternary PTQ."""

import pytest
import torch
from torch import nn

from neuropress.calibration.probe import StatisticsRegistry
from neuropress.estimators import GPTQEstimator
from neuropress.methods.gptq import GPTQCalibration
from neuropress.methods.zp_scale_calib import NormZpScaleCalibration
from neuropress.qscheme import Granularity


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gptq_accepts_hessian_on_another_device():
    """A disk-cache hit restores the Hessian on CPU while weights are on CUDA."""
    granularity = Granularity.per_group([32])
    module = nn.Linear(64, 32, bias=False).cuda()
    calibration = GPTQCalibration(
        granularity=granularity,
        possible_quantizers=[
            NormZpScaleCalibration(nbits=4, symmetric=True, granularity=granularity)
        ],
    )
    calibration.weight_tensor = module.weight
    calibration.module_class = nn.Linear
    calibration.module = module
    calibration.name = "q_proj"
    calibration.init()
    x = torch.randn(256, 64)
    calibration.hessian = 2.0 * x.T @ x / x.shape[0]
    calibration.nsamples = x.shape[0]

    calibration.compute_qscheme()

    assert module.weight.device.type == "cuda"
    assert torch.isfinite(module.weight).all()


def test_gptq_estimator_raises_when_a_layer_fails():
    model = nn.Sequential(nn.Linear(64, 32, bias=False))
    estimator = GPTQEstimator(group_size=32, nbits=4)
    estimator.prepare(model)

    # No Hessian was collected, so GPTQ fails for the only target.
    with pytest.raises(RuntimeError, match="GPTQ failed for 1/1 modules"):
        estimator.estimate(None, StatisticsRegistry(), None)
