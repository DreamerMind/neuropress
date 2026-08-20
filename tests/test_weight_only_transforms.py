"""Regression tests for weight-only quantization transforms.

Covers the bug where ``offloaded_tensor_qtensor`` calls the internal
``_calib`` callback with a flat 1-D storage view instead of the original
multi-dimensional shape, causing ``QTensorTractScaleOnly`` to raise an
``AssertionError`` (it requires ``ndim in {2, 3, 4}``).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

# t2n must be imported first to resolve the circular-import via transforms/__init__.
import neuropress.t2n  # noqa: F401
from neuropress.transforms.weight_only import (
    _make_packed_qtensor,
    _make_q4_0_packed_tensor,
    _make_q4_0_tensor,
    _make_q4_1_packed_tensor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OFFLOAD_PATH = "neuropress.transforms.weight_only.offloaded_tensor_qtensor"


def _flat_offload(q_fn, tensor, suffix):
    """Simulate offloaded_tensor_qtensor passing a 1-D storage view to q_fn."""
    return q_fn(tensor.flatten())


def _normal_offload(q_fn, tensor, suffix):
    """Simulate offloaded_tensor_qtensor passing the tensor as-is to q_fn."""
    return q_fn(tensor)


# ---------------------------------------------------------------------------
# Regression: flat 1-D callback (the original crash)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (64, 32),  # plain Linear weight
        (16, 4, 32),  # 3-D weight (Conv-like)
        (8, 2, 4, 4),  # 4-D weight (Conv 2-D)
    ],
)
def test_q4_0_tensor_survives_flat_callback(shape):
    """_make_q4_0_tensor must not crash when _calib receives a 1-D view."""
    weight = torch.randn(*shape)
    with patch(_OFFLOAD_PATH, _flat_offload):
        result = _make_q4_0_tensor(weight, "mod")
    assert result is not None


@pytest.mark.parametrize(
    "shape",
    [
        (64, 32),
        (16, 4, 32),
        (8, 2, 4, 4),
    ],
)
def test_q4_0_packed_tensor_survives_flat_callback(shape):
    """_make_q4_0_packed_tensor must not crash when _calib receives a 1-D view."""
    weight = torch.randn(*shape)
    with patch(_OFFLOAD_PATH, _flat_offload):
        result = _make_q4_0_packed_tensor(weight, "mod")
    assert result is not None


def test_make_packed_qtensor_survives_flat_callback():
    """_make_packed_qtensor must not crash when _calib receives a 1-D view."""
    weight = torch.randn(64, 32)
    # QTensorTractScaleOnly asserts n_bits == 4; only test the Q4 path here.
    with patch(_OFFLOAD_PATH, _flat_offload):
        result = _make_packed_qtensor(weight, "mod", nbits=4)
    assert result is not None


def test_q4_1_packed_tensor_survives_flat_callback():
    """_make_q4_1_packed_tensor must not crash when _calib receives a 1-D view."""
    weight = torch.randn(64, 32)
    with patch(_OFFLOAD_PATH, _flat_offload):
        result = _make_q4_1_packed_tensor(weight, "mod")
    assert result is not None


# ---------------------------------------------------------------------------
# Sanity: normal (non-flat) path still works
# ---------------------------------------------------------------------------


def test_q4_0_packed_tensor_normal_path():
    weight = torch.randn(64, 32)
    with patch(_OFFLOAD_PATH, _normal_offload):
        result = _make_q4_0_packed_tensor(weight, "mod")
    assert result is not None


def test_q4_1_packed_tensor_normal_path():
    weight = torch.randn(64, 32)
    with patch(_OFFLOAD_PATH, _normal_offload):
        result = _make_q4_1_packed_tensor(weight, "mod")
    assert result is not None


def test_make_packed_qtensor_normal_path():
    # QTensorTractScaleOnly asserts n_bits == 4; only test the Q4 path here.
    weight = torch.randn(64, 32)
    with patch(_OFFLOAD_PATH, _normal_offload):
        result = _make_packed_qtensor(weight, "mod", nbits=4)
    assert result is not None
