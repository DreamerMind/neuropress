"""Tests for ZpScaleScheme, ZpScaleCalibration, MinMaxZpScaleCalibration,
NormZpScaleCalibration."""

import math

import pytest
import torch

from neuropress.methods.zp_scale_calib import (
    MinMaxZpScaleCalibration,
    NormZpScaleCalibration,
)
from neuropress.qscheme import Granularity
from neuropress.qscheme.zp_scale import ZpScaleScheme

# ---------------------------------------------------------------------------
# ZpScaleScheme helpers
# ---------------------------------------------------------------------------


def _make_per_group_scheme(rows=4, cols=32, group_size=32, nbits=4, symmetric=True):
    """Create a minimal ZpScaleScheme for per-group [group_size]."""
    n_groups = cols // group_size
    qscheme_data_shape = [rows, 1, n_groups, 1]
    calibration_data_shape = [rows, 1, n_groups, group_size]
    scale = torch.ones(*qscheme_data_shape) * 0.1
    zp = torch.full(qscheme_data_shape, (2**nbits) // 2, dtype=torch.int32)
    return ZpScaleScheme(
        zero_point=zp,
        scale=scale,
        nbits=nbits,
        calibration_data_shape=calibration_data_shape,
        qscheme_data_shape=qscheme_data_shape,
    )


# ---------------------------------------------------------------------------
# ZpScaleScheme
# ---------------------------------------------------------------------------


class TestZpScaleScheme:
    def test_quantize_dequantize_roundtrip_full_weight(self):
        rows, cols, group_size = 4, 64, 32
        scheme = _make_per_group_scheme(rows=rows, cols=cols, group_size=group_size)
        w = torch.randn(rows, cols)
        q = scheme.quantize_as_u8(w)
        dq = scheme.dequantize(q, w.dtype)
        assert q.shape == w.shape
        assert dq.shape == w.shape

    def test_quantize_as_u8_dtype(self):
        scheme = _make_per_group_scheme()
        w = torch.randn(4, 32)
        q = scheme.quantize_as_u8(w)
        assert q.dtype == torch.uint8

    def test_quantize_as_torch_dtype(self):
        scheme = _make_per_group_scheme()
        w = torch.randn(4, 32)
        q = scheme.quantize_as_torch(w)
        assert q.dtype == torch.int32

    def test_column_broadcast(self):
        """Single column [rows] must work with a per-group scheme."""
        rows = 8
        scheme = _make_per_group_scheme(rows=rows, cols=32, group_size=32)
        col = torch.randn(rows)
        q = scheme.quantize_as_u8(col)
        dq = scheme.dequantize(q, col.dtype)
        assert q.shape == col.shape
        assert dq.shape == col.shape

    def test_maxq(self):
        scheme = _make_per_group_scheme(nbits=4)
        assert scheme.maxq == 15

    def test_dims_to_collapse(self):
        scheme = _make_per_group_scheme()
        assert scheme.dims_indexes_to_collapse == [1, 3]

    def test_dims_to_keep(self):
        scheme = _make_per_group_scheme()
        assert scheme.dims_indexes_to_keep == [0, 2]

    def test_quantize_clamps_to_range(self):
        scheme = _make_per_group_scheme(nbits=4)
        w = torch.full((4, 32), 1e6)  # extreme values
        q = scheme.quantize_as_u8(w)
        assert q.max().item() <= scheme.maxq
        assert q.min().item() >= 0

    def test_inherits_from_torch_to_nnef_qscheme(self):
        from torch_to_nnef.tensor.quant import QScheme

        scheme = _make_per_group_scheme()
        assert isinstance(scheme, QScheme)


# ---------------------------------------------------------------------------
# ZpScaleCalibration helpers
# ---------------------------------------------------------------------------


def _make_weight(rows=4, cols=64):
    return torch.randn(rows, cols)


# ---------------------------------------------------------------------------
# MinMaxZpScaleCalibration
# ---------------------------------------------------------------------------


class TestMinMaxZpScaleCalibration:
    def test_compute_qscheme_per_group(self):
        calib = MinMaxZpScaleCalibration(
            nbits=4,
            symmetric=True,
            granularity=Granularity.per_group([32]),
        )
        w = _make_weight(4, 64)
        calib.init()
        calib.update(w)
        scheme = calib.compute_qscheme()
        assert isinstance(scheme, ZpScaleScheme)
        assert scheme.nbits == 4

    def test_shapes_per_group(self):
        calib = MinMaxZpScaleCalibration(
            nbits=4,
            symmetric=True,
            granularity=Granularity.per_group([32]),
        )
        w = _make_weight(4, 64)
        calib.init()
        calib.update(w)
        scheme = calib.compute_qscheme()
        # qscheme_data_shape = [rows, 1, n_groups, 1] = [4, 1, 2, 1]
        assert scheme.qscheme_data_shape == [4, 1, 2, 1]

    def test_accumulates_multiple_batches(self):
        calib = MinMaxZpScaleCalibration(
            nbits=8,
            symmetric=False,
            granularity=Granularity.per_channel(),
        )
        calib.init()
        calib.update(_make_weight(4, 16))
        calib.update(_make_weight(4, 16))
        scheme = calib.compute_qscheme()
        assert scheme is not None

    def test_bpw(self):
        calib = MinMaxZpScaleCalibration(
            nbits=4,
            symmetric=True,
            granularity=Granularity.per_group([32]),
        )
        bpw = calib.bpw()
        # 4 * 32 + 16 = 144 bits / 32 elements = 4.5
        assert math.isclose(bpw, 4.5, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# NormZpScaleCalibration  (MSE)
# ---------------------------------------------------------------------------


class TestNormZpScaleCalibration:
    def test_compute_qscheme_per_group(self):
        calib = NormZpScaleCalibration(
            nbits=4,
            symmetric=True,
            granularity=Granularity.per_group([32]),
            grid=10,  # fast for testing
        )
        w = _make_weight(4, 64)
        calib.init()
        calib.update(w)
        scheme = calib.compute_qscheme()
        assert isinstance(scheme, ZpScaleScheme)
        assert scheme.nbits == 4

    def test_qscheme_closer_than_minmax(self):
        """NormZpScale should achieve equal or better MSE than MinMax."""
        torch.manual_seed(0)
        w = torch.randn(8, 32)

        def _mse(calib_cls, **kw):
            c = calib_cls(
                nbits=4, symmetric=True, granularity=Granularity.per_group([32]), **kw
            )
            c.init()
            c.update(w)
            s = c.compute_qscheme()
            dq = s.dequantize(s.quantize_as_u8(w), w.dtype)
            return (w - dq).pow(2).mean().item()

        mse_mm = _mse(MinMaxZpScaleCalibration)
        mse_norm = _mse(NormZpScaleCalibration, grid=20)
        assert mse_norm <= mse_mm + 1e-5  # norm should not be worse

    def test_rejects_multiple_distinct_samples(self):
        calib = NormZpScaleCalibration(
            nbits=4,
            symmetric=True,
            granularity=Granularity.per_group([32]),
        )
        calib.init()
        calib.update(_make_weight(4, 32))
        with pytest.raises(NotImplementedError):
            calib.update(_make_weight(4, 32))  # different data
