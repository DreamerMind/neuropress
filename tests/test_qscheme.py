"""Tests for Granularity, QSchemeDescriptor, QSchemeGroup, PanelSpec, TensorSlice."""

import pytest
import torch
from pydantic import ValidationError

from neuropress.qscheme import (
    Granularity,
    GranularityType,
    PanelSpec,
    QSchemeDescriptor,
    QSchemeGroup,
    TensorSlice,
)
from neuropress.qscheme.zp_scale import ZpScaleScheme


def _zp_scale() -> ZpScaleScheme:
    """Minimal ZpScaleScheme for use as a QScheme stub in tests."""
    return ZpScaleScheme(
        zero_point=torch.zeros(1),
        scale=torch.ones(1),
        nbits=4,
        calibration_data_shape=[1],
        qscheme_data_shape=[1],
    )


# ---------------------------------------------------------------------------
# Granularity
# ---------------------------------------------------------------------------


def test_granularity_all_tensor():
    g = Granularity.all_tensor()
    assert g.granularity_type is GranularityType.ALL
    assert g.group_sizes is None


def test_granularity_per_channel():
    g = Granularity.per_channel()
    assert g.granularity_type is GranularityType.PER_CHANNEL


def test_granularity_per_group():
    g = Granularity.per_group([32])
    assert g.granularity_type is GranularityType.PER_GROUP
    assert g.group_sizes == [32]


def test_granularity_per_group_requires_sizes():
    with pytest.raises(ValidationError):
        Granularity(granularity_type=GranularityType.PER_GROUP)


def test_granularity_equality():
    assert Granularity.per_group([32]) == Granularity.per_group([32])
    assert Granularity.per_group([32]) != Granularity.per_group([64])


def test_granularity_hashable():
    g = Granularity.per_group([32])
    d = {g: "value"}
    assert d[g] == "value"


# ---------------------------------------------------------------------------
# QSchemeDescriptor
# ---------------------------------------------------------------------------


def test_qscheme_descriptor_construction():
    d = QSchemeDescriptor(
        domain="weight",
        nbits=4,
        symmetric=True,
        granularity=Granularity.per_group([32]),
    )
    assert d.nbits == 4
    assert d.domain == "weight"


def test_qscheme_descriptor_activation_domain():
    d = QSchemeDescriptor(
        domain="activation",
        nbits=8,
        symmetric=False,
        granularity=Granularity.per_channel(),
    )
    assert d.domain == "activation"


def test_qscheme_descriptor_hashable():
    d = QSchemeDescriptor(
        domain="weight",
        nbits=4,
        symmetric=True,
        granularity=Granularity.per_group([32]),
    )
    s = {d}
    assert d in s


# ---------------------------------------------------------------------------
# TensorSlice
# ---------------------------------------------------------------------------


def test_tensor_slice_size():
    s = TensorSlice(dim=0, start=0, end=32)
    assert s.size == 32


def test_tensor_slice_equality():
    a = TensorSlice(dim=1, start=4, end=8)
    b = TensorSlice(dim=1, start=4, end=8)
    assert a == b


# ---------------------------------------------------------------------------
# QSchemeGroup
# ---------------------------------------------------------------------------


def test_qscheme_group_construction():
    panels = [
        PanelSpec(region=TensorSlice(dim=1, start=0, end=32), qscheme=_zp_scale()),
        PanelSpec(region=TensorSlice(dim=1, start=32, end=64), qscheme=_zp_scale()),
    ]
    g = QSchemeGroup(panels=panels)
    assert len(g.panels) == 2


def test_qscheme_group_total_coverage():
    panels = [
        PanelSpec(region=TensorSlice(dim=1, start=0, end=32), qscheme=_zp_scale()),
        PanelSpec(region=TensorSlice(dim=1, start=32, end=64), qscheme=_zp_scale()),
    ]
    g = QSchemeGroup(panels=panels)
    assert g.total_coverage(dim=1) == 64
