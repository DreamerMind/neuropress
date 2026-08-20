"""Tests for WeightTarget and ActivationTarget."""

from neuropress.targets import ActivationTarget, WeightTarget

# ---------------------------------------------------------------------------
# WeightTarget
# ---------------------------------------------------------------------------


def test_weight_target_key():
    t = WeightTarget(module_id="n1", param_name="weight")
    assert t.key == "n1/weight"


def test_weight_target_equality():
    t1 = WeightTarget(module_id="n1", param_name="weight")
    t2 = WeightTarget(module_id="n1", param_name="weight")
    assert t1 == t2


def test_weight_target_inequality():
    t1 = WeightTarget(module_id="n1", param_name="weight")
    t2 = WeightTarget(module_id="n2", param_name="weight")
    assert t1 != t2


# ---------------------------------------------------------------------------
# ActivationTarget
# ---------------------------------------------------------------------------


def test_activation_target_key():
    t = ActivationTarget(value_id="v1")
    assert t.key == "v1"


def test_activation_target_equality():
    t1 = ActivationTarget(value_id="v1")
    t2 = ActivationTarget(value_id="v1")
    assert t1 == t2
