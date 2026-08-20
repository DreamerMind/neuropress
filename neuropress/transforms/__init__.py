"""Transforms sub-package: structural and weight transforms for model adaptation.

Structural transforms
---------------------
- :mod:`.split_mixed_linear` — split mixed-precision linears into
  per-precision sub-linears (:class:`MixedPrecisionLinear`).

Weight transforms (registered into :data:`.WEIGHT_TRANSFORM_REGISTRY`)
-----------------------------------------------------------------------
- :mod:`.weight_only` — tract Q4 / Q2 quantization (``"q4_0"`` /
  ``"q4_0_unpacked"``, ``"q4_0_packed"``, ``"q2_0_packed"``, ``"identity"``
  keys); also exports :func:`_make_packed_qtensor` for use by
  :class:`~neuropress.estimators.GPTQPanelEstimator`.

Importing this package registers all weight transforms as a side effect.
"""

# weight_only must be imported before split_mixed_linear: t2n.registry imports
# Q4_X_GROUP_SIZE from this package during its own initialization, so the name
# must be present on the partially-initialized transforms module.
from .weight_only import Q4_X_GROUP_SIZE, _make_packed_qtensor  # noqa: I001
from .split_mixed_linear import (
    MixedPrecisionLinear,
    apply_split_to_model,
    split_mixed_precision_linear,
)

__all__ = [
    "Q4_X_GROUP_SIZE",
    "MixedPrecisionLinear",
    "_make_packed_qtensor",
    "apply_split_to_model",
    "split_mixed_precision_linear",
]
