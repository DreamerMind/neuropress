"""Heterogeneous QScheme grouping for multi-panel / mixed-precision layouts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from torch_to_nnef.tensor.quant import QScheme


class TensorSlice(BaseModel):
    """A contiguous slice along a single dimension of a tensor."""

    dim: int
    start: int
    end: int  # exclusive

    @property
    def size(self) -> int:
        """Number of elements in this slice."""
        return self.end - self.start


class PanelSpec(BaseModel):
    """Associates a tensor region with a concrete QScheme."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    region: TensorSlice
    qscheme: QScheme


class QSchemeGroup(BaseModel):
    """A heterogeneous collection of panels covering a single weight tensor.

    Used when mixed precision or multi-panel layout is required.
    Each panel covers a distinct slice of the tensor with its own scheme.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    panels: list[PanelSpec]

    def total_coverage(self, dim: int) -> int:
        """Return total number of elements covered by panels along *dim*."""
        return sum(p.region.size for p in self.panels if p.region.dim == dim)
