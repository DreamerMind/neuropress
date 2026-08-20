"""QScheme descriptors: backend-constrained layout definitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, model_validator


class GranularityType(StrEnum):
    """Enumeration of supported quantization granularities."""

    ALL = "all"  # per-tensor
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


class Granularity(BaseModel):
    """Quantization granularity spec."""

    granularity_type: GranularityType
    group_sizes: list[int] | None = None

    @model_validator(mode="after")
    def _check_group_sizes(self) -> Granularity:
        if self.granularity_type is GranularityType.PER_GROUP:
            if not self.group_sizes:
                raise ValueError("group_sizes must be set for PER_GROUP granularity")
        return self

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def all_tensor(cls) -> Granularity:
        """Return a per-tensor granularity spec."""
        return cls(granularity_type=GranularityType.ALL)

    @classmethod
    def per_channel(cls) -> Granularity:
        """Return a per-channel granularity spec."""
        return cls(granularity_type=GranularityType.PER_CHANNEL)

    @classmethod
    def per_group(cls, group_sizes: list[int]) -> Granularity:
        """Return a per-group granularity spec with the given *group_sizes*."""
        return cls(granularity_type=GranularityType.PER_GROUP, group_sizes=group_sizes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Granularity):
            return NotImplemented
        return (
            self.granularity_type == other.granularity_type
            and self.group_sizes == other.group_sizes
        )

    def __hash__(self) -> int:
        return hash((self.granularity_type, tuple(self.group_sizes or [])))


class QSchemeDescriptor(BaseModel):
    """Static backend-supported quantization layout definition.

    Descriptors are finite and backend-constrained; the estimator
    selects among a list of allowed descriptors.
    """

    domain: Literal["weight", "activation"]
    nbits: int
    symmetric: bool
    granularity: Granularity

    def __hash__(self) -> int:
        """Hash based on all descriptor fields."""
        return hash((self.domain, self.nbits, self.symmetric, self.granularity))
