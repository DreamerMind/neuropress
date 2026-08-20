"""QGraph node and value types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class QNode(BaseModel):
    """Represents a computation node (initially: module call)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    kind: str = "module_call"
    module_path: str = ""  # fully-qualified dotted path in the model
    metadata: dict[str, Any] = {}


class QValue(BaseModel):
    """Represents a produced tensor (activation identity)."""

    id: str
    producer_id: str  # QNode.id that produced this value
    consumer_ids: set[str] = set()  # QNode.ids that consume this value

    model_config = ConfigDict(frozen=False)
