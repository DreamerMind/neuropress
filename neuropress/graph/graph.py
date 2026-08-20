"""QGraph: dataflow graph built from runtime execution traces."""

from __future__ import annotations

from pydantic import BaseModel

from .node import QNode, QValue


class QGraph(BaseModel):
    """Module-level dataflow graph built from runtime execution.

    Edges are implicit via producer_id → QValue → consumer_ids.
    """

    nodes: dict[str, QNode] = {}
    values: dict[str, QValue] = {}

    def add_node(self, node: QNode) -> None:
        """Register *node* in the graph by its id."""
        self.nodes[node.id] = node

    def add_value(self, value: QValue) -> None:
        """Register *value* in the graph by its id."""
        self.values[value.id] = value

    def get_node(self, node_id: str) -> QNode:
        """Return the node with *node_id*."""
        return self.nodes[node_id]

    def get_value(self, value_id: str) -> QValue:
        """Return the value with *value_id*."""
        return self.values[value_id]

    def producers_of(self, value_id: str) -> QNode:
        """Return the node that produced *value_id*."""
        value = self.values[value_id]
        return self.nodes[value.producer_id]

    def consumers_of(self, value_id: str) -> list[QNode]:
        """Return all nodes that consume *value_id*."""
        value = self.values[value_id]
        return [self.nodes[nid] for nid in value.consumer_ids if nid in self.nodes]
