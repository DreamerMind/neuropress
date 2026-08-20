"""Runtime module graph provider.

Builds a QGraph by running a forward pass and recording module calls
via forward hooks.  No symbolic tracing (FX/Dynamo) is required; the
graph captures the actual execution order.

Upgrade path: replace this class with a DynamoOpGraphProvider that
reuses the same QGraph API without touching any downstream code.
"""

from __future__ import annotations

import uuid
from typing import Any

import torch
from torch import nn

from .graph import QGraph
from .node import QNode, QValue


class RuntimeModuleGraphProvider:
    """Build a QGraph by tracing a single forward pass via module hooks."""

    def build(
        self,
        model: nn.Module,
        example_inputs: Any,
    ) -> QGraph:
        """Run *example_inputs* through *model* and return the captured graph.

        Args:
            model: The nn.Module to trace.
            example_inputs: A single tensor, a tuple/list of tensors, or a
                dict of keyword arguments accepted by ``model.forward``.
        """
        graph = QGraph()
        hooks: list[Any] = []

        # Map from Python object id(module) → QNode.id for deduplication.
        module_to_node: dict[int, str] = {}
        # Track the most-recently produced QValue id so consumers can link back.
        last_value_id: list[str | None] = [None]

        def _make_pre_hook(path: str, module: nn.Module):
            def hook(mod, inputs):
                node_id = str(uuid.uuid4())
                module_to_node[id(mod)] = node_id
                node = QNode(
                    id=node_id,
                    kind="module_call",
                    module_path=path,
                )
                graph.add_node(node)
                # Record that this node consumes the previous value.
                prev = last_value_id[0]
                if prev is not None and prev in graph.values:
                    graph.values[prev].consumer_ids.add(node_id)

            return hook

        def _make_post_hook(module: nn.Module):
            def hook(mod, inputs, output):
                node_id = module_to_node.get(id(mod))
                if node_id is None:
                    return
                value_id = str(uuid.uuid4())
                value = QValue(id=value_id, producer_id=node_id)
                graph.add_value(value)
                last_value_id[0] = value_id

            return hook

        for name, module in model.named_modules():
            if name == "":
                continue  # skip root
            hooks.append(module.register_forward_pre_hook(_make_pre_hook(name, module)))
            hooks.append(module.register_forward_hook(_make_post_hook(module)))

        try:
            with torch.inference_mode():
                if isinstance(example_inputs, dict):
                    model(**example_inputs)
                elif isinstance(example_inputs, list | tuple):
                    model(*example_inputs)
                else:
                    model(example_inputs)
        finally:
            for h in hooks:
                h.remove()

        return graph
