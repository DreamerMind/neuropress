"""Tests for QNode, QValue, QGraph, RuntimeModuleGraphProvider."""

import pytest
import torch
from torch import nn

from neuropress.graph import QGraph, QNode, QValue, RuntimeModuleGraphProvider

# ---------------------------------------------------------------------------
# QNode / QValue
# ---------------------------------------------------------------------------


def test_qnode_basic_construction():
    node = QNode(id="n1", kind="module_call", module_path="layers.0")
    assert node.id == "n1"
    assert node.kind == "module_call"
    assert node.module_path == "layers.0"


def test_qnode_default_kind():
    node = QNode(id="n2")
    assert node.kind == "module_call"


def test_qnode_metadata():
    node = QNode(id="n3", metadata={"foo": 42})
    assert node.metadata["foo"] == 42


def test_qvalue_basic_construction():
    v = QValue(id="v1", producer_id="n1")
    assert v.id == "v1"
    assert v.producer_id == "n1"
    assert len(v.consumer_ids) == 0


def test_qvalue_consumers():
    v = QValue(id="v1", producer_id="n1", consumer_ids={"n2", "n3"})
    assert "n2" in v.consumer_ids
    assert "n3" in v.consumer_ids


# ---------------------------------------------------------------------------
# QGraph
# ---------------------------------------------------------------------------


def _make_graph():
    g = QGraph()
    n1 = QNode(id="n1", module_path="a")
    n2 = QNode(id="n2", module_path="b")
    v1 = QValue(id="v1", producer_id="n1", consumer_ids={"n2"})
    g.add_node(n1)
    g.add_node(n2)
    g.add_value(v1)
    return g, n1, n2, v1


def test_qgraph_add_and_retrieve():
    g, n1, n2, v1 = _make_graph()
    assert g.get_node("n1") is n1
    assert g.get_node("n2") is n2
    assert g.get_value("v1") is v1


def test_qgraph_producers_of():
    g, n1, _, _v1 = _make_graph()
    assert g.producers_of("v1") is n1


def test_qgraph_consumers_of():
    g, _, n2, _ = _make_graph()
    consumers = g.consumers_of("v1")
    assert n2 in consumers


def test_qgraph_missing_key_raises():
    g = QGraph()
    with pytest.raises(KeyError):
        g.get_node("missing")


# ---------------------------------------------------------------------------
# RuntimeModuleGraphProvider
# ---------------------------------------------------------------------------


def _simple_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_graph_has_nodes():
    model = _simple_model()
    provider = RuntimeModuleGraphProvider()
    x = torch.randn(2, 4)
    graph = provider.build(model, x)
    assert len(graph.nodes) > 0


def test_graph_has_values():
    model = _simple_model()
    provider = RuntimeModuleGraphProvider()
    x = torch.randn(2, 4)
    graph = provider.build(model, x)
    assert len(graph.values) > 0


def test_module_paths_captured():
    model = _simple_model()
    provider = RuntimeModuleGraphProvider()
    x = torch.randn(2, 4)
    graph = provider.build(model, x)
    paths = {n.module_path for n in graph.nodes.values()}
    # Sequential children include the Linear and ReLU layers
    assert any("0" in p for p in paths)


def test_build_with_dict_input():
    class DictModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    model = DictModel()
    provider = RuntimeModuleGraphProvider()
    graph = provider.build(model, {"x": torch.randn(1, 4)})
    assert len(graph.nodes) > 0
