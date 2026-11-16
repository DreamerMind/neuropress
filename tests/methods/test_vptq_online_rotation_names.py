"""An online-rotated layer keeps its type's codebook (down_proj, not 'linear')."""

import pytest
import torch
from torch import nn

from neuropress.estimators import VPTQGroupEstimator
from neuropress.transforms.rotation import HadamardInputLinear


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = HadamardInputLinear(
            nn.Linear(16, 8, bias=False), torch.ones(16)
        )


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({"mlp": _Mlp()}) for _ in range(2)])


@pytest.mark.parametrize("grouping", ["by_leaf_name", "per_layer"])
def test_wrapped_down_proj_uses_the_down_proj_codebook_size(grouping):
    est = VPTQGroupEstimator(
        num_centroids=64,
        vector_len=4,
        grouping=grouping,
        num_centroids_per_group={"up_proj": 16, "down_proj": 32},
    )
    est.prepare(_Model())
    groups = est._make_groups()
    sizes = {
        name: est.centroids_for_group(key)
        for key, members in groups.items()
        for name, _ in members
    }
    assert sizes == {
        "layers.0.mlp.up_proj": 16,
        "layers.1.mlp.up_proj": 16,
        "layers.0.mlp.down_proj.linear": 32,
        "layers.1.mlp.down_proj.linear": 32,
    }
    if grouping == "by_leaf_name":
        assert set(groups) == {"up_proj", "down_proj"}
