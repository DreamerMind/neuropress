"""Named modules get a codebook of their own, outside their sharing group."""

import pytest
import torch
from torch import nn

import neuropress.t2n.qtensors  # noqa: F401  (breaks a circular import)
from neuropress.estimators import VPTQGroupEstimator
from neuropress.pipeline import QuantizationPipeline
from neuropress.recipe_config import QuantizationRecipeConfig


class _Layer(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.up_proj = nn.Linear(32, 48, bias=False)
        self.down_proj = nn.Linear(48, 32, bias=False)
        self.down_proj.weight.data.mul_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down_proj(torch.relu(self.up_proj(x)))


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        # layer 1's down_proj is 20x larger: an outlier the shared codebook
        # would have to cover
        self.layers = nn.ModuleList([_Layer(1.0), _Layer(20.0), _Layer(1.0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _run(own):
    model = _Model().eval()
    recipe = QuantizationRecipeConfig.model_validate(
        {
            "estimator": {
                "type": "vptq",
                "target": "linear",
                "vector_len": 4,
                "num_centroids": 64,
                "grouping": "by_leaf_name",
                "own_codebook_modules": own,
                "seed": 0,
            },
            "kernel_lowering": None,
        }
    )
    pipeline = QuantizationPipeline(recipe.build())
    pipeline.run(model, [torch.randn(4, 8, 32)], estimation_device="cpu")
    return model, pipeline.last_plan


def _books(plan, leaf):
    return {
        name: qt.qscheme.centroids
        for name, qt in plan.weight_assignments.items()
        if name.endswith(leaf)
    }


def test_own_module_gets_its_own_codebook():
    _, plan = _run(["layers.1.down_proj"])
    books = _books(plan, "down_proj")
    torch.testing.assert_close(books["layers.0.down_proj"], books["layers.2.down_proj"])
    assert not torch.equal(books["layers.1.down_proj"], books["layers.0.down_proj"])
    # the other leaf groups are untouched
    ups = list(_books(plan, "up_proj").values())
    assert all(torch.equal(ups[0], u) for u in ups[1:])
    # The outlier dominates a shared fit; once it has its own codebook the
    # ordinary layers of its group are fitted better (over 2x with the CPU
    # k-means backend, 1.3x with the CUDA one).
    _, shared_plan = _run([])
    ref = _Model()

    def rel_err(p, i):
        q = p.weight_assignments[f"layers.{i}.down_proj"].decompress().float()
        w = ref.layers[i].down_proj.weight.detach().float()
        return float((q.cpu() - w).norm() / w.norm())

    for i in (0, 2):
        assert rel_err(plan, i) < 0.9 * rel_err(shared_plan, i)


def test_unknown_own_module_is_rejected():
    with pytest.raises(ValueError, match=r"layers\.9\.down_proj"):
        _run(["layers.9.down_proj"])


def test_group_keys():
    est = VPTQGroupEstimator(grouping="by_leaf_name", own_codebook_modules=("a.x",))
    est._target_modules = {"a.x": nn.Linear(4, 4), "b.x": nn.Linear(4, 4)}
    assert set(est._make_groups()) == {"a.x", "x"}
