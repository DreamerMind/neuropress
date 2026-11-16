"""Selective-depth residual VQ: two-stage coding of the leading column groups."""

from copy import deepcopy

import pytest
import torch
from torch import nn

import neuropress.t2n.qtensors  # noqa: F401  (breaks a circular import)
from neuropress.estimators import VPTQGroupEstimator
from neuropress.methods.residual_vq import (
    residual_groups_for,
    two_stage_assign,
    weighted_kmeans,
)
from neuropress.t2n.qtensors import load_qtensor, save_qtensor
from neuropress.transforms.vq import ProxySelector, QTensorTractVQ, VQScheme
from neuropress.transforms.vq_packing import IndexPacking

VL = 4


def test_residual_groups_rounds_and_caps():
    assert residual_groups_for(1024, 0.25) == 256
    assert residual_groups_for(10, 1 / 3) == 3
    assert residual_groups_for(8, 1.0) == 8


def test_two_stage_assign_recovers_exact_sums():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(64, VL, generator=g)
    res = torch.randn(16, VL, generator=g) * 0.1
    i = torch.randint(0, 64, (500,), generator=g)
    j = torch.randint(0, 16, (500,), generator=g)
    x = base[i] + res[j]
    gi, gj = two_stage_assign(x, base, res)
    torch.testing.assert_close(base[gi] + res[gj], x)


def test_two_stage_assign_beats_greedy():
    g = torch.Generator().manual_seed(1)
    base = torch.randn(32, VL, generator=g)
    res = torch.randn(16, VL, generator=g) * 0.3
    x = torch.randn(2000, VL, generator=g)
    gi, gj = two_stage_assign(x, base, res)
    err = ((x - base[gi] - res[gj]) ** 2).sum(1)
    # greedy: nearest base, then its best residual; the beam contains it
    i0 = torch.cdist(x, base).argmin(1)
    greedy = (torch.cdist(x - base[i0], res).min(1).values) ** 2
    assert (err <= greedy + 1e-5).all()
    assert err.mean() < greedy.mean()
    assert err.mean() < (torch.cdist(x, base).min(1).values ** 2).mean()


def test_weighted_kmeans_follows_the_weights():
    x = torch.tensor([[0.0] * VL, [1.0] * VL])
    w = torch.tensor([3.0, 1.0])
    c = weighted_kmeans(x.repeat(8, 1), w.repeat(8), 1, n_iter=3)
    torch.testing.assert_close(c, torch.full((1, VL), 0.25))


def _residual_tensor(rows=6, cols=16, n_res=2, perm=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(32, VL, generator=g).to(torch.float16)
    rc = (torch.randn(16, VL, generator=g) * 0.1).to(torch.float16)
    groups = cols // VL
    idx = torch.randint(0, 32, (rows * groups,), generator=g)
    ridx = torch.randint(0, 16, (rows * n_res,), generator=g)
    p = torch.randperm(cols, generator=g) if perm else None
    vq = VQScheme(
        centroids=base,
        proxy_selector=ProxySelector.L2,
        original_shape=torch.Size([rows, cols]),
        column_permutation=p,
        residual_centroids=rc,
        residual_groups=n_res,
        residual_indices=IndexPacking.pack(ridx, 4),
    )
    q = QTensorTractVQ.from_packed(idx, vq, dequant_to_dtype=torch.float32)
    # expected weight, built by hand in the permuted order
    v = base.float()[idx].reshape(rows, groups, VL)
    v[:, :n_res] += rc.float()[ridx].reshape(rows, n_res, VL)
    w_perm = v.reshape(rows, cols)
    expected = torch.empty_like(w_perm)
    if p is None:
        expected = w_perm
    else:
        expected[:, p] = w_perm
    return q, expected


@pytest.mark.parametrize("perm", [False, True])
def test_dequantize_adds_residual_on_leading_groups(perm):
    q, expected = _residual_tensor(perm=perm)
    torch.testing.assert_close(q.decompress().float(), expected, rtol=0, atol=1e-3)


def test_residual_serde_round_trip(tmp_path):
    q, expected = _residual_tensor()
    path = tmp_path / "res.safetensors"
    save_qtensor(q, path)
    q2 = load_qtensor(path)
    assert q2.qscheme.residual_groups == 2
    assert torch.equal(q2.qscheme.residual_indices, q.qscheme.residual_indices)
    torch.testing.assert_close(q2.decompress().float(), expected, rtol=0, atol=1e-3)


def test_scheme_rejects_partial_residual_and_reassignment():
    base = torch.randn(32, VL).to(torch.float16)
    with pytest.raises(ValueError, match="go together"):
        VQScheme(
            centroids=base,
            proxy_selector=ProxySelector.L2,
            original_shape=torch.Size([4, 8]),
            residual_centroids=base[:16],
        )
    q, _ = _residual_tensor()
    with pytest.raises(Exception, match="from_packed"):
        q.qscheme.quantize_as_u8(torch.randn(6, 16))


def test_lowering_refuses_residual_weights():
    from neuropress.nn.np_vq_linear import NPVQLinear, VQProxy

    q, _ = _residual_tensor(rows=8, cols=16, perm=False)
    lin = nn.Linear(16, 8, bias=False)
    lin._parameters["weight"] = q
    with pytest.raises(ValueError, match="residual VQ"):
        NPVQLinear.from_qtensor_linear(lin, VQProxy(q.qscheme.centroids), "torch")


def test_estimator_validation():
    ok = {"gptq_correction_iters": 1, "residual_num_centroids": 16}
    VPTQGroupEstimator(**ok, residual_fraction=0.25)
    with pytest.raises(ValueError, match="power of two"):
        VPTQGroupEstimator(
            **{**ok, "residual_num_centroids": 24}, residual_fraction=0.5
        )
    with pytest.raises(ValueError, match="needs a residual_fraction"):
        VPTQGroupEstimator(**ok, residual_fraction=0.0)
    with pytest.raises(ValueError, match="one sequential GPTQ pass"):
        VPTQGroupEstimator(**ok, residual_fraction=0.5, gptq_block_ldlq=True)
    with pytest.raises(ValueError, match="needs residual_num_centroids"):
        VPTQGroupEstimator(gptq_correction_iters=1, residual_fraction=0.5)
    with pytest.raises(ValueError, match="power of two"):
        VPTQGroupEstimator(
            gptq_correction_iters=1,
            residual_num_centroids_per_group={"o_proj": 100},
            residual_fraction_per_group={"o_proj": 0.25},
        )
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        VPTQGroupEstimator(
            gptq_correction_iters=1,
            residual_num_centroids_per_group={"o_proj": 256},
            residual_fraction_per_group={"o_proj": 1.5},
        )


def test_residual_for_group_lookup():
    est = VPTQGroupEstimator(
        gptq_correction_iters=1,
        residual_num_centroids=16,
        residual_fraction=0.5,
        residual_num_centroids_per_group={"o_proj": 256},
        residual_fraction_per_group={
            "o_proj": 0.25,
            "gate_proj": 0.0,
            "model.layers.1.mlp.down_proj": 0.75,
        },
    )
    assert est.residual_for_group("o_proj") == (256, 0.25)
    assert est.residual_for_group("gate_proj") is None  # fraction 0 opts out
    assert est.residual_for_group("up_proj") == (16, 0.5)  # global default
    # own-codebook groups: full name first, then the leaf name
    assert est.residual_for_group("model.layers.1.mlp.down_proj") == (16, 0.75)
    assert est.residual_for_group("model.layers.2.self_attn.o_proj") == (256, 0.25)
    only = VPTQGroupEstimator(
        gptq_correction_iters=1,
        residual_num_centroids_per_group={"o_proj": 256},
        residual_fraction_per_group={"o_proj": 0.25},
    )
    assert only.residual_for_group("o_proj") == (256, 0.25)
    assert only.residual_for_group("up_proj") is None


# ---------------------------------------------------------------------------
# End to end (the sequential GPTQ pass needs CUDA)
# ---------------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(64, 96, bias=False)
        self.down_proj = nn.Linear(96, 64, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.up_proj(x)))


def _run(model, x, **extra):
    from neuropress.pipeline import QuantizationPipeline
    from neuropress.recipe_config import QuantizationRecipeConfig

    recipe = QuantizationRecipeConfig.model_validate(
        {
            "estimator": {
                "type": "vptq",
                "target": "linear",
                "vector_len": VL,
                "num_centroids": 64,
                "grouping": "by_leaf_name",
                "gptq_correction_iters": 1,
                "seed": 0,
                **extra,
            },
            "kernel_lowering": None,
        }
    )
    pipeline = QuantizationPipeline(recipe.build())
    pipeline.run(model, x, estimation_device="cuda")
    return pipeline.last_plan


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_residual_stage_lowers_output_error_end_to_end(tmp_path):
    torch.manual_seed(0)
    original = _Block().eval()
    x = [torch.randn(8, 32, 64) * torch.linspace(0.2, 3.0, 64) for _ in range(2)]
    plain, deep = deepcopy(original), deepcopy(original)
    _run(plain, x)
    plan = _run(deep, x, residual_num_centroids=16, residual_fraction=0.5)
    up = plan.weight_assignments["up_proj"]
    assert up.qscheme.residual_groups == 8  # half of 64 / 4
    assert up.qscheme.residual_centroids.shape == (16, VL)
    with torch.no_grad():  # quantized weights live on the estimation device
        target = torch.cat([original(b) for b in x]).cuda()
        err_plain = (torch.cat([plain(b.cuda()) for b in x]) - target).norm()
        err_deep = (torch.cat([deep(b.cuda()) for b in x]) - target).norm()
    assert err_deep < 0.9 * err_plain
    path = tmp_path / "up.safetensors"
    save_qtensor(up, path)
    torch.testing.assert_close(
        load_qtensor(path).decompress().float().cpu(), up.decompress().float().cpu()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_per_group_residual_only_where_configured():
    torch.manual_seed(0)
    model = _Block().eval()
    x = [torch.randn(8, 32, 64)]
    plan = _run(
        model,
        x,
        residual_num_centroids_per_group={"down_proj": 32},
        residual_fraction_per_group={"down_proj": 0.25},
    )
    assert not plan.weight_assignments["up_proj"].qscheme.has_residual
    down = plan.weight_assignments["down_proj"].qscheme
    assert down.residual_groups == 6  # 96 / 4 = 24 groups, a quarter
    assert down.residual_centroids.shape == (32, VL)
