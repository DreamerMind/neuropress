"""GSQ: Gumbel-Softmax code refinement and its block refiner."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from neuropress.block_pipeline import BlockQuantizationPipeline
from neuropress.gsq_refiner import GSQGroup, GSQRefiner
from neuropress.methods.gsq import (
    GSQConfig,
    GumbelCodes,
    Lion,
    candidate_mode,
    evaluate_codes,
    fit_gumbel_codes,
)
from neuropress.pipeline import QuantizationPipeline
from neuropress.recipe_config import QuantizationRecipeConfig


def test_lion_step_matches_the_update_rule():
    p = nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    opt = Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.5)
    p.grad = torch.tensor([0.3, -0.1, 0.0])
    opt.step()
    expected = torch.tensor([1.0, -2.0, 0.5]) * (1 - 0.05) - 0.1 * torch.tensor(
        [1.0, -1.0, 0.0]
    )
    torch.testing.assert_close(p.detach(), expected)
    torch.testing.assert_close(
        opt.state[p]["exp_avg"], 0.01 * torch.tensor([0.3, -0.1, 0.0])
    )


def _rtn(weight, nbits, group):
    rows, cols = weight.shape
    g = weight.reshape(rows, cols // group, group)
    qmax = 2 ** (nbits - 1)
    scale = g.abs().amax(-1) / qmax
    levels = (g / scale[..., None]).round().clamp(-qmax, qmax - 1)
    return levels.reshape(rows, cols).to(torch.int32), scale


@pytest.mark.parametrize("nbits", [2, 3])
def test_strong_prior_starts_at_the_given_levels(nbits):
    torch.manual_seed(0)
    levels, scale = _rtn(torch.randn(8, 64), nbits, 32)
    config = GSQConfig(init_strength=100.0)
    lo, hi = -(2 ** (nbits - 1)), 2 ** (nbits - 1) - 1
    codes = GumbelCodes(
        levels, scale, lo, hi, candidate_mode(hi - lo + 1, config), config
    )
    assert codes.mode == ("full" if nbits == 2 else "shift")
    assert codes.logits.shape[0] == (4 if nbits == 2 else 5)
    assert torch.equal(codes.hard_levels(), levels)
    torch.testing.assert_close(codes.hard(), codes.initial())


def test_shifts_never_leave_the_grid():
    levels = torch.tensor([[-4, 3, 0, 2]] * 2)
    scale = torch.ones(2, 1)
    codes = GumbelCodes(levels, scale, -4, 3, "shift", GSQConfig())
    with torch.no_grad():
        codes.logits.zero_()
        codes.logits[0] = 10.0  # shift -2: out of range for -4
        codes.logits[4, :, 1] = 20.0  # shift +2: out of range for 3
    hard = codes.hard_levels()
    assert hard.min() >= -4 and hard.max() <= 3
    assert hard[0, 0] == -4 and hard[0, 2] == -2 and hard[0, 3] == 0
    assert hard[0, 1] == 1  # +2 masked; the best valid shift is -2
    soft = codes.soft(0.05, 500.0)
    assert soft.min() >= -4 - 1e-4 and soft.max() <= 3 + 1e-4


def test_soft_weight_is_differentiable_in_logits_and_scales():
    torch.manual_seed(1)
    levels, scale = _rtn(torch.randn(4, 32), 2, 16)
    codes = GumbelCodes(levels, scale, -2, 1, "full", GSQConfig())
    codes.soft(2.0, 100.0).square().sum().backward()
    assert codes.logits.grad.abs().sum() > 0
    assert codes.scale.grad.abs().sum() > 0


@pytest.mark.parametrize("nbits", [2, 3])
def test_fit_lowers_the_output_error_of_rtn_codes(nbits):
    torch.manual_seed(2)
    weight = torch.randn(16, 64)
    # anisotropic inputs, so that RTN is far from the output-optimal codes
    x = [torch.randn(32, 64) * torch.linspace(0.1, 3.0, 64) for _ in range(8)]
    levels, scale = _rtn(weight, nbits, 32)
    lo, hi = -(2 ** (nbits - 1)), 2 ** (nbits - 1) - 1
    config = GSQConfig(epochs=40, batch_size=2, lr_logits=2e-3, lr_scales=1e-3)
    codes = {
        "w": GumbelCodes(
            levels, scale, lo, hi, candidate_mode(hi - lo + 1, config), config
        )
    }

    def loss(weights, i):
        return (x[i] @ weights["w"].t() - x[i] @ weight.t()).square().mean()

    start = evaluate_codes({"w": codes["w"].initial()}, loss, len(x))
    history = fit_gumbel_codes(
        codes, loss, len(x), config, generator=torch.Generator().manual_seed(0)
    )
    fitted = evaluate_codes({"w": codes["w"].hard()}, loss, len(x))
    assert len(history) == config.epochs
    # exact coordinate descent over the levels (fixed scales) reaches 0.84 / 0.80 here
    assert fitted < 0.9 * start


class _Block(nn.Module):
    """Attention-like and MLP-like pairs with a residual, called with keywords."""

    def __init__(self, width=64, hidden=128):
        super().__init__()
        self.attn = nn.ModuleDict(
            {
                "v": nn.Linear(width, width, bias=False),
                "o": nn.Linear(width, width, bias=False),
            }
        )
        self.mix = _Pair(self.attn)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, hidden_states, gain):
        h = hidden_states + self.mix(hidden_states) * gain
        return h + self.down(torch.relu(self.up(h)))


class _Pair(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = [attn]  # list: one owner in the module tree

    def forward(self, x):
        return self.attn[0]["o"](torch.tanh(self.attn[0]["v"](x)))


def _recipe(nbits=2, transform=None):
    return QuantizationRecipeConfig.model_validate(
        {
            "estimator": {
                "type": "gptq",
                "group_size": 32,
                "nbits": nbits,
                "symmetric": True,
                "weight_transform": transform,
            },
            "kernel_lowering": None,
        }
    )


STAGES = (("attn.v", "attn.o"), ("up", "down"))
GROUPS = (GSQGroup(("attn.v", "attn.o"), objective="mix"), GSQGroup(("up", "down")))


def _batches(n=6):
    torch.manual_seed(3)
    return [
        {
            "hidden_states": torch.randn(1, 24, 64) * torch.linspace(0.2, 2.0, 64),
            "gain": 0.5,
        }
        for _ in range(n)
    ]


def _block_error(block, reference, batches):
    with torch.no_grad():
        return sum(
            (block(**b) - reference(**b)).square().mean().item() for b in batches
        )


@pytest.mark.parametrize("transform", [None, "q2_0_packed"])
def test_refiner_lowers_block_error_and_keeps_the_format(transform):
    torch.manual_seed(4)
    float_block = _Block().eval()
    batches = _batches()
    results = {}
    for use in (False, True):
        refiner = (
            GSQRefiner(
                GROUPS,
                GSQConfig(epochs=15, batch_size=2, lr_logits=1e-3, lr_scales=5e-4),
            )
            if use
            else None
        )
        runner = BlockQuantizationPipeline(
            _recipe(transform=transform), stages=STAGES, refiner=refiner
        )
        block = deepcopy(float_block)
        (result,) = list(runner.run([("b", block)], batches))
        assert not result.kept_float
        results[use] = (_block_error(result.model, float_block, batches), refiner)
    gptq, (gsq, refiner) = results[False][0], results[True]
    assert gsq < gptq
    assert [r["targets"] for r in refiner.history] == [g.targets for g in GROUPS]
    for record in refiner.history:
        assert record["kept"] and record["fitted_loss"] < record["start_loss"]


def test_refined_plan_is_what_the_adapter_applies():
    block = _Block().eval()
    batches = _batches(4)
    seen = {}

    class Recording(GSQRefiner):
        def refine(self, block, plan, batches, teacher, *, device, name=""):
            super().refine(block, plan, batches, teacher, device=device, name=name)
            for path, qw in plan.weight_assignments.items():
                seen[path] = qw.scheme.dequantize(qw.codes, torch.float32).reshape(
                    block.get_submodule(path).weight.shape
                )

    refiner = Recording(GROUPS, GSQConfig(epochs=2, batch_size=2))
    runner = BlockQuantizationPipeline(_recipe(), stages=STAGES, refiner=refiner)
    (result,) = list(runner.run([("b", block)], batches))
    assert set(seen) == {p for s in STAGES for p in s}
    for path, dense in seen.items():
        torch.testing.assert_close(
            result.model.get_submodule(path).weight.float(), dense
        )


def test_planned_weights_outside_groups_are_used_quantized():
    block = _Block().eval()
    batches = _batches(2)
    refiner = GSQRefiner(
        (GSQGroup(("attn.o",), objective="mix"),), GSQConfig(epochs=1, batch_size=1)
    )
    runner = BlockQuantizationPipeline(_recipe(), stages=STAGES, refiner=refiner)
    calls = []
    original = refiner._fixed_weights

    def spy(block, plan, exclude, device):
        fixed = original(block, plan, exclude, device)
        calls.append(set(fixed))
        return fixed

    refiner._fixed_weights = spy
    list(runner.run([("b", block)], batches))
    assert calls == [{"attn.v.weight"}]


def test_groups_and_stages_are_validated():
    with pytest.raises(ValueError, match="distinct"):
        GSQRefiner((GSQGroup(("a",)), GSQGroup(("a", "b"))))
    with pytest.raises(ValueError, match="either"):
        BlockQuantizationPipeline(_recipe(), target_paths=("a",), stages=(("a",),))
    with pytest.raises(ValueError, match="sequential"):
        BlockQuantizationPipeline(
            _recipe(), input_source="float", refiner=GSQRefiner(GROUPS)
        )
    refiner = GSQRefiner((GSQGroup(("attn.o", "up")),), GSQConfig(epochs=1))
    runner = BlockQuantizationPipeline(_recipe(), stages=STAGES, refiner=refiner)
    with pytest.raises(ValueError, match="spans pipeline stages"):
        list(runner.run([("b", _Block().eval())], _batches(2)))


def test_refine_plan_runs_between_estimation_and_adaptation():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(64, 32, bias=False)).eval()
    weight = model[0].weight.detach().clone()
    seen = []

    def refine(plan):
        seen.append(set(plan.weight_assignments))
        assert torch.equal(model[0].weight, weight)  # not adapted yet
        qw = plan.weight_assignments["0"]
        plan.weight_assignments["0"] = type(qw)(
            codes=torch.full_like(qw.codes, int(qw.scheme.zero_point.flatten()[0]) + 1),
            scheme=qw.scheme,
            dtype=qw.dtype,
            destination=qw.destination,
        )

    QuantizationPipeline(_recipe().build()).run(
        model, [torch.randn(1, 8, 64)], refine_plan=refine
    )
    assert seen == [{"0"}]
    scale = model[0].weight.detach().reshape(32, 2, 32)
    assert (scale > 0).all()  # every level is +1: the weight is the scale itself


@pytest.mark.parametrize("shift", [False, True])
def test_relaxation_backward_matches_autograd(shift):
    from neuropress.methods.gsq import _Relax, _relaxed_level

    torch.manual_seed(6)
    init = torch.randint(-4, 4, (6, 8)).to(torch.int8)
    candidates = torch.arange(-2.0, 3.0) if shift else torch.arange(-4.0, 4.0)
    logits = torch.randn(len(candidates), 6, 8, requires_grad=True)
    uniform = torch.rand(logits.shape)
    scalars = torch.tensor([1 / 0.7, 3.0])
    grad = torch.randn(6, 8)
    args = (uniform, init, candidates, scalars, -4, 3, shift)
    expected = torch.autograd.grad(
        (_relaxed_level(logits, *args) * grad).sum(), logits
    )[0]
    level = _Relax.apply(logits, *args)
    torch.testing.assert_close(level, _relaxed_level(logits, *args))
    (actual,) = torch.autograd.grad((level * grad).sum(), logits)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_group_paths_missing_from_the_block_are_rejected():
    refiner = GSQRefiner((GSQGroup(("attn.v", "attn.typo")),), GSQConfig(epochs=1))
    runner = BlockQuantizationPipeline(_recipe(), stages=STAGES, refiner=refiner)
    with pytest.raises(ValueError, match="not in the block"):
        list(runner.run([("b", _Block().eval())], _batches(2)))


def test_targets_kept_float_leave_the_rest_of_the_group_fitted():
    class DropDown(GSQRefiner):
        def refine(self, block, plan, batches, teacher, *, device, name=""):
            if "down" in plan.weight_assignments:
                del plan.weight_assignments["down"]
                plan.kept_float["down"] = "test"
            super().refine(block, plan, batches, teacher, device=device, name=name)

    refiner = DropDown(GROUPS, GSQConfig(epochs=1, batch_size=2))
    runner = BlockQuantizationPipeline(_recipe(), stages=STAGES, refiner=refiner)
    list(runner.run([("b", _Block().eval())], _batches(2)))
    assert [r["targets"] for r in refiner.history] == [("attn.v", "attn.o"), ("up",)]


def test_groups_of_different_stages_draw_different_seeds():
    refiner = GSQRefiner(GROUPS, GSQConfig(epochs=1, batch_size=2))
    runner = BlockQuantizationPipeline(_recipe(), stages=STAGES, refiner=refiner)
    list(runner.run([("b", _Block().eval())], _batches(2)))
    assert len({r["seed"] for r in refiner.history}) == 2


def test_scales_keep_their_sign():
    levels, scale = _rtn(torch.randn(4, 64), 3, 32)
    codes = GumbelCodes(levels, scale, -4, 3, "shift", GSQConfig())
    with torch.no_grad():
        codes.scale.mul_(-1.0)
    codes.clamp_scale()
    torch.testing.assert_close(codes.scale.detach(), scale * 1e-3)


def test_fit_runs_inside_no_grad():
    torch.manual_seed(1)
    weight = torch.randn(8, 64)
    levels, scale = _rtn(weight, 3, 32)
    codes = {"w": GumbelCodes(levels, scale, -4, 3, "shift", GSQConfig())}
    x = torch.randn(16, 64)

    def loss(weights, i):
        return (x @ weights["w"].t() - x @ weight.t()).square().mean()

    with torch.no_grad():
        history = fit_gumbel_codes(codes, loss, 2, GSQConfig(epochs=2, batch_size=1))
    assert len(history) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled kernels need CUDA")
def test_many_weight_shapes_compile_without_hitting_the_recompile_limit():
    torch.manual_seed(2)
    config = GSQConfig(epochs=1, batch_size=1)
    for rows in range(4, 4 + 12):
        weight = torch.randn(rows, 64, device="cuda")
        levels, scale = _rtn(weight, 3, 32)
        codes = {"w": GumbelCodes(levels, scale, -4, 3, "shift", config)}

        def loss(weights, i, weight=weight):
            return (weights["w"] - weight).square().mean()

        fit_gumbel_codes(codes, loss, 1, config)


# "up" reaches up but not down: the case that used to pass silently.
@pytest.mark.parametrize("objective", ["mix", "up"])
def test_an_objective_upstream_of_its_targets_is_refused(objective):
    refiner = GSQRefiner(
        (GSQGroup(("up", "down"), objective=objective),), GSQConfig(epochs=1)
    )
    runner = BlockQuantizationPipeline(_recipe(), stages=STAGES, refiner=refiner)
    with pytest.raises(ValueError, match="does not depend on"):
        list(runner.run([("b", _Block().eval())], _batches(2)))


def test_a_tie_keeps_the_gptq_codes(monkeypatch):
    import neuropress.gsq_refiner as gsq_refiner

    monkeypatch.setattr(gsq_refiner, "evaluate_codes", lambda *a, **k: 1.0)
    group = (GSQGroup(("up", "down")),)
    plans = {}

    class Spy(GSQRefiner):
        def refine(self, block, plan, batches, teacher, *, device, name=""):
            before = {
                p: plan.weight_assignments[p].codes.clone()
                for p in plan.weight_assignments
            }
            super().refine(block, plan, batches, teacher, device=device, name=name)
            plans.update(
                {p: (before[p], plan.weight_assignments[p].codes) for p in before}
            )

    refiner = Spy(group, GSQConfig(epochs=1, batch_size=2))
    runner = BlockQuantizationPipeline(_recipe(), stages=STAGES, refiner=refiner)
    list(runner.run([("b", _Block().eval())], _batches(2)))
    assert refiner.history[0]["kept"] is False
    for before, after in plans.values():
        assert torch.equal(before, after)


def test_weights_that_are_not_2d_are_refused():
    class ConvBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(64, 64, 3, padding=1, bias=False)

        def forward(self, hidden_states, gain):
            x = hidden_states.transpose(1, 2)[..., None]
            return self.conv(x)[..., 0].transpose(1, 2) * gain

    refiner = GSQRefiner((GSQGroup(("conv",)),), GSQConfig(epochs=1))
    recipe = _recipe()
    recipe = recipe.model_copy(
        update={"estimator": recipe.estimator.model_copy(update={"target": "conv2d"})}
    )
    runner = BlockQuantizationPipeline(recipe, stages=(("conv",),), refiner=refiner)
    with pytest.raises(ValueError, match="2-D weights"):
        list(runner.run([("b", ConvBlock().eval())], _batches(2)))


def test_the_teacher_leaves_the_device_before_the_fit(monkeypatch):
    import neuropress.gsq_refiner as gsq_refiner

    events = []
    fit = gsq_refiner.fit_gumbel_codes

    def spy_fit(*args, **kwargs):
        events.append("fit")
        return fit(*args, **kwargs)

    monkeypatch.setattr(gsq_refiner, "fit_gumbel_codes", spy_fit)

    class Spy(GSQRefiner):
        def refine(self, block, plan, batches, teacher, *, device, name=""):
            def placed():
                module = teacher()
                cpu = module.cpu

                def record():
                    events.append("teacher to host")
                    return cpu()

                module.cpu = record
                return module

            super().refine(block, plan, batches, placed, device=device, name=name)

    refiner = Spy((GSQGroup(("up", "down")),), GSQConfig(epochs=1, batch_size=2))
    runner = BlockQuantizationPipeline(_recipe(), stages=STAGES, refiner=refiner)
    list(runner.run([("b", _Block().eval())], _batches(2)))
    assert events[:2] == ["teacher to host", "fit"]


def test_an_unreached_target_is_refused_before_any_step(monkeypatch):
    import neuropress.methods.gsq as gsq

    steps = []
    monkeypatch.setattr(gsq.Lion, "step", lambda self, *a, **k: steps.append(1))
    refiner = GSQRefiner(
        (GSQGroup(("up", "down"), objective="up"),), GSQConfig(epochs=3)
    )
    runner = BlockQuantizationPipeline(_recipe(), stages=STAGES, refiner=refiner)
    with pytest.raises(ValueError, match=r"does not depend on \['down'\]"):
        list(runner.run([("b", _Block().eval())], _batches(4)))
    assert steps == []


def test_a_flat_tuple_of_stage_paths_is_refused():
    with pytest.raises(ValueError, match="tuple of stages"):
        BlockQuantizationPipeline(
            _recipe(), input_source="sequential", stages=("attn.v", "attn.o")
        )


def test_the_cache_device_is_not_part_of_the_fingerprint():
    assert repr(GSQRefiner(GROUPS, cache_device="cpu")) == repr(GSQRefiner(GROUPS))


def test_a_finite_fit_beats_a_non_finite_start(monkeypatch):
    import neuropress.gsq_refiner as gsq_refiner

    losses = iter([float("nan"), 1.0])
    monkeypatch.setattr(gsq_refiner, "evaluate_codes", lambda *a, **k: next(losses))
    refiner = GSQRefiner(
        (GSQGroup(("attn.v", "attn.o"), objective="mix"),),
        GSQConfig(epochs=1, batch_size=2),
    )
    runner = BlockQuantizationPipeline(_recipe(), stages=(STAGES[0],), refiner=refiner)
    list(runner.run([("b", _Block().eval())], _batches(2)))
    assert refiner.history[0]["kept"] is True


def test_targets_in_no_explicit_stage_are_reported_float():
    runner = BlockQuantizationPipeline(
        _recipe(), input_source="sequential", stages=(STAGES[0],)
    )
    (result,) = runner.run([("b", _Block().eval())], _batches(2))
    assert result.kept_float == {
        "up": "in no explicit stage",
        "down": "in no explicit stage",
    }


def test_a_group_no_stage_plans_is_refused():
    refiner = GSQRefiner(GROUPS, GSQConfig(epochs=1, batch_size=2))
    runner = BlockQuantizationPipeline(_recipe(), stages=(STAGES[0],), refiner=refiner)
    with pytest.raises(
        ValueError, match=r"no stage planned the groups \[\('up', 'down'\)\]"
    ):
        list(runner.run([("b", _Block().eval())], _batches(2)))
