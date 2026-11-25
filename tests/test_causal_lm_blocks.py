import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
qwen3 = pytest.importorskip("transformers.models.qwen3")
qwen3_moe = pytest.importorskip("transformers.models.qwen3_moe")

from neuropress.causal_lm_blocks import _capture, _Head  # noqa: E402
from neuropress.recipe import CalibrationConfig  # noqa: E402
from neuropress.t2n.registry import (  # noqa: E402
    NEUROPRESS_COMPRESSION,
    _same_recipe,
    run_quantization_pipeline,
)

from .test_t2n_register import _mock_load_dataset, _MockTokenizer  # noqa: E402


def _moe_model() -> nn.Module:
    cfg = qwen3_moe.Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
    )
    torch.manual_seed(0)
    return qwen3_moe.Qwen3MoeForCausalLM(cfg).eval()


def _hybrid_model() -> nn.Module:
    """Sliding and full attention layers: each gets its own eager mask."""
    cfg = qwen3.Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
        use_sliding_window=True,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return qwen3.Qwen3ForCausalLM(cfg).eval()


def _projected_moe_model() -> nn.Module:
    """An MoE model with a linear target outside its decoder layers and head."""
    model = _moe_model()
    torch.manual_seed(1)
    model.model.embed_tokens = nn.Sequential(
        model.model.embed_tokens, nn.Linear(64, 64, bias=False)
    )
    return model


def _config(entry: str):
    return NEUROPRESS_COMPRESSION[entry].__config__.model_copy(
        update={
            "calibration": CalibrationConfig(n_samples=4, seq_len=32, seed=0),
            "kernel_lowering": None,
        }
    )


def _quantize(model, entry: str, block_calibration, cache_dir=None, recipe=None):
    config = _config(entry)
    run_quantization_pipeline(
        model,
        tokenizer=_MockTokenizer(),
        recipe=recipe if recipe is not None else config.build(),
        recipe_config_json=config.model_dump_json(),
        block_calibration=block_calibration,
        cache_dir=cache_dir,
        _load_dataset=_mock_load_dataset,
    )
    return model


def _dense_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, mod in model.named_modules():
        for pname, p in mod.named_parameters(recurse=False):
            decompress = getattr(p, "decompress", None)
            state[f"{name}.{pname}"] = (decompress() if decompress else p).detach()
    return state


@pytest.mark.parametrize(
    ("make_model", "entry"),
    [
        (_moe_model, "open_text_c4_gptq_q4_0_moe"),
        (_hybrid_model, "open_text_c4_gptq_q4_0"),
        (_projected_moe_model, "open_text_c4_gptq_q4_0_moe"),
    ],
    ids=["moe", "hybrid_attention", "target_outside_the_blocks"],
)
def test_float_block_calibration_matches_a_whole_model_run(make_model, entry):
    whole = _dense_state(_quantize(make_model(), entry, None))
    blocks = _dense_state(_quantize(make_model(), entry, "float"))
    assert whole.keys() == blocks.keys()
    for key in whole:
        torch.testing.assert_close(blocks[key], whole[key], msg=key)


def test_float_block_calibration_keeps_the_moe_export_markers():
    from neuropress.moe_experts import expert_q40_from_gptq_codes

    model = _quantize(_moe_model(), "open_text_c4_gptq_q4_0_moe", "float")
    for layer in model.model.layers:
        moe = layer.mlp
        assert moe._t2n_quantize_moe_experts_q40 is True
        w2 = moe.experts.down_proj.detach().contiguous()
        q = expert_q40_from_gptq_codes(w2, **moe._t2n_quantize_moe_experts_q40_kwargs)
        torch.testing.assert_close(q.decompress(), w2, rtol=0, atol=0)


def test_sequential_block_calibration_quantizes_every_target():
    from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

    model = _quantize(
        _projected_moe_model(), "open_text_c4_gptq_q4_0_moe", "sequential"
    )
    assert isinstance(model.lm_head.weight, QTensorTractScaleOnly)
    assert isinstance(model.model.embed_tokens[1].weight, QTensorTractScaleOnly)
    assert model.__neuropress_manifest__.block_calibration == "sequential"
    for layer in model.model.layers:
        assert isinstance(layer.self_attn.q_proj.weight, QTensorTractScaleOnly)
        assert layer.mlp._t2n_quantize_moe_experts_q40 is True


def test_capture_keeps_each_layers_own_mask_and_shares_tensors():
    model = _hybrid_model()
    ids = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(0))
    seen = []
    hooks = [
        layer.register_forward_pre_hook(
            lambda m, a, k: seen.append(k["attention_mask"]), with_kwargs=True
        )
        for layer in model.model.layers
    ]
    model(input_ids=ids, use_cache=False)
    for hook in hooks:
        hook.remove()

    hidden, kwargs = _capture(model, model.model.layers, [{"input_ids": ids}])
    assert len(hidden) == 1
    masks = [kwargs[i][0]["attention_mask"] for i in range(2)]
    assert not torch.equal(masks[0], masks[1])
    for mask, expected in zip(masks, seen, strict=True):
        torch.testing.assert_close(mask, expected)
    # Rotary embeddings are one tensor pair shared by both layers.
    assert (
        kwargs[0][0]["position_embeddings"][0] is kwargs[1][0]["position_embeddings"][0]
    )


def test_head_block_hands_back_no_logits():
    model = _moe_model()
    head = _Head(model, "model.norm", "lm_head", None)
    assert head(torch.randn(2, 8, 64)).numel() == 0


def test_a_recipe_the_json_does_not_describe_is_refused():
    other = _config("open_text_c4_gptq_q4_0").build()
    with pytest.raises(ValueError, match="does not describe the recipe passed"):
        _quantize(_moe_model(), "open_text_c4_gptq_q4_0_moe", "float", recipe=other)


def test_a_recipe_stricter_than_its_json_is_refused():
    recipe = _config("open_text_c4_gptq_q4_0_moe").build()
    strict = recipe.model_copy(update={"require_all_quantized": True})
    with pytest.raises(ValueError, match="does not describe the recipe passed"):
        _quantize(_moe_model(), "open_text_c4_gptq_q4_0_moe", "float", recipe=strict)


def test_a_recipe_used_before_still_matches_its_json():
    config = _config("open_text_c4_gptq_q4_0_moe")
    recipe = config.build()
    recipe.param_estimator.prepare(_moe_model())
    assert _same_recipe(recipe, config.build())
    changed = config.model_copy(
        update={"calibration": CalibrationConfig(n_samples=8, seq_len=32, seed=0)}
    )
    assert not _same_recipe(recipe, changed.build())


def test_moe_entry_calibrates_blocks_sequentially():
    entry = NEUROPRESS_COMPRESSION["open_text_c4_gptq_q4_0_moe"]
    assert entry.keywords["block_calibration"] == "sequential"


def test_cache_dir_resumes_every_block(tmp_path, caplog):
    entry = "open_text_c4_gptq_q4_0_moe"
    first = _quantize(_moe_model(), entry, "sequential", cache_dir=tmp_path)
    with caplog.at_level("INFO", logger="neuropress.causal_lm_blocks"):
        second = _quantize(_moe_model(), entry, "sequential", cache_dir=tmp_path)
    assert caplog.text.count("resumed") == 2
    assert "done" not in caplog.text
    expected, resumed = _dense_state(first), _dense_state(second)
    assert expected.keys() == resumed.keys()
    for key in expected:
        torch.testing.assert_close(resumed[key], expected[key], rtol=0, atol=0)
    for layer in second.model.layers:
        assert layer.mlp._t2n_quantize_moe_experts_q40 is True


def test_a_checkpoint_keeps_the_block_calibration_mode(tmp_path):
    from neuropress.t2n.model_io import load_quantized_model, save_quantized_model

    entry = "open_text_c4_gptq_q4_0_moe"
    model = _quantize(_moe_model(), entry, "sequential")
    path = tmp_path / "model.safetensors"
    save_quantized_model(model, path, _config(entry).model_dump_json())
    assert model.__neuropress_manifest__.block_calibration == "sequential"
    manifest = load_quantized_model(_moe_model(), path, kernel_lowering=None)
    assert manifest.block_calibration == "sequential"


def test_a_whole_model_run_records_no_block_calibration():
    model = _quantize(_moe_model(), "open_text_c4_gptq_q4_0_moe", None)
    assert model.__neuropress_manifest__.block_calibration is None


def test_a_tied_embedding_is_quantized_once_in_the_head_block(monkeypatch):
    import neuropress.causal_lm_blocks as blocks
    from neuropress.estimator_configs import MinMaxZpWeightEstimatorConfig
    from neuropress.recipe_config import (
        QuantizationRecipeConfig,
        WeightAssignmentAdapterConfig,
    )

    model = _moe_model()
    model.lm_head.weight = model.model.embed_tokens.weight
    float_weight = model.lm_head.weight.detach().clone()
    config = QuantizationRecipeConfig(
        estimator=MinMaxZpWeightEstimatorConfig(target="embedding"),
        adapter=WeightAssignmentAdapterConfig(),
    )
    outside_passes = []
    monkeypatch.setattr(
        blocks, "_quantize_outside", lambda *a: outside_passes.append(a[-1]) or {}
    )
    ids = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(0))
    blocks.calibrate_causal_lm_blockwise(model, [{"input_ids": ids}], config)
    assert outside_passes == []
    weight = model.model.embed_tokens.weight
    assert not torch.equal(weight, float_weight)


def test_a_model_spread_over_devices_is_refused():
    from neuropress.causal_lm_blocks import calibrate_causal_lm_blockwise

    model = _moe_model()
    model.model.layers[1].to("meta")
    model.hf_device_map = {"model.layers.0": "cpu", "model.layers.1": "disk"}
    config = NEUROPRESS_COMPRESSION["open_text_c4_gptq_q4_0_moe"].__config__
    with pytest.raises(ValueError, match="spread over"):
        calibrate_causal_lm_blockwise(model, [], config)


def test_a_model_gathered_on_one_device_is_accepted_despite_its_old_map():
    from neuropress.causal_lm_blocks import _check_one_device

    model = _moe_model()
    model.hf_device_map = {"model.layers.0": 0, "model.layers.1": "cpu"}
    _check_one_device(model)


class _Bounds:
    def __init__(self, limit: int) -> None:
        self.limit = limit

    def keep(self, name: str, module: nn.Module) -> bool:
        return module.weight.shape[0] <= self.limit


def test_recipe_structure_reads_method_instances_partials_and_closures():
    from functools import partial

    from neuropress.t2n.registry import _structure

    assert _structure(_Bounds(1).keep) != _structure(_Bounds(2).keep)
    assert _structure(_Bounds(1).keep) == _structure(_Bounds(1).keep)

    def make():
        def inner(x):
            return x

        return partial(inner, 1)

    assert _structure(make()) == _structure(make())

    def recursive(n):
        return recursive(n - 1) if n else 0

    _structure(recursive)  # refers to itself through its closure

    def empty_cell():
        def uses():
            return later

        return uses
        later = 1

    _structure(empty_cell())


@pytest.mark.parametrize("resumed", [False, True], ids=["fresh", "resumed"])
def test_mixed_precision_splits_reach_the_checkpoint(tmp_path, resumed):
    from neuropress.estimator_configs import GptqPanelEstimatorConfig
    from neuropress.recipe_config import QuantizationRecipeConfig
    from neuropress.t2n.model_io import load_quantized_model, save_quantized_model

    config = QuantizationRecipeConfig(
        calibration=CalibrationConfig(n_samples=4, seq_len=32, seed=0),
        estimator=GptqPanelEstimatorConfig(
            target="linear", possible_nbits=[4, 8], target_bpw=6.0
        ),
    )

    def run():
        model = _hybrid_model()
        run_quantization_pipeline(
            model,
            tokenizer=_MockTokenizer(),
            recipe=config.build(),
            recipe_config_json=config.model_dump_json(),
            block_calibration="float",
            cache_dir=tmp_path / "cache",
            _load_dataset=_mock_load_dataset,
        )
        return model

    model = run()
    if resumed:
        model = run()
    split = {
        path
        for t in model.__neuropress_manifest__.transforms
        if getattr(t, "type", None) == "mixed_precision_split"
        for path in t.modules
    }
    assert {"model.layers.0.self_attn.q_proj", "lm_head"} <= split
    assert type(model.lm_head).__name__ == "MixedPrecisionLinear"
    for layer in model.model.layers:
        assert layer.__dict__.get("__neuropress_manifest__") is None
    path = tmp_path / "model.safetensors"
    save_quantized_model(model, path, config.model_dump_json())
    loaded = _hybrid_model()
    load_quantized_model(loaded, path, kernel_lowering=None)
    ids = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, model(ids).logits)


def _linear_and_embedding_config():
    from neuropress.estimator_configs import (
        GptqEstimatorConfig,
        MinMaxZpWeightEstimatorConfig,
    )
    from neuropress.recipe_config import (
        CompositeEstimatorConfig,
        QuantizationRecipeConfig,
    )

    return QuantizationRecipeConfig(
        calibration=CalibrationConfig(n_samples=4, seq_len=32, seed=0),
        estimator=CompositeEstimatorConfig(
            methods=[
                GptqEstimatorConfig(
                    target="linear",
                    weight_transform="q4_0_packed",
                    group_size=32,
                    nbits=4,
                ),
                MinMaxZpWeightEstimatorConfig(target="embedding"),
            ]
        ),
        kernel_lowering=None,
    )


def test_estimators_without_statistics_declare_their_targets():
    config = _linear_and_embedding_config()
    composite = config.build().param_estimator
    model = _moe_model()
    composite.prepare(model)
    minmax = composite.estimators[1]
    assert minmax.declare_statistics() == []
    assert minmax.declare_targets() == ["model.embed_tokens"]
    targets = composite.declare_targets()
    assert targets[-1] == "model.embed_tokens"
    assert "lm_head" in targets and "model.layers.0.self_attn.q_proj" in targets

    moe = NEUROPRESS_COMPRESSION["open_text_c4_gptq_q4_0_moe"].__config__.build()
    moe.param_estimator.prepare(model)
    assert "model.layers.0.mlp.experts" in moe.param_estimator.declare_targets()


def test_an_untied_embedding_without_statistics_is_quantized_in_block_mode():
    config = _linear_and_embedding_config()

    def run(block_calibration):
        model = _moe_model()
        run_quantization_pipeline(
            model,
            tokenizer=_MockTokenizer(),
            recipe=config.build(),
            recipe_config_json=config.model_dump_json(),
            block_calibration=block_calibration,
            _load_dataset=_mock_load_dataset,
        )
        return model

    float_embedding = _moe_model().model.embed_tokens.weight.detach()
    whole, blocks = _dense_state(run(None)), _dense_state(run("float"))
    assert not torch.equal(blocks["model.embed_tokens.weight"], float_embedding)
    assert whole.keys() == blocks.keys()
    for key in whole:
        torch.testing.assert_close(blocks[key], whole[key], msg=key)


def test_an_estimator_that_cannot_declare_targets_is_refused():
    from types import SimpleNamespace

    from neuropress.causal_lm_blocks import _targets_outside
    from neuropress.estimator import QuantizationParamEstimator

    class Opaque(QuantizationParamEstimator):
        def declare_statistics(self):
            return []

        def estimate(self, graph, registry, recipe):
            raise NotImplementedError

    config = SimpleNamespace(build=lambda: SimpleNamespace(param_estimator=Opaque()))
    with pytest.raises(ValueError, match="must know every target"):
        _targets_outside(_moe_model(), config, [])


class _Slotted:
    __slots__ = ("_cache", "limit")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._cache = object()


def test_recipe_structure_reads_sets_slots_and_nan():
    from neuropress.t2n.registry import _structure

    def fresh():
        return {"names": frozenset({"a", "b"}), "nan": float("nan")}

    assert _structure(fresh()) == _structure(fresh())
    assert _structure({1, 2}) != _structure({1, 3})
    assert _structure(_Slotted(1)) == _structure(_Slotted(1))
    assert _structure(_Slotted(1)) != _structure(_Slotted(2))
    assert _structure(torch.tensor([float("nan")])) == _structure(
        torch.tensor([float("nan")])
    )


@pytest.mark.parametrize(
    ("entry", "mode"),
    [("open_text_c4_gptq_q4_0_moe", "sequential"), ("open_text_c4_gptq_q4_0", None)],
)
def test_eval_run_config_records_the_block_calibration_mode(tmp_path, entry, mode):
    import argparse
    import json

    from neuropress.eval.llm_cli import _write_run_config

    args = argparse.Namespace(
        model="m", compression=entry, tasks=[], output=str(tmp_path / "out.json")
    )
    _write_run_config(args)
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["quantization_config"]["block_calibration"] == mode


@pytest.mark.parametrize(
    ("block_calibration", "export"),
    [("sequential", False), ("float", False), (None, False), ("sequential", True)],
    ids=["sequential", "float", "whole_model", "export"],
)
def test_kernel_lowering_runs_once_after_calibration(
    monkeypatch, block_calibration, export
):
    from neuropress.kernel_lowering import AutoKernelLowering

    entry = "open_text_c4_gptq_q4_0_moe"
    config = NEUROPRESS_COMPRESSION[entry].__config__.model_copy(
        update={"calibration": CalibrationConfig(n_samples=4, seq_len=32, seed=0)}
    )
    assert config.kernel_lowering is not None
    model = _projected_moe_model()
    lowered_when = []
    monkeypatch.setattr(
        AutoKernelLowering,
        "apply",
        lambda self, m: lowered_when.append(
            all(
                getattr(layer.mlp, "_t2n_quantize_moe_experts_q40", False)
                for layer in model.model.layers
            )
        ),
    )
    run_quantization_pipeline(
        model,
        tokenizer=_MockTokenizer(),
        recipe=config.build(),
        recipe_config_json=config.model_dump_json(),
        block_calibration=block_calibration,
        _load_dataset=_mock_load_dataset,
        **({"export_dirpath": "unused"} if export else {}),
    )
    # Once, with every layer already calibrated; never on the export path.
    assert lowered_when == ([] if export else [True])


def test_block_estimators_see_full_model_paths(monkeypatch):
    from neuropress.estimators import GPTQEstimator

    seen: list[str] = []
    prepare = GPTQEstimator.prepare

    def spy(self, model):
        prepare(self, model)
        seen.extend(self._target_modules)

    monkeypatch.setattr(GPTQEstimator, "prepare", spy)
    _quantize(_moe_model(), "open_text_c4_gptq_q4_0", "float")
    assert "model.layers.1.self_attn.q_proj" in seen
    assert "lm_head" in seen
    assert all(name.startswith(("model.layers.", "lm_head")) for name in seen)


def test_vptq_own_codebooks_may_name_modules_of_other_blocks():
    from neuropress.block_pipeline import BlockAtPaths
    from neuropress.estimators import VPTQGroupEstimator

    model = _moe_model()
    estimator = VPTQGroupEstimator(
        own_codebook_modules=("model.layers.0.self_attn.o_proj",)
    )
    estimator.prepare(BlockAtPaths(model, ["model.layers.1"], entry="model.layers.1"))
    assert "model.layers.1.self_attn.o_proj" in estimator._target_modules
    with pytest.raises(ValueError, match="not among the target modules"):
        VPTQGroupEstimator(own_codebook_modules=("model.layers.9.mlp",)).prepare(
            BlockAtPaths(model, ["model.layers.1"], entry="model.layers.1")
        )


def test_a_teacher_copy_of_a_block_calls_its_own_modules():
    import copy

    from neuropress.block_pipeline import BlockAtPaths

    model = _moe_model()
    block = BlockAtPaths(model, ["model.norm"], entry="model.norm")
    teacher = copy.deepcopy(block)
    with torch.no_grad():
        model.model.norm.weight.zero_()
    x = torch.randn(2, 4, 64)
    assert block(x).abs().sum() == 0
    assert teacher(x).abs().sum() > 0
    assert sum(p.numel() for p in teacher.parameters()) == 64


def test_sequential_teacher_stream_stays_float_with_targets_outside(monkeypatch):
    from neuropress.block_pipeline import BlockQuantizationPipeline
    from neuropress.causal_lm_blocks import _capture as capture

    model = _projected_moe_model()
    ids = torch.randint(0, 128, (1, 32), generator=torch.Generator().manual_seed(0))
    float_hidden, _ = capture(model, model.model.layers, [{"input_ids": ids}])
    streams = {}
    run_block = BlockQuantizationPipeline.run_block

    def spy(self, name, block, student, reference):
        streams.setdefault(name, (list(student), list(reference)))
        return run_block(self, name, block, student, reference)

    monkeypatch.setattr(BlockQuantizationPipeline, "run_block", spy)
    from neuropress.causal_lm_blocks import calibrate_causal_lm_blockwise

    calibrate_causal_lm_blockwise(
        model,
        [{"input_ids": ids}],
        _config("open_text_c4_gptq_q4_0_moe"),
        input_source="sequential",
    )
    student, reference = streams["model.layers.0"]
    torch.testing.assert_close(reference[0]["hidden_states"], float_hidden[0])
    assert not torch.equal(student[0]["hidden_states"], float_hidden[0])


def test_vptq_own_codebooks_may_name_modules_outside_the_stage():
    from neuropress.estimators import VPTQGroupEstimator

    model = _moe_model()
    estimator = VPTQGroupEstimator(
        own_codebook_modules=("model.layers.0.self_attn.o_proj",)
    )
    with estimator.select_targets({"model.layers.1.self_attn.q_proj"}):
        estimator.prepare(model)
        assert list(estimator._target_modules) == ["model.layers.1.self_attn.q_proj"]
    assert estimator.unstaged_filter() is estimator.filter_fn is None
