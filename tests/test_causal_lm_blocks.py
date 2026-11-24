import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
qwen3 = pytest.importorskip("transformers.models.qwen3")
qwen3_moe = pytest.importorskip("transformers.models.qwen3_moe")

from neuropress.causal_lm_blocks import (  # noqa: E402
    _capture,
    calibrate_causal_lm_blockwise,
)
from neuropress.recipe import CalibrationConfig  # noqa: E402
from neuropress.t2n.registry import (  # noqa: E402
    NEUROPRESS_COMPRESSION,
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


def _quantize(model, entry: str, block_calibration, cache_dir=None):
    config = NEUROPRESS_COMPRESSION[entry].__config__.model_copy(
        update={
            "calibration": CalibrationConfig(n_samples=4, seq_len=32, seed=0),
            "kernel_lowering": None,
        }
    )
    run_quantization_pipeline(
        model,
        tokenizer=_MockTokenizer(),
        recipe=config.build(),
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
    ],
    ids=["moe", "hybrid_attention"],
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

    model = _quantize(_moe_model(), "open_text_c4_gptq_q4_0_moe", "sequential")
    assert isinstance(model.lm_head.weight, QTensorTractScaleOnly)
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


def test_targets_outside_the_blocks_are_refused():
    model = _moe_model()
    model.extra = nn.Linear(64, 64)
    config = NEUROPRESS_COMPRESSION["open_text_c4_gptq_q4_0_moe"].__config__
    with pytest.raises(ValueError, match=r"outside them: \['extra'\]"):
        calibrate_causal_lm_blockwise(model, [], config)


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
