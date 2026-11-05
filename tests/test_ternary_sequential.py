"""Tests for dense modules, tied embeddings and sequential ternary GPTQ."""

import json

import torch
from safetensors import safe_open
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from neuropress.formats.prism_ternary import export_prism_ternary_checkpoint
from neuropress.preprocessing import TernaryHadamardStep
from neuropress.transforms.ternary_hadamard import TernaryHadamardInputLinear


def _tiny_llama(*, tie: bool = False, layers: int = 2) -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=layers,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=1024,
        tie_word_embeddings=tie,
    )
    return LlamaForCausalLM(config).eval()


def _batches(n: int = 4, seq: int = 16) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1)
    return [
        {"input_ids": torch.randint(0, 1024, (1, seq), generator=generator)}
        for _ in range(n)
    ]


def _make_wrapped_weights_ternary(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, TernaryHadamardInputLinear):
            weight = module.linear.weight
            levels = torch.arange(weight.numel()).reshape_as(weight) % 3 - 1
            weight.data.copy_(levels * 0.125)


def test_dense_modules_are_not_wrapped_and_are_exported_dense(tmp_path):
    model = _tiny_llama()
    original = model.lm_head.weight.detach().clone()
    TernaryHadamardStep().apply(model, [])

    assert type(model.lm_head) is nn.Linear
    torch.testing.assert_close(model.lm_head.weight, original)
    assert isinstance(
        model.model.layers[0].self_attn.q_proj, TernaryHadamardInputLinear
    )

    _make_wrapped_weights_ternary(model)
    export_prism_ternary_checkpoint(model, tmp_path / "out", model.config.to_dict())

    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["quantization_config"]["dense_modules"] == ["lm_head"]
    with safe_open(tmp_path / "out" / "model.safetensors", framework="pt") as ckpt:
        assert ckpt.get_tensor("lm_head.weight").dtype == original.dtype
        assert "lm_head.scales" not in ckpt.keys()


def test_all_ternary_export_omits_dense_modules_key(tmp_path):
    model = _tiny_llama()
    TernaryHadamardStep(dense_modules=()).apply(model, [])
    _make_wrapped_weights_ternary(model)
    export_prism_ternary_checkpoint(model, tmp_path / "out", model.config.to_dict())

    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert "dense_modules" not in config["quantization_config"]


def test_ternary_lm_head_unties_embeddings(tmp_path):
    model = _tiny_llama(tie=True)
    embedding = model.model.embed_tokens.weight.detach().clone()
    TernaryHadamardStep(dense_modules=()).apply(model, [])

    assert model.config.tie_word_embeddings is False
    torch.testing.assert_close(model.model.embed_tokens.weight, embedding)

    _make_wrapped_weights_ternary(model)
    base = model.config.to_dict()
    base["tie_word_embeddings"] = True  # as in the original checkpoint
    export_prism_ternary_checkpoint(model, tmp_path / "out", base)

    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["tie_word_embeddings"] is False
    with safe_open(tmp_path / "out" / "model.safetensors", framework="pt") as ckpt:
        keys = set(ckpt.keys())
    assert {"model.embed_tokens.weight", "lm_head.weight", "lm_head.scales"} <= keys


def test_dense_lm_head_keeps_tie():
    model = _tiny_llama(tie=True)
    TernaryHadamardStep().apply(model, [])

    assert model.config.tie_word_embeddings is True
    assert model.lm_head.weight is model.model.embed_tokens.weight
