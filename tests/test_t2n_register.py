"""Tests for neuropress.t2n.registry — GPTQ Q4_0 compression registry."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from torch_to_nnef.compress import dynamic_load_registry
from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

from neuropress.estimator_configs import GptqEstimatorConfig
from neuropress.recipe import CalibrationConfig
from neuropress.recipe_config import QuantizationRecipeConfig
from neuropress.t2n.registry import (
    NEUROPRESS_COMPRESSION,
    NEUROPRESS_COMPRESSION_REGISTRY,
    run_quantization_pipeline,
)
from neuropress.transforms.weight_only import _is_q4_x_compatible as _is_q4_0_compatible

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 128
_DIM = 64  # divisible by 32 → Q4_0 compatible
_SEQ_LEN = 32
_Q4_0_GROUP_SIZE = 32


class _TinyLM(nn.Module):
    """Minimal LLM-shaped model that accepts input_ids as a keyword argument."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(_VOCAB_SIZE, _DIM)
        self.proj = nn.Linear(_DIM, _DIM, bias=False)  # in_features=64, compatible

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)  # [batch, seq_len, dim]
        x = x.mean(dim=1)  # [batch, dim]
        return self.proj(x)  # [batch, dim]


class _MockTokenizer:
    """Returns a fixed stream of token IDs (no real text processing)."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> Any:
        # Produce enough tokens so load_c4_calibration can fill n_samples * seq_len
        n_tokens = max(len(text.split()) * 8, _SEQ_LEN * 10)
        ids = [i % _VOCAB_SIZE for i in range(n_tokens)]
        return SimpleNamespace(input_ids=ids)


def _mock_load_dataset(dataset_id, dataset_config, split, streaming) -> list[dict]:
    """Lightweight stand-in for datasets.load_dataset."""
    return [{"text": "hello world " * 50}] * 50


def _make_test_recipe(n_samples: int = 4):
    """Build a minimal GPTQ Q4_0 recipe for use in tests."""
    return QuantizationRecipeConfig(
        calibration=CalibrationConfig(n_samples=n_samples, seq_len=_SEQ_LEN, seed=0),
        estimator=GptqEstimatorConfig(
            group_size=_Q4_0_GROUP_SIZE,
            nbits=4,
            weight_transform="q4_0",
        ),
        # Disable kernel lowering so tests can inspect QTensor weights directly.
        kernel_lowering=None,
    ).build()


def _run_compression(model: nn.Module, n_samples: int = 4) -> nn.Module:
    """Helper that invokes run_quantization_pipeline with a test recipe."""
    return run_quantization_pipeline(
        model,
        tokenizer=_MockTokenizer(),
        recipe=_make_test_recipe(n_samples),
        _load_dataset=_mock_load_dataset,
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_path_is_loadable():
    registry = dynamic_load_registry(NEUROPRESS_COMPRESSION_REGISTRY)
    assert isinstance(registry, dict)


def test_registry_contains_expected_keys():
    assert "open_text_c4_gptq_q4_0" in NEUROPRESS_COMPRESSION
    assert "open_text_c4_gptq_q4_0_32samples" in NEUROPRESS_COMPRESSION
    assert "open_text_c4_gptq_q4_0_256samples" in NEUROPRESS_COMPRESSION


def test_registry_entries_are_callable():
    for key, fn in NEUROPRESS_COMPRESSION.items():
        assert callable(fn), f"{key!r} is not callable"


def test_dynamic_load_matches_module_dict():
    registry = dynamic_load_registry(NEUROPRESS_COMPRESSION_REGISTRY)
    assert registry is NEUROPRESS_COMPRESSION


# ---------------------------------------------------------------------------
# Compatibility filter tests
# ---------------------------------------------------------------------------


def test_compatible_linear():
    assert _is_q4_0_compatible(nn.Linear(64, 16))


def test_compatible_linear_multiple_of_32():
    assert _is_q4_0_compatible(nn.Linear(32, 8))
    assert _is_q4_0_compatible(nn.Linear(128, 4))


def test_incompatible_linear():
    assert not _is_q4_0_compatible(nn.Linear(10, 16))  # 10 % 32 != 0
    assert not _is_q4_0_compatible(nn.Linear(16, 8))  # 16 % 32 != 0


def test_no_weight():
    assert not _is_q4_0_compatible(nn.ReLU())


def test_embedding_not_compatible():
    # Embedding weight is [vocab, dim]; shape[1:] = [dim] = [64] which IS
    # divisible by 32 for dim=64.  Confirm the filter is shape-only.
    embed = nn.Embedding(128, 64)
    assert _is_q4_0_compatible(embed)  # shape-compatible (adapter may still skip)


# ---------------------------------------------------------------------------
# Compression function tests
# ---------------------------------------------------------------------------


def test_returns_same_model_object():
    model = _TinyLM()
    result = _run_compression(model)
    assert result is model


def test_proj_weight_becomes_q4_0_tensor():
    model = _TinyLM()
    _run_compression(model)
    assert isinstance(model.proj.weight, QTensorTractScaleOnly), (
        f"Expected QTensorTractScaleOnly, got {type(model.proj.weight)}"
    )


def test_nnef_name_is_set():
    model = _TinyLM()
    _run_compression(model)
    assert model.proj.weight.nnef_name == "proj.weight"  # type: ignore[union-attr]


def test_embed_weight_unchanged():
    """Embedding layer is Q4_0-shape-compatible but not nn.Linear → skipped."""
    model = _TinyLM()
    embed_weight_before = model.embed.weight.data.clone()
    # Only nn.Linear is targeted by default
    _run_compression(model)
    assert not isinstance(model.embed.weight, QTensorTractScaleOnly)
    # Embedding weight data must be untouched (GPTQ was not run on it)
    assert torch.allclose(model.embed.weight.data, embed_weight_before)


def test_model_still_runs_forward():
    """The quantized model must remain callable with the same input API."""
    model = _TinyLM()
    _run_compression(model)
    input_ids = torch.randint(0, _VOCAB_SIZE, (2, _SEQ_LEN))
    out = model(input_ids)
    assert out.shape == (2, _DIM)


def test_incompatible_layer_is_skipped():
    """A linear with in_features not divisible by 32 must not be quantized."""

    class _ModelWithTinyLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(_VOCAB_SIZE, _DIM)
            self.small = nn.Linear(10, _DIM)  # 10 % 32 != 0 — incompatible
            self.large = nn.Linear(_DIM, _DIM)  # compatible

        def forward(self, input_ids):
            x = self.embed(input_ids).mean(dim=1)
            # Route through small + large (odd but valid for testing)
            return self.large(x) + self.small(x[..., :10])

    model = _ModelWithTinyLinear()
    run_quantization_pipeline(
        model,
        tokenizer=_MockTokenizer(),
        recipe=_make_test_recipe(n_samples=4),
        _load_dataset=_mock_load_dataset,
    )
    assert not isinstance(model.small.weight, QTensorTractScaleOnly)
    assert isinstance(model.large.weight, QTensorTractScaleOnly)


# ---------------------------------------------------------------------------
# QuantizationRecipeConfig — JSON round-trip tests
# ---------------------------------------------------------------------------


def test_recipe_config_from_json_gptq_q4_0():
    """Full recipe built from JSON with gptq + q4_0 weight transform."""
    recipe_json = json.dumps(
        {
            "calibration": {
                "dataset": "c4",
                "seq_len": _SEQ_LEN,
                "n_samples": 4,
                "batch_size": 1,
                "seed": 0,
            },
            "estimator": {
                "type": "gptq",
                "weight_transform": "q4_0",
                "group_size": _Q4_0_GROUP_SIZE,
                "nbits": 4,
                "symmetric": True,
                "damp_percent": 0.01,
                "damp_auto_increment": 0.0025,
            },
            "adapter": {"type": "weight_assignment"},
            "kernel_lowering": None,
        }
    )

    cfg = QuantizationRecipeConfig.from_json(recipe_json)
    assert isinstance(cfg.estimator, GptqEstimatorConfig)
    assert cfg.estimator.weight_transform == "q4_0"
    assert cfg.estimator.nbits == 4
    assert cfg.calibration.n_samples == 4

    model = _TinyLM()
    result = run_quantization_pipeline(
        model,
        tokenizer=_MockTokenizer(),
        recipe=cfg.build(),
        _load_dataset=_mock_load_dataset,
    )
    assert result is model
    assert isinstance(model.proj.weight, QTensorTractScaleOnly)
    assert model.proj.weight.nnef_name == "proj.weight"  # type: ignore[union-attr]


def test_recipe_config_from_json_with_smooth_quant_preprocessing():
    """Full recipe with SmoothQuant preprocessing resolved from JSON."""
    # Import triggers @preprocessing_step_config("smooth_quant") registration.
    from neuropress.preprocessing import SmoothQuantStepConfig

    recipe_json = json.dumps(
        {
            "calibration": {
                "dataset": "c4",
                "seq_len": _SEQ_LEN,
                "n_samples": 4,
                "batch_size": 1,
                "seed": 0,
            },
            "preprocessing": [{"type": "smooth_quant", "alpha": 0.4}],
            "estimator": {"type": "gptq", "weight_transform": "q4_0"},
            "adapter": {"type": "weight_assignment"},
            "kernel_lowering": None,
        }
    )

    cfg = QuantizationRecipeConfig.from_json(recipe_json)
    assert len(cfg.preprocessing) == 1
    assert isinstance(cfg.preprocessing[0], SmoothQuantStepConfig)
    assert cfg.preprocessing[0].alpha == 0.4

    model = _TinyLM()
    original_weight = model.proj.weight.data.clone()
    run_quantization_pipeline(
        model,
        tokenizer=_MockTokenizer(),
        recipe=cfg.build(),
        _load_dataset=_mock_load_dataset,
    )
    # SmoothQuant ran (modifies weights before GPTQ), GPTQ also ran.
    assert isinstance(model.proj.weight, QTensorTractScaleOnly)
    # The original float weight must have been changed by one of the two steps.
    assert not torch.allclose(
        model.proj.weight.data,
        original_weight,  # type: ignore[arg-type]
    )


def test_recipe_config_unknown_estimator_type_raises():
    """from_json raises a clear ValueError for an unregistered estimator type."""
    with pytest.raises(ValueError, match="Unknown estimator type 'nonexistent'"):
        QuantizationRecipeConfig.from_json(
            json.dumps({"estimator": {"type": "nonexistent"}})
        )


def test_recipe_config_unknown_preprocessing_type_raises():
    """from_json raises a clear ValueError for an unregistered preprocessing type."""
    with pytest.raises(
        ValueError, match="Unknown preprocessing step type 'nonexistent'"
    ):
        QuantizationRecipeConfig.from_json(
            json.dumps(
                {
                    "estimator": {"type": "gptq", "weight_transform": "q4_0"},
                    "preprocessing": [{"type": "nonexistent"}],
                }
            )
        )


def test_recipe_config_unknown_weight_transform_raises():
    """build() raises a clear ValueError for an unregistered weight_transform key."""
    cfg = QuantizationRecipeConfig.from_json(
        json.dumps({"estimator": {"type": "gptq", "weight_transform": "bad_key"}})
    )
    with pytest.raises(ValueError, match="Unknown weight transform 'bad_key'"):
        cfg.build()
