"""Tests for the modular recipe config framework.

Covers CompositeEstimatorConfig and the generic estimator configs
(GptqEstimatorConfig, GptqPanelEstimatorConfig, MinMaxZpWeightEstimatorConfig),
including a full JSON round-trip that runs GPTQ on nn.Linear, panel GPTQ on
nn.Conv2d, and min-max weight quantization on nn.Embedding — all orchestrated
by a single composite recipe with SmoothQuant preprocessing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from neuropress.estimator_configs import (
    GptqEstimatorConfig,
    GptqPanelEstimatorConfig,
    MinMaxZpWeightEstimatorConfig,
)
from neuropress.estimators import (
    CompositeEstimator,
    GPTQEstimator,
    GPTQPanelEstimator,
    MinMaxZpWeightEstimator,
)
from neuropress.recipe_config import CompositeEstimatorConfig, QuantizationRecipeConfig
from neuropress.t2n.registry import run_quantization_pipeline

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 128
_EMBED_DIM = 64  # divisible by 32 → GPTQ-compatible
_SEQ_LEN = 32
_GROUP_SIZE = 32


# ---------------------------------------------------------------------------
# Fixtures: model, tokenizer, dataset stub
# ---------------------------------------------------------------------------


class _MixedModel(nn.Module):
    """Model with Embedding + Conv2d + Linear for multi-estimator tests.

    Architecture:
        embed: nn.Embedding(128, 64)
        conv:  nn.Conv2d(64, 64, 3, padding=1)   — weight cols = 64*3*3 = 576
        proj:  nn.Linear(64, 64, bias=False)
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(_VOCAB_SIZE, _EMBED_DIM)
        # weight shape [64, 64, 3, 3] → 576 cols → 576 % 32 == 0
        self.conv = nn.Conv2d(_EMBED_DIM, _EMBED_DIM, 3, padding=1)
        self.proj = nn.Linear(_EMBED_DIM, _EMBED_DIM, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)  # [B, seq_len, 64]
        x = x.mean(dim=1)  # [B, 64]
        x = x.unsqueeze(-1).unsqueeze(-1)  # [B, 64, 1, 1]
        x = self.conv(x)  # [B, 64, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [B, 64]
        return self.proj(x)  # [B, 64]


class _MockTokenizer:
    """Returns a fixed stream of token IDs (no real text processing)."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> Any:
        n_tokens = max(len(text.split()) * 8, _SEQ_LEN * 10)
        ids = [i % _VOCAB_SIZE for i in range(n_tokens)]
        return SimpleNamespace(input_ids=ids)


def _mock_load_dataset(dataset_id, dataset_config, split, streaming) -> list[dict]:
    return [{"text": "hello world " * 50}] * 50


def _composite_recipe_json(n_samples: int = 4) -> str:
    """JSON string for a full composite recipe: GPTQ + Panel + MinMax + SmoothQuant."""
    return json.dumps(
        {
            "calibration": {
                "dataset": "c4",
                "seq_len": _SEQ_LEN,
                "n_samples": n_samples,
                "batch_size": 1,
                "seed": 0,
            },
            "preprocessing": [{"type": "smooth_quant", "alpha": 0.5}],
            "estimator": {
                "type": "composite",
                "methods": [
                    {
                        "type": "gptq",
                        "target": "linear",
                        "group_size": _GROUP_SIZE,
                        "nbits": 4,
                        "symmetric": True,
                        "damp_percent": 0.01,
                        "damp_auto_increment": 0.0025,
                    },
                    {
                        "type": "gptq_panel",
                        "target": "conv2d",
                        "group_size": _GROUP_SIZE,
                        "possible_nbits": [4, 8],
                        "target_bpw": 5.0,
                        "symmetric": True,
                        "damp_percent": 0.01,
                        "damp_auto_increment": 0.0025,
                    },
                    {
                        "type": "min_max_zp_weight",
                        "target": "embedding",
                        "nbits": 8,
                        "symmetric": True,
                    },
                ],
            },
            "adapter": {"type": "weight_assignment"},
        }
    )


def _run_pipeline(model: nn.Module, recipe_json: str) -> nn.Module:
    """Parse recipe from JSON and run the quantization pipeline."""
    cfg = QuantizationRecipeConfig.from_json(recipe_json)
    return run_quantization_pipeline(
        model,
        tokenizer=_MockTokenizer(),
        recipe=cfg.build(),
        _load_dataset=_mock_load_dataset,
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_composite_config_registered():
    from neuropress.recipe_config import ESTIMATOR_CONFIG_REGISTRY

    assert "composite" in ESTIMATOR_CONFIG_REGISTRY
    assert ESTIMATOR_CONFIG_REGISTRY["composite"] is CompositeEstimatorConfig


def test_gptq_config_registered():
    from neuropress.recipe_config import ESTIMATOR_CONFIG_REGISTRY

    assert "gptq" in ESTIMATOR_CONFIG_REGISTRY
    assert ESTIMATOR_CONFIG_REGISTRY["gptq"] is GptqEstimatorConfig


def test_gptq_panel_config_registered():
    from neuropress.recipe_config import ESTIMATOR_CONFIG_REGISTRY

    assert "gptq_panel" in ESTIMATOR_CONFIG_REGISTRY
    assert ESTIMATOR_CONFIG_REGISTRY["gptq_panel"] is GptqPanelEstimatorConfig


def test_min_max_zp_weight_config_registered():
    from neuropress.recipe_config import ESTIMATOR_CONFIG_REGISTRY

    assert "min_max_zp_weight" in ESTIMATOR_CONFIG_REGISTRY
    assert (
        ESTIMATOR_CONFIG_REGISTRY["min_max_zp_weight"] is MinMaxZpWeightEstimatorConfig
    )


# ---------------------------------------------------------------------------
# Config build tests
# ---------------------------------------------------------------------------


def test_gptq_config_builds_gptq_estimator():
    est = GptqEstimatorConfig(target="linear", group_size=32, nbits=4).build()
    assert isinstance(est, GPTQEstimator)
    assert est.module_classes == (nn.Linear,)
    assert est.group_size == 32
    assert est.nbits == 4


def test_gptq_panel_config_builds_gptq_panel_estimator():
    est = GptqPanelEstimatorConfig(
        target="conv2d", group_size=32, possible_nbits=[4, 8], target_bpw=5.0
    ).build()
    assert isinstance(est, GPTQPanelEstimator)
    assert est.module_classes == (nn.Conv2d,)
    assert est.group_size == 32
    assert est.possible_nbits == [4, 8]


def test_min_max_zp_weight_config_builds_estimator():
    est = MinMaxZpWeightEstimatorConfig(target="embedding", nbits=8).build()
    assert isinstance(est, MinMaxZpWeightEstimator)
    assert est.module_classes == (nn.Embedding,)
    assert est.nbits == 8


def test_gptq_config_unknown_target_raises():
    with pytest.raises(ValueError, match="Unknown target module 'transformer'"):
        GptqEstimatorConfig(target="transformer").build()


def test_composite_config_builds_composite_estimator():
    cfg = CompositeEstimatorConfig(
        methods=[
            GptqEstimatorConfig(target="linear"),
            MinMaxZpWeightEstimatorConfig(target="embedding"),
        ]
    )
    est = cfg.build()
    assert isinstance(est, CompositeEstimator)
    assert len(est.estimators) == 2
    assert isinstance(est.estimators[0], GPTQEstimator)
    assert isinstance(est.estimators[1], MinMaxZpWeightEstimator)


# ---------------------------------------------------------------------------
# JSON round-trip tests
# ---------------------------------------------------------------------------


def test_composite_json_parses_correctly():
    cfg = QuantizationRecipeConfig.from_json(_composite_recipe_json())

    assert isinstance(cfg.estimator, CompositeEstimatorConfig)
    assert len(cfg.estimator.methods) == 3
    assert isinstance(cfg.estimator.methods[0], GptqEstimatorConfig)
    assert isinstance(cfg.estimator.methods[1], GptqPanelEstimatorConfig)
    assert isinstance(cfg.estimator.methods[2], MinMaxZpWeightEstimatorConfig)

    assert cfg.estimator.methods[0].target == "linear"
    assert cfg.estimator.methods[1].target == "conv2d"
    assert cfg.estimator.methods[2].target == "embedding"

    assert len(cfg.preprocessing) == 1
    assert cfg.preprocessing[0].alpha == 0.5  # type: ignore[union-attr]


def test_composite_json_unknown_sub_estimator_raises():
    bad_json = json.dumps(
        {
            "estimator": {
                "type": "composite",
                "methods": [{"type": "nonexistent_estimator"}],
            }
        }
    )
    match = "Unknown estimator type 'nonexistent_estimator'"
    with pytest.raises(ValueError, match=match):
        QuantizationRecipeConfig.from_json(bad_json)


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------


def test_composite_pipeline_runs_and_modifies_weights():
    """Full pipeline: GPTQ on Linear, panel GPTQ on Conv2d, MinMax on Embedding."""
    model = _MixedModel()
    embed_w0 = model.embed.weight.data.clone()
    conv_w0 = model.conv.weight.data.clone()
    proj_w0 = model.proj.weight.data.clone()

    result = _run_pipeline(model, _composite_recipe_json())

    assert result is model

    # GPTQ modifies proj in-place (dequantized corrected weights)
    assert not torch.allclose(model.proj.weight.data, proj_w0), (
        "proj weight unchanged — GPTQ did not run"
    )

    # Panel GPTQ modifies conv in-place
    assert not torch.allclose(model.conv.weight.data, conv_w0), (
        "conv weight unchanged — panel GPTQ did not run"
    )

    # MinMax quantizes embed in-place
    assert not torch.allclose(model.embed.weight.data, embed_w0), (
        "embed weight unchanged — min-max quantization did not run"
    )


def test_composite_pipeline_model_remains_callable():
    """Quantized model must still accept the same input API."""
    model = _MixedModel()
    _run_pipeline(model, _composite_recipe_json())
    input_ids = torch.randint(0, _VOCAB_SIZE, (2, _SEQ_LEN))
    out = model(input_ids)
    assert out.shape == (2, _EMBED_DIM)


def test_composite_pipeline_returns_same_model_object():
    model = _MixedModel()
    result = _run_pipeline(model, _composite_recipe_json())
    assert result is model


# ---------------------------------------------------------------------------
# Estimator settings MUST survive serialisation.
#
# `estimator` is declared as the base type `EstimatorConfig`, and Pydantic v2
# serialises by the DECLARED type, so every subclass field was silently dropped
# and model_dump_json() emitted `{"estimator": {"type": ...}}`.
#
# The recipe cache fingerprint is a SHA-256 of that JSON, so all VPTQ recipes
# collapsed to one cache scope: on 2026-08-07 a per-group-NC run loaded the
# 12 codebooks of an earlier uniform NC=16384 run and never trained its own,
# which silently turned three "different" arms into the same model.
# ---------------------------------------------------------------------------


def test_estimator_subclass_fields_survive_dump():
    from neuropress.estimator_configs import VptqGroupEstimatorConfig
    from neuropress.recipe_config import QuantizationRecipeConfig

    cfg = QuantizationRecipeConfig(
        estimator=VptqGroupEstimatorConfig(num_centroids=4096, vector_len=8)
    )
    dumped = json.loads(cfg.model_dump_json())["estimator"]
    assert dumped["num_centroids"] == 4096
    assert dumped["vector_len"] == 8


def test_recipes_differing_only_in_per_group_nc_have_distinct_fingerprints():
    from neuropress.estimator_configs import VptqGroupEstimatorConfig
    from neuropress.quantization.cache import _recipe_fingerprint
    from neuropress.recipe_config import QuantizationRecipeConfig

    uniform = QuantizationRecipeConfig(
        estimator=VptqGroupEstimatorConfig(num_centroids=16384)
    )
    per_group = QuantizationRecipeConfig(
        estimator=VptqGroupEstimatorConfig(
            num_centroids=16384, num_centroids_per_group={"down_proj": 256}
        )
    )
    assert _recipe_fingerprint(uniform.model_dump_json()) != _recipe_fingerprint(
        per_group.model_dump_json()
    )


def test_recipes_differing_only_in_vector_len_have_distinct_fingerprints():
    from neuropress.estimator_configs import VptqGroupEstimatorConfig
    from neuropress.quantization.cache import _recipe_fingerprint
    from neuropress.recipe_config import QuantizationRecipeConfig

    v4 = QuantizationRecipeConfig(estimator=VptqGroupEstimatorConfig(vector_len=4))
    v8 = QuantizationRecipeConfig(estimator=VptqGroupEstimatorConfig(vector_len=8))
    assert _recipe_fingerprint(v4.model_dump_json()) != _recipe_fingerprint(
        v8.model_dump_json()
    )


def test_preprocessing_step_fields_survive_dump():
    """A rotated run must not share a cache scope with an unrotated one."""
    from neuropress.estimator_configs import VptqGroupEstimatorConfig
    from neuropress.preprocessing import HadamardRotationStepConfig
    from neuropress.quantization.cache import _recipe_fingerprint
    from neuropress.recipe_config import QuantizationRecipeConfig

    plain = QuantizationRecipeConfig(estimator=VptqGroupEstimatorConfig())
    rotated = QuantizationRecipeConfig(
        estimator=VptqGroupEstimatorConfig(),
        preprocessing=[HadamardRotationStepConfig(seed=1234)],
    )
    assert _recipe_fingerprint(plain.model_dump_json()) != _recipe_fingerprint(
        rotated.model_dump_json()
    )
