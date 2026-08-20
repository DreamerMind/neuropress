"""Tests for quantized model save/load (neuropress.t2n.model_io)."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

from neuropress.estimator_configs import GptqEstimatorConfig
from neuropress.recipe import CalibrationConfig
from neuropress.recipe_config import QuantizationRecipeConfig
from neuropress.t2n.model_io import (
    QuantizationManifest,
    attach_manifest,
    load_quantized_model,
    save_quantized_model,
)
from neuropress.t2n.registry import run_quantization_pipeline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 128
_DIM = 64
_SEQ_LEN = 32
_Q4_0_GROUP_SIZE = 32


class _TinyLM(nn.Module):
    """Minimal LLM-shaped model with one quantizable linear layer."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(_VOCAB_SIZE, _DIM)
        self.proj = nn.Linear(_DIM, _DIM, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids).mean(dim=1)
        return self.proj(x)


class _MockTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False) -> Any:
        n_tokens = max(len(text.split()) * 8, _SEQ_LEN * 10)
        ids = [i % _VOCAB_SIZE for i in range(n_tokens)]
        return SimpleNamespace(input_ids=ids)


def _mock_load_dataset(dataset_id, dataset_config, split, streaming):
    return [{"text": "hello world " * 50}] * 50


def _make_recipe(n_samples: int = 4) -> QuantizationRecipeConfig:
    return QuantizationRecipeConfig(
        calibration=CalibrationConfig(n_samples=n_samples, seq_len=_SEQ_LEN, seed=0),
        estimator=GptqEstimatorConfig(
            group_size=_Q4_0_GROUP_SIZE, nbits=4, weight_transform="q4_0"
        ),
        # Disable kernel lowering so QTensor weights remain available for
        # manifest scanning and save/load round-trips.
        kernel_lowering=None,
    )


def _quantize(model: nn.Module, n_samples: int = 4) -> nn.Module:
    cfg = _make_recipe(n_samples)
    return run_quantization_pipeline(
        model,
        tokenizer=_MockTokenizer(),
        recipe=cfg.build(),
        recipe_config_json=cfg.model_dump_json(),
        _load_dataset=_mock_load_dataset,
    )


# ---------------------------------------------------------------------------
# attach_manifest
# ---------------------------------------------------------------------------


def test_manifest_attached_after_quantization():
    model = _TinyLM()
    _quantize(model)
    assert hasattr(model, "__neuropress_manifest__")
    manifest = model.__neuropress_manifest__
    assert isinstance(manifest, QuantizationManifest)


def test_manifest_records_qtensor_param():
    model = _TinyLM()
    _quantize(model)
    manifest = model.__neuropress_manifest__
    assert "proj.weight" in manifest.qtensor_params
    assert manifest.qtensor_params["proj.weight"] == "QTensorTractScaleOnly"


def test_manifest_does_not_record_float_param():
    model = _TinyLM()
    _quantize(model)
    manifest = model.__neuropress_manifest__
    assert "embed.weight" not in manifest.qtensor_params


def test_manifest_stores_recipe_config_json():
    model = _TinyLM()
    _quantize(model)
    manifest = model.__neuropress_manifest__
    assert manifest.recipe_config_json is not None
    parsed = json.loads(manifest.recipe_config_json)
    assert parsed["estimator"]["type"] == "gptq"


def test_manifest_has_transforms_list():

    model = _TinyLM()
    _quantize(model)
    manifest = model.__neuropress_manifest__
    assert isinstance(manifest.transforms, list)
    types = [t.type for t in manifest.transforms]
    assert "qtensor_assignment" in types


def test_qtensor_assignment_transform_has_metadata():
    from neuropress.t2n.model_io import QtensorAssignmentTransform

    model = _TinyLM()
    _quantize(model)
    manifest = model.__neuropress_manifest__
    qt_transform = next(
        t for t in manifest.transforms if isinstance(t, QtensorAssignmentTransform)
    )
    entry = qt_transform.qtensor_params["proj.weight"]
    assert entry.type == "QTensorTractScaleOnly"
    assert entry.tensor_keys
    assert entry.metadata is not None


def test_attach_manifest_explicit():
    model = _TinyLM()
    _quantize(model)
    manifest = attach_manifest(model, recipe_config_json='{"custom": true}')
    assert manifest.recipe_config_json == '{"custom": true}'
    assert model.__neuropress_manifest__ is manifest


def test_attach_manifest_preserves_other_transforms():
    """attach_manifest must not wipe non-qtensor transforms."""

    model = _TinyLM()
    _quantize(model)
    manifest = model.__neuropress_manifest__
    manifest.transforms.append({"type": "custom_transform", "value": 42})
    attach_manifest(model)
    transforms = model.__neuropress_manifest__.transforms

    def _type(t: object) -> str:
        return t.type if hasattr(t, "type") else t["type"]  # type: ignore[index]

    types = [_type(t) for t in transforms]
    assert "qtensor_assignment" in types
    assert "custom_transform" in types


# ---------------------------------------------------------------------------
# save / load roundtrip — weights
# ---------------------------------------------------------------------------


def test_roundtrip_qtensor_type(tmp_path):
    model = _TinyLM()
    _quantize(model)
    save_quantized_model(model, tmp_path / "m.safetensors")

    model2 = _TinyLM()
    load_quantized_model(model2, tmp_path / "m.safetensors", kernel_lowering=None)
    assert isinstance(model2.proj.weight, QTensorTractScaleOnly)


def test_roundtrip_u8_blob_identical(tmp_path):
    model = _TinyLM()
    _quantize(model)
    save_quantized_model(model, tmp_path / "m.safetensors")

    model2 = _TinyLM()
    load_quantized_model(model2, tmp_path / "m.safetensors", kernel_lowering=None)
    assert torch.equal(model.proj.weight.u8_blob, model2.proj.weight.u8_blob)


def test_roundtrip_scale_identical(tmp_path):
    model = _TinyLM()
    _quantize(model)
    save_quantized_model(model, tmp_path / "m.safetensors")

    model2 = _TinyLM()
    load_quantized_model(model2, tmp_path / "m.safetensors", kernel_lowering=None)
    assert torch.equal(
        model.proj.weight.qscheme.scale, model2.proj.weight.qscheme.scale
    )


def test_roundtrip_float_param_identical(tmp_path):
    model = _TinyLM()
    _quantize(model)
    save_quantized_model(model, tmp_path / "m.safetensors")

    model2 = _TinyLM()
    load_quantized_model(model2, tmp_path / "m.safetensors", kernel_lowering=None)
    assert torch.allclose(model.embed.weight.data, model2.embed.weight.data)


def test_roundtrip_nnef_name_preserved(tmp_path):
    model = _TinyLM()
    _quantize(model)
    # nnef_name is set by the pipeline transform
    assert model.proj.weight.nnef_name == "proj.weight"
    save_quantized_model(model, tmp_path / "m.safetensors")

    model2 = _TinyLM()
    load_quantized_model(model2, tmp_path / "m.safetensors", kernel_lowering=None)
    assert model2.proj.weight.nnef_name == "proj.weight"


def test_roundtrip_forward_runs(tmp_path):
    model = _TinyLM()
    _quantize(model)
    save_quantized_model(model, tmp_path / "m.safetensors")

    model2 = _TinyLM()
    load_quantized_model(model2, tmp_path / "m.safetensors", kernel_lowering=None)
    ids = torch.randint(0, _VOCAB_SIZE, (2, _SEQ_LEN))
    out = model2(ids)
    assert out.shape == (2, _DIM)


# ---------------------------------------------------------------------------
# save / load roundtrip — manifest
# ---------------------------------------------------------------------------


def test_load_returns_manifest(tmp_path):
    model = _TinyLM()
    _quantize(model)
    save_quantized_model(model, tmp_path / "m.safetensors")

    model2 = _TinyLM()
    manifest = load_quantized_model(
        model2,
        tmp_path / "m.safetensors",
        kernel_lowering=None,
    )
    assert isinstance(manifest, QuantizationManifest)


def test_load_attaches_manifest_to_model(tmp_path):
    model = _TinyLM()
    _quantize(model)
    save_quantized_model(model, tmp_path / "m.safetensors")

    model2 = _TinyLM()
    load_quantized_model(model2, tmp_path / "m.safetensors", kernel_lowering=None)
    assert hasattr(model2, "__neuropress_manifest__")
    assert "proj.weight" in model2.__neuropress_manifest__.qtensor_params


def test_load_restores_recipe_config_json(tmp_path):
    model = _TinyLM()
    _quantize(model)
    save_quantized_model(model, tmp_path / "m.safetensors")

    model2 = _TinyLM()
    manifest = load_quantized_model(
        model2,
        tmp_path / "m.safetensors",
        kernel_lowering=None,
    )
    assert manifest.recipe_config_json is not None
    parsed = json.loads(manifest.recipe_config_json)
    assert parsed["estimator"]["type"] == "gptq"


def test_load_restores_transforms_list(tmp_path):
    model = _TinyLM()
    _quantize(model)
    save_quantized_model(model, tmp_path / "m.safetensors")

    model2 = _TinyLM()
    manifest = load_quantized_model(
        model2,
        tmp_path / "m.safetensors",
        kernel_lowering=None,
    )
    types = [t.type if hasattr(t, "type") else t["type"] for t in manifest.transforms]  # type: ignore[index]
    assert "qtensor_assignment" in types


# ---------------------------------------------------------------------------
# File format
# ---------------------------------------------------------------------------


def test_file_readable_as_safetensors(tmp_path):
    """Saved file must be directly readable by safetensors (no pickle)."""
    from safetensors import safe_open

    model = _TinyLM()
    _quantize(model)
    path = tmp_path / "m.safetensors"
    save_quantized_model(model, path)

    with safe_open(str(path), framework="pt") as f:
        keys = list(f.keys())
        meta = f.metadata()

    assert "embed.weight" in keys
    assert "proj.weight::u8_blob" in keys
    assert "proj.weight::scale" in keys
    assert "proj.weight" not in keys  # no raw QTensor key
    raw = json.loads(meta["__neuropress_manifest__"])
    assert raw["format_version"] == "2"
    transforms_types = [t["type"] for t in raw["transforms"]]
    assert "qtensor_assignment" in transforms_types


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_load_missing_manifest_raises(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "bad.safetensors"
    save_file({"w": torch.zeros(4)}, str(path))
    model = _TinyLM()
    with pytest.raises(ValueError, match="__neuropress_manifest__"):
        load_quantized_model(model, path)


def test_save_recipe_config_json_override(tmp_path):
    """recipe_config_json arg overrides the attached manifest's value."""
    model = _TinyLM()
    _quantize(model)
    path = tmp_path / "m.safetensors"
    save_quantized_model(model, path, recipe_config_json='{"override": 1}')

    model2 = _TinyLM()
    manifest = load_quantized_model(model2, path)
    assert json.loads(manifest.recipe_config_json) == {"override": 1}
