"""Tests for _model_bytes accounting of packed QTensor storage.

Verifies that after Q4_0 quantization the measured model size decreases
significantly (the linear layer goes from float32 to packed 4-bit + f16
scale), and that _model_bytes matches the naive formula for unquantized
float models.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from neuropress.estimator_configs import GptqEstimatorConfig
from neuropress.eval.llm_cli import _model_bytes
from neuropress.recipe import CalibrationConfig
from neuropress.recipe_config import QuantizationRecipeConfig
from neuropress.t2n.packing import (
    BitPacking,
    BitPacking1,
    BitPacking2,
    BitPacking4,
    BitPacking8,
)
from neuropress.t2n.registry import run_quantization_pipeline

# ---------------------------------------------------------------------------
# Shared fixtures (same pattern as test_model_io.py)
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


def _make_q4_0_recipe(n_samples: int = 4) -> QuantizationRecipeConfig:
    return QuantizationRecipeConfig(
        calibration=CalibrationConfig(n_samples=n_samples, seq_len=_SEQ_LEN, seed=0),
        estimator=GptqEstimatorConfig(
            group_size=_Q4_0_GROUP_SIZE, nbits=4, weight_transform="q4_0"
        ),
    )


def _quantize_model(model: nn.Module) -> nn.Module:
    cfg = _make_q4_0_recipe()
    return run_quantization_pipeline(
        model,
        tokenizer=_MockTokenizer(),
        recipe=cfg.build(),
        recipe_config_json=cfg.model_dump_json(),
        _load_dataset=_mock_load_dataset,
    )


# ---------------------------------------------------------------------------
# _model_bytes tests
# ---------------------------------------------------------------------------


def test_model_bytes_baseline_matches_naive():
    """For a plain float model, _model_bytes matches the naive formula."""
    model = _TinyLM()
    expected = sum(p.numel() * p.element_size() for p in model.parameters())
    assert _model_bytes(model) == expected


def test_model_bytes_decreases_after_q4_0():
    """Q4_0 quantization should reduce the measured model size by at least 1.2x.

    _TinyLM breakdown (float32 baseline):
      - embed: Embedding(128, 64) = 8192 params * 4 B = 32768 B
      - proj:  Linear(64, 64)     = 4096 params * 4 B = 16384 B
      - Total baseline: 49152 B

    After Q4_0 quantization of proj (stored as u8_blob + f16 scale):
      - u8_blob: 4096 uint8 values (one 4-bit val per byte) = 4096 B
      - scale:   (4096 // 32) groups * 2 B (float16)        =  256 B
      - Total quantized proj: 4352 B  (~3.8x smaller than float32)

    Embedding stays float32.  Overall model ratio ≈ 49152 / 37120 = 1.32x.
    The test asserts >= 1.2x to confirm the QTensor serde path is active.
    """
    model = _TinyLM()
    baseline_bytes = _model_bytes(model)

    _quantize_model(model)
    quantized_bytes = _model_bytes(model)

    ratio = baseline_bytes / quantized_bytes
    assert ratio >= 1.2, (
        f"Expected >= 1.2x size reduction after Q4_0, "
        f"got {baseline_bytes} B -> {quantized_bytes} B (ratio {ratio:.2f}x)"
    )
    assert quantized_bytes > 0


# ---------------------------------------------------------------------------
# U8Compressor tests
# ---------------------------------------------------------------------------


def test_bit_packing_invalid_nbits():
    """BitPacking rejects unsupported bit-widths."""
    import pytest

    with pytest.raises(ValueError, match="nbits must be one of"):
        BitPacking(3)


def test_bit_packing_roundtrip_4bit():
    """compress/decompress roundtrip for 4-bit values is lossless."""
    x = torch.arange(16, dtype=torch.uint8)
    packed = BitPacking4.compress(x)
    assert packed.numel() == 8
    assert packed.dtype == torch.uint8
    restored = BitPacking4.decompress(packed)
    assert (restored == x).all()


def test_bit_packing_roundtrip_2bit():
    """compress/decompress roundtrip for 2-bit values is lossless."""
    x = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.uint8)
    packed = BitPacking2.compress(x)
    assert packed.numel() == 2
    restored = BitPacking2.decompress(packed)
    assert (restored == x).all()


def test_bit_packing_roundtrip_1bit():
    """compress/decompress roundtrip for 1-bit values is lossless."""
    x = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.uint8)
    packed = BitPacking1.compress(x)
    assert packed.numel() == 1
    restored = BitPacking1.decompress(packed)
    assert (restored == x).all()


def test_bit_packing_roundtrip_8bit():
    """compress/decompress roundtrip for 8-bit values is identity."""
    x = torch.randint(0, 256, (32,), dtype=torch.uint8)
    packed = BitPacking8.compress(x)
    assert packed.numel() == 32
    restored = BitPacking8.decompress(packed)
    assert (restored == x).all()


def test_bit_packing_non_divisible_numel():
    """compress raises if numel is not divisible by values_per_byte."""
    import pytest

    x = torch.tensor([5, 9, 3], dtype=torch.uint8)  # 3 not divisible by 2
    with pytest.raises(ValueError, match="must be divisible"):
        BitPacking4.compress(x)


def test_bit_packing_compressed_numel():
    """compressed_numel returns the correct output byte count."""
    assert BitPacking4.compressed_numel(8) == 4
    assert BitPacking4.compressed_numel(16) == 8
    assert BitPacking2.compressed_numel(8) == 2
    assert BitPacking1.compressed_numel(8) == 1
    assert BitPacking8.compressed_numel(8) == 8
