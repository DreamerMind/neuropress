"""Integration test: GPTQ calibration with C4 data → NNEF export.

Demonstrates the full neuropress + torch_to_nnef LLM workflow:

    1. Load a HuggingFace causal LM (Gemma-3-270M, ~270 M params).
    2. Build C4 calibration batches with ``load_c4_calibration``.
    3. Apply 4-bit GPTQ to every ``nn.Linear`` in the model via
       :class:`.QuantizationPipeline` + :class:`.GPTQEstimator`.
    4. Export the quantized model to an NNEF archive with
       ``torch_to_nnef``'s dedicated LLM exporter.

Gate
----
This test downloads a ~500 MB model and runs several minutes.
It is skipped by default.  To run it::

    RUN_INTEGRATION=1 uv run pytest tests/integration/ -v

Optional dependencies required::

    pip install neuropress[llm]      # datasets
    pip install transformers         # HuggingFace model loading
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Gate — skip the whole module unless explicitly opted in
# ---------------------------------------------------------------------------

if not os.getenv("RUN_INTEGRATION"):
    pytest.skip(
        "Set RUN_INTEGRATION=1 to run LLM integration tests.",
        allow_module_level=True,
    )

# Require optional dependencies — skip with a clear message if absent.
pytest.importorskip("transformers", reason="pip install transformers")
pytest.importorskip("datasets", reason="pip install neuropress[llm]")
pytest.importorskip("torch_to_nnef_llm", reason="pip install neuropress[eval]")

# ---------------------------------------------------------------------------
# Imports (only reached when gate + deps are satisfied)
# ---------------------------------------------------------------------------

from torch_to_nnef.inference_target import TractNNEF

# torch-to-nnef 0.24 split the LLM exporter into the torch_to_nnef_llm
# distribution (the "llm-tract" extra); it was torch_to_nnef.llm_tract before.
from torch_to_nnef_llm.exporter import LLMExporter

from neuropress.adapter import WeightAssignmentAdapter
from neuropress.calibration.data import load_c4_calibration
from neuropress.estimators import GPTQEstimator
from neuropress.pipeline import QuantizationPipeline
from neuropress.recipe import QuantizationRecipe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_SLUG = "google/gemma-3-270m"
SEQ_LEN = 512  # standard LLM calibration sequence length
N_SAMPLES = 32  # small for speed; use 128+ for production quality
BATCH_SIZE = 1
GROUP_SIZE = 128  # per-group-128 is standard for 4-bit LLM quantization
N_BITS = 4


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_gptq_calibration_and_nnef_export(tmp_path: Any) -> None:
    """Full workflow: load → C4 calibration → GPTQ → NNEF export."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- 1. Load model via torch_to_nnef LLM exporter ----------------------
    # Force float32: Gemma-3 loads as bfloat16, but the exporter builds its
    # past-KV placeholders in the default float32, and Gemma-3's eager attention
    # matmuls query against the cached keys without reconciling dtypes.  NNEF /
    # tract targets float32 anyway, so pin the whole path to it.
    exporter = LLMExporter.load(model_slug=MODEL_SLUG)
    model = exporter.hf_model_causal.to(device=device, dtype=torch.float32).eval()
    tokenizer = exporter.tokenizer

    # -- 2. Build calibration data as a lazy generator (device transfer
    #        happens one batch at a time — no OOM on large models) ----------
    batches = (
        {k: v.to(device) for k, v in b.items()}
        for b in load_c4_calibration(
            tokenizer,
            seq_len=SEQ_LEN,
            n_samples=N_SAMPLES,
            batch_size=BATCH_SIZE,
            seed=42,
        )
    )

    # -- 3. Apply GPTQ to all linear layers via QuantizationPipeline --------
    # GPTQEstimator.prepare() is called automatically by the pipeline;
    # it discovers nn.Linear modules and the device from the model.
    # weight_transform=None → GPTQ corrects weights in-place (no plan entries);
    # the NNEF exporter handles subsequent re-quantization.
    recipe = QuantizationRecipe(
        param_estimator=GPTQEstimator(group_size=GROUP_SIZE, nbits=N_BITS),
        adapter=WeightAssignmentAdapter(),
    )
    QuantizationPipeline(recipe).run(model, batches)

    # -- 4. Export the quantized model to NNEF ------------------------------
    # export_model asserts the directory does NOT exist yet (unless
    # ignore_already_exist_dir=True), so hand it a path and let it create one.
    export_dir = tmp_path / "gemma3_270m_gptq_nnef"

    # prepare() wraps the model for deterministic IO (required for export).
    # We pass wrapper_io_check=False to skip the tolerance check after
    # quantization (the outputs will differ from fp32 by design).
    exporter.prepare(compression_method=None, wrapper_io_check=False)

    exporter.export_model(
        export_dirpath=export_dir,
        inference_target=TractNNEF.latest(),
        # export_model asserts this is >= 2; it controls how many generation
        # steps get dumped as npz IO fixtures beside the archive.
        sample_generation_total_size=2,
    )

    # -- 5. Sanity checks ---------------------------------------------------
    # torch-to-nnef 0.24 writes into export_dirpath and returns None, so check
    # the artifacts on disk rather than a returned path.
    assert export_dir.exists(), f"export directory not created at {export_dir}"
    produced = sorted(p for p in export_dir.rglob("*") if p.is_file())
    assert produced, f"export produced no files under {export_dir}"

    archives = [p for p in produced if ".nnef" in p.name]
    assert archives, f"no NNEF archive produced; got {[p.name for p in produced][:20]}"
    for archive in archives:
        assert archive.stat().st_size > 0, f"NNEF archive is empty: {archive}"
