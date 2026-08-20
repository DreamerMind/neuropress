#!/usr/bin/env python3
"""Smoke test: random VQ assignment on Gemma 4 E2B → adapter apply → save.

Verifies the full save pipeline without running actual quantization.
Loads the real model, assigns random QTensorTractVQ to every eligible
layer (linear + embedding, matching production config), runs adapter
apply, then save_quantized_model.  Any remaining OOM sites will surface.

Target: <5 min on RTX 4090 (download + assign + save).

Usage:
    uv run python scripts/test_save.py
    uv run python scripts/test_save.py --no-embeddings   # linear only
    uv run python scripts/test_save.py --nc 256          # smaller codebook
    uv run python scripts/test_save.py --out /tmp/my.safetensors
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", flush=True)
    sys.exit(1)


def _step(msg: str) -> None:
    print(f"\n--- {msg}", flush=True)


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(0)
    return (total - free) / 2**30


def _make_vq(
    n_vecs: int,
    nc: int,
    vl: int,
    dev: torch.device,
) -> object:
    from neuropress.transforms.vq import ProxySelector, QTensorTractVQ, VQScheme

    centroids = torch.randn(nc, vl, dtype=torch.float16, device=dev)
    scheme = VQScheme(centroids=centroids, proxy_selector=ProxySelector.L2)
    indices = torch.randint(0, nc, (n_vecs,), dtype=torch.int64, device=dev)
    return QTensorTractVQ.from_packed(
        indices=indices, vq=scheme, dequant_to_dtype=torch.float16
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save pipeline smoke test (Gemma 4 E2B)"
    )
    parser.add_argument(
        "--model",
        default="google/gemma-4-E2B",
        help="HuggingFace model ID (default: google/gemma-4-E2B).",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=32768,
        help="Codebook size (default: 32768, matching production).",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Quantize linear layers only (skip nn.Embedding).",
    )
    parser.add_argument(
        "--out",
        default="/tmp/test_save.safetensors",
        help="Output path (default: /tmp/test_save.safetensors).",
    )
    args = parser.parse_args()

    nc = args.nc
    vl = 4
    out = Path(args.out)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model  : {args.model}")
    print(f"Device : {dev}")
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}  ({props.total_memory / 2**30:.1f} GB)")
    print(
        f"NC     : {nc}  VL={vl}  embeddings={'yes' if not args.no_embeddings else 'no'}"
    )

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    _step(f"Loading {args.model}")
    t0 = time.perf_counter()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    _ok(
        f"loaded in {time.perf_counter() - t0:.1f}s  "
        f"({n_params / 1e9:.2f}B params  VRAM used: {_vram_gb():.1f} GB)"
    )

    # ------------------------------------------------------------------
    # 2. Build random VQ assignments
    # ------------------------------------------------------------------
    _step("Building random QTensorTractVQ assignments")
    import torch.nn as nn

    assignments: dict = {}

    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is None or w.dim() != 2:
            continue

        is_linear = isinstance(mod, nn.Linear)
        is_embedding = isinstance(mod, nn.Embedding)

        if not (is_linear or (is_embedding and not args.no_embeddings)):
            continue

        out_f, in_f = w.shape
        if in_f % vl != 0:
            continue

        n_vecs = out_f * (in_f // vl)
        assignments[name] = _make_vq(n_vecs, nc, vl, dev)

    if not assignments:
        _fail("No eligible layers found.")

    n_linear = sum(
        1
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n in assignments
    )
    n_embedding = sum(
        1
        for n, m in model.named_modules()
        if isinstance(m, nn.Embedding) and n in assignments
    )
    _ok(
        f"{len(assignments)} layers assigned  "
        f"({n_linear} linear, {n_embedding} embedding)  "
        f"VRAM: {_vram_gb():.1f} GB"
    )

    # ------------------------------------------------------------------
    # 3. Adapter apply
    # ------------------------------------------------------------------
    _step("WeightAssignmentAdapter.apply()")
    from neuropress.adapter import WeightAssignmentAdapter
    from neuropress.plan import QuantizationPlan

    plan = QuantizationPlan(weight_assignments=assignments)
    t0 = time.perf_counter()
    try:
        WeightAssignmentAdapter().apply(model, plan)
    except Exception as exc:
        _fail(f"adapter raised: {type(exc).__name__}: {exc}")
    _ok(f"adapter done in {time.perf_counter() - t0:.2f}s  VRAM: {_vram_gb():.1f} GB")

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    _step(f"save_quantized_model → {out}")
    from neuropress.t2n.model_io import save_quantized_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t0 = time.perf_counter()
    try:
        save_quantized_model(model, out)
    except Exception as exc:
        _fail(f"save raised: {type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - t0
    size_mb = out.stat().st_size / 2**20
    _ok(f"saved in {elapsed:.1f}s  ({size_mb:.0f} MB  at {size_mb / elapsed:.0f} MB/s)")

    # ------------------------------------------------------------------
    # 5. Verify
    # ------------------------------------------------------------------
    _step("Load back + verify")
    from safetensors import safe_open

    with safe_open(str(out), framework="pt", device="cpu") as f:
        keys = list(f.keys())

    vq_blobs = [k for k in keys if k.endswith("::u8_blob")]
    vq_cents = [k for k in keys if k.endswith("::centroids")]
    plain_keys = [k for k in keys if "::" not in k]

    _ok(
        f"{len(keys)} total keys: "
        f"{len(vq_blobs)} ::u8_blob, "
        f"{len(vq_cents)} ::centroids, "
        f"{len(plain_keys)} plain tensors"
    )

    if len(vq_blobs) == 0:
        _fail("No VQ keys found — QTensorTractVQ was not serialised.")
    if len(vq_blobs) != len(assignments):
        print(f"  [WARN] expected {len(assignments)} VQ blobs, got {len(vq_blobs)}")

    print(f"\n{'=' * 60}")
    print("  PASS — save pipeline works end-to-end.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
