"""Export NeuroPress ternary models for the ``prism_ternary`` vLLM plugin."""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn

from ..transforms.ternary_hadamard import (
    TernaryHadamardInputLinear,
    is_prism_ternary_source_module,
)

PRISM_TERNARY_METHOD = "prism_ternary"
DEFAULT_GROUP_SIZE = 128
DEFAULT_HADAMARD_BLOCK = 1024
PACKED_LANES = 16
# The plugin's Triton layout is organized in 32-row blocks, but its compiled
# decode kernel requires the total output width to be divisible by 64.
VLLM_DECODE_ROW_BLOCK = 64

_AUXILIARY_FILES = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def pack_signs(signs: torch.Tensor) -> str:
    """Encode signs using the same MSB-first bit layout as ``numpy.packbits``."""
    flat_signs = signs.detach().reshape(-1).cpu()
    if not torch.isfinite(flat_signs).all() or not torch.all(flat_signs.abs() == 1):
        raise ValueError("Hadamard signs must contain only -1 and +1")
    negative = (flat_signs < 0).to(torch.uint8)
    padding = (-negative.numel()) % 8
    if padding:
        negative = torch.cat([negative, torch.zeros(padding, dtype=torch.uint8)])
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.int16)
    packed = (negative.view(-1, 8).to(torch.int16) * weights).sum(dim=1).to(torch.uint8)
    return base64.b64encode(bytes(packed.tolist())).decode("ascii")


def pack_ternary_weight(
    weight: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    row_chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a strict-ternary matrix using bounded host-memory row chunks."""
    if weight.dim() != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(weight.shape)}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    rows, width = weight.shape
    if width % group_size or width % PACKED_LANES:
        raise ValueError(
            f"input width {width} must be divisible by group_size={group_size} "
            f"and packed lanes={PACKED_LANES}"
        )
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")

    words = torch.empty((rows, width // PACKED_LANES), dtype=torch.int32)
    packed_scales = torch.empty((rows, width // group_size), dtype=torch.float16)
    shifts = torch.arange(PACKED_LANES, dtype=torch.int64) * 2
    for start in range(0, rows, row_chunk_size):
        stop = min(start + row_chunk_size, rows)
        chunk = weight[start:stop].detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(chunk).all():
            raise ValueError(
                f"weight contains non-finite values in rows [{start}, {stop})"
            )
        grouped = chunk.reshape(stop - start, -1, group_size)
        scales = grouped.abs().amax(dim=-1)
        safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        levels = (grouped / safe_scales[..., None]).round().clamp(-1, 1)
        levels = torch.where(scales[..., None] == 0, torch.zeros_like(levels), levels)
        reconstructed = levels * scales[..., None]
        tolerance = 1e-5 + scales[..., None] * 1e-3
        if torch.any((grouped - reconstructed).abs() > tolerance):
            max_error = (grouped - reconstructed).abs().max().item()
            raise ValueError(
                "weight is not strict ternary per group; run ternary_gptq before "
                f"export (max reconstruction error {max_error:.6g})"
            )
        scales_f16 = scales.to(torch.float16)
        if not torch.isfinite(scales_f16).all():
            raise ValueError("weight scales overflow float16 during export")
        codes = (levels.reshape(stop - start, width).to(torch.int64) + 1).reshape(
            stop - start, -1, PACKED_LANES
        )
        words[start:stop] = torch.sum(codes << shifts, dim=-1).to(torch.int32)
        packed_scales[start:stop] = scales_f16
    return words.contiguous(), packed_scales.contiguous()


def unpack_ternary_weight(
    words: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode a packed matrix; useful for format verification and tests."""
    rows = words.shape[0]
    shifts = torch.arange(PACKED_LANES, device=words.device, dtype=torch.int64) * 2
    codes = ((words.to(torch.int64)[..., None] >> shifts) & 3).reshape(rows, -1)
    if torch.any(codes == 3):
        raise ValueError("packed tensor contains reserved ternary code 3")
    values = codes.to(dtype) - 1
    return (values.reshape(rows, -1, group_size) * scales.to(dtype)[..., None]).reshape(
        rows, -1
    )


def _load_config(
    config: dict[str, Any] | str | Path,
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(config, dict):
        return dict(config), None
    path = Path(config)
    config_path = path / "config.json" if path.is_dir() else path
    return json.loads(config_path.read_text()), config_path.parent


def export_prism_ternary_checkpoint(
    model: nn.Module,
    output_dir: str | Path,
    base_config: dict[str, Any] | str | Path,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    hadamard_block: int = DEFAULT_HADAMARD_BLOCK,
    max_shard_bytes: int = 5 * 1024**3,
    packing_row_chunk_size: int = 256,
) -> Path:
    """Write a text-only Hugging Face checkpoint consumable by ``bonsai-vllm``.

    The model must first pass through ``ternary_hadamard`` preprocessing and
    ``ternary_gptq`` estimation. Vision-tower tensors are omitted because the
    current plugin contract is only verified with vLLM's
    ``--language-model-only`` mode. Existing checkpoint files are never
    replaced.
    """
    destination = Path(output_dir)
    if group_size != DEFAULT_GROUP_SIZE or hadamard_block != DEFAULT_HADAMARD_BLOCK:
        raise ValueError(
            "the prism_ternary vLLM plugin requires group_size=128 and "
            "hadamard_block=1024"
        )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory is not empty: {destination}")
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    if packing_row_chunk_size <= 0:
        raise ValueError("packing_row_chunk_size must be positive")

    config, source_dir = _load_config(base_config)
    if source_dir is not None and source_dir.resolve() == destination.resolve():
        raise ValueError("output directory must differ from the base checkpoint")

    wrappers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, TernaryHadamardInputLinear)
    }
    if not wrappers:
        raise ValueError("model has no TernaryHadamardInputLinear modules")
    unsupported_wrappers = [
        name for name in wrappers if not is_prism_ternary_source_module(name)
    ]
    if unsupported_wrappers:
        raise ValueError(
            "the prism_ternary plugin does not select these wrapped modules: "
            + ", ".join(unsupported_wrappers)
        )
    unwrapped_targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and is_prism_ternary_source_module(name)
    ]
    if unwrapped_targets:
        raise ValueError(
            "plugin-targeted linear modules were not Hadamard wrapped: "
            + ", ".join(unwrapped_targets)
        )
    signs_by_width: dict[int, torch.Tensor] = {}
    for name, wrapper in wrappers.items():
        if wrapper.block_size != hadamard_block:
            raise ValueError(
                f"{name} uses Hadamard block {wrapper.block_size}, expected {hadamard_block}"
            )
        width = wrapper.linear.in_features
        if wrapper.linear.out_features % VLLM_DECODE_ROW_BLOCK:
            raise ValueError(
                f"{name} output width {wrapper.linear.out_features} must be divisible "
                f"by vLLM decode row block {VLLM_DECODE_ROW_BLOCK}"
            )
        previous = signs_by_width.setdefault(
            width, wrapper.hadamard_signs.detach().cpu()
        )
        if not torch.equal(previous, wrapper.hadamard_signs.detach().cpu()):
            raise ValueError(f"layers with input width {width} do not share signs")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    wrapper_prefixes = sorted(wrappers, key=len, reverse=True)
    state = model.state_dict()
    shards: list[tuple[Path, list[str]]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    total_tensor_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        path = staging / f"model-{len(shards) + 1:05d}.safetensors"
        save_file(current, path, metadata={"format": "pt"})
        shards.append((path, list(current)))
        current = {}
        current_bytes = 0

    try:
        for key, tensor in state.items():
            if key.startswith("model.visual."):
                continue
            output_name = key
            output_tensors: list[tuple[str, torch.Tensor]] = []
            wrapper_name = next(
                (name for name in wrapper_prefixes if key.startswith(f"{name}.")), None
            )
            if wrapper_name is not None:
                relative = key[len(wrapper_name) + 1 :]
                if relative == "hadamard_signs":
                    continue
                if relative == "linear.weight":
                    words, scales = pack_ternary_weight(
                        tensor, group_size, packing_row_chunk_size
                    )
                    output_tensors = [
                        (f"{wrapper_name}.weight", words),
                        (f"{wrapper_name}.scales", scales),
                    ]
                elif relative.startswith("linear."):
                    output_tensors = [
                        (f"{wrapper_name}.{relative.removeprefix('linear.')}", tensor)
                    ]
                else:
                    continue
            else:
                output_tensors = [(output_name, tensor)]

            for name, value in output_tensors:
                stored = value.detach().cpu().contiguous().clone()
                nbytes = stored.numel() * stored.element_size()
                if current and current_bytes + nbytes > max_shard_bytes:
                    flush()
                current[name] = stored
                current_bytes += nbytes
                total_tensor_bytes += nbytes
        flush()

        weight_map: dict[str, str] = {}
        if len(shards) == 1:
            source, names = shards[0]
            target = staging / "model.safetensors"
            source.rename(target)
            weight_map = dict.fromkeys(names, target.name)
        else:
            total = len(shards)
            for index, (source, names) in enumerate(shards, 1):
                target = staging / f"model-{index:05d}-of-{total:05d}.safetensors"
                source.rename(target)
                weight_map.update(dict.fromkeys(names, target.name))
            index = {
                "metadata": {"total_size": total_tensor_bytes},
                "weight_map": weight_map,
            }
            (staging / "model.safetensors.index.json").write_text(
                json.dumps(index, indent=2) + "\n"
            )

        config["quantization_config"] = {
            "quant_method": PRISM_TERNARY_METHOD,
            "bits": 2,
            "group_size": group_size,
            "hadamard_block": hadamard_block,
            "signs": {
                str(width): pack_signs(signs)
                for width, signs in sorted(signs_by_width.items())
            },
        }
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if source_dir is not None:
            for filename in _AUXILIARY_FILES:
                source = source_dir / filename
                if source.exists():
                    shutil.copyfile(source, staging / filename)
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
