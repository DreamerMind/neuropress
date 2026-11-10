"""Compare float-statistics, sequential, and teacher-reconstruction GPTQ on Bonsai.

Requires the original Bonsai checkpoint, the deterministic prompts.json produced
by the published calibration script, transformers, and torch-to-nnef 0.25.0.
Held-out prompts are never used for calibration or parameter selection.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from neuropress.block_pipeline import BlockQuantizationPipeline
from neuropress.recipe_config import QuantizationRecipeConfig


class DecoderBlock(nn.Module):
    """One Qwen decoder block with Bonsai's causal/padding mask contract."""

    def __init__(self, block, cos, sin):
        super().__init__()
        self.block = block
        self.register_buffer("cos", cos.cpu())
        self.register_buffer("sin", sin.cpu())
        self.register_buffer("positions", torch.arange(cos.shape[1])[None])

    def forward(self, hidden_states, attention_mask):
        """Apply the block with its explicit causal and padding mask."""
        positions = self.positions
        causal = positions.t() >= positions
        allowed = causal[None, None] & attention_mask[:, None, None, :].bool()
        mask = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)
        result = self.block(
            hidden_states,
            attention_mask=mask,
            position_ids=positions,
            position_embeddings=(self.cos, self.sin),
        )
        return result[0] if isinstance(result, tuple) else result


def float_view(model):
    """Materialize final packed codes/scales in f32 for calibration forwards."""
    memo = {}
    for weight in model.parameters():
        if hasattr(weight, "decompress_to_u8"):
            value = weight.qscheme.dequantize(weight.decompress_to_u8(), torch.float32)
            memo[id(weight)] = nn.Parameter(value, requires_grad=False)
    return copy.deepcopy(model, memo)


def metrics(actual, reference, mask):
    """Report final-weight encoder error for all and non-padding tokens."""
    result = {}
    for name, a, r in [
        ("all_tokens", actual, reference),
        ("valid_tokens", actual[mask.bool()], reference[mask.bool()]),
    ]:
        delta = a.double() - r.double()
        result[name] = {
            "relative_l2": float(delta.norm() / r.double().norm()),
            "rmse": float(delta.square().mean().sqrt()),
        }
    return result


@torch.inference_mode()
def main():
    """Run the matched calibration experiment and save held-out measurements."""
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer,
        Qwen3RotaryEmbedding,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=27)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["float", "sequential", "teacher"],
        default=["float", "sequential", "teacher"],
    )
    parser.add_argument("--dependency-stages", action="store_true")
    args = parser.parse_args()
    if args.dependency_stages and "float" in args.modes:
        parser.error(
            "Dependency stages require --modes sequential teacher (or either mode)"
        )
    torch.set_num_threads(8)
    torch.manual_seed(3309)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.prompts.read_text())
    cfg = AutoConfig.from_pretrained(args.snapshot / "text_encoder")
    cfg._attn_implementation = "eager"
    shards = [
        safe_open(p, framework="pt")
        for p in sorted((args.snapshot / "text_encoder").glob("model-*.safetensors"))
    ]
    index = {key: shard for shard in shards for key in shard.keys()}

    def weight(key):
        return index[key].get_tensor(key).float()

    ids = {
        split: torch.tensor([row["input_ids"] for row in rows])
        for split, rows in data["splits"].items()
    }
    masks = {
        split: torch.tensor([row["attention_mask"] for row in rows])
        for split, rows in data["splits"].items()
    }
    embedding = weight("model.embed_tokens.weight")
    calibration = torch.nn.functional.embedding(ids["train"], embedding)
    evaluation = torch.nn.functional.embedding(ids["eval"], embedding)
    del embedding

    def batches(hidden, mask):
        return [
            {
                "hidden_states": hidden[start : start + args.batch_size],
                "attention_mask": mask[start : start + args.batch_size],
            }
            for start in range(0, len(hidden), args.batch_size)
        ]

    reference_batches = batches(calibration, masks["train"])
    students = dict.fromkeys(args.modes, reference_batches)
    evaluation_states = {mode: evaluation.clone() for mode in args.modes}
    reference_evaluation = evaluation
    positions = torch.arange(64, device=device)[None]
    rotary = Qwen3RotaryEmbedding(cfg).to(device)
    cos, sin = rotary(torch.zeros(1, 64, cfg.hidden_size, device=device), positions)
    runners = {}
    for mode in args.modes:
        recipe = QuantizationRecipeConfig.model_validate(
            {
                "calibration": {
                    "dataset": data["dataset"],
                    "n_samples": 256,
                    "seq_len": 64,
                    "batch_size": args.batch_size,
                    "seed": 3309,
                },
                "estimator": {
                    "type": "gptq",
                    "target": "linear",
                    "group_size": 32,
                    "nbits": 4,
                    "symmetric": True,
                    "weight_transform": "q4_0",
                    "reconstruction": "teacher" if mode == "teacher" else "self",
                    "reconstruction_damp_percent": 0.01,
                },
                "kernel_lowering": None,
            }
        )
        runners[mode] = BlockQuantizationPipeline(
            recipe,
            input_source="float" if mode == "float" else "sequential",
            forward_device=device,
            estimation_device="cpu",
            checkpoint_dir=args.out / mode,
            run_id="bonsai-30bbc032-explicit-f32-v2:" + args.prompts.read_text(),
            inference_model=float_view,
            target_paths=tuple(
                "block." + name
                for name in [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                    "mlp.down_proj",
                ]
            )
            if args.dependency_stages
            else None,
        )
        (args.out / f"{mode}-recipe.json").write_text(recipe.model_dump_json(indent=2))
    reports = []
    for i in range(args.layers):
        begin = time.monotonic()
        decoder = Qwen3DecoderLayer(cfg, i).eval().requires_grad_(False)
        decoder.load_state_dict(
            {name: weight(f"model.layers.{i}.{name}") for name in decoder.state_dict()}
        )
        original = DecoderBlock(decoder, cos, sin).eval()
        evaluator = runners[args.modes[0]]
        reference_evaluation = torch.cat(
            evaluator.forward(original, batches(reference_evaluation, masks["eval"]))
        )
        block_report = {"layer": i + 1}
        next_reference = None
        for mode, runner in runners.items():
            start = time.monotonic()
            result = runner.run_block(
                str(i), copy.deepcopy(original), students[mode], reference_batches
            )
            if result.kept_float:
                raise RuntimeError(
                    f"{mode}: unexpected float fallback: {result.kept_float}"
                )
            packed = {}
            for name, mod in result.model.block.named_modules():
                if isinstance(mod, nn.Linear):
                    q = mod.weight
                    packed[name] = {
                        "codes": q.decompress_to_u8().cpu(),
                        "scale": q.qscheme.scale.cpu(),
                        "shape": list(q.shape),
                    }
            torch.save(packed, args.out / mode / f"layer-{i:02d}.q4.pt")
            view = float_view(result.model)
            evaluation_states[mode] = torch.cat(
                runner.forward(view, batches(evaluation_states[mode], masks["eval"]))
            )
            students[mode] = [
                dict(batch, hidden_states=output)
                for batch, output in zip(
                    students[mode], result.student_outputs, strict=True
                )
            ]
            next_reference = [
                dict(batch, hidden_states=output)
                for batch, output in zip(
                    reference_batches, result.reference_outputs, strict=True
                )
            ]
            block_report[mode] = {
                "seconds": time.monotonic() - start,
                "resumed": result.resumed,
                "schedule": result.schedule.stages if result.schedule else None,
                **metrics(evaluation_states[mode], reference_evaluation, masks["eval"]),
            }
            print(
                json.dumps({"layer": i + 1, "mode": mode, **block_report[mode]}),
                flush=True,
            )
            del result, view, packed
            if device == "mps":
                torch.mps.empty_cache()
        reference_batches = next_reference
        if i in [8, 17, 26]:
            torch.save(
                {"reference": reference_evaluation, **evaluation_states},
                args.out / f"evaluation-{i + 1}.pt",
            )
        block_report["seconds"] = time.monotonic() - begin
        reports.append(block_report)
        (args.out / "layers.json").write_text(json.dumps(reports, indent=2))
        del original, decoder
    if args.layers == 27:
        pieces = [
            torch.load(args.out / f"evaluation-{i}.pt", weights_only=True)
            for i in [9, 18, 27]
        ]
        reference = torch.cat([piece["reference"] for piece in pieces], dim=-1)
        summary = {}
        for mode in args.modes:
            actual = torch.cat([piece[mode] for piece in pieces], dim=-1)
            summary[mode] = metrics(actual, reference, masks["eval"])
            summary[mode]["per_prompt_relative_l2"] = (
                (actual.double() - reference.double()).flatten(1).norm(dim=1)
                / reference.double().flatten(1).norm(dim=1)
            ).tolist()
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
