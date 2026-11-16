"""Fine-tune the codebooks of a saved VQ checkpoint against the bf16 model.

    python scripts/vq_finetune_llm.py meta-llama/Llama-3.1-8B \
        runs/my_vq/quantized_model.safetensors \
        runs/my_vq_ft/quantized_model.safetensors \
        --calibration eval_targeted --n-samples 512 --lr 1e-4

The output checkpoint has the same format and bit-rate as the input; evaluate
it with ``neuropress-eval compare ... --save-model <output>`` (resume path).
"""

from __future__ import annotations

import argparse
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuropress.calibration.data.llm import (
    load_c4_calibration,
    load_eval_targeted_calibration,
)
from neuropress.methods.vq_finetune import (
    finetune_codebooks,
    kd_loss,
    prepare_vq_finetune,
    write_back_codebooks,
)
from neuropress.t2n import load_quantized_model, save_quantized_model

LOGGER = logging.getLogger("vq_finetune_llm")


def _batches(
    tokenizer, name: str, n_samples: int, seq_len: int, batch_size: int, seed: int
):
    loader = {
        "c4": load_c4_calibration,
        "eval_targeted": load_eval_targeted_calibration,
    }[name]
    rows = [
        b["input_ids"]
        for b in loader(tokenizer, seq_len=seq_len, n_samples=n_samples, seed=seed)
    ]
    ids = torch.cat(rows)
    return [ids[i : i + batch_size].cuda() for i in range(0, len(ids), batch_size)]


@torch.no_grad()
def _heldout_kd(student, teacher, batches) -> float:
    return sum(
        kd_loss(student(input_ids=b).logits, teacher(input_ids=b).logits).item()
        for b in batches
    ) / len(batches)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    ap.add_argument(
        "--calibration", choices=("eval_targeted", "c4"), default="eval_targeted"
    )
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="data seed (differs from calibration's 42)",
    )
    ap.add_argument("--heldout-batches", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    teacher = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    student = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).cuda()
    load_quantized_model(student, args.checkpoint, kernel_lowering=None)
    student.cuda()
    groups = prepare_vq_finetune(student, device="cuda")
    student.config.use_cache = False
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    train = _batches(
        tok, args.calibration, args.n_samples, args.seq_len, args.batch_size, args.seed
    )
    heldout = _batches(
        tok,
        "c4",
        args.heldout_batches * args.batch_size,
        args.seq_len,
        args.batch_size,
        args.seed + 1,
    )
    student.eval()
    before = _heldout_kd(student, teacher, heldout)
    LOGGER.info("held-out c4 KD before: %.5f", before)

    student.train()  # activates gradient checkpointing; Llama has no dropout
    t0 = time.time()
    finetune_codebooks(
        groups,
        lambda b: student(input_ids=b).logits,
        lambda b: teacher(input_ids=b).logits,
        train,
        lr=args.lr,
        epochs=args.epochs,
    )
    student.eval()
    after = _heldout_kd(student, teacher, heldout)
    LOGGER.info(
        "held-out c4 KD after: %.5f (before %.5f, %.1f%%), training %.1f min",
        after,
        before,
        100 * (after - before) / before,
        (time.time() - t0) / 60,
    )

    student.gradient_checkpointing_disable()
    write_back_codebooks(student, groups)
    save_quantized_model(student, args.output)
    LOGGER.info("saved %s", args.output)


if __name__ == "__main__":
    main()
