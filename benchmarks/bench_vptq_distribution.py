r"""Benchmark: measure VQ assignment distribution on a quantized model.

Loads a model that has been quantized with VPTQGroupEstimator (i.e. has
QTensorTractVQ weights), measures per-layer assignment counts, fits Zipf s,
and prints the distribution report.

Usage::

    # Quantize first (takes a few minutes):
    uv run python -c "
    from neuropress.t2n.registry import NEUROPRESS_COMPRESSION
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained('google/gemma-3-270m-it')
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-270m-it')
    fn = NEUROPRESS_COMPRESSION['open_text_c4_vptq_256c_v4_256samples']
    fn(model, tokenizer=tokenizer)
    from neuropress.t2n.model_io import save_quantized_model
    save_quantized_model(model, 'tmp_vptq_model')
    "

    # Then analyze:
    uv run python benchmarks/bench_vptq_distribution.py tmp_vptq_model

Or pipe through an in-memory quantization (no save/load):

    uv run python benchmarks/bench_vptq_distribution.py --inline \\
        --model google/gemma-3-270m-it \\
        --compression open_text_c4_vptq_256c_v4_256samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure VQ codebook assignment distribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "model_dir",
        nargs="?",
        type=Path,
        help="Path to a saved quantized model directory (from save_quantized_model).",
    )
    p.add_argument(
        "--inline",
        action="store_true",
        help=(
            "Quantize in-memory (no save/load).  Requires --model and --compression."
        ),
    )
    p.add_argument(
        "--model",
        default="google/gemma-3-270m-it",
        metavar="HF_ID",
        help=(
            "HuggingFace model ID.  Used with --inline, and to build the "
            "architecture a saved checkpoint is loaded into."
        ),
    )
    p.add_argument(
        "--compression",
        default="open_text_c4_vptq_256c_v4_256samples",
        metavar="KEY",
        help="NEUROPRESS_COMPRESSION registry key (used with --inline).",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=30,
        metavar="N",
        help="Max number of layers to show in the report (default: 30).",
    )
    p.add_argument(
        "--theoretical",
        action="store_true",
        help="Also print theoretical bpw table for several s values.",
    )
    return p


def _load_model(model_dir: Path, model_id: str) -> object:
    """Instantiate *model_id*'s architecture and load the checkpoint at *model_dir*.

    ``load_quantized_model`` restores state into an existing module and returns the
    manifest, not a model, so the architecture has to be built first.
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("ERROR: transformers not installed.  Run: uv sync --extra llm")
        sys.exit(1)

    from neuropress.t2n.model_io import load_quantized_model

    print(f"Building {model_id} architecture …")
    model = AutoModelForCausalLM.from_pretrained(model_id)

    print(f"Loading quantized weights from {model_dir} …")
    load_quantized_model(model, model_dir)
    return model


def _inline_quantize(model_id: str, compression_key: str) -> object:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed.  Run: uv sync --extra llm")
        sys.exit(1)

    from neuropress.t2n.registry import NEUROPRESS_COMPRESSION

    if compression_key not in NEUROPRESS_COMPRESSION:
        available = ", ".join(sorted(NEUROPRESS_COMPRESSION))
        print(f"ERROR: unknown compression key {compression_key!r}")
        print(f"Available: {available}")
        sys.exit(1)

    print(f"Loading {model_id} …")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Quantizing with {compression_key!r} …")
    fn = NEUROPRESS_COMPRESSION[compression_key]
    fn(model, tokenizer=tokenizer)
    return model


def _print_theoretical_table() -> None:
    from neuropress.methods.vptq_analysis import theoretical_tiered_bpw

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title="Theoretical bpw under Zipf (NC=256, VL=4)", border_style="dim"
        )
        table.add_column("s", justify="right")
        table.add_column("fixed bpw", justify="right")
        table.add_column("2-tier bpw", justify="right", style="green")
        table.add_column("zero-escape bpw", justify="right", style="cyan")
        table.add_column("entropy bpw", justify="right", style="dim")
        table.add_column("2tier saving", justify="right", style="bold green")
        for s_val in [0.7, 0.9, 1.0, 1.2, 1.5, 2.0]:
            t = theoretical_tiered_bpw(s_val, 256, 4)
            saving = (t["fixed"] - t["2tier"]) / t["fixed"] * 100
            table.add_row(
                f"{s_val:.1f}",
                f"{t['fixed']:.3f}",
                f"{t['2tier']:.3f}",
                f"{t['2tier_zero_escape']:.3f}",
                f"{t['entropy']:.3f}",
                f"{saving:.1f}%",
            )
        console.print(table)
    except ImportError:
        print(f"\n{'s':>5} {'fixed':>8} {'2tier':>8} {'entropy':>8} {'saving':>8}")
        print("-" * 45)
        for s_val in [0.7, 0.9, 1.0, 1.2, 1.5, 2.0]:
            t = theoretical_tiered_bpw(s_val, 256, 4)
            saving = (t["fixed"] - t["2tier"]) / t["fixed"] * 100
            print(
                f"{s_val:>5.1f} {t['fixed']:>8.3f} {t['2tier']:>8.3f} "
                f"{t['entropy']:>8.3f} {saving:>7.1f}%"
            )


def main() -> None:
    """Run the VPTQ index-distribution benchmark from the command line."""
    args = _build_parser().parse_args()

    if args.inline:
        model = _inline_quantize(args.model, args.compression)
    elif args.model_dir is not None:
        model = _load_model(args.model_dir, args.model)
    else:
        _build_parser().print_help()
        sys.exit(1)

    from neuropress.methods.vptq_analysis import (
        measure_assignment_distribution,
        print_distribution_report,
    )

    print("\nMeasuring assignment distributions …")
    stats = measure_assignment_distribution(model)

    if not stats:
        print("No QTensorTractVQ layers found in the model.")
        sys.exit(1)

    print(f"Found {len(stats)} VQ layers.\n")
    print_distribution_report(stats, top_n=args.top_n)

    if args.theoretical:
        _print_theoretical_table()


if __name__ == "__main__":
    main()
