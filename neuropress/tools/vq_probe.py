"""CLI: probe VQ assignment distribution from a safetensors checkpoint.

Usage::

    neuropress-vq-probe model.safetensors
    neuropress-vq-probe model.safetensors --plot dist.png
    neuropress-vq-probe model.safetensors --top-n 40 --plot dist.pdf

The command reads a ``.safetensors`` checkpoint written by
:func:`~neuropress.t2n.model_io.save_quantized_model`, extracts every
``QTensorTractVQ`` parameter, and reports per-centroid assignment statistics.

Output
------
* A Rich table showing, per layer: codebook size, VL, fixed bpw, Zipf s/R²,
  top-16% hot fraction, empirical 2-tier bpw, entropy lower bound, and the
  *efficiency score* — what fraction of the achievable headroom 2-tier captures.
* A pooled summary line with the same metrics weighted by vector count.
* A recommendation string: flat / moderate / heavy skew.
* Optional matplotlib figure (``--plot PATH``) — saved as PNG/PDF/SVG based
  on the file extension.
"""

import sys


def main() -> int:
    """Entry-point for the ``neuropress-vq-probe`` CLI.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="neuropress-vq-probe",
        description=(
            "Probe VQ assignment distribution from a neuropress safetensors checkpoint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoint",
        metavar="CHECKPOINT",
        help="Path to a .safetensors file written by save_quantized_model().",
    )
    parser.add_argument(
        "--plot",
        metavar="PATH",
        default=None,
        help=(
            "Save the rank-frequency + efficiency figure to this path "
            "(PNG / PDF / SVG).  Requires matplotlib."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        metavar="N",
        help="Maximum number of layers to show in the table (default: 20).",
    )
    parser.add_argument(
        "--title",
        default=None,
        metavar="TEXT",
        help="Optional title for the plot.",
    )
    args = parser.parse_args()

    from pathlib import Path

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    from ..methods.vptq_analysis import (
        plot_distribution,
        print_distribution_report,
        probe_safetensors,
    )

    try:
        stats = probe_safetensors(str(ckpt_path))
    except Exception as exc:
        print(f"Error reading checkpoint: {exc}", file=sys.stderr)
        return 1

    if not stats:
        print(
            f"No QTensorTractVQ layers found in {ckpt_path}.\n"
            "Make sure the file was written by save_quantized_model() after "
            "VPTQ quantization.",
            file=sys.stderr,
        )
        return 1

    print_distribution_report(stats, top_n=args.top_n)

    if args.plot:
        plot_path = Path(args.plot)
        try:
            plot_distribution(stats, output_path=str(plot_path), title=args.title)
            print(f"\nPlot saved → {plot_path}")
        except ImportError as exc:
            print(f"Warning: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
