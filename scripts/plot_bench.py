"""Generate benchmark analysis plots from neuropress-bench JSON results."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> tuple[dict, list[dict]]:
    with open(path) as f:
        d = json.load(f)
    return d["meta"], d["results"]


def plot_backend_comparison_by_shape(results: list[dict], meta: dict, out_dir: Path):
    """Bar chart: median_ms per backend, grouped by shape, for Q4_0 linear bs=1."""
    filtered = [
        r
        for r in results
        if r["module_type"] == "NPLinear (Q4_0 sym)" and r["batch"] == 1
    ]
    if not filtered:
        return

    shapes = []
    seen = set()
    for r in filtered:
        if r["shape"] not in seen:
            shapes.append(r["shape"])
            seen.add(r["shape"])

    backends = []
    seen_b = set()
    for r in filtered:
        if r["backend"] not in seen_b:
            backends.append(r["backend"])
            seen_b.add(r["backend"])

    data = {}
    for r in filtered:
        data[(r["shape"], r["backend"])] = r["median_ms"]

    x = np.arange(len(shapes))
    width = 0.8 / len(backends)

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, b in enumerate(backends):
        vals = [data.get((s, b), 0) for s in shapes]
        ax.bar(x + i * width, vals, width, label=b)

    ax.set_xlabel("Weight shape (outxin)")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title(
        f"NPLinear Q4_0 sym — Backend comparison (batch=1)\n{meta.get('gpu_name', '?')}"
    )
    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels(shapes, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "backend_comparison_q4_bs1.png", dpi=150)
    plt.close(fig)
    print("  saved backend_comparison_q4_bs1.png")


def plot_batch_scaling(results: list[dict], meta: dict, out_dir: Path):
    """Line chart: latency vs batch size for each backend, fixed shape 4096x4096."""
    target_shape = "4096x4096"
    filtered = [
        r
        for r in results
        if r["module_type"] == "NPLinear (Q4_0 sym)" and r["shape"] == target_shape
    ]
    if not filtered:
        return

    backends = sorted({r["backend"] for r in filtered})
    batches = sorted({r["batch"] for r in filtered})

    fig, ax = plt.subplots(figsize=(10, 6))
    for b in backends:
        vals = [
            next(
                (
                    r["median_ms"]
                    for r in filtered
                    if r["backend"] == b and r["batch"] == bs
                ),
                None,
            )
            for bs in batches
        ]
        ax.plot(batches, vals, "o-", label=b, linewidth=2, markersize=6)

    ax.set_xlabel("Batch size")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title(
        f"NPLinear Q4_0 sym {target_shape} — Batch scaling\n{meta.get('gpu_name', '?')}"
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "batch_scaling_q4_4096.png", dpi=150)
    plt.close(fig)
    print("  saved batch_scaling_q4_4096.png")


def plot_quant_method_comparison(results: list[dict], meta: dict, out_dir: Path):
    """Grouped bar: best backend latency per quant method, shape 4096x4096, bs=1."""
    target_shape = "4096x4096"
    filtered = [
        r
        for r in results
        if r["shape"] == target_shape
        and r["batch"] == 1
        and "NPLinear" in r["module_type"]
    ]
    if not filtered:
        return

    methods = []
    seen = set()
    for r in filtered:
        if r["module_type"] not in seen:
            methods.append(r["module_type"])
            seen.add(r["module_type"])

    backends = sorted({r["backend"] for r in filtered})

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.8 / len(backends)

    for i, b in enumerate(backends):
        vals = []
        for m in methods:
            v = next(
                (
                    r["median_ms"]
                    for r in filtered
                    if r["module_type"] == m and r["backend"] == b
                ),
                0,
            )
            vals.append(v)
        ax.bar(x + i * width, vals, width, label=b)

    ax.set_xlabel("Quantization method")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title(
        f"All quant methods — {target_shape} batch=1\n{meta.get('gpu_name', '?')}"
    )
    ax.set_xticks(x + width * (len(backends) - 1) / 2)
    ax.set_xticklabels(
        [m.replace("NPLinear ", "") for m in methods],
        rotation=30,
        ha="right",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "quant_method_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved quant_method_comparison.png")


def plot_speedup_vs_default(results: list[dict], meta: dict, out_dir: Path):
    """Heatmap: speedup of each backend vs default(f32), for Q4_0 bs=1."""
    filtered = [
        r
        for r in results
        if r["module_type"] == "NPLinear (Q4_0 sym)" and r["batch"] == 1
    ]
    if not filtered:
        return

    shapes = []
    seen = set()
    for r in filtered:
        if r["shape"] not in seen:
            shapes.append(r["shape"])
            seen.add(r["shape"])

    backends = [
        b for b in sorted({r["backend"] for r in filtered}) if b != "default(f32)"
    ]
    baseline = {
        r["shape"]: r["median_ms"] for r in filtered if r["backend"] == "default(f32)"
    }

    data = np.zeros((len(backends), len(shapes)))
    for i, b in enumerate(backends):
        for j, s in enumerate(shapes):
            val = next(
                (
                    r["median_ms"]
                    for r in filtered
                    if r["backend"] == b and r["shape"] == s
                ),
                None,
            )
            if val and baseline.get(s):
                data[i, j] = baseline[s] / val
            else:
                data[i, j] = float("nan")

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=4.0)
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shapes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(backends)))
    ax.set_yticklabels(backends, fontsize=9)
    ax.set_title(
        f"Speedup vs default(f32) — Q4_0 sym batch=1\n{meta.get('gpu_name', '?')}"
    )

    for i in range(len(backends)):
        for j in range(len(shapes)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.1f}x",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > 2.5 or v < 0.8 else "black",
                )

    fig.colorbar(im, ax=ax, label="Speedup factor")
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_heatmap.png", dpi=150)
    plt.close(fig)
    print("  saved speedup_heatmap.png")


def plot_vq_comparison(results: list[dict], meta: dict, out_dir: Path):
    """Bar chart: VQ fixed vs tiered, by shape, bs=1."""
    filtered = [
        r for r in results if "NPVQLinear" in r["module_type"] and r["batch"] == 1
    ]
    if not filtered:
        return

    shapes = []
    seen = set()
    for r in filtered:
        if r["shape"] not in seen:
            shapes.append(r["shape"])
            seen.add(r["shape"])

    # Group by (module_type, backend)
    combos = []
    seen_c = set()
    for r in filtered:
        key = (r["module_type"], r["backend"])
        if key not in seen_c:
            combos.append(key)
            seen_c.add(key)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(shapes))
    width = 0.8 / len(combos)

    for i, (mt, b) in enumerate(combos):
        vals = [
            next(
                (
                    r["median_ms"]
                    for r in filtered
                    if r["module_type"] == mt and r["backend"] == b and r["shape"] == s
                ),
                0,
            )
            for s in shapes
        ]
        short = mt.replace("NPVQLinear ", "") + " " + b
        ax.bar(x + i * width, vals, width, label=short)

    ax.set_xlabel("Weight shape")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title(f"NPVQLinear — Fixed vs Tiered (batch=1)\n{meta.get('gpu_name', '?')}")
    ax.set_xticks(x + width * (len(combos) - 1) / 2)
    ax.set_xticklabels(shapes, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "vq_comparison.png", dpi=150)
    plt.close(fig)
    print("  saved vq_comparison.png")


def main():
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/bench/vastai/2026_04_neuropress-bench/bench/bench_results.json"
    )
    meta, results = load_results(path)
    out_dir = Path(path).parent / "plots"
    out_dir.mkdir(exist_ok=True)

    print(
        f"GPU: {meta.get('gpu_name')} | CPU: {meta.get('cpu_model')} | {len(results)} results"
    )

    plot_backend_comparison_by_shape(results, meta, out_dir)
    plot_batch_scaling(results, meta, out_dir)
    plot_quant_method_comparison(results, meta, out_dir)
    plot_speedup_vs_default(results, meta, out_dir)
    plot_vq_comparison(results, meta, out_dir)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
