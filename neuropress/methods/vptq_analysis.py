"""VQ codebook assignment distribution analysis.

Measures whether the empirical assignment distribution follows a Zipf law,
estimates the Zipf exponent ``s``, and reports per-layer statistics that
inform tier boundary choice for the tiered index coding scheme.

The Zipf model: ``P(rank=k) ∝ 1/k^s``.

- ``s < 0.9`` — flat distribution; fixed-width packing is near-optimal.
- ``s ∈ [0.9, 1.3]`` — moderate skew; 2-tier escape saves ~25–35%.
- ``s > 1.3`` — heavy skew; 2-tier (or more) saves 35–50%+.

Usage::

    from neuropress.methods.vptq_analysis import measure_assignment_distribution

    stats = measure_assignment_distribution(model)
    print_distribution_report(stats)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class AssignmentStats:
    """Per-layer VQ assignment distribution statistics.

    Attributes:
        layer_name: Name of the ``nn.Linear`` (or other) module.
        num_centroids: Codebook size ``NC``.
        vector_len: VQ vector length ``VL``.
        num_vectors: Total number of VQ vectors (``rows × (K // VL)``).
        counts: Assignment count per centroid, sorted descending ``[NC]``.
        zipf_s: Fitted Zipf exponent (slope of log-rank vs log-count).
        zipf_r2: R² of the log-log linear fit (1.0 = perfect Zipf).
        top1_fraction: Fraction of all assignments taken by centroid 0.
        top16_fraction: Fraction taken by the top-16 centroids.
        bits_per_index: Current storage bits per index (4 or 8).
        effective_bpw: Current bits-per-weight = ``bits_per_index / vector_len``.
        entropy_bpi: Empirical Shannon entropy of the index distribution
            (bits per index).  Lower bound on lossless index coding.
        tiered_bpi: Effective bits per index under the 2-tier prefix coding
            scheme at the empirical hot fraction.
    """

    layer_name: str
    num_centroids: int
    vector_len: int
    num_vectors: int
    counts: torch.Tensor  # [NC] sorted descending
    zipf_s: float
    zipf_r2: float
    top1_fraction: float
    top16_fraction: float
    bits_per_index: int
    effective_bpw: float
    entropy_bpi: float = 0.0
    tiered_bpi: float = 0.0

    def tiered_efficiency(self) -> float:
        """Fraction of entropy headroom captured by 2-tier coding.

        Computed as ``(fixed_bpi - tiered_bpi) / (fixed_bpi - entropy_bpi)``.
        A value of 1.0 means 2-tier coding achieves the entropy lower bound.
        A value of 0.0 means the distribution is uniform (no savings possible).

        Returns:
            Float in ``[0, 1]``.
        """
        headroom = self.bits_per_index - self.entropy_bpi
        savings = self.bits_per_index - self.tiered_bpi
        if headroom < 1e-9:
            return 1.0
        return max(0.0, min(1.0, savings / headroom))


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------


def _fit_zipf(counts_sorted: torch.Tensor) -> tuple[float, float]:
    """Fit Zipf exponent via log-log linear regression.

    Regresses ``log(count)`` on ``log(rank)`` using only entries with
    positive counts.  Returns ``(s, r2)`` where ``s`` is the negative slope
    and ``r2`` is the coefficient of determination.

    Args:
        counts_sorted: Assignment counts sorted **descending** ``[NC]``.

    Returns:
        ``(s, r2)`` — Zipf exponent and goodness-of-fit.
    """
    positive = counts_sorted[counts_sorted > 0].float()
    n = positive.numel()
    if n < 2:
        return 0.0, 0.0

    # ranks are 1-indexed; log-log regression: log(count) = a - s * log(rank)
    log_rank = torch.log(torch.arange(1, n + 1, dtype=torch.float32))
    log_count = torch.log(positive)

    # OLS: s = -cov(log_rank, log_count) / var(log_rank)
    lr_mean = log_rank.mean()
    lc_mean = log_count.mean()
    cov = ((log_rank - lr_mean) * (log_count - lc_mean)).mean()
    var_rank = ((log_rank - lr_mean) ** 2).mean()
    if var_rank.item() == 0.0:
        return 0.0, 0.0
    s = -(cov / var_rank).item()

    # R²
    lc_pred = lc_mean - s * (log_rank - lr_mean)
    ss_res = ((log_count - lc_pred) ** 2).sum()
    ss_tot = ((log_count - lc_mean) ** 2).sum()
    r2 = (1.0 - ss_res / ss_tot).item() if ss_tot.item() > 0 else 1.0

    return max(s, 0.0), max(min(r2, 1.0), 0.0)


def _counts_to_stats(
    name: str,
    counts_sorted: torch.Tensor,
    NC: int,
    VL: int,
    bits: int,
    n_vectors: int,
) -> AssignmentStats:
    """Build an :class:`AssignmentStats` from pre-sorted assignment counts.

    Args:
        name: Layer or parameter name (for display).
        counts_sorted: Assignment counts sorted **descending** ``[NC]``.
        NC: Codebook size.
        VL: VQ vector length.
        bits: Fixed-width bits per index.
        n_vectors: Total number of vectors (sum of counts).

    Returns:
        :class:`AssignmentStats` with all fields populated.
    """
    from ..transforms.vq_packing import TieredIndexPacking

    zipf_s, zipf_r2 = _fit_zipf(counts_sorted)

    total = max(n_vectors, 1)
    top1_fraction = counts_sorted[0].item() / total
    top16_n = min(NC, 16)
    top16_fraction = counts_sorted[:top16_n].sum().item() / total

    # Empirical Shannon entropy (bits per index).
    probs = counts_sorted.float() / (counts_sorted.float().sum() + 1e-10)
    probs_pos = probs[probs > 0]
    entropy_bpi = float(-(probs_pos * probs_pos.log2()).sum().item())

    # Empirical 2-tier bits per index at actual hot fraction.
    n_hot = int(counts_sorted[:top16_n].sum().item())
    tiered_bpi = TieredIndexPacking.effective_bits(
        n_hot=n_hot, n_total=total, cold_bits=bits
    )

    return AssignmentStats(
        layer_name=name,
        num_centroids=NC,
        vector_len=VL,
        num_vectors=n_vectors,
        counts=counts_sorted,
        zipf_s=zipf_s,
        zipf_r2=zipf_r2,
        top1_fraction=top1_fraction,
        top16_fraction=top16_fraction,
        bits_per_index=bits,
        effective_bpw=bits / VL,
        entropy_bpi=entropy_bpi,
        tiered_bpi=tiered_bpi,
    )


def measure_assignment_distribution(
    model: nn.Module,
) -> dict[str, AssignmentStats]:
    """Measure VQ assignment distributions for all VQ-weight layers in *model*.

    Walks ``model.named_modules()``, finds every ``nn.Linear`` whose weight
    is a :class:`~neuropress.transforms.vq.QTensorTractVQ`, unpacks its
    index buffer, and computes per-centroid assignment counts.

    Args:
        model: An ``nn.Module`` whose parameters may contain
            :class:`~neuropress.transforms.vq.QTensorTractVQ` weights.
            Compatible with both the raw estimator output (QTensorTractVQ
            weights on ``nn.Linear``) and linearized models
            (:class:`~neuropress.nn.np_vq_linear.NPVQLinear`).

    Returns:
        Dict mapping layer name → :class:`AssignmentStats`.
        Layers without a VQ weight are omitted.
    """
    from ..transforms.vq import QTensorTractVQ
    from ..transforms.vq_packing import IndexPacking, TieredIndexPacking

    stats: dict[str, AssignmentStats] = {}

    for name, module in model.named_modules():
        weight_q = getattr(module, "weight", None)
        if not isinstance(weight_q, QTensorTractVQ):
            continue

        qscheme = weight_q.qscheme
        NC = qscheme.num_centroids
        VL = qscheme.vector_len
        bits = qscheme.is_packed_indices
        original_shape = qscheme.original_shape

        if original_shape is None:
            LOGGER.warning(
                "vptq_analysis: skipping '%s' — VQScheme has no original_shape.",
                name,
            )
            continue

        n_vectors = 1
        for d in original_shape:
            n_vectors *= d
        n_vectors //= VL

        if qscheme.use_tiered:
            indices = TieredIndexPacking.unpack(weight_q.u8_blob.cpu())
        else:
            indices = IndexPacking.unpack(weight_q.u8_blob, bits, n_indices=n_vectors)
        indices = indices.cpu()

        counts_raw = torch.zeros(NC, dtype=torch.long)
        counts_raw.scatter_add_(0, indices, torch.ones(len(indices), dtype=torch.long))
        counts_sorted, _ = counts_raw.sort(descending=True)

        st = _counts_to_stats(name, counts_sorted, NC, VL, bits, n_vectors)
        stats[name] = st
        LOGGER.debug(
            "vptq_analysis: '%s'  NC=%d  VL=%d  s=%.3f  r²=%.3f  top1=%.1f%%",
            name,
            NC,
            VL,
            st.zipf_s,
            st.zipf_r2,
            st.top1_fraction * 100,
        )

    return stats


def probe_safetensors(path: str) -> dict[str, AssignmentStats]:
    """Load VQ assignment statistics directly from a safetensors checkpoint.

    Reads the embedded neuropress manifest to discover VQ parameter entries,
    then loads ``u8_blob`` and ``centroids`` tensors from the file to compute
    assignment counts — without requiring model reconstruction.

    Args:
        path: Path to a ``.safetensors`` file written by
            :func:`~neuropress.t2n.model_io.save_quantized_model`.

    Returns:
        Dict mapping parameter path → :class:`AssignmentStats`.
        Returns an empty dict if no VQ layers are found or the file has no
        neuropress manifest.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file exists but contains an invalid manifest.
    """
    import json

    from safetensors import safe_open

    from ..transforms.vq_packing import IndexPacking, TieredIndexPacking

    stats: dict[str, AssignmentStats] = {}

    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        manifest_json = meta.get("__neuropress_manifest__")
        if not manifest_json:
            LOGGER.warning("probe_safetensors: no manifest in %s; no VQ layers.", path)
            return stats

        manifest = json.loads(manifest_json)
        qt_params: dict[str, dict] = {}
        for transform in manifest.get("transforms", []):
            if transform.get("type") == "qtensor_assignment":
                qt_params = transform.get("qtensor_params", {})
                break

        all_keys: set[str] = set(f.keys())

        for param_path, entry in qt_params.items():
            if entry.get("type") != "QTensorTractVQ":
                continue

            u8_key = f"{param_path}::u8_blob"
            cent_key = f"{param_path}::centroids"
            if u8_key not in all_keys or cent_key not in all_keys:
                LOGGER.warning(
                    "probe_safetensors: missing tensors for '%s'; skipping.",
                    param_path,
                )
                continue

            u8_blob = f.get_tensor(u8_key)
            centroids = f.get_tensor(cent_key)

            m = entry.get("metadata", {})
            bits = int(m.get("bits_per_index", 8))
            shape_str = m.get("original_shape", "")
            original_shape = (
                torch.Size(int(d) for d in shape_str.split(",")) if shape_str else None
            )
            use_tiered = m.get("use_tiered", "0") == "1"

            NC = centroids.shape[0]
            VL = centroids.shape[1]

            if original_shape is None:
                LOGGER.warning(
                    "probe_safetensors: '%s' has no original_shape; skipping.",
                    param_path,
                )
                continue

            n_vectors = 1
            for d in original_shape:
                n_vectors *= d
            n_vectors //= VL

            if use_tiered:
                indices = TieredIndexPacking.unpack(u8_blob.cpu())
            else:
                indices = IndexPacking.unpack(u8_blob, bits, n_indices=n_vectors)
            indices = indices.cpu()

            counts_raw = torch.zeros(NC, dtype=torch.long)
            counts_raw.scatter_add_(
                0, indices, torch.ones(len(indices), dtype=torch.long)
            )
            counts_sorted, _ = counts_raw.sort(descending=True)

            stats[param_path] = _counts_to_stats(
                param_path, counts_sorted, NC, VL, bits, n_vectors
            )

    return stats


def pooled_stats(stats: dict[str, AssignmentStats]) -> dict[str, float]:
    """Compute pooled summary statistics across all layers.

    Pools assignment counts (weighted by number of vectors) to produce a
    single global picture of the distribution.

    Args:
        stats: Output of :func:`measure_assignment_distribution`.

    Returns:
        Dict with keys ``"zipf_s"``, ``"zipf_r2"``, ``"top1_fraction"``,
        ``"top16_fraction"``, ``"mean_bpw"``.
    """
    if not stats:
        return {}

    all_s = [st.zipf_s for st in stats.values()]
    all_r2 = [st.zipf_r2 for st in stats.values()]
    weights = [st.num_vectors for st in stats.values()]
    total_w = sum(weights) or 1

    weighted_s = sum(s * w for s, w in zip(all_s, weights, strict=False)) / total_w
    weighted_r2 = sum(r * w for r, w in zip(all_r2, weights, strict=False)) / total_w
    weighted_top1 = (
        sum(st.top1_fraction * st.num_vectors for st in stats.values()) / total_w
    )
    weighted_top16 = (
        sum(st.top16_fraction * st.num_vectors for st in stats.values()) / total_w
    )
    mean_bpw = sum(st.effective_bpw * st.num_vectors for st in stats.values()) / total_w

    return {
        "zipf_s": weighted_s,
        "zipf_r2": weighted_r2,
        "top1_fraction": weighted_top1,
        "top16_fraction": weighted_top16,
        "mean_bpw": mean_bpw,
    }


def theoretical_tiered_bpw(
    zipf_s: float,
    num_centroids: int,
    vector_len: int,
) -> dict[str, float]:
    """Compute theoretical bpw for several coding schemes under a Zipf model.

    All schemes assume centroids are sorted by frequency (index 0 = most used).

    Args:
        zipf_s: Fitted Zipf exponent ``s``.
        num_centroids: Codebook size ``NC``.
        vector_len: VQ vector length ``VL``.

    Returns:
        Dict mapping scheme name → bits-per-weight.
        Keys: ``"fixed"``, ``"2tier"``, ``"2tier_zero_escape"``, ``"entropy"``.
    """
    NC = num_centroids
    bits_fixed = math.ceil(math.log2(max(NC, 2)))

    # Zipf CDF: P(rank <= K) = H(K,s) / H(NC,s)
    def harmonic(K: int, s: float) -> float:
        return sum(1.0 / k**s for k in range(1, K + 1))

    H_NC = (
        harmonic(NC, zipf_s)
        if NC <= 4096
        else sum(1.0 / k**zipf_s for k in range(1, NC + 1))
    )

    def tier_prob(lo: int, hi: int) -> float:
        h_lo = harmonic(lo, zipf_s) if lo > 0 else 0.0
        h_hi = harmonic(min(hi, NC), zipf_s)
        return (h_hi - h_lo) / H_NC

    # 2-tier: 1 prefix bit + 4-bit hot (top 16) / bits_fixed cold
    p_hot = tier_prob(0, 16)
    p_cold = 1.0 - p_hot
    bpw_2tier = (p_hot * (1 + 4) + p_cold * (1 + bits_fixed)) / vector_len

    # 2-tier with zero escape: top centroid gets 2-bit code
    p0 = tier_prob(0, 1)
    p_hot2 = tier_prob(1, 16)
    p_cold2 = 1.0 - p0 - p_hot2
    bpw_zero = (p0 * 2 + p_hot2 * (2 + 4) + p_cold2 * (2 + bits_fixed)) / vector_len

    # Shannon entropy lower bound
    entropy_bits = 0.0
    for k in range(1, NC + 1):
        p = (1.0 / k**zipf_s) / H_NC
        if p > 0:
            entropy_bits -= p * math.log2(p)
    bpw_entropy = entropy_bits / vector_len

    return {
        "fixed": bits_fixed / vector_len,
        "2tier": bpw_2tier,
        "2tier_zero_escape": bpw_zero,
        "entropy": bpw_entropy,
    }


# ---------------------------------------------------------------------------
# Rich report
# ---------------------------------------------------------------------------


def print_distribution_report(
    stats: dict[str, AssignmentStats],
    *,
    top_n: int = 20,
) -> None:
    """Print a formatted report of the assignment distribution to stdout.

    Args:
        stats: Output of :func:`measure_assignment_distribution`.
        top_n: Maximum number of layers to show (sorted by ``zipf_s`` desc).
    """
    try:
        from rich.console import Console

        console = Console()
        _rich_report(stats, console, top_n=top_n)
    except ImportError:
        _plain_report(stats, top_n=top_n)


def _rich_report(
    stats: dict[str, AssignmentStats],
    console: object,
    top_n: int,
) -> None:
    from rich.table import Table

    pooled = pooled_stats(stats)

    table = Table(title="VQ Assignment Distribution", border_style="dim")
    table.add_column("Layer", style="dim", no_wrap=True, max_width=40)
    table.add_column("NC", justify="right")
    table.add_column("VL", justify="right")
    table.add_column("bpw", justify="right")
    table.add_column("s", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("top-16%", justify="right")
    table.add_column("2-tier bpw", justify="right", style="green")
    table.add_column("entropy bpw", justify="right", style="cyan")
    table.add_column("efficiency", justify="right", style="bold")

    rows = sorted(stats.values(), key=lambda x: x.zipf_s, reverse=True)
    for st in rows[:top_n]:
        eff = st.tiered_efficiency()
        eff_str = f"{eff * 100:.0f}%"
        table.add_row(
            st.layer_name,
            str(st.num_centroids),
            str(st.vector_len),
            f"{st.effective_bpw:.2f}",
            f"{st.zipf_s:.3f}",
            f"{st.zipf_r2:.3f}",
            f"{st.top16_fraction * 100:.1f}%",
            f"{st.tiered_bpi / st.vector_len:.3f}",
            f"{st.entropy_bpi / st.vector_len:.3f}",
            eff_str,
        )
    if len(rows) > top_n:
        table.add_row(f"… {len(rows) - top_n} more layers …", *[""] * 9)

    console.print(table)  # type: ignore[attr-defined]

    if pooled:
        # Compute pooled entropy/tiered scores.
        total_w = sum(st.num_vectors for st in stats.values()) or 1
        pooled_entropy_bpi = (
            sum(st.entropy_bpi * st.num_vectors for st in stats.values()) / total_w
        )
        pooled_tiered_bpi = (
            sum(st.tiered_bpi * st.num_vectors for st in stats.values()) / total_w
        )
        pooled_bpi = (
            pooled["mean_bpw"]
            * sum(st.vector_len * st.num_vectors for st in stats.values())
            / total_w
        )
        headroom = pooled_bpi - pooled_entropy_bpi
        savings = pooled_bpi - pooled_tiered_bpi
        eff = min(1.0, max(0.0, savings / headroom)) if headroom > 1e-9 else 1.0

        vl_avg = pooled_bpi / pooled["mean_bpw"]
        tiered_bpw_str = f"{pooled_tiered_bpi / vl_avg:.3f}"
        entropy_bpw_str = f"{pooled_entropy_bpi / vl_avg:.3f}"
        console.print(  # type: ignore[attr-defined]
            f"\n[bold]Pooled[/bold]  "
            f"s={pooled['zipf_s']:.3f}  "
            f"R²={pooled['zipf_r2']:.3f}  "
            f"top-16={pooled['top16_fraction'] * 100:.1f}%  "
            f"mean bpw={pooled['mean_bpw']:.3f}  "
            f"[green]2-tier bpw={tiered_bpw_str}[/green]  "
            f"[cyan]entropy bpw={entropy_bpw_str}[/cyan]  "
            f"[bold]efficiency={eff * 100:.0f}%[/bold]\n"
        )
        s = pooled["zipf_s"]
        if s < 0.9:
            msg = (
                "[yellow]s < 0.9 — distribution too flat; "
                "tiered coding unlikely to pay off.[/yellow]"
            )
        elif s < 1.3:
            msg = (
                "[green]s ∈ [0.9, 1.3] — moderate skew; "
                "2-tier escape recommended.[/green]"
            )
        else:
            msg = (
                "[bold green]s > 1.3 — heavy skew; "
                "2-tier (or zero-escape) strongly recommended.[/bold green]"
            )
        console.print(f"  Recommendation: {msg}\n")  # type: ignore[attr-defined]


def _plain_report(stats: dict[str, AssignmentStats], top_n: int) -> None:
    pooled = pooled_stats(stats)
    rows = sorted(stats.values(), key=lambda x: x.zipf_s, reverse=True)
    print(
        f"{'Layer':<40} {'NC':>6} {'VL':>4} {'bpw':>5} {'s':>6} {'R2':>5} "
        f"{'top16%':>7} {'2tier':>6} {'entropy':>8} {'eff':>4}"
    )
    print("-" * 100)
    for st in rows[:top_n]:
        print(
            f"{st.layer_name:<40} {st.num_centroids:>6} {st.vector_len:>4} "
            f"{st.effective_bpw:>5.2f} {st.zipf_s:>6.3f} {st.zipf_r2:>5.3f} "
            f"{st.top16_fraction * 100:>6.1f}% "
            f"{st.tiered_bpi / st.vector_len:>6.3f} "
            f"{st.entropy_bpi / st.vector_len:>8.3f} "
            f"{st.tiered_efficiency() * 100:>3.0f}%"
        )
    if pooled:
        print(
            f"\nPooled: s={pooled['zipf_s']:.3f}  R²={pooled['zipf_r2']:.3f}  "
            f"top-16={pooled['top16_fraction'] * 100:.1f}%  "
            f"mean bpw={pooled['mean_bpw']:.2f}"
        )


# ---------------------------------------------------------------------------
# Matplotlib distribution plot
# ---------------------------------------------------------------------------


def plot_distribution(
    stats: dict[str, AssignmentStats],
    output_path: str | None = None,
    *,
    title: str | None = None,
) -> None:
    """Plot the rank-frequency distribution and coding efficiency summary.

    Produces a two-panel figure:

    **Left — Zipf log-log plot** (pooled over all layers by vector count):
    Hot-tier centroids (rank 1–16) are shown in orange, cold-tier in blue.
    The fitted Zipf power-law is overlaid as a dashed red line.  A vertical
    dashed line marks the hot/cold boundary.

    **Right — coding efficiency thermometer**: A horizontal bar decomposing
    the fixed-width bits per weight into three zones:

    * Blue   — entropy lower bound (unavoidable under any lossless scheme).
    * Orange — entropy gap (what 2-tier misses vs the entropy bound).
    * Green  — 2-tier savings (what 2-tier recovers vs fixed-width).

    Two scores are printed above the bar:

    * **Headroom captured** = (fixed − tiered) / (fixed − entropy).
      How much of the theoretically-achievable compression 2-tier delivers.
    * **Entropy efficiency** = 1 − entropy / fixed.
      How compressible the distribution is in principle.

    Args:
        stats: Output of :func:`measure_assignment_distribution` or
            :func:`probe_safetensors`.
        output_path: Destination path for the figure (PNG / PDF / SVG).
            When ``None`` the figure is shown interactively via
            ``plt.show()``.
        title: Optional figure super-title.  Defaults to
            ``"VQ Codebook Assignment Distribution"``.

    Raises:
        ImportError: If ``matplotlib`` is not installed.
        ValueError: If *stats* is empty.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_distribution.  "
            "Install it with:  pip install matplotlib"
        ) from exc

    if not stats:
        raise ValueError("plot_distribution: stats dict is empty.")

    # ------------------------------------------------------------------
    # Pooled quantities (weighted by num_vectors)
    # ------------------------------------------------------------------
    total_vecs = sum(st.num_vectors for st in stats.values()) or 1

    def _wavg(attr: str) -> float:
        return (
            sum(getattr(st, attr) * st.num_vectors for st in stats.values())
            / total_vecs
        )

    pooled_bpi = _wavg("bits_per_index")
    pooled_entropy_bpi = _wavg("entropy_bpi")
    pooled_tiered_bpi = _wavg("tiered_bpi")
    pooled_top16 = _wavg("top16_fraction")
    pooled_vl = (
        sum(st.vector_len * st.num_vectors for st in stats.values()) / total_vecs
    )

    bpw_fixed = pooled_bpi / pooled_vl
    bpw_tiered = pooled_tiered_bpi / pooled_vl
    bpw_entropy = pooled_entropy_bpi / pooled_vl

    headroom = bpw_fixed - bpw_entropy
    savings = bpw_fixed - bpw_tiered
    eff_score = min(1.0, max(0.0, savings / headroom)) if headroom > 1e-9 else 1.0
    entropy_eff = (1.0 - bpw_entropy / bpw_fixed) if bpw_fixed > 1e-9 else 0.0

    # ------------------------------------------------------------------
    # Build pooled rank-frequency curve
    # ------------------------------------------------------------------
    # Use the layer with the largest num_vectors as the representative shape.
    representative = max(stats.values(), key=lambda s: s.num_vectors)
    NC = representative.num_centroids
    VL = representative.vector_len

    # Accumulate pooled counts, aligning by rank (all should share NC if same group).
    # For heterogeneous NC, each layer contributes its own sorted-rank probabilities.
    pooled_probs = torch.zeros(NC, dtype=torch.float32)
    for st in stats.values():
        n = min(st.num_centroids, NC)
        p = st.counts[:n].float() / (st.counts.float().sum() + 1e-10)
        pooled_probs[:n] += p * st.num_vectors / total_vecs
    # Re-sort in case layers had different NC.
    pooled_probs, _ = pooled_probs.sort(descending=True)
    ranks = torch.arange(1, NC + 1, dtype=torch.float32)

    # Zipf fit on pooled data.
    zipf_s_pool, zipf_r2_pool = _fit_zipf((pooled_probs * total_vecs).long())

    # ------------------------------------------------------------------
    # Colours
    # ------------------------------------------------------------------
    C_HOT = "#E87722"  # orange
    C_COLD = "#2E86AB"  # steel blue
    C_ENTROPY = "#2980B9"  # dark blue
    C_GAP = "#E67E22"  # orange-brown
    C_SAVINGS = "#27AE60"  # green
    C_FIT = "#E74C3C"  # red

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 5.5), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[6, 4], wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    suptitle = title or "VQ Codebook Assignment Distribution"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)

    # ------------------------------------------------------------------
    # Left panel: Zipf log-log scatter
    # ------------------------------------------------------------------
    hot_mask = (ranks <= 16).numpy()
    cold_mask = ~hot_mask

    probs_np = pooled_probs.numpy()
    ranks_np = ranks.numpy()

    ax1.scatter(
        ranks_np[hot_mask],
        probs_np[hot_mask],
        c=C_HOT,
        s=40,
        alpha=0.9,
        zorder=3,
        label=f"Hot (rank ≤ 16, {pooled_top16 * 100:.1f}% of mass)",
    )
    ax1.scatter(
        ranks_np[cold_mask],
        probs_np[cold_mask],
        c=C_COLD,
        s=8,
        alpha=0.35,
        zorder=2,
        label=f"Cold (rank > 16, NC={NC})",
    )

    # Zipf fit overlay.
    if zipf_s_pool > 0 and zipf_r2_pool > 0.2:
        positive = pooled_probs[pooled_probs > 0]
        n_pos = positive.numel()
        if n_pos >= 2:
            log_r = torch.log(torch.arange(1, n_pos + 1, dtype=torch.float32))
            log_p = torch.log(positive)
            a_fit = log_p.mean() + zipf_s_pool * log_r.mean()
            x_fit = torch.arange(1, NC + 1, dtype=torch.float32)
            y_fit = torch.exp(a_fit - zipf_s_pool * torch.log(x_fit))
            ax1.plot(
                x_fit.numpy(),
                y_fit.numpy(),
                "--",
                color=C_FIT,
                alpha=0.7,
                linewidth=1.8,
                zorder=4,
                label=f"Zipf fit  s={zipf_s_pool:.3f}, R²={zipf_r2_pool:.3f}",
            )

    # Hot/cold boundary.
    ax1.axvline(x=16.5, color="#888888", linestyle=":", alpha=0.6, linewidth=1)
    ax1.text(
        16.5,
        probs_np[probs_np > 0].max() * 0.35,
        "hot │ cold",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#888888",
        rotation=90,
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Centroid rank", fontsize=11)
    ax1.set_ylabel("Frequency (normalized)", fontsize=11)
    ax1.set_title(
        f"Rank-frequency  (NC={NC}, VL={VL}, {len(stats)} layer(s))", fontsize=10
    )
    ax1.legend(fontsize=8.5, framealpha=0.85, loc="upper right")
    ax1.grid(True, which="both", alpha=0.15)
    ax1.set_xlim(0.8, NC * 1.1)

    # ------------------------------------------------------------------
    # Right panel: coding efficiency thermometer
    # ------------------------------------------------------------------
    bar_y = 0.0
    bar_h = 0.42

    # Stacked bar: entropy | gap | savings
    #  [0 ─── entropy ─── | ─── gap ─── | ─── savings ─── bpw_fixed]
    ax2.barh(
        bar_y,
        bpw_entropy,
        height=bar_h,
        color=C_ENTROPY,
        zorder=3,
        label=f"Entropy bound  {bpw_entropy:.3f} bpw",
    )
    ax2.barh(
        bar_y,
        bpw_tiered - bpw_entropy,
        height=bar_h,
        left=bpw_entropy,
        color=C_GAP,
        zorder=3,
        label=f"Entropy gap  {bpw_tiered - bpw_entropy:.3f} bpw",
    )
    ax2.barh(
        bar_y,
        bpw_fixed - bpw_tiered,
        height=bar_h,
        left=bpw_tiered,
        color=C_SAVINGS,
        zorder=3,
        label=f"2-tier savings  {bpw_fixed - bpw_tiered:.3f} bpw",
    )

    # Fixed-width marker.
    ax2.axvline(bpw_fixed, color="#2C3E50", linewidth=2.2, zorder=4, linestyle="-")
    ax2.text(
        bpw_fixed,
        bar_y + bar_h / 2 + 0.03,
        f"fixed  {bpw_fixed:.3f}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#2C3E50",
    )

    # Score overlay — prominent numbers.
    score_color = C_SAVINGS if eff_score >= 0.5 else C_GAP
    ax2.text(
        0.5,
        0.88,
        f"{eff_score * 100:.0f}%",
        transform=ax2.transAxes,
        fontsize=32,
        fontweight="bold",
        ha="center",
        va="top",
        color=score_color,
    )
    ax2.text(
        0.5,
        0.68,
        "headroom captured by 2-tier",
        transform=ax2.transAxes,
        fontsize=9,
        ha="center",
        va="top",
        color="#555555",
    )
    ax2.text(
        0.5,
        0.56,
        f"(entropy efficiency {entropy_eff * 100:.1f}%)",
        transform=ax2.transAxes,
        fontsize=8.5,
        ha="center",
        va="top",
        color="#888888",
    )

    # Per-layer scatter along the efficiency axis.
    layer_tiered_bpw = [st.tiered_bpi / st.vector_len for st in stats.values()]
    jitter = torch.linspace(-bar_h * 0.35, bar_h * 0.35, len(layer_tiered_bpw))
    ax2.scatter(
        layer_tiered_bpw,
        jitter.numpy() + bar_y,
        c="#555555",
        s=18,
        alpha=0.55,
        zorder=5,
    )

    ax2.set_xlim(0, bpw_fixed * 1.18)
    ax2.set_ylim(-0.55, 1.0)
    ax2.set_yticks([])
    ax2.set_xlabel("Bits per weight element", fontsize=11)
    ax2.set_title("Coding efficiency (pooled)", fontsize=10)
    ax2.legend(fontsize=8.5, loc="lower right", framealpha=0.85)
    ax2.grid(True, axis="x", alpha=0.15)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
