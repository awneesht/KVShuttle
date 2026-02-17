"""GPU calibration visualization: CPU vs GPU timing, speedup, and pipeline plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_cpu_vs_gpu_bars(
    compressor_data: dict[str, dict],
    output_path: str | Path | None = None,
    title: str = "CPU vs GPU Compression Kernel Time",
) -> plt.Figure:
    """Grouped bar chart: CPU vs GPU compress/decompress times.

    Args:
        compressor_data: {compressor: {cpu_compress_ms, gpu_compress_ms,
                          cpu_decompress_ms, gpu_decompress_ms, compress_speedup,
                          decompress_speedup}}.
        output_path: Save path.
        title: Plot title.
    """
    names = sorted(compressor_data.keys())
    x = np.arange(len(names))
    width = 0.2

    cpu_comp = [compressor_data[n]["cpu_compress_ms"] for n in names]
    gpu_comp = [compressor_data[n]["gpu_compress_ms"] for n in names]
    cpu_dec = [compressor_data[n]["cpu_decompress_ms"] for n in names]
    gpu_dec = [compressor_data[n]["gpu_decompress_ms"] for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, cpu_comp, width, label="CPU Compress", color="#2196F3", alpha=0.85)
    ax.bar(x - 0.5 * width, gpu_comp, width, label="GPU Compress", color="#1565C0", alpha=0.85)
    ax.bar(x + 0.5 * width, cpu_dec, width, label="CPU Decompress", color="#FF9800", alpha=0.85)
    ax.bar(x + 1.5 * width, gpu_dec, width, label="GPU Decompress", color="#E65100", alpha=0.85)

    # Add speedup annotations
    for i, n in enumerate(names):
        d = compressor_data[n]
        ax.text(x[i] - 0.5 * width, gpu_comp[i], f'{d["compress_speedup"]:.0f}x',
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1565C0")
        ax.text(x[i] + 1.5 * width, gpu_dec[i], f'{d["decompress_speedup"]:.0f}x',
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#E65100")

    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ") for n in names], fontsize=10)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved CPU vs GPU bars to %s", output_path)
    plt.close(fig)
    return fig


def plot_gpu_speedup_scaling(
    scaling_data: dict[str, list[dict]],
    output_path: str | Path | None = None,
    title: str = "GPU Speedup vs Cache Size",
) -> plt.Figure:
    """Line plot showing GPU speedup increasing with cache size.

    Args:
        scaling_data: {compressor: [{cache_mb, compress_speedup, decompress_speedup}, ...]}.
        output_path: Save path.
        title: Plot title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, entries in sorted(scaling_data.items()):
        sizes = [e["cache_mb"] for e in entries]
        comp_sp = [e["compress_speedup"] for e in entries]
        dec_sp = [e["decompress_speedup"] for e in entries]
        label = name.replace("_", " ")
        axes[0].plot(sizes, comp_sp, "-o", label=label, linewidth=2, markersize=6)
        axes[1].plot(sizes, dec_sp, "-s", label=label, linewidth=2, markersize=6)

    for ax, op in zip(axes, ["Compress", "Decompress"]):
        ax.set_xlabel("KV Cache Size (MB)", fontsize=12)
        ax.set_ylabel(f"{op} Speedup (x faster than CPU)", fontsize=12)
        ax.set_title(f"{op} Speedup", fontsize=13)
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved GPU speedup scaling to %s", output_path)
    plt.close(fig)
    return fig


def plot_pipeline_comparison(
    pipeline_data: list[dict],
    output_path: str | Path | None = None,
    title: str = "Sequential vs Pipelined Transfer (GPU-Calibrated)",
) -> plt.Figure:
    """Side-by-side speedup curves: sequential vs pipelined, for each compressor.

    Args:
        pipeline_data: List of dicts with keys: compressor, bandwidth_gbps,
            sequential_speedup, pipelined_speedup, bottleneck_stage.
        output_path: Save path.
        title: Plot title.
    """
    compressors = sorted(set(e["compressor"] for e in pipeline_data))
    fig, axes = plt.subplots(1, len(compressors), figsize=(7 * len(compressors), 5))
    if len(compressors) == 1:
        axes = [axes]

    for ax, comp_name in zip(axes, compressors):
        entries = sorted(
            [e for e in pipeline_data if e["compressor"] == comp_name],
            key=lambda e: e["bandwidth_gbps"],
        )
        bws = [e["bandwidth_gbps"] for e in entries]
        seq = [e["sequential_speedup"] for e in entries]
        pipe = [e["pipelined_speedup"] for e in entries]

        ax.semilogx(bws, seq, "-o", label="Sequential", linewidth=2, markersize=7, color="steelblue")
        ax.semilogx(bws, pipe, "-s", label="Pipelined", linewidth=2, markersize=7, color="darkorange")
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label="Break-even")

        # Shade beneficial region
        ax.fill_between(bws, 1.0, [max(s, p) for s, p in zip(seq, pipe)],
                        where=[max(s, p) > 1.0 for s, p in zip(seq, pipe)],
                        alpha=0.08, color="green")

        ax.set_xlabel("Bandwidth (Gbps)", fontsize=12)
        ax.set_ylabel("Speedup vs Raw Transfer", fontsize=12)
        ax.set_title(comp_name.replace("_", " ").title(), fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved pipeline comparison to %s", output_path)
    plt.close(fig)
    return fig


def plot_gpu_calibrated_speedup(
    cpu_results: dict[str, dict[float, float]],
    gpu_results: dict[str, dict[float, float]],
    raw_transfer_ms: dict[float, float],
    output_path: str | Path | None = None,
    title: str = "Speedup: CPU-Timed vs GPU-Calibrated",
) -> plt.Figure:
    """Overlay CPU-timed and GPU-calibrated speedup curves.

    Args:
        cpu_results: {compressor: {bw: total_ms}} from CPU numpy timing.
        gpu_results: {compressor: {bw: total_ms}} with GPU-adjusted timing
                     (from pipeline data with realistic seq_lens).
        raw_transfer_ms: {bw: raw_ms}.
        output_path: Save path.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Combine all compressor names for color assignment
    all_names = sorted(set(list(cpu_results.keys()) + list(gpu_results.keys())))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_names), 10)))
    color_map = {n: colors[i] for i, n in enumerate(all_names)}

    for name in all_names:
        if name == "identity":
            continue
        color = color_map[name]

        # CPU dashed (from model_sweep, if available)
        if name in cpu_results:
            bws = sorted(cpu_results[name].keys())
            cpu_speedups = []
            for b in bws:
                raw = raw_transfer_ms.get(b)
                if raw is None:
                    # CPU raw_transfer at this bw not available, interpolate from nearest
                    closest = min(raw_transfer_ms.keys(), key=lambda x: abs(x - b))
                    raw = raw_transfer_ms[closest] * (closest / b)
                cpu_speedups.append(raw / cpu_results[name][b] if cpu_results[name][b] > 0 else 0)
            ax.plot(bws, cpu_speedups, "--", color=color, alpha=0.4, linewidth=1.5)

        # GPU solid (from pipeline data, if available)
        if name in gpu_results:
            gpu_bws = sorted(gpu_results[name].keys())
            gpu_speedups = []
            for b in gpu_bws:
                raw = raw_transfer_ms.get(b)
                if raw is None:
                    closest = min(raw_transfer_ms.keys(), key=lambda x: abs(x - b))
                    raw = raw_transfer_ms[closest] * (closest / b)
                gpu_speedups.append(raw / gpu_results[name][b] if gpu_results[name][b] > 0 else 0)
            ax.plot(gpu_bws, gpu_speedups, "-o", color=color, linewidth=2.5,
                    markersize=6, label=f"{name} (GPU)")
        elif name in cpu_results:
            # No GPU data — show CPU as solid
            ax.plot(bws, cpu_speedups, "-", color=color, linewidth=1.5,
                    alpha=0.6, label=f"{name} (CPU)")

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="Break-even")
    ax.set_xlabel("Bandwidth (Gbps)", fontsize=12)
    ax.set_ylabel("Speedup (vs raw transfer)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale("log")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved GPU-calibrated speedup to %s", output_path)
    plt.close(fig)
    return fig


def plot_multi_gpu_speedup(
    gpu_data: dict[str, dict[str, dict]],
    output_path: str | Path | None = None,
    title: str = "GPU Speedup Comparison Across Hardware",
) -> plt.Figure:
    """Grouped bars comparing speedup across multiple GPUs.

    Args:
        gpu_data: {gpu_name: {compressor: {compress_speedup, decompress_speedup}}}.
        output_path: Save path.
        title: Plot title.
    """
    gpu_names = sorted(gpu_data.keys())
    all_compressors = sorted({c for g in gpu_data.values() for c in g.keys()})
    x = np.arange(len(all_compressors))
    n_gpus = len(gpu_names)
    width = 0.35 / max(n_gpus, 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_gpus, 8)))

    for gi, gpu_name in enumerate(gpu_names):
        comp_sp = [gpu_data[gpu_name].get(c, {}).get("compress_speedup", 0) for c in all_compressors]
        dec_sp = [gpu_data[gpu_name].get(c, {}).get("decompress_speedup", 0) for c in all_compressors]
        offset = (gi - n_gpus / 2 + 0.5) * width

        axes[0].bar(x + offset, comp_sp, width, label=gpu_name, color=colors[gi], alpha=0.85)
        axes[1].bar(x + offset, dec_sp, width, label=gpu_name, color=colors[gi], alpha=0.85)

    for ax, op in zip(axes, ["Compress", "Decompress"]):
        ax.set_ylabel(f"{op} Speedup (x faster than CPU)", fontsize=12)
        ax.set_title(f"{op} Speedup by GPU", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ") for c in all_compressors], fontsize=9, rotation=20)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved multi-GPU speedup to %s", output_path)
    plt.close(fig)
    return fig


def plot_breakeven_shift(
    breakeven_data: dict[str, dict[str, float]],
    output_path: str | Path | None = None,
    title: str = "Break-Even Bandwidth Shift Across GPUs",
) -> plt.Figure:
    """Show how break-even bandwidth changes with GPU generation.

    Args:
        breakeven_data: {gpu_name: {compressor: max_beneficial_bw_gbps}}.
        output_path: Save path.
        title: Plot title.
    """
    gpu_names = sorted(breakeven_data.keys())
    all_compressors = sorted({c for g in breakeven_data.values() for c in g.keys()})
    x = np.arange(len(all_compressors))
    n_gpus = len(gpu_names)
    width = 0.7 / max(n_gpus, 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_gpus, 8)))

    for gi, gpu_name in enumerate(gpu_names):
        bws = [breakeven_data[gpu_name].get(c, 0) for c in all_compressors]
        offset = (gi - n_gpus / 2 + 0.5) * width
        ax.bar(x + offset, bws, width, label=gpu_name, color=colors[gi], alpha=0.85)

    ax.set_ylabel("Max Beneficial Bandwidth (Gbps)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in all_compressors], fontsize=9, rotation=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved break-even shift to %s", output_path)
    plt.close(fig)
    return fig


def plot_transfer_validation(
    results: list[dict],
    output_path: str | Path | None = None,
    title: str = "Transfer Model Validation: Analytical vs TCP",
) -> plt.Figure:
    """Dual-axis chart: analytical (line) vs measured (scatter) transfer times.

    Args:
        results: List of dicts from measure_transfer_overhead() with
            payload_mb, analytical_ms, real_ms, overhead_pct.
        output_path: Save path.
        title: Plot title.
    """
    valid = [r for r in results if r.get("real_ms") is not None]
    if not valid:
        logger.warning("No valid transfer measurements for plot")
        fig, ax = plt.subplots()
        plt.close(fig)
        return fig

    sizes_mb = [r["payload_mb"] for r in valid]
    analytical_ms = [r["analytical_ms"] for r in valid]
    real_ms = [r["real_ms"] for r in valid]
    overhead_pct = [r["overhead_pct"] for r in valid]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary axis: transfer times
    ax1.loglog(sizes_mb, analytical_ms, "o-", label="Analytical Model", linewidth=2,
               markersize=7, color="steelblue")
    ax1.loglog(sizes_mb, real_ms, "s-", label="Real TCP", linewidth=2,
               markersize=7, color="darkorange")
    ax1.set_xlabel("Payload Size (MB)", fontsize=12)
    ax1.set_ylabel("Transfer Time (ms)", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # Secondary axis: overhead percentage
    ax2 = ax1.twinx()
    ax2.semilogx(sizes_mb, overhead_pct, "D--", linewidth=1.5, markersize=6,
                 color="crimson", alpha=0.7, label="Overhead %")
    ax2.set_ylabel("Overhead (%)", fontsize=12, color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.axhline(y=0, color="crimson", linestyle=":", linewidth=0.8, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)

    ax1.set_title(title, fontsize=14)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved transfer validation to %s", output_path)
    plt.close(fig)
    return fig
