"""Bandwidth sweep plots: latency vs bandwidth curves."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_bandwidth_sweep(
    results: dict[str, dict[float, float]],
    output_path: str | Path | None = None,
    title: str = "Total Latency vs Network Bandwidth",
) -> plt.Figure:
    """Plot total pipeline latency across bandwidth points for each compressor.

    Args:
        results: Nested dict {compressor_name: {bandwidth_gbps: total_ms}}.
        output_path: If provided, save figure.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, bw_data in sorted(results.items()):
        bws = sorted(bw_data.keys())
        latencies = [bw_data[b] for b in bws]
        ax.plot(bws, latencies, marker="o", label=name, linewidth=2)

    ax.set_xlabel("Bandwidth (Gbps)", fontsize=12)
    ax.set_ylabel("Total Pipeline Latency (ms)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved bandwidth sweep plot to %s", output_path)

    return fig


def plot_speedup_curves(
    results: dict[str, dict[float, float]],
    raw_transfer_ms: dict[float, float],
    output_path: str | Path | None = None,
    title: str = "Speedup vs Bandwidth (Break-even Analysis)",
) -> plt.Figure:
    """Plot speedup curves showing where compression helps vs hurts.

    Args:
        results: {compressor_name: {bandwidth_gbps: total_ms}}.
        raw_transfer_ms: {bandwidth_gbps: raw_transfer_ms} (identity baseline).
        output_path: Save path.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, bw_data in sorted(results.items()):
        if name == "identity":
            continue
        bws = sorted(bw_data.keys())
        speedups = []
        for b in bws:
            raw = raw_transfer_ms.get(b, bw_data[b])
            speedups.append(raw / bw_data[b] if bw_data[b] > 0 else 1.0)
        ax.plot(bws, speedups, marker="o", label=name, linewidth=2)

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
        logger.info("Saved speedup curves to %s", output_path)

    return fig
