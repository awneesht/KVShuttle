"""Validate analytical transfer model against real TCP measurements.

Runs payload sizes from 1 KB to 100 MB through both analytical and TCP transfer,
generates a comparison plot showing overhead.

Usage:
    python validate_transfer_model.py [output_dir]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_validation(output_dir: str = "paper/figures") -> None:
    """Run transfer model validation and generate comparison plot."""
    from kvshuttle.transfer.real_transfer import measure_transfer_overhead

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Payload sizes: 1 KB to 100 MB (logarithmic spacing)
    payload_sizes = [
        1 * 1024,           # 1 KB
        10 * 1024,          # 10 KB
        100 * 1024,         # 100 KB
        1024 * 1024,        # 1 MB
        5 * 1024 * 1024,    # 5 MB
        10 * 1024 * 1024,   # 10 MB
        25 * 1024 * 1024,   # 25 MB
        50 * 1024 * 1024,   # 50 MB
        100 * 1024 * 1024,  # 100 MB
    ]

    logger.info("Running transfer validation with %d payload sizes", len(payload_sizes))

    # Measure at a nominal 100 Gbps analytical bandwidth
    # (real TCP localhost will likely be faster or slower depending on system)
    results = measure_transfer_overhead(
        payload_sizes=payload_sizes,
        bandwidth_gbps=100.0,
        repeats=5,
    )

    # Save raw results
    results_path = output_dir / "transfer_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", results_path)

    # Generate comparison plot
    valid = [r for r in results if r["real_ms"] is not None]
    if not valid:
        logger.error("No successful measurements!")
        return

    sizes_mb = [r["payload_mb"] for r in valid]
    analytical_ms = [r["analytical_ms"] for r in valid]
    real_ms = [r["real_ms"] for r in valid]
    overhead_pct = [r["overhead_pct"] for r in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: analytical vs real transfer time
    ax1.loglog(sizes_mb, analytical_ms, "o-", label="Analytical Model", linewidth=2,
               markersize=7, color="steelblue")
    ax1.loglog(sizes_mb, real_ms, "s-", label="Real TCP (localhost)", linewidth=2,
               markersize=7, color="darkorange")
    ax1.set_xlabel("Payload Size (MB)", fontsize=12)
    ax1.set_ylabel("Transfer Time (ms)", fontsize=12)
    ax1.set_title("Analytical vs Real Transfer Time", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which="both")

    # Right: overhead percentage
    ax2.semilogx(sizes_mb, overhead_pct, "D-", linewidth=2, markersize=7, color="crimson")
    ax2.set_xlabel("Payload Size (MB)", fontsize=12)
    ax2.set_ylabel("Overhead (%)", fontsize=12)
    ax2.set_title("TCP Overhead vs Analytical Model", fontsize=13)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # Annotate key thresholds
    for r in valid:
        if r["payload_mb"] >= 1.0:
            ax2.annotate(
                f'{r["overhead_pct"]:.1f}%',
                (r["payload_mb"], r["overhead_pct"]),
                textcoords="offset points", xytext=(5, 8), fontsize=8,
            )

    plt.suptitle("Transfer Model Validation: Analytical vs TCP Localhost", fontsize=14, y=1.02)
    plt.tight_layout()

    fig_path = output_dir / "transfer_validation.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved validation plot to %s", fig_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("TRANSFER VALIDATION SUMMARY")
    logger.info("=" * 60)
    for r in valid:
        logger.info(
            "  %8.2f MB: analytical=%7.3f ms, real=%7.3f ms, overhead=%+.1f%%",
            r["payload_mb"], r["analytical_ms"], r["real_ms"], r["overhead_pct"],
        )

    # Check if overhead < 20% for payloads > 1 MB
    large_payloads = [r for r in valid if r["payload_mb"] >= 1.0]
    if large_payloads:
        max_overhead = max(abs(r["overhead_pct"]) for r in large_payloads)
        logger.info("-" * 60)
        logger.info("Max overhead for payloads >= 1 MB: %.1f%%", max_overhead)
        if max_overhead < 20:
            logger.info("PASS: Overhead < 20%% for large payloads")
        else:
            logger.warning("NOTE: Overhead >= 20%% for some large payloads (TCP != analytical)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "paper/figures"
    run_validation(out)
