"""Pareto frontier plots: compression ratio vs quality."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_pareto_frontier(
    compression_ratios: dict[str, float],
    cosine_similarities: dict[str, float],
    output_path: str | Path | None = None,
    title: str = "Compression Ratio vs Quality (Pareto Frontier)",
) -> plt.Figure:
    """Plot compression ratio vs cosine similarity with Pareto frontier.

    Args:
        compression_ratios: Compressor name -> compression ratio.
        cosine_similarities: Compressor name -> mean cosine similarity.
        output_path: If provided, save figure to this path.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    names = list(compression_ratios.keys())
    ratios = np.array([compression_ratios[n] for n in names])
    quality = np.array([cosine_similarities[n] for n in names])

    # Plot all points
    ax.scatter(ratios, quality, s=100, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(
            name, (ratios[i], quality[i]),
            textcoords="offset points", xytext=(5, 5), fontsize=8,
        )

    # Draw Pareto frontier
    pareto_mask = _pareto_mask(ratios, quality)
    pareto_idx = np.where(pareto_mask)[0]
    pareto_idx = pareto_idx[np.argsort(ratios[pareto_idx])]
    ax.plot(ratios[pareto_idx], quality[pareto_idx], "r--", linewidth=2, label="Pareto frontier")

    ax.set_xlabel("Compression Ratio (x)", fontsize=12)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved Pareto plot to %s", output_path)

    return fig


def _pareto_mask(ratios: np.ndarray, quality: np.ndarray) -> np.ndarray:
    """Find Pareto-optimal points (maximize both ratio and quality)."""
    n = len(ratios)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                if ratios[j] >= ratios[i] and quality[j] >= quality[i]:
                    if ratios[j] > ratios[i] or quality[j] > quality[i]:
                        mask[i] = False
                        break
    return mask
