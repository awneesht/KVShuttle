"""Heatmap visualizations for per-layer sensitivity analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_layer_sensitivity_heatmap(
    cosine_sim_per_layer: dict[str, list[float]],
    output_path: str | Path | None = None,
    title: str = "Per-Layer Key Cosine Similarity by Compressor",
) -> plt.Figure:
    """Plot heatmap of per-layer quality across compressors.

    Args:
        cosine_sim_per_layer: {compressor_name: [layer_0_sim, layer_1_sim, ...]}.
        output_path: Save path.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    names = sorted(cosine_sim_per_layer.keys())
    num_layers = len(next(iter(cosine_sim_per_layer.values())))

    data = np.array([cosine_sim_per_layer[n] for n in names])

    fig, ax = plt.subplots(figsize=(max(12, num_layers * 0.4), max(6, len(names) * 0.5)))
    sns.heatmap(
        data,
        annot=num_layers <= 32,
        fmt=".3f" if num_layers <= 32 else "",
        xticklabels=[f"L{i}" for i in range(num_layers)],
        yticklabels=names,
        cmap="RdYlGn",
        vmin=0.8,
        vmax=1.0,
        ax=ax,
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Compressor", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved heatmap to %s", output_path)

    return fig
