"""Keys vs Values comparison plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_kv_comparison(
    key_cosine_sims: dict[str, float],
    val_cosine_sims: dict[str, float],
    output_path: str | Path | None = None,
    title: str = "Keys vs Values: Compression Quality Comparison",
) -> plt.Figure:
    """Grouped bar chart comparing key vs value quality per compressor.

    Args:
        key_cosine_sims: {compressor_name: mean_key_cosine_sim}.
        val_cosine_sims: {compressor_name: mean_val_cosine_sim}.
        output_path: Save path.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    names = sorted(key_cosine_sims.keys())
    x = np.arange(len(names))
    width = 0.35

    k_vals = [key_cosine_sims[n] for n in names]
    v_vals = [val_cosine_sims[n] for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, k_vals, width, label="Keys", color="#2196F3")
    bars2 = ax.bar(x + width / 2, v_vals, width, label="Values", color="#FF9800")

    ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved K/V comparison to %s", output_path)

    return fig
