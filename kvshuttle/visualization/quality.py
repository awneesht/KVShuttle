"""Visualization for end-to-end generation quality metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_perplexity_delta(
    data: dict[str, dict[str, list[float]]],
    output_path: str | Path | None = None,
    title: str = "Perplexity Delta by Compressor",
) -> plt.Figure:
    """Bar chart of perplexity delta per compressor, grouped by model.

    Args:
        data: {compressor: {model: [perplexity_delta_values]}}.
        output_path: Save path for the figure.
        title: Plot title.
    """
    compressors = sorted(data.keys())
    models = sorted({m for d in data.values() for m in d.keys()})
    x = np.arange(len(compressors))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 10)))

    for i, model in enumerate(models):
        means = []
        stds = []
        for comp in compressors:
            vals = data.get(comp, {}).get(model, [0.0])
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        # Clip lower error bars so they don't go below zero
        yerr_lower = np.minimum(stds_arr, means_arr.clip(0))
        yerr_upper = stds_arr
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=[yerr_lower, yerr_upper], label=model,
               color=colors[i], alpha=0.85, capsize=2)

    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Perplexity Delta (log scale, lower = better)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in compressors], fontsize=10, rotation=15)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved perplexity delta plot to %s", output_path)
    plt.close(fig)
    return fig


def plot_token_agreement(
    data: dict[str, dict[str, list[float]]],
    output_path: str | Path | None = None,
    title: str = "Token Agreement by Compressor",
) -> plt.Figure:
    """Bar chart of token agreement per compressor, grouped by model.

    Args:
        data: {compressor: {model: [token_agreement_values]}}.
        output_path: Save path for the figure.
        title: Plot title.
    """
    compressors = sorted(data.keys())
    models = sorted({m for d in data.values() for m in d.keys()})
    x = np.arange(len(compressors))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 10)))

    for i, model in enumerate(models):
        means = []
        for comp in compressors:
            vals = data.get(comp, {}).get(model, [1.0])
            means.append(np.mean(vals))
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=model,
               color=colors[i], alpha=0.85)

    ax.set_ylabel("Token Agreement Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in compressors], fontsize=10, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="green", linestyle="--", linewidth=0.8, alpha=0.5, label="Perfect")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved token agreement plot to %s", output_path)
    plt.close(fig)
    return fig


def plot_cosine_vs_perplexity(
    cosine_sims: dict[str, float],
    perplexity_deltas: dict[str, float],
    output_path: str | Path | None = None,
    title: str = "Cosine Similarity vs Perplexity Delta",
) -> plt.Figure:
    """Scatter plot: cosine similarity vs perplexity delta, one point per compressor.

    Args:
        cosine_sims: {compressor: mean_cosine_sim}.
        perplexity_deltas: {compressor: mean_perplexity_delta}.
        output_path: Save path for the figure.
        title: Plot title.
    """
    common = sorted(set(cosine_sims.keys()) & set(perplexity_deltas.keys()))
    if not common:
        logger.warning("No common compressors for cosine vs perplexity plot")
        fig, ax = plt.subplots()
        plt.close(fig)
        return fig

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(common), 10)))

    for i, name in enumerate(common):
        cos = cosine_sims[name]
        ppl = perplexity_deltas[name]
        ax.scatter(cos, ppl, s=120, color=colors[i], zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(name.replace("_", " "), (cos, ppl),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Mean Cosine Similarity (KV cache)", fontsize=12)
    ax.set_ylabel("Perplexity Delta (log scale)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved cosine vs perplexity plot to %s", output_path)
    plt.close(fig)
    return fig
