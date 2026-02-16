"""Router evaluation plots."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kvshuttle.router.evaluator import RouterEvalResult

logger = logging.getLogger(__name__)


def plot_router_comparison(
    eval_results: list[RouterEvalResult],
    output_path: str | Path | None = None,
    title: str = "Router Comparison: Regret vs Oracle",
) -> plt.Figure:
    """Bar chart comparing different routers by mean/median regret.

    Args:
        eval_results: List of evaluation results for different routers.
        output_path: Save path.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    names = [r.router_name for r in eval_results]
    mean_regret = [r.mean_regret * 100 for r in eval_results]  # as percentage
    median_regret = [r.median_regret * 100 for r in eval_results]
    p95_regret = [r.p95_regret * 100 for r in eval_results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, mean_regret, width, label="Mean Regret", color="#2196F3")
    ax.bar(x, median_regret, width, label="Median Regret", color="#4CAF50")
    ax.bar(x + width, p95_regret, width, label="P95 Regret", color="#FF5722")

    ax.set_ylabel("Regret (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved router comparison to %s", output_path)

    return fig
