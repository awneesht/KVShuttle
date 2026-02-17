"""Generate all paper figures from benchmark results."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

from kvshuttle.visualization.pareto import plot_pareto_frontier
from kvshuttle.visualization.bandwidth_sweep import plot_bandwidth_sweep, plot_speedup_curves
from kvshuttle.visualization.heatmaps import plot_layer_sensitivity_heatmap
from kvshuttle.visualization.kv_comparison import plot_kv_comparison
from kvshuttle.visualization.quality import (
    plot_perplexity_delta,
    plot_token_agreement,
    plot_cosine_vs_perplexity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_figures(results_path: str, output_dir: str = "paper/figures") -> None:
    """Generate all paper figures from a results.json file.

    Args:
        results_path: Path to results.json from experiment runner.
        output_dir: Directory to save figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]
    logger.info("Loaded %d results from %s", len(results), results_path)

    # 1. Pareto frontier (compression ratio vs quality)
    logger.info("Generating Pareto frontier...")
    comp_ratios = {}
    cosine_sims_k = {}
    cosine_sims_v = {}

    for r in results:
        name = r["compressor"]
        comp_ratios.setdefault(name, []).append(r["compression_ratio"])
        if "mean_key_cosine_sim" in r:
            cosine_sims_k.setdefault(name, []).append(r["mean_key_cosine_sim"])
            cosine_sims_v.setdefault(name, []).append(r.get("mean_val_cosine_sim", 1.0))

    avg_ratios = {n: float(np.mean(v)) for n, v in comp_ratios.items()}
    avg_k_cos = {n: float(np.mean(v)) for n, v in cosine_sims_k.items()}
    avg_v_cos = {n: float(np.mean(v)) for n, v in cosine_sims_v.items()}

    if avg_k_cos:
        plot_pareto_frontier(avg_ratios, avg_k_cos, output_dir / "fig1_pareto.pdf")

    # 2. Bandwidth sweep
    logger.info("Generating bandwidth sweep...")
    bw_data: dict[str, dict[float, list[float]]] = {}
    raw_transfer: dict[float, list[float]] = {}

    for r in results:
        name = r["compressor"]
        bw = r["bandwidth_gbps"]
        bw_data.setdefault(name, {}).setdefault(bw, []).append(r["total_ms"])
        raw_transfer.setdefault(bw, []).append(r["raw_transfer_ms"])

    bw_avg = {
        name: {bw: float(np.mean(vals)) for bw, vals in bw_dict.items()}
        for name, bw_dict in bw_data.items()
    }
    raw_avg = {bw: float(np.mean(vals)) for bw, vals in raw_transfer.items()}

    plot_bandwidth_sweep(bw_avg, output_dir / "fig2_bandwidth_sweep.pdf")
    plot_speedup_curves(bw_avg, raw_avg, output_dir / "fig3_speedup_curves.pdf")

    # 3. Keys vs Values comparison
    if avg_k_cos and avg_v_cos:
        logger.info("Generating K/V comparison...")
        plot_kv_comparison(avg_k_cos, avg_v_cos, output_dir / "fig4_kv_comparison.pdf")

    # 4. Layer sensitivity heatmap
    layer_data: dict[str, list[list[float]]] = {}
    for r in results:
        if "key_cosine_sim_per_layer" in r:
            name = r["compressor"]
            layer_data.setdefault(name, []).append(r["key_cosine_sim_per_layer"])

    if layer_data:
        logger.info("Generating layer sensitivity heatmap...")
        # Group by layer count to handle multi-model sweeps with different architectures
        avg_layer = {}
        for name, layers in layer_data.items():
            # Group by number of layers (different models have different layer counts)
            by_nlayers: dict[int, list[list[float]]] = {}
            for l in layers:
                by_nlayers.setdefault(len(l), []).append(l)
            # Use the most common layer count
            most_common = max(by_nlayers.keys(), key=lambda k: len(by_nlayers[k]))
            avg_layer[name] = np.mean(np.array(by_nlayers[most_common]), axis=0).tolist()
        plot_layer_sensitivity_heatmap(avg_layer, output_dir / "fig5_layer_sensitivity.pdf")

    # 5. Generation quality figures (perplexity delta, token agreement)
    ppl_data: dict[str, dict[str, list[float]]] = {}
    ta_data: dict[str, dict[str, list[float]]] = {}
    avg_ppl_by_comp: dict[str, list[float]] = {}

    for r in results:
        comp = r["compressor"]
        model = r["model"]
        if "perplexity_delta" in r:
            ppl_data.setdefault(comp, {}).setdefault(model, []).append(r["perplexity_delta"])
            avg_ppl_by_comp.setdefault(comp, []).append(r["perplexity_delta"])
        if "token_agreement" in r:
            ta_data.setdefault(comp, {}).setdefault(model, []).append(r["token_agreement"])

    if ppl_data:
        logger.info("Generating perplexity delta figure...")
        plot_perplexity_delta(ppl_data, output_dir / "fig6_perplexity_delta.pdf")

    if ta_data:
        logger.info("Generating token agreement figure...")
        plot_token_agreement(ta_data, output_dir / "fig7_token_agreement.pdf")

    if ppl_data and avg_k_cos:
        logger.info("Generating cosine vs perplexity scatter...")
        avg_ppl = {n: float(np.mean(v)) for n, v in avg_ppl_by_comp.items()}
        plot_cosine_vs_perplexity(avg_k_cos, avg_ppl, output_dir / "fig8_cosine_vs_perplexity.pdf")

    logger.info("All figures saved to %s", output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_paper_figures.py <results.json> [output_dir]")
        sys.exit(1)
    results_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "paper/figures"
    generate_figures(results_file, out_dir)
