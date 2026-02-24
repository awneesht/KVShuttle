"""Compute bootstrap confidence intervals from existing benchmark data.

Reads model_sweep and generation_quality results, computes 95% CIs
for compression ratio, cosine similarity, token agreement, and perplexity delta.
Outputs a JSON summary used by paper table generators.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kvshuttle.analysis.statistical_tests import (
    bootstrap_ci,
    wilcoxon_signed_rank,
    spearman_correlation,
    friedman_test,
    BootstrapCI,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parents[2]

# Result file paths
MODEL_SWEEP = PROJ_ROOT / "experiments/results/model_sweep/results.json"
GEN_QUALITY = PROJ_ROOT / "experiments/results/generation_quality_fp16_merged/results.json"
COMP_SWEEP = PROJ_ROOT / "experiments/results/compression_sweep/results.json"

OUTPUT_PATH = PROJ_ROOT / "experiments/results/confidence_intervals.json"


def _ci_to_dict(ci: BootstrapCI) -> dict:
    return {
        "estimate": ci.estimate,
        "ci_lower": ci.ci_lower,
        "ci_upper": ci.ci_upper,
        "ci_level": ci.ci_level,
    }


def compute_model_sweep_cis(results: list[dict]) -> dict:
    """Compute CIs for model_sweep: compression ratio and cosine similarity."""
    by_comp: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        comp = r["compressor"]
        by_comp[comp]["compression_ratio"].append(r["compression_ratio"])
        if "mean_key_cosine_sim" in r:
            by_comp[comp]["key_cosine_sim"].append(r["mean_key_cosine_sim"])
            by_comp[comp]["val_cosine_sim"].append(r.get("mean_val_cosine_sim", 1.0))

    out = {}
    for comp, metrics in by_comp.items():
        out[comp] = {}
        for metric, values in metrics.items():
            out[comp][metric] = _ci_to_dict(bootstrap_ci(values))
    return out


def compute_generation_quality_cis(results: list[dict]) -> dict:
    """Compute CIs for generation quality: token agreement and perplexity delta."""
    by_comp_model: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    by_comp: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        comp = r["compressor"]
        model = r["model"]
        if "token_agreement" in r:
            by_comp_model[comp][model]["token_agreement"].append(r["token_agreement"])
            by_comp[comp]["token_agreement"].append(r["token_agreement"])
        if "perplexity_delta" in r:
            by_comp_model[comp][model]["perplexity_delta"].append(r["perplexity_delta"])
            by_comp[comp]["perplexity_delta"].append(r["perplexity_delta"])
        if "mean_key_cosine_sim" in r:
            by_comp[comp]["key_cosine_sim"].append(r["mean_key_cosine_sim"])

    # Overall CIs per compressor
    overall = {}
    for comp, metrics in by_comp.items():
        overall[comp] = {}
        for metric, values in metrics.items():
            overall[comp][metric] = _ci_to_dict(bootstrap_ci(values))

    # Per-model CIs
    per_model = {}
    for comp, model_dict in by_comp_model.items():
        per_model[comp] = {}
        for model, metrics in model_dict.items():
            per_model[comp][model] = {}
            for metric, values in metrics.items():
                per_model[comp][model][metric] = _ci_to_dict(bootstrap_ci(values))

    return {"overall": overall, "per_model": per_model}


def compute_pairwise_tests(results: list[dict]) -> dict:
    """Wilcoxon signed-rank tests between compressor pairs on token agreement."""
    # Group by (compressor, prompt_idx) → token_agreement
    by_comp_prompt: dict[str, dict[int, float]] = defaultdict(dict)
    for r in results:
        if "token_agreement" in r:
            comp = r["compressor"]
            idx = r["prompt_idx"]
            by_comp_prompt[comp][idx] = r["token_agreement"]

    compressors = sorted(by_comp_prompt.keys())
    pairwise = {}
    for i, a in enumerate(compressors):
        for b in compressors[i + 1:]:
            # Find common prompts
            common = sorted(set(by_comp_prompt[a].keys()) & set(by_comp_prompt[b].keys()))
            if len(common) < 10:
                continue
            x = [by_comp_prompt[a][idx] for idx in common]
            y = [by_comp_prompt[b][idx] for idx in common]
            result = wilcoxon_signed_rank(x, y)
            pairwise[f"{a}_vs_{b}"] = {
                "statistic": result.statistic,
                "p_value": result.p_value,
                "effect_size": result.effect_size,
                "significant_05": result.significant_at_05,
                "n_pairs": len(common),
            }
    return pairwise


def compute_correlation_analysis(results: list[dict]) -> dict:
    """Spearman correlation between cosine similarity and generation quality."""
    cosine_vals = []
    ta_vals = []
    ppl_vals = []

    for r in results:
        if "mean_key_cosine_sim" in r and "token_agreement" in r:
            cosine_vals.append(r["mean_key_cosine_sim"])
            ta_vals.append(r["token_agreement"])
        if "mean_key_cosine_sim" in r and "perplexity_delta" in r:
            ppl_vals.append(r["perplexity_delta"])

    out = {}
    if cosine_vals and ta_vals:
        corr = spearman_correlation(cosine_vals, ta_vals)
        out["cosine_vs_token_agreement"] = {
            "coefficient": corr.coefficient,
            "p_value": corr.p_value,
            "n": corr.n,
        }
    if cosine_vals and ppl_vals and len(ppl_vals) == len(cosine_vals):
        corr = spearman_correlation(cosine_vals[:len(ppl_vals)], ppl_vals)
        out["cosine_vs_perplexity_delta"] = {
            "coefficient": corr.coefficient,
            "p_value": corr.p_value,
            "n": corr.n,
        }
    return out


def main() -> None:
    output: dict = {}

    # Model sweep CIs
    if MODEL_SWEEP.exists():
        logger.info("Loading model_sweep results...")
        with open(MODEL_SWEEP) as f:
            data = json.load(f)
        output["model_sweep_cis"] = compute_model_sweep_cis(data["results"])
        logger.info(
            "Computed CIs for %d compressors from model_sweep",
            len(output["model_sweep_cis"]),
        )

    # Generation quality CIs
    if GEN_QUALITY.exists():
        logger.info("Loading generation_quality results...")
        with open(GEN_QUALITY) as f:
            data = json.load(f)
        results = data["results"]
        output["generation_quality_cis"] = compute_generation_quality_cis(results)
        output["pairwise_tests"] = compute_pairwise_tests(results)
        output["correlation_analysis"] = compute_correlation_analysis(results)
        logger.info(
            "Computed generation quality CIs for %d compressors",
            len(output["generation_quality_cis"]["overall"]),
        )
        logger.info("Computed %d pairwise tests", len(output["pairwise_tests"]))

    # Compression sweep CIs (14 compressors on llama-3.1-8b)
    if COMP_SWEEP.exists():
        logger.info("Loading compression_sweep results...")
        with open(COMP_SWEEP) as f:
            data = json.load(f)
        output["compression_sweep_cis"] = compute_model_sweep_cis(data["results"])
        logger.info(
            "Computed CIs for %d compressors from compression_sweep",
            len(output["compression_sweep_cis"]),
        )

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved confidence intervals to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
