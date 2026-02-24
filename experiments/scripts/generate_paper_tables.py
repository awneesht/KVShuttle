"""Generate LaTeX tables from benchmark results."""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parents[2]


def generate_tables(results_path: str, output_dir: str = "paper/tables") -> None:
    """Generate LaTeX tables from results.

    Args:
        results_path: Path to results.json.
        output_dir: Directory to save .tex files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]

    # Table 1: Main results table (compression ratio, quality, throughput)
    _generate_main_table(results, output_dir / "table1_main.tex")

    # Table 2: Bandwidth break-even points
    _generate_breakeven_table(results, output_dir / "table2_breakeven.tex")

    logger.info("Tables saved to %s", output_dir)


def generate_full_paper_tables(output_dir: str = "paper/tables/full") -> None:
    """Generate all tables needed for the full TMLR paper.

    Reads from multiple result files and the confidence intervals JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all data sources
    comp_sweep_path = PROJ_ROOT / "experiments/results/compression_sweep/results.json"
    gen_quality_path = PROJ_ROOT / "experiments/results/generation_quality_fp16_merged/results.json"
    ci_path = PROJ_ROOT / "experiments/results/confidence_intervals.json"
    router_path = PROJ_ROOT / "experiments/results/router_training/router_results.json"
    model_sweep_path = PROJ_ROOT / "experiments/results/model_sweep/results.json"

    comp_sweep = _load_json(comp_sweep_path)
    gen_quality = _load_json(gen_quality_path)
    ci_data = _load_json(ci_path)
    router_data = _load_json(router_path)
    model_sweep = _load_json(model_sweep_path)

    # Table: Full 14-compressor results
    if comp_sweep and gen_quality:
        _generate_full_14_compressor_table(
            comp_sweep["results"], gen_quality["results"], output_dir / "table_full_14_compressors.tex"
        )

    # Table: Model sensitivity with CIs
    if gen_quality and ci_data:
        _generate_model_sensitivity_with_ci(
            gen_quality["results"], ci_data, output_dir / "table_model_sensitivity_with_ci.tex"
        )

    # Table: KV cache sizes
    if model_sweep:
        _generate_kv_cache_sizes_table(model_sweep["results"], output_dir / "table_kv_cache_sizes.tex")

    # Table: Practitioner recipe
    _generate_recipe_table(output_dir / "table_recipe.tex")

    # Table: Router accuracy
    if router_data:
        _generate_router_accuracy_table(router_data, output_dir / "table_router_accuracy.tex")

    # Table: Statistical significance summary
    if ci_data and "pairwise_tests" in ci_data:
        _generate_significance_table(ci_data, output_dir / "table_significance.tex")

    logger.info("Full paper tables saved to %s", output_dir)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def _generate_full_14_compressor_table(
    comp_results: list[dict], gen_results: list[dict], output_path: Path
) -> None:
    """Full 14-compressor table with ratio, cosine, token agreement, ppl delta."""
    # Aggregate compression_sweep data by compressor
    by_comp: dict[str, list[dict]] = defaultdict(list)
    for r in comp_results:
        by_comp[r["compressor"]].append(r)

    # Aggregate generation quality data by compressor
    gen_by_comp: dict[str, list[dict]] = defaultdict(list)
    for r in gen_results:
        gen_by_comp[r["compressor"]].append(r)

    # Family assignment
    families = {
        "identity": "baseline",
        "uniform_int8": "quantization",
        "uniform_int4": "quantization",
        "fp8_e4m3": "quantization",
        "kivi_2bit": "quantization",
        "kvquant_2bit": "quantization",
        "cachegen": "structured",
        "palu_lr": "low-rank",
        "topk_prune_50": "pruning",
        "cascade_prune50_int4": "hybrid",
        "palu_int4": "hybrid",
        "mixed_k8v4": "hybrid",
        "lossless_zstd": "lossless",
        "lossless_lz4": "lossless",
    }

    # Display order: by family then by ratio
    display_order = [
        "identity",
        "lossless_zstd", "lossless_lz4",
        "uniform_int8", "fp8_e4m3", "uniform_int4", "kivi_2bit", "kvquant_2bit",
        "cachegen",
        "palu_lr",
        "topk_prune_50",
        "mixed_k8v4", "palu_int4", "cascade_prune50_int4",
    ]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{All 14 compression strategies evaluated in KVShuttle. Ratio and cosine similarity are from the compression sweep on Llama-3.1-8B (50 prompts $\times$ 8 bandwidths). Token agreement (TA) and perplexity delta ($\Delta$ppl) are means over 6 models $\times$ 50 prompts each where available. --- indicates the compressor was not evaluated for generation quality.}",
        r"\label{tab:full_compressors}",
        r"\small",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Strategy} & \textbf{Family} & \textbf{Ratio} & \textbf{Key $\cos$} & \textbf{Val $\cos$} & \textbf{TA} & \textbf{$\Delta$ppl} \\",
        r"\midrule",
    ]

    prev_family = None
    for name in display_order:
        if name not in by_comp:
            continue
        entries = by_comp[name]
        family = families.get(name, "other")

        if prev_family is not None and family != prev_family:
            lines.append(r"\midrule")
        prev_family = family

        ratio = np.mean([e["compression_ratio"] for e in entries])
        k_cos = np.mean([e.get("mean_key_cosine_sim", 1.0) for e in entries])
        v_cos = np.mean([e.get("mean_val_cosine_sim", 1.0) for e in entries])

        # Generation quality
        gen_entries = gen_by_comp.get(name, [])
        if gen_entries:
            ta = np.mean([e["token_agreement"] for e in gen_entries if "token_agreement" in e])
            ppl = np.mean([e["perplexity_delta"] for e in gen_entries if "perplexity_delta" in e])
            ta_str = f"{ta:.3f}"
            ppl_str = f"{ppl:.2f}" if ppl < 100 else f"{ppl:,.0f}"
        else:
            ta_str = "---"
            ppl_str = "---"

        ratio_str = f"{ratio:.1f}$\\times$"
        lines.append(
            f"  {_escape_latex(name)} & {family} & {ratio_str} & {k_cos:.4f} & {v_cos:.4f} "
            f"& {ta_str} & {ppl_str} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    output_path.write_text("\n".join(lines))
    logger.info("Generated full 14-compressor table: %s", output_path)


def _generate_model_sensitivity_with_ci(
    gen_results: list[dict], ci_data: dict, output_path: Path
) -> None:
    """Model sensitivity table with 95% CIs on token agreement."""
    # Aggregate by (model, compressor)
    by_model_comp: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in gen_results:
        if "token_agreement" in r:
            by_model_comp[r["model"]][r["compressor"]].append(r["token_agreement"])

    models = sorted(by_model_comp.keys())
    compressors = ["uniform_int8", "cachegen", "uniform_int4", "kivi_2bit", "palu_lr", "cascade_prune50_int4"]

    # Get per-model CIs from ci_data
    per_model_ci = ci_data.get("generation_quality_cis", {}).get("per_model", {})

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Token agreement by model and compressor with 95\% bootstrap confidence intervals (50 prompts per cell). \textcolor{red}{\textbf{Red}} indicates catastrophic failure ($<$0.20).}",
        r"\label{tab:model_sensitivity_ci}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(compressors) + "}",
        r"\toprule",
        r"\textbf{Model} & " + " & ".join(
            r"\textbf{" + _escape_latex(c) + "}" for c in compressors
        ) + r" \\",
        r"\midrule",
    ]

    for model in models:
        cells = []
        for comp in compressors:
            vals = by_model_comp.get(model, {}).get(comp, [])
            if not vals:
                cells.append("---")
                continue
            mean_val = np.mean(vals)
            # Get CI from precomputed data
            ci_entry = per_model_ci.get(comp, {}).get(model, {}).get("token_agreement", {})
            ci_lo = ci_entry.get("ci_lower", mean_val)
            ci_hi = ci_entry.get("ci_upper", mean_val)

            if mean_val < 0.20:
                cell = f"\\textcolor{{red}}{{\\textbf{{{mean_val:.3f}}}}}"
            else:
                cell = f"{mean_val:.3f}"

            # Add CI as subscript
            cell += f" \\tiny{{[{ci_lo:.2f}, {ci_hi:.2f}]}}"
            cells.append(cell)

        model_display = _escape_latex(model.split("/")[-1])
        lines.append(f"  {model_display} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    output_path.write_text("\n".join(lines))
    logger.info("Generated model sensitivity with CI table: %s", output_path)


def _generate_kv_cache_sizes_table(results: list[dict], output_path: Path) -> None:
    """KV cache size table: all models × representative sequence lengths."""
    # Gather unique models and their cache sizes from identity compressor
    model_info: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["compressor"] == "identity":
            model_info[r["model"]][r["seq_len"]].append(r["original_bytes"])

    models = sorted(model_info.keys())

    # Model architecture info (layers, kv_heads, head_dim)
    arch_info = {
        "qwen2.5-3b": (36, 2, 128),
        "llama-3.2-3b": (28, 8, 128),
        "phi-3.5-mini": (32, 32, 96),
        "qwen2.5-7b": (28, 4, 128),
        "llama-3.1-8b": (32, 8, 128),
        "mistral-7b": (32, 8, 128),
        "llama-3.1-70b": (80, 8, 128),
    }

    seq_lens = [128, 256, 512, 1024, 2048]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{KV cache size (MB, FP16) by model and sequence length. Size = $2 \times L \times H_{kv} \times d \times S \times 2$ bytes, where $L$=layers, $H_{kv}$=KV heads, $d$=head dim, $S$=seq len.}",
        r"\label{tab:kv_sizes}",
        r"\small",
        r"\begin{tabular}{lccc" + "c" * len(seq_lens) + "}",
        r"\toprule",
        r"\textbf{Model} & $L$ & $H_{kv}$ & $d$ & " + " & ".join(
            f"\\textbf{{{s}}}" for s in seq_lens
        ) + r" \\",
        r"\midrule",
    ]

    for model in models:
        short_name = model.split("/")[-1]
        L, H, d = arch_info.get(short_name, (0, 0, 0))

        cells = []
        for s in seq_lens:
            # Compute analytically: 2 (K+V) * L * H * d * s * 2 (FP16 bytes)
            size_bytes = 2 * L * H * d * s * 2
            size_mb = size_bytes / (1024 * 1024)
            cells.append(f"{size_mb:.1f}")

        lines.append(
            f"  {_escape_latex(short_name)} & {L} & {H} & {d} & "
            + " & ".join(cells) + r" \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines))
    logger.info("Generated KV cache sizes table: %s", output_path)


def _generate_recipe_table(output_path: Path) -> None:
    """Practitioner recipe table: deployment scenario → recommended compressor."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Recommended compression strategy by deployment scenario. BW = network bandwidth, TA = token agreement, GPU = GPU-accelerated kernels available.}",
        r"\label{tab:recipe}",
        r"\small",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Conditions} & \textbf{Strategy} & \textbf{Expected TA} \\",
        r"\midrule",
        r"NVLink/IB & BW $\geq$ 100\,Gbps & identity (no compression) & 1.000 \\",
        r"Quality-critical & BW $\leq$ 50\,Gbps, GPU & uniform\_int8 & $\geq$0.95 \\",
        r"Balanced & BW $\leq$ 25\,Gbps, GPU & CacheGen & $\geq$0.90 \\",
        r"Max compression & BW $\leq$ 10\,Gbps, GPU & KIVI 2-bit & $\geq$0.70 \\",
        r"CPU-only & BW $\leq$ 1\,Gbps & uniform\_int8 & $\geq$0.95 \\",
        r"Streaming & Variable BW & CacheGen (delta) & $\geq$0.90 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines))
    logger.info("Generated recipe table: %s", output_path)


def _generate_router_accuracy_table(router_data: dict, output_path: Path) -> None:
    """Router accuracy table across quality thresholds."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Learned router accuracy for automated compressor selection. Decision tree, gradient boosting, and MLP classifiers trained on (bandwidth, model features) $\to$ best compressor.}",
        r"\label{tab:router_accuracy}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Quality Threshold} & \textbf{Decision Tree} & \textbf{Grad. Boosting} & \textbf{MLP} \\",
        r"\midrule",
    ]

    for qt_key in sorted(router_data.keys()):
        qt_data = router_data[qt_key]
        qt_val = qt_data["quality_threshold"]
        models = qt_data.get("models", {})

        dt_acc = models.get("decision_tree", {}).get("test_accuracy", 0)
        gb_acc = models.get("gradient_boosting", {}).get("test_accuracy", 0)
        mlp_acc = models.get("mlp", {}).get("test_accuracy", 0)

        lines.append(
            f"  TA $\\geq$ {qt_val:.2f} & {dt_acc:.1%} & {gb_acc:.1%} & {mlp_acc:.1%} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines))
    logger.info("Generated router accuracy table: %s", output_path)


def _generate_significance_table(ci_data: dict, output_path: Path) -> None:
    """Statistical significance summary for pairwise compressor comparisons."""
    pairwise = ci_data.get("pairwise_tests", {})
    if not pairwise:
        return

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Wilcoxon signed-rank tests between compressor pairs on token agreement (50 paired prompts per model, pooled across 6 models). Effect size is Cohen's $d$.}",
        r"\label{tab:significance}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{$W$} & \textbf{$p$-value} & \textbf{$d$} & \textbf{Sig.} \\",
        r"\midrule",
    ]

    # Show key comparisons (sorted by effect size magnitude)
    sorted_pairs = sorted(pairwise.items(), key=lambda x: abs(x[1]["effect_size"]), reverse=True)

    for pair_name, result in sorted_pairs[:12]:
        display_name = pair_name.replace("_vs_", r" vs.\ ").replace("_", r"\_")
        p_val = result["p_value"]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"

        lines.append(
            f"  {display_name} & {result['statistic']:.0f} & {p_str} "
            f"& {result['effect_size']:.2f} & {sig} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines))
    logger.info("Generated significance table: %s", output_path)


def _generate_main_table(results: list[dict], output_path: Path) -> None:
    """Generate the main results table."""
    by_comp: dict[str, list] = {}
    for r in results:
        by_comp.setdefault(r["compressor"], []).append(r)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Compression strategies: ratio, quality, and throughput}",
        r"\label{tab:main}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Strategy & Ratio & Key $\cos$ & Val $\cos$ & Comp. (ms) & Decomp. (ms) \\",
        r"\midrule",
    ]

    for name in sorted(by_comp.keys()):
        entries = by_comp[name]
        ratio = np.mean([e["compression_ratio"] for e in entries])
        k_cos = np.mean([e.get("mean_key_cosine_sim", 1.0) for e in entries])
        v_cos = np.mean([e.get("mean_val_cosine_sim", 1.0) for e in entries])
        comp_ms = np.mean([e["compress_ms"] for e in entries])
        decomp_ms = np.mean([e["decompress_ms"] for e in entries])

        lines.append(
            f"  {_escape_latex(name)} & {ratio:.1f}x & {k_cos:.4f} & {v_cos:.4f} "
            f"& {comp_ms:.2f} & {decomp_ms:.2f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines))
    logger.info("Generated main table: %s", output_path)


def _generate_breakeven_table(results: list[dict], output_path: Path) -> None:
    """Generate the bandwidth break-even table."""
    by_comp_bw: dict[str, dict[float, list[float]]] = {}
    for r in results:
        name = r["compressor"]
        bw = r["bandwidth_gbps"]
        by_comp_bw.setdefault(name, {}).setdefault(bw, []).append(r["speedup"])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Bandwidth break-even points (lowest Gbps where speedup $> 1$)}",
        r"\label{tab:breakeven}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Strategy & Break-even (Gbps) \\",
        r"\midrule",
    ]

    for name in sorted(by_comp_bw.keys()):
        if name == "identity":
            continue
        bw_dict = by_comp_bw[name]
        breakeven = "N/A"
        for bw in sorted(bw_dict.keys()):
            avg_speedup = np.mean(bw_dict[bw])
            if avg_speedup > 1.0:
                breakeven = f"{bw:.0f}"
                break
        lines.append(f"  {_escape_latex(name)} & {breakeven} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.write_text("\n".join(lines))
    logger.info("Generated break-even table: %s", output_path)


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    return text.replace("_", r"\_")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: generate full paper tables
        generate_full_paper_tables()
    elif sys.argv[1] == "--full":
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "paper/tables/full"
        generate_full_paper_tables(out_dir)
    else:
        results_file = sys.argv[1]
        out_dir = sys.argv[2] if len(sys.argv) > 2 else "paper/tables"
        generate_tables(results_file, out_dir)
