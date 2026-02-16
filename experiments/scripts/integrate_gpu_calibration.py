"""Integrate GPU calibration results into paper figures and tables.

Reads the main benchmark results (model_sweep) and GPU calibration results,
applies GPU speedup factors to compute GPU-adjusted timing, and generates
all updated paper figures and LaTeX tables.

Usage:
    python integrate_gpu_calibration.py \
        experiments/results/model_sweep/results.json \
        experiments/notebooks/gpu_calibration_results.json \
        [output_dir]
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

# Reuse existing visualization modules
from kvshuttle.visualization.pareto import plot_pareto_frontier
from kvshuttle.visualization.bandwidth_sweep import plot_bandwidth_sweep, plot_speedup_curves
from kvshuttle.visualization.heatmaps import plot_layer_sensitivity_heatmap
from kvshuttle.visualization.kv_comparison import plot_kv_comparison
from kvshuttle.visualization.gpu_calibration import (
    plot_cpu_vs_gpu_bars,
    plot_gpu_speedup_scaling,
    plot_pipeline_comparison,
    plot_gpu_calibrated_speedup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── GPU speedup factor mapping ──────────────────────────────────────────
# Maps (compressor, operation) → list of calibration entries.
# We derive per-model, per-seq_len factors and also use overall means.

_GPU_COMPRESSORS = {
    "uniform_int8", "kivi_2bit", "uniform_int4", "fp8_e4m3",
    "cachegen", "cascade_prune50_int4", "palu_lr",
}

# Maps calibration JSON section keys → compressor names
_CALIBRATION_SECTIONS = [
    ("int8_calibration", "uniform_int8"),
    ("kivi_calibration", "kivi_2bit"),
    ("int4_calibration", "uniform_int4"),
    ("fp8_calibration", "fp8_e4m3"),
    ("cachegen_calibration", "cachegen"),
    ("cascade_calibration", "cascade_prune50_int4"),
    ("palu_calibration", "palu_lr"),
]


def _build_speedup_lookup(gpu_cal: dict) -> dict:
    """Build lookup: (compressor) → {compress_speedup, decompress_speedup}.

    Uses mean speedup factors across all models and seq_lens, since the
    model_sweep seq_lens (~173 tokens) don't match the calibration grid
    (256-2048). The mean is a conservative, representative estimate.
    """
    lookup = {}

    for section_key, comp_name in _CALIBRATION_SECTIONS:
        entries = gpu_cal.get(section_key, [])
        if entries:
            lookup[comp_name] = {
                "compress_speedup": float(np.mean([e["compress_speedup"] for e in entries])),
                "decompress_speedup": float(np.mean([e["decompress_speedup"] for e in entries])),
            }

    return lookup


def _get_gpu_factor(lookup, compressor):
    """Get mean GPU speedup factor for a compressor, or None if not calibrated."""
    if compressor not in _GPU_COMPRESSORS:
        return None
    return lookup.get(compressor)


def _apply_gpu_timing(result: dict, factor: dict) -> dict:
    """Apply GPU speedup to a single benchmark result, returning GPU-adjusted copy."""
    r = dict(result)
    r["cpu_compress_ms"] = r["compress_ms"]
    r["cpu_decompress_ms"] = r["decompress_ms"]
    r["compress_ms"] = r["compress_ms"] / factor["compress_speedup"]
    r["decompress_ms"] = r["decompress_ms"] / factor["decompress_speedup"]
    # Recompute total_ms = compress + serialize + transfer + deserialize + decompress
    r["total_ms"] = (r["compress_ms"] + r.get("serialize_ms", 0)
                     + r["transfer_ms"] + r.get("deserialize_ms", 0)
                     + r["decompress_ms"])
    r["speedup"] = r["raw_transfer_ms"] / r["total_ms"] if r["total_ms"] > 0 else float("inf")
    r["gpu_calibrated"] = True
    r["compress_speedup_factor"] = factor["compress_speedup"]
    r["decompress_speedup_factor"] = factor["decompress_speedup"]
    return r


# ── Table generators ────────────────────────────────────────────────────

def _escape_latex(text: str) -> str:
    return text.replace("_", r"\_")


def generate_table1_gpu(results: list[dict], gpu_lookup: dict, output_path: Path) -> None:
    """Table 1: Main results with CPU and GPU timing columns."""
    by_comp: dict[str, list] = {}
    for r in results:
        by_comp.setdefault(r["compressor"], []).append(r)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Compression strategies with GPU-calibrated timing (Tesla T4)}",
        r"\label{tab:main_gpu}",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"Strategy & Ratio & Key $\cos$ & Val $\cos$ & \multicolumn{2}{c}{CPU (ms)} & \multicolumn{2}{c}{GPU (ms)} & \multicolumn{2}{c}{Speedup} \\",
        r"\cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}",
        r" & & & & Comp. & Decomp. & Comp. & Decomp. & Comp. & Decomp. \\",
        r"\midrule",
    ]

    for name in sorted(by_comp.keys()):
        entries = by_comp[name]
        ratio = np.mean([e["compression_ratio"] for e in entries])
        k_cos = np.mean([e.get("mean_key_cosine_sim", 1.0) for e in entries])
        v_cos = np.mean([e.get("mean_val_cosine_sim", 1.0) for e in entries])
        comp_ms = np.mean([e["compress_ms"] for e in entries])
        decomp_ms = np.mean([e["decompress_ms"] for e in entries])

        # GPU timing
        factor = _get_gpu_factor(gpu_lookup, name)
        if factor:
            gpu_comp = comp_ms / factor["compress_speedup"]
            gpu_decomp = decomp_ms / factor["decompress_speedup"]
            lines.append(
                f"  {_escape_latex(name)} & {ratio:.1f}x & {k_cos:.4f} & {v_cos:.4f} "
                f"& {comp_ms:.1f} & {decomp_ms:.1f} "
                f"& {gpu_comp:.2f} & {gpu_decomp:.2f} "
                f"& {factor['compress_speedup']:.0f}$\\times$ & {factor['decompress_speedup']:.0f}$\\times$ \\\\"
            )
        else:
            lines.append(
                f"  {_escape_latex(name)} & {ratio:.1f}x & {k_cos:.4f} & {v_cos:.4f} "
                f"& {comp_ms:.1f} & {decomp_ms:.1f} "
                f"& --- & --- & --- & --- \\\\"
            )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    output_path.write_text("\n".join(lines))
    logger.info("Generated GPU main table: %s", output_path)


def generate_table2_gpu_breakeven(
    cpu_results: list[dict],
    gpu_results: list[dict],
    pipeline_data: list[dict],
    output_path: Path,
) -> None:
    """Table 2: Break-even bandwidth — CPU sequential, GPU sequential, GPU pipelined.

    Uses GPU calibration pipeline data (finer bandwidth grid: 1-400 Gbps)
    for GPU sequential and pipelined columns. Falls back to main results
    for CPU sequential column.
    """
    def _find_breakeven(entries):
        by_bw: dict[float, list[float]] = {}
        for r in entries:
            by_bw.setdefault(r["bandwidth_gbps"], []).append(r["speedup"])
        for bw in sorted(by_bw.keys()):
            if np.mean(by_bw[bw]) > 1.0:
                return f"{bw:.0f}"
        return "$>$100"

    def _find_max_bw(entries, key):
        """Find the highest bandwidth where speedup > 1 (upper break-even)."""
        by_bw: dict[float, list[float]] = {}
        for e in entries:
            by_bw.setdefault(e["bandwidth_gbps"], []).append(e[key])
        max_bw = None
        for bw in sorted(by_bw.keys()):
            if np.mean(by_bw[bw]) > 1.0:
                max_bw = bw
        return max_bw

    # CPU break-even from main results (max beneficial BW)
    cpu_be: dict[str, str] = {}
    cpu_by_comp: dict[str, list] = {}
    for r in cpu_results:
        cpu_by_comp.setdefault(r["compressor"], []).append(r)
    for name, entries in cpu_by_comp.items():
        if name == "identity":
            continue
        # Find max bandwidth where CPU sequential speedup > 1
        by_bw: dict[float, list[float]] = {}
        for e in entries:
            by_bw.setdefault(e["bandwidth_gbps"], []).append(e["speedup"])
        max_bw = None
        for bw in sorted(by_bw.keys()):
            if np.mean(by_bw[bw]) > 1.0:
                max_bw = bw
        cpu_be[name] = f"$\\leq${int(max_bw)}" if max_bw else "N/A"

    # GPU sequential and pipelined break-even from GPU calibration pipeline data
    # (has finer bandwidth grid: 1, 5, 10, 25, 50, 100, 200, 400 Gbps)
    gpu_seq_be: dict[str, str] = {}
    gpu_pipe_be: dict[str, str] = {}
    pipe_by_comp: dict[str, list] = {}
    for e in pipeline_data:
        pipe_by_comp.setdefault(e["compressor"], []).append(e)

    for name, entries in pipe_by_comp.items():
        # Sequential: find max bandwidth where sequential_speedup > 1
        max_seq = _find_max_bw(entries, "sequential_speedup")
        gpu_seq_be[name] = f"$\\leq${int(max_seq)}" if max_seq else "N/A"

        # Pipelined: find max bandwidth where pipelined_speedup > 1
        max_pipe = _find_max_bw(entries, "pipelined_speedup")
        gpu_pipe_be[name] = f"$\\leq${int(max_pipe)}" if max_pipe else "N/A"

    # Also include non-calibrated compressors from CPU results
    all_names = sorted(set(list(cpu_be.keys()) + list(gpu_seq_be.keys())))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Maximum beneficial bandwidth (highest Gbps where speedup $> 1$)}",
        r"\label{tab:breakeven_gpu}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Strategy & CPU Seq. & GPU Seq. & GPU Pipelined \\",
        r"\midrule",
    ]

    for name in all_names:
        c = cpu_be.get(name, "---")
        g = gpu_seq_be.get(name, "---")
        p = gpu_pipe_be.get(name, "---")
        lines.append(f"  {_escape_latex(name)} & {c} & {g} & {p} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path.write_text("\n".join(lines))
    logger.info("Generated GPU break-even table: %s", output_path)


def generate_table3_gpu_summary(gpu_cal: dict, output_path: Path) -> None:
    """Table 3: GPU calibration summary — speedup factors by compressor."""
    summary = gpu_cal["summary"]
    meta = gpu_cal["metadata"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{GPU kernel speedup over CPU numpy (" + _escape_latex(meta["gpu"]) + r")}",
        r"\label{tab:gpu_summary}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Compressor & Operation & Mean Speedup & Range \\",
        r"\midrule",
    ]

    # Iterate over all calibration sections dynamically
    for i, (section_key, comp_name) in enumerate(_CALIBRATION_SECTIONS):
        entries = gpu_cal.get(section_key, [])
        if not entries:
            continue

        comp_sp = [e["compress_speedup"] for e in entries]
        dec_sp = [e["decompress_speedup"] for e in entries]

        lines.append(
            f"  {_escape_latex(comp_name)} & Compress & {np.mean(comp_sp):.0f}$\\times$ "
            f"& [{min(comp_sp):.0f}$\\times$, {max(comp_sp):.0f}$\\times$] \\\\"
        )
        lines.append(
            f"  {_escape_latex(comp_name)} & Decompress & {np.mean(dec_sp):.0f}$\\times$ "
            f"& [{min(dec_sp):.0f}$\\times$, {max(dec_sp):.0f}$\\times$] \\\\"
        )

        # Add midrule between compressors (but not after the last one)
        remaining = [s for s in _CALIBRATION_SECTIONS[i+1:] if gpu_cal.get(s[0])]
        if remaining:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path.write_text("\n".join(lines))
    logger.info("Generated GPU summary table: %s", output_path)


# ── Main integration ────────────────────────────────────────────────────

def integrate(
    main_results_path: str,
    gpu_cal_path: str,
    output_dir: str = "paper/figures/gpu_calibrated",
) -> None:
    """Generate all GPU-calibrated paper assets."""
    output_dir = Path(output_dir)
    fig_dir = output_dir
    table_dir = output_dir.parent.parent / "tables" / "gpu_calibrated"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(main_results_path) as f:
        main_data = json.load(f)
    with open(gpu_cal_path) as f:
        gpu_cal = json.load(f)

    results = main_data["results"]
    gpu_lookup = _build_speedup_lookup(gpu_cal)
    gpu_name = gpu_cal["metadata"]["gpu"]

    logger.info("Loaded %d main results, GPU: %s", len(results), gpu_name)

    # ── Apply GPU calibration to main results ──
    gpu_results = []
    for r in results:
        factor = _get_gpu_factor(gpu_lookup, r["compressor"])
        if factor:
            gpu_results.append(_apply_gpu_timing(r, factor))
        else:
            gpu_results.append(dict(r))

    # ══════════════════════════════════════════════════════════════════
    # TABLES
    # ══════════════════════════════════════════════════════════════════

    logger.info("Generating tables...")

    # Table 1: Main results with GPU columns
    generate_table1_gpu(results, gpu_lookup, table_dir / "table1_main_gpu.tex")

    # Table 2: Break-even comparison
    # Aggregate pipeline data from GPU calibration: average across models for each (compressor, bw)
    pipeline_raw = gpu_cal.get("pipeline_comparison", [])
    pipe_agg: dict[tuple, list] = {}
    for e in pipeline_raw:
        key = (e["compressor"], e["bandwidth_gbps"])
        pipe_agg.setdefault(key, []).append(e)
    pipeline_avg = []
    for (comp, bw), entries in pipe_agg.items():
        pipeline_avg.append({
            "compressor": comp,
            "bandwidth_gbps": bw,
            "sequential_speedup": float(np.mean([e["sequential_speedup"] for e in entries])),
            "pipelined_speedup": float(np.mean([e["pipelined_speedup"] for e in entries])),
            "pipeline_saving_pct": float(np.mean([e["pipeline_saving_pct"] for e in entries])),
            "bottleneck_stage": entries[0]["bottleneck_stage"],
        })

    generate_table2_gpu_breakeven(results, gpu_results, pipeline_avg, table_dir / "table2_breakeven_gpu.tex")

    # Table 3: GPU summary
    generate_table3_gpu_summary(gpu_cal, table_dir / "table3_gpu_summary.tex")

    # ══════════════════════════════════════════════════════════════════
    # FIGURES (existing ones, now GPU-calibrated)
    # ══════════════════════════════════════════════════════════════════

    logger.info("Generating figures...")

    # ── Fig 1: Pareto frontier (unchanged — quality not affected by GPU) ──
    comp_ratios = {}
    cosine_k = {}
    cosine_v = {}
    for r in results:
        name = r["compressor"]
        comp_ratios.setdefault(name, []).append(r["compression_ratio"])
        if "mean_key_cosine_sim" in r:
            cosine_k.setdefault(name, []).append(r["mean_key_cosine_sim"])
            cosine_v.setdefault(name, []).append(r.get("mean_val_cosine_sim", 1.0))
    avg_ratios = {n: float(np.mean(v)) for n, v in comp_ratios.items()}
    avg_k = {n: float(np.mean(v)) for n, v in cosine_k.items()}
    avg_v = {n: float(np.mean(v)) for n, v in cosine_v.items()}

    if avg_k:
        plot_pareto_frontier(avg_ratios, avg_k, fig_dir / "fig1_pareto.pdf")

    # ── Fig 2: Bandwidth sweep (GPU-calibrated latencies) ──
    bw_data: dict[str, dict[float, list[float]]] = {}
    raw_xfer: dict[float, list[float]] = {}
    for r in gpu_results:
        name = r["compressor"]
        bw = r["bandwidth_gbps"]
        bw_data.setdefault(name, {}).setdefault(bw, []).append(r["total_ms"])
        raw_xfer.setdefault(bw, []).append(r["raw_transfer_ms"])
    bw_avg = {n: {bw: float(np.mean(v)) for bw, v in d.items()} for n, d in bw_data.items()}
    raw_avg = {bw: float(np.mean(v)) for bw, v in raw_xfer.items()}

    plot_bandwidth_sweep(bw_avg, fig_dir / "fig2_bandwidth_sweep.pdf",
                         title=f"GPU-Calibrated Pipeline Latency ({gpu_name})")

    # ── Fig 3: Speedup curves (GPU-calibrated, from pipeline data) ──
    # IMPORTANT: Use GPU calibration pipeline data (seq=512-2048) NOT model_sweep
    # data (seq~173). Short sequences make GPU overhead negligible, which would
    # overstate the speedup at high bandwidth and contradict the break-even table.
    pipe_speedup_data: dict[str, dict[float, float]] = {}
    pipe_raw_data: dict[float, float] = {}
    for e in pipeline_raw:
        comp = e["compressor"]
        bw = e["bandwidth_gbps"]
        # Use sequential speedup (not pipelined) for the main speedup curve
        pipe_speedup_data.setdefault(comp, {}).setdefault(bw, []).append(e["sequential_total_ms"])
        pipe_raw_data.setdefault(bw, []).append(e["raw_transfer_ms"])

    # Average across models/seq_lens
    pipe_bw_avg = {
        n: {bw: float(np.mean(v)) for bw, v in d.items()}
        for n, d in pipe_speedup_data.items()
    }
    pipe_raw_avg = {bw: float(np.mean(v)) for bw, v in pipe_raw_data.items()}

    plot_speedup_curves(pipe_bw_avg, pipe_raw_avg, fig_dir / "fig3_speedup_curves.pdf",
                        title=f"GPU-Calibrated Speedup (seq=512-2048, {gpu_name})")

    # ── Fig 4: K/V comparison (unchanged) ──
    if avg_k and avg_v:
        plot_kv_comparison(avg_k, avg_v, fig_dir / "fig4_kv_comparison.pdf")

    # ── Fig 5: Layer sensitivity (unchanged) ──
    layer_data: dict[str, list[list[float]]] = {}
    for r in results:
        if "key_cosine_sim_per_layer" in r:
            layer_data.setdefault(r["compressor"], []).append(r["key_cosine_sim_per_layer"])
    if layer_data:
        avg_layer = {}
        for name, layers in layer_data.items():
            by_nl: dict[int, list[list[float]]] = {}
            for l in layers:
                by_nl.setdefault(len(l), []).append(l)
            most_common = max(by_nl.keys(), key=lambda k: len(by_nl[k]))
            avg_layer[name] = np.mean(np.array(by_nl[most_common]), axis=0).tolist()
        plot_layer_sensitivity_heatmap(avg_layer, fig_dir / "fig5_layer_sensitivity.pdf")

    # ══════════════════════════════════════════════════════════════════
    # NEW FIGURES (GPU-specific)
    # ══════════════════════════════════════════════════════════════════

    # ── Fig 6: CPU vs GPU kernel timing bar chart ──
    # Use calibration data averaged across models for seq=1024 as reference
    bar_data = {}
    for section_key, comp_name in _CALIBRATION_SECTIONS:
        cal_entries = gpu_cal.get(section_key, [])
        entries = [e for e in cal_entries if e["seq_len"] == 1024]
        if entries:
            bar_data[comp_name] = {
                "cpu_compress_ms": float(np.mean([e["cpu_compress_ms"] for e in entries])),
                "gpu_compress_ms": float(np.mean([e["gpu_compress_ms"] for e in entries])),
                "cpu_decompress_ms": float(np.mean([e["cpu_decompress_ms"] for e in entries])),
                "gpu_decompress_ms": float(np.mean([e["gpu_decompress_ms"] for e in entries])),
                "compress_speedup": float(np.mean([e["compress_speedup"] for e in entries])),
                "decompress_speedup": float(np.mean([e["decompress_speedup"] for e in entries])),
            }
    if bar_data:
        plot_cpu_vs_gpu_bars(bar_data, fig_dir / "fig6_cpu_vs_gpu_timing.pdf",
                             title=f"CPU vs GPU Kernel Time (seq=1024, {gpu_name})")

    # ── Fig 7: GPU speedup scaling with cache size ──
    scaling = {}
    for section_key, comp_name in _CALIBRATION_SECTIONS:
        cal_entries = gpu_cal.get(section_key, [])
        if not cal_entries:
            continue
        # Average across models for each cache size
        by_size: dict[float, list] = {}
        for e in cal_entries:
            by_size.setdefault(e["cache_mb"], []).append(e)
        scaling[comp_name] = sorted([
            {
                "cache_mb": mb,
                "compress_speedup": float(np.mean([x["compress_speedup"] for x in es])),
                "decompress_speedup": float(np.mean([x["decompress_speedup"] for x in es])),
            }
            for mb, es in by_size.items()
        ], key=lambda x: x["cache_mb"])
    if scaling:
        plot_gpu_speedup_scaling(scaling, fig_dir / "fig7_gpu_speedup_scaling.pdf",
                                  title=f"GPU Speedup vs KV Cache Size ({gpu_name})")

    # ── Fig 8: Pipeline comparison (sequential vs pipelined) ──
    # Use llama-3.1-8b seq=1024 from GPU calibration pipeline data
    pipe_fig_data = [e for e in pipeline_raw
                     if e["model"] == "llama-3.1-8b" and e["seq_len"] == 1024]
    if pipe_fig_data:
        plot_pipeline_comparison(pipe_fig_data, fig_dir / "fig8_pipeline_comparison.pdf",
                                  title=f"Sequential vs Pipelined ({gpu_name}, LLaMA-3.1-8B, seq=1024)")

    # ── Fig 9: CPU vs GPU calibrated speedup overlay ──
    # CPU: use model_sweep data (all compressors)
    # GPU: use pipeline_comparison data (realistic seq_lens, consistent with Fig 3)
    cpu_bw: dict[str, dict[float, list[float]]] = {}
    cpu_raw: dict[float, list[float]] = {}
    for r in results:
        n = r["compressor"]
        bw = r["bandwidth_gbps"]
        cpu_bw.setdefault(n, {}).setdefault(bw, []).append(r["total_ms"])
        cpu_raw.setdefault(bw, []).append(r["raw_transfer_ms"])
    cpu_bw_avg = {n: {bw: float(np.mean(v)) for bw, v in d.items()} for n, d in cpu_bw.items()}
    cpu_raw_avg = {bw: float(np.mean(v)) for bw, v in cpu_raw.items()}

    # GPU: use pipeline data (seq=512-2048, realistic overhead)
    plot_gpu_calibrated_speedup(
        cpu_bw_avg, pipe_bw_avg, pipe_raw_avg,
        fig_dir / "fig9_cpu_vs_gpu_speedup.pdf",
        title=f"Speedup: CPU-Timed (dashed) vs GPU-Calibrated (solid) — {gpu_name}",
    )

    # ══════════════════════════════════════════════════════════════════
    # Summary report
    # ══════════════════════════════════════════════════════════════════

    summary = gpu_cal["summary"]

    logger.info("=" * 70)
    logger.info("GPU CALIBRATION INTEGRATION COMPLETE")
    logger.info("=" * 70)
    logger.info("GPU: %s | PyTorch: %s | CUDA: %s",
                gpu_name, gpu_cal["metadata"]["pytorch_version"],
                gpu_cal["metadata"]["cuda_version"])

    # Log speedup for all calibrated compressors
    for section_key, comp_name in _CALIBRATION_SECTIONS:
        entries = gpu_cal.get(section_key, [])
        if not entries:
            continue
        comp_sp = [e["compress_speedup"] for e in entries]
        dec_sp = [e["decompress_speedup"] for e in entries]
        logger.info("%-25s compress: %4.0fx (%4.0f-%4.0fx)  decompress: %4.0fx (%4.0f-%4.0fx)",
                    comp_name,
                    np.mean(comp_sp), min(comp_sp), max(comp_sp),
                    np.mean(dec_sp), min(dec_sp), max(dec_sp))

    logger.info("Pipeline saving:         %.1f%% (%.1f-%.1f%%)",
                summary["pipeline_saving_mean_pct"],
                summary["pipeline_saving_range_pct"][0],
                summary["pipeline_saving_range_pct"][1])
    logger.info("-" * 70)
    logger.info("Figures saved to: %s", fig_dir)
    logger.info("Tables saved to:  %s", table_dir)
    logger.info("Generated: 9 figures + 3 tables")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python integrate_gpu_calibration.py <results.json> <gpu_calibration.json> [output_dir]")
        sys.exit(1)
    main_path = sys.argv[1]
    gpu_path = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "paper/figures/gpu_calibrated"
    integrate(main_path, gpu_path, out)
