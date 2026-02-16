"""Generate LaTeX tables from benchmark results."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def _generate_main_table(results: list[dict], output_path: Path) -> None:
    """Generate the main results table."""
    # Aggregate by compressor
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
    # Group by (compressor, bandwidth) and find speedup
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
        print("Usage: python generate_paper_tables.py <results.json> [output_dir]")
        sys.exit(1)
    results_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "paper/tables"
    generate_tables(results_file, out_dir)
