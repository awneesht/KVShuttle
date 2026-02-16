"""Calibrate CPU-to-GPU speedup for KV cache compression.

Runs both numpy (CPU) and MLX (Metal GPU) implementations of uniform_int8
and kivi_2bit side-by-side, measuring timing differences to establish a
calibration factor. Also compares sequential vs pipelined transfer models.

Usage:
    python3 experiments/scripts/calibrate_gpu.py [--seq-lens 512 1024 2048]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Model architectures: (layers, kv_heads, head_dim)
MODELS = {
    "llama-3.2-3b": (28, 8, 128),
    "qwen2.5-7b": (28, 4, 128),
}

DEFAULT_SEQ_LENS = [256, 512, 1024, 2048]
DEFAULT_BANDWIDTHS = [1.0, 10.0, 50.0, 100.0, 400.0]
WARMUP = 2
REPEATS = 5


def generate_kv_cache(
    num_layers: int, num_heads: int, seq_len: int, head_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic KV cache tensors."""
    rng = np.random.default_rng(42)
    shape = (num_layers, num_heads, seq_len, head_dim)
    keys = rng.standard_normal(shape).astype(np.float16)
    values = rng.standard_normal(shape).astype(np.float16)
    return keys, values


def time_compress_decompress(
    compressor, keys: np.ndarray, values: np.ndarray,
    warmup: int = WARMUP, repeats: int = REPEATS,
) -> dict:
    """Time compression and decompression with warmup."""
    # Warmup
    for _ in range(warmup):
        c = compressor.compress(keys, values)
        compressor.decompress(c)

    # Timed runs
    compress_times = []
    decompress_times = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        c = compressor.compress(keys, values)
        t1 = time.perf_counter_ns()
        compressor.decompress(c)
        t2 = time.perf_counter_ns()

        compress_times.append((t1 - t0) / 1e6)
        decompress_times.append((t2 - t1) / 1e6)

    return {
        "compress_ms": float(np.median(compress_times)),
        "decompress_ms": float(np.median(decompress_times)),
        "compress_std_ms": float(np.std(compress_times)),
        "decompress_std_ms": float(np.std(decompress_times)),
        "compressed_bytes": c.compressed_size_bytes,
        "ratio": c.compression_ratio,
    }


def run_calibration(seq_lens: list[int], bandwidths: list[float]) -> dict:
    """Run full CPU vs GPU calibration."""
    from kvshuttle.compression.registry import get_compressor
    from kvshuttle.transfer.pipeline import run_pipeline, run_pipeline_pipelined

    results = {
        "compressor_calibration": [],
        "pipeline_comparison": [],
    }

    for model_name, (num_layers, num_heads, head_dim) in MODELS.items():
        for seq_len in seq_lens:
            logger.info("=== %s, seq_len=%d ===", model_name, seq_len)
            keys, values = generate_kv_cache(num_layers, num_heads, seq_len, head_dim)
            cache_mb = (keys.nbytes + values.nbytes) / (1024 * 1024)
            logger.info("KV cache size: %.1f MB", cache_mb)

            # --- Compressor calibration ---
            pairs = [
                ("uniform_int8", "mlx_uniform_int8"),
                ("kivi_2bit", "mlx_kivi_2bit"),
            ]

            for cpu_name, gpu_name in pairs:
                cpu_comp = get_compressor(cpu_name)
                gpu_comp = get_compressor(gpu_name)

                cpu_timing = time_compress_decompress(cpu_comp, keys, values)
                gpu_timing = time_compress_decompress(gpu_comp, keys, values)

                compress_speedup = cpu_timing["compress_ms"] / gpu_timing["compress_ms"] if gpu_timing["compress_ms"] > 0 else float("inf")
                decompress_speedup = cpu_timing["decompress_ms"] / gpu_timing["decompress_ms"] if gpu_timing["decompress_ms"] > 0 else float("inf")

                entry = {
                    "model": model_name,
                    "seq_len": seq_len,
                    "cache_mb": round(cache_mb, 2),
                    "compressor": cpu_name,
                    "cpu_compress_ms": round(cpu_timing["compress_ms"], 3),
                    "gpu_compress_ms": round(gpu_timing["compress_ms"], 3),
                    "compress_speedup": round(compress_speedup, 2),
                    "cpu_decompress_ms": round(cpu_timing["decompress_ms"], 3),
                    "gpu_decompress_ms": round(gpu_timing["decompress_ms"], 3),
                    "decompress_speedup": round(decompress_speedup, 2),
                    "compression_ratio": round(cpu_timing["ratio"], 2),
                }
                results["compressor_calibration"].append(entry)

                logger.info(
                    "%s: CPU compress=%.2fms, GPU compress=%.2fms (%.1fx speedup)",
                    cpu_name,
                    cpu_timing["compress_ms"],
                    gpu_timing["compress_ms"],
                    compress_speedup,
                )
                logger.info(
                    "%s: CPU decompress=%.2fms, GPU decompress=%.2fms (%.1fx speedup)",
                    cpu_name,
                    cpu_timing["decompress_ms"],
                    gpu_timing["decompress_ms"],
                    decompress_speedup,
                )

            # --- Pipeline comparison: sequential vs pipelined ---
            for bw in bandwidths:
                for comp_name in ["uniform_int8", "kivi_2bit"]:
                    comp = get_compressor(comp_name)

                    seq_result = run_pipeline(comp, keys, values, bw)
                    pipe_result = run_pipeline_pipelined(comp, keys, values, bw)

                    pipeline_saving = (
                        (seq_result.total_ms - pipe_result.total_ms) / seq_result.total_ms * 100
                        if seq_result.total_ms > 0 else 0
                    )

                    entry = {
                        "model": model_name,
                        "seq_len": seq_len,
                        "bandwidth_gbps": bw,
                        "compressor": comp_name,
                        "sequential_total_ms": round(seq_result.total_ms, 3),
                        "pipelined_total_ms": round(pipe_result.total_ms, 3),
                        "pipeline_saving_pct": round(pipeline_saving, 1),
                        "sequential_speedup": round(seq_result.speedup, 3),
                        "pipelined_speedup": round(pipe_result.speedup, 3),
                        "raw_transfer_ms": round(seq_result.raw_transfer_ms, 3),
                    }
                    results["pipeline_comparison"].append(entry)

                    logger.info(
                        "%s @ %d Gbps: sequential=%.2fms (%.2fx), pipelined=%.2fms (%.2fx), saving=%.1f%%",
                        comp_name, bw,
                        seq_result.total_ms, seq_result.speedup,
                        pipe_result.total_ms, pipe_result.speedup,
                        pipeline_saving,
                    )

    return results


def print_summary(results: dict) -> None:
    """Print formatted summary tables."""
    print("\n" + "=" * 80)
    print("GPU CALIBRATION SUMMARY")
    print("=" * 80)

    print("\n--- CPU vs GPU Compression Timing ---")
    print(f"{'Compressor':<18} {'Model':<15} {'SeqLen':>6} {'CPU(ms)':>8} {'GPU(ms)':>8} {'Speedup':>8}")
    print("-" * 75)
    for e in results["compressor_calibration"]:
        print(
            f"{e['compressor']:<18} {e['model']:<15} {e['seq_len']:>6} "
            f"{e['cpu_compress_ms']:>8.2f} {e['gpu_compress_ms']:>8.2f} "
            f"{e['compress_speedup']:>7.1f}x"
        )

    # Aggregate speedups
    by_comp = {}
    for e in results["compressor_calibration"]:
        by_comp.setdefault(e["compressor"], []).append(e["compress_speedup"])

    print("\n--- Average GPU Speedup Factors ---")
    for comp, speedups in by_comp.items():
        print(f"  {comp}: {np.mean(speedups):.1f}x (range: {min(speedups):.1f}x - {max(speedups):.1f}x)")

    print("\n--- Sequential vs Pipelined Pipeline ---")
    print(f"{'Compressor':<18} {'BW(Gbps)':>8} {'Sequential':>12} {'Pipelined':>12} {'Saving':>8}")
    print("-" * 65)
    for e in results["pipeline_comparison"]:
        print(
            f"{e['compressor']:<18} {e['bandwidth_gbps']:>8.0f} "
            f"{e['sequential_total_ms']:>11.2f}ms {e['pipelined_total_ms']:>11.2f}ms "
            f"{e['pipeline_saving_pct']:>7.1f}%"
        )

    # Aggregate pipeline savings
    savings = [e["pipeline_saving_pct"] for e in results["pipeline_comparison"]]
    if savings:
        print(f"\n  Average pipeline saving: {np.mean(savings):.1f}% "
              f"(range: {min(savings):.1f}% - {max(savings):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Calibrate CPU vs GPU compression timing")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS)
    parser.add_argument("--bandwidths", nargs="+", type=float, default=DEFAULT_BANDWIDTHS)
    parser.add_argument("--output-dir", default="experiments/results/gpu_calibration")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_calibration(args.seq_lens, args.bandwidths)

    # Save results
    output_path = output_dir / "calibration_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)

    print_summary(results)


if __name__ == "__main__":
    main()
