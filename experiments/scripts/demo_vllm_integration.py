"""End-to-end demo: KVShuttle integration with disaggregated serving.

Demonstrates the full flow: prefill → compress → transfer → decompress → decode.
Runnable standalone without vLLM installed — uses KVShuttle's own pipeline.

Usage:
    python demo_vllm_integration.py [model_name] [compressor_name]
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_demo(
    model_name: str = "qwen2.5-3b",
    compressor_name: str = "uniform_int8",
) -> None:
    """Run the end-to-end integration demo."""
    from kvshuttle.serving.vllm_adapter import KVShuttleConnector
    from kvshuttle.transfer.simulator import simulate_transfer
    from kvshuttle.evaluation.attention_error import compute_attention_error

    print("=" * 70)
    print("KVShuttle — Disaggregated Serving Integration Demo")
    print("=" * 70)
    print()

    # Step 1: Load model and extract KV cache
    model_loaded = False
    try:
        from kvshuttle.models.loader import load_model
        from kvshuttle.models.kv_extractor import extract_kv_cache

        print(f"[1/5] Loading model: {model_name}...")
        model, tokenizer, model_info = load_model(model_name)
        model_loaded = True

        prompt = "Explain the concept of KV cache compression in transformer models and why it matters for disaggregated serving."
        print(f"[2/5] Running prefill on prompt ({len(prompt)} chars)...")
        kv = extract_kv_cache(model, tokenizer, prompt)
        keys, values = kv.keys, kv.values
        print(f"      Extracted KV cache: {kv}")
    except Exception as e:
        logger.warning("Could not load model %s: %s. Using synthetic KV cache.", model_name, e)
        print(f"[1/5] Using synthetic KV cache (model {model_name} not available)")
        print("[2/5] Generating synthetic KV cache...")
        # Use realistic shapes
        from experiments.scripts.run_experiment import _MODEL_KV_SHAPES, generate_synthetic_kv_cache
        if model_name in _MODEL_KV_SHAPES:
            layers, heads, head_dim = _MODEL_KV_SHAPES[model_name]
        else:
            layers, heads, head_dim = 32, 8, 128
        keys, values = generate_synthetic_kv_cache(layers, heads, 256, head_dim)
        print(f"      Synthetic KV cache: keys={keys.shape}, values={values.shape}, "
              f"size={keys.nbytes + values.nbytes:.1f} bytes")

    original_size = keys.nbytes + values.nbytes

    # Step 2: Compress and serialize (simulated prefill node → network)
    print(f"\n[3/5] Compressing with '{compressor_name}' via KVShuttleConnector...")
    connector = KVShuttleConnector(compressor_name)

    t_start = time.perf_counter_ns()
    wire_bytes = connector.send_kv_cache(keys, values)
    t_compress = (time.perf_counter_ns() - t_start) / 1_000_000

    stats = connector.last_stats
    print(f"      Original:   {original_size:>12,} bytes ({original_size / 1024 / 1024:.1f} MB)")
    print(f"      On wire:    {stats.wire_bytes:>12,} bytes ({stats.wire_bytes / 1024 / 1024:.1f} MB)")
    print(f"      Ratio:      {stats.compression_ratio:.2f}x")
    print(f"      Time:       {t_compress:.2f} ms")

    # Step 3: Simulate transfer at various bandwidths
    print("\n[4/5] Transfer time at various bandwidths:")
    bandwidths = [1, 10, 25, 50, 100, 200, 400]
    print(f"      {'Bandwidth':>12s}  {'Raw (ms)':>10s}  {'Compressed (ms)':>16s}  {'Speedup':>8s}")
    print(f"      {'-'*12}  {'-'*10}  {'-'*16}  {'-'*8}")

    for bw in bandwidths:
        raw = simulate_transfer(original_size, bw)
        compressed = simulate_transfer(len(wire_bytes), bw)
        total_with_overhead = t_compress + compressed.transfer_ms
        speedup = raw.transfer_ms / total_with_overhead if total_with_overhead > 0 else float("inf")
        print(f"      {bw:>8d} Gbps  {raw.transfer_ms:>10.2f}  {compressed.transfer_ms:>16.2f}  {speedup:>7.2f}x")

    # Step 4: Decompress (simulated decode node)
    print("\n[5/5] Decompressing on decode node...")
    t_start = time.perf_counter_ns()
    keys_out, values_out = connector.recv_kv_cache(wire_bytes)
    t_decompress = (time.perf_counter_ns() - t_start) / 1_000_000

    print(f"      Decompressed: keys={keys_out.shape}, values={values_out.shape}")
    print(f"      Time:         {t_decompress:.2f} ms")

    # Step 5: Quality check
    error = compute_attention_error(keys, values, keys_out, values_out)
    print(f"\n--- Quality Metrics ---")
    print(f"  Key cosine similarity:   {error.mean_key_cosine_sim:.6f}")
    print(f"  Value cosine similarity: {error.mean_val_cosine_sim:.6f}")
    print(f"  Key MSE:                 {error.mean_key_mse:.2e}")
    print(f"  Value MSE:               {error.mean_val_mse:.2e}")

    print(f"\n--- Summary ---")
    print(f"  Compressor:        {compressor_name}")
    print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
    print(f"  Compress time:     {t_compress:.2f} ms (CPU)")
    print(f"  Decompress time:   {t_decompress:.2f} ms (CPU)")
    print(f"  Quality:           cos_sim={error.mean_key_cosine_sim:.4f} (keys), "
          f"{error.mean_val_cosine_sim:.4f} (values)")
    print()
    print("This demo shows KVShuttle compressors can be used as drop-in")
    print("compression layers in disaggregated prefill/decode architectures.")
    print("=" * 70)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-3b"
    comp = sys.argv[2] if len(sys.argv) > 2 else "uniform_int8"
    run_demo(model, comp)
