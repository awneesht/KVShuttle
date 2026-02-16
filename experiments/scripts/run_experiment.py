"""Config-driven experiment runner for KVShuttle benchmarks."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from kvshuttle.compression.registry import get_compressor, list_compressors
from kvshuttle.evaluation.attention_error import compute_attention_error
from kvshuttle.profiling.timer import timer
from kvshuttle.transfer.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_prompts(count: int, min_tokens: int, max_tokens: int) -> list[str]:
    """Generate synthetic prompts of varying lengths."""
    rng = np.random.default_rng(42)
    prompts = []
    for _ in range(count):
        # Target token count (roughly 4 chars per token)
        target_tokens = rng.integers(min_tokens, max_tokens + 1)
        target_chars = target_tokens * 4
        # Generate a simple repeated pattern
        base = "The quick brown fox jumps over the lazy dog. "
        prompt = (base * (target_chars // len(base) + 1))[:target_chars]
        prompts.append(prompt)
    return prompts


def generate_synthetic_kv_cache(
    num_layers: int, num_heads: int, seq_len: int, head_dim: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic KV cache for testing without model loading."""
    rng = np.random.default_rng(seed)
    shape = (num_layers, num_heads, seq_len, head_dim)
    keys = rng.standard_normal(shape).astype(np.float16)
    values = (rng.standard_normal(shape) * 0.5).astype(np.float16)
    return keys, values


def run_experiment(config_path: str) -> None:
    """Run a full experiment from a YAML config file."""
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp = config["experiment"]
    logger.info("Starting experiment: %s — %s", exp["name"], exp["description"])

    # Setup output directory
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Get compressor names
    requested = config["compressors"]
    available = list_compressors()
    compressor_names = [c for c in requested if c in available]
    if missing := set(requested) - set(available):
        logger.warning("Skipping unavailable compressors: %s", missing)

    bandwidths = [float(b) for b in config["bandwidths_gbps"]]

    # Generate prompts config
    prompt_cfg = config["prompts"]
    prompt_count = prompt_cfg["count"]
    min_tokens = prompt_cfg.get("min_tokens", 64)
    max_tokens = prompt_cfg.get("max_tokens", 512)

    all_results = []
    eval_cfg = config.get("evaluation", {})

    # For each model
    for model_name in config["models"]:
        logger.info("=== Model: %s ===", model_name)

        # Try to load real model, fall back to synthetic
        model_loaded = False
        try:
            from kvshuttle.models.loader import load_model
            from kvshuttle.models.kv_extractor import extract_kv_cache

            model, tokenizer, model_info = load_model(model_name)
            model_loaded = True
            logger.info("Loaded model: %s", model_info)
        except Exception as e:
            logger.warning("Could not load model %s: %s. Using synthetic KV cache.", model_name, e)
            model_info = None

        # Generate token length range for synthetic caches
        rng = np.random.default_rng(42)
        seq_lens = rng.integers(min_tokens, max_tokens + 1, size=prompt_count)

        for prompt_idx in tqdm(range(prompt_count), desc=f"{model_name}"):
            seq_len = int(seq_lens[prompt_idx])

            if model_loaded:
                prompts = generate_synthetic_prompts(1, min_tokens, max_tokens)
                try:
                    kv = extract_kv_cache(model, tokenizer, prompts[0])
                    keys, values = kv.keys, kv.values
                    seq_len = kv.seq_len
                except Exception as e:
                    logger.warning("KV extraction failed: %s. Using synthetic.", e)
                    keys, values = _get_synthetic_kv(model_name, seq_len)
            else:
                keys, values = _get_synthetic_kv(model_name, seq_len)

            # Run each compressor
            for comp_name in compressor_names:
                compressor = get_compressor(comp_name)

                # Compute quality metrics ONCE per (prompt, compressor) — independent of bandwidth
                quality = {}
                if eval_cfg.get("attention_error", False):
                    try:
                        compressed = compressor.compress(keys, values)
                        keys_recon, values_recon = compressor.decompress(compressed)
                        error = compute_attention_error(keys, values, keys_recon, values_recon)
                        quality["mean_key_cosine_sim"] = error.mean_key_cosine_sim
                        quality["mean_val_cosine_sim"] = error.mean_val_cosine_sim
                        quality["mean_key_mse"] = error.mean_key_mse
                        quality["mean_val_mse"] = error.mean_val_mse

                        if config["output"].get("save_per_layer", False):
                            quality["key_cosine_sim_per_layer"] = error.key_cosine_sim_per_layer
                            quality["val_cosine_sim_per_layer"] = error.val_cosine_sim_per_layer
                    except Exception as e:
                        logger.warning("Quality eval failed for %s: %s", comp_name, e)

                for bw in bandwidths:
                    pipeline_result = run_pipeline(compressor, keys, values, bw)

                    result_record = {
                        "model": model_name,
                        "compressor": comp_name,
                        "bandwidth_gbps": bw,
                        "seq_len": seq_len,
                        "prompt_idx": prompt_idx,
                        "compress_ms": pipeline_result.compress_ms,
                        "serialize_ms": pipeline_result.serialize_ms,
                        "transfer_ms": pipeline_result.transfer_ms,
                        "deserialize_ms": pipeline_result.deserialize_ms,
                        "decompress_ms": pipeline_result.decompress_ms,
                        "total_ms": pipeline_result.total_ms,
                        "raw_transfer_ms": pipeline_result.raw_transfer_ms,
                        "speedup": pipeline_result.speedup,
                        "original_bytes": pipeline_result.original_bytes,
                        "compressed_bytes": pipeline_result.compressed_bytes,
                        "compression_ratio": pipeline_result.compression_ratio,
                        **quality,
                    }
                    all_results.append(result_record)

    # Save results
    results_path = output_dir / "results.json"
    metadata = {
        "experiment": exp["name"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_results": len(all_results),
        "models": config["models"],
        "compressors": compressor_names,
        "bandwidths_gbps": bandwidths,
    }

    with open(results_path, "w") as f:
        json.dump({"metadata": metadata, "results": all_results}, f, indent=2)

    logger.info("Saved %d results to %s", len(all_results), results_path)


# Default synthetic KV cache shapes per model
_MODEL_KV_SHAPES = {
    "qwen2.5-3b": (36, 2, 128),     # layers, kv_heads, head_dim
    "llama-3.2-3b": (28, 8, 128),
    "phi-3.5-mini": (32, 32, 96),
    "qwen2.5-7b": (28, 4, 128),
    "llama-3.1-8b": (32, 8, 128),
    "mistral-7b": (32, 8, 128),
}


def _get_synthetic_kv(model_name: str, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Get synthetic KV cache matching model architecture."""
    if model_name in _MODEL_KV_SHAPES:
        num_layers, num_heads, head_dim = _MODEL_KV_SHAPES[model_name]
    else:
        num_layers, num_heads, head_dim = 32, 8, 128
    return generate_synthetic_kv_cache(num_layers, num_heads, seq_len, head_dim)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <config.yaml>")
        sys.exit(1)
    run_experiment(sys.argv[1])
