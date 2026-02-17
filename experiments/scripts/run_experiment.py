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
from kvshuttle.datasets import load_dataset_prompts
from kvshuttle.evaluation.attention_error import compute_attention_error
from kvshuttle.evaluation.perplexity import compute_perplexity_delta
from kvshuttle.evaluation.token_agreement import compute_token_agreement
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
    backend = config.get("backend", "mlx")
    logger.info("Starting experiment: %s — %s (backend=%s)", exp["name"], exp["description"], backend)

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
    prompt_source = prompt_cfg.get("source", "synthetic")

    # Load real prompts if source is not synthetic
    real_prompts: list[str] | None = None
    if prompt_source != "synthetic":
        try:
            real_prompts = load_dataset_prompts(
                dataset_name=prompt_source,
                count=prompt_count,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
            )
            logger.info("Loaded %d real prompts from %s", len(real_prompts), prompt_source)
        except Exception as e:
            logger.warning("Failed to load %s prompts: %s. Falling back to synthetic.", prompt_source, e)
            prompt_source = "synthetic"

    all_results = []
    eval_cfg = config.get("evaluation", {})

    # For each model
    for model_name in config["models"]:
        logger.info("=== Model: %s ===", model_name)

        # Try to load real model, fall back to synthetic
        model_loaded = False
        model = tokenizer = model_info = None
        try:
            if backend == "torch":
                from kvshuttle.models.loader_torch import load_model_torch
                from kvshuttle.models.kv_extractor_torch import extract_kv_cache_torch

                model, tokenizer, model_info = load_model_torch(model_name)
                _extract_kv_cache = extract_kv_cache_torch
            else:
                from kvshuttle.models.loader import load_model
                from kvshuttle.models.kv_extractor import extract_kv_cache

                model, tokenizer, model_info = load_model(model_name)
                _extract_kv_cache = extract_kv_cache

            model_loaded = True
            logger.info("Loaded model: %s", model_info)
        except Exception as e:
            logger.warning("Could not load model %s: %s. Using synthetic KV cache.", model_name, e)

        # Generate token length range for synthetic caches
        rng = np.random.default_rng(42)
        seq_lens = rng.integers(min_tokens, max_tokens + 1, size=prompt_count)

        for prompt_idx in tqdm(range(prompt_count), desc=f"{model_name}"):
            seq_len = int(seq_lens[prompt_idx])
            prompt_text = None

            if model_loaded:
                # Use real prompt if available, else generate synthetic
                if real_prompts is not None and prompt_idx < len(real_prompts):
                    prompt_text = real_prompts[prompt_idx]
                else:
                    prompt_text = generate_synthetic_prompts(1, min_tokens, max_tokens)[0]
                try:
                    kv = _extract_kv_cache(model, tokenizer, prompt_text)
                    keys, values = kv.keys, kv.values
                    seq_len = kv.seq_len
                except Exception as e:
                    logger.warning("KV extraction failed: %s. Using synthetic.", e)
                    keys, values = _get_synthetic_kv(model_name, seq_len)
            else:
                keys, values = _get_synthetic_kv(model_name, seq_len)

            # Pre-compute original logits ONCE per (model, prompt) — shared across compressors
            # Uses continuation-based evaluation: split KV into prefix (80%) and
            # continuation (20%), inject prefix KV, forward on continuation tokens.
            logits_orig = None
            token_ids = None
            split_point = None
            continuation_ids = None
            needs_gen_quality = model_loaded and prompt_text is not None and (
                eval_cfg.get("perplexity", False) or eval_cfg.get("token_agreement", False)
            )
            if needs_gen_quality:
                try:
                    if backend == "torch":
                        from kvshuttle.models.kv_injector_torch import forward_continuation_with_kv_cache_torch as forward_continuation_with_kv_cache
                    else:
                        from kvshuttle.models.kv_injector import forward_continuation_with_kv_cache

                    # Tokenize with chat template (matching what extract_kv_cache used)
                    all_token_ids = _tokenize_prompt(tokenizer, prompt_text)

                    # Split at 80%: prefix KV + continuation tokens
                    total_len = len(all_token_ids)
                    split_point = max(1, int(total_len * 0.8))
                    # Ensure at least 10 continuation tokens
                    split_point = min(split_point, total_len - 10) if total_len > 10 else max(1, total_len - 1)
                    continuation_ids = all_token_ids[split_point:]

                    # Truncate original KV to prefix length
                    keys_prefix = keys[:, :, :split_point, :]
                    values_prefix = values[:, :, :split_point, :]

                    logits_orig = forward_continuation_with_kv_cache(
                        model, keys_prefix, values_prefix, continuation_ids
                    )
                    token_ids = np.array(continuation_ids)
                except Exception as e:
                    logger.warning("Original logits computation failed: %s", e, exc_info=True)
                    needs_gen_quality = False

            # Run each compressor
            for comp_name in compressor_names:
                compressor = get_compressor(comp_name)

                # Compute quality metrics ONCE per (prompt, compressor) — independent of bandwidth
                quality = {}
                keys_recon = values_recon = None
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

                # End-to-end generation quality (perplexity, token agreement)
                if needs_gen_quality and logits_orig is not None:
                    try:
                        if backend == "torch":
                            from kvshuttle.models.kv_injector_torch import forward_continuation_with_kv_cache_torch as forward_continuation_with_kv_cache
                        else:
                            from kvshuttle.models.kv_injector import forward_continuation_with_kv_cache

                        if keys_recon is None:
                            compressed = compressor.compress(keys, values)
                            keys_recon, values_recon = compressor.decompress(compressed)

                        # Truncate reconstructed KV to prefix length
                        keys_recon_prefix = keys_recon[:, :, :split_point, :]
                        values_recon_prefix = values_recon[:, :, :split_point, :]

                        logits_recon = forward_continuation_with_kv_cache(
                            model, keys_recon_prefix, values_recon_prefix, continuation_ids
                        )

                        if eval_cfg.get("perplexity", False):
                            # Continuation logits[i] predicts continuation_ids[i+1]
                            # So we use logits[:-1] and token_ids[1:]
                            min_len = min(logits_orig.shape[0], logits_recon.shape[0], len(token_ids)) - 1
                            quality["perplexity_delta"] = compute_perplexity_delta(
                                logits_orig[:min_len], logits_recon[:min_len], token_ids[1:min_len + 1]
                            )
                        if eval_cfg.get("token_agreement", False):
                            min_len = min(logits_orig.shape[0], logits_recon.shape[0])
                            quality["token_agreement"] = compute_token_agreement(
                                logits_orig[:min_len], logits_recon[:min_len]
                            )
                    except Exception as e:
                        logger.warning("Generation quality eval failed for %s: %s", comp_name, e, exc_info=True)

                for bw in bandwidths:
                    pipeline_result = run_pipeline(compressor, keys, values, bw)

                    result_record = {
                        "model": model_name,
                        "compressor": comp_name,
                        "bandwidth_gbps": bw,
                        "seq_len": seq_len,
                        "prompt_idx": prompt_idx,
                        "prompt_source": prompt_source,
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
        "backend": backend,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_results": len(all_results),
        "models": config["models"],
        "compressors": compressor_names,
        "bandwidths_gbps": bandwidths,
        "prompt_source": prompt_source,
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


def _tokenize_prompt(tokenizer, prompt: str) -> list[int]:
    """Tokenize a prompt with chat template, returning a plain list of ints."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        result = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        if isinstance(result, str):
            return tokenizer.encode(result)
        return [int(x) for x in result]
    return tokenizer.encode(prompt)


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
