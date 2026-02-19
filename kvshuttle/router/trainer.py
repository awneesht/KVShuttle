"""Train the KVShuttle Router on benchmark data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from kvshuttle.router.features import RouterInput
from kvshuttle.router.learned_router import LearnedRouter

logger = logging.getLogger(__name__)

# Model architecture info for feature extraction
_MODEL_ARCH = {
    "qwen2.5-3b": (36, 2, 128),     # (layers, kv_heads, head_dim)
    "llama-3.2-3b": (28, 8, 128),
    "phi-3.5-mini": (32, 32, 96),
    "qwen2.5-7b": (28, 4, 128),
    "llama-3.1-8b": (32, 8, 128),
    "mistral-7b": (32, 8, 128),
}


def load_training_data(
    results_paths: list[str | Path] | str | Path,
    quality_threshold: float = 0.99,
) -> dict:
    """Load and process benchmark results for router training.

    Args:
        results_paths: Path(s) to results.json from experiment runner.
        quality_threshold: Minimum cosine sim for a compressor to be considered.

    Returns:
        Dict with features, labels, and metadata for training.
    """
    if isinstance(results_paths, (str, Path)):
        results_paths = [results_paths]

    all_results = []
    for path in results_paths:
        with open(path) as f:
            data = json.load(f)
        all_results.extend(data["results"])

    logger.info("Loaded %d results from %d file(s)", len(all_results), len(results_paths))

    # Group by (model, prompt_idx, bandwidth)
    groups: dict[tuple, list] = {}
    for r in all_results:
        key = (r["model"], r["prompt_idx"], r["bandwidth_gbps"])
        groups.setdefault(key, []).append(r)

    features = []
    best_compressors = []
    all_compressors = sorted(set(r["compressor"] for r in all_results))
    comp_to_idx = {name: i for i, name in enumerate(all_compressors)}

    for group_key, group_results in groups.items():
        model_name = group_key[0]

        # Filter by quality
        valid = []
        for r in group_results:
            cos_sim = r.get("mean_key_cosine_sim", 1.0)
            if cos_sim >= quality_threshold or r["compressor"] == "identity":
                valid.append(r)

        if not valid:
            continue

        # Find best (lowest total_ms)
        best = min(valid, key=lambda r: r["total_ms"])
        best_name = best["compressor"]

        if best_name not in comp_to_idx:
            continue

        # Get model architecture
        if model_name in _MODEL_ARCH:
            num_layers, num_kv_heads, head_dim = _MODEL_ARCH[model_name]
        else:
            num_layers, num_kv_heads, head_dim = 32, 8, 128

        # Build feature vector
        r0 = group_results[0]
        ri = RouterInput(
            prompt_length=r0["seq_len"],
            model_num_layers=num_layers,
            model_num_kv_heads=num_kv_heads,
            model_head_dim=head_dim,
            kv_cache_size_bytes=r0["original_bytes"],
            available_bandwidth_gbps=r0["bandwidth_gbps"],
            quality_threshold=quality_threshold,
        )
        features.append(ri.to_feature_vector())
        best_compressors.append(comp_to_idx[best_name])

    return {
        "features": np.array(features),
        "labels": np.array(best_compressors),
        "compressor_names": all_compressors,
        "num_samples": len(features),
    }


def train_routers(
    results_path: str | Path,
    quality_threshold: float = 0.99,
    train_split: float = 0.8,
) -> dict:
    """Train all router types on benchmark data.

    Args:
        results_path: Path to results.json.
        quality_threshold: Minimum quality for compressor selection.
        train_split: Fraction of data for training.

    Returns:
        Dict of trained router objects.
    """
    data = load_training_data(results_path, quality_threshold)
    features = data["features"]
    labels = data["labels"]
    names = data["compressor_names"]

    if len(features) < 10:
        logger.warning("Only %d training samples, results may be unreliable", len(features))

    # Split
    n = len(features)
    idx = np.random.default_rng(42).permutation(n)
    split = int(n * train_split)
    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    routers = {}

    # Decision tree
    dt_router = LearnedRouter.train(X_train, y_train, names, model_type="decision_tree")
    if len(X_test) > 0:
        dt_acc = np.mean(dt_router.model.predict(X_test) == y_test)
        logger.info("Decision tree test accuracy: %.3f", dt_acc)
    routers["decision_tree"] = dt_router

    # MLP
    if len(X_train) >= 20:
        mlp_router = LearnedRouter.train(X_train, y_train, names, model_type="mlp")
        if len(X_test) > 0:
            mlp_acc = np.mean(mlp_router.model.predict(X_test) == y_test)
            logger.info("MLP test accuracy: %.3f", mlp_acc)
        routers["mlp"] = mlp_router

    logger.info("Trained %d routers on %d samples", len(routers), len(X_train))
    return routers
