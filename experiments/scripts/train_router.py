"""Train and evaluate the KVShuttle Router on benchmark data."""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from kvshuttle.router.evaluator import evaluate_router
from kvshuttle.router.features import RouterInput
from kvshuttle.router.learned_router import LearnedRouter
from kvshuttle.router.lookup_table import LookupTableRouter
from kvshuttle.router.trainer import load_training_data, _MODEL_ARCH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def train_and_evaluate(
    results_paths: list[str],
    quality_threshold: float = 0.99,
    output_dir: str = "experiments/results/router_training",
) -> None:
    """Train all router types and evaluate against oracle.

    Args:
        results_paths: Paths to results.json files (can combine multiple sweeps).
        quality_threshold: Minimum cosine similarity for compressor selection.
        output_dir: Directory to save router evaluation results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    data = load_training_data(results_paths, quality_threshold)
    features = data["features"]
    labels = data["labels"]
    names = data["compressor_names"]
    n_samples = data["num_samples"]

    logger.info("Training data: %d samples, %d compressors", n_samples, len(names))
    logger.info("Compressor names: %s", names)

    # Label distribution
    label_counts = Counter(labels)
    logger.info("Label distribution:")
    for idx, count in sorted(label_counts.items()):
        logger.info("  %s: %d (%.1f%%)", names[idx], count, 100 * count / n_samples)

    # Train/test split
    rng = np.random.default_rng(42)
    idx = rng.permutation(n_samples)
    split = int(n_samples * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    logger.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))

    # Also load raw results for oracle/regret evaluation
    all_results = []
    for path in results_paths:
        with open(path) as f:
            all_results.extend(json.load(f)["results"])

    # --- Train routers ---
    routers = {}
    router_eval_results = {}

    # 1. Decision Tree
    logger.info("=== Training Decision Tree ===")
    dt = LearnedRouter.train(X_train, y_train, names, model_type="decision_tree")
    dt_pred = dt.model.predict(X_test)
    dt_acc = np.mean(dt_pred == y_test)
    logger.info("Decision Tree test accuracy: %.3f", dt_acc)
    routers["decision_tree"] = dt

    # Feature importance
    importances = dt.model.feature_importances_
    feat_names = RouterInput.feature_names()
    logger.info("Decision Tree feature importances:")
    for fname, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
        if imp > 0.01:
            logger.info("  %s: %.3f", fname, imp)

    # 2. MLP
    logger.info("=== Training MLP ===")
    mlp = LearnedRouter.train(X_train, y_train, names, model_type="mlp")
    mlp_pred = mlp.model.predict(X_test)
    mlp_acc = np.mean(mlp_pred == y_test)
    logger.info("MLP test accuracy: %.3f", mlp_acc)
    routers["mlp"] = mlp

    # 3. Gradient Boosting
    logger.info("=== Training Gradient Boosting ===")
    gb = LearnedRouter.train(X_train, y_train, names, model_type="gradient_boosting")
    gb_pred = gb.model.predict(X_test)
    gb_acc = np.mean(gb_pred == y_test)
    logger.info("Gradient Boosting test accuracy: %.3f", gb_acc)
    routers["gradient_boosting"] = gb

    # 4. Lookup Table Router
    logger.info("=== Training Lookup Table ===")
    lt_router = _train_lookup_table(all_results, quality_threshold)
    routers["lookup_table"] = lt_router

    # 5. Fixed baselines (always pick one compressor)
    for baseline_name in ["identity", "uniform_int8", "kivi_2bit"]:
        if baseline_name in names:
            routers[f"fixed_{baseline_name}"] = baseline_name

    # --- Evaluate all routers via regret analysis ---
    logger.info("\n=== Regret Evaluation ===")

    # Build oracle: for each (model, prompt_idx, bandwidth), find best compressor
    groups: dict[tuple, list] = {}
    for r in all_results:
        key = (r["model"], r["prompt_idx"], r["bandwidth_gbps"])
        groups.setdefault(key, []).append(r)

    # Only use test set groups for evaluation
    test_groups = _get_test_groups(groups, test_idx, all_results, quality_threshold, names)

    for router_name, router in routers.items():
        router_times = []
        oracle_times = []

        for group_key, group_info in test_groups.items():
            oracle_time = group_info["oracle_time"]
            oracle_comp = group_info["oracle_compressor"]

            # Get router's pick
            if isinstance(router, str):
                # Fixed baseline
                router_pick = router
            elif isinstance(router, LearnedRouter):
                ri = group_info["router_input"]
                router_pick = router.predict(ri)
            elif isinstance(router, LookupTableRouter):
                ri = group_info["router_input"]
                router_pick = router.predict(ri)
            else:
                continue

            # Get time for router's pick
            comp_times = group_info["compressor_times"]
            if router_pick in comp_times:
                router_time = comp_times[router_pick]
            else:
                router_time = comp_times.get("identity", oracle_time * 2)

            router_times.append(router_time)
            oracle_times.append(oracle_time)

        if not router_times:
            continue

        router_times = np.array(router_times)
        oracle_times = np.array(oracle_times)

        eval_result = evaluate_router(router_name, router_times, oracle_times)
        router_eval_results[router_name] = eval_result

        logger.info(
            "%s: mean_regret=%.3f, median_regret=%.3f, p95_regret=%.3f, "
            "oracle_match=%.1f%%, mean_ms=%.2f (oracle=%.2f)",
            router_name,
            eval_result.mean_regret,
            eval_result.median_regret,
            eval_result.p95_regret,
            eval_result.oracle_match_rate * 100,
            eval_result.mean_total_ms,
            eval_result.oracle_mean_total_ms,
        )

    # --- Per-bandwidth analysis ---
    logger.info("\n=== Per-Bandwidth Router Accuracy ===")
    _per_bandwidth_analysis(test_groups, routers, names)

    # --- Save results ---
    results_out = {
        "training": {
            "num_samples": n_samples,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "quality_threshold": quality_threshold,
            "compressor_names": names,
            "label_distribution": {
                names[idx]: int(count) for idx, count in label_counts.items()
            },
        },
        "accuracy": {
            "decision_tree": float(dt_acc),
            "mlp": float(mlp_acc),
            "gradient_boosting": float(gb_acc),
        },
        "regret": {
            name: {
                "mean_regret": r.mean_regret,
                "median_regret": r.median_regret,
                "p95_regret": r.p95_regret,
                "oracle_match_rate": r.oracle_match_rate,
                "mean_total_ms": r.mean_total_ms,
                "oracle_mean_total_ms": r.oracle_mean_total_ms,
            }
            for name, r in router_eval_results.items()
        },
        "feature_importances": {
            fname: float(imp) for fname, imp in zip(feat_names, importances)
        },
    }

    results_path = output_dir / "router_results.json"
    with open(results_path, "w") as f:
        json.dump(results_out, f, indent=2)

    logger.info("\nRouter results saved to %s", results_path)


def _train_lookup_table(
    results: list[dict],
    quality_threshold: float,
) -> LookupTableRouter:
    """Train lookup table router from raw results."""
    bandwidths = np.array([r["bandwidth_gbps"] for r in results])
    prompt_lengths = np.array([r["seq_len"] for r in results])
    compressor_names = np.array([r["compressor"] for r in results])
    total_ms = np.array([r["total_ms"] for r in results])
    quality_ok = np.array([
        r.get("mean_key_cosine_sim", 1.0) >= quality_threshold
        or r["compressor"] == "identity"
        for r in results
    ])

    lt = LookupTableRouter.from_benchmark_data(
        bandwidths, prompt_lengths, compressor_names, total_ms, quality_ok
    )
    logger.info("Lookup table: %d entries", len(lt.table))
    return lt


def _get_test_groups(
    groups: dict,
    test_idx: np.ndarray,
    all_results: list[dict],
    quality_threshold: float,
    compressor_names: list[str],
) -> dict:
    """Build test evaluation groups with oracle, router inputs, and compressor times."""
    # We need to map test indices back to groups
    # Reconstruct which groups are in the test set
    group_keys = sorted(groups.keys())
    test_group_keys = set()

    # Map each group to its index
    group_order = []
    for gk in group_keys:
        group_results = groups[gk]
        # Filter valid
        valid = [
            r for r in group_results
            if r.get("mean_key_cosine_sim", 1.0) >= quality_threshold
            or r["compressor"] == "identity"
        ]
        if valid:
            group_order.append(gk)

    for ti in test_idx:
        if ti < len(group_order):
            test_group_keys.add(group_order[ti])

    test_groups = {}
    for gk in test_group_keys:
        group_results = groups[gk]
        model_name = gk[0]

        # Compressor -> time mapping
        comp_times = {}
        for r in group_results:
            comp_times[r["compressor"]] = r["total_ms"]

        # Oracle: best compressor meeting quality threshold
        valid_results = [
            r for r in group_results
            if r.get("mean_key_cosine_sim", 1.0) >= quality_threshold
            or r["compressor"] == "identity"
        ]
        if not valid_results:
            continue

        oracle = min(valid_results, key=lambda r: r["total_ms"])

        # Build RouterInput
        r0 = group_results[0]
        if model_name in _MODEL_ARCH:
            num_layers, num_kv_heads, head_dim = _MODEL_ARCH[model_name]
        else:
            num_layers, num_kv_heads, head_dim = 32, 8, 128

        ri = RouterInput(
            prompt_length=r0["seq_len"],
            model_num_layers=num_layers,
            model_num_kv_heads=num_kv_heads,
            model_head_dim=head_dim,
            kv_cache_size_bytes=r0["original_bytes"],
            available_bandwidth_gbps=r0["bandwidth_gbps"],
            quality_threshold=quality_threshold,
        )

        test_groups[gk] = {
            "oracle_time": oracle["total_ms"],
            "oracle_compressor": oracle["compressor"],
            "compressor_times": comp_times,
            "router_input": ri,
            "bandwidth": gk[2],
        }

    return test_groups


def _per_bandwidth_analysis(
    test_groups: dict,
    routers: dict,
    compressor_names: list[str],
) -> None:
    """Analyze router accuracy per bandwidth bucket."""
    bw_groups: dict[float, list] = {}
    for gk, info in test_groups.items():
        bw_groups.setdefault(info["bandwidth"], []).append(info)

    for bw in sorted(bw_groups.keys()):
        infos = bw_groups[bw]
        oracle_picks = Counter(i["oracle_compressor"] for i in infos)
        logger.info(
            "BW=%.0f Gbps (%d samples): oracle picks = %s",
            bw, len(infos),
            {k: v for k, v in oracle_picks.most_common(3)},
        )

        for router_name, router in routers.items():
            if isinstance(router, str):
                continue  # skip fixed baselines for per-bw
            matches = 0
            for info in infos:
                if isinstance(router, LearnedRouter):
                    pick = router.predict(info["router_input"])
                elif isinstance(router, LookupTableRouter):
                    pick = router.predict(info["router_input"])
                else:
                    continue
                if pick == info["oracle_compressor"]:
                    matches += 1
            acc = matches / len(infos) if infos else 0
            logger.info("  %s accuracy: %.1f%%", router_name, acc * 100)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_router.py <results1.json> [results2.json ...]")
        sys.exit(1)

    results_files = sys.argv[1:]
    train_and_evaluate(results_files)
