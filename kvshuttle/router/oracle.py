"""Oracle router: try all strategies, pick the best. Upper bound on performance."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from kvshuttle.compression.registry import get_compressor, list_compressors
from kvshuttle.transfer.pipeline import run_pipeline

logger = logging.getLogger(__name__)


@dataclass
class OracleResult:
    """Result from oracle routing.

    Attributes:
        best_compressor: Name of the optimal compressor.
        best_total_ms: Total pipeline time of the best compressor.
        all_results: Dict mapping compressor name to total_ms.
    """

    best_compressor: str
    best_total_ms: float
    all_results: dict[str, float]


def oracle_select(
    keys: np.ndarray,
    values: np.ndarray,
    bandwidth_gbps: float,
    compressor_names: list[str] | None = None,
    quality_threshold: float = 0.99,
) -> OracleResult:
    """Try all compressors and return the fastest one meeting quality threshold.

    Args:
        keys: Key tensors.
        values: Value tensors.
        bandwidth_gbps: Network bandwidth.
        compressor_names: List of compressor names to try. None = all registered.
        quality_threshold: Minimum cosine similarity.

    Returns:
        OracleResult with best compressor and full comparison.
    """
    from kvshuttle.evaluation.attention_error import compute_attention_error

    if compressor_names is None:
        compressor_names = list_compressors()

    results = {}

    for name in compressor_names:
        try:
            compressor = get_compressor(name)
            pipeline_result = run_pipeline(compressor, keys, values, bandwidth_gbps)

            # Check quality
            if compressor.is_lossy:
                compressed = compressor.compress(keys, values)
                k_recon, v_recon = compressor.decompress(compressed)
                error = compute_attention_error(keys, values, k_recon, v_recon)
                if error.mean_key_cosine_sim < quality_threshold:
                    logger.debug(
                        "Oracle: %s rejected (cosine_sim=%.4f < %.4f)",
                        name, error.mean_key_cosine_sim, quality_threshold,
                    )
                    results[name] = float("inf")
                    continue

            results[name] = pipeline_result.total_ms
        except Exception as e:
            logger.warning("Oracle: %s failed: %s", name, e)
            results[name] = float("inf")

    # Pick best
    best_name = min(results, key=results.get)
    return OracleResult(
        best_compressor=best_name,
        best_total_ms=results[best_name],
        all_results=results,
    )
