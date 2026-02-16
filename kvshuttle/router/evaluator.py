"""Evaluate router regret vs oracle."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RouterEvalResult:
    """Evaluation result for a router.

    Attributes:
        router_name: Name of the router being evaluated.
        mean_regret: Average regret (fraction slower than oracle).
        median_regret: Median regret.
        p95_regret: 95th percentile regret.
        oracle_match_rate: Fraction of times the router picks the oracle's choice.
        mean_total_ms: Average total pipeline time with this router.
        oracle_mean_total_ms: Average total pipeline time with oracle.
    """

    router_name: str
    mean_regret: float
    median_regret: float
    p95_regret: float
    oracle_match_rate: float
    mean_total_ms: float
    oracle_mean_total_ms: float


def compute_regret(
    router_total_ms: np.ndarray,
    oracle_total_ms: np.ndarray,
) -> np.ndarray:
    """Compute per-request regret.

    Regret = (router_time - oracle_time) / oracle_time

    Returns:
        Array of per-request regret values (0 = optimal, positive = slower).
    """
    safe_oracle = np.maximum(oracle_total_ms, 1e-9)
    return (router_total_ms - oracle_total_ms) / safe_oracle


def evaluate_router(
    router_name: str,
    router_total_ms: np.ndarray,
    oracle_total_ms: np.ndarray,
) -> RouterEvalResult:
    """Evaluate a router against the oracle.

    Args:
        router_name: Identifier for the router.
        router_total_ms: Pipeline times when using this router's picks.
        oracle_total_ms: Pipeline times when using oracle's picks.

    Returns:
        RouterEvalResult with regret statistics.
    """
    regret = compute_regret(router_total_ms, oracle_total_ms)
    oracle_match = np.isclose(router_total_ms, oracle_total_ms, rtol=0.01)

    return RouterEvalResult(
        router_name=router_name,
        mean_regret=float(np.mean(regret)),
        median_regret=float(np.median(regret)),
        p95_regret=float(np.percentile(regret, 95)),
        oracle_match_rate=float(np.mean(oracle_match)),
        mean_total_ms=float(np.mean(router_total_ms)),
        oracle_mean_total_ms=float(np.mean(oracle_total_ms)),
    )
