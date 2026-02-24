"""Statistical tests for KVShuttle benchmark analysis.

Provides bootstrap confidence intervals, nonparametric hypothesis tests,
effect sizes, and correlation measures for comparing compression strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""
    estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    def __repr__(self) -> str:
        return (
            f"{self.estimate:.4f} "
            f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"({self.ci_level*100:.0f}% CI, n_boot={self.n_bootstrap})"
        )


@dataclass
class HypothesisTestResult:
    """Result of a hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float | None = None

    @property
    def significant_at_05(self) -> bool:
        return self.p_value < 0.05

    @property
    def significant_at_01(self) -> bool:
        return self.p_value < 0.01

    def __repr__(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "ns"
        s = f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4g} {sig}"
        if self.effect_size is not None:
            s += f", d={self.effect_size:.3f}"
        return s


@dataclass
class CorrelationResult:
    """Result of a correlation test."""
    method: str
    coefficient: float
    p_value: float
    n: int

    def __repr__(self) -> str:
        return f"{self.method}: r={self.coefficient:.4f}, p={self.p_value:.4g}, n={self.n}"


def bootstrap_ci(
    data: ArrayLike,
    statistic: str = "mean",
    ci_level: float = 0.95,
    n_bootstrap: int = 10000,
    rng_seed: int | None = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1-D array of observations.
        statistic: One of "mean", "median", "std".
        ci_level: Confidence level (default 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap resamples.
        rng_seed: Random seed for reproducibility.

    Returns:
        BootstrapCI with point estimate and interval bounds.
    """
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return BootstrapCI(np.nan, np.nan, np.nan, ci_level, n_bootstrap)

    stat_fn = {"mean": np.mean, "median": np.median, "std": np.std}[statistic]
    point_estimate = float(stat_fn(data))

    rng = np.random.default_rng(rng_seed)
    boot_stats = np.empty(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        sample = data[rng.integers(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapCI(point_estimate, ci_lower, ci_upper, ci_level, n_bootstrap)


def wilcoxon_signed_rank(
    x: ArrayLike, y: ArrayLike
) -> HypothesisTestResult:
    """Wilcoxon signed-rank test for paired samples.

    Tests whether two related paired samples come from the same distribution.
    Use this to compare two compressors on the same set of prompts.

    Args:
        x: Metric values for compressor A (one per prompt).
        y: Metric values for compressor B (one per prompt).

    Returns:
        HypothesisTestResult with test statistic and p-value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    diff = diff[~np.isnan(diff)]
    if len(diff) < 10 or np.all(diff == 0):
        return HypothesisTestResult("Wilcoxon signed-rank", 0.0, 1.0, 0.0)

    stat, p = stats.wilcoxon(diff, alternative="two-sided")
    d = cohens_d(x, y)
    return HypothesisTestResult("Wilcoxon signed-rank", float(stat), float(p), d)


def cohens_d(x: ArrayLike, y: ArrayLike) -> float:
    """Compute Cohen's d effect size for paired samples.

    Args:
        x: Metric values for condition A.
        y: Metric values for condition B.

    Returns:
        Cohen's d (positive means x > y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diff = x - y
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0:
        return 0.0
    pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float(np.mean(diff) / pooled_std)


def spearman_correlation(
    x: ArrayLike, y: ArrayLike
) -> CorrelationResult:
    """Spearman rank correlation between two variables.

    Args:
        x: First variable (e.g., cosine similarity).
        y: Second variable (e.g., token agreement).

    Returns:
        CorrelationResult with coefficient and p-value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return CorrelationResult("Spearman", 0.0, 1.0, len(x))

    rho, p = stats.spearmanr(x, y)
    return CorrelationResult("Spearman", float(rho), float(p), len(x))


def friedman_test(
    *groups: ArrayLike,
) -> HypothesisTestResult:
    """Friedman test for repeated measures across multiple compressors.

    Each group is a compressor's metric values across the same set of prompts.
    Tests whether at least one compressor differs significantly.

    Args:
        *groups: One array per compressor, each of length n_prompts.

    Returns:
        HypothesisTestResult with chi-square statistic and p-value.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    if len(arrays) < 3:
        return HypothesisTestResult("Friedman", 0.0, 1.0)
    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]

    stat, p = stats.friedmanchisquare(*arrays)
    return HypothesisTestResult("Friedman", float(stat), float(p))


def compute_all_cis(
    results_by_compressor: dict[str, dict[str, list[float]]],
    metrics: list[str] | None = None,
    ci_level: float = 0.95,
) -> dict[str, dict[str, BootstrapCI]]:
    """Compute bootstrap CIs for all compressors and metrics.

    Args:
        results_by_compressor: {compressor: {metric_name: [values]}}.
        metrics: Which metrics to compute CIs for. If None, compute for all.
        ci_level: Confidence level.

    Returns:
        {compressor: {metric: BootstrapCI}}.
    """
    out: dict[str, dict[str, BootstrapCI]] = {}
    for comp, metric_dict in results_by_compressor.items():
        out[comp] = {}
        for metric_name, values in metric_dict.items():
            if metrics and metric_name not in metrics:
                continue
            out[comp][metric_name] = bootstrap_ci(values, ci_level=ci_level)
    return out
