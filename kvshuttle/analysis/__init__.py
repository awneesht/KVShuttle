"""Statistical analysis utilities for KVShuttle benchmark."""

from kvshuttle.analysis.statistical_tests import (
    bootstrap_ci,
    wilcoxon_signed_rank,
    spearman_correlation,
    cohens_d,
    friedman_test,
)

__all__ = [
    "bootstrap_ci",
    "wilcoxon_signed_rank",
    "spearman_correlation",
    "cohens_d",
    "friedman_test",
]
