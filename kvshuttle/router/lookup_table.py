"""Lookup table router: bin by (bandwidth, prompt_length) and pick precomputed best."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from kvshuttle.router.features import RouterInput

logger = logging.getLogger(__name__)


@dataclass
class LookupTableRouter:
    """Binned lookup table for routing.

    Discretizes (bandwidth_gbps, prompt_length) into bins and stores
    the best compressor per bin based on training data.
    """

    bandwidth_bins: np.ndarray  # bin edges
    length_bins: np.ndarray     # bin edges
    table: dict[tuple[int, int], str]  # (bw_bin, len_bin) -> compressor name
    default: str = "identity"

    def predict(self, router_input: RouterInput) -> str:
        """Select compressor based on binned lookup.

        Args:
            router_input: Features for the current request.

        Returns:
            Name of the recommended compressor.
        """
        bw_bin = int(np.searchsorted(self.bandwidth_bins, router_input.available_bandwidth_gbps))
        len_bin = int(np.searchsorted(self.length_bins, router_input.prompt_length))
        return self.table.get((bw_bin, len_bin), self.default)

    @classmethod
    def from_benchmark_data(
        cls,
        bandwidths: np.ndarray,
        prompt_lengths: np.ndarray,
        compressor_names: np.ndarray,
        total_ms: np.ndarray,
        quality_ok: np.ndarray,
        n_bw_bins: int = 8,
        n_len_bins: int = 6,
    ) -> LookupTableRouter:
        """Build lookup table from benchmark results.

        Args:
            bandwidths: Array of bandwidth values for each result.
            prompt_lengths: Array of prompt lengths.
            compressor_names: Array of compressor names.
            total_ms: Array of total pipeline times.
            quality_ok: Boolean array - True if quality threshold met.
            n_bw_bins: Number of bandwidth bins.
            n_len_bins: Number of prompt length bins.

        Returns:
            Trained LookupTableRouter.
        """
        bw_bins = np.percentile(bandwidths, np.linspace(0, 100, n_bw_bins + 1)[1:-1])
        len_bins = np.percentile(prompt_lengths, np.linspace(0, 100, n_len_bins + 1)[1:-1])

        bw_indices = np.searchsorted(bw_bins, bandwidths)
        len_indices = np.searchsorted(len_bins, prompt_lengths)

        table = {}
        for bw_bin in range(n_bw_bins):
            for len_bin in range(n_len_bins):
                mask = (bw_indices == bw_bin) & (len_indices == len_bin) & quality_ok
                if not mask.any():
                    continue

                # Find best (lowest total_ms) in this bin
                bin_ms = total_ms[mask]
                bin_names = compressor_names[mask]
                best_idx = np.argmin(bin_ms)
                table[(bw_bin, len_bin)] = str(bin_names[best_idx])

        return cls(
            bandwidth_bins=bw_bins,
            length_bins=len_bins,
            table=table,
        )
