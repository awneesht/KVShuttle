"""Simulate network transfer at configurable bandwidths."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Standard bandwidth test points (Gbps) spanning Ethernet to NVLink
BANDWIDTH_SWEEP_GBPS = [1, 5, 10, 25, 50, 100, 200, 400]


@dataclass
class TransferResult:
    """Result of a simulated transfer.

    Attributes:
        transfer_ms: Simulated transfer time in milliseconds.
        bandwidth_gbps: Bandwidth used for simulation.
        payload_bytes: Number of bytes transferred.
        effective_gbps: Effective bandwidth (payload / time).
    """

    transfer_ms: float
    bandwidth_gbps: float
    payload_bytes: int

    @property
    def effective_gbps(self) -> float:
        if self.transfer_ms == 0:
            return float("inf")
        return (self.payload_bytes * 8) / (self.transfer_ms * 1e6)


def simulate_transfer(payload_bytes: int, bandwidth_gbps: float) -> TransferResult:
    """Simulate transferring a payload at a given bandwidth.

    Uses analytical model: transfer_time = payload_size / bandwidth.
    Does not account for protocol overhead, congestion, or latency.

    Args:
        payload_bytes: Size of payload in bytes.
        bandwidth_gbps: Network bandwidth in gigabits per second.

    Returns:
        TransferResult with simulated timing.
    """
    if bandwidth_gbps <= 0:
        raise ValueError(f"Bandwidth must be positive, got {bandwidth_gbps}")
    if payload_bytes < 0:
        raise ValueError(f"Payload size must be non-negative, got {payload_bytes}")

    # Convert: bytes -> bits, Gbps -> bits/ms
    bits = payload_bytes * 8
    bits_per_ms = bandwidth_gbps * 1e6  # 1 Gbps = 1e9 bits/s = 1e6 bits/ms
    transfer_ms = bits / bits_per_ms if bits_per_ms > 0 else 0.0

    return TransferResult(
        transfer_ms=transfer_ms,
        bandwidth_gbps=bandwidth_gbps,
        payload_bytes=payload_bytes,
    )


def raw_transfer_ms(original_bytes: int, bandwidth_gbps: float) -> float:
    """Calculate raw (uncompressed) transfer time as baseline.

    Args:
        original_bytes: Size of uncompressed KV cache in bytes.
        bandwidth_gbps: Network bandwidth in Gbps.

    Returns:
        Transfer time in milliseconds.
    """
    return simulate_transfer(original_bytes, bandwidth_gbps).transfer_ms
