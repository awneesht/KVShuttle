"""Lightweight adapter showing how KVShuttle compressors integrate with vLLM.

This is a reference implementation demonstrating the integration pattern
for disaggregated prefill KV cache transfer, not a production-ready vLLM plugin.

Usage pattern (without vLLM installed):
    connector = KVShuttleConnector("uniform_int8")
    wire_bytes = connector.send_kv_cache(keys, values)
    keys_out, values_out = connector.recv_kv_cache(wire_bytes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from kvshuttle.compression.registry import get_compressor
from kvshuttle.transfer.serializer import deserialize, serialize

logger = logging.getLogger(__name__)


@dataclass
class TransferStats:
    """Statistics from a KV cache transfer via KVShuttleConnector.

    Attributes:
        original_bytes: Size of uncompressed KV cache.
        wire_bytes: Size of serialized compressed payload.
        compression_ratio: original_bytes / wire_bytes.
        compressor_name: Name of the compressor used.
    """

    original_bytes: int
    wire_bytes: int
    compression_ratio: float
    compressor_name: str


class KVShuttleConnector:
    """Adapter for integrating KVShuttle compression into serving frameworks.

    Provides a simple send/receive interface for KV cache transfer with
    compression. Designed to slot into the KV cache transfer path of
    disaggregated prefill architectures (e.g., vLLM, Splitwise, DistServe).

    Example:
        # On prefill node
        connector = KVShuttleConnector("kivi_2bit")
        wire_bytes = connector.send_kv_cache(keys, values)

        # Transfer wire_bytes over network...

        # On decode node
        keys_out, values_out = connector.recv_kv_cache(wire_bytes)
    """

    def __init__(self, compressor_name: str = "uniform_int8", **compressor_kwargs):
        """Initialize with a named compressor.

        Args:
            compressor_name: Registered compressor name (e.g., "uniform_int8", "kivi_2bit").
            **compressor_kwargs: Additional kwargs passed to the compressor constructor.
        """
        self.compressor = get_compressor(compressor_name, **compressor_kwargs)
        self._last_stats: TransferStats | None = None
        logger.info("KVShuttleConnector initialized with compressor: %s", compressor_name)

    def send_kv_cache(self, keys: np.ndarray, values: np.ndarray) -> bytes:
        """Compress and serialize KV cache for transfer.

        Args:
            keys: Key tensors, shape [num_layers, num_kv_heads, seq_len, head_dim].
            values: Value tensors, same shape as keys.

        Returns:
            Serialized compressed payload as bytes (ready for network transfer).
        """
        original_bytes = keys.nbytes + values.nbytes

        compressed = self.compressor.compress(keys, values)
        wire_bytes = serialize(compressed)

        self._last_stats = TransferStats(
            original_bytes=original_bytes,
            wire_bytes=len(wire_bytes),
            compression_ratio=(
                original_bytes / len(wire_bytes)
                if len(wire_bytes) > 0
                else float("inf")
            ),
            compressor_name=self.compressor.name,
        )

        logger.info(
            "Sent KV cache: %d bytes -> %d bytes (%.2fx compression)",
            original_bytes, len(wire_bytes), self._last_stats.compression_ratio,
        )

        return wire_bytes

    def recv_kv_cache(self, wire_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
        """Deserialize and decompress received KV cache.

        Args:
            wire_bytes: Serialized compressed payload from send_kv_cache().

        Returns:
            Tuple of (keys, values) as numpy arrays with shape
            [num_layers, num_kv_heads, seq_len, head_dim].
        """
        compressed = deserialize(wire_bytes)
        keys, values = self.compressor.decompress(compressed)

        logger.info(
            "Received KV cache: %d bytes -> keys%s + values%s",
            len(wire_bytes), keys.shape, values.shape,
        )

        return keys, values

    @property
    def last_stats(self) -> TransferStats | None:
        """Get statistics from the last send_kv_cache() call."""
        return self._last_stats
