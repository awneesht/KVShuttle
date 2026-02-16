"""Base classes and data structures for KV cache compression."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CompressedKVCache:
    """Container for a compressed KV cache with metadata.

    Attributes:
        data: Compressed payload bytes.
        metadata: Shapes, dtypes, scales, compression params.
        original_size_bytes: Size of the uncompressed KV cache.
        compressed_size_bytes: Size of the compressed payload.
        num_layers: Number of transformer layers.
        num_heads: Number of KV attention heads.
        seq_len: Sequence length (number of tokens).
        head_dim: Dimension per attention head.
    """

    data: bytes
    metadata: dict = field(default_factory=dict)
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    num_layers: int = 0
    num_heads: int = 0
    seq_len: int = 0
    head_dim: int = 0

    @property
    def compression_ratio(self) -> float:
        """Ratio of original size to compressed size (including metadata overhead)."""
        if self.original_size_bytes == 0:
            return 1.0
        metadata_bytes = len(json.dumps(self.metadata).encode())
        total = self.compressed_size_bytes + metadata_bytes
        if total == 0:
            return float("inf")
        return self.original_size_bytes / total


class BaseCompressor(ABC):
    """Abstract base class for all KV cache compressors.

    All compressors operate on raw numpy arrays. No MLX/PyTorch dependency
    should be introduced inside compression modules.
    """

    @abstractmethod
    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        """Compress KV cache arrays.

        Args:
            keys: Key tensors with shape [num_layers, num_heads, seq_len, head_dim].
            values: Value tensors with shape [num_layers, num_heads, seq_len, head_dim].

        Returns:
            CompressedKVCache containing the compressed data and metadata.
        """
        ...

    @abstractmethod
    def decompress(self, compressed: CompressedKVCache) -> tuple[np.ndarray, np.ndarray]:
        """Decompress a CompressedKVCache back to key/value arrays.

        Args:
            compressed: The compressed KV cache.

        Returns:
            Tuple of (keys, values) numpy arrays with original shapes.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this compression strategy."""
        ...

    @property
    def is_lossy(self) -> bool:
        """Whether this compressor is lossy. Override to return False for lossless methods."""
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, lossy={self.is_lossy})"
