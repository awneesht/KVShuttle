"""Identity compressor — no compression baseline (raw FP16)."""

from __future__ import annotations

import numpy as np

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


@register("identity")
class IdentityCompressor(BaseCompressor):
    """Pass-through compressor that serializes KV cache without compression.

    Serves as the baseline for compression ratio (1.0x) and quality (perfect).
    """

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        key_bytes = keys.tobytes()
        val_bytes = values.tobytes()
        data = key_bytes + val_bytes
        original_size = len(data)

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "key_dtype": str(keys.dtype),
                "val_dtype": str(values.dtype),
                "key_bytes_len": len(key_bytes),
            },
            original_size_bytes=original_size,
            compressed_size_bytes=len(data),
            num_layers=keys.shape[0],
            num_heads=keys.shape[1],
            seq_len=keys.shape[2],
            head_dim=keys.shape[3],
        )

    def decompress(self, compressed: CompressedKVCache) -> tuple[np.ndarray, np.ndarray]:
        meta = compressed.metadata
        key_len = meta["key_bytes_len"]
        key_dtype = np.dtype(meta["key_dtype"])
        val_dtype = np.dtype(meta["val_dtype"])

        keys = np.frombuffer(compressed.data[:key_len], dtype=key_dtype).reshape(
            meta["key_shape"]
        )
        values = np.frombuffer(compressed.data[key_len:], dtype=val_dtype).reshape(
            meta["val_shape"]
        )
        return keys.copy(), values.copy()

    @property
    def name(self) -> str:
        return "identity"

    @property
    def is_lossy(self) -> bool:
        return False
