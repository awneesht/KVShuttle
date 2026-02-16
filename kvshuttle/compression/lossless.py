"""Lossless compressors — zstd and lz4 on raw KV cache bytes."""

from __future__ import annotations

import numpy as np

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


class _LosslessBase(BaseCompressor):
    """Base class for lossless byte-level compressors."""

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        key_bytes = keys.tobytes()
        val_bytes = values.tobytes()
        raw = key_bytes + val_bytes
        original_size = len(raw)

        compressed_data = self._compress_bytes(raw)

        return CompressedKVCache(
            data=compressed_data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "key_dtype": str(keys.dtype),
                "val_dtype": str(values.dtype),
                "key_bytes_len": len(key_bytes),
                "original_len": original_size,
            },
            original_size_bytes=original_size,
            compressed_size_bytes=len(compressed_data),
            num_layers=keys.shape[0],
            num_heads=keys.shape[1],
            seq_len=keys.shape[2],
            head_dim=keys.shape[3],
        )

    def decompress(self, compressed: CompressedKVCache) -> tuple[np.ndarray, np.ndarray]:
        meta = compressed.metadata
        raw = self._decompress_bytes(compressed.data, meta["original_len"])

        key_len = meta["key_bytes_len"]
        key_dtype = np.dtype(meta["key_dtype"])
        val_dtype = np.dtype(meta["val_dtype"])

        keys = np.frombuffer(raw[:key_len], dtype=key_dtype).reshape(meta["key_shape"]).copy()
        values = np.frombuffer(raw[key_len:], dtype=val_dtype).reshape(meta["val_shape"]).copy()
        return keys, values

    def _compress_bytes(self, data: bytes) -> bytes:
        raise NotImplementedError

    def _decompress_bytes(self, data: bytes, original_len: int) -> bytes:
        raise NotImplementedError

    @property
    def is_lossy(self) -> bool:
        return False


@register("lossless_zstd")
class ZstdCompressor(_LosslessBase):
    """Zstandard compression at configurable level."""

    def __init__(self, level: int = 3):
        self._level = level

    def _compress_bytes(self, data: bytes) -> bytes:
        import zstandard as zstd

        cctx = zstd.ZstdCompressor(level=self._level)
        return cctx.compress(data)

    def _decompress_bytes(self, data: bytes, original_len: int) -> bytes:
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data, max_output_size=original_len)

    @property
    def name(self) -> str:
        return "lossless_zstd"


@register("lossless_lz4")
class Lz4Compressor(_LosslessBase):
    """LZ4 fast compression."""

    def _compress_bytes(self, data: bytes) -> bytes:
        import lz4.frame

        return lz4.frame.compress(data)

    def _decompress_bytes(self, data: bytes, original_len: int) -> bytes:
        import lz4.frame

        return lz4.frame.decompress(data)

    @property
    def name(self) -> str:
        return "lossless_lz4"
