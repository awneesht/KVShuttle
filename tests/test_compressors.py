"""Tests for compression strategies."""

from __future__ import annotations

import numpy as np
import pytest

from kvshuttle.compression.base import CompressedKVCache
from kvshuttle.compression.identity import IdentityCompressor
from kvshuttle.compression.registry import get_compressor, list_compressors


class TestIdentityCompressor:
    """Tests for the identity (no-op) compressor."""

    def test_round_trip(self, small_kv_cache):
        keys, values = small_kv_cache
        compressor = IdentityCompressor()

        compressed = compressor.compress(keys, values)
        keys_out, values_out = compressor.decompress(compressed)

        np.testing.assert_array_equal(keys, keys_out)
        np.testing.assert_array_equal(values, values_out)

    def test_compression_ratio_is_one(self, small_kv_cache):
        keys, values = small_kv_cache
        compressor = IdentityCompressor()

        compressed = compressor.compress(keys, values)
        assert compressed.compression_ratio == pytest.approx(1.0, abs=0.05)

    def test_metadata_correct(self, small_kv_cache):
        keys, values = small_kv_cache
        compressor = IdentityCompressor()

        compressed = compressor.compress(keys, values)
        assert compressed.num_layers == 4
        assert compressed.num_heads == 2
        assert compressed.seq_len == 32
        assert compressed.head_dim == 64

    def test_properties(self):
        compressor = IdentityCompressor()
        assert compressor.name == "identity"
        assert compressor.is_lossy is False

    def test_preserves_dtype(self, small_kv_cache):
        keys, values = small_kv_cache
        compressor = IdentityCompressor()

        compressed = compressor.compress(keys, values)
        keys_out, values_out = compressor.decompress(compressed)

        assert keys_out.dtype == np.float16
        assert values_out.dtype == np.float16


class TestRegistry:
    """Tests for the compressor registry."""

    def test_identity_registered(self):
        assert "identity" in list_compressors()

    def test_get_identity(self):
        compressor = get_compressor("identity")
        assert compressor.name == "identity"

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown compressor"):
            get_compressor("nonexistent_compressor")

    def test_list_returns_sorted(self):
        names = list_compressors()
        assert names == sorted(names)


class TestCompressedKVCache:
    """Tests for the CompressedKVCache dataclass."""

    def test_compression_ratio(self):
        cache = CompressedKVCache(
            data=b"x" * 500,
            metadata={"key": "value"},
            original_size_bytes=1000,
            compressed_size_bytes=500,
        )
        # metadata adds some overhead, so ratio is slightly less than 2.0
        assert 1.5 < cache.compression_ratio < 2.0

    def test_compression_ratio_zero_size(self):
        cache = CompressedKVCache(
            data=b"",
            metadata={},
            original_size_bytes=0,
            compressed_size_bytes=0,
        )
        assert cache.compression_ratio == 1.0
