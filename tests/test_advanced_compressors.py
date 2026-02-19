"""Tests for Phase 3 advanced compression strategies."""

from __future__ import annotations

import numpy as np
import pytest

from kvshuttle.compression.registry import get_compressor, list_compressors

ADVANCED_COMPRESSORS = ["kivi_2bit", "cachegen", "kvquant_2bit", "palu_lr", "topk_prune_50"]


class TestAdvancedCompressorsRegistered:
    """Verify all advanced compressors are registered."""

    @pytest.mark.parametrize("name", ADVANCED_COMPRESSORS)
    def test_registered(self, name):
        assert name in list_compressors()


class TestAdvancedRoundTrip:
    """Round-trip shape and dtype tests for all advanced compressors."""

    @pytest.mark.parametrize("name", ADVANCED_COMPRESSORS)
    def test_shape_preserved(self, small_kv_cache, name):
        keys, values = small_kv_cache
        compressor = get_compressor(name)
        compressed = compressor.compress(keys, values)
        keys_out, values_out = compressor.decompress(compressed)

        assert keys_out.shape == keys.shape, f"{name}: key shape mismatch"
        assert values_out.shape == values.shape, f"{name}: value shape mismatch"

    @pytest.mark.parametrize("name", ADVANCED_COMPRESSORS)
    def test_dtype_is_float16(self, small_kv_cache, name):
        keys, values = small_kv_cache
        compressor = get_compressor(name)
        compressed = compressor.compress(keys, values)
        keys_out, values_out = compressor.decompress(compressed)

        assert keys_out.dtype == np.float16, f"{name}: key dtype mismatch"
        assert values_out.dtype == np.float16, f"{name}: value dtype mismatch"


class TestCompressionRatios:
    """Verify expected compression ratios for advanced methods."""

    def test_kivi_high_compression(self, medium_kv_cache):
        """KIVI 2-bit should achieve high compression."""
        keys, values = medium_kv_cache
        compressor = get_compressor("kivi_2bit")
        compressed = compressor.compress(keys, values)
        # 2-bit = 8x data, plus scale/zero overhead
        assert compressed.compression_ratio > 3.0

    def test_cachegen_moderate_compression(self, small_kv_cache):
        """CacheGen should achieve moderate compression."""
        keys, values = small_kv_cache
        compressor = get_compressor("cachegen")
        compressed = compressor.compress(keys, values)
        assert compressed.compression_ratio > 1.5

    def test_kvquant_high_compression(self, medium_kv_cache):
        """KVQuant 2-bit should achieve high compression."""
        keys, values = medium_kv_cache
        compressor = get_compressor("kvquant_2bit")
        compressed = compressor.compress(keys, values)
        assert compressed.compression_ratio > 3.0

    def test_palu_compression(self, small_kv_cache):
        """Palu low-rank should compress significantly."""
        keys, values = small_kv_cache
        compressor = get_compressor("palu_lr")
        compressed = compressor.compress(keys, values)
        assert compressed.compression_ratio > 1.5

    def test_pruning_2x_compression(self, small_kv_cache):
        """50% pruning should give ~2x compression on data."""
        keys, values = small_kv_cache
        compressor = get_compressor("topk_prune_50")
        compressed = compressor.compress(keys, values)
        assert compressed.compression_ratio > 1.5


class TestQualityMetrics:
    """Verify reasonable quality for lossy methods."""

    @pytest.mark.parametrize("name", ["kivi_2bit", "cachegen", "kvquant_2bit", "palu_lr"])
    def test_cosine_similarity(self, small_kv_cache, name):
        """Lossy methods should maintain reasonable cosine similarity."""
        keys, values = small_kv_cache
        compressor = get_compressor(name)
        compressed = compressor.compress(keys, values)
        keys_out, values_out = compressor.decompress(compressed)

        k_flat = keys.astype(np.float32).reshape(-1)
        k_out_flat = keys_out.astype(np.float32).reshape(-1)

        norm_product = np.linalg.norm(k_flat) * np.linalg.norm(k_out_flat)
        if norm_product > 0:
            cosine_sim = np.dot(k_flat, k_out_flat) / norm_product
        else:
            cosine_sim = 1.0

        # 2-bit methods and aggressive low-rank may have lower similarity on random data
        # Real KV caches are more structured and achieve higher quality
        min_sim = 0.6 if name == "palu_lr" else 0.8
        assert cosine_sim > min_sim, f"{name}: cosine sim too low: {cosine_sim}"

    def test_pruning_preserves_kept_tokens(self, small_kv_cache):
        """Pruning should perfectly preserve the tokens it keeps."""
        keys, values = small_kv_cache
        compressor = get_compressor("topk_prune_50")
        compressed = compressor.compress(keys, values)
        keys_out, values_out = compressor.decompress(compressed)

        # Non-zero positions in output should match original
        nonzero_mask = np.any(keys_out != 0, axis=(0, 1, 3))  # [seq_len]
        for t in np.where(nonzero_mask)[0]:
            np.testing.assert_array_equal(keys[:, :, t, :], keys_out[:, :, t, :])


class TestPipelineIntegration:
    """Test advanced compressors through the full pipeline."""

    @pytest.mark.parametrize("name", ADVANCED_COMPRESSORS)
    def test_pipeline_runs(self, small_kv_cache, name):
        """Each compressor should work through the full pipeline."""
        from kvshuttle.transfer.pipeline import run_pipeline

        keys, values = small_kv_cache
        compressor = get_compressor(name)
        result = run_pipeline(compressor, keys, values, bandwidth_gbps=10.0)

        assert result.total_ms > 0
        assert result.compression_ratio > 0
        assert result.compressor_name == name
