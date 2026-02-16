"""Tests for the transfer pipeline and baseline compressors."""

from __future__ import annotations

import numpy as np
import pytest

from kvshuttle.compression.registry import get_compressor, list_compressors
from kvshuttle.transfer.serializer import serialize, deserialize
from kvshuttle.transfer.simulator import simulate_transfer
from kvshuttle.transfer.pipeline import run_pipeline, run_pipeline_sweep


class TestBaselineCompressors:
    """Round-trip and ratio tests for all baseline compressors."""

    @pytest.mark.parametrize("name", ["identity", "lossless_zstd", "lossless_lz4"])
    def test_lossless_round_trip(self, small_kv_cache, name):
        """Lossless compressors must produce exact round-trips."""
        keys, values = small_kv_cache
        compressor = get_compressor(name)
        compressed = compressor.compress(keys, values)
        keys_out, values_out = compressor.decompress(compressed)
        np.testing.assert_array_equal(keys, keys_out)
        np.testing.assert_array_equal(values, values_out)

    @pytest.mark.parametrize("name", ["lossless_zstd", "lossless_lz4"])
    def test_lossless_not_lossy(self, name):
        compressor = get_compressor(name)
        assert compressor.is_lossy is False

    @pytest.mark.parametrize("name", ["fp8_e4m3", "uniform_int8", "uniform_int4"])
    def test_lossy_round_trip_shapes(self, small_kv_cache, name):
        """Lossy compressors must preserve shapes."""
        keys, values = small_kv_cache
        compressor = get_compressor(name)
        compressed = compressor.compress(keys, values)
        keys_out, values_out = compressor.decompress(compressed)
        assert keys_out.shape == keys.shape
        assert values_out.shape == values.shape
        assert keys_out.dtype == np.float16
        assert values_out.dtype == np.float16

    @pytest.mark.parametrize("name", ["fp8_e4m3", "uniform_int8"])
    def test_2x_compression_ratio(self, small_kv_cache, name):
        """FP8 and INT8 should achieve ~2x compression."""
        keys, values = small_kv_cache
        compressor = get_compressor(name)
        compressed = compressor.compress(keys, values)
        assert 1.5 < compressed.compression_ratio < 2.5

    def test_int4_compression_ratio(self, medium_kv_cache):
        """INT4 should achieve ~4x compression on realistically sized caches."""
        keys, values = medium_kv_cache
        compressor = get_compressor("uniform_int4")
        compressed = compressor.compress(keys, values)
        # Metadata overhead is negligible on larger caches
        assert 3.0 < compressed.compression_ratio < 5.0

    @pytest.mark.parametrize("name", ["fp8_e4m3", "uniform_int8"])
    def test_lossy_quality_reasonable(self, small_kv_cache, name):
        """Lossy compressors should preserve values within reasonable tolerance."""
        keys, values = small_kv_cache
        compressor = get_compressor(name)
        compressed = compressor.compress(keys, values)
        keys_out, values_out = compressor.decompress(compressed)

        # Cosine similarity should be very high for 8-bit methods
        k_flat = keys.astype(np.float32).reshape(-1)
        k_out_flat = keys_out.astype(np.float32).reshape(-1)
        cosine_sim = np.dot(k_flat, k_out_flat) / (
            np.linalg.norm(k_flat) * np.linalg.norm(k_out_flat)
        )
        assert cosine_sim > 0.99

    def test_all_baselines_registered(self):
        """All expected baseline compressors should be registered."""
        registered = list_compressors()
        for name in ["identity", "lossless_zstd", "lossless_lz4",
                      "fp8_e4m3", "uniform_int8", "uniform_int4"]:
            assert name in registered, f"{name} not registered"


class TestSerializer:
    """Tests for serialize/deserialize round-trip."""

    def test_serialize_deserialize_round_trip(self, small_kv_cache):
        compressor = get_compressor("identity")
        keys, values = small_kv_cache
        compressed = compressor.compress(keys, values)

        wire = serialize(compressed)
        restored = deserialize(wire)

        assert restored.original_size_bytes == compressed.original_size_bytes
        assert restored.compressed_size_bytes == compressed.compressed_size_bytes
        assert restored.num_layers == compressed.num_layers
        assert restored.data == compressed.data

    def test_serialize_preserves_metadata(self, small_kv_cache):
        compressor = get_compressor("uniform_int8")
        keys, values = small_kv_cache
        compressed = compressor.compress(keys, values)

        wire = serialize(compressed)
        restored = deserialize(wire)

        assert restored.metadata["bits"] == 8
        assert restored.metadata["key_shape"] == list(keys.shape)


class TestSimulator:
    """Tests for the transfer simulator."""

    def test_basic_transfer(self):
        # 1 GB at 100 Gbps = 80ms
        result = simulate_transfer(1_000_000_000, 100.0)
        assert pytest.approx(result.transfer_ms, rel=0.01) == 80.0

    def test_zero_payload(self):
        result = simulate_transfer(0, 100.0)
        assert result.transfer_ms == 0.0

    def test_invalid_bandwidth(self):
        with pytest.raises(ValueError):
            simulate_transfer(100, 0.0)

    def test_effective_bandwidth(self):
        result = simulate_transfer(1_000_000_000, 100.0)
        assert pytest.approx(result.effective_gbps, rel=0.01) == 100.0


class TestPipeline:
    """Tests for the full pipeline."""

    def test_identity_pipeline(self, small_kv_cache):
        keys, values = small_kv_cache
        compressor = get_compressor("identity")
        result = run_pipeline(compressor, keys, values, bandwidth_gbps=10.0)

        assert result.compression_ratio == pytest.approx(1.0, abs=0.1)
        assert result.total_ms > 0
        assert result.raw_transfer_ms > 0
        assert result.compressor_name == "identity"

    def test_pipeline_speedup_at_low_bandwidth(self, small_kv_cache):
        """INT4 should give speedup > 1 at low bandwidth."""
        keys, values = small_kv_cache
        compressor = get_compressor("uniform_int4")
        result = run_pipeline(compressor, keys, values, bandwidth_gbps=1.0)

        # At 1 Gbps, compression should help
        assert result.compression_ratio > 2.0

    def test_pipeline_sweep(self, small_kv_cache):
        keys, values = small_kv_cache
        compressor = get_compressor("identity")
        results = run_pipeline_sweep(compressor, keys, values, bandwidths_gbps=[1.0, 10.0, 100.0])
        assert len(results) == 3
        # Higher bandwidth = lower transfer time
        assert results[0].transfer_ms > results[2].transfer_ms
