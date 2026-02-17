"""Tests for the vLLM adapter (KVShuttleConnector)."""

from __future__ import annotations

import numpy as np
import pytest

from kvshuttle.serving.vllm_adapter import KVShuttleConnector


class TestKVShuttleConnector:
    """Tests for the KVShuttleConnector send/receive interface."""

    @pytest.mark.parametrize(
        "compressor_name", ["identity", "uniform_int8", "kivi_2bit"]
    )
    def test_init_succeeds(self, compressor_name):
        connector = KVShuttleConnector(compressor_name)
        assert connector.compressor.name == compressor_name

    @pytest.mark.parametrize(
        "compressor_name", ["identity", "uniform_int8", "kivi_2bit"]
    )
    def test_last_stats_none_before_send(self, compressor_name):
        connector = KVShuttleConnector(compressor_name)
        assert connector.last_stats is None

    @pytest.mark.parametrize(
        "compressor_name", ["identity", "uniform_int8", "kivi_2bit"]
    )
    def test_round_trip_preserves_shape_and_dtype(self, compressor_name, small_kv_cache):
        keys, values = small_kv_cache
        connector = KVShuttleConnector(compressor_name)

        wire = connector.send_kv_cache(keys, values)
        keys_out, values_out = connector.recv_kv_cache(wire)

        assert keys_out.shape == keys.shape
        assert values_out.shape == values.shape
        assert keys_out.dtype == keys.dtype
        assert values_out.dtype == values.dtype

    def test_identity_arrays_match_exactly(self, small_kv_cache):
        keys, values = small_kv_cache
        connector = KVShuttleConnector("identity")

        wire = connector.send_kv_cache(keys, values)
        keys_out, values_out = connector.recv_kv_cache(wire)

        np.testing.assert_array_equal(keys, keys_out)
        np.testing.assert_array_equal(values, values_out)

    @pytest.mark.parametrize("compressor_name", ["uniform_int8", "kivi_2bit"])
    def test_lossy_cosine_sim_above_threshold(self, compressor_name, small_kv_cache):
        keys, values = small_kv_cache
        connector = KVShuttleConnector(compressor_name)

        wire = connector.send_kv_cache(keys, values)
        keys_out, values_out = connector.recv_kv_cache(wire)

        # Cosine similarity on flattened arrays
        for original, reconstructed in [(keys, keys_out), (values, values_out)]:
            a = original.flatten().astype(np.float32)
            b = reconstructed.flatten().astype(np.float32)
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            assert cos_sim > 0.9, f"{compressor_name} cosine sim too low: {cos_sim}"

    @pytest.mark.parametrize(
        "compressor_name", ["identity", "uniform_int8", "kivi_2bit"]
    )
    def test_last_stats_populated_after_send(self, compressor_name, small_kv_cache):
        keys, values = small_kv_cache
        connector = KVShuttleConnector(compressor_name)

        connector.send_kv_cache(keys, values)
        stats = connector.last_stats

        assert stats is not None
        assert stats.original_bytes > 0
        assert stats.wire_bytes > 0
        assert stats.compression_ratio > 0
        assert stats.compressor_name == compressor_name

    def test_int8_compression_ratio(self, small_kv_cache):
        keys, values = small_kv_cache
        connector = KVShuttleConnector("uniform_int8")

        connector.send_kv_cache(keys, values)
        stats = connector.last_stats

        assert stats.compression_ratio == pytest.approx(2.0, abs=0.5)
