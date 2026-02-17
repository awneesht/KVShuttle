"""Tests for real TCP transfer measurement."""

from __future__ import annotations

import statistics

import pytest

from kvshuttle.transfer.real_transfer import (
    RealTransferResult,
    measure_tcp_transfer,
    measure_transfer_overhead,
)


class TestRealTransferResult:
    """Tests for the RealTransferResult dataclass."""

    def test_fields_present(self):
        result = RealTransferResult(
            transfer_ms=1.5,
            payload_bytes=1024,
            connect_ms=0.1,
            all_times_ms=[1.0, 1.5, 2.0],
        )
        assert result.transfer_ms == 1.5
        assert result.payload_bytes == 1024
        assert result.connect_ms == 0.1
        assert result.all_times_ms == [1.0, 1.5, 2.0]


class TestTCPTransfer:
    """Tests for measure_tcp_transfer()."""

    @pytest.mark.parametrize("payload_size", [1024, 1_048_576])
    def test_transfer_completes_with_positive_time(self, payload_size):
        payload = bytes(bytearray(payload_size))
        result = measure_tcp_transfer(payload, repeats=3)
        assert result.transfer_ms > 0

    @pytest.mark.parametrize("payload_size", [1024, 1_048_576])
    def test_all_times_length_equals_repeats(self, payload_size):
        repeats = 3
        payload = bytes(bytearray(payload_size))
        result = measure_tcp_transfer(payload, repeats=repeats)
        assert len(result.all_times_ms) == repeats

    @pytest.mark.parametrize("payload_size", [1024, 1_048_576])
    def test_transfer_ms_is_median(self, payload_size):
        payload = bytes(bytearray(payload_size))
        result = measure_tcp_transfer(payload, repeats=3)
        assert result.transfer_ms == pytest.approx(
            statistics.median(result.all_times_ms)
        )

    @pytest.mark.parametrize("payload_size", [1024, 1_048_576])
    def test_connect_ms_positive(self, payload_size):
        payload = bytes(bytearray(payload_size))
        result = measure_tcp_transfer(payload, repeats=3)
        assert result.connect_ms > 0

    @pytest.mark.parametrize("payload_size", [1024, 1_048_576])
    def test_payload_bytes_matches_input(self, payload_size):
        payload = bytes(bytearray(payload_size))
        result = measure_tcp_transfer(payload, repeats=3)
        assert result.payload_bytes == payload_size


class TestTransferOverhead:
    """Tests for measure_transfer_overhead()."""

    def test_returns_list_of_dicts_with_expected_keys(self):
        results = measure_transfer_overhead([1024], bandwidth_gbps=100.0, repeats=3)
        assert isinstance(results, list)
        assert len(results) == 1
        expected_keys = {
            "payload_bytes", "payload_mb", "analytical_ms", "real_ms",
            "connect_ms", "overhead_pct", "all_real_ms",
        }
        assert expected_keys <= set(results[0].keys())

    def test_positive_analytical_and_real_ms(self):
        results = measure_transfer_overhead([4096], bandwidth_gbps=100.0, repeats=3)
        for entry in results:
            assert entry["analytical_ms"] > 0
            assert entry["real_ms"] > 0
