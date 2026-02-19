"""Real TCP transfer measurement for validating analytical transfer model."""

from __future__ import annotations

import logging
import socket
import statistics
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RealTransferResult:
    """Result of a real TCP transfer measurement.

    Attributes:
        transfer_ms: Measured transfer time in milliseconds (median).
        payload_bytes: Number of bytes transferred.
        connect_ms: TCP connection setup time (median).
        all_times_ms: All individual transfer times.
    """

    transfer_ms: float
    payload_bytes: int
    connect_ms: float
    all_times_ms: list[float]


def measure_tcp_transfer(
    payload: bytes,
    host: str = "127.0.0.1",
    port: int = 0,
    repeats: int = 5,
) -> RealTransferResult:
    """Measure actual TCP transfer time for a payload over localhost.

    Spawns a receiver thread, sends payload via TCP socket, measures wall time.
    Uses the median of `repeats` measurements for robustness.

    Args:
        payload: The byte payload to transfer.
        host: Target host (default: localhost).
        port: Port to use (0 = auto-assign).
        repeats: Number of measurement repetitions.

    Returns:
        RealTransferResult with timing measurements.
    """
    payload_size = len(payload)
    all_times_ms = []
    connect_times_ms = []

    for _ in range(repeats):
        # Set up server socket
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen(1)
        actual_port = server_sock.getsockname()[1]

        received = bytearray()
        recv_error = [None]

        def receiver():
            try:
                conn, _ = server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                # Set large receive buffer
                conn.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_RCVBUF,
                    min(payload_size, 16 * 1024 * 1024),
                )
                while len(received) < payload_size:
                    chunk = conn.recv(min(payload_size - len(received), 1024 * 1024))
                    if not chunk:
                        break
                    received.extend(chunk)
                conn.close()
            except Exception as e:
                recv_error[0] = e
            finally:
                server_sock.close()

        recv_thread = threading.Thread(target=receiver, daemon=True)
        recv_thread.start()

        # Measure: connect + send
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Set large send buffer
        client_sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_SNDBUF,
            min(payload_size, 16 * 1024 * 1024),
        )

        t_connect_start = time.perf_counter_ns()
        client_sock.connect((host, actual_port))
        t_connect_end = time.perf_counter_ns()
        connect_ms = (t_connect_end - t_connect_start) / 1_000_000

        t_start = time.perf_counter_ns()
        # Send in chunks to avoid memory pressure
        offset = 0
        while offset < payload_size:
            sent = client_sock.send(payload[offset:offset + 1024 * 1024])
            offset += sent
        client_sock.shutdown(socket.SHUT_WR)
        client_sock.close()

        # Wait for receiver to finish
        recv_thread.join(timeout=30)

        t_end = time.perf_counter_ns()
        elapsed_ms = (t_end - t_start) / 1_000_000

        if recv_error[0] is not None:
            logger.warning("Receiver error: %s", recv_error[0])
            continue

        all_times_ms.append(elapsed_ms)
        connect_times_ms.append(connect_ms)

    if not all_times_ms:
        raise RuntimeError("All TCP transfer attempts failed")

    return RealTransferResult(
        transfer_ms=statistics.median(all_times_ms),
        payload_bytes=payload_size,
        connect_ms=statistics.median(connect_times_ms),
        all_times_ms=all_times_ms,
    )


def measure_transfer_overhead(
    payload_sizes: list[int],
    bandwidth_gbps: float = 100.0,
    repeats: int = 5,
) -> list[dict]:
    """Compare analytical vs real TCP transfer for various payload sizes.

    Args:
        payload_sizes: List of payload sizes in bytes.
        bandwidth_gbps: Nominal bandwidth for analytical comparison.
        repeats: Repetitions per size for real measurement.

    Returns:
        List of dicts with analytical_ms, real_ms, overhead_pct for each size.
    """
    from kvshuttle.transfer.simulator import simulate_transfer

    results = []
    for size in payload_sizes:
        # Generate random payload
        payload = bytes(bytearray(size))

        # Analytical
        analytical = simulate_transfer(size, bandwidth_gbps)

        # Real TCP
        try:
            real = measure_tcp_transfer(payload, repeats=repeats)
            overhead_pct = (
                (real.transfer_ms - analytical.transfer_ms) / analytical.transfer_ms * 100
                if analytical.transfer_ms > 0 else float("inf")
            )
            results.append({
                "payload_bytes": size,
                "payload_mb": size / (1024 * 1024),
                "analytical_ms": analytical.transfer_ms,
                "real_ms": real.transfer_ms,
                "connect_ms": real.connect_ms,
                "overhead_pct": overhead_pct,
                "all_real_ms": real.all_times_ms,
            })
            logger.info(
                "Size=%7.2f MB: analytical=%.3f ms, real=%.3f ms, overhead=%.1f%%",
                size / (1024 * 1024), analytical.transfer_ms, real.transfer_ms, overhead_pct,
            )
        except Exception as e:
            logger.warning("Failed for size %d bytes: %s", size, e)
            results.append({
                "payload_bytes": size,
                "payload_mb": size / (1024 * 1024),
                "analytical_ms": analytical.transfer_ms,
                "real_ms": None,
                "connect_ms": None,
                "overhead_pct": None,
                "error": str(e),
            })

    return results
