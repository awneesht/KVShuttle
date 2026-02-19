"""Full compress-serialize-transfer-deserialize-decompress pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from kvshuttle.compression.base import BaseCompressor
from kvshuttle.profiling.timer import timer
from kvshuttle.transfer.serializer import deserialize, serialize
from kvshuttle.transfer.simulator import raw_transfer_ms, simulate_transfer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a full compress-transfer-decompress pipeline run.

    Attributes:
        compress_ms: Time to compress the KV cache.
        serialize_ms: Time to serialize to bytes.
        transfer_ms: Simulated transfer time.
        deserialize_ms: Time to deserialize from bytes.
        decompress_ms: Time to decompress back to arrays.
        total_ms: Total pipeline time (sum of all stages).
        raw_transfer_ms: Transfer time without compression (baseline).
        speedup: raw_transfer_ms / total_ms.
        original_bytes: Original KV cache size in bytes.
        compressed_bytes: Compressed payload size in bytes.
        compression_ratio: original_bytes / compressed_bytes.
        compressor_name: Name of the compressor used.
        bandwidth_gbps: Bandwidth used for simulation.
        attention_cosine_sim: Optional quality metric.
        perplexity_delta: Optional quality metric.
        token_agreement: Optional quality metric.
    """

    compress_ms: float
    serialize_ms: float
    transfer_ms: float
    deserialize_ms: float
    decompress_ms: float
    total_ms: float
    raw_transfer_ms: float
    speedup: float
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    compressor_name: str = ""
    bandwidth_gbps: float = 0.0
    attention_cosine_sim: float | None = None
    perplexity_delta: float | None = None
    token_agreement: float | None = None

    def summary(self) -> str:
        """Return a one-line summary of the pipeline result."""
        return (
            f"{self.compressor_name} @ {self.bandwidth_gbps:.0f} Gbps: "
            f"ratio={self.compression_ratio:.2f}x, "
            f"total={self.total_ms:.2f}ms, "
            f"raw={self.raw_transfer_ms:.2f}ms, "
            f"speedup={self.speedup:.2f}x"
        )


def run_pipeline(
    compressor: BaseCompressor,
    keys: np.ndarray,
    values: np.ndarray,
    bandwidth_gbps: float,
) -> PipelineResult:
    """Run the full compress-serialize-transfer-deserialize-decompress pipeline.

    Args:
        compressor: Compression strategy to use.
        keys: Key tensors, shape [num_layers, num_heads, seq_len, head_dim].
        values: Value tensors, same shape as keys.
        bandwidth_gbps: Simulated network bandwidth in Gbps.

    Returns:
        PipelineResult with timing and compression metrics.
    """
    original_bytes = keys.nbytes + values.nbytes

    # Compress
    with timer("compress") as t_compress:
        compressed = compressor.compress(keys, values)

    # Serialize
    with timer("serialize") as t_serialize:
        wire_bytes = serialize(compressed)

    # Simulate transfer
    payload_size = len(wire_bytes)
    transfer_result = simulate_transfer(payload_size, bandwidth_gbps)

    # Deserialize
    with timer("deserialize") as t_deserialize:
        compressed_back = deserialize(wire_bytes)

    # Decompress
    with timer("decompress") as t_decompress:
        _keys_out, _values_out = compressor.decompress(compressed_back)

    # Calculate metrics
    raw_ms = raw_transfer_ms(original_bytes, bandwidth_gbps)
    total_ms = (
        t_compress.elapsed_ms
        + t_serialize.elapsed_ms
        + transfer_result.transfer_ms
        + t_deserialize.elapsed_ms
        + t_decompress.elapsed_ms
    )
    speedup = raw_ms / total_ms if total_ms > 0 else float("inf")

    result = PipelineResult(
        compress_ms=t_compress.elapsed_ms,
        serialize_ms=t_serialize.elapsed_ms,
        transfer_ms=transfer_result.transfer_ms,
        deserialize_ms=t_deserialize.elapsed_ms,
        decompress_ms=t_decompress.elapsed_ms,
        total_ms=total_ms,
        raw_transfer_ms=raw_ms,
        speedup=speedup,
        original_bytes=original_bytes,
        compressed_bytes=compressed.compressed_size_bytes,
        compression_ratio=compressed.compression_ratio,
        compressor_name=compressor.name,
        bandwidth_gbps=bandwidth_gbps,
    )

    logger.info(result.summary())
    return result


def run_pipeline_pipelined(
    compressor: BaseCompressor,
    keys: np.ndarray,
    values: np.ndarray,
    bandwidth_gbps: float,
    num_chunks: int | None = None,
) -> PipelineResult:
    """Run pipeline with chunked pipelining to overlap compress/transfer/decompress.

    Models a realistic 3-stage pipeline where the KV cache is split into
    layer-chunks that flow through compress → transfer → decompress stages
    concurrently.

    Pipeline timing model:
        - Each chunk goes through: (compress+serialize) → transfer → (deserialize+decompress)
        - With N chunks, stages overlap:
          total = first_chunk_all_stages + (N-1) * bottleneck_stage
        - bottleneck_stage = max(compress_chunk, transfer_chunk, decompress_chunk)

    This models the key insight that in real systems (e.g., RDMA + GPU),
    compression of chunk i+1 overlaps with transfer of chunk i.

    Args:
        compressor: Compression strategy to use.
        keys: Key tensors, shape [num_layers, num_heads, seq_len, head_dim].
        values: Value tensors, same shape as keys.
        bandwidth_gbps: Simulated network bandwidth in Gbps.
        num_chunks: Number of pipeline chunks. Defaults to num_layers
            (one chunk per layer, the natural granularity).

    Returns:
        PipelineResult with pipelined timing.
    """
    original_bytes = keys.nbytes + values.nbytes
    num_layers = keys.shape[0]

    if num_chunks is None:
        num_chunks = num_layers
    num_chunks = min(num_chunks, num_layers)

    # Split layers into chunks
    chunk_boundaries = np.array_split(range(num_layers), num_chunks)

    # Measure per-chunk timings
    chunk_compress_ms = []
    chunk_transfer_ms = []
    chunk_decompress_ms = []
    total_compressed_bytes = 0

    for chunk_layers in chunk_boundaries:
        if len(chunk_layers) == 0:
            continue

        layer_slice = slice(chunk_layers[0], chunk_layers[-1] + 1)
        k_chunk = keys[layer_slice]
        v_chunk = values[layer_slice]

        # Compress + serialize
        with timer() as t_comp:
            compressed_chunk = compressor.compress(k_chunk, v_chunk)
        with timer() as t_ser:
            wire_bytes = serialize(compressed_chunk)

        chunk_compress_ms.append(t_comp.elapsed_ms + t_ser.elapsed_ms)

        # Transfer
        payload_size = len(wire_bytes)
        total_compressed_bytes += compressed_chunk.compressed_size_bytes
        xfer = simulate_transfer(payload_size, bandwidth_gbps)
        chunk_transfer_ms.append(xfer.transfer_ms)

        # Deserialize + decompress
        with timer() as t_deser:
            compressed_back = deserialize(wire_bytes)
        with timer() as t_decomp:
            compressor.decompress(compressed_back)

        chunk_decompress_ms.append(t_deser.elapsed_ms + t_decomp.elapsed_ms)

    n = len(chunk_compress_ms)
    if n == 0:
        raw_ms = raw_transfer_ms(original_bytes, bandwidth_gbps)
        return PipelineResult(
            compress_ms=0, serialize_ms=0, transfer_ms=0,
            deserialize_ms=0, decompress_ms=0, total_ms=0,
            raw_transfer_ms=raw_ms, speedup=float("inf"),
            original_bytes=original_bytes, compressed_bytes=0,
            compression_ratio=float("inf"),
            compressor_name=compressor.name, bandwidth_gbps=bandwidth_gbps,
        )

    # Pipeline timing model
    # First chunk must go through all 3 stages sequentially
    # Subsequent chunks overlap: only the bottleneck stage adds time
    total_compress = sum(chunk_compress_ms)
    total_transfer = sum(chunk_transfer_ms)
    total_decompress = sum(chunk_decompress_ms)

    if n == 1:
        # No pipelining possible with single chunk
        pipelined_total = total_compress + total_transfer + total_decompress
    else:
        # Avg per-chunk times
        avg_compress = total_compress / n
        avg_transfer = total_transfer / n
        avg_decompress = total_decompress / n

        # First chunk: sequential through all stages
        first_chunk_time = chunk_compress_ms[0] + chunk_transfer_ms[0] + chunk_decompress_ms[0]

        # Remaining chunks: each adds only the bottleneck stage duration
        bottleneck = max(avg_compress, avg_transfer, avg_decompress)
        pipelined_total = first_chunk_time + (n - 1) * bottleneck

    raw_ms = raw_transfer_ms(original_bytes, bandwidth_gbps)
    speedup = raw_ms / pipelined_total if pipelined_total > 0 else float("inf")

    # Report compression ratio using total compressed bytes
    if total_compressed_bytes > 0:
        ratio = original_bytes / total_compressed_bytes
    else:
        ratio = float("inf")

    result = PipelineResult(
        compress_ms=total_compress,
        serialize_ms=0.0,  # folded into compress_ms
        transfer_ms=total_transfer,
        deserialize_ms=0.0,  # folded into decompress_ms
        decompress_ms=total_decompress,
        total_ms=pipelined_total,
        raw_transfer_ms=raw_ms,
        speedup=speedup,
        original_bytes=original_bytes,
        compressed_bytes=total_compressed_bytes,
        compression_ratio=ratio,
        compressor_name=compressor.name,
        bandwidth_gbps=bandwidth_gbps,
    )

    logger.info(
        "PIPELINED %s @ %.0f Gbps: %d chunks, sequential=%.2fms, pipelined=%.2fms "
        "(bottleneck=%.2fms), speedup=%.2fx",
        compressor.name, bandwidth_gbps, n,
        total_compress + total_transfer + total_decompress,
        pipelined_total,
        max(total_compress / n, total_transfer / n, total_decompress / n) if n > 1 else 0,
        speedup,
    )

    return result


def run_pipeline_sweep(
    compressor: BaseCompressor,
    keys: np.ndarray,
    values: np.ndarray,
    bandwidths_gbps: list[float] | None = None,
) -> list[PipelineResult]:
    """Run the pipeline across multiple bandwidth points.

    Args:
        compressor: Compression strategy.
        keys: Key tensors.
        values: Value tensors.
        bandwidths_gbps: List of bandwidths to test. Defaults to standard sweep.

    Returns:
        List of PipelineResults, one per bandwidth.
    """
    from kvshuttle.transfer.simulator import BANDWIDTH_SWEEP_GBPS

    if bandwidths_gbps is None:
        bandwidths_gbps = [float(b) for b in BANDWIDTH_SWEEP_GBPS]

    results = []
    for bw in bandwidths_gbps:
        result = run_pipeline(compressor, keys, values, bw)
        results.append(result)
    return results
