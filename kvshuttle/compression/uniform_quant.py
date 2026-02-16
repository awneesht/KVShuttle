"""Uniform quantization — per-tensor symmetric INT8 and per-group INT4."""

from __future__ import annotations

import numpy as np

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


@register("uniform_int8")
class UniformInt8Compressor(BaseCompressor):
    """Per-tensor symmetric INT8 quantization.

    Each layer is quantized independently with a single scale factor.
    Gives 2x compression ratio.
    """

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes

        k_quant, k_scales = _symmetric_quantize(keys, bits=8)
        v_quant, v_scales = _symmetric_quantize(values, bits=8)

        data = k_quant.tobytes() + v_quant.tobytes()

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "k_scales": k_scales.tolist(),
                "v_scales": v_scales.tolist(),
                "key_bytes_len": len(k_quant.tobytes()),
                "bits": 8,
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

        k_quant = np.frombuffer(compressed.data[:key_len], dtype=np.int8).reshape(
            meta["key_shape"]
        )
        v_quant = np.frombuffer(compressed.data[key_len:], dtype=np.int8).reshape(
            meta["val_shape"]
        )

        k_scales = np.array(meta["k_scales"], dtype=np.float32)
        v_scales = np.array(meta["v_scales"], dtype=np.float32)

        keys = _symmetric_dequantize(k_quant, k_scales)
        values = _symmetric_dequantize(v_quant, v_scales)
        return keys, values

    @property
    def name(self) -> str:
        return "uniform_int8"


@register("uniform_int4")
class UniformInt4Compressor(BaseCompressor):
    """Per-group INT4 quantization with group size 128.

    Two INT4 values packed into one uint8 byte.
    Gives ~4x compression ratio.
    """

    def __init__(self, group_size: int = 128):
        self._group_size = group_size

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes

        k_packed, k_scales, k_zeros = _group_int4_quantize(keys, self._group_size)
        v_packed, v_scales, v_zeros = _group_int4_quantize(values, self._group_size)

        # Pack scales/zeros as binary (float32) instead of JSON to avoid metadata bloat
        k_packed_bytes = k_packed.tobytes()
        v_packed_bytes = v_packed.tobytes()
        k_scales_bytes = k_scales.tobytes()
        v_scales_bytes = v_scales.tobytes()
        k_zeros_bytes = k_zeros.tobytes()
        v_zeros_bytes = v_zeros.tobytes()

        data = (
            k_packed_bytes + v_packed_bytes
            + k_scales_bytes + v_scales_bytes
            + k_zeros_bytes + v_zeros_bytes
        )

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "key_packed_len": len(k_packed_bytes),
                "val_packed_len": len(v_packed_bytes),
                "k_scales_len": len(k_scales_bytes),
                "v_scales_len": len(v_scales_bytes),
                "k_zeros_len": len(k_zeros_bytes),
                "num_k_groups": len(k_scales),
                "num_v_groups": len(v_scales),
                "group_size": self._group_size,
                "bits": 4,
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
        buf = compressed.data

        # Parse offsets
        kp_end = meta["key_packed_len"]
        vp_end = kp_end + meta["val_packed_len"]
        ks_end = vp_end + meta["k_scales_len"]
        vs_end = ks_end + meta["v_scales_len"]
        kz_end = vs_end + meta["k_zeros_len"]
        # vz goes to end

        k_packed = np.frombuffer(buf[:kp_end], dtype=np.uint8)
        v_packed = np.frombuffer(buf[kp_end:vp_end], dtype=np.uint8)
        k_scales = np.frombuffer(buf[vp_end:ks_end], dtype=np.float32)
        v_scales = np.frombuffer(buf[ks_end:vs_end], dtype=np.float32)
        k_zeros = np.frombuffer(buf[vs_end:kz_end], dtype=np.float32)
        v_zeros = np.frombuffer(buf[kz_end:], dtype=np.float32)

        key_shape = meta["key_shape"]
        val_shape = meta["val_shape"]
        group_size = meta["group_size"]

        keys = _group_int4_dequantize(k_packed, k_scales, k_zeros, key_shape, group_size)
        values = _group_int4_dequantize(v_packed, v_scales, v_zeros, val_shape, group_size)
        return keys, values

    @property
    def name(self) -> str:
        return "uniform_int4"


def _symmetric_quantize(
    tensor: np.ndarray, bits: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer symmetric quantization.

    Args:
        tensor: Shape [num_layers, ...].
        bits: Number of bits (8 for INT8).

    Returns:
        Tuple of (quantized int8 array, per-layer scales).
    """
    fp32 = tensor.astype(np.float32)
    num_layers = fp32.shape[0]
    qmax = (1 << (bits - 1)) - 1  # 127 for 8-bit

    scales = np.zeros(num_layers, dtype=np.float32)
    quantized = np.zeros(fp32.shape, dtype=np.int8)

    for i in range(num_layers):
        layer = fp32[i]
        amax = np.abs(layer).max()
        if amax == 0:
            scales[i] = 1.0
        else:
            scales[i] = amax / qmax
        quantized[i] = np.clip(np.round(layer / scales[i]), -qmax, qmax).astype(np.int8)

    return quantized, scales


def _symmetric_dequantize(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize symmetric quantized tensor back to FP16."""
    num_layers = quantized.shape[0]
    result = np.zeros(quantized.shape, dtype=np.float32)

    for i in range(num_layers):
        result[i] = quantized[i].astype(np.float32) * scales[i]

    return result.astype(np.float16)


def _group_int4_quantize(
    tensor: np.ndarray, group_size: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-group asymmetric INT4 quantization.

    Flattens the tensor, groups elements, quantizes each group to 4-bit [0, 15],
    and packs two values per uint8 byte.

    Returns:
        Tuple of (packed uint8 array, scales per group, zeros per group).
    """
    fp32 = tensor.astype(np.float32)
    flat = fp32.reshape(-1)

    # Pad to multiple of group_size
    padded_len = ((len(flat) + group_size - 1) // group_size) * group_size
    padded = np.zeros(padded_len, dtype=np.float32)
    padded[: len(flat)] = flat

    num_groups = padded_len // group_size
    grouped = padded.reshape(num_groups, group_size)

    scales = np.zeros(num_groups, dtype=np.float32)
    zeros = np.zeros(num_groups, dtype=np.float32)
    quantized = np.zeros_like(grouped, dtype=np.uint8)

    for g in range(num_groups):
        group = grouped[g]
        gmin = group.min()
        gmax = group.max()
        rng = gmax - gmin
        if rng == 0:
            scales[g] = 1.0
            zeros[g] = gmin
        else:
            scales[g] = rng / 15.0
            zeros[g] = gmin
        quantized[g] = np.clip(np.round((group - zeros[g]) / scales[g]), 0, 15).astype(np.uint8)

    # Pack two 4-bit values per byte
    flat_q = quantized.reshape(-1)
    # Ensure even length for packing
    if len(flat_q) % 2 != 0:
        flat_q = np.append(flat_q, np.uint8(0))
    packed = (flat_q[0::2] << 4) | flat_q[1::2]

    return packed, scales, zeros


def _group_int4_dequantize(
    packed: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    original_shape: list[int],
    group_size: int = 128,
) -> np.ndarray:
    """Dequantize packed INT4 back to FP16."""
    # Unpack
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    flat_q = np.empty(len(packed) * 2, dtype=np.uint8)
    flat_q[0::2] = high
    flat_q[1::2] = low

    total_elements = 1
    for s in original_shape:
        total_elements *= s

    padded_len = len(scales) * group_size
    grouped = flat_q[:padded_len].reshape(len(scales), group_size)

    result = np.zeros_like(grouped, dtype=np.float32)
    for g in range(len(scales)):
        result[g] = grouped[g].astype(np.float32) * scales[g] + zeros[g]

    flat_result = result.reshape(-1)[:total_elements]
    return flat_result.reshape(original_shape).astype(np.float16)
