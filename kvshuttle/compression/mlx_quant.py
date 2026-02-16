"""Metal GPU-accelerated quantization compressors using MLX.

Implements uniform_int8 and kivi_2bit using mlx.core operations,
which run on Apple Silicon Metal GPU. Used to calibrate the
CPU-to-GPU speedup factor for compression timing.
"""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx

    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


def _require_mlx() -> None:
    if not _HAS_MLX:
        raise RuntimeError("MLX is required for Metal-accelerated compressors. Install with: pip install mlx")


# ---------------------------------------------------------------------------
# MLX Uniform INT8 — per-layer symmetric quantization on Metal GPU
# ---------------------------------------------------------------------------


@register("mlx_uniform_int8")
class MLXUniformInt8Compressor(BaseCompressor):
    """Per-layer symmetric INT8 quantization using MLX (Metal GPU).

    Functionally identical to UniformInt8Compressor but runs quantization
    on the Metal GPU via MLX for accurate GPU timing.
    """

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        _require_mlx()
        original_size = keys.nbytes + values.nbytes

        k_quant, k_scales = _mlx_symmetric_quantize(keys, bits=8)
        v_quant, v_scales = _mlx_symmetric_quantize(values, bits=8)

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
        _require_mlx()
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

        keys = _mlx_symmetric_dequantize(k_quant, k_scales)
        values = _mlx_symmetric_dequantize(v_quant, v_scales)
        return keys, values

    @property
    def name(self) -> str:
        return "mlx_uniform_int8"


# ---------------------------------------------------------------------------
# MLX KIVI 2-bit — per-channel keys + per-token values on Metal GPU
# ---------------------------------------------------------------------------


@register("mlx_kivi_2bit")
class MLXKiviCompressor(BaseCompressor):
    """KIVI-style 2-bit asymmetric quantization using MLX (Metal GPU).

    Keys: per-channel quantization (scale/zero per head_dim channel).
    Values: per-token quantization (scale/zero per token position).
    """

    def __init__(self, bits: int = 2):
        self._bits = bits
        self._qmax = (1 << bits) - 1  # 3 for 2-bit

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        _require_mlx()
        original_size = keys.nbytes + values.nbytes
        num_layers, num_heads, seq_len, head_dim = keys.shape

        # Keys: per-channel quantization
        k_quant, k_scales, k_zeros = _mlx_quantize_per_channel(keys, self._qmax)

        # Values: per-token quantization
        v_quant, v_scales, v_zeros = _mlx_quantize_per_token(values, self._qmax)

        # Pack 2-bit values: 4 values per byte (on CPU — packing is cheap)
        k_packed = _pack_nbits(k_quant, self._bits)
        v_packed = _pack_nbits(v_quant, self._bits)

        kp_bytes = k_packed.tobytes()
        vp_bytes = v_packed.tobytes()
        ks_bytes = k_scales.tobytes()
        vs_bytes = v_scales.tobytes()
        kz_bytes = k_zeros.tobytes()
        vz_bytes = v_zeros.tobytes()

        data = kp_bytes + vp_bytes + ks_bytes + vs_bytes + kz_bytes + vz_bytes

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "bits": self._bits,
                "kp_len": len(kp_bytes),
                "vp_len": len(vp_bytes),
                "ks_len": len(ks_bytes),
                "vs_len": len(vs_bytes),
                "kz_len": len(kz_bytes),
                "k_scales_shape": list(k_scales.shape),
                "v_scales_shape": list(v_scales.shape),
            },
            original_size_bytes=original_size,
            compressed_size_bytes=len(data),
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    def decompress(self, compressed: CompressedKVCache) -> tuple[np.ndarray, np.ndarray]:
        _require_mlx()
        meta = compressed.metadata
        buf = compressed.data
        bits = meta["bits"]

        kp_end = meta["kp_len"]
        vp_end = kp_end + meta["vp_len"]
        ks_end = vp_end + meta["ks_len"]
        vs_end = ks_end + meta["vs_len"]
        kz_end = vs_end + meta["kz_len"]

        k_packed = np.frombuffer(buf[:kp_end], dtype=np.uint8)
        v_packed = np.frombuffer(buf[kp_end:vp_end], dtype=np.uint8)
        k_scales = np.frombuffer(buf[vp_end:ks_end], dtype=np.float32).reshape(
            meta["k_scales_shape"]
        )
        v_scales = np.frombuffer(buf[ks_end:vs_end], dtype=np.float32).reshape(
            meta["v_scales_shape"]
        )
        k_zeros = np.frombuffer(buf[vs_end:kz_end], dtype=np.float32).reshape(
            meta["k_scales_shape"]
        )
        v_zeros = np.frombuffer(buf[kz_end:], dtype=np.float32).reshape(
            meta["v_scales_shape"]
        )

        key_shape = meta["key_shape"]
        val_shape = meta["val_shape"]
        total_k = 1
        total_v = 1
        for s in key_shape:
            total_k *= s
        for s in val_shape:
            total_v *= s

        k_quant = _unpack_nbits(k_packed, bits, total_k).reshape(key_shape)
        v_quant = _unpack_nbits(v_packed, bits, total_v).reshape(val_shape)

        # Dequantize on MLX GPU
        keys = _mlx_dequantize_per_channel(k_quant, k_scales, k_zeros)
        values = _mlx_dequantize_per_token(v_quant, v_scales, v_zeros)

        return keys.astype(np.float16), values.astype(np.float16)

    @property
    def name(self) -> str:
        return "mlx_kivi_2bit"


# ---------------------------------------------------------------------------
# MLX quantization helpers
# ---------------------------------------------------------------------------


def _mlx_symmetric_quantize(
    tensor: np.ndarray, bits: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer symmetric quantization using MLX Metal GPU."""
    qmax = (1 << (bits - 1)) - 1  # 127 for 8-bit

    # Move to MLX GPU
    x = mx.array(tensor.astype(np.float32))
    num_layers = x.shape[0]

    # Reshape to [num_layers, -1] for vectorized per-layer processing
    flat = mx.reshape(x, (num_layers, -1))

    # Compute per-layer absolute max
    amax = mx.max(mx.abs(flat), axis=1)  # [num_layers]
    # Avoid division by zero
    amax = mx.where(amax == 0, mx.ones_like(amax), amax)
    scales = amax / qmax  # [num_layers]

    # Quantize: round(x / scale), clip to [-qmax, qmax]
    scales_expanded = mx.reshape(scales, (num_layers,) + (1,) * (len(x.shape) - 1))
    quantized = mx.clip(mx.round(x / scales_expanded), -qmax, qmax)

    # Force evaluation on GPU before converting back
    mx.eval(quantized, scales)

    # Convert back to numpy
    q_np = np.array(quantized, copy=False).astype(np.int8)
    s_np = np.array(scales, copy=False).astype(np.float32)

    return q_np, s_np


def _mlx_symmetric_dequantize(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize symmetric quantized tensor using MLX Metal GPU."""
    x = mx.array(quantized.astype(np.float32))
    s = mx.array(scales)
    num_layers = x.shape[0]

    s_expanded = mx.reshape(s, (num_layers,) + (1,) * (len(x.shape) - 1))
    result = x * s_expanded

    mx.eval(result)
    return np.array(result, copy=False).astype(np.float16)


def _mlx_quantize_per_channel(
    tensor: np.ndarray, qmax: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel quantization (for keys): scale/zero per [layer, head, channel].

    Quantizes along seq_len axis (axis=2).
    """
    x = mx.array(tensor.astype(np.float32))

    # Min/max along seq_len axis (axis=2)
    tmin = mx.min(x, axis=2)  # [L, H, D]
    tmax = mx.max(x, axis=2)  # [L, H, D]
    rng = tmax - tmin
    rng = mx.where(rng == 0, mx.ones_like(rng), rng)

    scales = rng / qmax  # [L, H, D]
    zeros = tmin          # [L, H, D]

    # Expand dims for broadcasting: [L, H, 1, D]
    scales_exp = mx.expand_dims(scales, axis=2)
    zeros_exp = mx.expand_dims(zeros, axis=2)

    quantized = mx.clip(mx.round((x - zeros_exp) / scales_exp), 0, qmax)

    mx.eval(quantized, scales, zeros)

    q_np = np.array(quantized, copy=False).astype(np.uint8)
    s_np = np.array(scales, copy=False).astype(np.float32)
    z_np = np.array(zeros, copy=False).astype(np.float32)

    return q_np, s_np, z_np


def _mlx_dequantize_per_channel(
    quant: np.ndarray, scales: np.ndarray, zeros: np.ndarray
) -> np.ndarray:
    """Dequantize per-channel using MLX."""
    x = mx.array(quant.astype(np.float32))
    s = mx.array(scales)
    z = mx.array(zeros)

    s_exp = mx.expand_dims(s, axis=2)
    z_exp = mx.expand_dims(z, axis=2)
    result = x * s_exp + z_exp

    mx.eval(result)
    return np.array(result, copy=False).astype(np.float32)


def _mlx_quantize_per_token(
    tensor: np.ndarray, qmax: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-token quantization (for values): scale/zero per [layer, head, token].

    Quantizes along head_dim axis (axis=3).
    """
    x = mx.array(tensor.astype(np.float32))

    # Min/max along head_dim axis (axis=3)
    tmin = mx.min(x, axis=3)  # [L, H, S]
    tmax = mx.max(x, axis=3)  # [L, H, S]
    rng = tmax - tmin
    rng = mx.where(rng == 0, mx.ones_like(rng), rng)

    scales = rng / qmax  # [L, H, S]
    zeros = tmin          # [L, H, S]

    # Expand dims for broadcasting: [L, H, S, 1]
    scales_exp = mx.expand_dims(scales, axis=3)
    zeros_exp = mx.expand_dims(zeros, axis=3)

    quantized = mx.clip(mx.round((x - zeros_exp) / scales_exp), 0, qmax)

    mx.eval(quantized, scales, zeros)

    q_np = np.array(quantized, copy=False).astype(np.uint8)
    s_np = np.array(scales, copy=False).astype(np.float32)
    z_np = np.array(zeros, copy=False).astype(np.float32)

    return q_np, s_np, z_np


def _mlx_dequantize_per_token(
    quant: np.ndarray, scales: np.ndarray, zeros: np.ndarray
) -> np.ndarray:
    """Dequantize per-token using MLX."""
    x = mx.array(quant.astype(np.float32))
    s = mx.array(scales)
    z = mx.array(zeros)

    s_exp = mx.expand_dims(s, axis=3)
    z_exp = mx.expand_dims(z, axis=3)
    result = x * s_exp + z_exp

    mx.eval(result)
    return np.array(result, copy=False).astype(np.float32)


# ---------------------------------------------------------------------------
# Bit packing helpers (run on CPU — trivial cost vs quantization)
# ---------------------------------------------------------------------------


def _pack_nbits(arr: np.ndarray, bits: int) -> np.ndarray:
    """Pack n-bit unsigned integers into uint8 bytes."""
    flat = arr.reshape(-1).astype(np.uint8)
    vals_per_byte = 8 // bits

    pad_len = (vals_per_byte - len(flat) % vals_per_byte) % vals_per_byte
    if pad_len > 0:
        flat = np.append(flat, np.zeros(pad_len, dtype=np.uint8))

    packed = np.zeros(len(flat) // vals_per_byte, dtype=np.uint8)
    for i in range(vals_per_byte):
        packed |= flat[i::vals_per_byte] << (bits * (vals_per_byte - 1 - i))

    return packed


def _unpack_nbits(packed: np.ndarray, bits: int, total_elements: int) -> np.ndarray:
    """Unpack uint8 bytes back to n-bit values."""
    vals_per_byte = 8 // bits
    mask = (1 << bits) - 1

    flat = np.zeros(len(packed) * vals_per_byte, dtype=np.uint8)
    for i in range(vals_per_byte):
        flat[i::vals_per_byte] = (packed >> (bits * (vals_per_byte - 1 - i))) & mask

    return flat[:total_elements]
