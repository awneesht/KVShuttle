"""KIVI-style compression: per-channel keys + per-token values, asymmetric 2-bit.

Reference: KIVI (Liu et al., 2024) — arxiv:2402.02750

Key insight: Keys are quantized per-channel (each head_dim channel gets its own
scale/zero), while Values are quantized per-token (each token position gets its
own scale/zero). This asymmetry reflects the different statistical properties
of keys vs values in attention computation.
"""

from __future__ import annotations

import numpy as np

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


@register("kivi_2bit")
class KiviCompressor(BaseCompressor):
    """KIVI-style 2-bit asymmetric quantization.

    Keys: per-channel quantization (scale/zero per head_dim channel).
    Values: per-token quantization (scale/zero per token position).
    Group size 128 for residual quantization.
    Expected ~8x compression ratio.
    """

    def __init__(self, bits: int = 2, group_size: int = 128):
        self._bits = bits
        self._group_size = group_size
        self._qmax = (1 << bits) - 1  # 3 for 2-bit

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes
        num_layers, num_heads, seq_len, head_dim = keys.shape

        # Keys: per-channel quantization (quantize along seq_len, per channel)
        k_quant, k_scales, k_zeros = self._quantize_per_channel(keys)

        # Values: per-token quantization (quantize along head_dim, per token)
        v_quant, v_scales, v_zeros = self._quantize_per_token(values)

        # Pack 2-bit values: 4 values per byte
        k_packed = _pack_nbits(k_quant, self._bits)
        v_packed = _pack_nbits(v_quant, self._bits)

        # Binary pack scales/zeros
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
        meta = compressed.metadata
        buf = compressed.data
        bits = meta["bits"]

        # Parse sections
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

        # Dequantize keys: per-channel (scales along head_dim axis)
        keys = self._dequantize_per_channel(k_quant, k_scales, k_zeros)
        values = self._dequantize_per_token(v_quant, v_scales, v_zeros)

        return keys.astype(np.float16), values.astype(np.float16)

    def _quantize_per_channel(self, tensor: np.ndarray):
        """Per-channel quantization: scale/zero per [layer, head, channel]."""
        fp32 = tensor.astype(np.float32)
        num_layers, num_heads, seq_len, head_dim = fp32.shape

        # Compute min/max along seq_len axis (axis=2)
        tmin = fp32.min(axis=2)  # [L, H, D]
        tmax = fp32.max(axis=2)  # [L, H, D]
        rng = tmax - tmin
        rng[rng == 0] = 1.0

        scales = rng / self._qmax  # [L, H, D]
        zeros = tmin               # [L, H, D]

        # Quantize
        quantized = np.clip(
            np.round((fp32 - zeros[:, :, np.newaxis, :]) / scales[:, :, np.newaxis, :]),
            0, self._qmax,
        ).astype(np.uint8)

        return quantized, scales, zeros

    def _dequantize_per_channel(self, quant, scales, zeros):
        """Dequantize per-channel."""
        return quant.astype(np.float32) * scales[:, :, np.newaxis, :] + zeros[:, :, np.newaxis, :]

    def _quantize_per_token(self, tensor: np.ndarray):
        """Per-token quantization: scale/zero per [layer, head, token]."""
        fp32 = tensor.astype(np.float32)

        # Compute min/max along head_dim axis (axis=3)
        tmin = fp32.min(axis=3)  # [L, H, S]
        tmax = fp32.max(axis=3)  # [L, H, S]
        rng = tmax - tmin
        rng[rng == 0] = 1.0

        scales = rng / self._qmax  # [L, H, S]
        zeros = tmin               # [L, H, S]

        quantized = np.clip(
            np.round((fp32 - zeros[:, :, :, np.newaxis]) / scales[:, :, :, np.newaxis]),
            0, self._qmax,
        ).astype(np.uint8)

        return quantized, scales, zeros

    def _dequantize_per_token(self, quant, scales, zeros):
        """Dequantize per-token."""
        return quant.astype(np.float32) * scales[:, :, :, np.newaxis] + zeros[:, :, :, np.newaxis]

    @property
    def name(self) -> str:
        return "kivi_2bit"


def _pack_nbits(arr: np.ndarray, bits: int) -> np.ndarray:
    """Pack an array of n-bit unsigned integers into uint8 bytes.

    For 2-bit: 4 values per byte. For 4-bit: 2 values per byte.
    """
    flat = arr.reshape(-1).astype(np.uint8)
    vals_per_byte = 8 // bits

    # Pad to multiple of vals_per_byte
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
