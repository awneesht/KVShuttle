"""Hybrid compression strategies.

- palu_int4: SVD low-rank then INT4 quantize on latent
- mixed_k8v4: INT8 for keys, INT4 for values (asymmetric precision)
"""

from __future__ import annotations

import numpy as np

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


@register("mixed_k8v4")
class MixedK8V4Compressor(BaseCompressor):
    """Mixed precision: INT8 for keys, INT4 for values.

    Keys are more sensitive to quantization error, so we keep them at higher
    precision. Values tolerate more aggressive compression.
    Expected ~3x compression.
    """

    def __init__(self, group_size: int = 128):
        self._group_size = group_size

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes

        from kvshuttle.compression.uniform_quant import (
            _group_int4_quantize,
            _symmetric_quantize,
        )

        # Keys: INT8
        k_quant, k_scales = _symmetric_quantize(keys, bits=8)

        # Values: INT4
        v_packed, v_scales, v_zeros = _group_int4_quantize(values, self._group_size)

        kq = k_quant.tobytes()
        vp = v_packed.tobytes()
        vs = v_scales.tobytes()
        vz = v_zeros.tobytes()

        data = kq + vp + vs + vz

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "k_scales": k_scales.tolist(),
                "kq_len": len(kq),
                "vp_len": len(vp),
                "vs_len": len(vs),
                "num_v_groups": len(v_scales),
                "group_size": self._group_size,
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

        kq_end = meta["kq_len"]
        vp_end = kq_end + meta["vp_len"]
        vs_end = vp_end + meta["vs_len"]

        from kvshuttle.compression.uniform_quant import (
            _group_int4_dequantize,
            _symmetric_dequantize,
        )

        k_quant = np.frombuffer(buf[:kq_end], dtype=np.int8).reshape(meta["key_shape"])
        k_scales = np.array(meta["k_scales"], dtype=np.float32)
        keys = _symmetric_dequantize(k_quant, k_scales)

        v_packed = np.frombuffer(buf[kq_end:vp_end], dtype=np.uint8)
        v_scales = np.frombuffer(buf[vp_end:vs_end], dtype=np.float32)
        v_zeros = np.frombuffer(buf[vs_end:], dtype=np.float32)
        values = _group_int4_dequantize(
            v_packed, v_scales, v_zeros, meta["val_shape"], meta["group_size"]
        )

        return keys, values

    @property
    def name(self) -> str:
        return "mixed_k8v4"


@register("palu_int4")
class PaluInt4Compressor(BaseCompressor):
    """SVD low-rank followed by INT4 quantization on the latent representation.

    1. Apply truncated SVD to get low-rank factors.
    2. Quantize the factors to INT4.

    Expected 8-16x compression.
    """

    def __init__(self, rank_ratio: float = 0.25, group_size: int = 128):
        self._rank_ratio = rank_ratio
        self._group_size = group_size

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes
        num_layers, num_heads, seq_len, head_dim = keys.shape
        rank = max(1, int(min(seq_len, head_dim) * self._rank_ratio))

        from kvshuttle.compression.uniform_quant import _group_int4_quantize

        k_factors = self._svd_to_factors(keys, rank)
        v_factors = self._svd_to_factors(values, rank)

        # INT4 quantize the concatenated factors
        k_packed, k_scales, k_zeros = _group_int4_quantize(k_factors, self._group_size)
        v_packed, v_scales, v_zeros = _group_int4_quantize(v_factors, self._group_size)

        kp = k_packed.tobytes()
        vp = v_packed.tobytes()
        ks = k_scales.tobytes()
        vs = v_scales.tobytes()
        kz = k_zeros.tobytes()
        vz = v_zeros.tobytes()

        data = kp + vp + ks + vs + kz + vz

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "rank": rank,
                "factor_shape": list(k_factors.shape),
                "kp_len": len(kp),
                "vp_len": len(vp),
                "ks_len": len(ks),
                "vs_len": len(vs),
                "kz_len": len(kz),
                "num_k_groups": len(k_scales),
                "num_v_groups": len(v_scales),
                "group_size": self._group_size,
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
        rank = meta["rank"]

        kp_end = meta["kp_len"]
        vp_end = kp_end + meta["vp_len"]
        ks_end = vp_end + meta["ks_len"]
        vs_end = ks_end + meta["vs_len"]
        kz_end = vs_end + meta["kz_len"]

        from kvshuttle.compression.uniform_quant import _group_int4_dequantize

        k_packed = np.frombuffer(buf[:kp_end], dtype=np.uint8)
        v_packed = np.frombuffer(buf[kp_end:vp_end], dtype=np.uint8)
        k_scales = np.frombuffer(buf[vp_end:ks_end], dtype=np.float32)
        v_scales = np.frombuffer(buf[ks_end:vs_end], dtype=np.float32)
        k_zeros = np.frombuffer(buf[vs_end:kz_end], dtype=np.float32)
        v_zeros = np.frombuffer(buf[kz_end:], dtype=np.float32)

        factor_shape = meta["factor_shape"]
        k_factors = _group_int4_dequantize(k_packed, k_scales, k_zeros, factor_shape,
                                           meta["group_size"])
        v_factors = _group_int4_dequantize(v_packed, v_scales, v_zeros, factor_shape,
                                           meta["group_size"])

        keys = self._factors_to_tensor(k_factors, meta["key_shape"], rank)
        values = self._factors_to_tensor(v_factors, meta["val_shape"], rank)
        return keys, values

    def _svd_to_factors(self, tensor: np.ndarray, rank: int) -> np.ndarray:
        """Compute SVD factors and concatenate into a single array.

        For each (layer, head) slice: store [US | Vt] as [seq_len+head_dim, rank].
        """
        fp32 = tensor.astype(np.float32)
        num_layers, num_heads, seq_len, head_dim = fp32.shape

        # Output: [num_layers, num_heads, seq_len + head_dim, rank]
        factors = np.zeros((num_layers, num_heads, seq_len + head_dim, rank), dtype=np.float32)

        for l in range(num_layers):
            for h in range(num_heads):
                mat = fp32[l, h]
                U, S, Vt = np.linalg.svd(mat, full_matrices=False)
                US = U[:, :rank] * S[np.newaxis, :rank]
                Vt_r = Vt[:rank, :].T  # [head_dim, rank]
                factors[l, h, :seq_len, :] = US
                factors[l, h, seq_len:, :] = Vt_r

        return factors

    def _factors_to_tensor(self, factors: np.ndarray, shape: list[int], rank: int) -> np.ndarray:
        """Reconstruct tensor from SVD factors."""
        num_layers, num_heads, seq_len, head_dim = shape
        fp32 = factors.astype(np.float32)
        result = np.zeros(shape, dtype=np.float32)

        for l in range(num_layers):
            for h in range(num_heads):
                US = fp32[l, h, :seq_len, :]  # [seq_len, rank]
                Vt = fp32[l, h, seq_len:, :].T  # [rank, head_dim]
                result[l, h] = US @ Vt

        return result.astype(np.float16)

    @property
    def name(self) -> str:
        return "palu_int4"
