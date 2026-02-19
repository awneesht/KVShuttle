"""Cascade compression: prune then quantize (Titanus-style).

First prune low-importance tokens, then quantize the remainder.
Achieves multiplicative compression: prune_ratio * quant_ratio.
"""

from __future__ import annotations

import numpy as np

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.pruning import _select_important_tokens
from kvshuttle.compression.registry import register


@register("cascade_prune50_int4")
class CascadePruneInt4Compressor(BaseCompressor):
    """Prune 50% of tokens, then INT4 quantize the remainder.

    Expected ~8x compression (2x from pruning * 4x from INT4).
    """

    def __init__(self, keep_ratio: float = 0.5, group_size: int = 128):
        self._keep_ratio = keep_ratio
        self._group_size = group_size

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes
        num_layers, num_heads, seq_len, head_dim = keys.shape
        keep_count = max(1, int(seq_len * self._keep_ratio))

        # Step 1: Prune
        key_norms = np.linalg.norm(keys.astype(np.float32), axis=3).mean(axis=(0, 1))
        keep_indices = np.sort(_select_important_tokens(key_norms, keep_count, protect_ends=2))

        pruned_keys = keys[:, :, keep_indices, :].astype(np.float32)
        pruned_values = values[:, :, keep_indices, :].astype(np.float32)

        # Step 2: INT4 quantize
        from kvshuttle.compression.uniform_quant import _group_int4_quantize

        k_packed, k_scales, k_zeros = _group_int4_quantize(pruned_keys, self._group_size)
        v_packed, v_scales, v_zeros = _group_int4_quantize(pruned_values, self._group_size)

        # Pack binary
        kp = k_packed.tobytes()
        vp = v_packed.tobytes()
        ks = k_scales.tobytes()
        vs = v_scales.tobytes()
        kz = k_zeros.tobytes()
        vz = v_zeros.tobytes()
        idx = keep_indices.astype(np.uint32).tobytes()

        data = kp + vp + ks + vs + kz + vz + idx

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "keep_count": keep_count,
                "pruned_shape": [num_layers, num_heads, keep_count, head_dim],
                "kp_len": len(kp),
                "vp_len": len(vp),
                "ks_len": len(ks),
                "vs_len": len(vs),
                "kz_len": len(kz),
                "vz_len": len(vz),
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

        kp_end = meta["kp_len"]
        vp_end = kp_end + meta["vp_len"]
        ks_end = vp_end + meta["ks_len"]
        vs_end = ks_end + meta["vs_len"]
        kz_end = vs_end + meta["kz_len"]
        vz_end = kz_end + meta["vz_len"]

        k_packed = np.frombuffer(buf[:kp_end], dtype=np.uint8)
        v_packed = np.frombuffer(buf[kp_end:vp_end], dtype=np.uint8)
        k_scales = np.frombuffer(buf[vp_end:ks_end], dtype=np.float32)
        v_scales = np.frombuffer(buf[ks_end:vs_end], dtype=np.float32)
        k_zeros = np.frombuffer(buf[vs_end:kz_end], dtype=np.float32)
        v_zeros = np.frombuffer(buf[kz_end:vz_end], dtype=np.float32)
        indices = np.frombuffer(buf[vz_end:], dtype=np.uint32)

        from kvshuttle.compression.uniform_quant import _group_int4_dequantize

        pruned_shape = meta["pruned_shape"]
        keys_pruned = _group_int4_dequantize(
            k_packed, k_scales, k_zeros, pruned_shape, meta["group_size"]
        )
        values_pruned = _group_int4_dequantize(
            v_packed, v_scales, v_zeros, pruned_shape, meta["group_size"]
        )

        # Reconstruct full tensor
        orig_k_shape = meta["key_shape"]
        orig_v_shape = meta["val_shape"]
        keys = np.zeros(orig_k_shape, dtype=np.float16)
        values = np.zeros(orig_v_shape, dtype=np.float16)
        keys[:, :, indices, :] = keys_pruned
        values[:, :, indices, :] = values_pruned

        return keys, values

    @property
    def name(self) -> str:
        return "cascade_prune50_int4"
