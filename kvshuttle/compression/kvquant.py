"""KVQuant-style compression: non-uniform quantization + outlier handling.

Reference: KVQuant (Hooper et al., NeurIPS 2024) — arxiv:2401.18079

Key insight: KV cache values have non-uniform distributions with outliers.
Separate outlier channels and quantize them at higher precision, while
aggressively quantizing the rest at 2-bit. Uses per-channel sensitivity.
"""

from __future__ import annotations

import numpy as np

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


@register("kvquant_2bit")
class KvQuantCompressor(BaseCompressor):
    """KVQuant-style 2-bit quantization with outlier handling.

    1. Identify outlier channels (top percentile by magnitude).
    2. Store outliers at FP16 precision.
    3. Quantize remaining channels to 2-bit.

    Expected ~8x compression ratio.
    """

    def __init__(self, bits: int = 2, outlier_pct: float = 1.0):
        self._bits = bits
        self._outlier_pct = outlier_pct  # percentage of channels treated as outliers
        self._qmax = (1 << bits) - 1  # 3 for 2-bit

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes

        k_result = self._encode(keys)
        v_result = self._encode(values)

        data = k_result + v_result

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "bits": self._bits,
                "outlier_pct": self._outlier_pct,
                "k_len": len(k_result),
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
        k_len = meta["k_len"]

        keys = self._decode(compressed.data[:k_len], meta["key_shape"])
        values = self._decode(compressed.data[k_len:], meta["val_shape"])
        return keys, values

    def _encode(self, tensor: np.ndarray) -> bytes:
        """Encode with outlier separation and 2-bit quantization."""
        fp32 = tensor.astype(np.float32)
        num_layers, num_heads, seq_len, head_dim = fp32.shape

        # Find outlier channels globally by magnitude
        num_outliers = max(1, int(head_dim * self._outlier_pct / 100.0))
        channel_magnitudes = np.abs(fp32).mean(axis=(0, 1, 2))  # [head_dim]
        outlier_indices = np.argsort(channel_magnitudes)[-num_outliers:]
        normal_indices = np.setdiff1d(np.arange(head_dim), outlier_indices)

        # Outlier channels: store as FP16
        outlier_data = fp32[:, :, :, outlier_indices].astype(np.float16)

        # Normal channels: 2-bit per-layer quantization
        normal_data = fp32[:, :, :, normal_indices]

        # Per-layer asymmetric 2-bit quantization of normal channels
        scales = np.zeros(num_layers, dtype=np.float32)
        zeros = np.zeros(num_layers, dtype=np.float32)
        quant_flat = []

        for i in range(num_layers):
            layer = normal_data[i].reshape(-1)
            lmin, lmax = layer.min(), layer.max()
            rng = lmax - lmin
            if rng == 0:
                rng = 1.0
            scales[i] = rng / self._qmax
            zeros[i] = lmin
            q = np.clip(np.round((layer - zeros[i]) / scales[i]), 0, self._qmax).astype(np.uint8)
            quant_flat.append(q)

        quant_all = np.concatenate(quant_flat)

        # Pack 2-bit (4 values per byte)
        pad_len = (4 - len(quant_all) % 4) % 4
        if pad_len > 0:
            quant_all = np.append(quant_all, np.zeros(pad_len, dtype=np.uint8))
        packed = (
            (quant_all[0::4] << 6)
            | (quant_all[1::4] << 4)
            | (quant_all[2::4] << 2)
            | quant_all[3::4]
        )

        # Build binary payload
        parts = [
            np.array([num_outliers, len(normal_indices)], dtype=np.uint32).tobytes(),
            outlier_indices.astype(np.uint32).tobytes(),
            normal_indices.astype(np.uint32).tobytes(),
            outlier_data.tobytes(),
            packed.tobytes(),
            scales.tobytes(),
            zeros.tobytes(),
        ]

        # Store section lengths
        section_lens = np.array([len(p) for p in parts], dtype=np.uint32)
        return section_lens.tobytes() + b"".join(parts)

    def _decode(self, data: bytes, shape: list[int]) -> np.ndarray:
        """Decode outlier-separated 2-bit quantized tensor."""
        num_layers, num_heads, seq_len, head_dim = shape
        num_sections = 7
        offset = 0

        section_lens = np.frombuffer(data[: num_sections * 4], dtype=np.uint32)
        offset += num_sections * 4

        sections = []
        for slen in section_lens:
            sections.append(data[offset : offset + int(slen)])
            offset += int(slen)

        # Parse sections
        counts = np.frombuffer(sections[0], dtype=np.uint32)
        num_outliers = int(counts[0])
        num_normal = int(counts[1])

        outlier_indices = np.frombuffer(sections[1], dtype=np.uint32)
        normal_indices = np.frombuffer(sections[2], dtype=np.uint32)
        outlier_data = np.frombuffer(sections[3], dtype=np.float16).reshape(
            num_layers, num_heads, seq_len, num_outliers
        )
        packed = np.frombuffer(sections[4], dtype=np.uint8)
        scales = np.frombuffer(sections[5], dtype=np.float32)
        zeros = np.frombuffer(sections[6], dtype=np.float32)

        # Unpack 2-bit
        quant_all = np.zeros(len(packed) * 4, dtype=np.uint8)
        quant_all[0::4] = (packed >> 6) & 0x03
        quant_all[1::4] = (packed >> 4) & 0x03
        quant_all[2::4] = (packed >> 2) & 0x03
        quant_all[3::4] = packed & 0x03

        # Dequantize normal channels
        elements_per_layer = num_heads * seq_len * num_normal
        normal_result = np.zeros((num_layers, num_heads, seq_len, num_normal), dtype=np.float32)
        for i in range(num_layers):
            start = i * elements_per_layer
            end = start + elements_per_layer
            q = quant_all[start:end].astype(np.float32)
            normal_result[i] = (q * scales[i] + zeros[i]).reshape(num_heads, seq_len, num_normal)

        # Reconstruct full tensor
        result = np.zeros(shape, dtype=np.float32)
        result[:, :, :, outlier_indices] = outlier_data.astype(np.float32)
        result[:, :, :, normal_indices] = normal_result

        return result.astype(np.float16)

    @property
    def name(self) -> str:
        return "kvquant_2bit"
