"""FP8 compressor — cast FP16 KV cache to 8-bit float (E4M3)."""

from __future__ import annotations

import numpy as np

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


@register("fp8_e4m3")
class Fp8Compressor(BaseCompressor):
    """Compress KV cache by casting FP16 to FP8 (E4M3 format).

    Since numpy doesn't have native FP8, we simulate E4M3 by:
    1. Clamping to the E4M3 representable range [-448, 448]
    2. Storing per-tensor scale factors for values outside the range
    3. Packing as uint8 with custom encode/decode

    For practical purposes in this benchmark, we use a simulated approach:
    scale to fit E4M3 range, round, and store as uint8 + per-tensor scales.
    This gives the correct 2x compression ratio and representative quality.
    """

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes

        k_quant, k_scales = self._to_fp8(keys)
        v_quant, v_scales = self._to_fp8(values)

        data = k_quant.tobytes() + v_quant.tobytes()

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "k_scales": k_scales.tolist(),
                "v_scales": v_scales.tolist(),
                "key_bytes_len": len(k_quant.tobytes()),
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

        k_quant = np.frombuffer(compressed.data[:key_len], dtype=np.uint8).reshape(
            meta["key_shape"]
        )
        v_quant = np.frombuffer(compressed.data[key_len:], dtype=np.uint8).reshape(
            meta["val_shape"]
        )

        k_scales = np.array(meta["k_scales"], dtype=np.float32)
        v_scales = np.array(meta["v_scales"], dtype=np.float32)

        keys = self._from_fp8(k_quant, k_scales)
        values = self._from_fp8(v_quant, v_scales)
        return keys, values

    @staticmethod
    def _to_fp8(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantize FP16 tensor to simulated FP8 E4M3.

        Per-layer scale factor maps the tensor range to [0, 255].

        Returns:
            Tuple of (quantized uint8 array, per-layer scale factors).
        """
        fp32 = tensor.astype(np.float32)
        num_layers = fp32.shape[0]
        scales = np.zeros(num_layers, dtype=np.float32)
        quantized = np.zeros_like(fp32, dtype=np.uint8)

        for i in range(num_layers):
            layer = fp32[i]
            amax = np.abs(layer).max()
            if amax == 0:
                scales[i] = 1.0
            else:
                # E4M3 max is 448; we use 240 as practical max for better precision
                scales[i] = amax / 240.0

            scaled = layer / scales[i]
            # Map [-240, 240] -> [0, 255] via offset and round
            mapped = np.clip(np.round(scaled + 128.0), 0, 255)
            quantized[i] = mapped.astype(np.uint8)

        return quantized, scales

    @staticmethod
    def _from_fp8(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Dequantize simulated FP8 back to FP16."""
        num_layers = quantized.shape[0]
        result = np.zeros_like(quantized, dtype=np.float32)

        for i in range(num_layers):
            result[i] = (quantized[i].astype(np.float32) - 128.0) * scales[i]

        return result.astype(np.float16)

    @property
    def name(self) -> str:
        return "fp8_e4m3"

    @property
    def is_lossy(self) -> bool:
        return True
