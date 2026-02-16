"""CacheGen-style compression: delta encoding + grouped quantization.

Reference: CacheGen (Liu et al., SIGCOMM 2024) — arxiv:2310.07240

Key insight: Group tokens into chunks. First token in each chunk is an anchor
quantized independently. Subsequent tokens store deltas from the anchor.
Deltas have smaller range, enabling better quantization efficiency.
"""

from __future__ import annotations

import numpy as np

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


@register("cachegen")
class CacheGenCompressor(BaseCompressor):
    """CacheGen-style delta encoding with grouped quantization.

    Groups tokens into chunks of `chunk_size`. For each chunk:
    1. First token = anchor, quantized to `anchor_bits`
    2. Remaining tokens = delta from anchor, quantized to `delta_bits`

    Expected ~3.5-4.3x compression ratio.
    """

    def __init__(self, chunk_size: int = 10, anchor_bits: int = 8, delta_bits: int = 4):
        self._chunk_size = chunk_size
        self._anchor_bits = anchor_bits
        self._delta_bits = delta_bits

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes

        k_data = self._encode(keys)
        v_data = self._encode(values)

        k_bytes = k_data.tobytes()
        v_bytes = v_data.tobytes()
        data = k_bytes + v_bytes

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "chunk_size": self._chunk_size,
                "anchor_bits": self._anchor_bits,
                "delta_bits": self._delta_bits,
                "key_bytes_len": len(k_bytes),
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

        k_data = np.frombuffer(compressed.data[:key_len], dtype=np.uint8)
        v_data = np.frombuffer(compressed.data[key_len:], dtype=np.uint8)

        keys = self._decode(k_data, meta["key_shape"])
        values = self._decode(v_data, meta["val_shape"])
        return keys, values

    def _encode(self, tensor: np.ndarray) -> np.ndarray:
        """Encode tensor using delta-based grouped quantization.

        For simplicity and speed, we use a flat approach:
        - Anchors quantized to anchor_bits (stored as uint8)
        - Deltas quantized to delta_bits (packed)

        We store them sequentially: [all_anchors | all_deltas | scales_and_zeros]
        """
        fp32 = tensor.astype(np.float32)
        num_layers, num_heads, seq_len, head_dim = fp32.shape
        chunk_size = self._chunk_size

        # Reshape to process per (layer, head) slice
        flat = fp32.reshape(num_layers * num_heads, seq_len, head_dim)

        all_bytes = []
        for slice_2d in flat:
            # slice_2d: [seq_len, head_dim]
            num_chunks = (seq_len + chunk_size - 1) // chunk_size

            anchor_quants = []
            delta_quants = []
            params = []  # (anchor_scale, anchor_zero, delta_scale, delta_zero) per chunk

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, seq_len)
                chunk = slice_2d[start:end]  # [chunk_len, head_dim]

                # Anchor: first token
                anchor = chunk[0]  # [head_dim]
                a_min, a_max = anchor.min(), anchor.max()
                a_rng = a_max - a_min
                if a_rng == 0:
                    a_rng = 1.0
                a_scale = a_rng / 255.0
                a_zero = a_min
                a_quant = np.clip(np.round((anchor - a_zero) / a_scale), 0, 255).astype(np.uint8)
                anchor_quants.append(a_quant)

                # Deltas: remaining tokens - anchor
                if end - start > 1:
                    deltas = chunk[1:] - anchor[np.newaxis, :]  # [chunk_len-1, head_dim]
                    d_min, d_max = deltas.min(), deltas.max()
                    d_rng = d_max - d_min
                    if d_rng == 0:
                        d_rng = 1.0
                    d_scale = d_rng / 15.0
                    d_zero = d_min
                    d_quant = np.clip(
                        np.round((deltas - d_zero) / d_scale), 0, 15
                    ).astype(np.uint8)
                    delta_quants.append(d_quant.reshape(-1))
                else:
                    d_scale = 1.0
                    d_zero = 0.0

                params.append(
                    np.array([a_scale, a_zero, d_scale, d_zero], dtype=np.float32)
                )

            # Concatenate anchor bytes
            anchors_flat = np.concatenate(anchor_quants)  # uint8

            # Pack deltas (4-bit, 2 per byte)
            if delta_quants:
                deltas_flat = np.concatenate(delta_quants)
                if len(deltas_flat) % 2 != 0:
                    deltas_flat = np.append(deltas_flat, np.uint8(0))
                packed_deltas = (deltas_flat[0::2] << 4) | deltas_flat[1::2]
            else:
                packed_deltas = np.array([], dtype=np.uint8)

            params_flat = np.concatenate(params)  # float32

            # Header: lengths as uint32
            header = np.array(
                [len(anchors_flat), len(packed_deltas), len(params_flat)], dtype=np.uint32
            )

            all_bytes.append(header.tobytes())
            all_bytes.append(anchors_flat.tobytes())
            all_bytes.append(packed_deltas.tobytes())
            all_bytes.append(params_flat.tobytes())

        return np.frombuffer(b"".join(all_bytes), dtype=np.uint8)

    def _decode(self, data: np.ndarray, shape: list[int]) -> np.ndarray:
        """Decode delta-encoded tensor."""
        num_layers, num_heads, seq_len, head_dim = shape
        chunk_size = self._chunk_size
        buf = data.tobytes()
        offset = 0

        result = np.zeros((num_layers * num_heads, seq_len, head_dim), dtype=np.float32)

        for s in range(num_layers * num_heads):
            # Read header
            header = np.frombuffer(buf[offset : offset + 12], dtype=np.uint32)
            anchor_len, delta_len, params_len = int(header[0]), int(header[1]), int(header[2])
            offset += 12

            anchors = np.frombuffer(buf[offset : offset + anchor_len], dtype=np.uint8)
            offset += anchor_len

            packed_deltas = np.frombuffer(buf[offset : offset + delta_len], dtype=np.uint8)
            offset += delta_len

            params = np.frombuffer(buf[offset : offset + params_len * 4], dtype=np.float32)
            offset += params_len * 4

            # Unpack deltas
            if len(packed_deltas) > 0:
                high = (packed_deltas >> 4) & 0x0F
                low = packed_deltas & 0x0F
                deltas_flat = np.empty(len(packed_deltas) * 2, dtype=np.uint8)
                deltas_flat[0::2] = high
                deltas_flat[1::2] = low
            else:
                deltas_flat = np.array([], dtype=np.uint8)

            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            anchor_offset = 0
            delta_offset = 0

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start

                p = params[c * 4 : (c + 1) * 4]
                a_scale, a_zero, d_scale, d_zero = p[0], p[1], p[2], p[3]

                # Decode anchor
                a_quant = anchors[anchor_offset : anchor_offset + head_dim]
                anchor_offset += head_dim
                anchor = a_quant.astype(np.float32) * a_scale + a_zero
                result[s, start] = anchor

                # Decode deltas
                if chunk_len > 1:
                    n_delta_vals = (chunk_len - 1) * head_dim
                    d_quant = deltas_flat[delta_offset : delta_offset + n_delta_vals]
                    delta_offset += n_delta_vals
                    deltas = d_quant.astype(np.float32) * d_scale + d_zero
                    deltas = deltas.reshape(chunk_len - 1, head_dim)
                    result[s, start + 1 : end] = anchor[np.newaxis, :] + deltas

        return result.reshape(shape).astype(np.float16)

    @property
    def name(self) -> str:
        return "cachegen"
