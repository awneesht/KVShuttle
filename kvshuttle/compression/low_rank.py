"""Palu-style low-rank compression via SVD projection.

Reference: Palu (ICLR 2025)

Key insight: KV projections have low effective rank. Apply truncated SVD to
compress the seq_len dimension. Store low-rank factors instead of full matrix.
Rank can be selected per layer based on singular value spectrum.
"""

from __future__ import annotations

import numpy as np

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


@register("palu_lr")
class PaluLowRankCompressor(BaseCompressor):
    """Palu-style SVD low-rank compression.

    For each (layer, head) slice of shape [seq_len, head_dim]:
    - Compute truncated SVD: A ≈ U @ diag(S) @ Vt
    - Store U[:, :rank] @ diag(S[:rank]) and Vt[:rank, :] as compressed form
    - Compression ratio ≈ seq_len * head_dim / (rank * (seq_len + head_dim))

    Expected 4-11x compression depending on rank ratio.
    """

    def __init__(self, rank_ratio: float = 0.25):
        """Initialize with rank ratio.

        Args:
            rank_ratio: Fraction of min(seq_len, head_dim) to keep. Lower = more compression.
        """
        self._rank_ratio = rank_ratio

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes
        num_layers, num_heads, seq_len, head_dim = keys.shape

        rank = max(1, int(min(seq_len, head_dim) * self._rank_ratio))

        k_factors = self._svd_compress(keys, rank)
        v_factors = self._svd_compress(values, rank)

        k_bytes = k_factors.tobytes()
        v_bytes = v_factors.tobytes()
        data = k_bytes + v_bytes

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "rank": rank,
                "rank_ratio": self._rank_ratio,
                "key_bytes_len": len(k_bytes),
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
        key_len = meta["key_bytes_len"]
        rank = meta["rank"]

        keys = self._svd_decompress(compressed.data[:key_len], meta["key_shape"], rank)
        values = self._svd_decompress(compressed.data[key_len:], meta["val_shape"], rank)
        return keys, values

    def _svd_compress(self, tensor: np.ndarray, rank: int) -> np.ndarray:
        """Compress via truncated SVD.

        For each (layer, head) slice, compute U_r @ diag(S_r) and V_r^T.
        Store as float16 for space efficiency.

        Returns concatenated bytes of all factor pairs.
        """
        fp32 = tensor.astype(np.float32)
        num_layers, num_heads, seq_len, head_dim = fp32.shape

        # Each slice: US has shape [seq_len, rank], Vt has shape [rank, head_dim]
        # Total per slice: rank * (seq_len + head_dim) float16 values
        all_factors = []

        for l in range(num_layers):
            for h in range(num_heads):
                mat = fp32[l, h]  # [seq_len, head_dim]
                U, S, Vt = np.linalg.svd(mat, full_matrices=False)

                # Truncate to rank
                US = U[:, :rank] * S[np.newaxis, :rank]  # [seq_len, rank]
                Vt_r = Vt[:rank, :]  # [rank, head_dim]

                all_factors.append(US.astype(np.float16).tobytes())
                all_factors.append(Vt_r.astype(np.float16).tobytes())

        return np.frombuffer(b"".join(all_factors), dtype=np.uint8)

    def _svd_decompress(self, data: bytes, shape: list[int], rank: int) -> np.ndarray:
        """Reconstruct from SVD factors."""
        num_layers, num_heads, seq_len, head_dim = shape
        result = np.zeros(shape, dtype=np.float32)

        us_size = seq_len * rank * 2  # float16 bytes
        vt_size = rank * head_dim * 2  # float16 bytes
        slice_size = us_size + vt_size

        offset = 0
        for l in range(num_layers):
            for h in range(num_heads):
                us_bytes = data[offset : offset + us_size]
                vt_bytes = data[offset + us_size : offset + slice_size]
                offset += slice_size

                US = np.frombuffer(us_bytes, dtype=np.float16).reshape(seq_len, rank)
                Vt = np.frombuffer(vt_bytes, dtype=np.float16).reshape(rank, head_dim)

                result[l, h] = (US.astype(np.float32) @ Vt.astype(np.float32))

        return result.astype(np.float16)

    @property
    def name(self) -> str:
        return "palu_lr"
