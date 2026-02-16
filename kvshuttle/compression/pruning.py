"""Token pruning — keep only the most important tokens.

Supports attention-free importance scoring (key norm based) for use in
transfer scenarios where attention weights are not available on the sender.
"""

from __future__ import annotations

import numpy as np

from kvshuttle.compression.base import BaseCompressor, CompressedKVCache
from kvshuttle.compression.registry import register


@register("topk_prune_50")
class TopKPruneCompressor(BaseCompressor):
    """Keep top-K% tokens by importance, discard the rest.

    Uses key L2 norm as attention-free importance proxy:
    tokens with larger key norms tend to receive more attention.

    Args:
        keep_ratio: Fraction of tokens to keep (0.5 = 50%).
    """

    def __init__(self, keep_ratio: float = 0.5):
        self._keep_ratio = keep_ratio

    def compress(self, keys: np.ndarray, values: np.ndarray) -> CompressedKVCache:
        original_size = keys.nbytes + values.nbytes
        num_layers, num_heads, seq_len, head_dim = keys.shape

        keep_count = max(1, int(seq_len * self._keep_ratio))

        # Compute importance: average key L2 norm across layers and heads
        key_norms = np.linalg.norm(keys.astype(np.float32), axis=3)  # [L, H, S]
        importance = key_norms.mean(axis=(0, 1))  # [S]

        # Always keep first token (BOS/system) and last few tokens (recent context)
        # Then fill remaining from highest importance
        keep_indices = _select_important_tokens(importance, keep_count, protect_ends=2)
        keep_indices = np.sort(keep_indices)  # Maintain positional order

        # Select tokens
        pruned_keys = keys[:, :, keep_indices, :]
        pruned_values = values[:, :, keep_indices, :]

        k_bytes = pruned_keys.tobytes()
        v_bytes = pruned_values.tobytes()
        idx_bytes = keep_indices.astype(np.uint32).tobytes()
        data = k_bytes + v_bytes + idx_bytes

        return CompressedKVCache(
            data=data,
            metadata={
                "key_shape": list(keys.shape),
                "val_shape": list(values.shape),
                "key_dtype": str(keys.dtype),
                "val_dtype": str(values.dtype),
                "keep_count": keep_count,
                "key_bytes_len": len(k_bytes),
                "val_bytes_len": len(v_bytes),
            },
            original_size_bytes=original_size,
            compressed_size_bytes=len(data),
            num_layers=num_layers,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    def decompress(self, compressed: CompressedKVCache) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct KV cache with pruned tokens zero-filled."""
        meta = compressed.metadata
        key_len = meta["key_bytes_len"]
        val_len = meta["val_bytes_len"]
        keep_count = meta["keep_count"]

        key_dtype = np.dtype(meta["key_dtype"])
        val_dtype = np.dtype(meta["val_dtype"])

        orig_shape_k = meta["key_shape"]
        orig_shape_v = meta["val_shape"]
        num_layers, num_heads, seq_len, head_dim = orig_shape_k

        pruned_keys = np.frombuffer(compressed.data[:key_len], dtype=key_dtype).reshape(
            num_layers, num_heads, keep_count, head_dim
        )
        pruned_values = np.frombuffer(
            compressed.data[key_len : key_len + val_len], dtype=val_dtype
        ).reshape(num_layers, num_heads, keep_count, head_dim)

        indices = np.frombuffer(compressed.data[key_len + val_len :], dtype=np.uint32)

        # Reconstruct (zero-filled for pruned positions)
        keys = np.zeros(orig_shape_k, dtype=key_dtype)
        values = np.zeros(orig_shape_v, dtype=val_dtype)
        keys[:, :, indices, :] = pruned_keys
        values[:, :, indices, :] = pruned_values

        return keys, values

    @property
    def name(self) -> str:
        return "topk_prune_50"

    @property
    def is_lossy(self) -> bool:
        return True


def _select_important_tokens(
    importance: np.ndarray, keep_count: int, protect_ends: int = 2
) -> np.ndarray:
    """Select token indices to keep based on importance scores.

    Args:
        importance: Per-token importance scores, shape [seq_len].
        keep_count: Total number of tokens to keep.
        protect_ends: Number of tokens to protect at start and end.

    Returns:
        Array of indices to keep.
    """
    seq_len = len(importance)
    if keep_count >= seq_len:
        return np.arange(seq_len)

    # Protected indices
    protected = set()
    for i in range(min(protect_ends, seq_len)):
        protected.add(i)
    for i in range(max(0, seq_len - protect_ends), seq_len):
        protected.add(i)

    protected = sorted(protected)
    remaining_budget = keep_count - len(protected)

    if remaining_budget <= 0:
        return np.array(protected[:keep_count], dtype=np.int64)

    # Select from non-protected by importance
    candidates = np.array([i for i in range(seq_len) if i not in set(protected)])
    if len(candidates) <= remaining_budget:
        return np.arange(seq_len)

    candidate_importance = importance[candidates]
    top_k_idx = np.argsort(candidate_importance)[-remaining_budget:]
    selected_candidates = candidates[top_k_idx]

    return np.array(sorted(set(protected) | set(selected_candidates)), dtype=np.int64)
