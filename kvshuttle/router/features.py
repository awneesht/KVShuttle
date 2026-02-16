"""Feature extraction for the KVShuttle Router."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RouterInput:
    """Input features for the KVShuttle Router.

    Attributes:
        prompt_length: Number of tokens in the prompt.
        model_num_layers: Number of transformer layers.
        model_num_kv_heads: Number of KV attention heads.
        model_head_dim: Dimension per attention head.
        kv_cache_size_bytes: Total size of KV cache in bytes.
        available_bandwidth_gbps: Current network bandwidth in Gbps.
        quality_threshold: Minimum acceptable quality (e.g., 0.99 cosine sim).
    """

    prompt_length: int
    model_num_layers: int
    model_num_kv_heads: int
    model_head_dim: int
    kv_cache_size_bytes: int
    available_bandwidth_gbps: float
    quality_threshold: float = 0.99

    def to_feature_vector(self) -> np.ndarray:
        """Convert to numpy feature vector for ML models."""
        return np.array([
            self.prompt_length,
            self.model_num_layers,
            self.model_num_kv_heads,
            self.model_head_dim,
            self.kv_cache_size_bytes,
            self.available_bandwidth_gbps,
            self.quality_threshold,
            # Derived features
            np.log2(self.prompt_length + 1),
            np.log2(self.kv_cache_size_bytes + 1),
            np.log2(self.available_bandwidth_gbps + 0.01),
            self.kv_cache_size_bytes / (self.available_bandwidth_gbps * 1e9 / 8 + 1e-9),  # raw xfer s
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "prompt_length",
            "model_num_layers",
            "model_num_kv_heads",
            "model_head_dim",
            "kv_cache_size_bytes",
            "available_bandwidth_gbps",
            "quality_threshold",
            "log2_prompt_length",
            "log2_kv_cache_bytes",
            "log2_bandwidth",
            "raw_transfer_seconds",
        ]

    @classmethod
    def from_kv_cache(
        cls,
        keys: np.ndarray,
        bandwidth_gbps: float,
        quality_threshold: float = 0.99,
    ) -> RouterInput:
        """Create RouterInput from a KV cache array."""
        num_layers, num_heads, seq_len, head_dim = keys.shape
        kv_size = keys.nbytes * 2  # keys + values
        return cls(
            prompt_length=seq_len,
            model_num_layers=num_layers,
            model_num_kv_heads=num_heads,
            model_head_dim=head_dim,
            kv_cache_size_bytes=kv_size,
            available_bandwidth_gbps=bandwidth_gbps,
            quality_threshold=quality_threshold,
        )
