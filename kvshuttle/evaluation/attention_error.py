"""Attention error metrics: MSE and cosine similarity vs uncompressed KV cache."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttentionErrorResult:
    """Per-layer and aggregate attention error metrics.

    Attributes:
        key_cosine_sim_per_layer: Cosine similarity between original and reconstructed keys.
        val_cosine_sim_per_layer: Same for values.
        key_mse_per_layer: MSE per layer for keys.
        val_mse_per_layer: MSE per layer for values.
        mean_key_cosine_sim: Average cosine similarity across layers (keys).
        mean_val_cosine_sim: Average cosine similarity across layers (values).
        mean_key_mse: Average MSE across layers (keys).
        mean_val_mse: Average MSE across layers (values).
    """

    key_cosine_sim_per_layer: list[float]
    val_cosine_sim_per_layer: list[float]
    key_mse_per_layer: list[float]
    val_mse_per_layer: list[float]
    mean_key_cosine_sim: float
    mean_val_cosine_sim: float
    mean_key_mse: float
    mean_val_mse: float


def compute_attention_error(
    keys_orig: np.ndarray,
    values_orig: np.ndarray,
    keys_recon: np.ndarray,
    values_recon: np.ndarray,
) -> AttentionErrorResult:
    """Compute per-layer attention error metrics.

    Args:
        keys_orig: Original keys, shape [num_layers, num_heads, seq_len, head_dim].
        values_orig: Original values, same shape.
        keys_recon: Reconstructed keys after compression round-trip.
        values_recon: Reconstructed values.

    Returns:
        AttentionErrorResult with per-layer and aggregate metrics.
    """
    num_layers = keys_orig.shape[0]

    k_cos = []
    v_cos = []
    k_mse = []
    v_mse = []

    for i in range(num_layers):
        k_orig_flat = keys_orig[i].astype(np.float32).reshape(-1)
        k_recon_flat = keys_recon[i].astype(np.float32).reshape(-1)
        v_orig_flat = values_orig[i].astype(np.float32).reshape(-1)
        v_recon_flat = values_recon[i].astype(np.float32).reshape(-1)

        k_cos.append(_cosine_similarity(k_orig_flat, k_recon_flat))
        v_cos.append(_cosine_similarity(v_orig_flat, v_recon_flat))
        k_mse.append(float(np.mean((k_orig_flat - k_recon_flat) ** 2)))
        v_mse.append(float(np.mean((v_orig_flat - v_recon_flat) ** 2)))

    return AttentionErrorResult(
        key_cosine_sim_per_layer=k_cos,
        val_cosine_sim_per_layer=v_cos,
        key_mse_per_layer=k_mse,
        val_mse_per_layer=v_mse,
        mean_key_cosine_sim=float(np.mean(k_cos)),
        mean_val_cosine_sim=float(np.mean(v_cos)),
        mean_key_mse=float(np.mean(k_mse)),
        mean_val_mse=float(np.mean(v_mse)),
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two flat vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0 if (norm_a == 0 and norm_b == 0) else 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
