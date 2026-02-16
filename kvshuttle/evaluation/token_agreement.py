"""Token agreement rate: greedy decode match percentage."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_token_agreement(
    logits_original: np.ndarray,
    logits_reconstructed: np.ndarray,
    top_k: int = 1,
) -> float:
    """Compute token agreement rate between original and reconstructed outputs.

    Compares greedy (top-1) or top-k token predictions.

    Args:
        logits_original: Logits from model with original KV cache, shape [seq_len, vocab_size].
        logits_reconstructed: Logits with reconstructed KV cache, same shape.
        top_k: Compare top-k predictions (1 = greedy).

    Returns:
        Agreement rate in [0, 1].
    """
    if top_k == 1:
        tokens_orig = np.argmax(logits_original, axis=-1)
        tokens_recon = np.argmax(logits_reconstructed, axis=-1)
        return float(np.mean(tokens_orig == tokens_recon))

    # Top-k agreement: check if top-1 of reconstructed is in top-k of original
    top_k_orig = np.argsort(logits_original, axis=-1)[:, -top_k:]
    top_1_recon = np.argmax(logits_reconstructed, axis=-1)

    matches = 0
    for i in range(len(top_1_recon)):
        if top_1_recon[i] in top_k_orig[i]:
            matches += 1

    return matches / len(top_1_recon)
