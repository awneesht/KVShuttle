"""Perplexity measurement with reconstructed KV cache."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_perplexity_from_logits(
    logits: np.ndarray,
    token_ids: np.ndarray,
) -> float:
    """Compute perplexity from pre-computed logits and target tokens.

    Args:
        logits: Model output logits, shape [seq_len, vocab_size].
        token_ids: Target token IDs, shape [seq_len].

    Returns:
        Perplexity (exp of average cross-entropy loss).
    """
    fp32_logits = logits.astype(np.float64)

    # Numerically stable log-softmax
    max_logits = np.max(fp32_logits, axis=-1, keepdims=True)
    shifted = fp32_logits - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))
    log_probs = shifted[np.arange(len(token_ids)), token_ids] - log_sum_exp

    avg_nll = -np.mean(log_probs)
    perplexity = float(np.exp(avg_nll))

    return perplexity


def compute_perplexity_delta(
    logits_original: np.ndarray,
    logits_reconstructed: np.ndarray,
    token_ids: np.ndarray,
) -> float:
    """Compute the perplexity increase from using reconstructed KV cache.

    Args:
        logits_original: Logits from original KV cache.
        logits_reconstructed: Logits from reconstructed (compressed) KV cache.
        token_ids: Target token IDs.

    Returns:
        Perplexity delta (positive = worse).
    """
    ppl_orig = compute_perplexity_from_logits(logits_original, token_ids)
    ppl_recon = compute_perplexity_from_logits(logits_reconstructed, token_ids)
    return ppl_recon - ppl_orig
