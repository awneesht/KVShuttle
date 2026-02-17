"""Inject compressed/reconstructed KV caches into PyTorch models for forward pass evaluation."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def forward_continuation_with_kv_cache_torch(
    model,
    keys: np.ndarray,
    values: np.ndarray,
    continuation_ids: list[int],
) -> np.ndarray:
    """Run forward pass on continuation tokens with injected prefix KV cache.

    Injects pre-computed (possibly compressed/reconstructed) KV tensors as a
    DynamicCache, then runs a forward pass on only the continuation tokens.
    This mirrors how disaggregated serving works: prefix KV is transferred,
    then the decode node continues generation.

    Args:
        model: HuggingFace model loaded via AutoModelForCausalLM.
        keys: Prefix key tensors, shape [num_layers, num_kv_heads, prefix_len, head_dim].
        values: Prefix value tensors, same shape as keys.
        continuation_ids: Token IDs for the continuation (after the prefix).

    Returns:
        Logits as numpy array, shape [len(continuation_ids), vocab_size].
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_layers = keys.shape[0]
    prefix_len = keys.shape[2]
    cont_len = len(continuation_ids)

    # Build past_key_values from numpy arrays
    past = _build_kv_cache(keys, values, num_layers, device, dtype)

    # Continuation tokens
    input_tensor = torch.tensor([continuation_ids], dtype=torch.long, device=device)

    # Attention mask must cover prefix (cached) + continuation (new) tokens
    attention_mask = torch.ones(1, prefix_len + cont_len, dtype=torch.long, device=device)

    # cache_position: absolute positions of the new tokens being processed
    cache_position = torch.arange(
        prefix_len, prefix_len + cont_len, dtype=torch.long, device=device
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            past_key_values=past,
            cache_position=cache_position,
            use_cache=False,
        )

    # [1, len(continuation_ids), vocab_size] -> [len(continuation_ids), vocab_size]
    logits_np = outputs.logits[0].cpu().float().numpy()

    logger.debug(
        "Continuation forward pass: prefix_len=%d, continuation_len=%d, logits_shape=%s",
        prefix_len, cont_len, logits_np.shape,
    )

    return logits_np


def _build_kv_cache(keys, values, num_layers, device, dtype):
    """Build HF-compatible KV cache, with DynamicCache → tuple fallback."""
    kv_pairs = []
    for layer_idx in range(num_layers):
        # [num_kv_heads, prefix_len, head_dim] -> [1, num_kv_heads, prefix_len, head_dim]
        k = torch.from_numpy(keys[layer_idx]).unsqueeze(0).to(device=device, dtype=dtype)
        v = torch.from_numpy(values[layer_idx]).unsqueeze(0).to(device=device, dtype=dtype)
        kv_pairs.append((k, v))

    try:
        from transformers import DynamicCache

        past = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv_pairs):
            past.update(k, v, layer_idx)
        return past
    except (ImportError, Exception) as e:
        logger.debug("DynamicCache unavailable (%s), using tuple format", e)
        return tuple(kv_pairs)
