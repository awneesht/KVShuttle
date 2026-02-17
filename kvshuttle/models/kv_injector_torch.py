"""Inject compressed/reconstructed KV caches into PyTorch models for forward pass evaluation."""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import DynamicCache

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

    # Build DynamicCache from numpy arrays
    past = DynamicCache()
    for layer_idx in range(num_layers):
        # [num_kv_heads, prefix_len, head_dim] -> [1, num_kv_heads, prefix_len, head_dim]
        k_tensor = torch.from_numpy(keys[layer_idx]).unsqueeze(0).to(device=device, dtype=dtype)
        v_tensor = torch.from_numpy(values[layer_idx]).unsqueeze(0).to(device=device, dtype=dtype)
        past.update(k_tensor, v_tensor, layer_idx)

    # Continuation tokens
    input_tensor = torch.tensor([continuation_ids], dtype=torch.long, device=device)

    # Position IDs: continuation starts after the prefix
    position_ids = torch.arange(
        prefix_len, prefix_len + len(continuation_ids),
        dtype=torch.long, device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            input_ids=input_tensor,
            past_key_values=past,
            position_ids=position_ids,
        )

    # [1, len(continuation_ids), vocab_size] -> [len(continuation_ids), vocab_size]
    logits_np = outputs.logits[0].cpu().float().numpy()

    logger.debug(
        "Continuation forward pass: prefix_len=%d, continuation_len=%d, logits_shape=%s",
        prefix_len, len(continuation_ids), logits_np.shape,
    )

    return logits_np
