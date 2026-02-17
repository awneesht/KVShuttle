"""Inject compressed/reconstructed KV caches into a model for forward pass evaluation."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def forward_with_kv_cache(
    model,
    tokenizer,
    prompt: str,
    keys: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Run model forward pass with injected KV cache, return logits.

    Injects pre-computed (possibly compressed/reconstructed) KV tensors into
    the model's cache, then runs a forward pass to obtain logits. This allows
    comparing generation quality between original and reconstructed caches.

    Args:
        model: MLX model loaded via mlx-lm.
        tokenizer: Corresponding tokenizer.
        prompt: Input text (used for tokenization to get continuation tokens).
        keys: Key tensors, shape [num_layers, num_kv_heads, seq_len, head_dim].
        values: Value tensors, same shape as keys.

    Returns:
        Logits as numpy array, shape [seq_len, vocab_size].
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    # Tokenize the prompt
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if isinstance(text, str):
            input_ids = tokenizer.encode(text)
        else:
            input_ids = text
    else:
        input_ids = tokenizer.encode(prompt)

    # Create fresh cache and inject KV tensors
    cache = make_prompt_cache(model)
    num_layers = keys.shape[0]

    for layer_idx in range(min(num_layers, len(cache))):
        k_layer = mx.array(keys[layer_idx])  # [num_heads, seq_len, head_dim]
        v_layer = mx.array(values[layer_idx])

        # Add batch dimension: [1, num_heads, seq_len, head_dim]
        if k_layer.ndim == 3:
            k_layer = mx.expand_dims(k_layer, axis=0)
            v_layer = mx.expand_dims(v_layer, axis=0)

        # Inject via .state setter (preferred) or direct attribute assignment
        if hasattr(cache[layer_idx], "state"):
            cache[layer_idx].state = (k_layer, v_layer)
        else:
            cache[layer_idx].keys = k_layer
            cache[layer_idx].values = v_layer
            if hasattr(cache[layer_idx], "offset"):
                cache[layer_idx].offset = keys.shape[2]  # seq_len

    mx.eval(*[c.state for c in cache if hasattr(c, "state")])

    # Run forward pass with the injected cache
    # Use the full input_ids so the model produces logits at every position
    input_tensor = mx.array([input_ids])
    mx.eval(input_tensor)

    logits = model(input_tensor, cache=cache)
    mx.eval(logits)

    # Convert to numpy: [1, seq_len, vocab_size] -> [seq_len, vocab_size]
    logits_np = np.array(logits[0])

    logger.debug(
        "Forward pass with injected KV: input_len=%d, logits_shape=%s",
        len(input_ids), logits_np.shape,
    )

    return logits_np
