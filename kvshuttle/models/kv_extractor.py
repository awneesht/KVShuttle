"""Hook into MLX models to capture KV cache after prefill."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedKVCache:
    """Raw KV cache extracted from a model after prefill.

    Attributes:
        keys: Key tensors, shape [num_layers, num_kv_heads, seq_len, head_dim].
        values: Value tensors, shape [num_layers, num_kv_heads, seq_len, head_dim].
        prompt: The prompt text used for prefill.
        prompt_tokens: Number of tokens in the prompt.
    """

    keys: np.ndarray
    values: np.ndarray
    prompt: str
    prompt_tokens: int

    @property
    def num_layers(self) -> int:
        return self.keys.shape[0]

    @property
    def num_heads(self) -> int:
        return self.keys.shape[1]

    @property
    def seq_len(self) -> int:
        return self.keys.shape[2]

    @property
    def head_dim(self) -> int:
        return self.keys.shape[3]

    @property
    def size_bytes(self) -> int:
        return self.keys.nbytes + self.values.nbytes

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def __repr__(self) -> str:
        return (
            f"ExtractedKVCache(layers={self.num_layers}, heads={self.num_heads}, "
            f"seq_len={self.seq_len}, head_dim={self.head_dim}, "
            f"size={self.size_mb:.1f} MB)"
        )


def extract_kv_cache(model, tokenizer, prompt: str) -> ExtractedKVCache:
    """Run prefill on a prompt and extract the KV cache.

    Args:
        model: MLX model loaded via mlx-lm.
        tokenizer: Corresponding tokenizer.
        prompt: Input text to prefill.

    Returns:
        ExtractedKVCache with numpy arrays.
    """
    import mlx.core as mx

    # Tokenize
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if isinstance(text, str):
            input_ids = tokenizer.encode(text)
        else:
            input_ids = text
    else:
        input_ids = tokenizer.encode(prompt)

    input_ids = mx.array([input_ids])
    mx.eval(input_ids)

    # Run prefill to populate KV cache
    cache = _run_prefill(model, input_ids)

    # Extract KV tensors from cache objects
    all_keys = []
    all_values = []

    for layer_cache in cache:
        # mlx-lm cache objects store keys and values as attributes
        k, v = _extract_from_cache_object(layer_cache)
        mx.eval(k, v)
        all_keys.append(np.array(k))
        all_values.append(np.array(v))

    keys = np.stack(all_keys, axis=0)
    values = np.stack(all_values, axis=0)

    # Ensure float16 for consistent benchmarking
    if keys.dtype != np.float16:
        keys = keys.astype(np.float16)
    if values.dtype != np.float16:
        values = values.astype(np.float16)

    seq_len = keys.shape[2]
    logger.info(
        "Extracted KV cache: %d layers, %d heads, seq_len=%d, head_dim=%d (%.1f MB)",
        keys.shape[0],
        keys.shape[1],
        seq_len,
        keys.shape[3],
        (keys.nbytes + values.nbytes) / (1024 * 1024),
    )

    return ExtractedKVCache(
        keys=keys,
        values=values,
        prompt=prompt,
        prompt_tokens=seq_len,
    )


def _run_prefill(model, input_ids):
    """Run the model forward pass to populate KV cache.

    Args:
        model: MLX model.
        input_ids: Tokenized input, shape [1, seq_len].

    Returns:
        List of cache objects, one per layer.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)

    # Run forward pass — this populates the cache
    logits = model(input_ids, cache=cache)
    mx.eval(logits)

    return cache


def _extract_from_cache_object(cache_obj) -> tuple:
    """Extract key and value tensors from an mlx-lm cache object.

    mlx-lm KVCache pre-allocates buffers (e.g., 256 tokens) but only fills up to
    `offset`. The `.state` property returns only the filled portion with correct
    seq_len, while `.keys`/`.values` return the full pre-allocated buffer.

    Args:
        cache_obj: A single layer's cache object from mlx-lm.

    Returns:
        Tuple of (keys, values) as MLX arrays with shape [num_heads, seq_len, head_dim].
    """
    if hasattr(cache_obj, "state"):
        # .state returns (keys, values) with only the filled portion
        # Shape: (batch=1, num_kv_heads, seq_len, head_dim)
        keys, values = cache_obj.state
    elif hasattr(cache_obj, "keys") and hasattr(cache_obj, "values"):
        keys = cache_obj.keys
        values = cache_obj.values
        # If offset is available, trim to actual content
        if hasattr(cache_obj, "offset"):
            offset = cache_obj.offset
            keys = keys[:, :, :offset, :]
            values = values[:, :, :offset, :]
    else:
        raise TypeError(
            f"Cannot extract KV from cache object of type {type(cache_obj)}. "
            f"Available attributes: {dir(cache_obj)}"
        )

    # Squeeze batch dimension (always 1 for our use case)
    if keys.ndim == 4 and keys.shape[0] == 1:
        keys = keys[0]
        values = values[0]

    return keys, values
