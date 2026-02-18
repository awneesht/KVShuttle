"""Extract KV cache from PyTorch / HuggingFace models after prefill."""

from __future__ import annotations

import logging

import numpy as np
import torch

from kvshuttle.models.kv_extractor import ExtractedKVCache

logger = logging.getLogger(__name__)


def extract_kv_cache_torch(model, tokenizer, prompt: str) -> ExtractedKVCache:
    """Run prefill on a prompt and extract the KV cache.

    Args:
        model: HuggingFace model loaded via AutoModelForCausalLM.
        tokenizer: Corresponding tokenizer.
        prompt: Input text to prefill.

    Returns:
        ExtractedKVCache with numpy float16 arrays.
    """
    # Tokenize with chat template (matching MLX version)
    input_ids = _tokenize_prompt(tokenizer, prompt)

    # Determine device from model parameters
    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Run forward pass with cache
    with torch.no_grad():
        outputs = model(input_tensor, use_cache=True)

    past_kv = outputs.past_key_values

    # Extract per-layer K/V tensors (version-safe across transformers versions)
    all_keys = []
    all_values = []

    if hasattr(past_kv, "layers"):
        # transformers >= 4.54: DynamicCache with .layers API
        num_layers = len(past_kv.layers)
        for layer_idx in range(num_layers):
            k = past_kv.layers[layer_idx].keys
            v = past_kv.layers[layer_idx].values
            k_np = k[0].cpu().float().numpy().astype(np.float16)
            v_np = v[0].cpu().float().numpy().astype(np.float16)
            all_keys.append(k_np)
            all_values.append(v_np)
    elif hasattr(past_kv, "key_cache"):
        # transformers 4.36-4.53: DynamicCache with key_cache/value_cache lists
        for layer_idx in range(len(past_kv.key_cache)):
            k = past_kv.key_cache[layer_idx]
            v = past_kv.value_cache[layer_idx]
            k_np = k[0].cpu().float().numpy().astype(np.float16)
            v_np = v[0].cpu().float().numpy().astype(np.float16)
            all_keys.append(k_np)
            all_values.append(v_np)
    else:
        # Legacy tuple-of-tuples format (pre-4.36)
        for layer_idx in range(len(past_kv)):
            k = past_kv[layer_idx][0]
            v = past_kv[layer_idx][1]
            k_np = k[0].cpu().float().numpy().astype(np.float16)
            v_np = v[0].cpu().float().numpy().astype(np.float16)
            all_keys.append(k_np)
            all_values.append(v_np)

    keys = np.stack(all_keys, axis=0)    # [num_layers, num_kv_heads, seq_len, head_dim]
    values = np.stack(all_values, axis=0)

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


def _tokenize_prompt(tokenizer, prompt: str) -> list[int]:
    """Tokenize a prompt with chat template, returning a plain list of ints.

    Handles all transformers versions: apply_chat_template may return
    List[int], str, dict, or tensor depending on version and tokenizer.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        result = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        # Newer transformers may return dict/BatchEncoding {"input_ids": [...]}
        if hasattr(result, "keys") and "input_ids" in result:
            result = result["input_ids"]
        if isinstance(result, str):
            return tokenizer.encode(result)
        # Ensure plain Python ints (not numpy/torch types)
        return [int(x) for x in result]
    return tokenizer.encode(prompt)
