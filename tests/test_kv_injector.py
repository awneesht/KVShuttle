"""Tests for KV cache injection and forward pass evaluation."""

from __future__ import annotations

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from kvshuttle.models.kv_injector import forward_with_kv_cache  # noqa: E402


@pytest.mark.slow
class TestForwardWithKVCache:
    """Tests for forward_with_kv_cache() using Qwen2.5-3B on MLX."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-3B-Instruct-4bit")
        return model, tokenizer

    def test_logits_shape(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = "Hello, world!"

        # Extract KV cache dimensions from the model config
        config = model.args if hasattr(model, "args") else model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        seq_len = 16

        rng = np.random.default_rng(42)
        shape = (num_layers, num_kv_heads, seq_len, head_dim)
        keys = rng.standard_normal(shape).astype(np.float16)
        values = rng.standard_normal(shape).astype(np.float16)

        logits = forward_with_kv_cache(model, tokenizer, prompt, keys, values)

        # forward_with_kv_cache applies chat template, so seq_len > raw encode
        assert logits.ndim == 2
        assert logits.shape[0] > 0
        assert logits.shape[1] == config.vocab_size

    def test_logits_are_finite(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = "What is 2+2?"

        config = model.args if hasattr(model, "args") else model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = config.num_key_value_heads
        head_dim = config.hidden_size // config.num_attention_heads
        seq_len = 8

        rng = np.random.default_rng(123)
        shape = (num_layers, num_kv_heads, seq_len, head_dim)
        keys = rng.standard_normal(shape).astype(np.float16)
        values = rng.standard_normal(shape).astype(np.float16)

        logits = forward_with_kv_cache(model, tokenizer, prompt, keys, values)

        assert np.all(np.isfinite(logits)), "Logits contain NaN or inf"
