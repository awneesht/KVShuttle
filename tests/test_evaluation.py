"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from kvshuttle.evaluation.attention_error import compute_attention_error
from kvshuttle.evaluation.perplexity import compute_perplexity_from_logits
from kvshuttle.evaluation.token_agreement import compute_token_agreement


class TestAttentionError:
    """Tests for attention error computation."""

    def test_identical_inputs(self, small_kv_cache):
        keys, values = small_kv_cache
        result = compute_attention_error(keys, values, keys, values)

        assert result.mean_key_cosine_sim == pytest.approx(1.0)
        assert result.mean_val_cosine_sim == pytest.approx(1.0)
        assert result.mean_key_mse == pytest.approx(0.0, abs=1e-6)

    def test_noisy_inputs(self, small_kv_cache):
        keys, values = small_kv_cache
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(keys.shape).astype(np.float16) * 0.01
        keys_noisy = keys + noise

        result = compute_attention_error(keys, values, keys_noisy, values)

        assert result.mean_key_cosine_sim > 0.99
        assert result.mean_val_cosine_sim == pytest.approx(1.0)
        assert result.mean_key_mse > 0

    def test_per_layer_results(self, small_kv_cache):
        keys, values = small_kv_cache
        result = compute_attention_error(keys, values, keys, values)
        assert len(result.key_cosine_sim_per_layer) == 4
        assert len(result.val_cosine_sim_per_layer) == 4


class TestTokenAgreement:
    """Tests for token agreement metric."""

    def test_identical_logits(self):
        logits = np.random.default_rng(42).standard_normal((10, 100)).astype(np.float32)
        assert compute_token_agreement(logits, logits) == 1.0

    def test_random_logits(self):
        rng = np.random.default_rng(42)
        logits_a = rng.standard_normal((100, 1000)).astype(np.float32)
        logits_b = rng.standard_normal((100, 1000)).astype(np.float32)
        agreement = compute_token_agreement(logits_a, logits_b)
        assert 0.0 <= agreement <= 1.0

    def test_top_k_agreement(self):
        logits = np.random.default_rng(42).standard_normal((10, 100)).astype(np.float32)
        assert compute_token_agreement(logits, logits, top_k=5) == 1.0


class TestPerplexity:
    """Tests for perplexity computation."""

    def test_uniform_logits(self):
        vocab_size = 100
        seq_len = 10
        logits = np.zeros((seq_len, vocab_size), dtype=np.float32)  # uniform
        tokens = np.arange(seq_len) % vocab_size

        ppl = compute_perplexity_from_logits(logits, tokens)
        # Uniform distribution over 100 tokens should give ppl ≈ 100
        assert pytest.approx(ppl, rel=0.1) == vocab_size

    def test_confident_logits(self):
        vocab_size = 100
        seq_len = 10
        tokens = np.arange(seq_len) % vocab_size

        logits = np.zeros((seq_len, vocab_size), dtype=np.float32)
        for i in range(seq_len):
            logits[i, tokens[i]] = 100.0  # very confident

        ppl = compute_perplexity_from_logits(logits, tokens)
        assert ppl < 1.1  # near-perfect predictions
