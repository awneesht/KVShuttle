"""Tests for the KVShuttle Router."""

from __future__ import annotations

import numpy as np
import pytest

from kvshuttle.router.features import RouterInput
from kvshuttle.router.evaluator import compute_regret, evaluate_router
from kvshuttle.router.learned_router import LearnedRouter


class TestRouterInput:
    """Tests for RouterInput features."""

    def test_feature_vector_shape(self):
        ri = RouterInput(
            prompt_length=256,
            model_num_layers=32,
            model_num_kv_heads=8,
            model_head_dim=128,
            kv_cache_size_bytes=100_000_000,
            available_bandwidth_gbps=10.0,
        )
        features = ri.to_feature_vector()
        assert features.shape == (len(RouterInput.feature_names()),)
        assert features.dtype == np.float32

    def test_from_kv_cache(self, small_kv_cache):
        keys, _ = small_kv_cache
        ri = RouterInput.from_kv_cache(keys, bandwidth_gbps=50.0)
        assert ri.prompt_length == keys.shape[2]
        assert ri.model_num_layers == keys.shape[0]


class TestRegret:
    """Tests for regret computation."""

    def test_zero_regret_for_oracle(self):
        oracle_ms = np.array([10.0, 20.0, 30.0])
        regret = compute_regret(oracle_ms, oracle_ms)
        np.testing.assert_allclose(regret, 0.0)

    def test_positive_regret(self):
        oracle_ms = np.array([10.0, 20.0, 30.0])
        router_ms = np.array([15.0, 25.0, 30.0])
        regret = compute_regret(router_ms, oracle_ms)
        assert np.all(regret >= 0)

    def test_evaluate_router(self):
        oracle_ms = np.array([10.0, 20.0, 30.0])
        router_ms = np.array([12.0, 22.0, 30.0])
        result = evaluate_router("test_router", router_ms, oracle_ms)
        assert result.mean_regret > 0
        assert result.oracle_match_rate < 1.0
        assert result.router_name == "test_router"


class TestLearnedRouter:
    """Tests for the learned router."""

    def test_decision_tree_training(self):
        rng = np.random.default_rng(42)
        n_samples = 100
        n_features = 11

        features = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        # Simple rule: label = 0 if feature[0] > 0, else 1
        labels = (features[:, 0] > 0).astype(int)

        router = LearnedRouter.train(
            features, labels, ["compressor_a", "compressor_b"],
            model_type="decision_tree",
        )

        # Should learn the simple rule
        test_features = np.array([[1.0] + [0.0] * 10, [-1.0] + [0.0] * 10], dtype=np.float32)
        preds = router.model.predict(test_features)
        assert preds[0] == 1  # feature[0] > 0 -> label 1
        assert preds[1] == 0  # feature[0] < 0 -> label 0

    def test_predict_single(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((50, 11)).astype(np.float32)
        labels = (features[:, 0] > 0).astype(int)

        router = LearnedRouter.train(features, labels, ["a", "b"], model_type="decision_tree")

        ri = RouterInput(
            prompt_length=256,
            model_num_layers=32,
            model_num_kv_heads=8,
            model_head_dim=128,
            kv_cache_size_bytes=100_000_000,
            available_bandwidth_gbps=10.0,
        )
        result = router.predict(ri)
        assert result in ["a", "b"]
