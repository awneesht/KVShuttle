"""Shared test fixtures for KVShuttle tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def small_kv_cache() -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic KV cache for fast tests.

    Returns:
        Tuple of (keys, values) with shape [4, 2, 32, 64] (4 layers, 2 heads, 32 tokens, dim 64).
    """
    rng = np.random.default_rng(42)
    shape = (4, 2, 32, 64)
    keys = rng.standard_normal(shape).astype(np.float16)
    values = rng.standard_normal(shape).astype(np.float16)
    return keys, values


@pytest.fixture
def medium_kv_cache() -> tuple[np.ndarray, np.ndarray]:
    """Medium synthetic KV cache mimicking a small model.

    Returns:
        Tuple of (keys, values) with shape [24, 8, 256, 128]
        (24 layers, 8 heads, 256 tokens, dim 128).
    """
    rng = np.random.default_rng(123)
    shape = (24, 8, 256, 128)
    keys = rng.standard_normal(shape).astype(np.float16)
    values = rng.standard_normal(shape).astype(np.float16)
    return keys, values


@pytest.fixture
def realistic_kv_cache() -> tuple[np.ndarray, np.ndarray]:
    """KV cache with realistic value distributions.

    Uses a distribution closer to real transformer KV caches:
    keys tend to have some outlier channels, values are smoother.

    Returns:
        Tuple of (keys, values) with shape [4, 4, 128, 64].
    """
    rng = np.random.default_rng(999)
    shape = (4, 4, 128, 64)

    # Keys: some channels have larger magnitudes (outliers)
    keys = rng.standard_normal(shape).astype(np.float32)
    outlier_channels = rng.choice(64, size=4, replace=False)
    keys[:, :, :, outlier_channels] *= 5.0
    keys = keys.astype(np.float16)

    # Values: smoother distribution
    values = (rng.standard_normal(shape) * 0.5).astype(np.float16)

    return keys, values
