"""Tests for dataset loading utilities."""

from __future__ import annotations

import pytest

from kvshuttle.datasets import list_datasets, load_dataset_prompts


class TestListDatasets:
    """Tests for list_datasets()."""

    def test_returns_expected_names(self):
        names = list_datasets()
        assert names == ["c4", "gsm8k", "wikitext"]

    def test_returns_sorted(self):
        names = list_datasets()
        assert names == sorted(names)


@pytest.mark.slow
class TestLoadDatasetPrompts:
    """Tests for load_dataset_prompts() using gsm8k test split."""

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset_prompts("nonexistent_dataset")

    def test_returns_correct_count(self):
        prompts = load_dataset_prompts("gsm8k", split="test", count=5, seed=42)
        assert len(prompts) == 5

    def test_all_prompts_are_nonempty_strings(self):
        prompts = load_dataset_prompts("gsm8k", split="test", count=5, seed=42)
        for p in prompts:
            assert isinstance(p, str)
            assert len(p) > 0

    @pytest.mark.parametrize("min_tokens,max_tokens", [(128, 512)])
    def test_prompts_within_char_length_bounds(self, min_tokens, max_tokens):
        prompts = load_dataset_prompts(
            "gsm8k", split="test", count=5,
            min_tokens=min_tokens, max_tokens=max_tokens, seed=42,
        )
        min_chars = min_tokens * 4
        max_chars = max_tokens * 4
        for p in prompts:
            assert min_chars <= len(p) <= max_chars

    def test_deterministic_same_seed(self):
        p1 = load_dataset_prompts("gsm8k", split="test", count=5, seed=42)
        p2 = load_dataset_prompts("gsm8k", split="test", count=5, seed=42)
        assert p1 == p2

    def test_different_seed_different_selection(self):
        p1 = load_dataset_prompts("gsm8k", split="test", count=5, seed=42)
        p2 = load_dataset_prompts("gsm8k", split="test", count=5, seed=99)
        assert p1 != p2
