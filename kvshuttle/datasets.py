"""Load real text prompts from HuggingFace datasets for realistic KV cache benchmarks."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Supported dataset configurations
_DATASET_CONFIGS = {
    "wikitext": {
        "path": "wikitext",
        "name": "wikitext-103-v1",
        "text_field": "text",
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "text_field": "question",
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "text_field": "text",
        "streaming": True,
    },
}


def load_dataset_prompts(
    dataset_name: str,
    split: str = "test",
    count: int = 30,
    min_tokens: int = 128,
    max_tokens: int = 512,
    seed: int = 42,
) -> list[str]:
    """Load real text prompts from HuggingFace datasets.

    Supported datasets:
    - "wikitext": WikiText-103 (long-form prose)
    - "gsm8k": Grade school math (reasoning chains)
    - "c4": Common Crawl web text (diverse domains)

    Prompts are filtered by estimated token count (4 chars/token heuristic)
    and deterministically sampled for reproducibility.

    Args:
        dataset_name: One of "wikitext", "gsm8k", "c4".
        split: Dataset split to use (e.g., "test", "train").
        count: Number of prompts to return.
        min_tokens: Minimum estimated tokens per prompt.
        max_tokens: Maximum estimated tokens per prompt.
        seed: Random seed for deterministic sampling.

    Returns:
        List of text strings suitable for model prefill.

    Raises:
        ValueError: If dataset_name is not supported.
        ImportError: If the datasets library is not installed.
    """
    if dataset_name not in _DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {list(_DATASET_CONFIGS.keys())}"
        )

    import numpy as np
    from datasets import load_dataset

    cfg = _DATASET_CONFIGS[dataset_name]
    text_field = cfg["text_field"]

    # Rough chars-per-token estimate for filtering before tokenization
    min_chars = min_tokens * 4
    max_chars = max_tokens * 4

    logger.info(
        "Loading %s dataset (split=%s), targeting %d prompts with %d-%d tokens",
        dataset_name, split, count, min_tokens, max_tokens,
    )

    # Handle GSM8K which only has train/test splits
    if dataset_name == "gsm8k" and split == "validation":
        split = "test"

    # Load dataset
    if cfg.get("streaming"):
        ds = load_dataset(cfg["path"], cfg["name"], split=split, streaming=True)
        # For streaming, collect candidates in one pass
        candidates = []
        for example in ds:
            text = example[text_field].strip()
            if min_chars <= len(text) <= max_chars and len(text) > 0:
                candidates.append(text)
                if len(candidates) >= count * 10:
                    break
    else:
        ds = load_dataset(cfg["path"], cfg.get("name"), split=split)
        candidates = [
            row[text_field].strip()
            for row in ds
            if min_chars <= len(row[text_field].strip()) <= max_chars
            and len(row[text_field].strip()) > 0
        ]

    if len(candidates) == 0:
        raise ValueError(
            f"No prompts found in {dataset_name}/{split} with "
            f"{min_tokens}-{max_tokens} estimated tokens. "
            f"Try adjusting token range."
        )

    # Deterministic sampling
    rng = np.random.default_rng(seed)
    if len(candidates) > count:
        indices = rng.choice(len(candidates), size=count, replace=False)
        indices.sort()
        prompts = [candidates[i] for i in indices]
    else:
        prompts = candidates[:count]
        logger.warning(
            "Only %d prompts available (requested %d) from %s/%s",
            len(prompts), count, dataset_name, split,
        )

    logger.info(
        "Loaded %d prompts from %s (avg length: %.0f chars ≈ %.0f tokens)",
        len(prompts), dataset_name,
        np.mean([len(p) for p in prompts]),
        np.mean([len(p) / 4 for p in prompts]),
    )

    return prompts


def list_datasets() -> list[str]:
    """Return list of supported dataset names."""
    return sorted(_DATASET_CONFIGS.keys())
