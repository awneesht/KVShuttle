"""Merge all FP16 generation quality results from multiple Colab runs.

Updated version that includes:
  - Original 5 models (from merge_fp16_results.py)
  - Experiment 1: Llama-3.2-3B (fills last model gap)
  - Experiment 3: 7 missing compressors on Llama-3.1-8B + Qwen2.5-3B
  - Experiment 4: Llama-3.1-70B (extends range to 70B)
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"

# Original sources (5 models, 7 compressors each)
ORIGINAL_SOURCES = {
    RESULTS_DIR / "generation_quality_fp16" / "results.json": ["qwen2.5-3b", "phi-3.5-mini"],
    RESULTS_DIR / "generation_quality_fp16_llama" / "results.json": ["llama-3.1-8b"],
    RESULTS_DIR / "generation_quality_fp16_7b" / "results.json": ["mistral-7b", "qwen2.5-7b"],
}

# New sources from experiments 1, 3, and 4
NEW_SOURCES = {
    # Experiment 1: Llama-3.2-3B generation quality
    RESULTS_DIR / "generation_quality_fp16_llama32" / "results.json": None,  # None = keep all models
    # Experiment 3: 7 missing compressors
    RESULTS_DIR / "generation_quality_fp16_missing7" / "results.json": None,
    # Experiment 4: Llama-3.1-70B generation quality (4-bit model, FP16 KV cache)
    RESULTS_DIR / "generation_quality_fp16_70b" / "results.json": None,
}

OUTPUT = RESULTS_DIR / "generation_quality_fp16_merged" / "results.json"


def merge() -> None:
    all_results: list[dict] = []
    all_models: list[str] = []
    all_compressors: set[str] = set()
    source_info: dict[str, dict] = {}

    # Process original sources
    for path, models_to_keep in ORIGINAL_SOURCES.items():
        if not path.exists():
            logger.warning("Missing: %s — skipping", path)
            continue

        with open(path) as f:
            data = json.load(f)

        if models_to_keep:
            kept = [r for r in data["results"] if r["model"] in models_to_keep]
        else:
            kept = data["results"]

        logger.info("From %s: kept %d results", path.name, len(kept))
        all_results.extend(kept)
        all_models.extend(models_to_keep or data["metadata"].get("models", []))
        all_compressors.update(data["metadata"]["compressors"])
        source_info[str(path)] = {
            "models": models_to_keep or data["metadata"].get("models", []),
            "num_results": len(kept),
        }

    # Process new sources
    for path, models_to_keep in NEW_SOURCES.items():
        if not path.exists():
            logger.warning("New source not yet available: %s — skipping", path)
            continue

        with open(path) as f:
            data = json.load(f)

        if models_to_keep:
            kept = [r for r in data["results"] if r["model"] in models_to_keep]
        else:
            kept = data["results"]

        # Deduplicate: don't add (model, compressor, prompt_idx) combos that already exist
        existing_keys = {
            (r["model"], r["compressor"], r.get("prompt_idx", 0))
            for r in all_results
        }
        new_kept = []
        dupes = 0
        for r in kept:
            key = (r["model"], r["compressor"], r.get("prompt_idx", 0))
            if key in existing_keys:
                dupes += 1
                continue
            new_kept.append(r)
            existing_keys.add(key)

        if dupes:
            logger.info("From %s: %d new results, %d duplicates skipped", path.name, len(new_kept), dupes)
        else:
            logger.info("From %s: %d new results", path.name, len(new_kept))

        all_results.extend(new_kept)
        new_models = set(r["model"] for r in new_kept)
        for m in new_models:
            if m not in all_models:
                all_models.append(m)
        all_compressors.update(r["compressor"] for r in new_kept)
        source_info[str(path)] = {
            "models": sorted(new_models),
            "num_results": len(new_kept),
            "duplicates_skipped": dupes,
        }

    # Sort by model, compressor, prompt_idx
    all_results.sort(key=lambda r: (r["model"], r["compressor"], r.get("prompt_idx", 0)))

    metadata = {
        "experiment": "generation_quality_fp16_merged",
        "backend": "torch",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_results": len(all_results),
        "models": sorted(set(all_models)),
        "compressors": sorted(all_compressors),
        "bandwidths_gbps": [10.0],
        "prompt_source": "wikitext",
        "source_files": source_info,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({"metadata": metadata, "results": all_results}, f, indent=2)

    logger.info("Merged %d results (%d models, %d compressors) → %s",
                len(all_results), len(metadata["models"]), len(metadata["compressors"]), OUTPUT)

    # Summary
    model_counts = Counter(r["model"] for r in all_results)
    comp_counts = Counter(r["compressor"] for r in all_results)
    print("\n=== Per-model summary ===")
    for model, count in sorted(model_counts.items()):
        has_ppl = sum(1 for r in all_results if r["model"] == model and "perplexity_delta" in r)
        comps = sorted(set(r["compressor"] for r in all_results if r["model"] == model))
        print(f"  {model}: {count} results ({has_ppl} with ppl), {len(comps)} compressors")
    print(f"\n=== Per-compressor summary ===")
    for comp, count in sorted(comp_counts.items()):
        models = sorted(set(r["model"] for r in all_results if r["compressor"] == comp))
        print(f"  {comp}: {count} results across {len(models)} models")


if __name__ == "__main__":
    merge()
