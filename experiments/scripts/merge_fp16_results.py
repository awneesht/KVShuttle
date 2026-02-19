"""Merge FP16 generation quality results from multiple Colab runs.

Sources:
  - generation_quality_fp16/results.json: T4 run (qwen2.5-3b, phi-3.5-mini real; llama synthetic)
  - generation_quality_fp16_llama/results.json: A100 run (llama-3.1-8b real)
  - generation_quality_fp16_7b/results.json: A100 run (mistral-7b, qwen2.5-7b real)

Merging strategy:
  - From T4 run: keep qwen2.5-3b and phi-3.5-mini only (llama had no ppl/token metrics)
  - From llama run: keep llama-3.1-8b
  - From 7b run: keep mistral-7b and qwen2.5-7b
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("experiments/results")

SOURCES = {
    RESULTS_DIR / "generation_quality_fp16" / "results.json": ["qwen2.5-3b", "phi-3.5-mini"],
    RESULTS_DIR / "generation_quality_fp16_llama" / "results.json": ["llama-3.1-8b"],
    RESULTS_DIR / "generation_quality_fp16_7b" / "results.json": ["mistral-7b", "qwen2.5-7b"],
}

OUTPUT = RESULTS_DIR / "generation_quality_fp16_merged" / "results.json"


def merge() -> None:
    all_results = []
    all_models = []
    all_compressors = set()

    for path, models_to_keep in SOURCES.items():
        if not path.exists():
            logger.warning("Missing: %s — skipping", path)
            continue

        with open(path) as f:
            data = json.load(f)

        kept = [r for r in data["results"] if r["model"] in models_to_keep]
        logger.info("From %s: kept %d results (%s)", path.name, len(kept), models_to_keep)
        all_results.extend(kept)
        all_models.extend(models_to_keep)
        all_compressors.update(data["metadata"]["compressors"])

    # Sort by model name, then compressor, then prompt_idx
    all_results.sort(key=lambda r: (r["model"], r["compressor"], r["prompt_idx"]))

    metadata = {
        "experiment": "generation_quality_fp16_merged",
        "backend": "torch",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_results": len(all_results),
        "models": all_models,
        "compressors": sorted(all_compressors),
        "bandwidths_gbps": [10.0],
        "prompt_source": "wikitext",
        "source_files": {str(k): v for k, v in SOURCES.items()},
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({"metadata": metadata, "results": all_results}, f, indent=2)

    logger.info("Merged %d results (%d models) → %s", len(all_results), len(all_models), OUTPUT)

    # Quick summary
    from collections import Counter
    model_counts = Counter(r["model"] for r in all_results)
    for model, count in sorted(model_counts.items()):
        has_ppl = sum(1 for r in all_results if r["model"] == model and "perplexity_delta" in r)
        logger.info("  %s: %d results (%d with perplexity)", model, count, has_ppl)


if __name__ == "__main__":
    merge()
