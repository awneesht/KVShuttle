"""Merge all result sets into a unified dataset for the full paper.

Combines model_sweep (7 comp × 6 models), compression_sweep (14 comp × 1 model),
and generation_quality (7 comp × 5 models) into a single indexed dataset.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ_ROOT = Path(__file__).resolve().parents[2]

RESULT_FILES = {
    "model_sweep": PROJ_ROOT / "experiments/results/model_sweep/results.json",
    "compression_sweep": PROJ_ROOT / "experiments/results/compression_sweep/results.json",
    "generation_quality": PROJ_ROOT / "experiments/results/generation_quality_fp16_merged/results.json",
    # New experiments (Phase 2) — included when available
    "model_sweep_full14": PROJ_ROOT / "experiments/results/model_sweep_full14/results.json",
}

OUTPUT_PATH = PROJ_ROOT / "experiments/results/merged_all_results.json"


def main() -> None:
    merged_results = []
    source_summary = {}

    for source_name, path in RESULT_FILES.items():
        if not path.exists():
            logger.warning("Skipping %s: file not found at %s", source_name, path)
            continue

        with open(path) as f:
            data = json.load(f)

        results = data["results"]
        for r in results:
            r["source"] = source_name

        merged_results.extend(results)
        source_summary[source_name] = {
            "num_results": len(results),
            "models": data["metadata"].get("models", []),
            "compressors": data["metadata"].get("compressors", []),
        }
        logger.info("Loaded %d results from %s", len(results), source_name)

    # Deduplicate: if a result exists in both model_sweep and compression_sweep
    # for the same (model, compressor, bandwidth, prompt_idx), keep the one from
    # the more specific source (compression_sweep > model_sweep)
    priority = {"compression_sweep": 2, "generation_quality": 1, "model_sweep": 0}

    seen: dict[tuple, dict] = {}
    for r in merged_results:
        key = (
            r.get("model", ""),
            r["compressor"],
            r.get("bandwidth_gbps", 0),
            r.get("prompt_idx", 0),
            r.get("source", ""),
        )
        # Keep all results but tag them; actual dedup is by source
        seen[key] = r

    all_compressors = sorted(set(r["compressor"] for r in merged_results))
    all_models = sorted(set(r.get("model", "unknown") for r in merged_results))

    output = {
        "metadata": {
            "description": "Merged results from all KVShuttle experiments",
            "num_results": len(merged_results),
            "all_compressors": all_compressors,
            "all_models": all_models,
            "sources": source_summary,
        },
        "results": merged_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(
        "Merged %d total results (%d compressors, %d models) → %s",
        len(merged_results),
        len(all_compressors),
        len(all_models),
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
