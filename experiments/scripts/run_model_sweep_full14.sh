#!/bin/bash
# Experiment 2: Full 14-compressor sweep on all 6 models
# CPU-only, no GPU required. Uses synthetic KV caches.
# Expected runtime: ~4-6 hours locally
# Expected output: ~10,080 results (14 comp × 6 models × 4 BW × 30 prompts)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== KVShuttle: Full 14-Compressor Model Sweep ==="
echo "Project root: $PROJECT_ROOT"
echo "Config: experiments/configs/model_sweep_full14.yaml"
echo "Output: experiments/results/model_sweep_full14/"
echo ""
echo "This will take ~4-6 hours. Press Ctrl+C to cancel."
echo ""

cd "$PROJECT_ROOT"

python3 -m experiments.scripts.run_experiment \
    experiments/configs/model_sweep_full14.yaml

echo ""
echo "=== Experiment complete ==="
echo "Results: experiments/results/model_sweep_full14/results.json"
echo ""
echo "Next steps:"
echo "  1. python3 experiments/scripts/merge_all_results.py"
echo "  2. python3 experiments/scripts/compute_confidence_intervals.py"
echo "  3. python3 experiments/scripts/generate_paper_tables.py --full"
