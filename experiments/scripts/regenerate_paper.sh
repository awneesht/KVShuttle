#!/bin/bash
# Master pipeline: after experiments complete, regenerate all paper assets
#
# Run this after:
#   1. Experiment 1 results → experiments/results/generation_quality_fp16_llama32/results.json
#   2. Experiment 2 results → experiments/results/model_sweep_full14/results.json
#   3. Experiment 3 results → experiments/results/generation_quality_fp16_missing7/results.json
#
# It's OK to run this with only some experiments done — it skips missing files.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== KVShuttle: Regenerate Paper Assets ==="
echo ""

# Step 1: Merge generation quality results (includes new experiments if available)
echo "[1/5] Merging generation quality results..."
python3 experiments/scripts/merge_fp16_results_v2.py
echo ""

# Step 2: Merge all result sets
echo "[2/5] Merging all result sets..."
python3 experiments/scripts/merge_all_results.py
echo ""

# Step 3: Compute confidence intervals
echo "[3/5] Computing confidence intervals..."
python3 experiments/scripts/compute_confidence_intervals.py
echo ""

# Step 4: Generate paper tables
echo "[4/5] Generating paper tables..."
python3 experiments/scripts/generate_paper_tables.py --full
echo ""

# Step 5: Compile paper
echo "[5/5] Compiling paper..."
PDFLATEX="/Library/TeX/texbin/pdflatex"
BIBTEX="/Library/TeX/texbin/bibtex"

if [ -x "$PDFLATEX" ]; then
    cd paper/latex
    "$PDFLATEX" -interaction=nonstopmode kvshuttle.tex > /dev/null 2>&1
    "$BIBTEX" kvshuttle > /dev/null 2>&1
    "$PDFLATEX" -interaction=nonstopmode kvshuttle.tex > /dev/null 2>&1
    "$PDFLATEX" -interaction=nonstopmode kvshuttle.tex > /dev/null 2>&1

    SIZE=$(ls -lh kvshuttle.pdf | awk '{print $5}')
    PAGES=$("$PDFLATEX" -interaction=nonstopmode kvshuttle.tex 2>&1 | grep -oE 'Output written on .* \([0-9]+ pages' | grep -oE '[0-9]+ pages' || echo "? pages")
    echo "  PDF: kvshuttle.pdf ($SIZE, $PAGES)"

    # Check for warnings
    WARNINGS=$(grep -c "Warning" kvshuttle.log 2>/dev/null; true)
    ERRORS=$(grep -c "Error" kvshuttle.log 2>/dev/null; true)
    echo "  Warnings: $WARNINGS, Errors: $ERRORS"
    cd "$PROJECT_ROOT"
else
    echo "  pdflatex not found at $PDFLATEX — skipping compilation"
fi

echo ""
echo "=== Done ==="
echo ""
echo "Generated assets:"
echo "  experiments/results/generation_quality_fp16_merged/results.json"
echo "  experiments/results/merged_all_results.json"
echo "  experiments/results/confidence_intervals.json"
echo "  paper/tables/full/*.tex"
echo "  paper/latex/kvshuttle.pdf"
