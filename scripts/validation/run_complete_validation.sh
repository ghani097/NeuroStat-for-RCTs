#!/bin/bash
# Master validation script - runs everything in sequence
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=========================================="
echo "NeuroStat Validation Pipeline"
echo "=========================================="

echo ""
echo "Step 1: Generating simulated EEG data..."
python scripts/validation/generate_simulated_eeg.py

echo ""
echo "Step 2: Processing through NeuroStat (GEDAI + Traditional)..."
python scripts/validation/run_neurostat_batch.py

echo ""
echo "Step 3: Analyzing results and generating figures..."
python scripts/validation/analyze_validation_results.py

echo ""
echo "=========================================="
echo "Validation Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - validation_data/results/validation_results.csv"
echo "  - validation_data/results/validation_scatter.png"
echo "  - validation_data/results/method_comparison.png"
echo "  - validation_data/results/pattern_recovery.png"
