#!/usr/bin/env python
"""
Master validation script -- runs everything in sequence.

Steps
-----
1. Generate simulated EEG data with known connectivity
2. Process through NeuroStat (GEDAI + Traditional)
3. Analyse results and generate figures

Usage
-----
    python scripts/validation/run_complete_validation.py
    python scripts/validation/run_complete_validation.py --skip-generate
    python scripts/validation/run_complete_validation.py --skip-pipeline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BASE_DIR    = Path(__file__).resolve().parent
SIM_DIR     = BASE_DIR / "simulated_data"
VAL_DIR     = BASE_DIR / "validation_data"
RESULTS_DIR = VAL_DIR / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete NeuroStat validation pipeline.")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip data generation (reuse existing simulated data)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip pipeline processing (reuse existing PLI outputs)")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip analysis and figure generation")
    args = parser.parse_args()

    wall_start = time.time()

    print()
    print("=" * 60)
    print("  NeuroStat Validation Pipeline")
    print("=" * 60)

    # -- Step 1: Generate ---------------------------------------------
    if not args.skip_generate:
        print()
        print("-" * 60)
        print("  Step 1/3: Generating simulated EEG data …")
        print("-" * 60)
        from generate_simulated_eeg import main as gen_main
        gen_main(SIM_DIR)
    else:
        print("\n  [SKIP] Step 1 -- using existing simulated data")

    # -- Step 2: Pipeline ---------------------------------------------
    if not args.skip_pipeline:
        print()
        print("-" * 60)
        print("  Step 2/3: Processing through NeuroStat (GEDAI + Traditional) …")
        print("-" * 60)
        from run_neurostat_batch import main as batch_main
        batch_main(SIM_DIR, VAL_DIR)
    else:
        print("\n  [SKIP] Step 2 -- using existing pipeline outputs")

    # -- Step 3: Analysis ---------------------------------------------
    if not args.skip_analysis:
        print()
        print("-" * 60)
        print("  Step 3/3: Analysing results and generating figures …")
        print("-" * 60)
        from analyze_validation_results import main as analysis_main
        analysis_main(VAL_DIR)
    else:
        print("\n  [SKIP] Step 3 -- analysis")

    # -- Done ---------------------------------------------------------
    wall_min = (time.time() - wall_start) / 60
    print()
    print("=" * 60)
    print(f"  Validation Complete!  ({wall_min:.1f} min)")
    print("=" * 60)
    print()
    print("  Results saved to:")
    print(f"    - {RESULTS_DIR / 'validation_results.csv'}")
    print(f"    - {RESULTS_DIR / 'validation_scatter.png'}")
    print(f"    - {RESULTS_DIR / 'method_comparison.png'}")
    print(f"    - {RESULTS_DIR / 'pattern_recovery.png'}")
    print()


if __name__ == "__main__":
    main()
