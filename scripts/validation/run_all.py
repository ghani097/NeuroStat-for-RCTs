"""
Run the complete NeuroStat validation pipeline.

Steps:
    1. Generate simulated EEG datasets with known connectivity
    2. Process through both GEDAI and Traditional pipelines
    3. Analyze results and generate manuscript figures/tables

Usage:
    python scripts/validation/run_all.py [--skip-generate] [--skip-pipeline] [--skip-analysis]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Run complete NeuroStat validation pipeline."
    )
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip data generation (use existing simulated data)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip pipeline processing (use existing results)")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip analysis and figure generation")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Base output directory")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "simulated_data"
    results_dir = base_dir / "validation_results"

    total_start = time.time()

    # Step 1: Generate simulated data
    if not args.skip_generate:
        print("\n" + "=" * 70)
        print("STEP 1/3: Generating simulated EEG datasets")
        print("=" * 70)
        from generate_simulated_eeg import main as generate_main
        generate_main(data_dir)
    else:
        print("\n[SKIP] Step 1: Using existing simulated data")

    # Step 2: Run pipeline
    if not args.skip_pipeline:
        print("\n" + "=" * 70)
        print("STEP 2/3: Processing through NeuroStat pipeline")
        print("=" * 70)
        from run_validation import main as run_main
        run_main(data_dir, results_dir)
    else:
        print("\n[SKIP] Step 2: Using existing pipeline results")

    # Step 3: Analyze results
    if not args.skip_analysis:
        print("\n" + "=" * 70)
        print("STEP 3/3: Analyzing results and generating figures")
        print("=" * 70)
        from analyze_validation import main as analyze_main
        analyze_main(results_dir)
    else:
        print("\n[SKIP] Step 3: Analysis")

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"VALIDATION COMPLETE ({total_time / 60:.1f} minutes)")
    print(f"{'=' * 70}")
    print(f"\nOutputs:")
    print(f"  Simulated data: {data_dir}")
    print(f"  Pipeline results: {results_dir}")
    print(f"  Figures/tables: {results_dir / 'figures'}")


if __name__ == "__main__":
    main()
