"""
Run NeuroStat pipeline on simulated EEG datasets for validation.

Processes each simulated .set file through both GEDAI and Traditional
(ASR+ICLabel) pipelines, then also computes sensor-level PLI for
direct comparison with ground truth.

Usage:
    python scripts/validation/run_validation.py [--data-dir PATH] [--output-dir PATH]

Expects simulated_data/ from generate_simulated_eeg.py to exist already.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import mne

# Add project root to path so we can import the pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pli_pipeline import (
    PipelineConfig,
    PreprocessingConfig,
    ConnectivityConfig,
    SourceConfig,
    preprocess_file,
    run_source_loc_and_connectivity,
    compute_pli,
)

# Try to import spectral_connectivity_epochs for sensor-level PLI
try:
    from mne.connectivity import spectral_connectivity_epochs
except ImportError:
    from mne_connectivity import spectral_connectivity_epochs


# ---------------------------------------------------------------------------
# Sensor-level PLI (bypasses source localization for direct comparison)
# ---------------------------------------------------------------------------

def compute_sensor_level_pli(
    raw: mne.io.BaseRaw,
    frequency_bands: dict,
    epoch_length: float = 4.0,
) -> dict:
    """Compute PLI directly on sensor-level data (no source localization).

    This provides a direct test of whether connectivity patterns survive
    preprocessing, without the confound of source localization.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed (or raw) EEG data.
    frequency_bands : dict
        Mapping of band name to (fmin, fmax) tuples.
    epoch_length : float
        Duration of each epoch in seconds.

    Returns
    -------
    pli_matrices : dict
        Mapping of band name to (n_channels, n_channels) PLI matrix.
    """
    data = raw.get_data(picks="eeg")
    sfreq = raw.info["sfreq"]
    ch_names = [ch for ch in raw.ch_names if raw.get_channel_types([ch])[0] == "eeg"]
    n_channels = len(ch_names)

    # Create epochs
    n_samples_per_epoch = int(epoch_length * sfreq)
    n_epochs = data.shape[1] // n_samples_per_epoch

    if n_epochs == 0:
        raise ValueError("Data too short for even one epoch")

    epochs_data = np.zeros((n_epochs, n_channels, n_samples_per_epoch))
    for i in range(n_epochs):
        start = i * n_samples_per_epoch
        end = start + n_samples_per_epoch
        epochs_data[i] = data[:n_channels, start:end]

    # Create EpochsArray
    info = mne.create_info(ch_names[:n_channels], sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(epochs_data, info, verbose=False)

    pli_matrices = {}
    for band_name, (fmin, fmax) in frequency_bands.items():
        try:
            pli_mat = compute_pli(epochs, (fmin, fmax), method="pli", use_gpu=False)
            pli_matrices[band_name] = pli_mat
        except Exception as exc:
            print(f"    WARNING: Sensor-level PLI failed for {band_name}: {exc}")
            pli_matrices[band_name] = None

    return pli_matrices


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def setup_validation_folder(
    set_file: Path,
    method_name: str,
    output_base: Path,
) -> tuple:
    """Set up folder structure for pipeline processing.

    The pipeline expects: base/group/session/file.set
    We create:  output_base/method_name/artifact_level/Session1/file.set

    Returns (set_copy_path, pipeline_output_dir).
    """
    artifact_level = set_file.stem
    # Create input structure: method/artifact_level/Session1/file.set
    input_dir = output_base / f"{method_name}_input" / artifact_level / "Session1"
    input_dir.mkdir(parents=True, exist_ok=True)

    set_copy = input_dir / set_file.name
    if not set_copy.exists():
        shutil.copy2(set_file, set_copy)
        # Also copy .fdt if it exists
        fdt_file = set_file.with_suffix(".fdt")
        if fdt_file.exists():
            shutil.copy2(fdt_file, input_dir / fdt_file.name)

    # Pipeline output directory
    pipeline_output = output_base / f"{method_name}_output"
    pipeline_output.mkdir(parents=True, exist_ok=True)

    return set_copy, pipeline_output


def run_pipeline_on_file(
    set_file: Path,
    method_name: str,
    output_base: Path,
    use_gedai: bool = False,
) -> dict:
    """Run the NeuroStat pipeline on a single .set file.

    Parameters
    ----------
    set_file : Path
        Path to the simulated .set file.
    method_name : str
        Either "gedai" or "traditional".
    output_base : Path
        Base output directory for validation results.
    use_gedai : bool
        If True, use GEDAI; otherwise use ASR+ICLabel.

    Returns
    -------
    result : dict
        Processing results including PLI matrices and timing.
    """
    artifact_level = set_file.stem
    print(f"\n  Processing {artifact_level} with {method_name}...")

    set_copy, pipeline_output = setup_validation_folder(
        set_file, method_name, output_base
    )

    # Configure pipeline
    preproc_config = PreprocessingConfig(
        l_freq=1.0,
        h_freq=40.0,
        notch_freqs=(50.0, 100.0),
        use_ged=use_gedai,
        ged_threshold=3.0,
        use_asr=not use_gedai,
        use_iclabel=not use_gedai,
        iclabel_brain_threshold=0.85,
    )

    config = PipelineConfig(
        preprocessing=preproc_config,
        source=SourceConfig(),
        connectivity=ConnectivityConfig(),
        output_root=pipeline_output,
    )

    result = {
        "artifact_level": artifact_level,
        "method": method_name,
        "preprocessing_ok": False,
        "connectivity_ok": False,
        "pli_matrices": {},
        "sensor_pli_matrices": {},
        "preprocessing_metrics": {},
        "timing": {},
    }

    def _progress(msg):
        print(f"    [{method_name}] {msg}")

    # Step 1: Preprocessing
    t0 = time.time()
    try:
        preproc_summary = preprocess_file(set_copy, config, _progress)
        result["preprocessing_ok"] = preproc_summary.get("preprocessed_fif") is not None
        result["preprocessing_metrics"] = preproc_summary.get("metrics", {})
        result["timing"]["preprocessing_sec"] = round(time.time() - t0, 1)
        print(f"    Preprocessing: {'OK' if result['preprocessing_ok'] else 'FAILED'} "
              f"({result['timing']['preprocessing_sec']:.1f}s)")
    except Exception as exc:
        print(f"    Preprocessing FAILED: {exc}")
        result["timing"]["preprocessing_sec"] = round(time.time() - t0, 1)
        return result

    if not result["preprocessing_ok"]:
        return result

    fif_path = Path(preproc_summary["preprocessed_fif"])

    # Step 1b: Compute sensor-level PLI on preprocessed data
    t1 = time.time()
    try:
        raw_preprocessed = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
        sensor_pli = compute_sensor_level_pli(
            raw_preprocessed,
            config.connectivity.frequency_bands,
            config.connectivity.epoch_length,
        )
        result["sensor_pli_matrices"] = {
            band: mat.tolist() if mat is not None else None
            for band, mat in sensor_pli.items()
        }
        result["timing"]["sensor_pli_sec"] = round(time.time() - t1, 1)
        print(f"    Sensor-level PLI: OK ({result['timing']['sensor_pli_sec']:.1f}s)")
    except Exception as exc:
        print(f"    Sensor-level PLI FAILED: {exc}")
        result["timing"]["sensor_pli_sec"] = round(time.time() - t1, 1)

    # Step 2: Source localization + connectivity
    t2 = time.time()
    try:
        conn_summary = run_source_loc_and_connectivity(
            fif_path, set_copy, config, _progress
        )
        result["connectivity_ok"] = bool(conn_summary.get("pli_outputs"))
        result["timing"]["connectivity_sec"] = round(time.time() - t2, 1)

        # Load the PLI CSVs that were saved
        for band_name, csv_path_str in conn_summary.get("pli_outputs", {}).items():
            csv_path = Path(csv_path_str)
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0)
                result["pli_matrices"][band_name] = df.values.tolist()

        print(f"    Source + Connectivity: {'OK' if result['connectivity_ok'] else 'FAILED'} "
              f"({result['timing']['connectivity_sec']:.1f}s)")
    except Exception as exc:
        print(f"    Source + Connectivity FAILED: {exc}")
        result["timing"]["connectivity_sec"] = round(time.time() - t2, 1)

    total = sum(v for v in result["timing"].values() if isinstance(v, (int, float)))
    result["timing"]["total_sec"] = round(total, 1)

    return result


# ---------------------------------------------------------------------------
# Sensor-level PLI on raw (unprocessed) data for baseline
# ---------------------------------------------------------------------------

def compute_raw_sensor_pli(set_file: Path, frequency_bands: dict) -> dict:
    """Compute sensor-level PLI on raw (unprocessed) data.

    This establishes the baseline: what PLI looks like before any
    preprocessing.
    """
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    return compute_sensor_level_pli(raw, frequency_bands, epoch_length=4.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
    """Run validation pipeline on all simulated datasets.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing simulated .set files and ground_truth.json.
        Defaults to scripts/validation/simulated_data/.
    output_dir : Path, optional
        Output directory for validation results.
        Defaults to scripts/validation/validation_results/.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "simulated_data"
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "validation_results"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt_path = data_dir / "ground_truth.json"
    if not gt_path.exists():
        print(f"ERROR: Ground truth not found at {gt_path}")
        print("Run generate_simulated_eeg.py first.")
        sys.exit(1)

    with open(gt_path) as f:
        ground_truth = json.load(f)

    frequency_bands = {k: tuple(v) for k, v in ground_truth["frequency_bands"].items()}

    print("=" * 60)
    print("NeuroStat Validation: Pipeline Processing")
    print("=" * 60)
    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Datasets:         {len(ground_truth['datasets'])}")

    # Collect results for all datasets and methods
    all_results = []
    methods = [
        ("traditional", False),
        ("gedai", True),
    ]

    for dataset_info in ground_truth["datasets"]:
        filename = dataset_info["filename"]
        set_file = data_dir / filename
        artifact_level = dataset_info["artifact_level"]

        if not set_file.exists():
            print(f"\nWARNING: {set_file} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {filename} ({dataset_info['label']})")
        print(f"{'='*60}")

        # Compute raw baseline PLI
        print("  Computing raw (unprocessed) sensor-level PLI...")
        try:
            raw_pli = compute_raw_sensor_pli(set_file, frequency_bands)
            raw_pli_serializable = {
                band: mat.tolist() if mat is not None else None
                for band, mat in raw_pli.items()
            }
        except Exception as exc:
            print(f"  WARNING: Raw PLI failed: {exc}")
            raw_pli_serializable = {}

        # Run each pipeline method
        for method_name, use_gedai in methods:
            try:
                result = run_pipeline_on_file(
                    set_file, method_name, output_dir, use_gedai
                )
                result["raw_sensor_pli"] = raw_pli_serializable
                all_results.append(result)
            except Exception as exc:
                print(f"  ERROR: {method_name} pipeline failed for {artifact_level}: {exc}")
                all_results.append({
                    "artifact_level": artifact_level,
                    "method": method_name,
                    "preprocessing_ok": False,
                    "connectivity_ok": False,
                    "error": str(exc),
                })

    # Save all results
    results_path = output_dir / "validation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "ground_truth": ground_truth,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("VALIDATION PROCESSING SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<30s} {'Method':<15s} {'Preproc':<10s} {'Connect':<10s} {'Time (s)':<10s}")
    print("-" * 75)
    for r in all_results:
        preproc = "OK" if r.get("preprocessing_ok") else "FAIL"
        connect = "OK" if r.get("connectivity_ok") else "FAIL"
        timing = r.get("timing", {}).get("total_sec", "N/A")
        print(f"{r['artifact_level']:<30s} {r['method']:<15s} {preproc:<10s} {connect:<10s} {timing}")

    print(f"\nFull results: {results_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run NeuroStat pipeline on simulated EEG for validation."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
