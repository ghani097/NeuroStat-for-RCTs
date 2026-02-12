"""
Batch-process all simulated EEG datasets through the NeuroStat pipeline.

Runs each of the 4 artifact-severity datasets through both GEDAI and
Traditional (ASR + ICLabel) preprocessing, followed by source localization
and PLI connectivity computation with 50 % overlapping epochs.

Output structure
----------------
validation_data/
├-- neurostat_outputs/
│   ├-- clean_high_snr/
│   │   ├-- GEDAI/
│   │   │   ├-- delta_PLI.csv          # 68×68 source-level PLI
│   │   │   ├-- theta_PLI.csv
│   │   │   ├-- alpha_PLI.csv
│   │   │   ├-- beta_PLI.csv
│   │   │   ├-- gamma_PLI.csv
│   │   │   ├-- sensor_alpha_PLI.csv   # 64×64 sensor-level PLI
│   │   │   ├-- sensor_beta_PLI.csv
│   │   │   ├-- sensor_theta_PLI.csv
│   │   │   ├-- preprocessing_metrics.json
│   │   │   └-- preprocessing_info.json
│   │   └-- Traditional/
│   │       └-- ...  (same layout)
│   ├-- light_medium_snr/  ...
│   ├-- moderate_medium_snr/  ...
│   └-- heavy_low_snr/  ...
└-- raw_sensor_pli/
    ├-- clean_high_snr/
    │   ├-- sensor_alpha_PLI.csv
    │   └-- ...
    └-- ...

Usage
-----
    python scripts/validation/run_neurostat_batch.py [--sim-dir PATH] [--out-dir PATH]

Requires
--------
* Simulated .set files produced by ``generate_simulated_eeg.py``
* MNE-Python with fsaverage (auto-downloaded on first run)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import mne
from mne import EpochsArray
from mne.channels import make_standard_montage

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pli_pipeline import (                       # noqa: E402
    PipelineConfig,
    PreprocessingConfig,
    ConnectivityConfig,
    SourceConfig,
    compute_pli,
    preprocess_raw,
    ensure_subjects_dir,
)
from pli_pipeline import (                       # noqa: E402
    _prepare_inverse_operator,
    _run_source_localization,
    _apply_montage,
    _save_connectivity_outputs,
)

LOGGER = logging.getLogger("validation_batch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

# Suppress noisy MNE INFO messages (keep warnings)
mne.set_log_level("WARNING")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

METHODS = [
    ("GEDAI",       True),   # (label, use_gedai)
    ("Traditional", False),
]

BIOSEMI64_MONTAGE = make_standard_montage("biosemi64")


def _load_set(set_path: Path, preload: bool = True) -> mne.io.Raw:
    """Load an EEGLAB .set file and re-apply the biosemi64 montage.

    Re-applying the standard montage after load silences the 'head radius
    above 99th percentile' warning that occurs when channel positions
    stored inside the .set file have slightly different scaling.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*head radius.*", category=RuntimeWarning)
        raw = mne.io.read_raw_eeglab(str(set_path), preload=preload, verbose=False)
    raw.set_montage(BIOSEMI64_MONTAGE, on_missing="ignore", verbose=False)
    return raw

EPOCH_LENGTH  = 4.0   # seconds
EPOCH_OVERLAP = 0.50   # 50 % overlap


# ---------------------------------------------------------------------------
# Overlapping epoch creation
# ---------------------------------------------------------------------------
def _make_overlapping_epochs(
    data_2d: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    epoch_length: float = EPOCH_LENGTH,
    overlap: float = EPOCH_OVERLAP,
    ch_type: str = "misc",
) -> EpochsArray:
    """Segment a (n_channels, n_samples) array into overlapping epochs.

    Parameters
    ----------
    data_2d : ndarray, shape (n_channels, n_samples)
    sfreq   : float
    ch_names: list of str
    epoch_length : float – seconds per epoch.
    overlap : float – fraction of overlap, e.g. 0.50 for 50 %.
    ch_type : str – MNE channel type label.

    Returns
    -------
    EpochsArray  – shape (n_epochs, n_channels, n_samples_per_epoch)
    """
    n_ch, n_total = data_2d.shape
    spe  = int(epoch_length * sfreq)          # samples per epoch
    step = int(spe * (1.0 - overlap))         # hop size
    if step < 1:
        step = 1

    starts = list(range(0, n_total - spe + 1, step))
    if not starts:
        raise ValueError("Data too short for even one epoch")

    epoch_data = np.empty((len(starts), n_ch, spe), dtype=data_2d.dtype)
    for idx, s in enumerate(starts):
        epoch_data[idx] = data_2d[:, s : s + spe]

    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types=ch_type)
    return EpochsArray(epoch_data, info, verbose=False)


# ---------------------------------------------------------------------------
# Sensor-level PLI (bypasses source localisation)
# ---------------------------------------------------------------------------
def _compute_sensor_pli(
    raw: mne.io.BaseRaw,
    bands: Dict[str, Tuple[float, float]] = FREQUENCY_BANDS,
) -> Dict[str, np.ndarray]:
    """Return {band_name: (n_eeg, n_eeg) PLI matrix} from raw EEG."""
    picks = mne.pick_types(raw.info, eeg=True, meg=False)
    data  = raw.get_data(picks=picks)
    ch_names = [raw.ch_names[p] for p in picks]
    epochs = _make_overlapping_epochs(data, raw.info["sfreq"], ch_names, ch_type="eeg")

    result = {}
    for bname, band in bands.items():
        try:
            result[bname] = compute_pli(epochs, band, method="pli", use_gpu=False)
        except Exception as exc:
            LOGGER.warning("Sensor PLI (%s) failed: %s", bname, exc)
            result[bname] = None
    return result


# ---------------------------------------------------------------------------
# Single-dataset processing
# ---------------------------------------------------------------------------
def _build_config(use_gedai: bool) -> Tuple[PreprocessingConfig, SourceConfig, ConnectivityConfig]:
    """Build a deterministic config matching the paper parameters."""
    preproc = PreprocessingConfig(
        l_freq=1.0,
        h_freq=40.0,
        notch_freqs=(50.0, 100.0),
        ica_method="fastica",
        ica_n_components=25,
        random_state=42,
        use_ged=use_gedai,
        ged_threshold=3.0,
        use_asr=not use_gedai,
        use_iclabel=not use_gedai,
        # ICLabel was trained on real EEG; for synthetic data with simpler
        # spectral structure, a lower threshold retains more signal-carrying
        # components.  0.50 = "more likely brain than not".
        iclabel_brain_threshold=0.50,
        bad_channel_threshold=4.0,
        interpolate_bad_channels=True,
        ica_reject_threshold=200e-6,
    )
    source  = SourceConfig()
    connect = ConnectivityConfig(
        epoch_length=EPOCH_LENGTH,
        frequency_bands=FREQUENCY_BANDS,
        method="pli",
        use_gpu=False,
    )
    return preproc, source, connect


def process_single(
    set_path: Path,
    method_label: str,
    use_gedai: bool,
    out_dir: Path,
) -> dict:
    """Run the full NeuroStat pipeline on one .set file.

    Steps
    -----
    1.  Load raw EEGLAB .set
    2.  Preprocess (GEDAI *or* Traditional ASR + ICLabel)
    3.  Compute sensor-level PLI (50 % overlap) → save CSVs
    4.  Source localisation → label time-courses
    5.  Compute source-level PLI (50 % overlap) → save CSVs
    6.  Persist metrics & info JSON

    Returns a summary dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "set_file": str(set_path),
        "method": method_label,
        "preproc_ok": False,
        "source_ok": False,
        "bands_ok": [],
        "timing": {},
    }

    preproc_cfg, source_cfg, conn_cfg = _build_config(use_gedai)

    # -- 1. Load ----------------------------------------------------------
    t0 = time.time()
    try:
        raw = _load_set(set_path, preload=False)
    except Exception as exc:
        LOGGER.error("Failed to load %s: %s", set_path.name, exc)
        return summary

    # -- 2. Preprocess ----------------------------------------------------
    try:
        raw_before, raw_after, ica, preproc_info = preprocess_raw(raw, preproc_cfg)
        summary["preproc_ok"] = True
    except Exception as exc:
        LOGGER.error("[%s] Preprocessing failed for %s: %s", method_label, set_path.name, exc)
        summary["timing"]["preprocess_sec"] = round(time.time() - t0, 1)
        return summary
    summary["timing"]["preprocess_sec"] = round(time.time() - t0, 1)

    # Save preprocessing metrics
    try:
        from pli_pipeline import _compute_preprocessing_metrics
        metrics = _compute_preprocessing_metrics(raw_before, raw_after, preproc_info)
        metrics["ica_components_total"] = ica.n_components_ if hasattr(ica, "n_components_") else None
        with open(out_dir / "preprocessing_metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2, default=str)
        summary["metrics"] = metrics
    except Exception as exc:
        LOGGER.warning("Metrics failed: %s", exc)

    # Save preprocessing info
    try:
        with open(out_dir / "preprocessing_info.json", "w") as fh:
            json.dump(preproc_info, fh, indent=2, default=str)
    except Exception:
        pass

    # -- 3. Sensor-level PLI ----------------------------------------------
    t1 = time.time()
    try:
        sensor_pli = _compute_sensor_pli(raw_after, conn_cfg.frequency_bands)
        picks = mne.pick_types(raw_after.info, eeg=True, meg=False)
        sensor_ch = [raw_after.ch_names[p] for p in picks]
        for bname, mat in sensor_pli.items():
            if mat is not None:
                df = pd.DataFrame(mat, index=sensor_ch, columns=sensor_ch)
                df.to_csv(out_dir / f"sensor_{bname}_PLI.csv", float_format="%.6f")
        summary["timing"]["sensor_pli_sec"] = round(time.time() - t1, 1)
    except Exception as exc:
        LOGGER.warning("[%s] Sensor PLI failed: %s", method_label, exc)
        summary["timing"]["sensor_pli_sec"] = round(time.time() - t1, 1)

    # -- 4. Source localisation -------------------------------------------
    t2 = time.time()
    try:
        inv_op, labels, src, _sdir = _prepare_inverse_operator(raw_after, source_cfg)
        label_tc, label_names, _stc = _run_source_localization(
            raw_after, inv_op, labels, src, source_cfg
        )
        summary["source_ok"] = True
    except Exception as exc:
        LOGGER.error("[%s] Source localization failed for %s: %s",
                     method_label, set_path.name, exc)
        summary["timing"]["source_sec"] = round(time.time() - t2, 1)
        return summary
    summary["timing"]["source_sec"] = round(time.time() - t2, 1)

    # -- 5. Source-level PLI with 50 % overlap ----------------------------
    t3 = time.time()
    sfreq = raw_after.info["sfreq"]
    epochs = _make_overlapping_epochs(
        label_tc, sfreq, list(label_names),
        epoch_length=conn_cfg.epoch_length,
        overlap=EPOCH_OVERLAP,
        ch_type="misc",
    )
    LOGGER.info(
        "[%s] %s: %d source epochs (%.1f s, %d%% overlap, %d labels)",
        method_label, set_path.stem, len(epochs),
        conn_cfg.epoch_length, int(EPOCH_OVERLAP * 100), len(label_names),
    )

    for bname, band in conn_cfg.frequency_bands.items():
        try:
            pli_mat = compute_pli(epochs, band, method="pli", use_gpu=False)
            # Save CSV  (same format as pipeline: label_names as index/columns)
            df = pd.DataFrame(pli_mat, index=label_names, columns=label_names)
            df.to_csv(out_dir / f"{bname}_PLI.csv", float_format="%.6f")
            summary["bands_ok"].append(bname)
        except Exception as exc:
            LOGGER.warning("[%s] Source PLI (%s) failed: %s", method_label, bname, exc)

    summary["timing"]["source_pli_sec"] = round(time.time() - t3, 1)
    summary["timing"]["total_sec"] = round(time.time() - t0, 1)
    return summary


# ---------------------------------------------------------------------------
# Raw (un-processed) sensor-level PLI baseline
# ---------------------------------------------------------------------------
def _compute_raw_baseline(
    set_path: Path,
    out_dir: Path,
    bands: Dict[str, Tuple[float, float]] = FREQUENCY_BANDS,
) -> None:
    """Compute sensor PLI on the raw file (no preprocessing) as a baseline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _load_set(set_path, preload=True)
    picks = mne.pick_types(raw.info, eeg=True, meg=False)
    ch_names = [raw.ch_names[p] for p in picks]
    data = raw.get_data(picks=picks)
    epochs = _make_overlapping_epochs(data, raw.info["sfreq"], ch_names, ch_type="eeg")
    for bname, band in bands.items():
        try:
            mat = compute_pli(epochs, band, method="pli", use_gpu=False)
            df = pd.DataFrame(mat, index=ch_names, columns=ch_names)
            df.to_csv(out_dir / f"sensor_{bname}_PLI.csv", float_format="%.6f")
        except Exception as exc:
            LOGGER.warning("Raw baseline PLI (%s) failed: %s", bname, exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(sim_dir: Optional[Path] = None, out_dir: Optional[Path] = None) -> list:
    """Process every simulated dataset through both pipeline methods.

    Parameters
    ----------
    sim_dir : Path
        Folder with .set files and ground_truth.json
        (default: ``scripts/validation/simulated_data/``).
    out_dir : Path
        Root for outputs
        (default: ``scripts/validation/validation_data/``).
    """
    if sim_dir is None:
        sim_dir = Path(__file__).resolve().parent / "simulated_data"
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "validation_data"

    gt_path = sim_dir / "ground_truth.json"
    if not gt_path.exists():
        sys.exit(f"ERROR  ground_truth.json not found at {gt_path}\n"
                 "       Run generate_simulated_eeg.py first.")
    with open(gt_path) as fh:
        ground_truth = json.load(fh)

    datasets = ground_truth["datasets"]
    neuro_out = out_dir / "neurostat_outputs"
    raw_out   = out_dir / "raw_sensor_pli"

    print("=" * 65)
    print("  NeuroStat Validation -- Batch Processing")
    print("=" * 65)
    print(f"  Simulated data : {sim_dir}")
    print(f"  Output root    : {out_dir}")
    print(f"  Datasets       : {len(datasets)}")
    print(f"  Methods        : {', '.join(m for m, _ in METHODS)}")
    print(f"  Epoch length   : {EPOCH_LENGTH} s  ({int(EPOCH_OVERLAP*100)} % overlap)")
    print(f"  Frequency bands: {', '.join(FREQUENCY_BANDS)}")
    print("=" * 65)

    all_summaries: list = []
    wall_start = time.time()

    for ds in datasets:
        fname = ds["filename"]
        level = ds["artifact_level"]
        set_path = sim_dir / fname

        if not set_path.exists():
            LOGGER.warning("Missing %s -- skipping", set_path)
            continue

        print(f"\n{'-'*65}")
        print(f"  Dataset: {fname}  ({ds['label']})")
        print(f"{'-'*65}")

        # -- Raw baseline ---------------------------------------------
        print("  > Computing raw (unprocessed) sensor-level PLI baseline …")
        try:
            _compute_raw_baseline(set_path, raw_out / level)
            print("    OK Raw baseline saved")
        except Exception as exc:
            print(f"    FAIL Raw baseline failed: {exc}")

        # -- Each pipeline method -------------------------------------
        for method_label, use_gedai in METHODS:
            method_out = neuro_out / level / method_label
            print(f"\n  > [{method_label}]  preprocessing + source-loc + PLI …")

            summary = process_single(set_path, method_label, use_gedai, method_out)
            all_summaries.append(summary)

            ok_bands = summary.get("bands_ok", [])
            t_total  = summary.get("timing", {}).get("total_sec", "?")
            print(f"    preproc={'OK' if summary['preproc_ok'] else 'FAIL'}  "
                  f"source={'OK' if summary['source_ok'] else 'FAIL'}  "
                  f"bands={','.join(ok_bands) if ok_bands else '--'}  "
                  f"time={t_total} s")

    # -- Persist run manifest -----------------------------------------
    manifest_path = out_dir / "batch_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump({
            "ground_truth": ground_truth,
            "summaries": all_summaries,
            "wall_time_sec": round(time.time() - wall_start, 1),
        }, fh, indent=2, default=str)

    # -- Summary table ------------------------------------------------
    wall_min = (time.time() - wall_start) / 60
    print(f"\n{'='*65}")
    print(f"  BATCH PROCESSING COMPLETE  ({wall_min:.1f} min)")
    print(f"{'='*65}")
    print(f"  {'Dataset':<28s} {'Method':<14s} {'Preproc':<9s} {'Source':<9s} {'Bands':<28s} {'Time':>6s}")
    print(f"  {'-'*28} {'-'*14} {'-'*9} {'-'*9} {'-'*28} {'-'*6}")
    for s in all_summaries:
        ds = Path(s["set_file"]).stem
        meth = s["method"]
        pp = "OK" if s["preproc_ok"] else "FAIL"
        sl = "OK" if s["source_ok"] else "FAIL"
        bn = ",".join(s.get("bands_ok", [])) or "--"
        tt = str(s.get("timing", {}).get("total_sec", "--"))
        print(f"  {ds:<28s} {meth:<14s} {pp:<9s} {sl:<9s} {bn:<28s} {tt:>6s}")

    print(f"\n  Manifest : {manifest_path}")
    print(f"  Outputs  : {neuro_out}")
    return all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-process simulated EEG through NeuroStat for validation.",
    )
    parser.add_argument("--sim-dir",  type=Path, default=None,
                        help="Directory with simulated .set files and ground_truth.json")
    parser.add_argument("--out-dir",  type=Path, default=None,
                        help="Root output directory  (default: scripts/validation/validation_data/)")
    args = parser.parse_args()
    main(args.sim_dir, args.out_dir)
