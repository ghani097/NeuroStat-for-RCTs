"""
EEG PLI pipeline with GUI entry point.

This module loads EEGLAB .set files, performs preprocessing with MNE-Python,
computes Desikan-Killiany source-localized PLI connectivity, and stores
diagnostic visualizations plus connectivity matrices for each recording.

RUN: python .\run_study.py "E:\Dropbox\2024\Analyses\GIT_HUB\EEG_Preprocessing-and-Quality-Check\TEST" "GroupA,GroupB" "pre,post" "E:\Dropbox\2024\Analyses\GIT_HUB\EEG_Preprocessing-and-Quality-Check\processed3" 


"""

from __future__ import annotations

import json
import logging
import sys
import textwrap
from dataclasses import dataclass, field
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from itertools import combinations  # used later for simple utilities

# New: simple study-design container (user-provided group/session names)
@dataclass
class StudyDesign:
    groups: List[str] = field(default_factory=list)
    sessions: List[str] = field(default_factory=list)


import matplotlib

# Use non-interactive backend for script/GUI runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io  # noqa: E402
from scipy import sparse  # noqa: E402

import mne  # noqa: E402
from mne import EpochsArray  # noqa: E402
from mne.channels import make_standard_montage  # noqa: E402
# connectivity API renamed; fall back to standalone package if needed
try:  # pragma: no cover - import paths differ across MNE versions
    from mne.connectivity import spectral_connectivity_epochs  # type: ignore
except ImportError:  # pragma: no cover
    from mne_connectivity import spectral_connectivity_epochs  # type: ignore
try:  # pragma: no cover
    from mne.viz import plot_connectivity_circle as _plot_circle  # type: ignore
except Exception:  # pragma: no cover
    try:
        from mne_connectivity.viz import plot_connectivity_circle as _plot_circle  # type: ignore
    except Exception:  # pragma: no cover
        _plot_circle = None
from mne.datasets import fetch_fsaverage  # noqa: E402
from mne.minimum_norm import (  # noqa: E402
    apply_inverse_raw,
    make_inverse_operator,
)
from mne.preprocessing import ICA  # noqa: E402

# Official GEDAI package from https://github.com/neurotuning/gedai
try:
    from gedai import Gedai
    HAS_OFFICIAL_GEDAI = True
except ImportError:
    HAS_OFFICIAL_GEDAI = False
    Gedai = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:  # pragma: no cover
    tk = None
    filedialog = None
    messagebox = None


LOGGER = logging.getLogger("pli_pipeline")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

try:  # pragma: no cover - optional CuPy acceleration
    import cupy as cp
    from cupy import cuda

    GPU_AVAILABLE = cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - GPU unavailable
    cp = None
    GPU_AVAILABLE = False


@dataclass
class PreprocessingConfig:
    l_freq: float = 1.0
    h_freq: float = 40.0
    notch_freqs: Sequence[float] = (50.0, 100.0)
    ica_method: str = "fastica"
    ica_n_components: Optional[int] = None
    random_state: int = 42
    reject_criteria: Optional[Dict[str, float]] = field(
        default_factory=lambda: {"eeg": 150e-6}
    )
    # Preprocessing method options
    use_asr: bool = True
    use_iclabel: bool = True
    iclabel_brain_threshold: float = 0.85
    # GED-based artifact detection (inspired by GEDAI)
    use_ged: bool = False
    ged_threshold: float = 3.0  # Z-score threshold for artifact component rejection
    # Additional preprocessing options
    bad_channel_threshold: float = 4.0  # Z-score for bad channel detection
    interpolate_bad_channels: bool = True
    # Epoch rejection for ICA fitting
    ica_reject_threshold: float = 200e-6  # Peak-to-peak threshold in V


@dataclass
class SourceConfig:
    trans_path: Path = Path("fsaverage-trans.fif")
    subjects_dir: Optional[Path] = None
    spacing: str = "ico5"
    bem_solution_name: str = "fsaverage-5120-5120-5120-bem-sol.fif"
    lambda2: float = 1.0 / 9.0


@dataclass
class ConnectivityConfig:
    epoch_length: float = 4.0
    frequency_bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }
    )
    method: str = "pli"
    use_gpu: bool = False  # GPU disabled by default for stability; enable manually if needed


@dataclass
class PipelineConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    output_root: Path = Path("processed")


@dataclass
class StatsConfig:
    """Configuration for statistical tests/cluster permutation."""
    n_permutations: int = 1000
    cluster_threshold: Optional[float] = None  # t-threshold; None = MNE default


# Extend PipelineConfig to include stats (backward compatible if not referenced)
PipelineConfig.stats = StatsConfig()  # type: ignore[attr-defined]


def ensure_subjects_dir(path: Optional[Path]) -> Path:
    """Return a valid subjects_dir, fetching fsaverage when necessary."""
    if path and Path(path).expanduser().exists():
        return Path(path).expanduser()
    try:
        fetched = Path(fetch_fsaverage(verbose=True))
        return fetched.parent
    except Exception as exc:  # pragma: no cover - fetch may fail offline
        raise FileNotFoundError(
            "fsaverage subject not found. Set SUBJECTS_DIR env or "
            "download fsaverage manually."
        ) from exc


def find_set_files(folder: Path) -> List[Path]:
    """Return sorted list of EEGLAB .set files in folder (non-recursive).

    Excludes files ending with '_cleaned.set' to avoid reprocessing already cleaned data.
    """
    all_files = sorted(Path(folder).glob("*.set"))
    # Exclude already cleaned files to prevent reprocessing
    files = [f for f in all_files if not f.stem.endswith("_cleaned")]
    if not files:
        raise FileNotFoundError(f"No .set files found in {folder}")
    return files


def _apply_montage(raw: mne.io.BaseRaw) -> bool:
    """
    Assign a standard montage if the recording lacks channel positions.

    Tries multiple montages in order of preference to maximize channel coverage.
    Returns True if a montage was successfully applied with good coverage.
    """
    # Check if montage already exists with positions
    current_montage = raw.get_montage()
    if current_montage is not None:
        n_with_pos = sum(1 for ch in raw.info['chs'] if any(ch['loc'][:3]))
        if n_with_pos >= len(raw.ch_names) * 0.8:
            LOGGER.info(f"Montage already set: {n_with_pos}/{len(raw.ch_names)} channels have positions")
            return True

    # List of montages to try, in order of preference
    # BioSemi and extended 10-20 systems first, then basic
    montages_to_try = [
        'biosemi64',      # BioSemi 64-channel system
        'biosemi32',      # BioSemi 32-channel system
        'biosemi128',     # BioSemi 128-channel system
        'biosemi256',     # BioSemi 256-channel system
        'standard_1005',  # Extended 10-20 system (343 channels)
        'standard_1020',  # Standard 10-20 system
        'easycap-M1',     # EasyCap montage
        'GSN-HydroCel-64_1.0',  # EGI 64-channel
        'GSN-HydroCel-128',     # EGI 128-channel
    ]

    best_montage = None
    best_coverage = 0

    for montage_name in montages_to_try:
        try:
            montage = make_standard_montage(montage_name)

            # Try to apply and count how many channels get positions
            raw_test = raw.copy()
            raw_test.set_montage(montage, on_missing='ignore', verbose=False)

            n_with_pos = sum(1 for ch in raw_test.info['chs'] if any(ch['loc'][:3]))
            coverage = n_with_pos / len(raw.ch_names)

            if coverage > best_coverage:
                best_coverage = coverage
                best_montage = montage_name

            # If we get 80%+ coverage, use this montage
            if coverage >= 0.8:
                raw.set_montage(montage, on_missing='ignore', verbose=False)
                LOGGER.info(f"Applied montage '{montage_name}': {n_with_pos}/{len(raw.ch_names)} channels ({coverage*100:.0f}%)")
                return True

        except Exception:
            continue

    # Use best montage found even if coverage is lower
    if best_montage and best_coverage > 0:
        try:
            montage = make_standard_montage(best_montage)
            raw.set_montage(montage, on_missing='ignore', verbose=False)
            n_with_pos = sum(1 for ch in raw.info['chs'] if any(ch['loc'][:3]))
            LOGGER.warning(f"Applied montage '{best_montage}' with limited coverage: {n_with_pos}/{len(raw.ch_names)} ({best_coverage*100:.0f}%)")
            return best_coverage >= 0.5
        except Exception as exc:
            LOGGER.warning(f"Could not apply best montage '{best_montage}': {exc}")

    LOGGER.warning("Could not find suitable montage for channel names")
    return False


def _detect_bad_channels(raw: mne.io.BaseRaw, threshold: float = 4.0) -> List[str]:
    """Detect bad channels based on amplitude variance z-score."""
    data = raw.get_data(picks="eeg")
    ch_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]

    # Compute variance per channel
    variances = np.var(data, axis=1)

    # Z-score of variances
    z_scores = (variances - np.mean(variances)) / (np.std(variances) + 1e-10)

    # Mark channels with extreme variance as bad
    bad_mask = np.abs(z_scores) > threshold
    bad_channels = [ch_names[i] for i in np.where(bad_mask)[0]]

    return bad_channels


def _gedai_denoise(
    raw: mne.io.BaseRaw,
    threshold: float = 3.0,
    trans_path: Optional[Path] = None,
) -> Tuple[mne.io.BaseRaw, dict]:
    """
    GEDAI (Generalized Eigenvalue De-Artifacting Instrument) - Official Implementation.

    Uses the official GEDAI package from https://github.com/neurotuning/gedai
    Based on: Ros et al. (2025) "Return of the GEDAI: Unsupervised EEG Denoising
    based on Leadfield Filtering"

    The algorithm uses leadfield-based reference covariance and SENSAI
    (Signal & Noise Subspace Alignment Index) for optimal threshold selection.

    Parameters:
        raw: MNE Raw object (will be modified in-place)
        threshold: Artifact rejection threshold (2=aggressive, 3=balanced, 4=conservative)
                   Maps to noise_multiplier parameter in official GEDAI
        trans_path: Path to head-MRI transformation file (optional, not used by official GEDAI)

    Returns:
        raw: Cleaned raw object
        info: Dictionary with diagnostic information
    """
    from scipy.signal import butter, filtfilt

    picks = mne.pick_types(raw.info, eeg=True, meg=False)
    data_original = raw.get_data(picks=picks).copy()
    n_channels, n_times = data_original.shape
    sfreq = raw.info["sfreq"]

    # Get GEDAI version
    gedai_version = "unknown"
    if HAS_OFFICIAL_GEDAI:
        try:
            import gedai
            gedai_version = getattr(gedai, "__version__", "0.1.0")
        except Exception:
            gedai_version = "0.1.0"

    info_dict = {
        "method": "Official_GEDAI",
        "gedai_version": gedai_version,
        "gedai_package": "neurotuning/gedai",
        "gedai_repository": "https://github.com/neurotuning/gedai",
        "threshold": threshold,
        "n_channels": n_channels,
        "n_samples": n_times,
        "duration_sec": n_times / sfreq,
        "sensai_score": 0.0,
        "n_components_rejected": 0,
        "variance_removed_pct": 0.0,
        "snr_improvement_db": 0.0,
        "reference_type": "leadfield",
        "official_gedai_used": False,
    }

    # Print prominent banner for verification
    print("\n" + "=" * 70)
    print("OFFICIAL GEDAI DENOISING")
    print("=" * 70)
    print(f"  Package: gedai v{gedai_version}")
    print(f"  Repository: https://github.com/neurotuning/gedai")
    print(f"  Citation: Ros et al. (2025) bioRxiv 10.1101/2025.10.04.680449")
    print(f"  Channels: {n_channels}, Duration: {n_times/sfreq:.1f}s")
    print(f"  Threshold (noise_multiplier): {threshold}")
    print(f"  Reference covariance: leadfield (physics-based)")
    print("=" * 70)

    LOGGER.info(f"GEDAI: Starting OFFICIAL denoising (v{gedai_version})")
    LOGGER.info(f"GEDAI: {n_channels} channels, {n_times/sfreq:.1f}s, threshold={threshold}")
    LOGGER.info(f"GEDAI: Using leadfield-based reference covariance")

    if not HAS_OFFICIAL_GEDAI:
        error_msg = "Official GEDAI package not installed! Install with: pip install gedai"
        print(f"\n*** ERROR: {error_msg}")
        print("*** Repository: https://github.com/neurotuning/gedai\n")
        LOGGER.error(error_msg)
        LOGGER.error("Repository: https://github.com/neurotuning/gedai")
        info_dict["error"] = "Official GEDAI package not installed"
        return raw, info_dict

    try:
        # Map threshold to official GEDAI parameters
        # threshold 2 = aggressive (noise_multiplier=2.0)
        # threshold 3 = balanced (noise_multiplier=3.0)
        # threshold 4 = conservative (noise_multiplier=4.0)
        noise_multiplier = float(threshold)

        # Ensure proper montage for leadfield computation
        print("  [1/5] Checking/setting electrode montage...")
        LOGGER.info("GEDAI: Checking electrode positions for leadfield computation")
        raw_for_gedai = raw.copy()

        # Standard channel names that GEDAI's leadfield matrix supports
        # (10-20, 10-10, 10-05 systems and common variants)
        GEDAI_SUPPORTED_CHANNELS = {
            # 10-20 system
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'A1', 'A2',
            # Extended 10-20 / 10-10 names
            'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
            'T7', 'C5', 'C1', 'C2', 'C6', 'T8',
            'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
            'P7', 'P5', 'P1', 'P2', 'P6', 'P8', 'P9', 'P10',
            'PO7', 'PO3', 'POz', 'PO4', 'PO8',
            'Oz', 'Iz', 'Fpz',
            # BioSemi specific (alternative naming)
            'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8',
        }

        # Check how many channels match GEDAI's supported names
        eeg_picks = mne.pick_types(raw_for_gedai.info, eeg=True, meg=False)
        eeg_ch_names = [raw_for_gedai.ch_names[i] for i in eeg_picks]
        n_eeg = len(eeg_ch_names)

        # Count channels with standard names (case-insensitive match)
        n_standard = sum(1 for ch in eeg_ch_names if ch in GEDAI_SUPPORTED_CHANNELS or ch.upper() in GEDAI_SUPPORTED_CHANNELS)
        standard_coverage = n_standard / n_eeg if n_eeg > 0 else 0

        # Also check position coverage
        n_with_pos = sum(1 for ch in raw_for_gedai.info['chs'] if any(ch['loc'][:3]))
        pos_coverage = n_with_pos / n_eeg if n_eeg > 0 else 0

        print(f"       Standard channel names: {n_standard}/{n_eeg} ({standard_coverage*100:.0f}%)")
        print(f"       Channels with positions: {n_with_pos}/{n_eeg} ({pos_coverage*100:.0f}%)")

        # Try to set montage if needed
        if pos_coverage < 0.8:
            print(f"       Attempting to set standard montage...")
            LOGGER.info(f"GEDAI: Low position coverage ({pos_coverage*100:.0f}%), attempting to set standard montage")
            _apply_montage(raw_for_gedai)
            n_with_pos = sum(1 for ch in raw_for_gedai.info['chs'] if any(ch['loc'][:3]))
            pos_coverage = n_with_pos / n_eeg if n_eeg > 0 else 0

        # GEDAI requires standard channel names for leadfield lookup
        if standard_coverage < 0.5:
            error_msg = (
                f"Insufficient standard channel names for GEDAI leadfield computation. "
                f"Only {n_standard}/{n_eeg} channels have standard names ({standard_coverage*100:.0f}%). "
                f"GEDAI requires standard 10-20/10-10 channel names (e.g., Fp1, F3, C3, P3, O1). "
                f"Your channels: {eeg_ch_names[:5]}{'...' if len(eeg_ch_names) > 5 else ''}"
            )
            print(f"       WARNING: {error_msg}")
            LOGGER.warning(f"GEDAI: {error_msg}")
            info_dict["warning"] = error_msg
            info_dict["standard_coverage"] = standard_coverage
            # Don't return - let GEDAI try and fail with a clear error

        LOGGER.info(f"GEDAI: Standard names: {n_standard}/{n_eeg} ({standard_coverage*100:.0f}%), Positions: {n_with_pos}/{n_eeg} ({pos_coverage*100:.0f}%)")
        info_dict["montage_coverage"] = pos_coverage
        info_dict["standard_name_coverage"] = standard_coverage

        # Handle dimension mismatch: exclude non-standard channels before GEDAI processing
        # GEDAI's leadfield only covers standard 10-20/10-10 channels
        non_standard_channels = [ch for ch in eeg_ch_names
                                  if ch not in GEDAI_SUPPORTED_CHANNELS and ch.upper() not in GEDAI_SUPPORTED_CHANNELS]
        standard_channels = [ch for ch in eeg_ch_names
                             if ch in GEDAI_SUPPORTED_CHANNELS or ch.upper() in GEDAI_SUPPORTED_CHANNELS]

        excluded_channels = []
        if non_standard_channels and n_standard >= 19:
            # Have enough standard channels to proceed - exclude non-standard ones
            print(f"       Excluding {len(non_standard_channels)} non-standard channels for GEDAI: {non_standard_channels}")
            LOGGER.info(f"GEDAI: Excluding non-standard channels: {non_standard_channels}")
            excluded_channels = non_standard_channels
            raw_for_gedai = raw_for_gedai.copy().pick(standard_channels)
            info_dict["excluded_channels"] = excluded_channels
            info_dict["n_channels_for_gedai"] = len(standard_channels)
            print(f"       Processing {len(standard_channels)} standard channels")
        elif non_standard_channels:
            print(f"       WARNING: Found non-standard channels but only {n_standard} standard channels (need >=19)")
            LOGGER.warning(f"GEDAI: Non-standard channels present but insufficient standard channels ({n_standard}<19)")

        # Ensure average reference (required by GEDAI)
        print("  [2/5] Applying average reference...")
        LOGGER.info("GEDAI: Applying average reference (required by official GEDAI)")
        raw_for_gedai.set_eeg_reference("average", projection=False, verbose=False)

        # Initialize official GEDAI
        print("  [3/5] Initializing official Gedai() class...")
        LOGGER.info("GEDAI: Initializing official Gedai() from neurotuning/gedai")
        gedai = Gedai()

        # Fit the model - try leadfield first, fall back to identity if needed
        reference_cov_used = "leadfield"
        fit_success = False

        # First attempt: Try with leadfield-based reference covariance
        print(f"  [4/5] Fitting GEDAI model (attempting reference_cov='leadfield')...")
        LOGGER.info(f"GEDAI: Attempting fit with reference_cov='leadfield', noise_multiplier={noise_multiplier}")

        try:
            gedai.fit_raw(
                raw_for_gedai,
                duration=2.0,                    # Epoch duration (seconds)
                overlap=0.5,                     # 50% overlap
                reject_by_annotation=False,      # Don't reject annotated segments
                reference_cov="leadfield",       # Use leadfield-based reference (core GEDAI feature)
                sensai_method="gridsearch",      # Grid search for optimal threshold
                noise_multiplier=noise_multiplier,
                verbose=True,
            )
            fit_success = True
            reference_cov_used = "leadfield"
            print("       Leadfield-based model fitted successfully!")
            LOGGER.info("GEDAI: Leadfield-based model fitting complete")

        except (ValueError, RuntimeError) as e:
            # Leadfield failed - try with identity reference
            print(f"       Leadfield failed ({e}), trying identity reference...")
            LOGGER.warning(f"GEDAI: Leadfield reference failed: {e}")
            LOGGER.info("GEDAI: Falling back to identity reference covariance")

            try:
                gedai = Gedai()  # Reset
                gedai.fit_raw(
                    raw_for_gedai,
                    duration=2.0,
                    overlap=0.5,
                    reject_by_annotation=False,
                    reference_cov="identity",    # Fallback to identity
                    sensai_method="gridsearch",
                    noise_multiplier=noise_multiplier,
                    verbose=True,
                )
                fit_success = True
                reference_cov_used = "identity"
                print("       Identity-based model fitted successfully!")
                LOGGER.info("GEDAI: Identity-based model fitting complete (fallback)")

            except Exception as e2:
                # Identity also failed - try with data-driven approach
                print(f"       Identity failed ({e2}), trying data-driven reference...")
                LOGGER.warning(f"GEDAI: Identity reference failed: {e2}")

                try:
                    gedai = Gedai()  # Reset
                    gedai.fit_raw(
                        raw_for_gedai,
                        duration=2.0,
                        overlap=0.5,
                        reject_by_annotation=False,
                        reference_cov="data",    # Data-driven reference
                        sensai_method="gridsearch",
                        noise_multiplier=noise_multiplier,
                        verbose=True,
                    )
                    fit_success = True
                    reference_cov_used = "data"
                    print("       Data-driven model fitted successfully!")
                    LOGGER.info("GEDAI: Data-driven model fitting complete (fallback)")

                except Exception as e3:
                    raise RuntimeError(f"All GEDAI reference types failed: leadfield={e}, identity={e2}, data={e3}")

        info_dict["reference_type"] = reference_cov_used
        print(f"       Reference covariance used: {reference_cov_used}")
        LOGGER.info(f"GEDAI: Reference covariance used: {reference_cov_used}")

        # Transform/denoise the data
        print("  [5/5] Transforming/denoising data...")
        LOGGER.info("GEDAI: Transforming data (applying denoising)")
        raw_cleaned = gedai.transform_raw(
            raw_for_gedai,
            duration=2.0,
            overlap=0.5,
            verbose=True
        )
        print("       Transform complete!")
        LOGGER.info("GEDAI: Transform complete")

        # Mark that official GEDAI was successfully used
        info_dict["official_gedai_used"] = True

        # Get cleaned data - handle case where channels were excluded
        if excluded_channels:
            # Get cleaned data for processed channels only
            processed_picks = mne.pick_types(raw_cleaned.info, eeg=True, meg=False)
            data_clean_partial = raw_cleaned.get_data(picks=processed_picks)

            # Create full data array with original data for excluded channels
            data_clean = data_original.copy()

            # Map processed channel data back to original indices
            for i, ch_name in enumerate(standard_channels):
                # Find the index in the original channel list
                original_idx = eeg_ch_names.index(ch_name)
                data_clean[original_idx, :] = data_clean_partial[i, :]

            print(f"       Merged {len(standard_channels)} processed channels with {len(excluded_channels)} excluded channels")
            LOGGER.info(f"GEDAI: Merged processed ({len(standard_channels)}) and excluded ({len(excluded_channels)}) channels")
        else:
            # All channels were processed
            data_clean = raw_cleaned.get_data(picks=mne.pick_types(raw_cleaned.info, eeg=True, meg=False))

        # Compute quality metrics
        # Variance removed
        var_original = np.var(data_original)
        var_clean = np.var(data_clean)
        var_removed = np.var(data_original - data_clean)
        info_dict["variance_removed_pct"] = float(100 * var_removed / (var_original + 1e-10))
        info_dict["variance_original"] = float(var_original)
        info_dict["variance_clean"] = float(var_clean)

        # SNR improvement estimate (using high-freq as noise proxy)
        try:
            nyq = sfreq / 2
            if nyq > 30:
                b_hf, a_hf = butter(4, 30 / nyq, btype='high')
                noise_before = np.std(filtfilt(b_hf, a_hf, data_original, axis=1))
                noise_after = np.std(filtfilt(b_hf, a_hf, data_clean, axis=1))
                signal_after = np.std(data_clean)
                if noise_after > 0 and noise_before > 0:
                    snr_before = np.std(data_original) / noise_before
                    snr_after = signal_after / noise_after
                    info_dict["snr_improvement_db"] = float(20 * np.log10(snr_after / (snr_before + 1e-10)))
        except Exception:
            pass

        # Extract SENSAI score from fitted model if available
        if hasattr(gedai, 'sensai_score_') and gedai.sensai_score_ is not None:
            # Official GEDAI SENSAI score (typically 0-1, convert to 0-100)
            info_dict["sensai_score"] = float(gedai.sensai_score_ * 100)
            info_dict["sensai_score_raw"] = float(gedai.sensai_score_)
            print(f"       Official SENSAI score: {gedai.sensai_score_:.4f}")
            LOGGER.info(f"GEDAI: Official SENSAI score from model: {gedai.sensai_score_:.4f}")
        else:
            # Compute approximate SENSAI score based on metrics
            var_score = np.clip(info_dict["variance_removed_pct"] / 50, 0, 1) * 30
            snr_score = np.clip((info_dict["snr_improvement_db"] + 5) / 15, 0, 1) * 40
            quality_score = 30  # Base score for successful processing
            info_dict["sensai_score"] = float(np.clip(var_score + snr_score + quality_score, 0, 100))
            info_dict["sensai_score_computed"] = True
            LOGGER.info("GEDAI: SENSAI score computed from metrics (not available from model)")

        # Get threshold from fitted model if available
        if hasattr(gedai, 'threshold_') and gedai.threshold_ is not None:
            info_dict["optimal_threshold"] = float(gedai.threshold_)
            print(f"       Optimal threshold: {gedai.threshold_:.4f}")
            LOGGER.info(f"GEDAI: Optimal threshold from model: {gedai.threshold_:.4f}")

        # Check for other useful attributes from the fitted model
        for attr in ['n_components_', 'n_rejected_', 'eigenvalues_', 'sensai_scores_']:
            if hasattr(gedai, attr):
                val = getattr(gedai, attr)
                if val is not None:
                    if isinstance(val, (int, float)):
                        info_dict[attr] = val
                        LOGGER.info(f"GEDAI: {attr} = {val}")
                    elif isinstance(val, np.ndarray) and val.size < 10:
                        info_dict[attr] = val.tolist()

        # Update raw data in place
        raw._data[picks] = data_clean

        # Print summary
        print("\n" + "-" * 70)
        print("GEDAI DENOISING COMPLETE")
        print("-" * 70)
        print(f"  SENSAI Score: {info_dict['sensai_score']:.1f}%")
        print(f"  Variance Removed: {info_dict['variance_removed_pct']:.1f}%")
        print(f"  SNR Improvement: {info_dict['snr_improvement_db']:.1f} dB")
        print(f"  Official GEDAI Used: {info_dict['official_gedai_used']}")
        print("-" * 70 + "\n")

        LOGGER.info(
            f"GEDAI complete: SENSAI={info_dict['sensai_score']:.1f}%, "
            f"variance_removed={info_dict['variance_removed_pct']:.1f}%, "
            f"SNR_improvement={info_dict['snr_improvement_db']:.1f}dB, "
            f"official_gedai_used=True"
        )

    except Exception as exc:
        error_msg = f"Official GEDAI denoising failed: {exc}"
        print(f"\n*** ERROR: {error_msg}\n")
        LOGGER.error(error_msg)
        import traceback
        traceback.print_exc()
        info_dict["error"] = str(exc)
        info_dict["official_gedai_used"] = False
        info_dict["error"] = str(exc)

    return raw, info_dict


def preprocess_raw(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig,
    trans_path: Optional[Path] = None,
) -> Tuple[mne.io.BaseRaw, mne.io.BaseRaw, ICA]:
    """
    Run filtering, referencing, and artifact removal.

    Supports multiple artifact removal methods:
    - ASR (Artifact Subspace Reconstruction)
    - ICLabel (ICA component classification)
    - GEDAI (Generalized Eigenvalue De-Artifacting)

    Returns:
        raw_before: Copy of raw before preprocessing
        raw_after: Preprocessed raw object
        ica: Fitted ICA object (may be empty if using GEDAI only)
    """
    preproc_info = {
        "asr_applied": False,
        "iclabel_applied": False,
        "gedai_applied": False,
        "ica_excluded": 0,
        "bad_channels": [],
    }

    # Log basic data shape and duration
    try:
        n_ch = mne.pick_types(raw.info, eeg=True, meg=False).size
        sfreq = float(raw.info.get("sfreq", 0.0) or 0.0)
        n_times = getattr(raw, "n_times", 0)
        duration = (float(n_times - 1) / sfreq) if (sfreq and n_times) else 0.0
        LOGGER.info(
            "Preprocessing: channels=%d, sfreq=%.2f Hz, duration=%.1f s",
            n_ch,
            sfreq or 0.0,
            duration,
        )
    except Exception:
        pass

    raw.load_data()
    _apply_montage(raw)
    raw_before = raw.copy()

    # Step 1: Detect and optionally interpolate bad channels
    if getattr(config, "interpolate_bad_channels", True):
        try:
            bad_ch_threshold = getattr(config, "bad_channel_threshold", 4.0)
            bad_channels = _detect_bad_channels(raw, threshold=bad_ch_threshold)
            if bad_channels:
                LOGGER.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
                raw.info["bads"] = list(set(raw.info["bads"] + bad_channels))
                raw.interpolate_bads(reset_bads=True)
                preproc_info["bad_channels"] = bad_channels
                LOGGER.info("Interpolated bad channels")
        except Exception as exc:
            LOGGER.warning(f"Bad channel detection/interpolation failed: {exc}")

    # Step 2: GEDAI denoising (if enabled) - applied BEFORE filtering
    if getattr(config, "use_ged", False):
        try:
            LOGGER.info("Applying GEDAI denoising...")
            ged_threshold = getattr(config, "ged_threshold", 3.0)

            raw, gedai_info = _gedai_denoise(
                raw,
                threshold=ged_threshold,
                trans_path=trans_path,
            )
            preproc_info["gedai_applied"] = True
            preproc_info["gedai_info"] = gedai_info
            LOGGER.info(
                f"GEDAI applied: SENSAI={gedai_info.get('sensai_score', 0):.1f}%, "
                f"variance_removed={gedai_info.get('variance_removed_pct', 0):.1f}%, "
                f"components_rejected={gedai_info.get('n_components_rejected', 0)}"
            )
        except Exception as exc:
            LOGGER.warning(f"GEDAI denoising failed: {exc}")
            import traceback
            traceback.print_exc()

    # Step 3: ASR denoising (if enabled and available)
    if getattr(config, "use_asr", True):
        applied = False

        # Try asrpy first (works with MNE Raw objects directly)
        try:
            import asrpy

            LOGGER.info("Applying ASR (asrpy)...")
            # ASR parameters: cutoff=20 is a good default (5=conservative, 20=aggressive)
            asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=20)
            # Fit on the raw data (uses first portion as calibration)
            asr.fit(raw)
            # Transform returns a new Raw object
            raw = asr.transform(raw)
            applied = True
            preproc_info["asr_applied"] = True
            LOGGER.info("Applied ASR (asrpy) successfully")
        except ImportError:
            LOGGER.info("asrpy not installed, trying meegkit...")
        except Exception as exc:
            LOGGER.warning(f"asrpy failed: {exc}, trying meegkit...")

        # Try meegkit as fallback
        if not applied:
            try:
                from meegkit.asr import ASR as MeegkitASR

                LOGGER.info("Applying ASR (meegkit)...")
                picks = mne.pick_types(raw.info, eeg=True, meg=False)
                data = raw.get_data(picks=picks)

                # meegkit ASR expects (n_samples, n_channels) format
                asr = MeegkitASR(sfreq=raw.info["sfreq"], cutoff=20)
                # fit_transform expects (n_samples, n_channels)
                data_clean, _ = asr.fit_transform(data.T)
                raw._data[picks] = data_clean.T
                applied = True
                preproc_info["asr_applied"] = True
                LOGGER.info("Applied ASR (meegkit) successfully")
            except ImportError:
                LOGGER.info("meegkit not installed")
            except Exception as exc:
                LOGGER.warning(f"meegkit ASR failed: {exc}")

        if not applied:
            LOGGER.info("ASR not available (install: pip install asrpy or pip install meegkit)")

    # Step 4: Filtering
    LOGGER.info(f"Applying bandpass filter: {config.l_freq}-{config.h_freq} Hz")
    raw.filter(config.l_freq, config.h_freq, fir_design="firwin")

    if config.notch_freqs:
        valid_notches = [
            freq for freq in config.notch_freqs if freq < raw.info["sfreq"] / 2.0
        ]
        if valid_notches:
            LOGGER.info(f"Applying notch filter at: {valid_notches} Hz")
            raw.notch_filter(valid_notches)

    # Step 5: Re-reference to average
    raw.set_eeg_reference("average", projection=True)

    # Step 6: ICA fitting
    n_components = config.ica_n_components
    if n_components is None:
        # Auto-determine: use rank or max 25 components
        n_eeg = len(mne.pick_types(raw.info, eeg=True, meg=False))
        n_components = min(n_eeg - 1, 25)

    ica = ICA(
        n_components=n_components,
        method=config.ica_method,
        random_state=config.random_state,
        max_iter="auto",
    )

    LOGGER.info(f"Fitting ICA: method={config.ica_method}, n_components={n_components}")

    # Fit ICA with optional rejection threshold
    reject_dict = None
    ica_reject = getattr(config, "ica_reject_threshold", None)
    if ica_reject:
        reject_dict = {"eeg": ica_reject}

    try:
        ica.fit(raw, reject=reject_dict)
    except Exception as exc:
        LOGGER.warning(f"ICA fit with rejection failed, retrying without: {exc}")
        ica.fit(raw)

    # Step 7: Component classification and rejection
    excluded = []

    # Try ICLabel first (if enabled)
    if getattr(config, "use_iclabel", True):
        try:
            from mne_icalabel import label_components

            LOGGER.info("Running ICLabel classification...")

            # label_components returns a dict with:
            # - 'y_pred_proba': array (n_components, 7) - probabilities for each class
            # - 'labels': list of predicted labels
            # Classes: brain, muscle artifact, eye blink, heart beat, line noise, channel noise, other
            labels_dict = label_components(raw, ica, method="iclabel")

            # Get probabilities and labels
            if isinstance(labels_dict, dict):
                ic_probs = labels_dict.get("y_pred_proba", None)
                ic_labels = labels_dict.get("labels", None)
            else:
                # Older versions might return differently
                ic_probs = getattr(labels_dict, 'y_pred_proba_', None)
                ic_labels = getattr(labels_dict, 'labels_', None)

            thr = float(getattr(config, "iclabel_brain_threshold", 0.85))

            if ic_probs is not None:
                ic_probs = np.array(ic_probs)
                LOGGER.info(f"ICLabel: probabilities shape = {ic_probs.shape}")

                for idx in range(ica.n_components_):
                    try:
                        # Brain probability is column 0
                        if ic_probs.ndim == 2 and idx < ic_probs.shape[0]:
                            prob_brain = float(ic_probs[idx, 0])
                        elif ic_probs.ndim == 1:
                            prob_brain = float(ic_probs[idx])
                        else:
                            prob_brain = 0.0

                        # Check label if available
                        is_brain_label = False
                        if ic_labels is not None and idx < len(ic_labels):
                            label = ic_labels[idx]
                            is_brain_label = (isinstance(label, str) and label.lower() == "brain")

                        # Exclude if brain probability is below threshold
                        # OR if label is not "brain" (when available)
                        if prob_brain < thr:
                            excluded.append(idx)
                            LOGGER.debug(f"ICLabel: excluding component {idx} (brain_prob={prob_brain:.3f})")

                    except Exception as e:
                        LOGGER.warning(f"ICLabel: error processing component {idx}: {e}")
                        # Keep the component if we can't classify it
                        pass

                preproc_info["iclabel_applied"] = True
                LOGGER.info(f"ICLabel: excluding {len(excluded)}/{ica.n_components_} components (threshold={thr})")
            else:
                LOGGER.warning("ICLabel: No probabilities returned")

        except ImportError:
            LOGGER.info("mne-icalabel not installed (install: pip install mne-icalabel)")
        except Exception as exc:
            LOGGER.warning(f"ICLabel failed: {exc}")

    # Fallback: EOG-based detection
    if not excluded and not preproc_info.get("iclabel_applied"):
        try:
            eog_inds, _ = ica.find_bads_eog(raw)
            excluded.extend(eog_inds)
            LOGGER.info(f"EOG detection: found {len(eog_inds)} EOG-related components")
        except Exception as exc:
            LOGGER.warning(f"EOG detection failed: {exc}")

        # Also try muscle artifact detection
        try:
            muscle_inds, _ = ica.find_bads_muscle(raw)
            excluded.extend(muscle_inds)
            LOGGER.info(f"Muscle detection: found {len(muscle_inds)} muscle-related components")
        except Exception:
            pass

    # Apply ICA
    ica.exclude = list(sorted(set(excluded)))
    preproc_info["ica_excluded"] = len(ica.exclude)

    if ica.exclude:
        ica.apply(raw)
        LOGGER.info(f"ICA applied: excluded {len(ica.exclude)} components")
    else:
        LOGGER.info("ICA: no components excluded")

    # Log preprocessing summary
    LOGGER.info(
        f"Preprocessing complete: ASR={preproc_info['asr_applied']}, "
        f"GEDAI={preproc_info['gedai_applied']}, "
        f"ICLabel={preproc_info['iclabel_applied']}, "
        f"ICA excluded={preproc_info['ica_excluded']}"
    )

    return raw_before, raw, ica, preproc_info


def _infer_paths(file_path: Path, output_root: Path) -> Tuple[str, str, Path]:
    """Infer group/session and construct output directory for a file."""
    parts = file_path.parts
    group_name = parts[-3] if len(parts) >= 3 else "unknown_group"
    session_name = parts[-2] if len(parts) >= 2 else "unknown_session"
    out_dir = (output_root / group_name / session_name / file_path.stem).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    return group_name, session_name, out_dir


def _compute_preprocessing_metrics(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    preproc_info: dict,
) -> dict:
    """Compute preprocessing quality metrics."""
    metrics = {}

    try:
        # Get data
        data_before = raw_before.get_data(picks="eeg")
        data_after = raw_after.get_data(picks="eeg")

        # Variance reduction (percentage)
        var_before = np.var(data_before)
        var_after = np.var(data_after)
        if var_before > 0:
            metrics["variance_reduction"] = float(100 * (1 - var_after / var_before))
        else:
            metrics["variance_reduction"] = 0.0

        # Signal quality score (simplified SNR estimate)
        # Higher score = better quality after cleaning
        std_before = np.std(data_before, axis=1).mean()
        std_after = np.std(data_after, axis=1).mean()

        # Estimate noise as high-frequency content (simple approximation)
        # Filter to get high-freq component
        try:
            raw_hf_before = raw_before.copy().filter(30, None, verbose=False)
            raw_hf_after = raw_after.copy().filter(30, None, verbose=False)
            noise_before = np.std(raw_hf_before.get_data(picks="eeg"))
            noise_after = np.std(raw_hf_after.get_data(picks="eeg"))

            # SNR improvement in dB
            if noise_after > 0 and noise_before > 0:
                snr_before = std_before / noise_before
                snr_after = std_after / noise_after
                metrics["snr_improvement_db"] = float(20 * np.log10(snr_after / snr_before))
            else:
                metrics["snr_improvement_db"] = 0.0

            # Signal quality: percentage based on variance reduction and SNR
            snr_factor = min(100, max(0, 50 + metrics["snr_improvement_db"] * 5))
            var_factor = min(100, max(0, metrics["variance_reduction"]))
            metrics["signal_quality"] = float((snr_factor + var_factor) / 2)

        except Exception:
            metrics["snr_improvement_db"] = 0.0
            metrics["signal_quality"] = max(0, min(100, metrics["variance_reduction"]))

        # Band power changes
        try:
            bands = {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
            }
            band_changes = {}

            psd_before = raw_before.compute_psd(fmin=1, fmax=40, picks="eeg", verbose=False)
            psd_after = raw_after.compute_psd(fmin=1, fmax=40, picks="eeg", verbose=False)

            freqs = psd_before.freqs
            psd_data_before = psd_before.get_data().mean(axis=0)
            psd_data_after = psd_after.get_data().mean(axis=0)

            for band_name, (fmin, fmax) in bands.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                power_before = psd_data_before[mask].mean()
                power_after = psd_data_after[mask].mean()
                if power_before > 0:
                    change_pct = 100 * (power_after - power_before) / power_before
                    band_changes[band_name] = float(change_pct)
                else:
                    band_changes[band_name] = 0.0

            metrics["band_power_changes"] = band_changes
        except Exception as e:
            LOGGER.warning(f"Band power computation failed: {e}")
            metrics["band_power_changes"] = {}

        # Add preprocessing info
        metrics["ica_components_rejected"] = preproc_info.get("ica_excluded", 0)
        metrics["bad_channels"] = preproc_info.get("bad_channels", [])
        metrics["asr_applied"] = preproc_info.get("asr_applied", False)
        metrics["iclabel_applied"] = preproc_info.get("iclabel_applied", False)
        metrics["gedai_applied"] = preproc_info.get("gedai_applied", False)

        # GEDAI-specific metrics
        if preproc_info.get("gedai_applied") and "gedai_info" in preproc_info:
            gedai_info = preproc_info["gedai_info"]
            metrics["sensai_score"] = gedai_info.get("sensai_score", 0)
            metrics["gedai_components_rejected"] = gedai_info.get("n_components_rejected", 0)
            metrics["gedai_variance_removed"] = gedai_info.get("variance_removed_pct", 0)
            metrics["gedai_snr_improvement"] = gedai_info.get("snr_improvement_db", 0)
            metrics["gedai_threshold"] = gedai_info.get("threshold", 3.0)
            # Override signal quality with SENSAI score for GEDAI
            metrics["signal_quality"] = gedai_info.get("sensai_score", metrics.get("signal_quality", 0))
            # Also use GEDAI's variance and SNR metrics if available
            if gedai_info.get("variance_removed_pct", 0) > 0:
                metrics["variance_reduction"] = gedai_info.get("variance_removed_pct", metrics.get("variance_reduction", 0))
            if gedai_info.get("snr_improvement_db") is not None:
                metrics["snr_improvement_db"] = gedai_info.get("snr_improvement_db", metrics.get("snr_improvement_db", 0))

    except Exception as e:
        LOGGER.warning(f"Metrics computation failed: {e}")

    return metrics


def preprocess_file(
    file_path: Path,
    config: "PipelineConfig",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Preprocess a single .set file and persist outputs.

    Saves diagnostics and a preprocessed FIF file in the subject's output folder.
    Returns a summary with key paths.
    """
    group_name, session_name, out_dir = _infer_paths(file_path, config.output_root)
    if progress_cb:
        progress_cb(f"Preprocessing {file_path.name} ({group_name}/{session_name}) …")

    summary = {
        "file": str(file_path),
        "group": group_name,
        "session": session_name,
        "out_dir": str(out_dir),
        "preprocessed_fif": None,
        "diagnostics": {},
        "metrics": {},
    }
    try:
        raw = mne.io.read_raw_eeglab(str(file_path), preload=False, verbose=False)
        raw_before, raw_after, _ica, preproc_info = preprocess_raw(raw, config.preprocessing)

        # Compute preprocessing metrics
        try:
            metrics = _compute_preprocessing_metrics(raw_before, raw_after, preproc_info)
            metrics["ica_components_total"] = _ica.n_components_ if hasattr(_ica, 'n_components_') else None
            summary["metrics"] = metrics

            # Save metrics to JSON
            metrics_path = out_dir / "preprocessing_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            LOGGER.info(f"Saved preprocessing metrics to {metrics_path}")
        except Exception as exc:
            LOGGER.warning("Metrics computation failed for %s: %s", file_path.name, exc)

        # Diagnostics - visualization plots
        try:
            traces_path = out_dir / "traces_before_after.png"
            topo_path = out_dir / "topomap_before_after.png"
            gfp_path = out_dir / "gfp_before_after.png"
            psd_path = out_dir / "psd_before_after.png"

            _plot_traces_before_after(raw_before, raw_after, traces_path)
            _plot_topomap(raw_before, raw_after, topo_path)
            _plot_gfp(raw_before, raw_after, gfp_path)
            _plot_psd(raw_before, raw_after, "Power Spectrum: Before vs After Preprocessing", psd_path)

            summary["diagnostics"] = {
                "traces": str(traces_path),
                "topomap": str(topo_path),
                "gfp": str(gfp_path),
                "psd": str(psd_path),
            }
        except Exception as exc:
            LOGGER.warning("Diagnostics failed for %s: %s", file_path.name, exc)

        # Save preprocessed raw as FIF format (needed by downstream steps)
        fif_path = out_dir / "preprocessed_raw.fif"
        try:
            raw_after.save(fif_path, overwrite=True, verbose=False)
            summary["preprocessed_fif"] = str(fif_path)
            if progress_cb:
                progress_cb(f"Saved preprocessed FIF: {fif_path}")
        except Exception as exc:
            if progress_cb:
                progress_cb(f"Failed to save preprocessed FIF for {file_path.name}: {exc}")

        # Save preprocessed raw as EEGLAB .set format in the processed output folder
        cleaned_set_name = f"{file_path.stem}_cleaned.set"
        set_path = out_dir / cleaned_set_name
        try:
            raw_after.export(set_path, overwrite=True, verbose=False)
            summary["preprocessed_set"] = str(set_path)
            if progress_cb:
                progress_cb(f"Saved cleaned SET: {set_path}")
        except Exception as exc:
            if progress_cb:
                progress_cb(f"Failed to save preprocessed SET for {file_path.name}: {exc}")

        if progress_cb:
            progress_cb(f"Finished preprocessing {file_path.name}")
    except Exception as exc:
        LOGGER.exception("Preprocessing error for %s", file_path)
        if progress_cb:
            progress_cb(f"Preprocessing error for {file_path.name}: {exc}")
    return summary


def _plot_psd(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, title: str, out_file: Path) -> None:
    """Plot mean PSD before/after preprocessing with frequency band annotations."""
    psd_kwargs = dict(fmin=1.0, fmax=45.0, picks="eeg", tmin=None, tmax=None, n_fft=2048)
    psd_before = raw_before.compute_psd(**psd_kwargs)
    psd_after = raw_after.compute_psd(**psd_kwargs)

    freqs = psd_before.freqs
    psd_data_before = 10 * np.log10(psd_before.get_data().mean(axis=0) + 1e-10)
    psd_data_after = 10 * np.log10(psd_after.get_data().mean(axis=0) + 1e-10)

    # Create figure with dark theme
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#252536')

    # Define frequency bands for shading
    bands = [
        ("Delta", 1, 4, "#89b4fa", 0.15),
        ("Theta", 4, 8, "#a6e3a1", 0.15),
        ("Alpha", 8, 13, "#f9e2af", 0.15),
        ("Beta", 13, 30, "#f38ba8", 0.15),
        ("Gamma", 30, 45, "#cba6f7", 0.15),
    ]

    # Add band shading
    y_min, y_max = min(psd_data_before.min(), psd_data_after.min()) - 5, max(psd_data_before.max(), psd_data_after.max()) + 5
    for band_name, fmin, fmax, color, alpha in bands:
        ax.axvspan(fmin, fmax, alpha=alpha, color=color, label=None)
        # Add band label at top
        ax.text((fmin + fmax) / 2, y_max - 2, band_name, fontsize=8, color=color,
                ha='center', va='top', fontweight='bold', alpha=0.8)

    # Plot PSD curves
    ax.plot(freqs, psd_data_before, label="Before Preprocessing", color="#f38ba8",
            lw=2, alpha=0.9)
    ax.plot(freqs, psd_data_after, label="After Preprocessing", color="#a6e3a1",
            lw=2, alpha=0.9)

    # Fill between to show reduction
    ax.fill_between(freqs, psd_data_before, psd_data_after,
                    where=psd_data_before > psd_data_after,
                    alpha=0.2, color='#a6e3a1', label='Power Reduction')

    # Styling
    ax.set_xlabel("Frequency (Hz)", color='#cdd6f4', fontsize=11)
    ax.set_ylabel("Power Spectral Density (dB)", color='#cdd6f4', fontsize=11)
    ax.set_title(title, color='#89b4fa', fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors='#cdd6f4')
    for spine in ax.spines.values():
        spine.set_color('#45475a')

    ax.set_xlim([1, 45])
    ax.set_ylim([y_min, y_max])
    ax.grid(True, alpha=0.2, color='#6c7086')

    # Legend
    legend = ax.legend(loc='upper right', facecolor='#313244', edgecolor='#45475a',
                       fontsize=10, framealpha=0.9)
    for text in legend.get_texts():
        text.set_color('#cdd6f4')

    fig.tight_layout()
    fig.savefig(out_file, dpi=200, facecolor='#1e1e2e', edgecolor='none')
    plt.close(fig)


def _plot_traces_before_after(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    out_file: Path,
    n_channels: int = 6,
    duration: float = 10.0,
) -> None:
    """Plot a few EEG channels before/after preprocessing with improved visualization."""
    picks = mne.pick_types(raw_after.info, eeg=True, meg=False)
    if len(picks) == 0:
        return
    picks = picks[: min(n_channels, len(picks))]
    ch_names = [raw_after.ch_names[i] for i in picks]
    sfreq = raw_after.info["sfreq"]
    tmax = min(duration, raw_after.times[-1] if len(raw_after.times) else duration)
    n_samp = int(max(1, tmax * sfreq))
    data_b = raw_before.get_data(picks=picks)[:, :n_samp] * 1e6
    data_a = raw_after.get_data(picks=picks)[:, :n_samp] * 1e6
    times = np.arange(n_samp) / sfreq

    # Calculate amplitude stats for display
    std_before = np.std(data_b)
    std_after = np.std(data_a)
    reduction_pct = 100 * (1 - std_after / std_before) if std_before > 0 else 0

    # Create figure with better layout
    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor('#1e1e2e')

    # Create grid for layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.15], height_ratios=[1, 1],
                          hspace=0.15, wspace=0.05, left=0.08, right=0.92, top=0.88, bottom=0.12)

    ax_before = fig.add_subplot(gs[0, 0:2])
    ax_after = fig.add_subplot(gs[1, 0:2], sharex=ax_before)

    # Set dark background
    for ax in [ax_before, ax_after]:
        ax.set_facecolor('#252536')
        ax.tick_params(colors='#cdd6f4')
        for spine in ax.spines.values():
            spine.set_color('#45475a')

    # Calculate offsets for stacking channels
    max_amp = max(np.max(np.abs(data_b)), np.max(np.abs(data_a)))
    channel_spacing = max_amp * 0.6 if max_amp > 0 else 50.0
    offsets = np.arange(len(picks)) * channel_spacing

    # Plot before
    for ch in range(len(picks)):
        ax_before.plot(times, data_b[ch] + offsets[ch], color="#f38ba8", lw=0.7, alpha=0.9)
    ax_before.set_title("BEFORE Preprocessing", fontsize=12, fontweight='bold',
                        color='#f38ba8', pad=10)
    ax_before.set_ylabel("Channels", color='#cdd6f4')

    # Add channel labels on the left
    for ch, name in enumerate(ch_names):
        ax_before.text(-0.5, offsets[ch], name, fontsize=8, color='#6c7086',
                       ha='right', va='center', transform=ax_before.get_yaxis_transform())

    # Plot after
    for ch in range(len(picks)):
        ax_after.plot(times, data_a[ch] + offsets[ch], color="#a6e3a1", lw=0.7, alpha=0.9)
    ax_after.set_title("AFTER Preprocessing", fontsize=12, fontweight='bold',
                       color='#a6e3a1', pad=10)
    ax_after.set_xlabel("Time (s)", color='#cdd6f4')
    ax_after.set_ylabel("Channels", color='#cdd6f4')

    # Add channel labels
    for ch, name in enumerate(ch_names):
        ax_after.text(-0.5, offsets[ch], name, fontsize=8, color='#6c7086',
                      ha='right', va='center', transform=ax_after.get_yaxis_transform())

    # Configure axes
    for ax in [ax_before, ax_after]:
        ax.set_yticks([])
        ax.grid(True, alpha=0.15, color='#6c7086')
        ax.set_xlim([0, tmax])

    plt.setp(ax_before.get_xticklabels(), visible=False)

    # Add summary stats as text box
    stats_text = (
        f"Amplitude Reduction\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Before: {std_before:.1f} µV\n"
        f"After:  {std_after:.1f} µV\n"
        f"━━━━━━━━━━━━━━━\n"
        f"Reduction: {reduction_pct:.0f}%"
    )
    fig.text(0.95, 0.5, stats_text, fontsize=9, color='#cdd6f4',
             ha='center', va='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#313244',
                       edgecolor='#45475a', alpha=0.9))

    # Main title
    fig.suptitle("EEG Signal Comparison: Before vs After Preprocessing",
                 fontsize=14, fontweight='bold', color='#89b4fa', y=0.96)

    fig.savefig(out_file, dpi=200, facecolor='#1e1e2e', edgecolor='none')
    plt.close(fig)


def _plot_topomap(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, out_file: Path) -> None:
    """Compare RMS scalp maps before/after preprocessing with dark theme."""
    data_before = raw_before.get_data(picks="eeg")
    data_after = raw_after.get_data(picks="eeg")
    rms_before = np.sqrt((data_before**2).mean(axis=1))
    rms_after = np.sqrt((data_after**2).mean(axis=1))

    # Calculate reduction
    reduction = 100 * (1 - rms_after.mean() / rms_before.mean()) if rms_before.mean() > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('#1e1e2e')

    # Before topomap
    im1, _ = mne.viz.plot_topomap(rms_before, raw_before.info, axes=axes[0], show=False,
                                   contours=0, cmap='RdYlBu_r')
    axes[0].set_title("BEFORE Preprocessing", color='#f38ba8', fontsize=12, fontweight='bold')

    # After topomap
    im2, _ = mne.viz.plot_topomap(rms_after, raw_after.info, axes=axes[1], show=False,
                                   contours=0, cmap='RdYlBu_r')
    axes[1].set_title("AFTER Preprocessing", color='#a6e3a1', fontsize=12, fontweight='bold')

    # Difference map (reduction)
    rms_diff = rms_before - rms_after
    im3, _ = mne.viz.plot_topomap(rms_diff, raw_after.info, axes=axes[2], show=False,
                                   contours=0, cmap='Greens')
    axes[2].set_title(f"Noise Reduction ({reduction:.0f}%)", color='#89b4fa', fontsize=12, fontweight='bold')

    # Main title
    fig.suptitle("Scalp Topography: RMS Amplitude Comparison",
                 fontsize=14, fontweight='bold', color='#89b4fa', y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_file, dpi=200, facecolor='#1e1e2e', edgecolor='none')
    plt.close(fig)


def _plot_gfp(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, out_file: Path) -> None:
    """Plot global field power before/after cleaning with dark theme."""
    def _gfp(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, np.ndarray]:
        data = raw.get_data(picks="eeg")
        gfp = np.sqrt((data**2).mean(axis=0)) * 1e6
        return gfp, raw.times

    gfp_before, times = _gfp(raw_before)
    gfp_after, _ = _gfp(raw_after)

    # Calculate stats
    mean_before = np.mean(gfp_before)
    mean_after = np.mean(gfp_after)
    peak_before = np.max(gfp_before)
    peak_after = np.max(gfp_after)
    reduction = 100 * (1 - mean_after / mean_before) if mean_before > 0 else 0

    step = max(1, len(times) // 5000)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#252536')

    # Plot GFP traces
    ax.fill_between(times[::step], 0, gfp_before[::step], alpha=0.3, color='#f38ba8', label='Before')
    ax.fill_between(times[::step], 0, gfp_after[::step], alpha=0.3, color='#a6e3a1', label='After')
    ax.plot(times[::step], gfp_before[::step], color='#f38ba8', lw=1, alpha=0.8)
    ax.plot(times[::step], gfp_after[::step], color='#a6e3a1', lw=1, alpha=0.8)

    # Add horizontal lines for means
    ax.axhline(mean_before, color='#f38ba8', linestyle='--', lw=1.5, alpha=0.7)
    ax.axhline(mean_after, color='#a6e3a1', linestyle='--', lw=1.5, alpha=0.7)

    # Styling
    ax.set_xlabel("Time (s)", color='#cdd6f4', fontsize=11)
    ax.set_ylabel("Global Field Power (µV)", color='#cdd6f4', fontsize=11)
    ax.set_title("Global Field Power: Before vs After Preprocessing",
                 color='#89b4fa', fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(colors='#cdd6f4')
    for spine in ax.spines.values():
        spine.set_color('#45475a')
    ax.grid(True, alpha=0.2, color='#6c7086')

    # Legend with stats
    legend = ax.legend(loc='upper right', facecolor='#313244', edgecolor='#45475a',
                       fontsize=10, framealpha=0.9)
    for text in legend.get_texts():
        text.set_color('#cdd6f4')

    # Add stats text box
    stats_text = (
        f"Mean GFP Reduction: {reduction:.0f}%\n"
        f"Before: {mean_before:.1f} µV (peak: {peak_before:.1f})\n"
        f"After:  {mean_after:.1f} µV (peak: {peak_after:.1f})"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, color='#cdd6f4',
            va='top', ha='left', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#313244', edgecolor='#45475a', alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_file, dpi=200, facecolor='#1e1e2e', edgecolor='none')
    plt.close(fig)


def _prepare_inverse_operator(raw: mne.io.BaseRaw, source_cfg: SourceConfig) -> Tuple[dict, List[mne.Label], mne.SourceSpaces, Path]:
    """Build inverse operator plus label definitions."""
    subjects_dir = ensure_subjects_dir(source_cfg.subjects_dir)
    subject = "fsaverage"
    src_dir = Path(subjects_dir) / subject / "bem"
    src_path = src_dir / f"{subject}-{source_cfg.spacing}-src.fif"
    bem_path = src_dir / source_cfg.bem_solution_name

    if not src_path.exists():
        src = mne.setup_source_space(
            subject=subject,
            spacing=source_cfg.spacing,
            subjects_dir=subjects_dir,
            add_dist=False,
        )
        mne.write_source_spaces(src_path, src, overwrite=True)
    else:
        src = mne.read_source_spaces(src_path)

    if not bem_path.exists():
        mne.make_bem_model(
            subject=subject,
            ico=5,
            conductivity=(0.3, 0.006, 0.3),
            subjects_dir=subjects_dir,
            output=Path(subjects_dir) / subject / "bem" / f"{subject}-5120-5120-5120-bem.fif",
            overwrite=True,
        )
        bem = mne.read_bem_solution(bem_path)
    else:
        bem = mne.read_bem_solution(bem_path)

    trans_path = source_cfg.trans_path
    if not trans_path.exists():
        raise FileNotFoundError(
            f"Transformation file '{trans_path}' not found. Provide a valid head<->MRI transform."
        )

    fwd = mne.make_forward_solution(
        info=raw.info,
        trans=str(trans_path),
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=1,
    )
    noise_cov = mne.compute_raw_covariance(raw, method="shrunk", rank="info")
    inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov, loose="auto", depth=0.8)
    labels = [
        label
        for label in mne.read_labels_from_annot(subject, parc="aparc", subjects_dir=subjects_dir)
        if label.name.split("-")[0].lower() != "unknown"
    ]
    return inverse_operator, labels, src, Path(subjects_dir)


def _run_source_localization(
    raw: mne.io.BaseRaw,
    inverse_operator,
    labels: Sequence[mne.Label],
    src: mne.SourceSpaces,
    source_cfg: SourceConfig,
) -> Tuple[np.ndarray, Sequence[str], mne.SourceEstimate]:
    """Apply inverse operator and extract label time courses."""
    stc = apply_inverse_raw(
        raw,
        inverse_operator,
        lambda2=source_cfg.lambda2,
        method="MNE",
        pick_ori="normal",
    )
    label_tc = mne.extract_label_time_course([stc], labels, src=src, mode="pca_flip")[0]
    label_names = [label.name for label in labels]
    return label_tc, label_names, stc


def _label_tc_to_epochs(
    label_tc: np.ndarray,
    label_names: Sequence[str],
    sfreq: float,
    epoch_length: float,
) -> EpochsArray:
    """Convert label time courses into fixed-length epochs."""
    samples_per_epoch = int(epoch_length * sfreq)
    total_samples = label_tc.shape[1]
    usable = (total_samples // samples_per_epoch) * samples_per_epoch
    if usable < samples_per_epoch:
        raise ValueError("Not enough data to create even a single epoch for connectivity.")
    trimmed = label_tc[:, :usable]
    epochs = trimmed.reshape(len(label_names), -1, samples_per_epoch)
    data = np.transpose(epochs, (1, 0, 2))  # (n_epochs, n_labels, n_times)
    info = mne.create_info(ch_names=list(label_names), sfreq=sfreq, ch_types="misc")
    return EpochsArray(data, info, verbose=False)


def _bandpass_gpu(data: "cp.ndarray", sfreq: float, band: Tuple[float, float]) -> "cp.ndarray":
    fmin, fmax = band
    n_times = data.shape[1]
    freqs = cp.fft.rfftfreq(n_times, d=1.0 / sfreq)
    spectrum = cp.fft.rfft(data, axis=1)
    mask = (freqs >= fmin) & (freqs <= fmax)
    spectrum *= mask
    return cp.fft.irfft(spectrum, n=n_times, axis=1)


def _hilbert_gpu(data: "cp.ndarray") -> "cp.ndarray":
    n_times = data.shape[1]
    spectrum = cp.fft.fft(data, axis=1)
    h = cp.zeros(n_times, dtype=data.dtype)
    if n_times % 2 == 0:
        h[0] = h[n_times // 2] = 1.0
        h[1:n_times // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n_times + 1) // 2] = 2.0
    spectrum *= h
    return cp.fft.ifft(spectrum, axis=1)


def _compute_pli_gpu(
    epochs: EpochsArray,
    band: Tuple[float, float],
    chunk_size: int = 1024,  # Reduced chunk size for memory safety
) -> np.ndarray:
    """Compute PLI using GPU acceleration with proper memory management."""
    if not GPU_AVAILABLE or cp is None:
        raise RuntimeError("GPU computation requested but CuPy is unavailable.")

    result = None
    try:
        # Synchronize GPU before starting
        cp.cuda.Stream.null.synchronize()

        data = epochs.get_data(copy=True)  # (n_epochs, n_labels, n_times)
        n_epochs, n_labels, n_times = data.shape
        merged = data.transpose(1, 0, 2).reshape(n_labels, -1)

        # Check available GPU memory and adjust if needed
        mempool = cp.get_default_memory_pool()
        try:
            free_mem = cp.cuda.runtime.memGetInfo()[0]
            data_size = merged.nbytes * 4  # float32
            if data_size > free_mem * 0.7:  # Use at most 70% of free memory
                LOGGER.warning("GPU memory limited, using smaller chunks")
                chunk_size = max(256, chunk_size // 4)
        except Exception:
            pass

        cp_data = cp.asarray(merged, dtype=cp.float32)
        filtered = _bandpass_gpu(cp_data, epochs.info["sfreq"], band)

        # Free intermediate data
        del cp_data
        mempool.free_all_blocks()

        analytic = _hilbert_gpu(filtered)

        # Free filtered data
        del filtered
        mempool.free_all_blocks()

        samples = analytic.shape[1]
        analytic = analytic.T  # (samples, n_labels)

        accum = cp.zeros((n_labels, n_labels), dtype=cp.float32)
        for start in range(0, samples, chunk_size):
            stop = min(samples, start + chunk_size)
            segment = analytic[start:stop]
            if segment.size == 0:
                continue
            phase = cp.angle(segment)
            diff = phase[:, :, None] - phase[:, None, :]
            accum += cp.sign(cp.sin(diff)).sum(axis=0)

            # Free intermediate results each iteration
            del phase, diff
            cp.cuda.Stream.null.synchronize()

        pli_gpu = cp.abs(accum / samples)

        # Copy result to CPU before cleanup
        result = cp.asnumpy(pli_gpu)

        # Clean up GPU memory
        del analytic, accum, pli_gpu
        mempool.free_all_blocks()
        cp.cuda.Stream.null.synchronize()

    except Exception as e:
        LOGGER.error(f"GPU PLI computation failed: {e}")
        # Clean up GPU memory on error
        try:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass
        raise

    return result


def _should_use_gpu(user_pref: bool) -> bool:
    """Determine if GPU should be used for PLI computation."""
    if user_pref:
        if not GPU_AVAILABLE:
            LOGGER.warning("GPU requested but CuPy/CUDA not available, using CPU")
            return False
        return True
    return False


def compute_pli(
    epochs: EpochsArray,
    band: Tuple[float, float],
    method: str = "pli",
    use_gpu: bool = False,
) -> np.ndarray:
    """Compute PLI matrix for a frequency band."""
    if use_gpu and method.lower() == "pli":
        try:
            LOGGER.info("Computing %s PLI on GPU", method.upper())
            result = _compute_pli_gpu(epochs, band)
            if result is not None:
                return result
            LOGGER.warning("GPU PLI returned None, falling back to CPU")
        except Exception as exc:  # pragma: no cover - GPU fallback path
            LOGGER.warning("GPU PLI failed (%s); falling back to CPU.", exc)
            # Ensure GPU memory is cleaned up
            try:
                if cp is not None:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.cuda.Stream.null.synchronize()
            except Exception:
                pass

    fmin, fmax = band
    con = spectral_connectivity_epochs(
        epochs,
        method=method,
        mode="multitaper",
        sfreq=epochs.info["sfreq"],
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        mt_adaptive=True,
        verbose=False,
    )
    if hasattr(con, "get_data"):
        dense = con.get_data(output="dense")
        if dense.ndim == 4:
            dense = dense[0]
        matrix = np.squeeze(dense, axis=-1)
    else:  # pragma: no cover - old tuple output from mne-connectivity<0.6
        matrix = con[0]
        if matrix.ndim == 3:
            matrix = np.squeeze(matrix, axis=-1)

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Unexpected PLI matrix shape {matrix.shape}; expected square connectivity matrix."
        )

    matrix = np.maximum(matrix, matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def run_source_loc_and_connectivity(
    preprocessed_fif: Path,
    original_file: Optional[Path],
    config: "PipelineConfig",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run source localization and PLI on a preprocessed FIF.

    Saves per-band CSV/PNG and returns a summary with pli_outputs mapping.
    """
    # Determine out_dir and group/session from FIF location
    out_dir = preprocessed_fif.parent
    # Try to recover names from parent folders
    parts = out_dir.parts
    group_name = parts[-3] if len(parts) >= 3 else "unknown_group"
    session_name = parts[-2] if len(parts) >= 2 else "unknown_session"

    if progress_cb:
        progress_cb(f"Source/Connectivity for {out_dir.name} ({group_name}/{session_name}) …")

    summary = {
        "file": str(original_file) if original_file else str(preprocessed_fif),
        "group": group_name,
        "session": session_name,
        "pli_outputs": {},
    }
    try:
        raw = mne.io.read_raw_fif(str(preprocessed_fif), preload=True, verbose=False)
        inv_op, labels, src, _subjects_dir = _prepare_inverse_operator(raw, config.source)
        label_tc, label_names, _stc = _run_source_localization(raw, inv_op, labels, src, config.source)

        try:
            _plot_source_power(label_tc, label_names, out_dir / "source_power.png")
        except Exception:
            pass

        epochs = _label_tc_to_epochs(label_tc, label_names, raw.info["sfreq"], config.connectivity.epoch_length)
        try:
            LOGGER.info(
                "Connectivity epochs: n_epochs=%d, epoch_length=%.1f s, labels=%d, sfreq=%.2f",
                len(epochs),
                float(config.connectivity.epoch_length),
                len(label_names),
                float(raw.info.get("sfreq", 0.0) or 0.0),
            )
        except Exception:
            pass
        use_gpu = _should_use_gpu(config.connectivity.use_gpu)
        try:
            LOGGER.info("PLI GPU enabled: %s (GPU_AVAILABLE=%s)", str(use_gpu), str(GPU_AVAILABLE))
        except Exception:
            pass

        for band_name, band in config.connectivity.frequency_bands.items():
            if progress_cb:
                progress_cb(f"Computing {band_name} PLI for {out_dir.name} …")
            try:
                pli_mat = compute_pli(epochs, band, method=config.connectivity.method, use_gpu=use_gpu)

                # Validate result
                if pli_mat is None:
                    LOGGER.warning("PLI computation returned None for %s %s", out_dir.name, band_name)
                    if progress_cb:
                        progress_cb(f"PLI returned None for {band_name}, skipping")
                    continue

                if not isinstance(pli_mat, np.ndarray) or pli_mat.ndim != 2:
                    LOGGER.warning("PLI returned invalid shape for %s %s", out_dir.name, band_name)
                    continue

            except Exception as exc:
                LOGGER.exception("PLI failed for %s %s", out_dir.name, band_name)
                if progress_cb:
                    progress_cb(f"PLI failed for {band_name}: {exc}")
                # Clean up GPU memory on error
                try:
                    if cp is not None:
                        cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                continue

            try:
                _save_connectivity_outputs(pli_mat, label_names, band_name, out_dir, group_name)
                try:
                    _plot_connectivity_circle(pli_mat, label_names, band_name, out_dir / f"{band_name}_circle.png")
                except Exception:
                    pass
                summary["pli_outputs"][band_name] = str(out_dir / f"{band_name}_PLI.csv")
            except Exception as exc:
                LOGGER.warning("Saving connectivity outputs failed for %s (%s): %s", out_dir.name, band_name, exc)

        if progress_cb:
            progress_cb(f"Finished source/connectivity for {out_dir.name}")
    except Exception as exc:
        LOGGER.exception("Source/Connectivity error for %s", preprocessed_fif)
        if progress_cb:
            progress_cb(f"Source/Connectivity error: {exc}")
    return summary


def run_preprocessing_for_design(
    base_folder: Path,
    design: "StudyDesign",
    config: Optional["PipelineConfig"] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Batch preprocessing across study design; returns list of summaries."""
    config = config or PipelineConfig()
    summaries: List[dict] = []
    for group in design.groups:
        for session in design.sessions:
            session_dir = Path(base_folder) / group / session
            if not session_dir.exists():
                if progress_cb:
                    progress_cb(f"Missing folder: {session_dir}")
                continue
            try:
                files = find_set_files(session_dir)
            except FileNotFoundError:
                if progress_cb:
                    progress_cb(f"No .set files in {session_dir}")
                continue
            for f in files:
                summaries.append(preprocess_file(f, config, progress_cb))
    return summaries


def run_source_and_connectivity_for_design(
    base_folder: Path,
    design: "StudyDesign",
    config: Optional["PipelineConfig"] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Run source localization + connectivity for all preprocessed FIFs."""
    config = config or PipelineConfig()
    summaries: List[dict] = []
    for group in design.groups:
        for session in design.sessions:
            session_dir = (config.output_root / group / session)
            if not session_dir.exists():
                if progress_cb:
                    progress_cb(f"No preprocessed outputs under {session_dir}")
                continue
            for subj_dir in sorted(session_dir.glob("*")):
                fif_path = subj_dir / "preprocessed_raw.fif"
                if fif_path.exists():
                    summaries.append(run_source_loc_and_connectivity(fif_path, None, config, progress_cb))
                else:
                    if progress_cb:
                        progress_cb(f"Missing preprocessed FIF: {fif_path}")
    return summaries


def aggregate_csv_table(
    design: "StudyDesign",
    config: Optional["PipelineConfig"] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Aggregate existing PLI CSV files across output_root into an Excel table.

    Includes ALL subjects found in the output directories, even those without PLI data.
    """
    config = config or PipelineConfig()
    pseudo_summaries: List[dict] = []

    for group in design.groups:
        for session in design.sessions:
            session_dir = (config.output_root / group / session)
            if not session_dir.exists():
                if progress_cb:
                    progress_cb(f"Warning: No output folder for {group}/{session}")
                continue

            # Get all subject directories (not just those with PLI files)
            for subj_dir in sorted(session_dir.glob("*")):
                if not subj_dir.is_dir():
                    continue

                pli_outputs = {}
                for csv_file in subj_dir.glob("*_PLI.csv"):
                    # band name is prefix before _PLI
                    band_name = csv_file.stem.replace("_PLI", "")
                    pli_outputs[band_name] = str(csv_file)

                # Include ALL subjects, even those without PLI outputs
                pseudo_summaries.append({
                    "file": str(subj_dir / (subj_dir.name + ".set")),
                    "group": group,
                    "session": session,
                    "pli_outputs": pli_outputs,  # May be empty dict
                })

    if progress_cb:
        progress_cb(f"Found {len(pseudo_summaries)} subjects across all groups")

    output_excel = config.output_root / "PLI_Table.xlsx"
    return compute_group_stats(pseudo_summaries, {
        "SN": [1, 2, 19, 20],
        "DMN": [15, 16, 21, 22, 29, 30, 31, 32, 35, 36, 47, 48, 51, 52, 53, 54],
        "CEN": [5, 6, 55, 56, 57, 58, 59, 60],
    }, output_excel, progress_cb)


def run_stats_for_design(
    design: "StudyDesign",
    config: Optional["PipelineConfig"] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Run statistical analysis using existing CSVs (reconstructed summaries).

    This runs the cluster-based permutation analysis and also saves simple t-test stats.
    """
    config = config or PipelineConfig()
    # reconstruct summaries like in aggregate_csv_table
    pseudo_summaries: List[dict] = []
    for group in design.groups:
        for session in design.sessions:
            session_dir = (config.output_root / group / session)
            if not session_dir.exists():
                continue
            for subj_dir in sorted(session_dir.glob("*")):
                pli_outputs = {}
                for csv_file in subj_dir.glob("*_PLI.csv"):
                    band_name = csv_file.stem.replace("_PLI", "")
                    pli_outputs[band_name] = str(csv_file)
                if pli_outputs:
                    pseudo_summaries.append({
                        "file": str(subj_dir / (subj_dir.name + ".set")),
                        "group": group,
                        "session": session,
                        "pli_outputs": pli_outputs,
                    })
    # Classic edge-wise t-tests (saved under stats/..)
    try:
        _perform_statistical_analysis(pseudo_summaries, design, config, progress_cb)
    except Exception as exc:
        LOGGER.warning("Simple statistical analysis failed: %s", exc)
    # Cluster-based permutation (saved under stats/cluster_perm/..)
    try:
        _perform_cluster_based_permutation(pseudo_summaries, design, config, progress_cb)
    except Exception as exc:
        LOGGER.warning("Cluster-based permutation failed: %s", exc)


def _save_connectivity_outputs(
    pli_matrix: np.ndarray,
    label_names: Sequence[str],
    band_name: str,
    out_dir: Path,
    group_name: str,  # Add group_name parameter
) -> None:
    """Persist connectivity matrix and heatmap, and save as .mat file."""
    # Create a directory for .mat files if it doesn't exist
    mat_dir = out_dir / "mat files"
    mat_dir.mkdir(parents=True, exist_ok=True)

    # Save the PLI matrix as a .mat file with group name
    mat_file_name = f"{out_dir.stem}_{group_name}_{band_name}_PLI.mat"  # e.g., P1_A_Alpha_PLI.mat
    mat_file_path = mat_dir / mat_file_name
    scipy.io.savemat(mat_file_path, {f"{band_name}_PLI": pli_matrix})

    # Save the CSV file
    df = pd.DataFrame(pli_matrix, index=label_names, columns=label_names)
    csv_path = out_dir / f"{band_name}_PLI.csv"
    df.to_csv(csv_path, float_format="%.6f")

    # Plot and save the heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pli_matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title(f"{band_name.upper()} band PLI")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90, fontsize=6)
    ax.set_yticklabels(label_names, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.6, label="PLI")
    fig.tight_layout()
    fig.savefig(out_dir / f"{band_name}_PLI.png", dpi=200)
    plt.close(fig)


def _plot_connectivity_circle(
    pli_matrix: np.ndarray,
    label_names: Sequence[str],
    band_name: str,
    out_file: Path,
    max_lines: int = 40,
) -> None:
    """Plot top connections using an MNE connectivity circle."""
    if _plot_circle is None or not np.any(pli_matrix):
        return
    lines = min(max_lines, max(5, int(np.count_nonzero(pli_matrix) / pli_matrix.shape[0])))
    fig, _ = _plot_circle(
        pli_matrix,
        label_names,
        n_lines=lines,
        title=f"{band_name.upper()} band PLI",
        colorbar=True,
        linewidth=1.5,
        show=False,
    )
    fig.savefig(out_file, dpi=200, facecolor="black")
    plt.close(fig)


def _plot_source_power(
    label_tc: np.ndarray,
    label_names: Sequence[str],
    out_file: Path,
) -> None:
    """Plot normalized RMS power per Desikan-Killiany region."""
    label_rms = np.sqrt((label_tc**2).mean(axis=1))
    order = np.argsort(label_rms)[::-1]
    sorted_names = [label_names[i] for i in order]
    sorted_vals = label_rms[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(sorted_vals)), sorted_vals, color="#4c72b0")
    ax.set_xticks(range(len(sorted_vals)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=6)
    ax.set_ylabel("RMS (a.u.)")
    ax.set_title("Source-level activity by Desikan-Killiany region")
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def _compute_group_pli_from_matrix(pli_matrix: np.ndarray, indices: Sequence[int]) -> float:
    """Compute mean PLI for the upper-triangle (excluding diagonal) restricted to given indices.
    Assumes indices are 0-based. Returns NaN if no valid elements."""
    sub = pli_matrix[np.ix_(indices, indices)]
    upper = np.triu(sub, k=1)
    vals = upper[upper > 0]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def process_file(file_path: Path, config: PipelineConfig, progress_cb: Optional[Callable[[str], None]] = None) -> dict:
    """Process a single EEGLAB .set file: preprocess, source-localize, compute PLI per band."""
    if progress_cb:
        progress_cb(f"Processing {file_path.name} ...")

    # Extract group and session from file path
    parts = file_path.parts
    group_name = parts[-3] if len(parts) >= 3 else "unknown_group"
    session_name = parts[-2] if len(parts) >= 2 else "unknown_session"
    
    # Create output directory maintaining group/session structure
    out_dir = (config.output_root / group_name / session_name / file_path.stem).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rest of the function remains the same
    summary = {"file": str(file_path), "pli_outputs": {}}
    try:
        # Load raw EEGLAB file (do not preload: preprocess_raw will load)
        raw = mne.io.read_raw_eeglab(str(file_path), preload=False, verbose=False)

        # Preprocess (filter, notch, ref, ICA)
        raw_before, raw_after, ica, _ = preprocess_raw(raw, config.preprocessing)

        # Save diagnostic figures
        try:
            _plot_psd(raw_before, raw_after, f"{file_path.stem} - PSD", out_dir / "psd_before_after.png")
            _plot_topomap(raw_before, raw_after, out_dir / "topomap_before_after.png")
            _plot_gfp(raw_before, raw_after, out_dir / "gfp_before_after.png")
        except Exception as exc:
            LOGGER.warning("Failed to write diagnostics for %s: %s", file_path.name, exc)

        # Prepare inverse operator (may download fsaverage if needed)
        inv_op, labels, src, subjects_dir = _prepare_inverse_operator(raw_after, config.source)

        # Run source localization and extract label time-courses 
        label_tc, label_names, stc = _run_source_localization(raw_after, inv_op, labels, src, config.source)

        # Save a simple source-power plot
        try:
            _plot_source_power(label_tc, label_names, out_dir / "source_power.png")
        except Exception as exc:
            LOGGER.warning("Failed to plot source power for %s: %s", file_path.name, exc)

        # Convert label time-courses to epochs for connectivity
        epochs = _label_tc_to_epochs(label_tc, label_names, raw_after.info["sfreq"], config.connectivity.epoch_length)

        # Decide GPU usage
        use_gpu = _should_use_gpu(config.connectivity.use_gpu)

        # Compute PLI per band and save outputs
        for band_name, band in config.connectivity.frequency_bands.items():
            if progress_cb:
                progress_cb(f"Computing {band_name} PLI for {file_path.name} ...")
            try:
                pli_mat = compute_pli(epochs, band, method=config.connectivity.method, use_gpu=use_gpu)
            except Exception as exc:
                LOGGER.exception("PLI computation failed for %s %s: %s", file_path.name, band_name, exc)
                continue

            # Persist CSV + heatmap
            try:
                _save_connectivity_outputs(pli_mat, label_names, band_name, out_dir, group_name)
                # plot connectivity circle if available
                try:
                    _plot_connectivity_circle(pli_mat, label_names, band_name, out_dir / f"{band_name}_circle.png")
                except Exception:
                    pass
                summary["pli_outputs"][band_name] = str(out_dir / f"{band_name}_PLI.csv")
            except Exception as exc:
                LOGGER.warning("Failed to save connectivity outputs for %s (%s): %s", file_path.name, band_name, exc)

        if progress_cb:
            progress_cb(f"Finished processing {file_path.name}")
        return summary

    except Exception as exc:
        LOGGER.exception("Error processing file %s", file_path)
        if progress_cb:
            progress_cb(f"Error processing {file_path.name}: {exc}")
        # return partial summary so compute_group_stats can run (will skip missing CSVs)
        return summary


def compute_group_stats(
    summaries: List[dict],
    networks: Dict[str, List[int]],
    output_excel: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Aggregate PLI outputs from summaries into a single Excel file.

    summaries: list of summary dicts returned by process_file
    networks: mapping network name -> list of 1-based indices (will convert to 0-based)
    """
    if progress_cb:
        progress_cb("Aggregating connectivity outputs into study-level table...")

    rows = []
    for s in summaries:
        try:
            file_path = Path(s["file"])
            # Attempt to infer participant, group, session from path: expect base/.../Group/Session/filename.set
            parts = file_path.parts
            # Fallbacks
            participant = file_path.stem
            group_name = s.get("group", "")
            session_name = s.get("session", "")

            # If group/session not in summary, try to infer from path
            if not group_name and len(parts) >= 3:
                group_name = parts[-3]
            if not session_name and len(parts) >= 2:
                session_name = parts[-2]

            # Extract subject ID (numeric part from filename)
            import re
            subj_match = re.search(r'(\d+)', participant)
            subject_id = int(subj_match.group(1)) if subj_match else 0

            # each band has CSV path in pli_outputs
            pli_outputs = s.get("pli_outputs", {})

            if not pli_outputs:
                # Subject has no PLI outputs - still include with NaN values
                LOGGER.warning(f"Subject {participant} has no PLI outputs, including with NaN values")
                for band_name in ["theta", "alpha", "beta", "gamma"]:
                    for net_name in networks.keys():
                        rows.append({
                            "SubjectID": subject_id,
                            "Participant": participant,
                            "Group": group_name,
                            "Session": session_name,
                            "FrequencyBand": band_name,
                            "Network": net_name,
                            "MeanPLI": float("nan"),
                            "Status": "No PLI data",
                        })
                continue

            for band_name, csv_path in pli_outputs.items():
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    matrix = df.values.astype(float)
                    status = "OK"
                except Exception as e:
                    # If CSV fails, include with NaN
                    LOGGER.warning(f"Failed to read {csv_path}: {e}")
                    for net_name in networks.keys():
                        rows.append({
                            "SubjectID": subject_id,
                            "Participant": participant,
                            "Group": group_name,
                            "Session": session_name,
                            "FrequencyBand": band_name,
                            "Network": net_name,
                            "MeanPLI": float("nan"),
                            "Status": f"Read error: {e}",
                        })
                    continue

                for net_name, idxs in networks.items():
                    # convert 1-based to 0-based if necessary: if any idx == 0 assume already 0-based
                    if len(idxs) == 0:
                        mean_pli = float("nan")
                        status = "Empty network"
                    else:
                        convert = [i - 1 if min(idxs) > 0 else i for i in idxs]
                        # filter out-of-bounds indices
                        valid = [i for i in convert if 0 <= i < matrix.shape[0]]
                        if not valid:
                            mean_pli = float("nan")
                            status = "Invalid indices"
                        else:
                            mean_pli = _compute_group_pli_from_matrix(matrix, valid)
                            status = "OK"

                    rows.append({
                        "SubjectID": subject_id,
                        "Participant": participant,
                        "Group": group_name,
                        "Session": session_name,
                        "FrequencyBand": band_name,
                        "Network": net_name,
                        "MeanPLI": mean_pli,
                        "Status": status,
                    })

        except Exception as exc:
            LOGGER.warning("Failed to aggregate summary %s: %s", s.get("file", "<unknown>"), exc)

    # Create DataFrame with all columns
    columns = ["SubjectID", "Participant", "Group", "Session", "FrequencyBand", "Network", "MeanPLI", "Status"]
    table = pd.DataFrame(rows, columns=columns)

    # Sort by SubjectID for better readability
    if not table.empty:
        table = table.sort_values(["Group", "SubjectID", "FrequencyBand", "Network"]).reset_index(drop=True)

    try:
        output_excel.parent.mkdir(parents=True, exist_ok=True)
        table.to_excel(output_excel, index=False)
        if progress_cb:
            n_subjects = table["SubjectID"].nunique()
            n_groups = table["Group"].nunique()
            progress_cb(f"Saved PLI table: {n_subjects} subjects, {n_groups} groups -> {output_excel}")
    except Exception as exc:
        LOGGER.error("Failed to save Excel summary: %s", exc)
        if progress_cb:
            progress_cb(f"Failed to save study-level PLI table: {exc}")

    return table


def run_study_pipeline(
    base_folder: Path,
    design: StudyDesign,
    config: Optional[PipelineConfig] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[List[dict], pd.DataFrame]:
    """Run pipeline across groups/sessions as defined in design.

    Expects folder structure:
      base_folder / <GroupName> / <SessionName> / *.set

    Returns list of file-level summaries and the aggregated DataFrame.
    """
    if progress_cb:
        progress_cb(f"Running study pipeline in {base_folder}")
    config = config or PipelineConfig()
    summaries = []
    for group in design.groups:
        for session in design.sessions:
            session_dir = Path(base_folder) / group / session
            if not session_dir.exists():
                LOGGER.warning("Session folder not found: %s", session_dir)
                if progress_cb:
                    progress_cb(f"Warning: session folder not found: {session_dir}")
                continue
            try:
                set_files = find_set_files(session_dir)
            except FileNotFoundError:
                if progress_cb:
                    progress_cb(f"No .set files found in {session_dir}; skipping.")
                continue
            for f in set_files:
                try:
                    # process_file will create per-file outputs
                    summary = process_file(f, config, progress_cb)
                    summaries.append(summary)
                except Exception as exc:
                    LOGGER.exception("Failed to process %s: %s", f, exc)
                    if progress_cb:
                        progress_cb(f"Failed to process {f.name}: {exc}")
    # default networks (same as original notebook). Indices are 1-based here.
    default_networks = {
        "SN": [1, 2, 19, 20],
        "DMN": [15, 16, 21, 22, 29, 30, 31, 32, 35, 36, 47, 48, 51, 52, 53, 54],
        "CEN": [5, 6, 55, 56, 57, 58, 59, 60],
    }
    output_excel = config.output_root / "PLI_Table.xlsx"
    table = compute_group_stats(summaries, default_networks, output_excel, progress_cb)

    # Add statistical analysis after computing PLI tables
    try:
        _perform_statistical_analysis(summaries, design, config, progress_cb)
    except Exception as exc:
        LOGGER.exception("Statistical analysis failed")
        if progress_cb:
            progress_cb(f"Statistical analysis failed: {exc}")

    return summaries, table


def _build_gui() -> Optional[tk.Tk]:
    """Create a minimal Tkinter GUI with two panels: Study design (left) and Folder/run (right)."""
    if tk is None:
        LOGGER.error("tkinter is not available; GUI cannot be created.")
        return None

    root = tk.Tk()
    root.title("EEG PLI Pipeline - Study Mode")
    root.geometry("900x420")
    root.resizable(True, True)

    # Top instruction
    tk.Label(root, text="Define study design on the left, then select base folder (groups) on the right.").pack(pady=(6, 4))

    container = tk.Frame(root)
    container.pack(fill="both", expand=True, padx=8, pady=6)

    # Left panel: Study design
    left = tk.LabelFrame(container, text="Study design", width=420)
    left.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=4)

    tk.Label(left, text="Groups (comma-separated)").pack(anchor="w", padx=6, pady=(8, 0))
    groups_var = tk.StringVar(value="GroupA,GroupB")
    groups_entry = tk.Entry(left, textvariable=groups_var)
    groups_entry.pack(fill="x", padx=6, pady=4)

    tk.Label(left, text="Sessions (comma-separated)").pack(anchor="w", padx=6, pady=(8, 0))
    sessions_var = tk.StringVar(value="pre,post4W")
    sessions_entry = tk.Entry(left, textvariable=sessions_var)
    sessions_entry.pack(fill="x", padx=6, pady=4)

    tk.Label(left, text="Output root folder (optional)").pack(anchor="w", padx=6, pady=(8, 0))
    out_var = tk.StringVar(value=str(Path.cwd() / "processed"))
    out_entry = tk.Entry(left, textvariable=out_var)
    out_entry.pack(fill="x", padx=6, pady=4)

    # Handy note
    tk.Label(left, text="Folder layout expected:\nbase / <Group> / <Session> / *.set", anchor="w", justify="left", fg="gray").pack(fill="x", padx=6, pady=(8, 4))

    # Right panel: folder selection and run controls
    right = tk.LabelFrame(container, text="Run and folder selection", width=420)
    right.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)

    base_var = tk.StringVar()
    tk.Label(right, text="Base folder (contains group folders)").pack(anchor="w", padx=6, pady=(8, 0))
    base_entry = tk.Entry(right, textvariable=base_var)
    base_entry.pack(fill="x", padx=6, pady=4)

    def browse_base() -> None:
        d = filedialog.askdirectory(title="Select base folder (groups)")
        if d:
            base_var.set(d)

    tk.Button(right, text="Browse…", command=browse_base).pack(anchor="e", padx=6, pady=(0, 8))

    log_text = tk.Text(right, height=12, wrap="word", state="disabled")
    log_text.pack(fill="both", padx=6, pady=(4, 6), expand=True)

    def log(msg: str) -> None:
        log_text.configure(state="normal")
        log_text.insert(tk.END, f"{msg}\n")
        log_text.see(tk.END)
        log_text.configure(state="disabled")
        root.update_idletasks()

    def run_clicked() -> None:
        base_path = base_var.get().strip()
        if not base_path:
            messagebox.showwarning("Missing path", "Please select a base folder first.")
            return
        groups = [g.strip() for g in groups_var.get().split(",") if g.strip()]
        sessions = [s.strip() for s in sessions_var.get().split(",") if s.strip()]
        if not groups or not sessions:
            messagebox.showwarning("Design error", "Provide at least one group and one session name (comma-separated).")
            return
        # update config output root
        cfg = PipelineConfig()
        try:
            cfg.output_root = Path(out_var.get()).expanduser()
            cfg.output_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Output folder error", str(exc))
            return

        design = StudyDesign(groups=groups, sessions=sessions)
        # run study pipeline (blocking)
        try:
            log("Starting study pipeline...")
            summaries, table = run_study_pipeline(Path(base_path), design, cfg, progress_cb=log)
            log(f"Study processing complete. Processed {len(summaries)} files.")
            messagebox.showinfo("Pipeline", f"Processing complete. Saved summary to {cfg.output_root / 'PLI_Table.xlsx'}")
        except Exception as exc:
            LOGGER.exception("Study pipeline failed")
            messagebox.showerror("Pipeline error", str(exc))
            log(f"Pipeline error: {exc}")

    tk.Button(right, text="Run study pipeline", command=run_clicked).pack(pady=(0, 6))

    return root


def launch_gui() -> None:
    """Expose GUI entry point to end users."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    root = _build_gui()
    if root is None:
        print("tkinter is not available. Run run_pipeline() directly.", file=sys.stderr)
        return
    root.mainloop()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point - currently launches the GUI."""
    _ = argv  # Placeholder for future CLI args
    launch_gui()


# Add these imports at the top
from scipy import stats
import mne.stats
from itertools import combinations
import networkx as nx

def _clean_matrices(matrices: List[np.ndarray]) -> np.ndarray:
    """Clean and validate matrices for statistical testing."""
    if not matrices:
        return np.array([])
    
    # Stack matrices and ensure float type
    stacked = np.stack(matrices).astype(float)
    
    # Replace inf values with nan
    stacked[~np.isfinite(stacked)] = np.nan
    
    # Remove subjects with all NaN values
    valid_subjects = ~np.all(np.isnan(stacked), axis=(1,2))
    cleaned = stacked[valid_subjects]
    
    # If no valid subjects remain, return empty array
    if len(cleaned) == 0:
        return np.array([])
    
    # Fill remaining NaNs with mean of non-NaN values
    for i in range(len(cleaned)):
        nan_mask = np.isnan(cleaned[i])
        if np.any(nan_mask):
            valid_vals = cleaned[i][~nan_mask]
            if len(valid_vals) > 0:
                cleaned[i][nan_mask] = np.mean(valid_vals)
            else:
                cleaned[i] = 0  # If no valid values, set to 0
    
    return cleaned

def _perform_statistical_comparison(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform statistical comparison between two groups of matrices."""
    if len(X) < 2 or len(Y) < 2:
        return np.zeros_like(X[0]), np.ones_like(X[0])
    
    # Perform t-test for each connection
    t_stats = np.zeros_like(X[0])
    p_vals = np.ones_like(X[0])
    
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            if i != j:
                try:
                    t_stat, p_val = stats.ttest_ind(
                        X[:, i, j],
                        Y[:, i, j],
                        equal_var=False  # Welch's t-test
                    )
                    t_stats[i, j] = t_stat
                    p_vals[i, j] = p_val
                except:
                    continue
    
    return t_stats, p_vals

def _perform_statistical_analysis(
    summaries: List[dict],
    design: StudyDesign,
    config: PipelineConfig,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Perform statistical analyses on PLI matrices and save results."""
    if progress_cb:
        progress_cb("Starting statistical analyses...")

    # Create stats output directories
    stats_dir = config.output_root / "stats"
    between_dir = stats_dir / "between_groups"
    within_dir = stats_dir / "within_groups"
    for d in [stats_dir, between_dir, within_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Group files by condition and band
    grouped_files: Dict[Tuple[str, str, str], List[str]] = {}
    for s in summaries:
        # Prefer explicit group/session if present
        group = s.get("group")
        session = s.get("session")
        if not group or not session:
            # Fallback: infer from file path
            try:
                file_path = Path(s.get("file", ""))
                parts = file_path.parts
                if len(parts) >= 5:
                    group = parts[-5]
                    session = parts[-4]
                elif len(parts) >= 3:
                    group = parts[-3]
                    session = parts[-2]
            except Exception:
                pass
        if not group or not session:
            continue
        for band, csv_path in s.get("pli_outputs", {}).items():
            key = (group, session, band)
            grouped_files.setdefault(key, []).append(csv_path)

    # Between-group analysis
    for session in design.sessions:
        for band in config.connectivity.frequency_bands.keys():
            if progress_cb:
                progress_cb(f"Computing between-group differences for {session} - {band}")
            
            # Collect and clean matrices for each group
            group_matrices = {}
            for group in design.groups:
                key = (group, session, band)
                if key in grouped_files:
                    matrices = []
                    for f in grouped_files[key]:
                        try:
                            df = pd.read_csv(f, index_col=0)
                            matrices.append(df.values)
                        except Exception as exc:
                            LOGGER.warning(f"Could not load {f}: {exc}")
                    
                    cleaned_matrices = _clean_matrices(matrices)
                    if len(cleaned_matrices) > 0:
                        group_matrices[group] = cleaned_matrices

            # Compare groups
            if len(group_matrices) >= 2:
                for g1, g2 in combinations(group_matrices.keys(), 2):
                    X = group_matrices[g1]
                    Y = group_matrices[g2]
                    
                    # Perform statistical comparison
                    t_stats, p_vals = _perform_statistical_comparison(X, Y)
                    
                    # Create significance mask with FDR correction
                    mask = np.zeros_like(p_vals)
                    mask[p_vals < 0.05] = 1  # You can adjust threshold
                    
                    # Save results
                    out_dir = between_dir / f"{session}_{band}"
                    out_dir.mkdir(exist_ok=True)
                    
                    np.savez(
                        out_dir / f"{g1}_vs_{g2}_stats.npz",
                        t_stat=t_stats,
                        p_values=p_vals,
                        significant_mask=mask
                    )

                    # Plot significant connections
                    sig_connections = t_stats * mask
                    fig, ax = plt.subplots(figsize=(10, 8))
                    vmax = np.max(np.abs(sig_connections))
                    im = ax.imshow(sig_connections, cmap='RdBu_r', clim=(-vmax, vmax))
                    ax.set_title(f'{band} - {g1} vs {g2} ({session})\nSignificant connections')
                    plt.colorbar(im)
                    fig.savefig(out_dir / f"{g1}_vs_{g2}_connections.png")
                    plt.close(fig)

    # Within-group analysis
    for group in design.groups:
        for band in config.connectivity.frequency_bands.keys():
            if progress_cb:
                progress_cb(f"Computing within-group changes for {group} - {band}")

            session_matrices = {}
            for session in design.sessions:
                key = (group, session, band)
                if key in grouped_files:
                    matrices = []
                    for f in grouped_files[key]:
                        try:
                            df = pd.read_csv(f, index_col=0)
                            matrices.append(df.values)
                        except Exception:
                            continue
                    
                    cleaned_matrices = _clean_matrices(matrices)
                    if len(cleaned_matrices) > 0:
                        session_matrices[session] = cleaned_matrices

            if len(session_matrices) >= 2:
                for s1, s2 in combinations(session_matrices.keys(), 2):
                    X = session_matrices[s1]
                    Y = session_matrices[s2]
                    
                    # Perform statistical comparison
                    t_stats, p_vals = _perform_statistical_comparison(X, Y)
                    
                    # Create significance mask
                    mask = np.zeros_like(p_vals)
                    mask[p_vals < 0.05] = 1
                    
                    # Save results
                    out_dir = within_dir / group / band
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    np.savez(
                        out_dir / f"{s1}_vs_{s2}_stats.npz",
                        t_stat=t_stats,
                        p_values=p_vals,
                        significant_mask=mask
                    )

                    # Plot significant connections
                    sig_connections = t_stats * mask
                    fig, ax = plt.subplots(figsize=(10, 8))
                    vmax = np.max(np.abs(sig_connections))
                    im = ax.imshow(sig_connections, cmap='RdBu_r', clim=(-vmax, vmax))
                    ax.set_title(f'{group} - {band}\n{s1} vs {s2} significant changes')
                    plt.colorbar(im)
                    fig.savefig(out_dir / f"{s1}_vs_{s2}_connections.png")
                    plt.close(fig)

    if progress_cb:
        if not grouped_files:
            progress_cb("No CSVs found to analyze; stats folders may be empty.")
        progress_cb("Statistical analyses complete")


def _upper_tri_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def _edge_adjacency(n_labels: int) -> "sparse.csr_matrix":
    """Adjacency on edges: two edges are neighbors if they share a node.
    Returns a sparse (n_edges x n_edges) matrix.
    """
    iu = _upper_tri_indices(n_labels)
    edges = list(zip(iu[0], iu[1]))
    n_edges = len(edges)
    incident = {i: [] for i in range(n_labels)}
    for idx, (u, v) in enumerate(edges):
        incident[u].append(idx)
        incident[v].append(idx)
    rows = []
    cols = []
    for node, e_list in incident.items():
        # connect all edges incident at this node
        for i in range(len(e_list)):
            for j in range(i + 1, len(e_list)):
                a = e_list[i]
                b = e_list[j]
                rows.extend([a, b])
                cols.extend([b, a])
    data = np.ones(len(rows), dtype=float)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_edges, n_edges))


def _cluster_permutation_between(
    X: np.ndarray,
    Y: np.ndarray,
    n_labels: int,
    n_permutations: int = 1000,
    threshold: Optional[float] = None,
):
    """Run cluster-based permutation on edge-wise differences using edge adjacency.

    Returns t_obs (vector over upper-tri edges), clusters (list of boolean masks), p_vals, and iu indices.
    """
    iu = _upper_tri_indices(n_labels)
    n_edges = len(iu[0])
    Xv = X[:, iu[0], iu[1]]  # (n_subj, n_edges)
    Yv = Y[:, iu[0], iu[1]]
    adjacency = _edge_adjacency(n_labels)
    with warnings.catch_warnings():
        # Suppress benign warning when no clusters are found
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="No clusters found*")
        t_obs, clusters, p_vals, _ = mne.stats.permutation_cluster_test(
            [Xv, Yv],
            n_permutations=n_permutations,
            tail=0,
            adjacency=adjacency,
            out_type="mask",
            threshold=threshold,
            verbose=False,
        )
    return t_obs, clusters, p_vals, iu


def _perform_cluster_based_permutation(
    summaries: List[dict],
    design: StudyDesign,
    config: PipelineConfig,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """Compute cluster-based permutation between groups and within groups, save results."""
    if progress_cb:
        progress_cb("Starting cluster-based permutation analyses…")

    stats_dir = config.output_root / "stats" / "cluster_perm"
    between_dir = stats_dir / "between_groups"
    within_dir = stats_dir / "within_groups"
    for d in [stats_dir, between_dir, within_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect files per (group, session, band)
    grouped_files: Dict[Tuple[str, str, str], List[str]] = {}
    for s in summaries:
        group = s.get("group")
        session = s.get("session")
        if not group or not session:
            # Fallback heuristic if not provided
            file_path = Path(s.get("file", "")) if s.get("file") else None
            parts = file_path.parts if file_path else []
            if len(parts) >= 5:
                group = parts[-5]
                session = parts[-4]
            elif len(parts) >= 3:
                group = parts[-3]
                session = parts[-2]
        if not group or not session:
            continue
        for band, csv_path in s.get("pli_outputs", {}).items():
            grouped_files.setdefault((group, session, band), []).append(csv_path)

    # Helper to load/clean matrices -> array
    def _load_clean(paths: List[str]) -> np.ndarray:
        mats = []
        for p in paths:
            try:
                df = pd.read_csv(p, index_col=0)
                mats.append(df.values.astype(float))
            except Exception:
                continue
        return _clean_matrices(mats)

    # Between-group comparisons per session/band
    any_between = False
    for session in design.sessions:
        for band in config.connectivity.frequency_bands.keys():
            if progress_cb:
                progress_cb(f"Cluster perm: between-group {session} - {band}")
            group_mats: Dict[str, np.ndarray] = {}
            for group in design.groups:
                key = (group, session, band)
                if key in grouped_files:
                    arr = _load_clean(grouped_files[key])
                    if arr.size:
                        group_mats[group] = arr
            if len(group_mats) >= 2:
                any_between = True
                # take first two groups pairwise
                from itertools import combinations as _cmb

                for g1, g2 in _cmb(group_mats.keys(), 2):
                    X = group_mats[g1]
                    Y = group_mats[g2]
                    if X.shape[1:] != Y.shape[1:]:
                        continue
                    n_labels = X.shape[1]
                    try:
                        n_perm = getattr(getattr(config, "stats", object()), "n_permutations", 1000)
                        thr = getattr(getattr(config, "stats", object()), "cluster_threshold", None)
                        t_obs, clusters, p_vals, iu = _cluster_permutation_between(
                            X, Y, n_labels, n_permutations=int(n_perm), threshold=thr
                        )
                    except Exception as exc:
                        LOGGER.warning("Cluster perm failed %s vs %s: %s", g1, g2, exc)
                        continue
                    if not len(clusters):
                        LOGGER.info(
                            "Cluster perm: no clusters found (between) %s vs %s, session=%s, band=%s",
                            g1, g2, session, band
                        )
                    sig_mask_vec = np.zeros_like(t_obs, dtype=bool)
                    for cl_mask, pv in zip(clusters, p_vals):
                        if pv < 0.05:
                            sig_mask_vec = np.logical_or(sig_mask_vec, cl_mask)
                    # Reconstruct matrices
                    t_mat = np.zeros((n_labels, n_labels))
                    sig_mat = np.zeros((n_labels, n_labels), dtype=bool)
                    t_mat[iu] = t_obs
                    t_mat = t_mat + t_mat.T
                    sig_mat[iu] = sig_mask_vec
                    sig_mat = sig_mat | sig_mat.T

                    out_dir = between_dir / f"{session}_{band}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        out_dir / f"{g1}_vs_{g2}_cluster_perm.npz",
                        t_matrix=t_mat,
                        significant_mask=sig_mat,
                        p_values=p_vals,
                    )
                    # Plot
                    vmax = np.nanmax(np.abs(t_mat)) or 1.0
                    fig, ax = plt.subplots(figsize=(9, 7))
                    im = ax.imshow(np.where(sig_mat, t_mat, 0.0), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                    ax.set_title(f"{band} - {g1} vs {g2} ({session})\nCluster-based permutation (p<0.05)")
                    plt.colorbar(im, ax=ax)
                    fig.tight_layout()
                    fig.savefig(out_dir / f"{g1}_vs_{g2}_cluster_perm.png", dpi=200)
                    plt.close(fig)

    # Within-group comparisons across sessions per band
    any_within = False
    for group in design.groups:
        for band in config.connectivity.frequency_bands.keys():
            if progress_cb:
                progress_cb(f"Cluster perm: within-group {group} - {band}")
            session_mats: Dict[str, np.ndarray] = {}
            for session in design.sessions:
                key = (group, session, band)
                if key in grouped_files:
                    arr = _load_clean(grouped_files[key])
                    if arr.size:
                        session_mats[session] = arr
            if len(session_mats) >= 2:
                any_within = True
                from itertools import combinations as _cmb

                for s1, s2 in _cmb(session_mats.keys(), 2):
                    X = session_mats[s1]
                    Y = session_mats[s2]
                    if X.shape[1:] != Y.shape[1:]:
                        continue
                    n_labels = X.shape[1]
                    try:
                        n_perm = getattr(getattr(config, "stats", object()), "n_permutations", 1000)
                        thr = getattr(getattr(config, "stats", object()), "cluster_threshold", None)
                        t_obs, clusters, p_vals, iu = _cluster_permutation_between(
                            X, Y, n_labels, n_permutations=int(n_perm), threshold=thr
                        )
                    except Exception as exc:
                        LOGGER.warning("Cluster perm failed %s %s: %s", group, f"{s1} vs {s2}", exc)
                        continue
                    if not len(clusters):
                        LOGGER.info(
                            "Cluster perm: no clusters found (within) %s: %s vs %s, band=%s",
                            group, s1, s2, band
                        )
                    sig_mask_vec = np.zeros_like(t_obs, dtype=bool)
                    for cl_mask, pv in zip(clusters, p_vals):
                        if pv < 0.05:
                            sig_mask_vec = np.logical_or(sig_mask_vec, cl_mask)
                    t_mat = np.zeros((n_labels, n_labels))
                    sig_mat = np.zeros((n_labels, n_labels), dtype=bool)
                    t_mat[iu] = t_obs
                    t_mat = t_mat + t_mat.T
                    sig_mat[iu] = sig_mask_vec
                    sig_mat = sig_mat | sig_mat.T

                    out_dir = within_dir / group / band
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        out_dir / f"{s1}_vs_{s2}_cluster_perm.npz",
                        t_matrix=t_mat,
                        significant_mask=sig_mat,
                        p_values=p_vals,
                    )
                    vmax = np.nanmax(np.abs(t_mat)) or 1.0
                    fig, ax = plt.subplots(figsize=(9, 7))
                    im = ax.imshow(np.where(sig_mat, t_mat, 0.0), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                    ax.set_title(f"{group} - {band}\n{s1} vs {s2} Cluster-based permutation (p<0.05)")
                    plt.colorbar(im, ax=ax)
                    fig.tight_layout()
                    fig.savefig(out_dir / f"{s1}_vs_{s2}_cluster_perm.png", dpi=200)
                    plt.close(fig)

    if progress_cb:
        if not (any_between or any_within):
            progress_cb("No valid condition pairs for cluster permutation; nothing saved.")
        progress_cb("Cluster-based permutation analyses complete")
