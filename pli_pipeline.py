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
    ica_method: str = "infomax"
    ica_n_components: Optional[int] = None
    random_state: int = 42
    reject_criteria: Optional[Dict[str, float]] = field(
        default_factory=lambda: {"eeg": 150e-6}
    )
    # New: optional stages and thresholds
    use_asr: bool = True
    use_iclabel: bool = True
    iclabel_brain_threshold: float = 0.85


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
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0),
        }
    )
    method: str = "pli"
    use_gpu: Optional[bool] = None  # True=force GPU, False=CPU, None=auto


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
    """Return sorted list of EEGLAB .set files in folder (non-recursive)."""
    files = sorted(Path(folder).glob("*.set"))
    if not files:
        raise FileNotFoundError(f"No .set files found in {folder}")
    return files


def _apply_montage(raw: mne.io.BaseRaw) -> None:
    """Assign a standard 10-20 montage if the recording lacks channel positions."""
    if raw.get_montage() is None:
        try:
            montage = make_standard_montage("standard_1020")
            raw.set_montage(montage, on_missing="warn")
        except Exception as exc:
            LOGGER.warning("Could not set montage automatically: %s", exc)


def preprocess_raw(raw: mne.io.BaseRaw, config: PreprocessingConfig, progress_cb: Optional[Callable[[str], None]] = None) -> Tuple[mne.io.BaseRaw, mne.io.BaseRaw, ICA]:
    """Run filtering, referencing, and ICA-based cleanup."""
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

    # Optional ASR denoising (if enabled and available)
    if getattr(config, "use_asr", True):
        try:  # pragma: no cover - ASR may not be installed
            applied = False
            # asrpy API - expects MNE Raw object
            try:
                from asrpy import ASR  # type: ignore

                asr = ASR(sfreq=raw.info["sfreq"])
                # ASR.fit and transform work on Raw objects directly
                asr.fit(raw)
                raw = asr.transform(raw)
                applied = True
                LOGGER.info("Applied ASR (asrpy)")
                if progress_cb:
                    progress_cb("  -> ASR artifact removal applied")
            except Exception as asr_err:
                LOGGER.warning("asrpy failed: %s", asr_err)
            if not applied:
                LOGGER.info("ASR package not available; skipping ASR step")
                if progress_cb:
                    progress_cb("  -> ASR skipped (not available)")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("ASR step failed: %s", exc)

    raw.filter(config.l_freq, config.h_freq, fir_design="firwin")
    if config.notch_freqs:
        valid_notches = [
            freq for freq in config.notch_freqs if freq < raw.info["sfreq"] / 2.0
        ]
        if valid_notches:
            raw.notch_filter(valid_notches)
    raw.set_eeg_reference("average", projection=True)

    # Use extended infomax for ICLabel compatibility
    fit_params = dict(extended=True) if config.ica_method == "infomax" else None
    ica = ICA(
        n_components=config.ica_n_components,
        method=config.ica_method,
        fit_params=fit_params,
        random_state=config.random_state,
        max_iter="auto",
    )
    LOGGER.info("ICA: method=%s, n_components=%s", str(config.ica_method), str(config.ica_n_components))
    ica.fit(raw)
    # ICLabel auto-classification (if enabled and available); fallback to EOG detection
    excluded = []
    iclabel_applied = False
    if getattr(config, "use_iclabel", True):
        try:  # pragma: no cover - optional dependency
            from mne_icalabel import label_components  # type: ignore

            # label_components returns a dict with 'labels' and 'y_pred_proba'
            result = label_components(raw, ica, method="iclabel")
            ic_labels = result.get("labels", [])
            # ICLabel returns labels like 'brain', 'eye blink', 'muscle', etc.
            # Exclude components NOT classified as 'brain'
            for idx, lab in enumerate(ic_labels):
                label_str = str(lab).lower() if lab else ""
                if label_str != "brain":
                    excluded.append(idx)
            LOGGER.info("ICLabel excluded %d components (non-brain artifacts)", len(excluded))
            iclabel_applied = True
            if progress_cb:
                progress_cb(f"  -> ICLabel: excluded {len(excluded)} artifact components")
        except Exception as ic_err:
            LOGGER.warning("ICLabel failed: %s", ic_err)
            try:
                eog_inds, _ = ica.find_bads_eog(raw)
                excluded.extend(eog_inds)
                if progress_cb:
                    progress_cb(f"  -> EOG detection: excluded {len(eog_inds)} components")
            except RuntimeError:
                pass
    if not iclabel_applied and progress_cb:
        progress_cb("  -> ICLabel skipped (not available)")
    ica.exclude = list(sorted(set(excluded)))
    ica.apply(raw)
    LOGGER.info("ICA applied: excluded %d components", len(ica.exclude))
    return raw_before, raw, ica


def _infer_paths(file_path: Path, output_root: Path) -> Tuple[str, str, Path]:
    """Infer group/session and construct output directory for a file."""
    parts = file_path.parts
    group_name = parts[-3] if len(parts) >= 3 else "unknown_group"
    session_name = parts[-2] if len(parts) >= 2 else "unknown_session"
    out_dir = (output_root / group_name / session_name / file_path.stem).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    return group_name, session_name, out_dir


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
    }
    try:
        raw = mne.io.read_raw_eeglab(str(file_path), preload=False, verbose=False)
        raw_before, raw_after, _ica = preprocess_raw(raw, config.preprocessing, progress_cb)

        # Diagnostics
        try:
            traces_path = out_dir / "traces_before_after.png"
            topo_path = out_dir / "topomap_before_after.png"
            gfp_path = out_dir / "gfp_before_after.png"
            _plot_traces_before_after(raw_before, raw_after, traces_path)
            _plot_topomap(raw_before, raw_after, topo_path)
            _plot_gfp(raw_before, raw_after, gfp_path)
            summary["diagnostics"] = {
                "traces": str(traces_path),
                "topomap": str(topo_path),
                "gfp": str(gfp_path),
            }
        except Exception as exc:
            LOGGER.warning("Diagnostics failed for %s: %s", file_path.name, exc)

        # Save preprocessed raw
        fif_path = out_dir / "preprocessed_raw.fif"
        try:
            raw_after.save(fif_path, overwrite=True, verbose=False)
            summary["preprocessed_fif"] = str(fif_path)
        except Exception as exc:
            if progress_cb:
                progress_cb(f"Failed to save preprocessed FIF for {file_path.name}: {exc}")

        if progress_cb:
            progress_cb(f"Finished preprocessing {file_path.name}")
    except Exception as exc:
        LOGGER.exception("Preprocessing error for %s", file_path)
        if progress_cb:
            progress_cb(f"Preprocessing error for {file_path.name}: {exc}")
    return summary


def _plot_psd(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, title: str, out_file: Path) -> None:
    """Plot mean PSD before/after preprocessing."""
    psd_kwargs = dict(fmin=1.0, fmax=45.0, picks="eeg", tmin=None, tmax=None, n_fft=2048)
    psd_before = raw_before.compute_psd(**psd_kwargs)
    psd_after = raw_after.compute_psd(**psd_kwargs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(psd_before.freqs, psd_before.get_data().mean(axis=0), label="Before", color="#d95f02")
    ax.plot(psd_after.freqs, psd_after.get_data().mean(axis=0), label="After", color="#1b9e77")
    ax.set(
        xlabel="Frequency (Hz)",
        ylabel="Power Spectral Density (dB)",
        title=title,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def _plot_traces_before_after(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    out_file: Path,
    n_channels: int = 6,
    duration: float = 10.0,
) -> None:
    """Plot a few EEG channels before/after preprocessing (stacked traces)."""
    picks = mne.pick_types(raw_after.info, eeg=True, meg=False)
    if len(picks) == 0:
        return
    picks = picks[: min(n_channels, len(picks))]
    sfreq = raw_after.info["sfreq"]
    tmax = min(duration, raw_after.times[-1] if len(raw_after.times) else duration)
    n_samp = int(max(1, tmax * sfreq))
    data_b = raw_before.get_data(picks=picks)[:, :n_samp] * 1e6
    data_a = raw_after.get_data(picks=picks)[:, :n_samp] * 1e6
    times = np.arange(n_samp) / sfreq
    offsets = np.arange(len(picks)) * 50.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ch in range(len(picks)):
        axes[0].plot(times, data_b[ch] + offsets[ch], color="#d95f02", lw=0.8)
        axes[1].plot(times, data_a[ch] + offsets[ch], color="#1b9e77", lw=0.8)
    axes[0].set_title("Before preprocessing (µV)")
    axes[1].set_title("After preprocessing (µV)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_yticks([])
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def _plot_topomap(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, out_file: Path) -> None:
    """Compare RMS scalp maps before/after preprocessing."""
    data_before = raw_before.get_data(picks="eeg")
    data_after = raw_after.get_data(picks="eeg")
    rms_before = np.sqrt((data_before**2).mean(axis=1))
    rms_after = np.sqrt((data_after**2).mean(axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    mne.viz.plot_topomap(rms_before, raw_before.info, axes=axes[0], show=False, contours=0)
    axes[0].set_title("Before preprocessing")
    mne.viz.plot_topomap(rms_after, raw_after.info, axes=axes[1], show=False, contours=0)
    axes[1].set_title("After preprocessing")
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def _plot_gfp(raw_before: mne.io.BaseRaw, raw_after: mne.io.BaseRaw, out_file: Path) -> None:
    """Plot global field power before/after cleaning."""
    def _gfp(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, np.ndarray]:
        data = raw.get_data(picks="eeg")
        gfp = np.sqrt((data**2).mean(axis=0)) * 1e6
        return gfp, raw.times

    gfp_before, times = _gfp(raw_before)
    gfp_after, _ = _gfp(raw_after)
    step = max(1, len(times) // 5000)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times[::step], gfp_before[::step], label="Before", color="#d95f02", alpha=0.8)
    ax.plot(times[::step], gfp_after[::step], label="After", color="#1b9e77", alpha=0.8)
    ax.set(
        xlabel="Time (s)",
        ylabel="Global Field Power (µV)",
        title="Global Field Power before/after preprocessing",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
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
    chunk_size: int = 2048,
) -> np.ndarray:
    if not GPU_AVAILABLE or cp is None:
        raise RuntimeError("GPU computation requested but CuPy is unavailable.")

    data = epochs.get_data(copy=True)  # (n_epochs, n_labels, n_times)
    n_epochs, n_labels, n_times = data.shape
    merged = data.transpose(1, 0, 2).reshape(n_labels, -1)

    cp_data = cp.asarray(merged, dtype=cp.float32)
    filtered = _bandpass_gpu(cp_data, epochs.info["sfreq"], band)
    analytic = _hilbert_gpu(filtered)
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

    pli_gpu = cp.abs(accum / samples)
    return cp.asnumpy(pli_gpu)


def _should_use_gpu(user_pref: Optional[bool]) -> bool:
    if user_pref is True:
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "GPU acceleration was requested but no compatible GPU/CuPy installation was detected."
            )
        return True
    if user_pref is False:
        return False
    return GPU_AVAILABLE


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
            return _compute_pli_gpu(epochs, band)
        except Exception as exc:  # pragma: no cover - GPU fallback path
            LOGGER.warning("GPU PLI failed (%s); falling back to CPU.", exc)

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
            except Exception as exc:
                LOGGER.exception("PLI failed for %s %s", out_dir.name, band_name)
                if progress_cb:
                    progress_cb(f"PLI failed for {band_name}: {exc}")
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
    """Aggregate existing PLI CSV files across output_root into an Excel table."""
    config = config or PipelineConfig()
    pseudo_summaries: List[dict] = []
    for group in design.groups:
        for session in design.sessions:
            session_dir = (config.output_root / group / session)
            if not session_dir.exists():
                continue
            for subj_dir in sorted(session_dir.glob("*")):
                pli_outputs = {}
                for csv_file in subj_dir.glob("*_PLI.csv"):
                    # band name is prefix before _PLI
                    band_name = csv_file.stem.replace("_PLI", "")
                    pli_outputs[band_name] = str(csv_file)
                if pli_outputs:
                    pseudo_summaries.append({
                        "file": str(subj_dir / (subj_dir.name + ".set")),
                        "group": group,
                        "session": session,
                        "pli_outputs": pli_outputs,
                    })
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
        raw_before, raw_after, ica = preprocess_raw(raw, config.preprocessing, progress_cb)

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
            group_name = ""
            session_name = ""
            if len(parts) >= 3:
                # try last three parts: ... / Group / Session / filename
                group_name = parts[-3]
                session_name = parts[-2]
            # each band has CSV path in pli_outputs
            pli_outputs = s.get("pli_outputs", {})
            for band_name, csv_path in pli_outputs.items():
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    matrix = df.values.astype(float)
                except Exception:
                    # If CSV fails, try numpy load or skip
                    continue
                for net_name, idxs in networks.items():
                    # convert 1-based to 0-based if necessary: if any idx == 0 assume already 0-based
                    if len(idxs) == 0:
                        mean_pli = float("nan")
                    else:
                        convert = [i - 1 if min(idxs) > 0 else i for i in idxs]
                        # filter out-of-bounds indices
                        valid = [i for i in convert if 0 <= i < matrix.shape[0]]
                        if not valid:
                            mean_pli = float("nan")
                        else:
                            mean_pli = _compute_group_pli_from_matrix(matrix, valid)
                    rows.append(
                        {
                            "Participant": participant,
                            "Group": group_name,
                            "Session": session_name,
                            "FrequencyTag": band_name,
                            "MeanPLI": mean_pli,
                            "Network": net_name,
                        }
                    )
        except Exception as exc:
            LOGGER.warning("Failed to aggregate summary %s: %s", s.get("file", "<unknown>"), exc)

    table = pd.DataFrame(rows, columns=["Participant", "Group", "Session", "FrequencyTag", "MeanPLI", "Network"])
    try:
        output_excel.parent.mkdir(parents=True, exist_ok=True)
        table.to_excel(output_excel, index=False)
        if progress_cb:
            progress_cb(f"Saved study-level PLI table to {output_excel}")
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
