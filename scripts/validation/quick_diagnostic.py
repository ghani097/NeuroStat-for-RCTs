#!/usr/bin/env python
"""
Quick diagnostic script -- loads simulated EEG and produces three plots:

1. Raw EEG traces (3 seconds from each artifact level)
2. Power spectral density (showing planted oscillator peaks)
3. Sensor-level connectivity matrix for the clean dataset (alpha band)

Usage
-----
    python scripts/validation/quick_diagnostic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.channels import make_standard_montage

_MONTAGE = make_standard_montage("biosemi64")


def _load_set(fpath):
    """Load .set and re-apply biosemi64 montage to suppress head-radius warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*head radius.*", category=RuntimeWarning)
        raw = mne.io.read_raw_eeglab(str(fpath), preload=True, verbose=False)
    raw.set_montage(_MONTAGE, on_missing="ignore", verbose=False)
    return raw


# ── paths ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
SIM_DIR = BASE / "simulated_data"
OUT_DIR = BASE / "validation_data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("clean_high_snr",      "Clean (10 dB)"),
    ("light_medium_snr",    "Light (5 dB)"),
    ("moderate_medium_snr", "Moderate (5 dB)"),
    ("heavy_low_snr",       "Heavy (2 dB)"),
]

# ── Load ground truth ──────────────────────────────────────────────────
gt_path = SIM_DIR / "ground_truth.json"
with open(gt_path) as fh:
    gt = json.load(fh)

patterns = gt["connectivity_patterns"]


# ======================================================================
# Figure A: Raw EEG traces (first 3 s, selected channels)
# ======================================================================
def plot_raw_traces():
    """4-panel plot showing 3 s of raw EEG for each artifact level."""
    show_chs = ["Fp1", "AF3", "C3", "P3", "O1", "T7"]
    t_start, t_end = 5.0, 8.0  # seconds

    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(14, 3.2 * len(DATASETS)),
                             sharex=True)

    for idx, (level, label) in enumerate(DATASETS):
        ax = axes[idx]
        fpath = SIM_DIR / f"{level}.set"
        raw = _load_set(fpath)

        picks = [raw.ch_names.index(ch) for ch in show_chs if ch in raw.ch_names]
        sfreq = raw.info["sfreq"]
        s0, s1 = int(t_start * sfreq), int(t_end * sfreq)
        data = raw.get_data(picks=picks)[:, s0:s1] * 1e6  # V -> uV
        times = np.arange(data.shape[1]) / sfreq + t_start

        spacing = 40  # uV between channels
        for i, ch_idx in enumerate(picks):
            ch_name = raw.ch_names[ch_idx]
            trace = data[i] + i * spacing
            ax.plot(times, trace, linewidth=0.5, label=ch_name)

        ax.set_ylabel("Channels", fontsize=9)
        ax.set_yticks([i * spacing for i in range(len(picks))])
        ax.set_yticklabels([raw.ch_names[p] for p in picks], fontsize=8)
        ax.set_title(f"{label}", fontsize=11, fontweight="bold", loc="left")
        ax.set_xlim(t_start, t_end)
        ax.grid(alpha=0.15)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    fig.suptitle("Raw EEG Traces (3-second window)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "diagnostic_raw_traces.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ======================================================================
# Figure B: Power Spectral Density
# ======================================================================
def plot_psd():
    """PSD for clean and heavy datasets showing planted oscillator peaks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (level, label) in zip(axes, [DATASETS[0], DATASETS[-1]]):
        fpath = SIM_DIR / f"{level}.set"
        raw = _load_set(fpath)

        # Compute PSD using Welch
        psd = raw.compute_psd(method="welch", fmin=0.5, fmax=45, n_fft=1024,
                              verbose=False)
        freqs = psd.freqs
        psd_data = psd.get_data() * 1e12  # V^2/Hz -> uV^2/Hz

        # Mean across channels
        mean_psd = psd_data.mean(axis=0)
        ax.semilogy(freqs, mean_psd, "k-", linewidth=0.8, alpha=0.6, label="Mean (all ch)")

        # Highlight channels involved in each pattern
        colors = ["#E74C3C", "#3498DB", "#2ECC71"]
        for pat, color in zip(patterns, colors):
            chs = pat["region1_channels"] + pat["region2_channels"]
            ch_psd = psd_data[chs].mean(axis=0)
            ax.semilogy(freqs, ch_psd, color=color, linewidth=1.2,
                        label=f"{pat['name'].replace('_', ' ').title()} chs")
            # Mark the peak frequency
            ax.axvline(pat["frequency_hz"], color=color, linestyle="--",
                       alpha=0.5, linewidth=0.8)

        ax.set_xlabel("Frequency (Hz)", fontsize=10)
        ax.set_ylabel("PSD (uV^2/Hz)", fontsize=10)
        ax.set_title(f"{label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(0.5, 45)
        ax.grid(alpha=0.2, which="both")

    fig.suptitle("Power Spectral Density -- Planted Oscillator Peaks",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "diagnostic_psd.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ======================================================================
# Figure C: Sensor-level connectivity matrix (alpha band, clean dataset)
# ======================================================================
def plot_connectivity_matrix():
    """Compute and plot sensor-level PLI matrix for the alpha band (clean data)."""
    try:
        from mne.connectivity import spectral_connectivity_epochs
    except ImportError:
        from mne_connectivity import spectral_connectivity_epochs
    from mne import EpochsArray

    fpath = SIM_DIR / "clean_high_snr.set"
    raw = _load_set(fpath)

    # Epoch the data (4 s, 50% overlap)
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    n_ch, n_total = data.shape
    spe = int(4.0 * sfreq)
    step = int(spe * 0.5)
    starts = list(range(0, n_total - spe + 1, step))
    epochs_data = np.empty((len(starts), n_ch, spe), dtype=data.dtype)
    for i, s in enumerate(starts):
        epochs_data[i] = data[:, s : s + spe]

    info = mne.create_info(ch_names=raw.ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(epochs_data, info, verbose=False)

    # Compute PLI in alpha band
    con = spectral_connectivity_epochs(
        epochs,
        method="pli",
        mode="multitaper",
        fmin=8.0,
        fmax=13.0,
        faverage=True,
        verbose=False,
    )
    pli_matrix = con.get_data(output="dense")[:, :, 0]
    pli_matrix = np.maximum(pli_matrix, pli_matrix.T)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5),
                             gridspec_kw={"width_ratios": [1, 1]})

    # Full matrix
    ax = axes[0]
    im = ax.imshow(pli_matrix, cmap="hot", vmin=0, vmax=1, aspect="equal")
    ax.set_title("Alpha PLI (8-13 Hz) -- All 64 channels", fontsize=11, fontweight="bold")
    ax.set_xlabel("Channel", fontsize=9)
    ax.set_ylabel("Channel", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("PLI", fontsize=9)

    # Highlight planted pattern channels
    r1 = gt["connectivity_patterns"][0]["region1_channels"]  # frontal
    r2 = gt["connectivity_patterns"][0]["region2_channels"]  # parietal
    for i in r1:
        for j in r2:
            ax.plot(j, i, "s", markersize=3, markeredgecolor="cyan",
                    markerfacecolor="none", markeredgewidth=1.0)
            ax.plot(i, j, "s", markersize=3, markeredgecolor="cyan",
                    markerfacecolor="none", markeredgewidth=1.0)

    # Zoomed sub-matrix: frontal vs parietal
    ax2 = axes[1]
    all_chs = r1 + r2
    sub = pli_matrix[np.ix_(all_chs, all_chs)]
    ch_labels = [raw.ch_names[i] for i in all_chs]

    im2 = ax2.imshow(sub, cmap="hot", vmin=0, vmax=1, aspect="equal")
    ax2.set_xticks(range(len(ch_labels)))
    ax2.set_xticklabels(ch_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks(range(len(ch_labels)))
    ax2.set_yticklabels(ch_labels, fontsize=8)
    ax2.set_title("Zoomed: Frontal-Parietal Alpha Pattern\n(expected PLI = 0.80)",
                  fontsize=11, fontweight="bold")

    # Annotate cell values
    for i in range(len(ch_labels)):
        for j in range(len(ch_labels)):
            val = sub[i, j]
            color = "white" if val > 0.5 else "black"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=7, color=color)

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("PLI", fontsize=9)

    # Add a line separating frontal from parietal in the zoomed view
    n_frontal = len(r1)
    ax2.axhline(n_frontal - 0.5, color="cyan", linewidth=1.5, linestyle="--")
    ax2.axvline(n_frontal - 0.5, color="cyan", linewidth=1.5, linestyle="--")
    ax2.text(-0.5, n_frontal / 2 - 0.5, "Frontal", fontsize=8, color="cyan",
             ha="right", va="center", fontweight="bold")
    ax2.text(-0.5, n_frontal + (len(r2) / 2) - 0.5, "Parietal", fontsize=8,
             color="cyan", ha="right", va="center", fontweight="bold")

    fig.suptitle("Sensor-Level Connectivity -- Clean Dataset",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "diagnostic_connectivity.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  NeuroStat Validation -- Quick Diagnostic")
    print("=" * 60)
    print(f"  Simulated data: {SIM_DIR}")
    print(f"  Output:         {OUT_DIR}")
    print()

    print("  Figure A: Raw EEG traces ...")
    plot_raw_traces()

    print("  Figure B: Power spectral density ...")
    plot_psd()

    print("  Figure C: Connectivity matrix ...")
    plot_connectivity_matrix()

    print()
    print("=" * 60)
    print("  Diagnostic complete! Check:")
    print(f"    {OUT_DIR / 'diagnostic_raw_traces.png'}")
    print(f"    {OUT_DIR / 'diagnostic_psd.png'}")
    print(f"    {OUT_DIR / 'diagnostic_connectivity.png'}")
    print("=" * 60)
