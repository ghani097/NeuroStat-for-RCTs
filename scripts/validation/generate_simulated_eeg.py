"""
Generate simulated EEG datasets with known connectivity for pipeline validation.

Creates 4 EEGLAB .set files at different artifact severity levels, each
containing three planted connectivity patterns (alpha, beta, theta coupling
between specific channel groups). Ground truth PLI values and pattern
specifications are saved to ground_truth.json.

Usage:
    python scripts/validation/generate_simulated_eeg.py [--output-dir PATH]

Reference:
    Designed for validating the NeuroStat PLI pipeline as described in:
    "NeuroStat: An Automated EEG Processing Pipeline for Source-Localized
     Phase Lag Index Connectivity Analysis in Randomized Controlled Trials"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import firwin, filtfilt

import mne
from mne.channels import make_standard_montage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SFREQ = 250.0           # Sampling rate (Hz)
DURATION = 120.0         # Recording duration (seconds)
N_CHANNELS = 64          # Number of EEG channels
MONTAGE_NAME = "biosemi64"  # Standard 64-channel montage
RNG_SEED = 42            # Reproducibility

# Frequency bands matching the pipeline
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# ---------------------------------------------------------------------------
# Connectivity pattern definitions
# ---------------------------------------------------------------------------

# Biosemi64 channel order (from montage):
#  [ 0] Fp1   [ 1] AF7   [ 2] AF3   [ 3] F1    [ 4] F3    [ 5] F5
#  [ 6] F7    [ 7] FT7   [ 8] FC5   [ 9] FC3   [10] FC1   [11] C1
#  [12] C3    [13] C5    [14] T7    [15] TP7    [16] CP5   [17] CP3
#  [18] CP1   [19] P1    [20] P3    [21] P5    [22] P7    [23] P9
#  [24] PO7   [25] PO3   [26] O1    [27] Iz    [28] Oz    [29] POz
#  [30] Pz    [31] CPz   [32] Fpz   [33] Fp2   [34] AF8   [35] AF4
#  [36] AFz   [37] Fz    [38] F2    [39] F4    [40] F6    [41] F8
#  [42] FT8   [43] FC6   [44] FC4   [45] FC2   [46] FCz   [47] Cz
#  [48] C2    [49] C4    [50] C6    [51] T8    [52] TP8   [53] CP6
#  [54] CP4   [55] CP2   [56] P2    [57] P4    [58] P6    [59] P8
#  [60] P10   [61] PO8   [62] PO4   [63] O2

CONNECTIVITY_PATTERNS = [
    {
        "name": "alpha_frontal_parietal",
        "description": "Strong alpha coupling: frontal <-> parietal",
        "region1_channels": [0, 1, 2, 3],       # Fp1, AF7, AF3, F1
        "region2_channels": [19, 20, 21, 22],    # P1, P3, P5, P7
        "frequency_hz": 10.0,
        "band": "alpha",
        "expected_pli": 0.8,
        "phase_lag_rad": np.pi / 4,
        "coupling_strength": "strong",
    },
    {
        "name": "beta_central_occipital",
        "description": "Moderate beta coupling: central <-> occipital",
        "region1_channels": [11, 12, 48, 49],   # C1, C3, C2, C4
        "region2_channels": [26, 28, 63],         # O1, Oz, O2
        "frequency_hz": 20.0,
        "band": "beta",
        "expected_pli": 0.5,
        "phase_lag_rad": np.pi / 3,
        "coupling_strength": "moderate",
    },
    {
        "name": "theta_left_right_temporal",
        "description": "Weak theta coupling: left temporal <-> right temporal",
        "region1_channels": [13, 14],            # C5, T7
        "region2_channels": [50, 51],             # C6, T8
        "frequency_hz": 6.0,
        "band": "theta",
        "expected_pli": 0.3,
        "phase_lag_rad": np.pi / 6,
        "coupling_strength": "weak",
    },
]

# Artifact severity levels
ARTIFACT_LEVELS = {
    "clean_high_snr": {
        "label": "Clean (SNR=10dB)",
        "snr_db": 10.0,
        "blinks_per_min": 0,
        "muscle_bursts_per_min": 0,
        "line_noise_amplitude_uv": 0.0,
    },
    "light_medium_snr": {
        "label": "Light artifacts (SNR=5dB)",
        "snr_db": 5.0,
        "blinks_per_min": 10,
        "muscle_bursts_per_min": 5,
        "line_noise_amplitude_uv": 2.0,
    },
    "moderate_medium_snr": {
        "label": "Moderate artifacts (SNR=5dB)",
        "snr_db": 5.0,
        "blinks_per_min": 30,
        "muscle_bursts_per_min": 15,
        "line_noise_amplitude_uv": 5.0,
    },
    "heavy_low_snr": {
        "label": "Heavy artifacts (SNR=2dB)",
        "snr_db": 2.0,
        "blinks_per_min": 50,
        "muscle_bursts_per_min": 30,
        "line_noise_amplitude_uv": 10.0,
    },
}

# Frontal channel indices for eye blink artifacts (biosemi64)
# Fp1, AF7, AF3, Fp2, AF8, AF4, Fpz-like positions
FRONTAL_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7]

# Temporal/peripheral channels for muscle artifacts
TEMPORAL_CHANNELS = [5, 6, 7, 13, 14, 15, 16, 24, 25, 40, 41, 42]


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_pink_noise(n_samples: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f (pink) noise via spectral shaping.

    Parameters
    ----------
    n_samples : int
        Number of time samples.
    n_channels : int
        Number of channels.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    noise : ndarray, shape (n_channels, n_samples)
        Pink noise with unit variance per channel.
    """
    # Generate white noise in frequency domain and shape with 1/sqrt(f)
    n_fft = n_samples
    white = rng.standard_normal((n_channels, n_fft)) + \
            1j * rng.standard_normal((n_channels, n_fft))

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SFREQ)
    freqs[0] = 1.0  # avoid division by zero
    spectrum_shape = 1.0 / np.sqrt(freqs)
    spectrum_shape[0] = 0.0  # remove DC

    # Apply spectral shaping
    white_rfft = np.fft.rfft(rng.standard_normal((n_channels, n_fft)), axis=-1)
    shaped = white_rfft * spectrum_shape[np.newaxis, :]
    noise = np.fft.irfft(shaped, n=n_fft, axis=-1)

    # Normalize to unit variance per channel
    std = noise.std(axis=-1, keepdims=True)
    std[std == 0] = 1.0
    noise /= std

    return noise


def generate_coupled_oscillators(
    n_samples: int,
    freq_hz: float,
    phase_lag_rad: float,
    coupling_strength: float,
    region1_channels: List[int],
    region2_channels: List[int],
    n_total_channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate coupled oscillatory signals between two channel groups.

    Uses Kuramoto-style phase coupling: region2 oscillates at a fixed phase
    offset from region1, with jitter controlled by coupling_strength.

    Parameters
    ----------
    n_samples : int
        Number of time samples.
    freq_hz : float
        Center frequency of the oscillation.
    phase_lag_rad : float
        Constant phase lag from region1 to region2.
    coupling_strength : float
        PLI-like coupling target (0-1). Controls phase jitter: higher
        values produce less jitter and higher PLI.
    region1_channels, region2_channels : list of int
        Channel indices for each coupled region.
    n_total_channels : int
        Total number of channels in the dataset.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    signal : ndarray, shape (n_total_channels, n_samples)
        Oscillatory signal (zeros on non-participating channels).
    """
    t = np.arange(n_samples) / SFREQ
    signal = np.zeros((n_total_channels, n_samples))

    # Phase jitter inversely related to coupling strength
    # Higher coupling_strength -> less jitter -> higher PLI
    # PLI ≈ |mean(sign(phase_diff))| ; for constant lag, PLI=1
    # Add Brownian phase noise to reduce PLI to target
    if coupling_strength >= 0.95:
        phase_noise_std = 0.05
    elif coupling_strength >= 0.7:
        # For PLI~0.8, need moderate jitter
        phase_noise_std = 0.4
    elif coupling_strength >= 0.4:
        # For PLI~0.5, need more jitter (calibrated empirically)
        phase_noise_std = 0.45
    else:
        # For PLI~0.3, need substantial jitter
        phase_noise_std = 1.3

    # Generate base phase for region 1 (common reference + small per-channel jitter)
    base_phase = 2 * np.pi * freq_hz * t

    # Region 1: oscillate at the target frequency with small intra-region jitter
    for ch in region1_channels:
        ch_jitter = rng.standard_normal(n_samples) * 0.05
        ch_phase_offset = rng.uniform(0, 0.1)
        signal[ch] = np.sin(base_phase + ch_jitter + ch_phase_offset)

    # Region 2: oscillate at same frequency with controlled phase lag
    # Use cumulative random walk for phase noise (Brownian motion)
    phase_noise = np.cumsum(rng.standard_normal(n_samples) * phase_noise_std / np.sqrt(SFREQ))
    # High-pass the phase noise to prevent drift from dominating
    if len(phase_noise) > int(SFREQ):
        # Simple detrending
        from scipy.signal import detrend
        phase_noise = detrend(phase_noise)
        # Scale to target variance
        if phase_noise.std() > 0:
            phase_noise *= phase_noise_std / phase_noise.std()

    for ch in region2_channels:
        ch_jitter = rng.standard_normal(n_samples) * 0.05
        ch_phase_offset = rng.uniform(0, 0.1)
        signal[ch] = np.sin(base_phase + phase_lag_rad + phase_noise + ch_jitter + ch_phase_offset)

    # Amplitude modulation: sinusoidal envelope to make it more EEG-like
    envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 0.1 * t)
    all_channels = region1_channels + region2_channels
    for ch in all_channels:
        signal[ch] *= envelope

    return signal


def scale_to_snr(signal: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Scale noise to achieve a target SNR relative to the signal.

    Parameters
    ----------
    signal : ndarray, shape (n_channels, n_samples)
        Clean signal component.
    noise : ndarray, shape (n_channels, n_samples)
        Noise component to be scaled.
    target_snr_db : float
        Desired SNR in decibels.

    Returns
    -------
    scaled_noise : ndarray
        Noise scaled to achieve target SNR.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0 or signal_power == 0:
        return noise

    target_snr_linear = 10 ** (target_snr_db / 10)
    scale = np.sqrt(signal_power / (noise_power * target_snr_linear))
    return noise * scale


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------

def generate_eye_blinks(
    n_samples: int,
    n_channels: int,
    blinks_per_min: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate eye blink artifacts on frontal channels.

    Blinks are modeled as Gaussian-shaped voltage deflections (amplitude
    ~100 uV) centered on frontal electrodes with spatial falloff.

    Parameters
    ----------
    n_samples : int
        Number of time samples.
    n_channels : int
        Total number of channels.
    blinks_per_min : int
        Average number of blinks per minute.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    blink_signal : ndarray, shape (n_channels, n_samples)
        Blink artifact signal in volts.
    """
    if blinks_per_min == 0:
        return np.zeros((n_channels, n_samples))

    duration_sec = n_samples / SFREQ
    n_blinks = int(blinks_per_min * duration_sec / 60)
    artifact = np.zeros((n_channels, n_samples))

    # Blink parameters
    blink_duration_samples = int(0.3 * SFREQ)  # ~300 ms per blink
    half_width = blink_duration_samples // 2

    # Generate blink times (uniformly distributed, avoiding edges)
    margin = blink_duration_samples
    blink_centers = rng.integers(margin, n_samples - margin, size=n_blinks)

    for center in blink_centers:
        # Gaussian-shaped blink template
        amplitude = rng.uniform(80e-6, 120e-6)  # 80-120 uV in Volts
        t_blink = np.arange(-half_width, half_width + 1)
        sigma = half_width / 3  # 3-sigma covers the blink
        template = amplitude * np.exp(-0.5 * (t_blink / sigma) ** 2)

        start = max(0, center - half_width)
        end = min(n_samples, center + half_width + 1)
        t_start = start - (center - half_width)
        t_end = t_start + (end - start)

        # Apply to frontal channels with spatial falloff
        for ch_idx in range(n_channels):
            if ch_idx in FRONTAL_CHANNELS:
                # Full amplitude on frontal channels
                spatial_weight = 1.0
            elif ch_idx < 16:
                # Moderate amplitude on anterior channels
                spatial_weight = 0.3
            else:
                # Small amplitude on posterior channels
                spatial_weight = 0.05

            artifact[ch_idx, start:end] += spatial_weight * template[t_start:t_end]

    return artifact


def generate_muscle_artifacts(
    n_samples: int,
    n_channels: int,
    bursts_per_min: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate muscle (EMG) artifact bursts.

    Muscle artifacts are modeled as high-frequency bandpass-filtered noise
    bursts (20-100 Hz) on temporal/peripheral channels.

    Parameters
    ----------
    n_samples : int
        Number of time samples.
    n_channels : int
        Total number of channels.
    bursts_per_min : int
        Average number of muscle bursts per minute.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    muscle_signal : ndarray, shape (n_channels, n_samples)
        Muscle artifact signal in volts.
    """
    if bursts_per_min == 0:
        return np.zeros((n_channels, n_samples))

    duration_sec = n_samples / SFREQ
    n_bursts = int(bursts_per_min * duration_sec / 60)
    artifact = np.zeros((n_channels, n_samples))

    # Burst parameters
    burst_duration_range = (int(0.1 * SFREQ), int(0.5 * SFREQ))  # 100-500 ms

    # Bandpass filter for muscle frequency range (20-100 Hz)
    nyq = SFREQ / 2
    high_cutoff = min(100.0, nyq - 1)
    numtaps = min(101, int(SFREQ) - 1)
    if numtaps % 2 == 0:
        numtaps += 1
    bp_filter = firwin(numtaps, [20.0 / nyq, high_cutoff / nyq], pass_zero=False)

    for _ in range(n_bursts):
        burst_len = rng.integers(burst_duration_range[0], burst_duration_range[1])
        amplitude = rng.uniform(20e-6, 50e-6)  # 20-50 uV

        # Random start position
        start = rng.integers(0, max(1, n_samples - burst_len))
        end = min(start + burst_len, n_samples)
        actual_len = end - start

        # Generate burst on random subset of temporal channels
        n_affected = max(1, rng.integers(1, min(len(TEMPORAL_CHANNELS), 6)))
        affected = rng.choice(TEMPORAL_CHANNELS, size=n_affected, replace=False)

        for ch in affected:
            # Generate high-frequency noise burst
            noise_burst_full = rng.standard_normal(actual_len + 2 * numtaps)
            if len(noise_burst_full) > 3 * numtaps:
                filtered_burst = filtfilt(bp_filter, [1.0], noise_burst_full)
                burst = filtered_burst[numtaps:numtaps + actual_len]
            else:
                # Too short for filter, use raw high-freq noise
                burst = noise_burst_full[:actual_len]

            # Smooth onset/offset with Tukey window
            window = np.ones(actual_len)
            ramp = min(int(0.05 * SFREQ), actual_len // 4)
            if ramp > 0:
                window[:ramp] = np.linspace(0, 1, ramp)
                window[-ramp:] = np.linspace(1, 0, ramp)

            burst *= window
            if burst.std() > 0:
                burst = burst / burst.std() * amplitude

            artifact[ch, start:end] += burst

    return artifact


def generate_line_noise(
    n_samples: int,
    n_channels: int,
    amplitude_uv: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate 50 Hz power line noise (plus harmonics).

    Parameters
    ----------
    n_samples : int
        Number of time samples.
    n_channels : int
        Total number of channels.
    amplitude_uv : float
        Peak amplitude of the 50 Hz component in microvolts.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    line_signal : ndarray, shape (n_channels, n_samples)
        Line noise signal in volts.
    """
    if amplitude_uv == 0:
        return np.zeros((n_channels, n_samples))

    t = np.arange(n_samples) / SFREQ
    artifact = np.zeros((n_channels, n_samples))

    amplitude_v = amplitude_uv * 1e-6  # Convert to volts

    for ch in range(n_channels):
        # 50 Hz fundamental with random phase per channel
        phase_50 = rng.uniform(0, 2 * np.pi)
        line_50 = amplitude_v * np.sin(2 * np.pi * 50 * t + phase_50)

        # 100 Hz harmonic at half amplitude (if within Nyquist)
        if 100 < SFREQ / 2:
            phase_100 = rng.uniform(0, 2 * np.pi)
            line_100 = (amplitude_v * 0.5) * np.sin(2 * np.pi * 100 * t + phase_100)
        else:
            line_100 = 0.0

        # Small amplitude modulation for realism
        am = 1.0 + 0.05 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 2 * np.pi))

        artifact[ch] = (line_50 + line_100) * am

    return artifact


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def generate_dataset(
    artifact_level: str,
    output_dir: Path,
    rng: np.random.Generator,
) -> Dict:
    """Generate a single simulated EEG dataset.

    Parameters
    ----------
    artifact_level : str
        Key from ARTIFACT_LEVELS dict.
    output_dir : Path
        Directory to save the .set file.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    info_dict : dict
        Dataset metadata including ground truth patterns.
    """
    params = ARTIFACT_LEVELS[artifact_level]
    n_samples = int(DURATION * SFREQ)

    print(f"\n{'='*60}")
    print(f"Generating: {artifact_level} — {params['label']}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Generate coupled oscillatory signals (the "ground truth")
    # ------------------------------------------------------------------
    print("  [1/5] Generating coupled oscillatory signals...")
    signal = np.zeros((N_CHANNELS, n_samples))

    pattern_info = []
    for pattern in CONNECTIVITY_PATTERNS:
        osc = generate_coupled_oscillators(
            n_samples=n_samples,
            freq_hz=pattern["frequency_hz"],
            phase_lag_rad=pattern["phase_lag_rad"],
            coupling_strength=pattern["expected_pli"],
            region1_channels=pattern["region1_channels"],
            region2_channels=pattern["region2_channels"],
            n_total_channels=N_CHANNELS,
            rng=rng,
        )
        # Scale oscillation amplitude to realistic EEG range (~10-20 uV)
        osc_amplitude = 15e-6  # 15 uV in Volts
        osc_std = osc[osc != 0].std() if np.any(osc != 0) else 1.0
        osc = osc / osc_std * osc_amplitude
        signal += osc

        pattern_info.append({
            "name": pattern["name"],
            "description": pattern["description"],
            "region1_channels": pattern["region1_channels"],
            "region2_channels": pattern["region2_channels"],
            "frequency_hz": pattern["frequency_hz"],
            "band": pattern["band"],
            "expected_pli": pattern["expected_pli"],
            "phase_lag_rad": pattern["phase_lag_rad"],
        })

    # ------------------------------------------------------------------
    # 2. Generate 1/f background noise
    # ------------------------------------------------------------------
    print("  [2/5] Generating 1/f background noise...")
    pink = generate_pink_noise(n_samples, N_CHANNELS, rng)

    # Scale noise to achieve target SNR
    scaled_noise = scale_to_snr(signal, pink, params["snr_db"])
    data = signal + scaled_noise

    # ------------------------------------------------------------------
    # 3. Add artifacts
    # ------------------------------------------------------------------
    print("  [3/5] Adding artifacts...")

    # Eye blinks
    blinks = generate_eye_blinks(
        n_samples, N_CHANNELS,
        params["blinks_per_min"], rng
    )
    if params["blinks_per_min"] > 0:
        print(f"         Blinks: {params['blinks_per_min']}/min "
              f"({int(params['blinks_per_min'] * DURATION / 60)} total)")

    # Muscle bursts
    muscle = generate_muscle_artifacts(
        n_samples, N_CHANNELS,
        params["muscle_bursts_per_min"], rng
    )
    if params["muscle_bursts_per_min"] > 0:
        print(f"         Muscle: {params['muscle_bursts_per_min']}/min "
              f"({int(params['muscle_bursts_per_min'] * DURATION / 60)} total)")

    # Line noise
    line = generate_line_noise(
        n_samples, N_CHANNELS,
        params["line_noise_amplitude_uv"], rng
    )
    if params["line_noise_amplitude_uv"] > 0:
        print(f"         Line noise: {params['line_noise_amplitude_uv']} uV @ 50 Hz")

    data += blinks + muscle + line

    # ------------------------------------------------------------------
    # 4. Create MNE Raw object with montage
    # ------------------------------------------------------------------
    print("  [4/5] Creating MNE Raw object with biosemi64 montage...")

    montage = make_standard_montage(MONTAGE_NAME)
    ch_names = montage.ch_names[:N_CHANNELS]

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=SFREQ,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(montage, verbose=False)

    # Set measurement date for EEGLAB compatibility
    raw.set_meas_date(None)

    # ------------------------------------------------------------------
    # 5. Export as EEGLAB .set file
    # ------------------------------------------------------------------
    print("  [5/5] Exporting EEGLAB .set file...")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_path = output_dir / f"{artifact_level}.set"
    raw.export(str(set_path), overwrite=True, verbose=False)
    print(f"         Saved: {set_path}")

    # Verify file
    file_size_mb = set_path.stat().st_size / (1024 * 1024)
    print(f"         Size: {file_size_mb:.1f} MB")

    # Compute actual SNR for verification
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(scaled_noise ** 2)
    actual_snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    artifact_power = np.mean((blinks + muscle + line) ** 2)
    total_noise_power = noise_power + artifact_power
    effective_snr = 10 * np.log10(signal_power / total_noise_power) if total_noise_power > 0 else float('inf')

    print(f"         Actual SNR (signal vs pink noise): {actual_snr:.1f} dB")
    print(f"         Effective SNR (signal vs all noise): {effective_snr:.1f} dB")

    # Map channel indices to names for ground truth
    for p in pattern_info:
        p["region1_channel_names"] = [ch_names[i] for i in p["region1_channels"]]
        p["region2_channel_names"] = [ch_names[i] for i in p["region2_channels"]]

    return {
        "filename": f"{artifact_level}.set",
        "artifact_level": artifact_level,
        "label": params["label"],
        "snr_db_target": params["snr_db"],
        "snr_db_actual": round(actual_snr, 2),
        "snr_db_effective": round(effective_snr, 2),
        "duration_sec": DURATION,
        "sfreq": SFREQ,
        "n_channels": N_CHANNELS,
        "montage": MONTAGE_NAME,
        "channel_names": ch_names,
        "artifact_params": {
            "blinks_per_min": params["blinks_per_min"],
            "muscle_bursts_per_min": params["muscle_bursts_per_min"],
            "line_noise_amplitude_uv": params["line_noise_amplitude_uv"],
        },
        "connectivity_patterns": pattern_info,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: Path | None = None):
    """Generate all 4 simulated EEG datasets and ground truth JSON.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory. Defaults to ``scripts/validation/simulated_data/``.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "simulated_data"

    print("=" * 60)
    print("NeuroStat Validation: Simulated EEG Generation")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Sampling rate:    {SFREQ} Hz")
    print(f"Duration:         {DURATION} s")
    print(f"Channels:         {N_CHANNELS} ({MONTAGE_NAME})")
    print(f"Random seed:      {RNG_SEED}")
    print()
    print("Connectivity patterns:")
    for p in CONNECTIVITY_PATTERNS:
        print(f"  - {p['name']}: {p['frequency_hz']} Hz, "
              f"PLI={p['expected_pli']}, lag={p['phase_lag_rad']:.3f} rad")

    rng = np.random.default_rng(RNG_SEED)
    all_datasets = []

    for level_name in ARTIFACT_LEVELS:
        info = generate_dataset(level_name, output_dir, rng)
        all_datasets.append(info)

    # ------------------------------------------------------------------
    # Save ground truth JSON
    # ------------------------------------------------------------------
    ground_truth = {
        "description": (
            "Ground truth for NeuroStat pipeline validation. "
            "Contains simulated EEG datasets with known connectivity "
            "patterns at different artifact severity levels."
        ),
        "generation_params": {
            "sfreq": SFREQ,
            "duration_sec": DURATION,
            "n_channels": N_CHANNELS,
            "montage": MONTAGE_NAME,
            "random_seed": RNG_SEED,
        },
        "frequency_bands": {k: list(v) for k, v in FREQ_BANDS.items()},
        "connectivity_patterns": [
            {
                "name": p["name"],
                "description": p["description"],
                "region1_channels": p["region1_channels"],
                "region2_channels": p["region2_channels"],
                "frequency_hz": p["frequency_hz"],
                "band": p["band"],
                "expected_pli": p["expected_pli"],
                "phase_lag_rad": float(p["phase_lag_rad"]),
                "coupling_strength": p["coupling_strength"],
            }
            for p in CONNECTIVITY_PATTERNS
        ],
        "datasets": [
            {
                "filename": d["filename"],
                "artifact_level": d["artifact_level"],
                "label": d["label"],
                "snr_db_target": d["snr_db_target"],
                "snr_db_actual": d["snr_db_actual"],
                "snr_db_effective": d["snr_db_effective"],
                "artifact_params": d["artifact_params"],
            }
            for d in all_datasets
        ],
    }

    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\nGround truth saved: {gt_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nFiles generated in: {output_dir}")
    for d in all_datasets:
        print(f"  - {d['filename']:30s} SNR={d['snr_db_effective']:+.1f} dB (effective)")
    print(f"  - {'ground_truth.json':30s} (metadata)")
    print()

    return ground_truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate simulated EEG datasets for NeuroStat validation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: scripts/validation/simulated_data/)",
    )
    args = parser.parse_args()
    main(args.output_dir)
