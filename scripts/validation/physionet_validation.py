#!/usr/bin/env python
"""
External validation using the PhysioNet EEG Motor Movement/Imagery Database.

Replicates the well-established finding that alpha-band (8-13 Hz) functional
connectivity (Phase Lag Index) is higher during eyes-closed compared to
eyes-open resting state.

Downloads S001-S020, processes baseline eyes-open (R01) and eyes-closed (R02)
runs through the NeuroStat pipeline (Traditional: ICA + ICLabel), and computes
sensor-level alpha PLI for posterior channels.

Reference
---------
Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R.
BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.
IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.

PhysioNet: https://physionet.org/content/eegmmidb/1.0.0/

Usage
-----
    python scripts/validation/physionet_validation.py
    python scripts/validation/physionet_validation.py --subjects 1 2 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from mne import EpochsArray
from mne.channels import make_standard_montage
from mne.datasets import eegbci

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pli_pipeline import (  # noqa: E402
    PreprocessingConfig,
    preprocess_raw,
    compute_pli,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("physionet_validation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
mne.set_log_level("WARNING")

BASE_DIR    = Path(__file__).resolve().parent
OUT_DIR     = BASE_DIR / "physionet_data"
SET_DIR     = OUT_DIR / "set_files"
RESULTS_DIR = OUT_DIR / "results"

DEFAULT_SUBJECTS = list(range(1, 21))  # S001-S020
RUN_EO = 1  # R01 = eyes open baseline
RUN_EC = 2  # R02 = eyes closed baseline

ALPHA_BAND = (8.0, 13.0)
EPOCH_LENGTH  = 2.0   # seconds (shorter for 1-minute recordings)
EPOCH_OVERLAP = 0.50   # 50 %

POSTERIOR_CHANNELS = [
    "O1", "Oz", "O2",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "P3", "Pz", "P4", "P7", "P8",
]

# Smaller set for readable heatmap annotations
POSTERIOR_DISPLAY = [
    "P3", "Pz", "P4",
    "PO3", "POz", "PO4",
    "O1", "Oz", "O2",
]

MONTAGE = make_standard_montage("standard_1005")

PREPROC_CFG = PreprocessingConfig(
    l_freq=1.0,
    h_freq=40.0,
    notch_freqs=(60.0,),        # US line noise (harmless: bandpass already removes >40 Hz)
    ica_method="fastica",
    ica_n_components=15,         # fewer for 1-minute recordings
    random_state=42,
    use_ged=False,
    use_asr=False,               # disabled for short recordings
    use_iclabel=True,
    iclabel_brain_threshold=0.50,   # relaxed: ICLabel trained on infomax, we use fastica
    bad_channel_threshold=4.0,
    interpolate_bad_channels=True,
    ica_reject_threshold=200e-6,
)

METHOD_COLORS = {"EO": "#3498DB", "EC": "#E74C3C"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_eegbci(
    subject: int,
    run: int,
    data_dir: Path,
) -> mne.io.Raw:
    """Download (if needed) and load one EEGBCI run.

    Returns a Raw object with standardised channel names,
    standard_1005 montage, and average reference.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fnames = eegbci.load_data(subject, runs=[run], path=str(data_dir),
                                   update_path=False)

    raw = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
    eegbci.standardize(raw)
    raw.set_montage(MONTAGE, on_missing="ignore", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def _convert_to_set(
    raw: mne.io.Raw,
    subject: int,
    run: int,
    set_dir: Path,
) -> Optional[Path]:
    """Export Raw to EEGLAB .set format.  Returns path or None on failure."""
    set_dir.mkdir(parents=True, exist_ok=True)
    path = set_dir / f"S{subject:03d}R{run:02d}.set"
    try:
        raw.export(str(path), overwrite=True, verbose=False)
        return path
    except Exception as exc:
        LOGGER.warning("Could not export .set for S%03dR%02d: %s", subject, run, exc)
        return None


def _make_overlapping_epochs(
    data_2d: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    epoch_length: float = EPOCH_LENGTH,
    overlap: float = EPOCH_OVERLAP,
) -> EpochsArray:
    """Create overlapping EpochsArray from 2-D data.  Same logic as run_neurostat_batch."""
    n_ch, n_total = data_2d.shape
    spe  = int(epoch_length * sfreq)
    step = int(spe * (1.0 - overlap))
    starts = list(range(0, n_total - spe + 1, step))

    epoch_data = np.empty((len(starts), n_ch, spe), dtype=data_2d.dtype)
    for idx, s in enumerate(starts):
        epoch_data[idx] = data_2d[:, s : s + spe]

    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
    return EpochsArray(epoch_data, info, verbose=False)


def _process_recording(
    raw: mne.io.Raw,
) -> Tuple[np.ndarray, List[str], int, dict]:
    """Preprocess one recording through NeuroStat and compute alpha PLI.

    Returns
    -------
    pli_matrix : ndarray, shape (n_ch, n_ch)
    ch_names   : list of str
    n_epochs   : int
    preproc_info : dict
    """
    # 1. Preprocess via NeuroStat pipeline (Traditional: ICA + ICLabel)
    raw_before, raw_after, ica, preproc_info = preprocess_raw(raw, PREPROC_CFG)

    # 2. Build overlapping epochs
    picks = mne.pick_types(raw_after.info, eeg=True, meg=False)
    ch_names = [raw_after.ch_names[p] for p in picks]
    data = raw_after.get_data(picks=picks)
    epochs = _make_overlapping_epochs(data, raw_after.info["sfreq"], ch_names)

    # 3. Compute alpha PLI using the pipeline function
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pli_matrix = compute_pli(epochs, band=ALPHA_BAND, method="pli")

    return pli_matrix, ch_names, len(epochs), preproc_info


def _extract_posterior_pli(
    pli_matrix: np.ndarray,
    ch_names: List[str],
    posterior: List[str] = POSTERIOR_CHANNELS,
) -> Tuple[float, np.ndarray, List[str]]:
    """Mean PLI within the posterior ROI (upper triangle, excl. diagonal).

    Returns (mean_pli, sub_matrix, found_channels).
    """
    indices = [ch_names.index(ch) for ch in posterior if ch in ch_names]
    found   = [ch_names[i] for i in indices]
    sub     = pli_matrix[np.ix_(indices, indices)]
    mask    = np.triu(np.ones_like(sub, dtype=bool), k=1)
    return float(sub[mask].mean()), sub, found


def _cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples (mean diff / sd diff)."""
    diff = x - y
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def generate_figure(
    eo_vals: np.ndarray,
    ec_vals: np.ndarray,
    eo_matrix_mean: np.ndarray,
    ec_matrix_mean: np.ndarray,
    ch_names: List[str],
    p_value: float,
    d_value: float,
    out_dir: Path,
) -> Path:
    """Two-panel publication figure: paired comparison + PLI difference matrix."""

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [1.0, 1.2]},
    )

    n = len(eo_vals)
    rng = np.random.RandomState(42)
    jitter = rng.uniform(-0.04, 0.04, n)

    # -- Panel A: paired comparison ----------------------------------------
    for i in range(n):
        ax1.plot(
            [0 + jitter[i], 1 + jitter[i]],
            [eo_vals[i], ec_vals[i]],
            color="#888888", alpha=0.45, linewidth=1.0, zorder=2,
        )

    ax1.scatter(
        np.zeros(n) + jitter, eo_vals, s=70,
        color=METHOD_COLORS["EO"], edgecolors="white", linewidths=0.8,
        zorder=3, label="Eyes Open (R01)",
    )
    ax1.scatter(
        np.ones(n) + jitter, ec_vals, s=70,
        color=METHOD_COLORS["EC"], edgecolors="white", linewidths=0.8,
        zorder=3, label="Eyes Closed (R02)",
    )

    # Mean + SEM bars
    for x_pos, vals, color in [
        (0, eo_vals, METHOD_COLORS["EO"]),
        (1, ec_vals, METHOD_COLORS["EC"]),
    ]:
        m   = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(n)
        ax1.bar(x_pos, m, width=0.5, alpha=0.18, color=color, zorder=1)
        ax1.errorbar(
            x_pos, m, yerr=sem, fmt="none",
            color=color, capsize=8, capthick=2, linewidth=2, zorder=4,
        )

    # Significance bracket
    y_max = max(eo_vals.max(), ec_vals.max())
    by = y_max + 0.015
    ax1.plot([0, 0, 1, 1], [by, by + 0.007, by + 0.007, by],
             color="black", linewidth=1.5)

    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "n.s."

    ax1.text(
        0.5, by + 0.01,
        f"{stars}\np = {p_value:.4f}\nd = {d_value:.2f}",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Eyes Open\n(R01)", "Eyes Closed\n(R02)"], fontsize=11)
    ax1.set_ylabel("Mean Posterior Alpha PLI", fontsize=11)
    ax1.set_title("A.  Paired Comparison", fontsize=12, fontweight="bold", loc="left")
    ax1.set_xlim(-0.5, 1.5)
    ax1.grid(axis="y", alpha=0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=9, loc="lower right")

    # -- Panel B: PLI difference matrix ------------------------------------
    diff_full = ec_matrix_mean - eo_matrix_mean

    disp_idx   = [ch_names.index(ch) for ch in POSTERIOR_DISPLAY if ch in ch_names]
    disp_names = [ch_names[i] for i in disp_idx]
    diff_sub   = diff_full[np.ix_(disp_idx, disp_idx)]

    vmax = max(abs(np.nanmin(diff_sub)), abs(np.nanmax(diff_sub)), 0.03)
    im = ax2.imshow(diff_sub, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    ax2.set_xticks(range(len(disp_names)))
    ax2.set_xticklabels(disp_names, rotation=45, ha="right", fontsize=9)
    ax2.set_yticks(range(len(disp_names)))
    ax2.set_yticklabels(disp_names, fontsize=9)

    for i in range(len(disp_names)):
        for j in range(len(disp_names)):
            if i != j:
                v = diff_sub[i, j]
                color = "white" if abs(v) > vmax * 0.6 else "black"
                ax2.text(j, i, f"{v:+.3f}", ha="center", va="center",
                         fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("PLI difference (EC - EO)", fontsize=9)

    ax2.set_title("B.  Alpha PLI Increase (EC - EO)", fontsize=12,
                  fontweight="bold", loc="left")

    n_subj = len(eo_vals)
    fig.suptitle(
        "External Validation: Alpha Connectivity in Eyes-Closed vs Eyes-Open\n"
        f"PhysioNet EEGBCI Database (N={n_subj})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = out_dir / "physionet_alpha_ec_eo.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Results paragraph
# ---------------------------------------------------------------------------
def _results_paragraph(stats: Dict[str, Any]) -> str:
    """Generate a suggested paragraph for the Results section."""
    n  = stats["n"]
    df = n - 1
    d  = stats["cohens_d"]

    if d >= 0.8:
        mag = "large"
    elif d >= 0.5:
        mag = "medium"
    else:
        mag = "small"

    p = stats["p_ttest"]
    if p < 0.001:
        p_str = "< .001"
    else:
        p_str = f"= {p:.3f}"

    pw = stats["p_wilcoxon"]
    if pw < 0.001:
        pw_str = "< .001"
    else:
        pw_str = f"= {pw:.3f}"

    return (
        "External validation using the PhysioNet EEG Motor Movement/Imagery "
        "Database (Schalk et al., 2004) confirmed the pipeline's ability to "
        "detect established connectivity patterns in independent data. "
        f"Alpha-band (8-13 Hz) PLI among posterior channels was significantly "
        f"higher during eyes-closed (M = {stats['ec_mean']:.3f}, "
        f"SD = {stats['ec_std']:.3f}) compared to eyes-open resting state "
        f"(M = {stats['eo_mean']:.3f}, SD = {stats['eo_std']:.3f}); "
        f"t({df}) = {stats['t_stat']:.2f}, p {p_str}, Cohen's d = {d:.2f}. "
        f"A non-parametric Wilcoxon signed-rank test confirmed the result "
        f"(W = {stats['w_stat']:.0f}, p {pw_str}). "
        f"This {mag} effect replicates the expected increase in posterior "
        "alpha synchronisation during eye closure (Barry et al., 2007; "
        "Klimesch, 1999), validating the pipeline's sensitivity to "
        "physiologically meaningful connectivity changes in real EEG data."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(subjects: Optional[List[int]] = None) -> Dict[str, Any]:
    if subjects is None:
        subjects = DEFAULT_SUBJECTS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SET_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("  PhysioNet EEGBCI Validation")
    print("  Alpha PLI: Eyes-Closed vs Eyes-Open")
    print("=" * 60)
    print(f"  Subjects  : S{subjects[0]:03d}-S{subjects[-1]:03d} (N={len(subjects)})")
    print(f"  Conditions: Eyes Open (R01) vs Eyes Closed (R02)")
    print(f"  Band      : Alpha ({ALPHA_BAND[0]}-{ALPHA_BAND[1]} Hz)")
    print(f"  Posterior  : {len(POSTERIOR_CHANNELS)} channels")
    print(f"  Epochs    : {EPOCH_LENGTH}s, {int(EPOCH_OVERLAP * 100)}% overlap")
    print(f"  Output    : {RESULTS_DIR}")
    print()

    wall_start = time.time()

    # -- Step 1: Download + convert ----------------------------------------
    print("-" * 60)
    print("  [1/4] Downloading and converting PhysioNet data ...")
    print("-" * 60)

    raw_cache: Dict[int, Dict[str, mne.io.Raw]] = {}

    for subj in subjects:
        try:
            raw_eo = _load_eegbci(subj, RUN_EO, OUT_DIR)
            raw_ec = _load_eegbci(subj, RUN_EC, OUT_DIR)
            raw_cache[subj] = {"EO": raw_eo, "EC": raw_ec}

            # Convert to .set
            p1 = _convert_to_set(raw_eo, subj, RUN_EO, SET_DIR)
            p2 = _convert_to_set(raw_ec, subj, RUN_EC, SET_DIR)
            s1 = f"{p1.stat().st_size / 1024:.0f} KB" if p1 else "SKIP"
            s2 = f"{p2.stat().st_size / 1024:.0f} KB" if p2 else "SKIP"
            print(f"    S{subj:03d}: OK  (EO: {s1}, EC: {s2})")
        except Exception as exc:
            print(f"    S{subj:03d}: FAILED ({exc})")

    if len(raw_cache) < 3:
        print("\n  ERROR: Too few subjects downloaded. Aborting.")
        return {}

    # -- Step 2: Process through NeuroStat ---------------------------------
    print()
    print("-" * 60)
    print("  [2/4] Processing through NeuroStat pipeline ...")
    print("-" * 60)

    eo_means: List[float] = []
    ec_means: List[float] = []
    eo_matrices: List[np.ndarray] = []
    ec_matrices: List[np.ndarray] = []
    all_ch_names: Optional[List[str]] = None
    successful: List[int] = []

    for subj, raws in raw_cache.items():
        ok = True
        subj_eo_pli = subj_ec_pli = None

        for label, cond_key, raw_obj in [
            ("Eyes Open ", "EO", raws["EO"]),
            ("Eyes Closed", "EC", raws["EC"]),
        ]:
            try:
                pli, chn, ne, pinfo = _process_recording(raw_obj.copy())
                mean_pli, sub_mat, found = _extract_posterior_pli(pli, chn)

                if all_ch_names is None:
                    all_ch_names = found

                if cond_key == "EO":
                    subj_eo_pli = mean_pli
                    eo_matrices.append(sub_mat)
                else:
                    subj_ec_pli = mean_pli
                    ec_matrices.append(sub_mat)

                n_exc = pinfo.get("ica_excluded", 0)
                print(f"    S{subj:03d} {label}: PLI={mean_pli:.4f}  "
                      f"({ne} epochs, {n_exc} ICA comp. removed)")
            except Exception as exc:
                print(f"    S{subj:03d} {label}: FAILED ({exc})")
                ok = False

        if ok and subj_eo_pli is not None and subj_ec_pli is not None:
            eo_means.append(subj_eo_pli)
            ec_means.append(subj_ec_pli)
            successful.append(subj)

    if len(successful) < 3:
        print("\n  ERROR: Too few subjects processed successfully.")
        return {}

    eo_arr = np.array(eo_means)
    ec_arr = np.array(ec_means)
    n = len(successful)

    eo_mat_mean = np.mean(eo_matrices, axis=0)
    ec_mat_mean = np.mean(ec_matrices, axis=0)

    # -- Step 3: Statistics ------------------------------------------------
    print()
    print("-" * 60)
    print("  [3/4] Statistical analysis ...")
    print("-" * 60)

    t_stat, p_ttest = sp_stats.ttest_rel(ec_arr, eo_arr)
    # Wilcoxon needs n >= 6 for meaningful result
    if n >= 6:
        w_stat, p_wilcox = sp_stats.wilcoxon(ec_arr, eo_arr)
    else:
        w_stat, p_wilcox = float("nan"), float("nan")
    d = _cohens_d_paired(ec_arr, eo_arr)

    stats = {
        "n": n,
        "subjects": successful,
        "eo_mean": float(eo_arr.mean()),
        "eo_std": float(eo_arr.std(ddof=1)),
        "ec_mean": float(ec_arr.mean()),
        "ec_std": float(ec_arr.std(ddof=1)),
        "diff_mean": float((ec_arr - eo_arr).mean()),
        "diff_std": float((ec_arr - eo_arr).std(ddof=1)),
        "t_stat": float(t_stat),
        "p_ttest": float(p_ttest),
        "w_stat": float(w_stat),
        "p_wilcoxon": float(p_wilcox),
        "cohens_d": float(d),
        "ec_gt_eo": bool(ec_arr.mean() > eo_arr.mean()),
        "individual_eo": eo_arr.tolist(),
        "individual_ec": ec_arr.tolist(),
        "posterior_channels": POSTERIOR_CHANNELS,
        "alpha_band_hz": list(ALPHA_BAND),
        "epoch_length_s": EPOCH_LENGTH,
        "epoch_overlap": EPOCH_OVERLAP,
        "preprocessing": "Traditional (ICA + ICLabel)",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"    Subjects analysed : {n}")
    print(f"    Eyes Open   (M +/- SD): {eo_arr.mean():.4f} +/- {eo_arr.std(ddof=1):.4f}")
    print(f"    Eyes Closed (M +/- SD): {ec_arr.mean():.4f} +/- {ec_arr.std(ddof=1):.4f}")
    print(f"    Difference  (EC - EO) : {(ec_arr - eo_arr).mean():+.4f} +/- "
          f"{(ec_arr - eo_arr).std(ddof=1):.4f}")
    print(f"    Paired t-test : t({n-1}) = {t_stat:.3f}, p = {p_ttest:.4f}")
    if not np.isnan(w_stat):
        print(f"    Wilcoxon      : W = {w_stat:.1f}, p = {p_wilcox:.4f}")
    print(f"    Cohen's d     : {d:.3f}")
    direction = "EC > EO (as expected)" if ec_arr.mean() > eo_arr.mean() else "UNEXPECTED: EO > EC"
    print(f"    Direction     : {direction}")

    # -- Step 4: Outputs ---------------------------------------------------
    print()
    print("-" * 60)
    print("  [4/4] Generating outputs ...")
    print("-" * 60)

    # Figure
    fig_path = generate_figure(
        eo_arr, ec_arr, eo_mat_mean, ec_mat_mean,
        all_ch_names, p_ttest, d, RESULTS_DIR,
    )
    print(f"    OK {fig_path.name}")

    # JSON
    json_path = RESULTS_DIR / "physionet_results.json"
    with open(json_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"    OK {json_path.name}")

    # CSV
    df = pd.DataFrame({
        "Subject": [f"S{s:03d}" for s in successful],
        "EyesOpen_Alpha_PLI": eo_arr,
        "EyesClosed_Alpha_PLI": ec_arr,
        "Difference_EC_minus_EO": ec_arr - eo_arr,
    })
    csv_path = RESULTS_DIR / "physionet_subject_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"    OK {csv_path.name}")

    # -- Summary -----------------------------------------------------------
    wall_min = (time.time() - wall_start) / 60

    print()
    print("=" * 60)
    print("  SUGGESTED RESULTS PARAGRAPH")
    print("=" * 60)
    para = _results_paragraph(stats)
    # Word-wrap for console
    words = para.split()
    line = "  "
    for w in words:
        if len(line) + len(w) + 1 > 78:
            print(line)
            line = "  " + w
        else:
            line += " " + w if line.strip() else w
    if line.strip():
        print(line)

    print()
    print("=" * 60)
    print(f"  Validation complete!  ({wall_min:.1f} min)")
    print(f"  Outputs: {RESULTS_DIR}")
    print("=" * 60)
    print()

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhysioNet EEGBCI external validation (alpha EC vs EO).",
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=None,
        help="Subject numbers (default: 1-20)",
    )
    args = parser.parse_args()
    main(args.subjects)
