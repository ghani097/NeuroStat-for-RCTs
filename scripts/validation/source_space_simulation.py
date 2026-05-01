#!/usr/bin/env python
"""
Source-space simulation benchmark for NeuroStat.

This validation plants phase-lagged alpha activity in two known fsaverage
Desikan-Killiany parcels, projects the source activity to scalp EEG with the
same BEM/source-space machinery used by NeuroStat, reconstructs parcel time
courses with the MNE inverse solution, and tests whether reconstructed
source-space PLI is strongest for the planted parcel pair.

Outputs are written to:
    scripts/validation/source_space_simulation/results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from mne.channels import make_standard_montage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pli_pipeline import (  # noqa: E402
    ConnectivityConfig,
    SourceConfig,
    _label_tc_to_epochs,
    _prepare_inverse_operator,
    _run_source_localization,
    compute_pli,
    ensure_subjects_dir,
)


SFREQ = 160.0
DURATION = 60.0
RNG_SEED = 42
N_REPEATS = 20
ALPHA_BAND = (8.0, 13.0)
SPACING = "ico3"
SOURCE_SIGNAL_AMPLITUDE = 2.0
SOURCE_BACKGROUND_SCALE = 0.005
PLANTED_LABEL_A = "lateraloccipital-lh"
PLANTED_LABEL_B = "lateraloccipital-rh"
CONTROL_LABELS = [
    "rostralmiddlefrontal-lh",
    "rostralmiddlefrontal-rh",
    "precentral-lh",
    "precentral-rh",
]

CONDITIONS = [
    {"name": "clean", "sensor_noise": 0.03, "blink_amp": 0.0, "muscle_amp": 0.0},
    {"name": "light_artifact", "sensor_noise": 0.08, "blink_amp": 0.20, "muscle_amp": 0.08},
    {"name": "moderate_artifact", "sensor_noise": 0.12, "blink_amp": 0.35, "muscle_amp": 0.16},
]

OUT_DIR = Path(__file__).resolve().parent / "source_space_simulation" / "results"


def _make_info() -> mne.Info:
    montage = make_standard_montage("standard_1005")
    ch_names = [
        "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
        "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
    ]
    info = mne.create_info(ch_names, SFREQ, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    return info


def _prepare_forward(info: mne.Info, source_cfg: SourceConfig) -> Tuple[dict, mne.SourceSpaces, List[mne.Label], Path]:
    subjects_dir = ensure_subjects_dir(source_cfg.subjects_dir)
    subject = "fsaverage"
    src_dir = Path(subjects_dir) / subject / "bem"
    src_path = src_dir / f"{subject}-{source_cfg.spacing}-src.fif"
    bem_path = src_dir / source_cfg.bem_solution_name

    if src_path.exists():
        src = mne.read_source_spaces(src_path, verbose=False)
    else:
        src = mne.setup_source_space(
            subject=subject,
            spacing=source_cfg.spacing,
            subjects_dir=subjects_dir,
            add_dist=False,
            verbose=False,
        )
        mne.write_source_spaces(src_path, src, overwrite=True, verbose=False)

    bem = mne.read_bem_solution(bem_path, verbose=False)
    fwd = mne.make_forward_solution(
        info=info,
        trans=str(source_cfg.trans_path),
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=1,
        verbose=False,
    )
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)
    labels = [
        label
        for label in mne.read_labels_from_annot(subject, parc="aparc", subjects_dir=subjects_dir, verbose=False)
        if label.name.split("-")[0].lower() != "unknown"
    ]
    return fwd, src, labels, Path(subjects_dir)


def _pink_noise(n_rows: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal((n_rows, n_samples))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / SFREQ)
    freqs[0] = 1.0
    shaped = np.fft.rfft(white, axis=1) / np.sqrt(freqs)[None, :]
    shaped[:, 0] = 0.0
    noise = np.fft.irfft(shaped, n=n_samples, axis=1)
    noise /= np.maximum(noise.std(axis=1, keepdims=True), 1e-12)
    return noise


def _label_vertex_indices(label: mne.Label, src: mne.SourceSpaces, offset: int) -> np.ndarray:
    hemi_idx = 0 if label.hemi == "lh" else 1
    vertno = np.asarray(src[hemi_idx]["vertno"])
    idx = np.where(np.isin(vertno, label.vertices))[0]
    if label.hemi == "rh":
        idx = idx + offset
    return idx


def _make_source_estimate(
    labels: Sequence[mne.Label],
    src: mne.SourceSpaces,
    condition: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[mne.SourceEstimate, Dict[str, Any]]:
    n_samples = int(SFREQ * DURATION)
    times = np.arange(n_samples) / SFREQ
    vertices = [src[0]["vertno"], src[1]["vertno"]]
    n_lh = len(vertices[0])
    n_src = n_lh + len(vertices[1])

    data = SOURCE_BACKGROUND_SCALE * _pink_noise(n_src, n_samples, rng)

    label_by_name = {label.name: label for label in labels}
    lab_a = label_by_name[PLANTED_LABEL_A]
    lab_b = label_by_name[PLANTED_LABEL_B]
    idx_a = _label_vertex_indices(lab_a, src, n_lh)
    idx_b = _label_vertex_indices(lab_b, src, n_lh)

    alpha_phase = 2 * np.pi * 10.0 * times
    jitter = signal.detrend(np.cumsum(rng.standard_normal(n_samples) * 0.015))
    sig_a = np.sin(alpha_phase) + 0.10 * rng.standard_normal(n_samples)
    sig_b = np.sin(alpha_phase + np.pi / 3 + jitter) + 0.10 * rng.standard_normal(n_samples)

    for idx, sig in [(idx_a, sig_a), (idx_b, sig_b)]:
        if len(idx) == 0:
            raise RuntimeError("No vertices found for planted label")
        weights = rng.uniform(0.7, 1.0, size=(len(idx), 1))
        data[idx] += SOURCE_SIGNAL_AMPLITUDE * weights * sig[None, :]

    stc = mne.SourceEstimate(data, vertices=vertices, tmin=0.0, tstep=1.0 / SFREQ, subject="fsaverage")
    truth = {
        "planted_pair": [PLANTED_LABEL_A, PLANTED_LABEL_B],
        "frequency_hz": 10.0,
        "phase_lag_rad": float(np.pi / 3),
        "duration_s": DURATION,
        "sfreq_hz": SFREQ,
        "n_vertices_label_a": int(len(idx_a)),
        "n_vertices_label_b": int(len(idx_b)),
        "source_signal_amplitude_au": SOURCE_SIGNAL_AMPLITUDE,
        "source_background_noise_scale_au": SOURCE_BACKGROUND_SCALE,
        "approx_source_snr_db": float(20 * np.log10(SOURCE_SIGNAL_AMPLITUDE / SOURCE_BACKGROUND_SCALE)),
    }
    return stc, truth


def _add_sensor_artifacts(data: np.ndarray, ch_names: Sequence[str], condition: Dict[str, float], rng: np.random.Generator) -> np.ndarray:
    n_ch, n_samples = data.shape
    out = data.copy()
    out += condition["sensor_noise"] * _pink_noise(n_ch, n_samples, rng)

    times = np.arange(n_samples) / SFREQ
    frontal = [i for i, ch in enumerate(ch_names) if ch in {"Fp1", "Fpz", "Fp2", "AF7", "AF3", "AF4", "AF8"}]
    temporal = [i for i, ch in enumerate(ch_names) if ch in {"T7", "T8", "FT7", "FT8", "TP7", "TP8"}]

    if condition["blink_amp"] > 0:
        for center in np.arange(5.0, DURATION, 12.0):
            blink = np.exp(-0.5 * ((times - center) / 0.12) ** 2)
            for rank, ch_idx in enumerate(frontal):
                out[ch_idx] += condition["blink_amp"] * (1.0 - 0.06 * rank) * blink

    if condition["muscle_amp"] > 0 and temporal:
        b, a = signal.butter(4, [25.0, 45.0], btype="bandpass", fs=SFREQ)
        for ch_idx in temporal:
            burst = signal.filtfilt(b, a, rng.standard_normal(n_samples))
            envelope = np.zeros(n_samples)
            for center in np.arange(8.0, DURATION, 10.0):
                envelope += np.exp(-0.5 * ((times - center) / 0.35) ** 2)
            out[ch_idx] += condition["muscle_amp"] * burst * envelope

    return out


def _simulate_raw(condition: Dict[str, float], fwd: dict, src: mne.SourceSpaces, labels: Sequence[mne.Label], info: mne.Info, rng: np.random.Generator) -> Tuple[mne.io.RawArray, Dict[str, Any]]:
    stc, truth = _make_source_estimate(labels, src, condition, rng)
    gain = fwd["sol"]["data"]
    raw_data = gain @ stc.data
    raw_data /= np.maximum(np.std(raw_data), 1e-12)
    raw_data = _add_sensor_artifacts(raw_data, info["ch_names"], condition, rng)
    raw = mne.io.RawArray(raw_data * 1e-5, info.copy(), verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    return raw, truth


def _extract_pair_stats(pli: np.ndarray, label_names: Sequence[str]) -> Dict[str, float]:
    idx_a = label_names.index(PLANTED_LABEL_A)
    idx_b = label_names.index(PLANTED_LABEL_B)
    planted_pli = float(pli[idx_a, idx_b])

    control_vals = []
    for c in CONTROL_LABELS:
        if c in label_names:
            idx_c = label_names.index(c)
            control_vals.extend([float(pli[idx_a, idx_c]), float(pli[idx_b, idx_c])])

    iu = np.triu_indices_from(pli, k=1)
    all_edges = pli[iu]
    rank = int((all_edges > planted_pli).sum() + 1)
    percentile = float(100.0 * (all_edges <= planted_pli).mean())
    return {
        "planted_pli": planted_pli,
        "control_mean_pli": float(np.mean(control_vals)),
        "control_sd_pli": float(np.std(control_vals, ddof=1)) if len(control_vals) > 1 else 0.0,
        "planted_minus_control": float(planted_pli - np.mean(control_vals)),
        "planted_edge_rank": rank,
        "planted_edge_percentile": percentile,
        "n_control_edges": int(len(control_vals)),
    }


def _ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    sem = stats.sem(arr)
    half = float(stats.t.ppf(0.975, len(arr) - 1) * sem)
    return mean, mean - half, mean + half


def _summarise_repeats(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for condition, sub in df.groupby("condition", sort=False):
        row: Dict[str, Any] = {"condition": condition, "n_repeats": int(len(sub))}
        for col in ["planted_pli", "control_mean_pli", "planted_minus_control", "planted_edge_percentile"]:
            mean, lo, hi = _ci95(sub[col])
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95_low"] = lo
            row[f"{col}_ci95_high"] = hi
        t_stat, p_val = stats.ttest_1samp(sub["planted_minus_control"], 0.0)
        row["planted_minus_control_t"] = float(t_stat)
        row["planted_minus_control_p"] = float(p_val)
        row["planted_minus_control_d"] = float(
            sub["planted_minus_control"].mean() / sub["planted_minus_control"].std(ddof=1)
        )
        row["all_planted_edges_top_5_percent"] = bool((sub["planted_edge_percentile"] >= 95.0).all())
        rows.append(row)

    diff = df["planted_minus_control"].to_numpy(float)
    t_stat, p_val = stats.ttest_1samp(diff, 0.0)
    mean, lo, hi = _ci95(diff)
    overall = {
        "n_runs": int(len(df)),
        "n_repeats_per_condition": int(df.groupby("condition").size().iloc[0]),
        "mean_planted_minus_control": mean,
        "mean_planted_minus_control_ci95": [lo, hi],
        "one_sample_t_vs_zero": float(t_stat),
        "one_sample_t_p": float(p_val),
        "cohens_d_vs_zero": float(diff.mean() / diff.std(ddof=1)),
        "all_planted_edges_top_5_percent": bool((df["planted_edge_percentile"] >= 95.0).all()),
    }
    return pd.DataFrame(rows), overall


def _plot_results(summary_df: pd.DataFrame, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(len(summary_df))
    planted = summary_df["planted_pli_mean"]
    planted_err = np.vstack([
        planted - summary_df["planted_pli_ci95_low"],
        summary_df["planted_pli_ci95_high"] - planted,
    ])
    control = summary_df["control_mean_pli_mean"]
    control_err = np.vstack([
        control - summary_df["control_mean_pli_ci95_low"],
        summary_df["control_mean_pli_ci95_high"] - control,
    ])
    ax.bar(x - 0.18, planted, width=0.36, yerr=planted_err, capsize=3,
           label="Known parcel pair", color="#2B6CB0")
    ax.bar(x + 0.18, control, width=0.36, yerr=control_err, capsize=3,
           label="Anterior control edges", color="#A0AEC0")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["condition"].str.replace("_", "\n"))
    ax.set_ylabel("Reconstructed source-space alpha PLI")
    ax.set_title("Source-Space Simulation: Known Parcel-Pair Recovery", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(n_repeats: int = N_REPEATS) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = _make_info()
    source_cfg = SourceConfig(spacing=SPACING)
    conn_cfg = ConnectivityConfig(epoch_length=4.0)

    started = time.time()
    fwd, src, labels, subjects_dir = _prepare_forward(info, source_cfg)

    rows = []
    truth_payload = None
    for repeat_idx in range(n_repeats):
        for condition in CONDITIONS:
            seed = RNG_SEED + repeat_idx * 100 + CONDITIONS.index(condition)
            rng = np.random.default_rng(seed)
            raw, truth = _simulate_raw(condition, fwd, src, labels, info, rng)
            truth_payload = truth

            inv_op, recon_labels, recon_src, _ = _prepare_inverse_operator(raw, source_cfg)
            label_tc, label_names, _ = _run_source_localization(raw, inv_op, recon_labels, recon_src, source_cfg)
            epochs = _label_tc_to_epochs(label_tc, label_names, SFREQ, conn_cfg.epoch_length)
            pli = compute_pli(epochs, ALPHA_BAND, method="pli", use_gpu=False)

            stats_row = _extract_pair_stats(pli, label_names)
            stats_row.update({
                "condition": condition["name"],
                "repeat": int(repeat_idx + 1),
                "seed": int(seed),
                "n_epochs": int(len(epochs)),
                "n_labels": int(len(label_names)),
            })
            rows.append(stats_row)

            if repeat_idx == 0:
                pd.DataFrame(pli, index=label_names, columns=label_names).to_csv(
                    OUT_DIR / f"{condition['name']}_source_alpha_pli_repeat01.csv"
                )

    df = pd.DataFrame(rows)
    summary_df, overall_summary = _summarise_repeats(df)
    shapiro = stats.shapiro(df["planted_minus_control"]) if len(df) >= 3 else None

    csv_path = OUT_DIR / "source_space_simulation_results.csv"
    summary_csv_path = OUT_DIR / "source_space_simulation_summary_by_condition.csv"
    fig_path = OUT_DIR / "source_space_simulation_recovery.png"
    json_path = OUT_DIR / "source_space_simulation_summary.json"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    summary_df.to_csv(summary_csv_path, index=False, float_format="%.6f")
    _plot_results(summary_df, fig_path)

    payload = {
        "analysis": "source_space_simulation",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runtime_min": round((time.time() - started) / 60.0, 2),
        "random_seed": RNG_SEED,
        "n_repeats": int(n_repeats),
        "subjects_dir": str(subjects_dir),
        "source_spacing": SPACING,
        "truth": truth_payload,
        "conditions": CONDITIONS,
        "control_edges": {
            "control_labels": CONTROL_LABELS,
            "n_control_edges": int(df["n_control_edges"].iloc[0]),
            "distance_matching": "not applied; controls were anterior/non-posterior Desikan-Killiany parcels",
        },
        "summary": {
            "mean_planted_pli": float(df["planted_pli"].mean()),
            "mean_control_pli": float(df["control_mean_pli"].mean()),
            "mean_planted_minus_control": float(df["planted_minus_control"].mean()),
            "all_planted_edges_top_5_percent": bool((df["planted_edge_percentile"] >= 95.0).all()),
            "shapiro_planted_minus_control_p": float(shapiro.pvalue) if shapiro is not None else None,
            **overall_summary,
        },
        "artifacts": {
            "csv": str(csv_path),
            "summary_csv": str(summary_csv_path),
            "figure": str(fig_path),
            "json": str(json_path),
        },
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(summary_df.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {json_path}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run source-space simulation benchmark.")
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()
    run(args.n_repeats)
