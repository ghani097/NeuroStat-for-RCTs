#!/usr/bin/env python
"""
Source-space PhysioNet EEGBCI benchmark with ICLabel-enabled preprocessing.

This script computes the eyes-closed vs eyes-open posterior alpha contrast
after BEM/MNE source reconstruction and Desikan-Killiany parcellation.
It is intentionally separate from physionet_validation.py, which computes
the same physiological contrast on posterior sensor channels.
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
from scipy import stats
from scipy.signal import find_peaks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pli_pipeline import (  # noqa: E402
    SourceConfig,
    preprocess_raw,
    _prepare_inverse_operator,
    _run_source_localization,
    compute_pli,
)
from scripts.validation.physionet_validation import _load_eegbci, PREPROC_CFG  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "physionet_data"
OUT_DIR = DATA_DIR / "source_space_iclabel"

RUN_EO = 1
RUN_EC = 2
ALPHA_BAND = (8.0, 13.0)
EPOCH_LENGTH = 2.0
EPOCH_OVERLAP = 0.50
SOURCE_SPACING = "ico3"
FRONTAL_CHANNELS = {
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
}
LARGE_ICA_DIFF_THRESHOLD = 3

POSTERIOR_DK_BASE = [
    "cuneus",
    "lateraloccipital",
    "lingual",
    "pericalcarine",
    "precuneus",
    "superiorparietal",
    "inferiorparietal",
]
POSTERIOR_DK_LABELS = [f"{base}-{hemi}" for base in POSTERIOR_DK_BASE for hemi in ("lh", "rh")]


def _make_overlapping_label_epochs(
    label_tc: np.ndarray,
    label_names: Sequence[str],
    sfreq: float,
    epoch_length: float = EPOCH_LENGTH,
    overlap: float = EPOCH_OVERLAP,
) -> mne.EpochsArray:
    n_labels, n_total = label_tc.shape
    spe = int(epoch_length * sfreq)
    step = int(spe * (1.0 - overlap))
    starts = list(range(0, n_total - spe + 1, step))
    data = np.empty((len(starts), n_labels, spe), dtype=label_tc.dtype)
    for idx, start in enumerate(starts):
        data[idx] = label_tc[:, start:start + spe]
    info = mne.create_info(list(label_names), sfreq, ch_types="misc")
    return mne.EpochsArray(data, info, verbose=False)


def _posterior_mean_pli(pli: np.ndarray, label_names: Sequence[str]) -> Tuple[float, List[str]]:
    found = [name for name in POSTERIOR_DK_LABELS if name in label_names]
    idx = [label_names.index(name) for name in found]
    sub = pli[np.ix_(idx, idx)]
    tri = np.triu_indices_from(sub, k=1)
    return float(sub[tri].mean()), found


def _frontal_metrics(raw: mne.io.Raw) -> Dict[str, Any]:
    picks = [idx for idx, name in enumerate(raw.ch_names) if name in FRONTAL_CHANNELS]
    if not picks:
        return {
            "frontal_channels": [],
            "frontal_var_uv2": np.nan,
            "frontal_transient_count": np.nan,
        }
    data_uv = raw.get_data(picks=picks) * 1e6
    mean_front = data_uv.mean(axis=0)
    centered = mean_front - mean_front.mean()
    sd = centered.std(ddof=1)
    min_distance = max(1, int(round(0.30 * raw.info["sfreq"])))
    threshold = max(50.0, 3.0 * sd)
    peaks, _ = find_peaks(np.abs(centered), height=threshold, distance=min_distance)
    return {
        "frontal_channels": [raw.ch_names[idx] for idx in picks],
        "frontal_var_uv2": float(np.mean(np.var(data_uv, axis=1, ddof=1))),
        "frontal_transient_count": int(len(peaks)),
    }


def _process(raw: mne.io.Raw, source_cfg: SourceConfig) -> Tuple[float, int, Dict[str, Any]]:
    before_metrics = _frontal_metrics(raw)
    _, raw_after, _, preproc_info = preprocess_raw(raw, PREPROC_CFG)
    raw_after.set_eeg_reference("average", projection=True, verbose=False)
    after_metrics = _frontal_metrics(raw_after)

    inv_op, labels, src, _ = _prepare_inverse_operator(raw_after, source_cfg)
    label_tc, label_names, _ = _run_source_localization(raw_after, inv_op, labels, src, source_cfg)
    epochs = _make_overlapping_label_epochs(label_tc, label_names, raw_after.info["sfreq"])
    pli = compute_pli(epochs, ALPHA_BAND, method="pli", use_gpu=False)
    mean_pli, found = _posterior_mean_pli(pli, list(label_names))

    preproc_info = dict(preproc_info)
    preproc_info["posterior_labels_found"] = found
    preproc_info["frontal_metrics_before"] = before_metrics
    preproc_info["frontal_metrics_after"] = after_metrics
    return mean_pli, len(epochs), preproc_info


def _cohens_d(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def _paired_stats(ec: np.ndarray, eo: np.ndarray) -> Dict[str, Any]:
    diff = ec - eo
    t_stat, p_t = stats.ttest_rel(ec, eo)
    if len(diff) > 0 and np.allclose(diff, diff[0]):
        w_stat, p_w = np.nan, np.nan
    else:
        w_stat, p_w = stats.wilcoxon(ec, eo)
    return {
        "n": int(len(diff)),
        "eo_mean": float(eo.mean()),
        "eo_sd": float(eo.std(ddof=1)),
        "ec_mean": float(ec.mean()),
        "ec_sd": float(ec.std(ddof=1)),
        "diff_mean": float(diff.mean()),
        "diff_sd": float(diff.std(ddof=1)),
        "paired_t": float(t_stat),
        "paired_t_p": float(p_t),
        "wilcoxon_w": float(w_stat),
        "wilcoxon_p": float(p_w),
        "cohens_d": _cohens_d(diff),
        "direction_ec_gt_eo": int((diff > 0).sum()),
    }


def _flatten_iclabel_counts(prefix: str, info: Dict[str, Any]) -> Dict[str, int]:
    counts = info.get("iclabel_class_counts", {}) or {}
    out: Dict[str, int] = {}
    for label, count in counts.items():
        key = str(label).lower().replace(" ", "_").replace("-", "_")
        out[f"{prefix}_ICLabel_{key}"] = int(count)
    return out


def _plot(df: pd.DataFrame, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for _, row in df.iterrows():
        ax.plot([0, 1], [row["EyesOpen_SourceAlpha_PLI"], row["EyesClosed_SourceAlpha_PLI"]],
                color="#718096", alpha=0.55, linewidth=1)
        ax.scatter([0, 1], [row["EyesOpen_SourceAlpha_PLI"], row["EyesClosed_SourceAlpha_PLI"]],
                   color=["#3182CE", "#E53E3E"], s=26, zorder=3)
    means = [df["EyesOpen_SourceAlpha_PLI"].mean(), df["EyesClosed_SourceAlpha_PLI"].mean()]
    ax.plot([0, 1], means, color="black", linewidth=3, marker="o", markersize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Eyes open", "Eyes closed"])
    ax.set_ylabel("Posterior DK alpha PLI")
    ax.set_title("PhysioNet Source-Space Posterior Alpha Benchmark", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(subjects: Sequence[int]) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_cfg = SourceConfig(spacing=SOURCE_SPACING)
    rows: List[Dict[str, Any]] = []
    started = time.time()

    for subject in subjects:
        print(f"S{subject:03d}")
        raw_eo = _load_eegbci(subject, RUN_EO, DATA_DIR)
        raw_ec = _load_eegbci(subject, RUN_EC, DATA_DIR)
        eo_pli, eo_epochs, eo_info = _process(raw_eo, source_cfg)
        ec_pli, ec_epochs, ec_info = _process(raw_ec, source_cfg)
        eo_before = eo_info["frontal_metrics_before"]
        eo_after = eo_info["frontal_metrics_after"]
        ec_before = ec_info["frontal_metrics_before"]
        ec_after = ec_info["frontal_metrics_after"]
        row = {
            "Subject": f"S{subject:03d}",
            "SubjectID": subject,
            "EyesOpen_SourceAlpha_PLI": eo_pli,
            "EyesClosed_SourceAlpha_PLI": ec_pli,
            "Difference_EC_minus_EO": ec_pli - eo_pli,
            "EO_Epochs": eo_epochs,
            "EC_Epochs": ec_epochs,
            "EO_ICA_Excluded": int(eo_info.get("ica_excluded", 0)),
            "EC_ICA_Excluded": int(ec_info.get("ica_excluded", 0)),
            "Abs_ICA_Excluded_Diff": int(abs(eo_info.get("ica_excluded", 0) - ec_info.get("ica_excluded", 0))),
            "EO_ICLabel_Applied": bool(eo_info.get("iclabel_applied", False)),
            "EC_ICLabel_Applied": bool(ec_info.get("iclabel_applied", False)),
            "EO_FrontalVar_Before_uV2": eo_before["frontal_var_uv2"],
            "EO_FrontalVar_After_uV2": eo_after["frontal_var_uv2"],
            "EC_FrontalVar_Before_uV2": ec_before["frontal_var_uv2"],
            "EC_FrontalVar_After_uV2": ec_after["frontal_var_uv2"],
            "EO_FrontalTransient_Before": eo_before["frontal_transient_count"],
            "EO_FrontalTransient_After": eo_after["frontal_transient_count"],
            "EC_FrontalTransient_Before": ec_before["frontal_transient_count"],
            "EC_FrontalTransient_After": ec_after["frontal_transient_count"],
        }
        row.update(_flatten_iclabel_counts("EO", eo_info))
        row.update(_flatten_iclabel_counts("EC", ec_info))
        rows.append(row)
        print(f"  EO={eo_pli:.4f}, EC={ec_pli:.4f}, diff={ec_pli - eo_pli:+.4f}")

    df = pd.DataFrame(rows)
    eo = df["EyesOpen_SourceAlpha_PLI"].to_numpy(float)
    ec = df["EyesClosed_SourceAlpha_PLI"].to_numpy(float)
    diff = ec - eo
    primary_stats = _paired_stats(ec, eo)
    shapiro = stats.shapiro(diff)
    ica_t = stats.ttest_rel(df["EO_ICA_Excluded"], df["EC_ICA_Excluded"])
    epoch_t = stats.ttest_rel(df["EO_Epochs"], df["EC_Epochs"])
    frontal_after_t = stats.ttest_rel(df["EC_FrontalVar_After_uV2"], df["EO_FrontalVar_After_uV2"])
    transient_after_t = stats.ttest_rel(df["EC_FrontalTransient_After"], df["EO_FrontalTransient_After"])
    sensitivity_df = df[df["Abs_ICA_Excluded_Diff"] < LARGE_ICA_DIFF_THRESHOLD].copy()
    sensitivity_stats = _paired_stats(
        sensitivity_df["EyesClosed_SourceAlpha_PLI"].to_numpy(float),
        sensitivity_df["EyesOpen_SourceAlpha_PLI"].to_numpy(float),
    )
    iclabel_count_cols = [
        col for col in df.columns
        if (col.startswith("EO_ICLabel_") or col.startswith("EC_ICLabel_"))
        and not col.endswith("_Applied")
    ]
    iclabel_counts = {
        col: int(df[col].fillna(0).sum())
        for col in iclabel_count_cols
    }

    csv_path = OUT_DIR / "physionet_source_subject_results.csv"
    fig_path = OUT_DIR / "physionet_source_paired.png"
    json_path = OUT_DIR / "physionet_source_results.json"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    _plot(df, fig_path)

    payload = {
        "analysis": "physionet_source_space_iclabel",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runtime_min": round((time.time() - started) / 60.0, 2),
        "subjects": list(subjects),
        "source_spacing": SOURCE_SPACING,
        "posterior_dk_labels": POSTERIOR_DK_LABELS,
        "alpha_band_hz": list(ALPHA_BAND),
        "epoch_length_s": EPOCH_LENGTH,
        "epoch_overlap": EPOCH_OVERLAP,
        "stats": {
            **primary_stats,
            "shapiro_w": float(shapiro.statistic),
            "shapiro_p": float(shapiro.pvalue),
            "reversed_subjects": df.loc[diff <= 0, "Subject"].tolist(),
            "eo_ica_mean": float(df["EO_ICA_Excluded"].mean()),
            "eo_ica_sd": float(df["EO_ICA_Excluded"].std(ddof=1)),
            "ec_ica_mean": float(df["EC_ICA_Excluded"].mean()),
            "ec_ica_sd": float(df["EC_ICA_Excluded"].std(ddof=1)),
            "ica_paired_t": float(ica_t.statistic),
            "ica_paired_t_p": float(ica_t.pvalue),
            "epoch_paired_t": float(epoch_t.statistic),
            "epoch_paired_t_p": float(epoch_t.pvalue),
            "frontal_var_after_eo_mean": float(df["EO_FrontalVar_After_uV2"].mean()),
            "frontal_var_after_ec_mean": float(df["EC_FrontalVar_After_uV2"].mean()),
            "frontal_var_after_paired_t": float(frontal_after_t.statistic),
            "frontal_var_after_paired_t_p": float(frontal_after_t.pvalue),
            "frontal_transient_after_eo_mean": float(df["EO_FrontalTransient_After"].mean()),
            "frontal_transient_after_ec_mean": float(df["EC_FrontalTransient_After"].mean()),
            "frontal_transient_after_paired_t": float(transient_after_t.statistic),
            "frontal_transient_after_paired_t_p": float(transient_after_t.pvalue),
            "sensitivity_excluding_abs_ica_diff_gte": LARGE_ICA_DIFF_THRESHOLD,
            "sensitivity_excluded_subjects": df.loc[
                df["Abs_ICA_Excluded_Diff"] >= LARGE_ICA_DIFF_THRESHOLD, "Subject"
            ].tolist(),
            "sensitivity": sensitivity_stats,
            "iclabel_component_class_counts": iclabel_counts,
        },
        "artifacts": {"csv": str(csv_path), "figure": str(fig_path), "json": str(json_path)},
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(df.to_string(index=False))
    print(json.dumps(payload["stats"], indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run source-space PhysioNet ICLabel benchmark.")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 21)))
    args = parser.parse_args()
    run(args.subjects)
