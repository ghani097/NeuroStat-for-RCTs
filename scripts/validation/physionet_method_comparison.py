#!/usr/bin/env python
"""
Standalone PhysioNet method comparison (GEDAI vs Traditional) on real EEG.

This script does not modify existing validation scripts. It reuses core loading
and connectivity utilities to compare preprocessing methods on the same endpoint:
posterior alpha-band (8-13 Hz) PLI difference between eyes-closed (R02) and
eyes-open (R01) resting state.

Outputs (under scripts/validation/physionet_data/method_comparison/):
- physionet_method_comparison_results.json
- physionet_method_subject_results.csv
- physionet_method_comparison.png
"""

from __future__ import annotations

import argparse
import json
import logging
import time
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

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pli_pipeline import PreprocessingConfig, preprocess_raw, compute_pli  # noqa: E402
from scripts.validation.physionet_validation import (  # noqa: E402
    _load_eegbci,
    _make_overlapping_epochs,
    _extract_posterior_pli,
)

LOGGER = logging.getLogger("physionet_method_comparison")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
mne.set_log_level("WARNING")

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "physionet_data" / "method_comparison"
DATA_DIR = BASE_DIR / "physionet_data"

DEFAULT_SUBJECTS = list(range(1, 21))
RUN_EO = 1
RUN_EC = 2

ALPHA_BAND = (8.0, 13.0)
EPOCH_LENGTH = 2.0
EPOCH_OVERLAP = 0.50

METHOD_COLORS = {
    "Traditional": "#E67E22",
    "GEDAI": "#2980B9",
}


def _build_configs() -> Dict[str, PreprocessingConfig]:
    """Build method configs without changing existing codepaths."""
    common = dict(
        l_freq=1.0,
        h_freq=40.0,
        notch_freqs=(60.0,),
        ica_method="fastica",
        ica_n_components=15,
        random_state=42,
        bad_channel_threshold=4.0,
        interpolate_bad_channels=True,
        ica_reject_threshold=200e-6,
    )

    return {
        "Traditional": PreprocessingConfig(
            **common,
            use_ged=False,
            use_asr=False,
            use_iclabel=True,
            iclabel_brain_threshold=0.50,
            ged_threshold=3.0,
        ),
        "GEDAI": PreprocessingConfig(
            **common,
            use_ged=True,
            use_asr=False,
            use_iclabel=False,
            iclabel_brain_threshold=0.50,
            ged_threshold=3.0,
        ),
    }


def _cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def _process_recording(raw: mne.io.Raw, cfg: PreprocessingConfig) -> Tuple[float, int, dict]:
    """Return posterior alpha mean PLI for one recording."""
    raw_before, raw_after, ica, preproc_info = preprocess_raw(raw, cfg)

    picks = mne.pick_types(raw_after.info, eeg=True, meg=False)
    ch_names = [raw_after.ch_names[p] for p in picks]
    data = raw_after.get_data(picks=picks)
    epochs = _make_overlapping_epochs(
        data,
        raw_after.info["sfreq"],
        ch_names,
        epoch_length=EPOCH_LENGTH,
        overlap=EPOCH_OVERLAP,
    )

    pli = compute_pli(epochs, band=ALPHA_BAND, method="pli")
    mean_pli, _, _ = _extract_posterior_pli(pli, ch_names)
    return float(mean_pli), len(epochs), preproc_info


def _method_stats(subject_df: pd.DataFrame) -> Dict[str, Any]:
    eo = subject_df["EyesOpen_Alpha_PLI"].to_numpy(float)
    ec = subject_df["EyesClosed_Alpha_PLI"].to_numpy(float)
    diff = ec - eo
    n = len(subject_df)

    t_stat, p_ttest = sp_stats.ttest_rel(ec, eo)
    if n >= 6:
        w_stat, p_wilcox = sp_stats.wilcoxon(ec, eo)
    else:
        w_stat, p_wilcox = float("nan"), float("nan")

    return {
        "n": int(n),
        "subjects": subject_df["Subject"].tolist(),
        "eo_mean": float(eo.mean()),
        "eo_std": float(eo.std(ddof=1)),
        "ec_mean": float(ec.mean()),
        "ec_std": float(ec.std(ddof=1)),
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std(ddof=1)),
        "t_stat": float(t_stat),
        "p_ttest": float(p_ttest),
        "w_stat": float(w_stat),
        "p_wilcoxon": float(p_wilcox),
        "cohens_d": float(_cohens_d_paired(ec, eo)),
        "direction_ec_gt_eo": int((diff > 0).sum()),
        "direction_ec_lt_eo": int((diff < 0).sum()),
        "direction_ec_eq_eo": int((diff == 0).sum()),
    }


def _winner(method_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Choose winner by strongest physiologically expected EC>EO effect."""

    def score(s: Dict[str, Any]) -> Tuple[float, float, int]:
        signed_d = s["cohens_d"] if s["ec_mean"] > s["eo_mean"] else -abs(s["cohens_d"])
        # maximize signed_d, then minimize p_ttest, then maximize direction count
        return (signed_d, -s["p_ttest"], s["direction_ec_gt_eo"])

    methods = list(method_stats.keys())
    best = max(methods, key=lambda m: score(method_stats[m]))

    ranked = sorted(methods, key=lambda m: score(method_stats[m]), reverse=True)
    return {
        "winner": best,
        "ranking": ranked,
        "criterion": "largest expected-direction paired effect (Cohen's d for EC-EO), tie-break by lower paired t-test p and more EC>EO subjects",
    }


def _plot(method_to_df: Dict[str, pd.DataFrame], method_stats: Dict[str, Dict[str, Any]], out_file: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.0, 1.2]})

    methods = list(method_to_df.keys())
    x = np.arange(len(methods))
    w = 0.35

    eo_means = [method_stats[m]["eo_mean"] for m in methods]
    ec_means = [method_stats[m]["ec_mean"] for m in methods]
    eo_sem = [method_to_df[m]["EyesOpen_Alpha_PLI"].std(ddof=1) / np.sqrt(len(method_to_df[m])) for m in methods]
    ec_sem = [method_to_df[m]["EyesClosed_Alpha_PLI"].std(ddof=1) / np.sqrt(len(method_to_df[m])) for m in methods]

    ax1.bar(x - w / 2, eo_means, w, yerr=eo_sem, capsize=4, color="#3498DB", alpha=0.85, label="EO")
    ax1.bar(x + w / 2, ec_means, w, yerr=ec_sem, capsize=4, color="#E74C3C", alpha=0.85, label="EC")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("Posterior alpha PLI")
    ax1.set_title("A. EO vs EC mean PLI by method", loc="left", fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend()

    # subject-level differences
    jitter_rng = np.random.RandomState(42)
    for i, m in enumerate(methods):
        d = method_to_df[m]["Difference_EC_minus_EO"].to_numpy(float)
        jitter = jitter_rng.uniform(-0.08, 0.08, len(d))
        ax2.scatter(np.full(len(d), i) + jitter, d, s=35, alpha=0.7, color=METHOD_COLORS[m], label=m if i == 0 else None)
        mean_d = d.mean()
        ax2.hlines(mean_d, i - 0.2, i + 0.2, colors="black", linewidth=2)
        ax2.text(i, mean_d + 0.005, f"d={method_stats[m]['cohens_d']:.2f}\np={method_stats[m]['p_ttest']:.4f}",
                 ha="center", va="bottom", fontsize=9)

    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel("EC - EO posterior alpha PLI")
    ax2.set_title("B. Subject-level EC-EO differences", loc="left", fontweight="bold")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("PhysioNet Real-EEG Method Comparison: GEDAI vs Traditional", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(subjects: Optional[List[int]] = None) -> Dict[str, Any]:
    if subjects is None:
        subjects = DEFAULT_SUBJECTS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PhysioNet Method Comparison (Real EEG): GEDAI vs Traditional")
    print("=" * 70)
    print(f"Subjects: S{subjects[0]:03d}-S{subjects[-1]:03d} (requested N={len(subjects)})")
    print(f"Endpoint: posterior alpha PLI ({ALPHA_BAND[0]}-{ALPHA_BAND[1]} Hz), EC vs EO")
    print(f"Output: {OUT_DIR}")

    t0 = time.time()

    # Load raw data once
    raw_cache: Dict[int, Dict[str, mne.io.Raw]] = {}
    print("\n[1/4] Loading PhysioNet recordings...")
    for subj in subjects:
        try:
            raw_eo = _load_eegbci(subj, RUN_EO, DATA_DIR)
            raw_ec = _load_eegbci(subj, RUN_EC, DATA_DIR)
            raw_cache[subj] = {"EO": raw_eo, "EC": raw_ec}
            print(f"  S{subj:03d}: OK")
        except Exception as exc:
            print(f"  S{subj:03d}: FAILED ({exc})")

    if len(raw_cache) < 3:
        raise RuntimeError("Too few subjects loaded for method comparison")

    cfgs = _build_configs()
    method_subject_rows: List[Dict[str, Any]] = []
    method_to_df: Dict[str, pd.DataFrame] = {}
    method_stats: Dict[str, Dict[str, Any]] = {}

    print("\n[2/4] Processing methods...")
    for method_name, cfg in cfgs.items():
        print(f"\n  Method: {method_name}")
        rows = []
        for subj, raws in raw_cache.items():
            try:
                eo_pli, eo_epochs, eo_info = _process_recording(raws["EO"].copy(), cfg)
                ec_pli, ec_epochs, ec_info = _process_recording(raws["EC"].copy(), cfg)

                row = {
                    "Subject": f"S{subj:03d}",
                    "SubjectID": subj,
                    "Method": method_name,
                    "EyesOpen_Alpha_PLI": eo_pli,
                    "EyesClosed_Alpha_PLI": ec_pli,
                    "Difference_EC_minus_EO": ec_pli - eo_pli,
                    "EO_Epochs": eo_epochs,
                    "EC_Epochs": ec_epochs,
                    "EO_ICA_Excluded": int(eo_info.get("ica_excluded", 0)),
                    "EC_ICA_Excluded": int(ec_info.get("ica_excluded", 0)),
                }
                rows.append(row)
                method_subject_rows.append(row)
                print(
                    f"    S{subj:03d}: EO={eo_pli:.4f}, EC={ec_pli:.4f}, "
                    f"Diff={ec_pli - eo_pli:+.4f}"
                )
            except Exception as exc:
                print(f"    S{subj:03d}: FAILED ({exc})")

        if len(rows) < 3:
            print(f"  {method_name}: insufficient successful subjects (n={len(rows)})")
            continue

        mdf = pd.DataFrame(rows).sort_values("SubjectID")
        method_to_df[method_name] = mdf
        method_stats[method_name] = _method_stats(mdf)

    if len(method_to_df) < 2:
        raise RuntimeError("Could not compute both methods successfully")

    print("\n[3/4] Computing winner and summary metrics...")
    winner_info = _winner(method_stats)
    winner = winner_info["winner"]

    for m, s in method_stats.items():
        print(
            f"  {m}: n={s['n']}, EO={s['eo_mean']:.4f}+-{s['eo_std']:.4f}, "
            f"EC={s['ec_mean']:.4f}+-{s['ec_std']:.4f}, "
            f"t({s['n']-1})={s['t_stat']:.3f}, p={s['p_ttest']:.4f}, d={s['cohens_d']:.3f}, "
            f"EC>EO {s['direction_ec_gt_eo']}/{s['n']}"
        )

    print(f"\n  Winner: {winner}")

    print("\n[4/4] Writing outputs...")
    csv_path = OUT_DIR / "physionet_method_subject_results.csv"
    json_path = OUT_DIR / "physionet_method_comparison_results.json"
    fig_path = OUT_DIR / "physionet_method_comparison.png"

    all_df = pd.DataFrame(method_subject_rows).sort_values(["Method", "SubjectID"])
    all_df.to_csv(csv_path, index=False, float_format="%.6f")

    _plot(method_to_df, method_stats, fig_path)

    payload = {
        "analysis": "real_eeg_method_comparison",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "subjects_requested": subjects,
        "methods": method_stats,
        "winner": winner_info,
        "endpoint": {
            "band_hz": list(ALPHA_BAND),
            "epoch_length_s": EPOCH_LENGTH,
            "epoch_overlap": EPOCH_OVERLAP,
            "contrast": "eyes_closed_minus_eyes_open",
            "metric": "posterior_mean_sensor_PLI",
        },
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "figure": str(fig_path),
        },
        "runtime_min": round((time.time() - t0) / 60.0, 2),
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  OK {csv_path}")
    print(f"  OK {fig_path}")
    print(f"  OK {json_path}")
    print(f"  Total runtime: {payload['runtime_min']:.2f} min")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GEDAI vs Traditional on PhysioNet real EEG.")
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Subject numbers (default: 1-20)")
    args = parser.parse_args()
    main(args.subjects)
