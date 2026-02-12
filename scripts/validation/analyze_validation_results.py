"""
Analyse NeuroStat validation results and generate manuscript outputs.

Reads ground_truth.json and all PLI CSV files produced by
``run_neurostat_batch.py``, then generates:

1. ``validation_scatter.png``   – Ground Truth vs Recovered PLI (4-panel)
2. ``method_comparison.png``    – Mean absolute error by condition
3. ``pattern_recovery.png``     – Pattern recovery for moderate-artifact case
4. ``validation_results.csv``   – One row per pattern × condition (12 rows)
5. Console summary with per-method statistics

Usage
-----
    python scripts/validation/analyze_validation_results.py [--data-dir PATH]

The default *data-dir* is ``scripts/validation/validation_data/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONDITION_ORDER = [
    "clean_high_snr",
    "light_medium_snr",
    "moderate_medium_snr",
    "heavy_low_snr",
]

CONDITION_LABELS = {
    "clean_high_snr":      "Clean\n(10 dB)",
    "light_medium_snr":    "Light\n(5 dB)",
    "moderate_medium_snr": "Moderate\n(5 dB)",
    "heavy_low_snr":       "Heavy\n(2 dB)",
}

CONDITION_LABELS_SHORT = {
    "clean_high_snr":      "Clean",
    "light_medium_snr":    "Light",
    "moderate_medium_snr": "Moderate",
    "heavy_low_snr":       "Heavy",
}

PATTERN_COLORS = {
    "alpha_frontal_parietal":      "#E74C3C",
    "beta_central_occipital":      "#3498DB",
    "theta_left_right_temporal":   "#2ECC71",
}

PATTERN_LABELS = {
    "alpha_frontal_parietal":    "Strong Alpha (10 Hz)",
    "beta_central_occipital":    "Moderate Beta (20 Hz)",
    "theta_left_right_temporal": "Weak Theta (6 Hz)",
}

METHOD_COLORS = {"GEDAI": "#2980B9", "Traditional": "#E67E22"}


# ---------------------------------------------------------------------------
# Desikan-Killiany region grouping (maps sensor channel groups → DK labels)
# ---------------------------------------------------------------------------
DK_REGION_GROUPS: Dict[str, List[str]] = {
    "frontal": [
        "superiorfrontal-lh", "superiorfrontal-rh",
        "rostralmiddlefrontal-lh", "rostralmiddlefrontal-rh",
        "caudalmiddlefrontal-lh", "caudalmiddlefrontal-rh",
        "frontalpole-lh", "frontalpole-rh",
        "lateralorbitofrontal-lh", "lateralorbitofrontal-rh",
        "medialorbitofrontal-lh", "medialorbitofrontal-rh",
        "parsopercularis-lh", "parsopercularis-rh",
        "parsorbitalis-lh", "parsorbitalis-rh",
        "parstriangularis-lh", "parstriangularis-rh",
    ],
    "parietal": [
        "superiorparietal-lh", "superiorparietal-rh",
        "inferiorparietal-lh", "inferiorparietal-rh",
        "precuneus-lh", "precuneus-rh",
        "supramarginal-lh", "supramarginal-rh",
        "postcentral-lh", "postcentral-rh",
    ],
    "central": [
        "precentral-lh", "precentral-rh",
        "postcentral-lh", "postcentral-rh",
        "paracentral-lh", "paracentral-rh",
    ],
    "occipital": [
        "lateraloccipital-lh", "lateraloccipital-rh",
        "cuneus-lh", "cuneus-rh",
        "pericalcarine-lh", "pericalcarine-rh",
        "lingual-lh", "lingual-rh",
    ],
    "temporal_left": [
        "superiortemporal-lh", "middletemporal-lh",
        "inferiortemporal-lh", "transversetemporal-lh",
        "bankssts-lh",
    ],
    "temporal_right": [
        "superiortemporal-rh", "middletemporal-rh",
        "inferiortemporal-rh", "transversetemporal-rh",
        "bankssts-rh",
    ],
}

# Pattern → (region_group_1, region_group_2)
PATTERN_TO_DK = {
    "alpha_frontal_parietal":    ("frontal",        "parietal"),
    "beta_central_occipital":    ("central",        "occipital"),
    "theta_left_right_temporal": ("temporal_left",  "temporal_right"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_ground_truth(data_dir: Path) -> dict:
    gt_path = data_dir / "ground_truth.json"
    if not gt_path.exists():
        # Try parent (simulated_data)
        gt_path = data_dir.parent / "simulated_data" / "ground_truth.json"
    if not gt_path.exists():
        sys.exit(f"ground_truth.json not found under {data_dir}")
    with open(gt_path) as fh:
        return json.load(fh)


def _extract_cross_region_pli(
    pli_csv: Path,
    region_group_1: str,
    region_group_2: str,
) -> float:
    """Mean PLI between two DK region groups from a 68×68 source-level CSV."""
    df = pd.read_csv(pli_csv, index_col=0)
    labels1 = [l for l in DK_REGION_GROUPS[region_group_1] if l in df.index]
    labels2 = [l for l in DK_REGION_GROUPS[region_group_2] if l in df.index]
    if not labels1 or not labels2:
        return np.nan
    sub = df.loc[labels1, labels2].values
    return float(np.mean(sub))


def _extract_sensor_pli(
    pli_csv: Path,
    ch_idx_1: List[int],
    ch_idx_2: List[int],
) -> float:
    """Mean PLI between two channel groups from a sensor-level CSV."""
    df = pd.read_csv(pli_csv, index_col=0)
    ch_names = list(df.index)
    r1 = [ch_names[i] for i in ch_idx_1 if i < len(ch_names)]
    r2 = [ch_names[i] for i in ch_idx_2 if i < len(ch_names)]
    if not r1 or not r2:
        return np.nan
    sub = df.loc[r1, r2].values
    return float(np.mean(sub))


# ---------------------------------------------------------------------------
# Build results DataFrame
# ---------------------------------------------------------------------------
def build_results_table(
    ground_truth: dict,
    data_dir: Path,
) -> pd.DataFrame:
    """Build a tidy table with one row per condition × pattern (12 rows).

    Columns:
        Condition, Pattern, Expected_PLI,
        GEDAI_PLI, GEDAI_Error,
        Traditional_PLI, Traditional_Error,
        GEDAI_Better
    """
    neuro = data_dir / "neurostat_outputs"
    raw_base = data_dir / "raw_sensor_pli"
    patterns = ground_truth["connectivity_patterns"]
    rows = []

    for cond in CONDITION_ORDER:
        for pat in patterns:
            pname = pat["name"]
            band  = pat["band"]
            expected = pat["expected_pli"]
            r1 = pat["region1_channels"]
            r2 = pat["region2_channels"]
            rg1, rg2 = PATTERN_TO_DK[pname]

            row = {
                "Condition": cond,
                "Condition_Label": CONDITION_LABELS_SHORT.get(cond, cond),
                "Pattern": pname,
                "Pattern_Label": PATTERN_LABELS.get(pname, pname),
                "Band": band,
                "Expected_PLI": expected,
            }

            # Raw baseline (sensor-level, unprocessed)
            raw_csv = raw_base / cond / f"sensor_{band}_PLI.csv"
            row["Raw_PLI"] = _extract_sensor_pli(raw_csv, r1, r2) if raw_csv.exists() else np.nan

            for method in ("GEDAI", "Traditional"):
                method_dir = neuro / cond / method

                # Sensor-level PLI (after preprocessing)
                sensor_csv = method_dir / f"sensor_{band}_PLI.csv"
                sensor_val = _extract_sensor_pli(sensor_csv, r1, r2) if sensor_csv.exists() else np.nan

                # Source-level PLI (after source localisation)
                source_csv = method_dir / f"{band}_PLI.csv"
                source_val = _extract_cross_region_pli(source_csv, rg1, rg2) if source_csv.exists() else np.nan

                row[f"{method}_Sensor_PLI"] = round(sensor_val, 4) if not np.isnan(sensor_val) else np.nan
                row[f"{method}_Source_PLI"] = round(source_val, 4) if not np.isnan(source_val) else np.nan

                # Primary comparison metric = sensor-level (direct validation)
                recovered = sensor_val
                row[f"{method}_PLI"]   = round(recovered, 4) if not np.isnan(recovered) else np.nan
                row[f"{method}_Error"] = round(abs(recovered - expected), 4) if not np.isnan(recovered) else np.nan

            # GEDAI better?
            ge = row.get("GEDAI_Error")
            te = row.get("Traditional_Error")
            if ge is not None and te is not None and not np.isnan(ge) and not np.isnan(te):
                row["GEDAI_Better"] = ge < te
            else:
                row["GEDAI_Better"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure1_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 1: Ground Truth vs Recovered PLI – 4 panels (one per condition)."""
    conditions = [c for c in CONDITION_ORDER if c in df["Condition"].values]
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.8), squeeze=False)

    for idx, cond in enumerate(conditions):
        ax = axes[0, idx]
        sub = df[df["Condition"] == cond]

        for method, marker, color in [("GEDAI", "o", METHOD_COLORS["GEDAI"]),
                                       ("Traditional", "s", METHOD_COLORS["Traditional"])]:
            x = sub["Expected_PLI"].values
            y = sub[f"{method}_PLI"].values
            mask = ~np.isnan(y)
            if mask.sum() == 0:
                continue
            ax.scatter(x[mask], y[mask], marker=marker, s=100,
                       color=color, edgecolors="k", linewidths=0.5,
                       label=method, zorder=5)

            # Correlation
            if mask.sum() >= 3:
                r, p = sp_stats.pearsonr(x[mask], y[mask])
                rmse = np.sqrt(np.mean((x[mask] - y[mask]) ** 2))
                ax.text(0.05, 0.95 if method == "GEDAI" else 0.85,
                        f"{method}: r={r:.2f}, RMSE={rmse:.3f}",
                        transform=ax.transAxes, fontsize=8,
                        va="top", color=color, fontweight="bold")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1, label="Identity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Expected PLI", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Recovered PLI", fontsize=11)
        ax.set_title(CONDITION_LABELS_SHORT.get(cond, cond), fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25, zorder=0)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Ground Truth vs Recovered PLI", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    path = out_dir / "validation_scatter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    OK {path.name}")


def figure2_method_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 2: Mean absolute error by condition (GEDAI vs Traditional)."""
    conditions = [c for c in CONDITION_ORDER if c in df["Condition"].values]

    gedai_mae, trad_mae = [], []
    for c in conditions:
        sub = df[df["Condition"] == c]
        gedai_mae.append(sub["GEDAI_Error"].mean())
        trad_mae.append(sub["Traditional_Error"].mean())

    x = np.arange(len(conditions))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w / 2, gedai_mae, w, label="GEDAI",
                   color=METHOD_COLORS["GEDAI"], edgecolor="white", zorder=3)
    bars2 = ax.bar(x + w / 2, trad_mae, w, label="Traditional",
                   color=METHOD_COLORS["Traditional"], edgecolor="white", zorder=3)

    # Value labels on bars
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], fontsize=10)
    ax.set_ylabel("Mean Absolute Error (PLI units)", fontsize=11)
    ax.set_title("Method Comparison: Mean Absolute Error by Artifact Severity",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = out_dir / "method_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    OK {path.name}")


def figure3_pattern_recovery(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 3: Pattern recovery (grouped bar) for moderate-artifact case."""
    # Use moderate condition; fall back to first available
    cond_pref = "moderate_medium_snr"
    available = df["Condition"].unique()
    cond = cond_pref if cond_pref in available else available[0]

    sub = df[df["Condition"] == cond].copy()
    patterns = sub["Pattern"].values
    labels   = [PATTERN_LABELS.get(p, p) for p in patterns]
    expected = sub["Expected_PLI"].values
    gedai    = sub["GEDAI_PLI"].values
    trad     = sub["Traditional_PLI"].values

    x = np.arange(len(patterns))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w, expected, w, label="Ground Truth", color="#2C3E50", edgecolor="white", zorder=3)
    ax.bar(x,     gedai,    w, label="GEDAI Recovered", color=METHOD_COLORS["GEDAI"],
           edgecolor="white", zorder=3)
    ax.bar(x + w, trad,     w, label="Traditional Recovered", color=METHOD_COLORS["Traditional"],
           edgecolor="white", zorder=3)

    # Error annotations
    for i in range(len(patterns)):
        if not np.isnan(gedai[i]):
            ax.annotate(f"{abs(gedai[i]-expected[i]):.3f}",
                        (x[i], gedai[i]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color=METHOD_COLORS["GEDAI"])
        if not np.isnan(trad[i]):
            ax.annotate(f"{abs(trad[i]-expected[i]):.3f}",
                        (x[i] + w, trad[i]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color=METHOD_COLORS["Traditional"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("PLI Value", fontsize=11)
    ax.set_title(f"Pattern Recovery -- {CONDITION_LABELS_SHORT.get(cond, cond)} Artifacts",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = out_dir / "pattern_recovery.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    OK {path.name}")


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
def _print_summary(df: pd.DataFrame) -> None:
    """Print the console summary block expected by the spec."""

    header = "\n" + "=" * 60 + "\n  VALIDATION SUMMARY\n" + "=" * 60

    for method in ("GEDAI", "Traditional"):
        ecol = f"{method}_Error"
        pcol = f"{method}_PLI"
        errs = df[ecol].dropna()
        plis = df[pcol].dropna()
        exp  = df.loc[plis.index, "Expected_PLI"]

        mean_ae  = errs.mean()
        med_ae   = errs.median()
        if len(plis) >= 3:
            r, p = sp_stats.pearsonr(exp, plis)
            rmse = np.sqrt(np.mean((exp.values - plis.values) ** 2))
        else:
            r, p, rmse = np.nan, np.nan, np.nan

        header += f"\n\n  {method} Performance:"
        header += f"\n    Mean absolute error   : {mean_ae:.4f}"
        header += f"\n    Median absolute error  : {med_ae:.4f}"
        header += f"\n    RMSE                   : {rmse:.4f}"
        header += f"\n    Correlation with truth  : r={r:.3f}, p={p:.4f}"

    # Head-to-head
    both = df.dropna(subset=["GEDAI_Error", "Traditional_Error"])
    n_gedai_wins = int((both["GEDAI_Error"] < both["Traditional_Error"]).sum())
    total = len(both)
    header += f"\n\n  GEDAI outperformed Traditional in {n_gedai_wins}/{total} cases"
    header += "\n" + "=" * 60
    print(header)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Run the full analysis pipeline.

    Parameters
    ----------
    data_dir : Path
        Root of validation_data/ produced by ``run_neurostat_batch.py``.
        Default: ``scripts/validation/validation_data/``
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "validation_data"

    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NeuroStat Validation -- Results Analysis")
    print("=" * 60)
    print(f"  Data directory   : {data_dir}")
    print(f"  Results directory : {results_dir}")

    # Load ground truth
    gt = _load_ground_truth(data_dir)

    # Build table
    print("\n  Building results table …")
    df = build_results_table(gt, data_dir)
    print(f"    Rows: {len(df)}")

    # Save CSV
    csv_path = results_dir / "validation_results.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"    OK {csv_path.name}")

    # Figures
    print("\n  Generating figures …")
    figure1_scatter(df, results_dir)
    figure2_method_comparison(df, results_dir)
    figure3_pattern_recovery(df, results_dir)

    # Summary
    _print_summary(df)

    # Print the table itself
    print("\n  Detailed results:")
    display_cols = [
        "Condition_Label", "Pattern_Label", "Expected_PLI",
        "GEDAI_PLI", "GEDAI_Error",
        "Traditional_PLI", "Traditional_Error",
        "GEDAI_Better",
    ]
    print(df[display_cols].to_string(index=False, float_format="%.4f"))

    print(f"\n  All outputs saved to: {results_dir}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse NeuroStat validation results.",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Root of validation_data/ (default: scripts/validation/validation_data/)")
    args = parser.parse_args()
    main(args.data_dir)
