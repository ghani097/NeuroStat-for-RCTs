"""
Analyze validation results and generate manuscript figures/tables.

Compares recovered PLI values from the NeuroStat pipeline against
ground truth connectivity patterns, producing:

1. Figure: PLI recovery accuracy across artifact severity levels
2. Figure: Sensor-level vs source-level PLI comparison
3. Figure: Preprocessing effect (raw vs cleaned) on PLI
4. Table: Quantitative results for manuscript (CSV + LaTeX)

Usage:
    python scripts/validation/analyze_validation.py [--results-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_validation_data(results_dir: Path) -> Tuple[dict, list]:
    """Load ground truth and validation results.

    Returns (ground_truth, results_list).
    """
    results_path = results_dir / "validation_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Validation results not found: {results_path}\n"
            "Run run_validation.py first."
        )

    with open(results_path) as f:
        data = json.load(f)

    return data["ground_truth"], data["results"]


# ---------------------------------------------------------------------------
# PLI extraction helpers
# ---------------------------------------------------------------------------

def extract_pattern_pli(
    pli_matrix: np.ndarray,
    region1_channels: List[int],
    region2_channels: List[int],
) -> float:
    """Extract mean PLI between two channel groups from a PLI matrix.

    Computes the mean of all cross-region channel pairs.
    """
    values = []
    for i in region1_channels:
        for j in region2_channels:
            if i < pli_matrix.shape[0] and j < pli_matrix.shape[1]:
                values.append(pli_matrix[i, j])
    return float(np.mean(values)) if values else np.nan


def extract_background_pli(
    pli_matrix: np.ndarray,
    exclude_channels: List[int],
) -> float:
    """Extract mean PLI for non-coupled channel pairs (background level).

    This represents the noise floor / false positive rate.
    """
    n = pli_matrix.shape[0]
    values = []
    for i in range(n):
        for j in range(i + 1, n):
            if i not in exclude_channels and j not in exclude_channels:
                values.append(pli_matrix[i, j])
    return float(np.mean(values)) if values else np.nan


def build_results_dataframe(
    ground_truth: dict,
    results: list,
) -> pd.DataFrame:
    """Build a tidy DataFrame with PLI measurements for all conditions.

    Columns:
        artifact_level, method, pattern_name, band, expected_pli,
        sensor_pli_raw, sensor_pli_cleaned, source_pli,
        background_pli, pli_error, pli_relative_error
    """
    patterns = ground_truth["connectivity_patterns"]
    rows = []

    for r in results:
        method = r["method"]
        artifact_level = r["artifact_level"]

        for pattern in patterns:
            band = pattern["band"]
            expected = pattern["expected_pli"]
            r1 = pattern["region1_channels"]
            r2 = pattern["region2_channels"]
            all_coupled = r1 + r2

            # Raw sensor-level PLI (before preprocessing)
            raw_sensor_pli = np.nan
            raw_pli_data = r.get("raw_sensor_pli", {}).get(band)
            if raw_pli_data is not None:
                mat = np.array(raw_pli_data)
                raw_sensor_pli = extract_pattern_pli(mat, r1, r2)

            # Cleaned sensor-level PLI (after preprocessing)
            cleaned_sensor_pli = np.nan
            cleaned_pli_data = r.get("sensor_pli_matrices", {}).get(band)
            if cleaned_pli_data is not None:
                mat = np.array(cleaned_pli_data)
                cleaned_sensor_pli = extract_pattern_pli(mat, r1, r2)

            # Source-level PLI (full pipeline)
            # Note: source-level uses DK atlas regions (68x68), not sensor channels.
            # For source-level, we record the overall mean as a reference;
            # direct channel-to-region mapping requires atlas knowledge.
            source_pli = np.nan
            source_pli_data = r.get("pli_matrices", {}).get(band)
            if source_pli_data is not None:
                mat = np.array(source_pli_data)
                # For source-level, we use the overall mean as a proxy
                # since the exact channel-to-region mapping depends on
                # source localization results
                upper = mat[np.triu_indices_from(mat, k=1)]
                source_pli = float(np.mean(upper)) if len(upper) > 0 else np.nan

            # Background PLI (non-coupled pairs, sensor-level)
            background_pli = np.nan
            if cleaned_pli_data is not None:
                mat = np.array(cleaned_pli_data)
                background_pli = extract_background_pli(mat, all_coupled)

            # Error metrics
            pli_error = cleaned_sensor_pli - expected if not np.isnan(cleaned_sensor_pli) else np.nan
            pli_rel_error = (pli_error / expected * 100) if (not np.isnan(pli_error) and expected > 0) else np.nan

            rows.append({
                "artifact_level": artifact_level,
                "method": method,
                "pattern_name": pattern["name"],
                "band": band,
                "frequency_hz": pattern["frequency_hz"],
                "expected_pli": expected,
                "coupling_strength": pattern["coupling_strength"],
                "sensor_pli_raw": round(raw_sensor_pli, 4),
                "sensor_pli_cleaned": round(cleaned_sensor_pli, 4),
                "source_pli_mean": round(source_pli, 4),
                "background_pli": round(background_pli, 4),
                "pli_error": round(pli_error, 4) if not np.isnan(pli_error) else np.nan,
                "pli_relative_error_pct": round(pli_rel_error, 1) if not np.isnan(pli_rel_error) else np.nan,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def figure1_pli_recovery(df: pd.DataFrame, output_dir: Path):
    """Figure 1: PLI recovery accuracy across artifact severity levels.

    Bar chart showing expected vs recovered PLI for each pattern,
    grouped by artifact level, with separate panels for each method.
    """
    methods = df["method"].unique()
    levels = ["clean_high_snr", "light_medium_snr", "moderate_medium_snr", "heavy_low_snr"]
    levels = [l for l in levels if l in df["artifact_level"].unique()]
    patterns = df["pattern_name"].unique()

    level_labels = {
        "clean_high_snr": "Clean\n(10 dB)",
        "light_medium_snr": "Light\n(5 dB)",
        "moderate_medium_snr": "Moderate\n(5 dB)",
        "heavy_low_snr": "Heavy\n(2 dB)",
    }

    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5),
                             sharey=True, squeeze=False)

    colors = {
        "alpha_frontal_parietal": "#E74C3C",
        "beta_central_occipital": "#3498DB",
        "theta_left_right_temporal": "#2ECC71",
    }
    pattern_labels = {
        "alpha_frontal_parietal": r"Alpha (10 Hz, PLI$_{exp}$=0.8)",
        "beta_central_occipital": r"Beta (20 Hz, PLI$_{exp}$=0.5)",
        "theta_left_right_temporal": r"Theta (6 Hz, PLI$_{exp}$=0.3)",
    }

    for ax_idx, method in enumerate(methods):
        ax = axes[0, ax_idx]
        method_df = df[df["method"] == method]

        n_levels = len(levels)
        n_patterns = len(patterns)
        width = 0.2
        x = np.arange(n_levels)

        for p_idx, pattern in enumerate(patterns):
            pli_values = []
            for level in levels:
                row = method_df[
                    (method_df["artifact_level"] == level) &
                    (method_df["pattern_name"] == pattern)
                ]
                if len(row) > 0:
                    pli_values.append(row.iloc[0]["sensor_pli_cleaned"])
                else:
                    pli_values.append(np.nan)

            offset = (p_idx - n_patterns / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, pli_values, width,
                label=pattern_labels.get(pattern, pattern),
                color=colors.get(pattern, f"C{p_idx}"),
                edgecolor="white", linewidth=0.5,
                zorder=3,
            )

        # Draw expected PLI lines
        for p_idx, pattern in enumerate(patterns):
            expected = df[df["pattern_name"] == pattern]["expected_pli"].iloc[0]
            ax.axhline(y=expected, color=colors.get(pattern, "gray"),
                       linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel("Artifact Severity (SNR)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([level_labels.get(l, l) for l in levels], fontsize=9)
        ax.set_title(f"{'GEDAI' if method == 'gedai' else 'Traditional (ASR+ICLabel)'}",
                     fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax_idx == 0:
            ax.set_ylabel("Recovered PLI (sensor-level)", fontsize=11)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.suptitle("PLI Recovery Accuracy Across Artifact Severity",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = output_dir / "fig1_pli_recovery.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Also save as PDF for manuscript
    pdf_path = output_dir / "fig1_pli_recovery.pdf"
    fig2, axes2 = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5),
                                sharey=True, squeeze=False)
    # Recreate for PDF (matplotlib sometimes has issues reusing figures)
    for ax_idx, method in enumerate(methods):
        ax = axes2[0, ax_idx]
        method_df = df[df["method"] == method]
        n_levels = len(levels)
        n_patterns = len(patterns)
        width = 0.2
        x = np.arange(n_levels)
        for p_idx, pattern in enumerate(patterns):
            pli_values = []
            for level in levels:
                row = method_df[
                    (method_df["artifact_level"] == level) &
                    (method_df["pattern_name"] == pattern)
                ]
                if len(row) > 0:
                    pli_values.append(row.iloc[0]["sensor_pli_cleaned"])
                else:
                    pli_values.append(np.nan)
            offset = (p_idx - n_patterns / 2 + 0.5) * width
            ax.bar(x + offset, pli_values, width,
                   label=pattern_labels.get(pattern, pattern),
                   color=colors.get(pattern, f"C{p_idx}"),
                   edgecolor="white", linewidth=0.5, zorder=3)
        for p_idx, pattern in enumerate(patterns):
            expected = df[df["pattern_name"] == pattern]["expected_pli"].iloc[0]
            ax.axhline(y=expected, color=colors.get(pattern, "gray"),
                       linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xlabel("Artifact Severity (SNR)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([level_labels.get(l, l) for l in levels], fontsize=9)
        ax.set_title(f"{'GEDAI' if method == 'gedai' else 'Traditional (ASR+ICLabel)'}",
                     fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_idx == 0:
            ax.set_ylabel("Recovered PLI (sensor-level)", fontsize=11)
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig2.suptitle("PLI Recovery Accuracy Across Artifact Severity",
                  fontsize=14, fontweight="bold", y=1.02)
    fig2.tight_layout()
    fig2.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {pdf_path}")


def figure2_raw_vs_cleaned(df: pd.DataFrame, output_dir: Path):
    """Figure 2: Effect of preprocessing on PLI (raw vs cleaned).

    Shows how preprocessing preserves true connectivity while reducing
    spurious PLI from artifacts.
    """
    methods = df["method"].unique()
    patterns = df["pattern_name"].unique()
    levels = ["clean_high_snr", "light_medium_snr", "moderate_medium_snr", "heavy_low_snr"]
    levels = [l for l in levels if l in df["artifact_level"].unique()]

    level_short = {
        "clean_high_snr": "Clean",
        "light_medium_snr": "Light",
        "moderate_medium_snr": "Moderate",
        "heavy_low_snr": "Heavy",
    }

    fig, axes = plt.subplots(len(patterns), len(methods),
                             figsize=(6 * len(methods), 4 * len(patterns)),
                             sharey="row", squeeze=False)

    for p_idx, pattern in enumerate(patterns):
        pattern_df = df[df["pattern_name"] == pattern]
        expected = pattern_df["expected_pli"].iloc[0]

        for m_idx, method in enumerate(methods):
            ax = axes[p_idx, m_idx]
            method_df = pattern_df[pattern_df["method"] == method]

            x = np.arange(len(levels))
            raw_vals = [method_df[method_df["artifact_level"] == l]["sensor_pli_raw"].values[0]
                        if l in method_df["artifact_level"].values else np.nan
                        for l in levels]
            cleaned_vals = [method_df[method_df["artifact_level"] == l]["sensor_pli_cleaned"].values[0]
                            if l in method_df["artifact_level"].values else np.nan
                            for l in levels]
            bg_vals = [method_df[method_df["artifact_level"] == l]["background_pli"].values[0]
                       if l in method_df["artifact_level"].values else np.nan
                       for l in levels]

            width = 0.25
            ax.bar(x - width, raw_vals, width, label="Raw", color="#95A5A6", alpha=0.8, zorder=3)
            ax.bar(x, cleaned_vals, width, label="Cleaned", color="#2ECC71", alpha=0.8, zorder=3)
            ax.bar(x + width, bg_vals, width, label="Background", color="#E67E22", alpha=0.6, zorder=3)

            ax.axhline(y=expected, color="red", linestyle="--", alpha=0.7,
                       label=f"Expected ({expected})")

            ax.set_xticks(x)
            ax.set_xticklabels([level_short.get(l, l) for l in levels], fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if m_idx == 0:
                band = pattern_df["band"].iloc[0]
                ax.set_ylabel(f"{band.capitalize()}\n({pattern.split('_')[0]})", fontsize=10)
            if p_idx == 0:
                ax.set_title(f"{'GEDAI' if method == 'gedai' else 'Traditional'}",
                             fontsize=11, fontweight="bold")
            if p_idx == 0 and m_idx == len(methods) - 1:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
            if p_idx == len(patterns) - 1:
                ax.set_xlabel("Artifact Level", fontsize=10)

    fig.suptitle("Effect of Preprocessing on PLI Recovery",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = output_dir / "fig2_raw_vs_cleaned.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def figure3_method_comparison(df: pd.DataFrame, output_dir: Path):
    """Figure 3: Head-to-head comparison of GEDAI vs Traditional.

    Scatter plot of PLI recovery for each method, with identity line.
    """
    methods = df["method"].unique()
    if len(methods) < 2:
        print("  Skipping Figure 3 (need both methods for comparison)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)

    patterns = df["pattern_name"].unique()
    colors = {
        "alpha_frontal_parietal": "#E74C3C",
        "beta_central_occipital": "#3498DB",
        "theta_left_right_temporal": "#2ECC71",
    }
    markers = {
        "clean_high_snr": "o",
        "light_medium_snr": "s",
        "moderate_medium_snr": "D",
        "heavy_low_snr": "^",
    }
    level_labels = {
        "clean_high_snr": "Clean",
        "light_medium_snr": "Light",
        "moderate_medium_snr": "Moderate",
        "heavy_low_snr": "Heavy",
    }

    # Panel A: Sensor-level PLI comparison
    ax = axes[0, 0]
    for pattern in patterns:
        for level in df["artifact_level"].unique():
            trad = df[(df["method"] == "traditional") &
                      (df["pattern_name"] == pattern) &
                      (df["artifact_level"] == level)]
            gedai = df[(df["method"] == "gedai") &
                       (df["pattern_name"] == pattern) &
                       (df["artifact_level"] == level)]
            if len(trad) > 0 and len(gedai) > 0:
                ax.scatter(
                    trad.iloc[0]["sensor_pli_cleaned"],
                    gedai.iloc[0]["sensor_pli_cleaned"],
                    color=colors.get(pattern, "gray"),
                    marker=markers.get(level, "o"),
                    s=80, edgecolors="black", linewidth=0.5,
                    label=f"{pattern.split('_')[0].title()} / {level_labels.get(level, level)}",
                    zorder=3,
                )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Identity")
    ax.set_xlabel("Traditional PLI", fontsize=11)
    ax.set_ylabel("GEDAI PLI", fontsize=11)
    ax.set_title("A. Sensor-Level PLI", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, zorder=0)

    # Panel B: Error relative to ground truth
    ax = axes[0, 1]
    for method in methods:
        method_df = df[df["method"] == method]
        errors = method_df.groupby("artifact_level")["pli_error"].mean()
        levels_ordered = ["clean_high_snr", "light_medium_snr", "moderate_medium_snr", "heavy_low_snr"]
        levels_ordered = [l for l in levels_ordered if l in errors.index]
        vals = [errors[l] for l in levels_ordered]
        x = np.arange(len(levels_ordered))
        label = "GEDAI" if method == "gedai" else "Traditional"
        ax.plot(x, vals, "o-", label=label, markersize=8)

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.set_xticks(range(len(levels_ordered)))
    ax.set_xticklabels([level_labels.get(l, l) for l in levels_ordered], fontsize=9)
    ax.set_xlabel("Artifact Severity", fontsize=11)
    ax.set_ylabel("Mean PLI Error", fontsize=11)
    ax.set_title("B. Mean PLI Error vs Ground Truth", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, zorder=0)

    # Panel C: Background PLI (false positive rate)
    ax = axes[0, 2]
    for method in methods:
        method_df = df[df["method"] == method]
        bg = method_df.groupby("artifact_level")["background_pli"].mean()
        levels_ordered = ["clean_high_snr", "light_medium_snr", "moderate_medium_snr", "heavy_low_snr"]
        levels_ordered = [l for l in levels_ordered if l in bg.index]
        vals = [bg[l] for l in levels_ordered]
        x = np.arange(len(levels_ordered))
        label = "GEDAI" if method == "gedai" else "Traditional"
        ax.plot(x, vals, "o-", label=label, markersize=8)

    ax.set_xticks(range(len(levels_ordered)))
    ax.set_xticklabels([level_labels.get(l, l) for l in levels_ordered], fontsize=9)
    ax.set_xlabel("Artifact Severity", fontsize=11)
    ax.set_ylabel("Background PLI", fontsize=11)
    ax.set_title("C. False Positive Rate", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, zorder=0)

    fig.tight_layout()
    out_path = output_dir / "fig3_method_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def figure4_heatmap_matrices(df: pd.DataFrame, ground_truth: dict,
                              results: list, output_dir: Path):
    """Figure 4: PLI matrix heatmaps for visual inspection.

    Shows the sensor-level PLI matrix for each artifact level (one row
    per level, columns = raw / Traditional / GEDAI) in the alpha band.
    """
    levels = ["clean_high_snr", "light_medium_snr", "moderate_medium_snr", "heavy_low_snr"]
    target_band = "alpha"  # Focus on the strongest pattern
    methods = ["traditional", "gedai"]
    col_labels = ["Raw", "Traditional", "GEDAI"]

    # Filter to available levels
    available_levels = [l for l in levels if any(r["artifact_level"] == l for r in results)]
    if not available_levels:
        print("  Skipping Figure 4 (no results)")
        return

    n_rows = len(available_levels)
    n_cols = 3  # raw + 2 methods

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    level_labels = {
        "clean_high_snr": "Clean (10 dB)",
        "light_medium_snr": "Light (5 dB)",
        "moderate_medium_snr": "Moderate (5 dB)",
        "heavy_low_snr": "Heavy (2 dB)",
    }

    for row_idx, level in enumerate(available_levels):
        # Find results for this level
        trad_result = next((r for r in results
                            if r["artifact_level"] == level and r["method"] == "traditional"), None)
        gedai_result = next((r for r in results
                             if r["artifact_level"] == level and r["method"] == "gedai"), None)

        matrices = [None, None, None]

        # Raw PLI matrix
        if trad_result and trad_result.get("raw_sensor_pli", {}).get(target_band):
            matrices[0] = np.array(trad_result["raw_sensor_pli"][target_band])

        # Traditional cleaned PLI matrix
        if trad_result and trad_result.get("sensor_pli_matrices", {}).get(target_band):
            matrices[1] = np.array(trad_result["sensor_pli_matrices"][target_band])

        # GEDAI cleaned PLI matrix
        if gedai_result and gedai_result.get("sensor_pli_matrices", {}).get(target_band):
            matrices[2] = np.array(gedai_result["sensor_pli_matrices"][target_band])

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            mat = matrices[col_idx]

            if mat is not None:
                im = ax.imshow(mat, vmin=0, vmax=0.6, cmap="hot",
                               interpolation="nearest", aspect="equal")
                # Mark coupled channel regions
                patterns = ground_truth["connectivity_patterns"]
                alpha_pattern = next((p for p in patterns if p["band"] == target_band), None)
                if alpha_pattern:
                    r1 = alpha_pattern["region1_channels"]
                    r2 = alpha_pattern["region2_channels"]
                    # Draw rectangles around coupled regions
                    from matplotlib.patches import Rectangle
                    rect1 = Rectangle((min(r2) - 0.5, min(r1) - 0.5),
                                      max(r2) - min(r2) + 1, max(r1) - min(r1) + 1,
                                      linewidth=1.5, edgecolor="cyan",
                                      facecolor="none", linestyle="--")
                    rect2 = Rectangle((min(r1) - 0.5, min(r2) - 0.5),
                                      max(r1) - min(r1) + 1, max(r2) - min(r2) + 1,
                                      linewidth=1.5, edgecolor="cyan",
                                      facecolor="none", linestyle="--")
                    ax.add_patch(rect1)
                    ax.add_patch(rect2)
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="gray")

            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(level_labels.get(level, level), fontsize=10)

            ax.set_xticks([])
            ax.set_yticks([])

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="PLI")

    fig.suptitle(f"Sensor-Level {target_band.capitalize()}-Band PLI Matrices",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    out_path = output_dir / "fig4_pli_heatmaps.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def generate_results_table(df: pd.DataFrame, output_dir: Path):
    """Generate Table 1 for the manuscript: quantitative validation results.

    Saves as CSV and LaTeX format.
    """
    level_order = ["clean_high_snr", "light_medium_snr", "moderate_medium_snr", "heavy_low_snr"]
    level_labels = {
        "clean_high_snr": "Clean (10 dB)",
        "light_medium_snr": "Light (5 dB)",
        "moderate_medium_snr": "Moderate (5 dB)",
        "heavy_low_snr": "Heavy (2 dB)",
    }

    # Create summary table rows
    table_rows = []
    for level in level_order:
        for method in df["method"].unique():
            subset = df[(df["artifact_level"] == level) & (df["method"] == method)]
            if len(subset) == 0:
                continue

            method_label = "GEDAI" if method == "gedai" else "Traditional"

            for _, row in subset.iterrows():
                table_rows.append({
                    "Artifact Level": level_labels.get(level, level),
                    "Method": method_label,
                    "Pattern": row["pattern_name"].replace("_", " ").title(),
                    "Band": row["band"].capitalize(),
                    "Expected PLI": row["expected_pli"],
                    "Raw PLI": row["sensor_pli_raw"],
                    "Cleaned PLI": row["sensor_pli_cleaned"],
                    "Background PLI": row["background_pli"],
                    "Error": row["pli_error"],
                    "Rel. Error (%)": row["pli_relative_error_pct"],
                })

    table_df = pd.DataFrame(table_rows)

    # Save CSV
    csv_path = output_dir / "table1_validation_results.csv"
    table_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved: {csv_path}")

    # Save LaTeX
    latex_path = output_dir / "table1_validation_results.tex"
    latex_str = table_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Pipeline validation results: PLI recovery across artifact severity levels.",
        label="tab:validation",
        column_format="lllccccccc",
        na_rep="--",
    )
    with open(latex_path, "w") as f:
        f.write(latex_str)
    print(f"  Saved: {latex_path}")

    # Also create a compact summary table
    summary_rows = []
    for method in df["method"].unique():
        method_label = "GEDAI" if method == "gedai" else "Traditional"
        for level in level_order:
            subset = df[(df["artifact_level"] == level) & (df["method"] == method)]
            if len(subset) == 0:
                continue

            mean_error = subset["pli_error"].mean()
            mean_abs_error = subset["pli_error"].abs().mean()
            mean_rel_error = subset["pli_relative_error_pct"].mean()
            mean_bg = subset["background_pli"].mean()

            summary_rows.append({
                "Method": method_label,
                "Artifact Level": level_labels.get(level, level),
                "MAE": round(mean_abs_error, 4),
                "Mean Error": round(mean_error, 4),
                "Mean Rel. Error (%)": round(mean_rel_error, 1) if not np.isnan(mean_rel_error) else np.nan,
                "Mean Background PLI": round(mean_bg, 4),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "table2_summary.csv"
    summary_df.to_csv(summary_csv, index=False, float_format="%.4f")
    print(f"  Saved: {summary_csv}")

    return table_df, summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_dir: Optional[Path] = None):
    """Analyze validation results and generate manuscript outputs.

    Parameters
    ----------
    results_dir : Path, optional
        Directory containing validation_results.json.
        Defaults to scripts/validation/validation_results/.
    """
    if results_dir is None:
        results_dir = Path(__file__).resolve().parent / "validation_results"

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NeuroStat Validation: Analysis & Figure Generation")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Figures output:    {figures_dir}")

    # Load data
    print("\nLoading validation data...")
    ground_truth, results = load_validation_data(results_dir)
    print(f"  Datasets: {len(ground_truth['datasets'])}")
    print(f"  Results:  {len(results)} processing runs")

    # Build tidy DataFrame
    print("\nBuilding results DataFrame...")
    df = build_results_dataframe(ground_truth, results)
    print(f"  Rows: {len(df)}")
    print(f"  Methods: {df['method'].unique().tolist()}")
    print(f"  Levels: {df['artifact_level'].unique().tolist()}")

    # Save raw DataFrame
    raw_df_path = results_dir / "validation_dataframe.csv"
    df.to_csv(raw_df_path, index=False, float_format="%.4f")
    print(f"  Saved DataFrame: {raw_df_path}")

    # Generate figures
    print("\nGenerating manuscript figures...")

    print("\n  Figure 1: PLI recovery accuracy...")
    try:
        figure1_pli_recovery(df, figures_dir)
    except Exception as exc:
        print(f"    ERROR: {exc}")

    print("\n  Figure 2: Raw vs cleaned PLI...")
    try:
        figure2_raw_vs_cleaned(df, figures_dir)
    except Exception as exc:
        print(f"    ERROR: {exc}")

    print("\n  Figure 3: Method comparison...")
    try:
        figure3_method_comparison(df, figures_dir)
    except Exception as exc:
        print(f"    ERROR: {exc}")

    print("\n  Figure 4: PLI heatmaps...")
    try:
        figure4_heatmap_matrices(df, ground_truth, results, figures_dir)
    except Exception as exc:
        print(f"    ERROR: {exc}")

    # Generate tables
    print("\nGenerating manuscript tables...")
    try:
        table_df, summary_df = generate_results_table(df, figures_dir)
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("VALIDATION ANALYSIS SUMMARY")
    print("=" * 60)
    print("\nPer-condition results:")
    print(df[["artifact_level", "method", "pattern_name", "expected_pli",
              "sensor_pli_cleaned", "pli_error"]].to_string(index=False))

    print("\n\nMean absolute error by method and artifact level:")
    pivot = df.pivot_table(
        values="pli_error",
        index="artifact_level",
        columns="method",
        aggfunc=lambda x: np.abs(x).mean(),
    )
    print(pivot.to_string(float_format="%.4f"))

    print(f"\n\nAll outputs saved to: {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze NeuroStat validation results and generate figures."
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    main(args.results_dir)
