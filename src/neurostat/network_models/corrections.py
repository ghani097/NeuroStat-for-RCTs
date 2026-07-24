"""Explicit multiple-testing families and deterministic adjustments."""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .config import NetworkModelConfig
from .data import NetworkModelError


def adjust_pvalues(values: Iterable[float], method: str) -> np.ndarray:
    """Match R's p.adjust definitions for BH, Holm, and Bonferroni."""
    p = np.asarray(tuple(values), dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise NetworkModelError("correction families require finite raw p-values inside [0, 1]")
    count = len(p)
    if method == "bonferroni":
        return np.minimum(1.0, p * count)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    if method == "BH":
        adjusted_ranked = ranked * count / np.arange(1, count + 1)
        adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    elif method == "holm":
        adjusted_ranked = ranked * np.arange(count, 0, -1)
        adjusted_ranked = np.maximum.accumulate(adjusted_ranked)
    else:
        raise NetworkModelError(f"unsupported correction method: {method}")
    adjusted_ranked = np.minimum(1.0, adjusted_ranked)
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = adjusted_ranked
    return adjusted


def apply_correction_families(
    config: NetworkModelConfig,
    anova_rows: list[dict[str, str]],
    between_rows: list[dict[str, str]],
    within_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Assign every inferential row to exactly one explicit family."""
    rows_by_target = {"anova": anova_rows, "between": between_rows, "within": within_rows}
    output: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str, str, str, str, str]] = set()
    for family in config.families:
        for correction in family.correction_families:
            identifiers = correction.terms if correction.target == "anova" else correction.contrasts
            candidates = [
                row
                for row in rows_by_target[correction.target]
                if row.get("analysis_family") == family.name
                and row.get("term" if correction.target == "anova" else "contrast") in identifiers
                and row.get("band") in correction.bands
                and row.get("network") in correction.networks
            ]
            expected = len(identifiers) * len(correction.bands) * len(correction.networks)
            # Within contrasts may intentionally select different arm subsets,
            # so calculate their exact expanded cardinality directly.
            if correction.target == "within":
                expected = len(correction.bands) * len(correction.networks) * sum(
                    len(contrast.arms)
                    for contrast in family.within_arm_contrasts
                    if contrast.name in identifiers
                )
            if len(candidates) != int(expected):
                raise NetworkModelError(
                    f"correction family {family.name}/{correction.name} expected {int(expected)} "
                    f"raw rows but received {len(candidates)}"
                )
            raw_p = [float(row["p_value"]) for row in candidates]
            for method in correction.methods:
                adjusted = adjust_pvalues(raw_p, method)
                for row, value in zip(candidates, adjusted, strict=True):
                    arm_or_session = (
                        row.get("term", "")
                        if correction.target == "anova"
                        else row.get("arm", row.get("session", ""))
                    )
                    identity = (
                        correction.target,
                        family.name,
                        row["band"],
                        row["network"],
                        row.get("term", row.get("contrast", "")),
                        arm_or_session,
                    )
                    seen_rows.add(identity)
                    output.append(
                        {
                            "analysis_family": family.name,
                            "correction_family": correction.name,
                            "target": correction.target,
                            "method": method,
                            "family_size": len(candidates),
                            "condition": row["condition"],
                            "band": row["band"],
                            "network": row["network"],
                            "term": row.get("term", ""),
                            "contrast": row.get("contrast", ""),
                            "arm": row.get("arm", ""),
                            "session": row.get("session", ""),
                            "raw_p_value": format(float(row["p_value"]), ".17g"),
                            "adjusted_p_value": format(float(value), ".17g"),
                        }
                    )
    expected_raw = len(anova_rows) + len(between_rows) + len(within_rows)
    # Multiple methods deliberately repeat a membership in output, while the
    # typed config parser already prohibits overlap between correction families.
    if len(seen_rows) != expected_raw:
        raise NetworkModelError(
            f"correction membership covers {len(seen_rows)} of {expected_raw} inferential rows"
        )
    return output
