"""Independent Python reconstruction of exported mixed-model estimands."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import NetworkModelConfig
from .corrections import adjust_pvalues
from .data import NetworkModelError, VerifiedConnectivity
from ..project import CleanProject


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise NetworkModelError(f"required statistical output is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise NetworkModelError(f"statistical output has no header: {path}")
        return [dict(row) for row in reader]


def _finite(row: dict[str, str], field: str, description: str) -> float:
    try:
        result = float(row[field])
    except (KeyError, TypeError, ValueError):
        raise NetworkModelError(f"{description} has invalid {field}") from None
    if not math.isfinite(result):
        raise NetworkModelError(f"{description} has non-finite {field}")
    return result


def _close(actual: float, expected: float, config: NetworkModelConfig, description: str) -> None:
    if not np.isclose(
        actual,
        expected,
        rtol=config.validation.relative_tolerance,
        atol=config.validation.absolute_tolerance,
    ):
        raise NetworkModelError(
            f"independent validation failed for {description}: R={actual:.17g}, Python={expected:.17g}"
        )


def validate_statistical_exports(
    project: CleanProject,
    config: NetworkModelConfig,
    upstream: VerifiedConnectivity,
    summary_rows: tuple[dict[str, Any], ...],
    count_rows: tuple[dict[str, Any], ...],
    r_output_dir: Path,
    correction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate cells, orientations, counts, and correction membership independently."""
    overview = read_csv_rows(r_output_dir / "model_overview.csv")
    emmeans = read_csv_rows(r_output_dir / "estimated_marginal_means.csv")
    between = read_csv_rows(r_output_dir / "between_arm_contrasts.csv")
    within = read_csv_rows(r_output_dir / "within_arm_contrasts.csv")
    residual_observations = read_csv_rows(r_output_dir / "residual_observations.csv")
    fixed = read_csv_rows(r_output_dir / "fixed_effects.csv")
    anova = read_csv_rows(r_output_dir / "anova_type3.csv")
    residual_diagnostics = read_csv_rows(r_output_dir / "residual_diagnostics.csv")
    sensitivity_fixed = read_csv_rows(r_output_dir / "sensitivity_models.csv")
    sensitivity_anova = read_csv_rows(r_output_dir / "sensitivity_anova_type3.csv")
    sensitivity_between = read_csv_rows(r_output_dir / "sensitivity_between_arm_contrasts.csv")
    allfit = read_csv_rows(r_output_dir / "allfit_diagnostics.csv")
    influence = read_csv_rows(r_output_dir / "influence_diagnostics.csv")
    loo = read_csv_rows(r_output_dir / "leave_one_out.csv")
    loo_summary = read_csv_rows(r_output_dir / "leave_one_out_summary.csv")

    family_by_name = {family.name: family for family in config.families}
    all_r_tables = {
        "fixed effects": fixed,
        "Type-III ANOVA": anova,
        "marginal means": emmeans,
        "between contrasts": between,
        "within contrasts": within,
        "model overview": overview,
        "residual diagnostics": residual_diagnostics,
        "residual observations": residual_observations,
        "sensitivity fixed effects": sensitivity_fixed,
        "sensitivity Type-III ANOVA": sensitivity_anova,
        "sensitivity between contrasts": sensitivity_between,
        "allFit diagnostics": allfit,
        "influence diagnostics": influence,
        "leave-one-out": loo,
        "leave-one-out summary": loo_summary,
    }
    for description, rows in all_r_tables.items():
        for row in rows:
            family = family_by_name.get(row.get("analysis_family", ""))
            if family is None or row.get("condition") != family.condition:
                raise NetworkModelError(
                    f"{description} contains an unknown family or wrong configured condition"
                )
    network_by_name = config.networks_by_name
    project_by_id = {record.record_id: record for record in project.records}
    expected_summary_keys: set[tuple[str, str, str, str]] = set()
    for family in config.families:
        records_by_subject: dict[str, dict[str, str]] = defaultdict(dict)
        for record in project.records:
            if record.key.condition == family.condition and record.key.session in family.sessions:
                records_by_subject[record.key.participant_id][record.key.session] = record.record_id
        for subject, by_session in records_by_subject.items():
            if set(by_session) == set(family.sessions):
                for record_id in by_session.values():
                    for band in family.bands:
                        for network in family.networks:
                            expected_summary_keys.add((family.name, record_id, band, network))
    actual_summary_keys = [
        (
            str(row["analysis_family"]),
            str(row["record_id"]),
            str(row["band"]),
            str(row["network"]),
        )
        for row in summary_rows
    ]
    if len(set(actual_summary_keys)) != len(actual_summary_keys) or set(actual_summary_keys) != expected_summary_keys:
        raise NetworkModelError("network summary rows do not have the exact independently expected identities")

    matrix_cache: dict[str, np.ndarray] = {}
    for row in summary_rows:
        family = family_by_name.get(str(row["analysis_family"]))
        record = project_by_id.get(str(row["record_id"]))
        if family is None or record is None:
            raise NetworkModelError("network summary references an unknown family or record")
        verified = upstream.records[record.record_id]
        if (
            str(row["condition"]) != family.condition
            or str(row["subject_id"]) != record.key.participant_id
            or str(row["arm"]) != record.key.arm
            or str(row["session"]) != record.key.session
            or str(row["record_sha256"]) != verified.connectivity_sha256
        ):
            raise NetworkModelError(f"network summary metadata differs from project record {record.record_id}")
        if record.record_id not in matrix_cache:
            import h5py

            with h5py.File(verified.connectivity_path, "r") as file:
                matrix_cache[record.record_id] = np.asarray(file["matrix"][:], dtype=np.float64)
        matrices = matrix_cache[record.record_id]
        try:
            band_index = verified.connectivity_header.bands.index(str(row["band"]))
            labels = verified.connectivity_header.labels
            indexes = tuple(labels.index(label) for label in network_by_name[str(row["network"])].labels)
        except (KeyError, ValueError):
            raise NetworkModelError("network summary band/network labels are inconsistent with verified connectivity") from None
        submatrix = matrices[band_index][np.ix_(indexes, indexes)]
        expected_raw = float(submatrix[np.triu_indices(len(indexes), 1)].mean())
        if family.boundary_observation_count == "scout_samples":
            expected_count = verified.scout_header.samples
            expected_policy = "verified_upstream_scout_samples"
        else:
            expected_count = verified.method_provenance.get("window_count")
            expected_policy = "verified_complete_connectivity_windows"
        if type(expected_count) is not int or int(row["sample_count"]) != expected_count:
            raise NetworkModelError(f"network summary observation count is invalid for {record.record_id}")
        epsilon = 1.0 / (2.0 * expected_count)
        if expected_raw == 0:
            expected_adjusted, expected_boundary = epsilon, "exact_zero_to_epsilon"
        elif expected_raw == 1:
            expected_adjusted, expected_boundary = 1.0 - epsilon, "exact_one_to_one_minus_epsilon"
        else:
            expected_adjusted, expected_boundary = expected_raw, "unchanged_interior"
        if str(row["sample_count_policy"]) != expected_policy or str(row["boundary_policy"]) != expected_boundary:
            raise NetworkModelError(f"network summary boundary policy is invalid for {record.record_id}")
        _close(float(row["raw_pli"]), expected_raw, config, f"network mean {record.record_id}")
        _close(float(row["adjusted_pli"]), expected_adjusted, config, f"adjusted network mean {record.record_id}")
        _close(float(row["z_pli"]), float(np.arctanh(expected_adjusted)), config, f"Fisher z {record.record_id}")

    cell_values: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    subjects: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in summary_rows:
        key = (
            str(row["analysis_family"]),
            str(row["band"]),
            str(row["network"]),
            str(row["arm"]),
            str(row["session"]),
        )
        cell_values[key].append(float(row["z_pli"]))
        subjects[(key[0], key[3], key[4])].add(str(row["subject_id"]))
    expected_job_keys = {
        (family.name, band, network)
        for family in config.families
        for band in family.bands
        for network in family.networks
    }
    actual_job_keys = [(row["analysis_family"], row["band"], row["network"]) for row in overview]
    expected_jobs = len(expected_job_keys)
    if len(set(actual_job_keys)) != len(actual_job_keys) or set(actual_job_keys) != expected_job_keys:
        raise NetworkModelError("model overview does not contain the exact configured job identities")
    overview_sigma: dict[tuple[str, str, str], float] = {}
    for row in overview:
        if row.get("converged", "").lower() not in {"true", "t", "1"}:
            raise NetworkModelError(
                f"primary model did not converge for {row.get('analysis_family')}/{row.get('band')}/{row.get('network')}"
            )
        family = next(
            (item for item in config.families if item.name == row.get("analysis_family")),
            None,
        )
        if family is None:
            raise NetworkModelError("model overview contains an unknown analysis family")
        expected_rows = sum(
            len(values)
            for key, values in cell_values.items()
            if key[:3] == (family.name, row["band"], row["network"])
        )
        if int(row["n_rows"]) != expected_rows:
            raise NetworkModelError("R model row count differs from the independently reconstructed cohort")
        expected_subjects = len(
            {
                str(item["subject_id"])
                for item in summary_rows
                if item["analysis_family"] == family.name
            }
        )
        if (
            int(row["subjects"]) != expected_subjects
            or int(row["arms"]) != len(project.study.arms)
            or int(row["sessions"]) != len(family.sessions)
        ):
            raise NetworkModelError("R model overview factor counts differ from the frozen cohort/config")
        sigma = _finite(row, "residual_sd", "model overview")
        if sigma <= 0:
            raise NetworkModelError(
                f"model overview residual SD is not positive for {(row['analysis_family'], row['band'], row['network'])}"
            )
        overview_sigma[(row["analysis_family"], row["band"], row["network"])] = sigma
    for description, rows in (("fixed-effects", fixed), ("Type-III ANOVA", anova)):
        keys = {(row["analysis_family"], row["band"], row["network"]) for row in rows}
        if keys != expected_job_keys:
            raise NetworkModelError(f"primary {description} output does not cover the exact model jobs")
    for job_key in expected_job_keys:
        family = family_by_name[job_key[0]]
        fixed_job = [
            row
            for row in fixed
            if (row["analysis_family"], row["band"], row["network"]) == job_key
        ]
        if (
            len(fixed_job) != 2 * len(family.sessions)
            or len({row["term"] for row in fixed_job}) != len(fixed_job)
        ):
            raise NetworkModelError(f"primary fixed-effects cardinality is wrong for {job_key}")
        anova_job = [
            row
            for row in anova
            if (row["analysis_family"], row["band"], row["network"]) == job_key
        ]
        if {row["term"] for row in anova_job} != {"arm", "session", "arm:session"} or len(anova_job) != 3:
            raise NetworkModelError(f"primary Type-III terms are wrong for {job_key}")
    residual_keys = {
        (row["analysis_family"], row["band"], row["network"])
        for row in residual_diagnostics
    }
    if config.diagnostics.residuals and residual_keys != expected_job_keys:
        raise NetworkModelError("residual diagnostics do not cover the exact model jobs")
    if config.diagnostics.residuals and len(residual_diagnostics) != len(expected_job_keys):
        raise NetworkModelError("residual diagnostics must contain exactly one summary per model job")
    if not config.diagnostics.residuals and (residual_diagnostics or residual_observations):
        raise NetworkModelError("residual outputs were emitted although diagnostics.residuals is false")

    expected_emmeans = {
        key: float(np.mean(values))
        for key, values in cell_values.items()
    }
    emmean_keys = [
        (row["analysis_family"], row["band"], row["network"], row["arm"], row["session"])
        for row in emmeans
    ]
    if len(set(emmean_keys)) != len(emmean_keys) or set(emmean_keys) != set(expected_emmeans):
        raise NetworkModelError("R marginal-mean rows do not match the exact configured model cells")
    for row in emmeans:
        key = (
            row["analysis_family"],
            row["band"],
            row["network"],
            row["arm"],
            row["session"],
        )
        if key not in expected_emmeans:
            raise NetworkModelError(f"R marginal means contain an unexpected cell: {key}")
        _close(_finite(row, "emmean", "marginal mean"), expected_emmeans[key], config, f"cell mean {key}")
        _close(_finite(row, "raw_emm", "marginal mean"), math.tanh(_finite(row, "emmean", "marginal mean")), config, f"raw EMM {key}")
        _close(_finite(row, "raw_lower_cl", "marginal mean"), math.tanh(_finite(row, "lower_cl", "marginal mean")), config, f"raw EMM lower {key}")
        _close(_finite(row, "raw_upper_cl", "marginal mean"), math.tanh(_finite(row, "upper_cl", "marginal mean")), config, f"raw EMM upper {key}")

    expected_between: dict[tuple[str, str, str, str], tuple[float, tuple[str, str, str]]] = {}
    expected_within: dict[tuple[str, str, str, str, str], tuple[float, tuple[str, str]]] = {}
    for family in config.families:
        for band in family.bands:
            for network in family.networks:
                for contrast in family.between_arm_contrasts:
                    numerator = expected_emmeans[
                        (family.name, band, network, contrast.numerator_arm, contrast.session)
                    ]
                    denominator = expected_emmeans[
                        (family.name, band, network, contrast.denominator_arm, contrast.session)
                    ]
                    expected_between[(family.name, band, network, contrast.name)] = (
                        numerator - denominator,
                        (contrast.session, contrast.numerator_arm, contrast.denominator_arm),
                    )
                for contrast in family.within_arm_contrasts:
                    for arm in contrast.arms:
                        numerator = expected_emmeans[
                            (family.name, band, network, arm, contrast.numerator_session)
                        ]
                        denominator = expected_emmeans[
                            (family.name, band, network, arm, contrast.denominator_session)
                        ]
                        expected_within[(family.name, band, network, contrast.name, arm)] = (
                            numerator - denominator,
                            (contrast.numerator_session, contrast.denominator_session),
                        )
    between_keys = [
        (row["analysis_family"], row["band"], row["network"], row["contrast"])
        for row in between
    ]
    within_keys = [
        (row["analysis_family"], row["band"], row["network"], row["contrast"], row["arm"])
        for row in within
    ]
    if (
        len(set(between_keys)) != len(between_keys)
        or set(between_keys) != set(expected_between)
        or len(set(within_keys)) != len(within_keys)
        or set(within_keys) != set(expected_within)
    ):
        raise NetworkModelError("R contrast rows do not match the explicitly requested estimands")
    for row in between:
        key = (row["analysis_family"], row["band"], row["network"], row["contrast"])
        expected_value, metadata = expected_between[key]
        if (row["session"], row["numerator_arm"], row["denominator_arm"]) != metadata:
            raise NetworkModelError(f"R between-arm contrast orientation metadata is wrong for {key}")
        estimate = _finite(row, "estimate", "between contrast")
        sigma = _finite(row, "residual_sigma", "between contrast")
        if sigma <= 0:
            raise NetworkModelError(f"R between-arm residual sigma is not positive for {key}")
        _close(estimate, expected_value, config, f"between contrast {key}")
        _close(sigma, overview_sigma[key[:3]], config, f"between residual sigma {key}")
        _close(_finite(row, "standardized_estimate", "between contrast"), estimate / sigma, config, f"standardized between contrast {key}")
    for row in within:
        key = (row["analysis_family"], row["band"], row["network"], row["contrast"], row["arm"])
        expected_value, metadata = expected_within[key]
        if (row["numerator_session"], row["denominator_session"]) != metadata:
            raise NetworkModelError(f"R within-arm contrast orientation metadata is wrong for {key}")
        estimate = _finite(row, "estimate", "within contrast")
        sigma = _finite(row, "residual_sigma", "within contrast")
        if sigma <= 0:
            raise NetworkModelError(f"R within-arm residual sigma is not positive for {key}")
        _close(estimate, expected_value, config, f"within contrast {key}")
        _close(sigma, overview_sigma[key[:3]], config, f"within residual sigma {key}")
        _close(_finite(row, "standardized_estimate", "within contrast"), estimate / sigma, config, f"standardized within contrast {key}")

    if config.diagnostics.residuals:
        expected_residual_keys = {
            (
                str(row["analysis_family"]),
                band,
                network,
                str(row["record_id"]),
            )
            for row in summary_rows
            for band in family_by_name[str(row["analysis_family"])].bands
            for network in family_by_name[str(row["analysis_family"])].networks
            if str(row["band"]) == band and str(row["network"]) == network
        }
        actual_residual_keys = [
            (row["analysis_family"], row["band"], row["network"], row["record_id"])
            for row in residual_observations
        ]
        if (
            len(set(actual_residual_keys)) != len(actual_residual_keys)
            or set(actual_residual_keys) != expected_residual_keys
        ):
            raise NetworkModelError("residual observation carrier does not match every modeled record/job")
        summary_z = {
            (str(row["analysis_family"]), str(row["band"]), str(row["network"]), str(row["record_id"])): float(row["z_pli"])
            for row in summary_rows
        }
        for row in residual_observations:
            key = (row["analysis_family"], row["band"], row["network"], row["record_id"])
            z_value = _finite(row, "z_pli", "residual observation")
            fitted = _finite(row, "fitted", "residual observation")
            residual = _finite(row, "residual", "residual observation")
            _close(z_value, summary_z[key], config, f"residual source z {key}")
            _close(residual, z_value - fitted, config, f"residual identity {key}")
            _close(
                _finite(row, "standardized_residual", "residual observation"),
                residual / overview_sigma[key[:3]],
                config,
                f"standardized residual {key}",
            )
    elif residual_observations:
        raise NetworkModelError("residual observations were emitted although diagnostics.residuals is false")

    expected_sensitivity_names: set[str] = set()
    if config.diagnostics.raw_scale:
        expected_sensitivity_names.add("raw_scale")
    winsor_name = f"cellwise_raw_pli_robust_winsor_z_{config.diagnostics.winsor_robust_z:g}"
    if config.diagnostics.winsorized:
        expected_sensitivity_names.add(winsor_name)
    for description, rows in (
        ("fixed-effects", sensitivity_fixed),
        ("Type-III ANOVA", sensitivity_anova),
    ):
        actual_names = {row["sensitivity"] for row in rows}
        if actual_names != expected_sensitivity_names:
            raise NetworkModelError(
                f"sensitivity {description} outputs do not match configured sensitivity types"
            )
        for sensitivity_name in expected_sensitivity_names:
            jobs = {
                (row["analysis_family"], row["band"], row["network"])
                for row in rows
                if row["sensitivity"] == sensitivity_name
            }
            if jobs != expected_job_keys:
                raise NetworkModelError(
                    f"sensitivity {description} {sensitivity_name} does not cover exact model jobs"
                )
            for job_key in expected_job_keys:
                family = family_by_name[job_key[0]]
                job_rows = [
                    row
                    for row in rows
                    if row["sensitivity"] == sensitivity_name
                    and (row["analysis_family"], row["band"], row["network"]) == job_key
                ]
                if description == "fixed-effects":
                    if (
                        len(job_rows) != 2 * len(family.sessions)
                        or len({row["term"] for row in job_rows}) != len(job_rows)
                    ):
                        raise NetworkModelError(
                            f"sensitivity fixed-effects cardinality is wrong for {sensitivity_name}/{job_key}"
                        )
                elif (
                    {row["term"] for row in job_rows} != {"arm", "session", "arm:session"}
                    or len(job_rows) != 3
                ):
                    raise NetworkModelError(
                        f"sensitivity Type-III terms are wrong for {sensitivity_name}/{job_key}"
                    )

    raw_cells: dict[tuple[str, str, str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for row in summary_rows:
        raw_cells[
            (
                str(row["analysis_family"]),
                str(row["band"]),
                str(row["network"]),
                str(row["arm"]),
                str(row["session"]),
            )
        ].append((float(row["raw_pli"]), int(row["sample_count"])))
    raw_cell_means = {key: float(np.mean([item[0] for item in values])) for key, values in raw_cells.items()}
    winsor_cell_means: dict[tuple[str, str, str, str, str], float] = {}
    winsor_flags_by_job: dict[tuple[str, str, str], int] = defaultdict(int)
    for key, pairs in raw_cells.items():
        values = np.asarray([item[0] for item in pairs], dtype=np.float64)
        center = float(np.median(values))
        scale = float(np.median(np.abs(values - center)) * 1.4826)
        if not math.isfinite(scale) or scale <= 0:
            scale = float((np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349)
        if (not math.isfinite(scale) or scale <= 0) and len(values) > 1:
            scale = float(np.std(values, ddof=1))
        winsor = values.copy()
        flags = np.zeros(len(values), dtype=bool)
        if math.isfinite(scale) and scale > 0:
            threshold = config.diagnostics.winsor_robust_z
            flags = np.abs((values - center) / scale) > threshold
            winsor = np.clip(values, center - threshold * scale, center + threshold * scale)
        transformed: list[float] = []
        for value, (_, count) in zip(winsor, pairs, strict=True):
            epsilon = 1.0 / (2.0 * count)
            adjusted = epsilon if value == 0 else (1.0 - epsilon if value == 1 else float(value))
            transformed.append(float(np.arctanh(adjusted)))
        winsor_cell_means[key] = float(np.mean(transformed))
        winsor_flags_by_job[key[:3]] += int(flags.sum())

    expected_sensitivity_between: dict[
        tuple[str, str, str, str, str],
        tuple[float, tuple[str, str, str], int],
    ] = {}
    for family in config.families:
        for band in family.bands:
            for network in family.networks:
                for contrast in family.between_arm_contrasts:
                    orientation = (contrast.session, contrast.numerator_arm, contrast.denominator_arm)
                    if config.diagnostics.raw_scale:
                        expected_sensitivity_between[
                            ("raw_scale", family.name, band, network, contrast.name)
                        ] = (
                            raw_cell_means[(family.name, band, network, contrast.numerator_arm, contrast.session)]
                            - raw_cell_means[(family.name, band, network, contrast.denominator_arm, contrast.session)],
                            orientation,
                            0,
                        )
                    if config.diagnostics.winsorized:
                        expected_sensitivity_between[
                            (winsor_name, family.name, band, network, contrast.name)
                        ] = (
                            winsor_cell_means[(family.name, band, network, contrast.numerator_arm, contrast.session)]
                            - winsor_cell_means[(family.name, band, network, contrast.denominator_arm, contrast.session)],
                            orientation,
                            winsor_flags_by_job[(family.name, band, network)],
                        )
    actual_sensitivity_keys = [
        (
            row["sensitivity"],
            row["analysis_family"],
            row["band"],
            row["network"],
            row["contrast"],
        )
        for row in sensitivity_between
    ]
    if (
        len(set(actual_sensitivity_keys)) != len(actual_sensitivity_keys)
        or set(actual_sensitivity_keys) != set(expected_sensitivity_between)
    ):
        raise NetworkModelError("sensitivity planned-between output has wrong identities/cardinality")
    for row in sensitivity_between:
        key = (
            row["sensitivity"],
            row["analysis_family"],
            row["band"],
            row["network"],
            row["contrast"],
        )
        expected_estimate, orientation, flagged = expected_sensitivity_between[key]
        if (row["session"], row["numerator_arm"], row["denominator_arm"]) != orientation:
            raise NetworkModelError(f"sensitivity between contrast orientation is wrong for {key}")
        estimate = _finite(row, "estimate", "sensitivity between contrast")
        sigma = _finite(row, "residual_sigma", "sensitivity between contrast")
        if sigma <= 0:
            raise NetworkModelError(f"sensitivity residual sigma is not positive for {key}")
        _close(estimate, expected_estimate, config, f"sensitivity between contrast {key}")
        _close(
            _finite(row, "standardized_estimate", "sensitivity between contrast"),
            estimate / sigma,
            config,
            f"standardized sensitivity contrast {key}",
        )
        if int(row["flagged_values"]) != flagged:
            raise NetworkModelError(f"sensitivity flagged-value count is wrong for {key}")

    def diagnostic_job_keys(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
        return {(row["analysis_family"], row["band"], row["network"]) for row in rows}

    if config.diagnostics.all_fit:
        if diagnostic_job_keys(allfit) != expected_job_keys:
            raise NetworkModelError("allFit output does not provide an auditable row for every model job")
        for row in allfit:
            success = row.get("success", "").lower() in {"true", "t", "1"}
            if not success and not row.get("message", "").strip():
                raise NetworkModelError("failed allFit row has no diagnostic message")
    elif allfit:
        raise NetworkModelError("allFit output was emitted although diagnostics.all_fit is false")
    if config.diagnostics.influence:
        if diagnostic_job_keys(influence) != expected_job_keys:
            raise NetworkModelError("influence output does not provide an auditable row for every model job")
        for row in influence:
            if not row.get("subject_id", "").strip() and not row.get("message", "").strip():
                raise NetworkModelError("influence failure row has no diagnostic message")
    elif influence:
        raise NetworkModelError("influence output was emitted although diagnostics.influence is false")

    if config.diagnostics.leave_one_out:
        expected_loo_keys = {
            (family.name, band, network, subject, contrast.name)
            for family in config.families
            for band in family.bands
            for network in family.networks
            for subject in {
                str(row["subject_id"])
                for row in summary_rows
                if row["analysis_family"] == family.name
            }
            for contrast in family.between_arm_contrasts
        }
        actual_loo_keys = [
            (
                row["analysis_family"],
                row["band"],
                row["network"],
                row["excluded_subject_id"],
                row["contrast"],
            )
            for row in loo
        ]
        if len(set(actual_loo_keys)) != len(actual_loo_keys) or set(actual_loo_keys) != expected_loo_keys:
            raise NetworkModelError("leave-one-out output has wrong subject/contrast cardinality")
        primary_between = {
            (row["analysis_family"], row["band"], row["network"], row["contrast"]): row
            for row in between
        }
        for row in loo:
            key = (row["analysis_family"], row["band"], row["network"], row["contrast"])
            reference = primary_between[key]
            if (
                row["session"],
                row["numerator_arm"],
                row["denominator_arm"],
            ) != (
                reference["session"],
                reference["numerator_arm"],
                reference["denominator_arm"],
            ):
                raise NetworkModelError(f"leave-one-out orientation is wrong for {key}")
            _close(
                _finite(row, "full_estimate", "leave-one-out"),
                _finite(reference, "estimate", "primary between contrast"),
                config,
                f"leave-one-out full estimate {key}",
            )
            _close(
                _finite(row, "full_t_ratio", "leave-one-out"),
                _finite(reference, "t_ratio", "primary between contrast"),
                config,
                f"leave-one-out full t {key}",
            )
            estimate = _finite(row, "estimate", "leave-one-out")
            _close(
                _finite(row, "estimate_change", "leave-one-out"),
                estimate - _finite(row, "full_estimate", "leave-one-out"),
                config,
                f"leave-one-out estimate change {key}",
            )
        expected_loo_summary = {
            (family.name, band, network, contrast.name)
            for family in config.families
            for band in family.bands
            for network in family.networks
            for contrast in family.between_arm_contrasts
        }
        actual_loo_summary = {
            (row["analysis_family"], row["band"], row["network"], row["contrast"])
            for row in loo_summary
        }
        if actual_loo_summary != expected_loo_summary:
            raise NetworkModelError("leave-one-out summary does not cover exact planned contrasts")
    elif loo or loo_summary:
        raise NetworkModelError("leave-one-out output was emitted although diagnostics.leave_one_out is false")

    expected_counts = {
        (str(row["analysis_family"]), str(row["arm"]), str(row["session"])): int(row["subjects"])
        for row in count_rows
    }
    actual_counts = {
        key: len(value)
        for key, value in subjects.items()
    }
    if expected_counts != actual_counts:
        raise NetworkModelError("independent group/session counts differ from the frozen cohort counts")

    raw_maps: dict[str, dict[tuple[str, str, str, str, str], float]] = {
        "anova": {
            (
                row["analysis_family"],
                row["band"],
                row["network"],
                row["term"],
                row["term"],
            ): _finite(row, "p_value", "Type-III ANOVA")
            for row in anova
        },
        "between": {
            (
                row["analysis_family"],
                row["band"],
                row["network"],
                row["contrast"],
                row["session"],
            ): _finite(row, "p_value", "between contrast")
            for row in between
        },
        "within": {
            (
                row["analysis_family"],
                row["band"],
                row["network"],
                row["contrast"],
                row["arm"],
            ): _finite(row, "p_value", "within contrast")
            for row in within
        },
    }
    expected_correction_keys: set[tuple[str, str, str, str, str, str, str]] = set()
    correction_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in correction_rows:
        group_key = (str(row["analysis_family"]), str(row["correction_family"]), str(row["method"]))
        correction_groups[group_key].append(row)
        target = str(row["target"])
        identifier = str(row["term"] if target == "anova" else row["contrast"])
        suffix = str(
            row["term"]
            if target == "anova"
            else (row["session"] if target == "between" else row["arm"])
        )
        raw_key = (
            str(row["analysis_family"]),
            str(row["band"]),
            str(row["network"]),
            identifier,
            suffix,
        )
        if target not in raw_maps or raw_key not in raw_maps[target]:
            raise NetworkModelError(f"correction row is not linked to an exact R contrast: {raw_key}")
        _close(
            float(row["raw_p_value"]),
            raw_maps[target][raw_key],
            config,
            f"correction raw p {raw_key}",
        )
        full_key = (*group_key, target, raw_key[1], raw_key[2], raw_key[3], raw_key[4])
        if full_key in expected_correction_keys:
            raise NetworkModelError(f"duplicate correction output row: {full_key}")
        expected_correction_keys.add(full_key)
    configured_correction_rows = 0
    for family in config.families:
        for correction in family.correction_families:
            expected_raw_keys: set[tuple[str, str, str, str, str]] = set()
            for band in correction.bands:
                for network in correction.networks:
                    if correction.target == "anova":
                        expected_raw_keys.update(
                            (family.name, band, network, term, term)
                            for term in correction.terms
                        )
                    elif correction.target == "between":
                        for contrast_name in correction.contrasts:
                            contrast = next(
                                item
                                for item in family.between_arm_contrasts
                                if item.name == contrast_name
                            )
                            expected_raw_keys.add(
                                (family.name, band, network, contrast.name, contrast.session)
                            )
                    else:
                        for contrast_name in correction.contrasts:
                            contrast = next(
                                item
                                for item in family.within_arm_contrasts
                                if item.name == contrast_name
                            )
                            expected_raw_keys.update(
                                (family.name, band, network, contrast.name, arm)
                                for arm in contrast.arms
                            )
            for method in correction.methods:
                group = correction_groups.get((family.name, correction.name, method), [])
                actual_raw_keys = {
                    (
                        str(row["analysis_family"]),
                        str(row["band"]),
                        str(row["network"]),
                        str(row["term"] if correction.target == "anova" else row["contrast"]),
                        str(
                            row["term"]
                            if correction.target == "anova"
                            else (
                                row["session"]
                                if correction.target == "between"
                                else row["arm"]
                            )
                        ),
                    )
                    for row in group
                }
                if actual_raw_keys != expected_raw_keys:
                    raise NetworkModelError(
                        f"correction family {family.name}/{correction.name}/{method} has wrong membership"
                    )
                configured_correction_rows += len(expected_raw_keys)
    if len(correction_rows) != configured_correction_rows:
        raise NetworkModelError("correction output has rows outside configured families/methods")
    for key, rows in correction_groups.items():
        rows.sort(
            key=lambda row: (
                str(row["band"]),
                str(row["network"]),
                str(row["term"]),
                str(row["contrast"]),
                str(row["session"]),
                str(row["arm"]),
            )
        )
        raw = [float(row["raw_p_value"]) for row in rows]
        expected = adjust_pvalues(raw, key[2])
        for row, value in zip(rows, expected, strict=True):
            _close(
                float(row["adjusted_p_value"]),
                float(value),
                config,
                f"correction {key}",
            )
            if int(row["family_size"]) != len(rows):
                raise NetworkModelError(f"correction family size is wrong for {key}")

    return {
        "schema": "neurostat.network_models_validation.v1",
        "status": "passed",
        "absolute_tolerance": config.validation.absolute_tolerance,
        "relative_tolerance": config.validation.relative_tolerance,
        "model_jobs": expected_jobs,
        "marginal_mean_cells": len(expected_emmeans),
        "between_arm_estimands": len(expected_between),
        "within_arm_estimands": len(expected_within),
        "correction_memberships": len(correction_rows),
        "cohort_count_cells": len(expected_counts),
    }
