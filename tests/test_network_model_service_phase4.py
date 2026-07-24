from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.io import savemat

from neurostat.analysis_profiles import project_analysis_profiles
from neurostat.connectivity.artifacts import write_connectivity_artifact
from neurostat.connectivity.methods import ConnectivityValues
from neurostat.network_models import (
    NetworkModelError,
    RuntimeMetadata,
    run_network_models,
    verify_network_model_run,
)
from neurostat.network_models.config import parse_network_model_config
from neurostat.network_models.runtime import preflight_r
from neurostat.project import CleanProject, load_project
from neurostat.scouts.artifacts import ScoutStream, write_scout_artifact
from neurostat.stage_io import atomic_json, digest_json, sha256_file


def _config(executable: str = "Rscript", library_paths: str = "[]") -> str:
    return """
network_models:
  networks:
    front: [a-lh, a-rh, b-lh]
  families:
    - name: primary
      condition: evoked
      bands: [alpha]
      networks: [front]
      sessions: [baseline, followup]
      boundary_observation_count: scout_samples
      between_arm_contrasts:
        - {name: active-minus-control-baseline, session: baseline, numerator_arm: active, denominator_arm: control}
        - {name: active-minus-control-followup, session: followup, numerator_arm: active, denominator_arm: control}
      within_arm_contrasts:
        - {name: followup-minus-baseline, arms: [active, control], numerator_session: followup, denominator_session: baseline}
      correction_families:
        - {name: anova-arm, target: anova, terms: [arm], bands: [alpha], networks: [front], methods: [BH, holm]}
        - {name: anova-session, target: anova, terms: [session], bands: [alpha], networks: [front], methods: [BH, holm]}
        - {name: anova-interaction, target: anova, terms: [arm:session], bands: [alpha], networks: [front], methods: [BH, holm]}
        - {name: between, target: between, contrasts: [active-minus-control-baseline, active-minus-control-followup], bands: [alpha], networks: [front], methods: [BH, holm, bonferroni]}
        - {name: within, target: within, contrasts: [followup-minus-baseline], bands: [alpha], networks: [front], methods: [BH, holm, bonferroni]}
  model: {reml: true, optimizer: nloptwrap, maxfun: 100000, arm_contrasts: sum, session_contrasts: poly, df_method: satterthwaite}
  diagnostics: {residuals: true, raw_scale: true, winsorized: false, winsor_robust_z: 3.5, all_fit: false, influence: false, leave_one_out: false}
  runtime: {executable: "__EXECUTABLE__", library_paths: __LIBRARY_PATHS__, dependency_contract: builtin, strict_renv: false, timeout_seconds: 600}
  validation: {absolute_tolerance: 1.0e-9, relative_tolerance: 1.0e-8}
""".replace("__EXECUTABLE__", executable).replace("__LIBRARY_PATHS__", library_paths)


def _project(
    tmp_path: Path,
    *,
    executable: str = "Rscript",
    library_paths: str = "[]",
) -> CleanProject:
    tmp_path.mkdir(parents=True, exist_ok=True)
    savemat(tmp_path / "operator.mat", {"SensorToScoutOperator": np.ones((3, 2))})
    savemat(tmp_path / "clean.mat", {"placeholder": np.asarray([1.0])})
    rows: list[str] = []
    counter = 0
    for arm, prefix in (("active", "a"), ("control", "c")):
        for subject_index in range(1, 5):
            for session in ("baseline", "followup"):
                counter += 1
                rows.append(
                    "  - {record_id: r%02d, participant_id: %s%d, arm: %s, session: %s, "
                    "condition: evoked, data_kind: epochs, format: fieldtrip_mat, path: clean.mat, "
                    "analysis_profile: fixed, data_unit: V}" %
                    (counter, prefix, subject_index, arm, session)
                )
    project_path = tmp_path / "project.yaml"
    project_path.write_text(
        """schema_version: 1
study: {id: synthetic, arms: [active, control], sessions: [baseline, followup], conditions: [evoked]}
input: {state: cleaned}
records:
"""
        + "\n".join(rows)
        + """
analysis_profiles:
  fixed:
    source:
      backend: fixed_linear_operator
      operator: operator.mat
      sensor_labels: [s1, s2]
      scout_labels: [a-lh, a-rh, b-lh]
      export_matlab: false
    connectivity:
      backend: evoked_average_hilbert_pli
      bands: {alpha: [8, 12]}
      filter_order: 4
output: {root: outputs}
"""
        + _config(executable, library_paths),
        encoding="utf-8",
    )
    return load_project(project_path)


def _upstream(project: CleanProject) -> Path:
    profiles = project_analysis_profiles(project)
    normalized_profiles = {name: value.normalized() for name, value in profiles.items()}
    scout_id = "synthetic-scout"
    scout_dir = project.output_root / "runs" / f"scouts-{scout_id}"
    scout_entries = []
    labels = ("a-lh", "a-rh", "b-lh")
    times = np.arange(512, dtype=np.float64) / 128.0
    for record in project.records:
        path = scout_dir / "scouts" / "records" / f"{record.record_id}.h5"
        stream = ScoutStream(
            labels,
            times,
            128.0,
            ((0, np.zeros((3, 512), dtype=np.float64)),),
            {"backend": "fixed_linear_operator", "synthetic": True},
        )
        digest, header = write_scout_artifact(
            path,
            stream,
            {
                "record_id": record.record_id,
                "participant_id": record.key.participant_id,
                "arm": record.key.arm,
                "session": record.key.session,
                "condition": record.key.condition,
                "analysis_profile": record.analysis_profile,
                "source_backend": "fixed_linear_operator",
                "canonical_sha256": f"canonical-{record.record_id}",
                "canonical_payload_digest": f"canonical-payload-{record.record_id}",
                "profile_digest": profiles[record.analysis_profile].digest,
            },
            chunk_samples=128,
        )
        scout_entries.append(
            {
                "record_id": record.record_id,
                "path": str(path),
                "sha256": digest,
                "payload_digest": header.payload_digest,
                "dimensions": [3, 512],
                "labels": list(labels),
                "sfreq": 128.0,
                "analysis_profile": record.analysis_profile,
                "source_backend": "fixed_linear_operator",
                "canonical_sha256": f"canonical-{record.record_id}",
            }
        )
    scout_manifest = {
        "schema": "neurostat.scout_manifest.v1",
        "run_id": scout_id,
        "upstream_run_id": "synthetic-ingest",
        "records": scout_entries,
    }
    atomic_json(
        scout_dir / "project.lock.json",
        {
            "stage": "scouts",
            "project": project.normalized(),
            "profiles": normalized_profiles,
            "upstream": {"stage": "ingest", "run_id": "synthetic-ingest"},
        },
    )
    atomic_json(scout_dir / "scouts" / "manifest.json", scout_manifest)
    atomic_json(
        scout_dir / "scouts" / "status.json",
        {"state": "succeeded", "total": len(project.records), "completed": len(project.records)},
    )

    connectivity_id = "synthetic-connectivity"
    connectivity_dir = project.output_root / "runs" / f"connectivity-{connectivity_id}"
    connectivity_entries = []
    scout_by_id = {item["record_id"]: item for item in scout_entries}
    for index, record in enumerate(project.records):
        subject_offset = (int(record.key.participant_id[1:]) - 1) * 0.01
        session_subject_offset = subject_offset * (
            1.35 if record.key.session == "followup" else 1.0
        )
        arm_session_jitter = (
            0.003 * (-1 if int(record.key.participant_id[1:]) % 2 else 1)
            if record.key.arm == "active" and record.key.session == "followup"
            else 0.0
        )
        raw = (
            0.20
            + (0.08 if record.key.arm == "active" else 0.0)
            + (0.05 if record.key.session == "followup" else 0.0)
            + session_subject_offset
            + arm_session_jitter
        )
        matrix = np.full((3, 3), raw, dtype=np.float64)
        np.fill_diagonal(matrix, 0.0)
        values = ConnectivityValues(
            matrices=matrix[np.newaxis, :, :],
            sos=np.zeros((1, 4, 6), dtype=np.float64),
            provenance={
                "backend": "evoked_average_hilbert_pli",
                "sample_policy": "synthetic all samples",
            },
        )
        scout = scout_by_id[record.record_id]
        target = connectivity_dir / "connectivity" / "records" / f"{record.record_id}.h5"
        digest, header = write_connectivity_artifact(
            target,
            values,
            labels,
            profiles[record.analysis_profile].connectivity.bands,
            {
                "record_id": record.record_id,
                "participant_id": record.key.participant_id,
                "arm": record.key.arm,
                "session": record.key.session,
                "condition": record.key.condition,
                "analysis_profile": record.analysis_profile,
                "connectivity_backend": "evoked_average_hilbert_pli",
                "sfreq": 128.0,
                "scout_sha256": scout["sha256"],
                "scout_payload_digest": scout["payload_digest"],
                "profile_digest": profiles[record.analysis_profile].digest,
            },
        )
        connectivity_entries.append(
            {
                "record_id": record.record_id,
                "path": str(target),
                "sha256": digest,
                "payload_digest": header.payload_digest,
                "dimensions": [1, 3, 3],
                "labels": list(labels),
                "bands": ["alpha"],
                "analysis_profile": record.analysis_profile,
                "connectivity_backend": "evoked_average_hilbert_pli",
                "scout_sha256": scout["sha256"],
            }
        )
    stable_scout = {
        "schema": scout_manifest["schema"],
        "run_id": scout_manifest["run_id"],
        "upstream_run_id": scout_manifest["upstream_run_id"],
        "records": scout_manifest["records"],
    }
    atomic_json(
        connectivity_dir / "project.lock.json",
        {
            "stage": "connectivity",
            "project": project.normalized(),
            "profiles": normalized_profiles,
            "upstream": {
                "stage": "scouts",
                "run_id": scout_id,
                "manifest_content_sha256": digest_json(stable_scout),
                "records": {
                    item["record_id"]: item["sha256"]
                    for item in scout_entries
                },
            },
        },
    )
    atomic_json(
        connectivity_dir / "connectivity" / "manifest.json",
        {
            "schema": "neurostat.connectivity_manifest.v1",
            "run_id": connectivity_id,
            "upstream_run_id": scout_id,
            "records": connectivity_entries,
        },
    )
    atomic_json(
        connectivity_dir / "connectivity" / "status.json",
        {"state": "succeeded", "total": len(project.records), "completed": len(project.records)},
    )
    return connectivity_dir


def _write(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fake_preflight(config, directory: Path):
    paths = tuple(
        directory / name
        for name in (
            "effective-r-package-contract.tsv",
            "r-runtime.tsv",
            "r-preflight.stdout.log",
            "r-preflight.stderr.log",
        )
    )
    for path in paths:
        path.write_text("synthetic\n", encoding="utf-8")
    return (
        RuntimeMetadata(
            Path(sys.executable),
            "R synthetic",
            (),
            (("lme4", "synthetic"), ("lmerTest", "synthetic"), ("emmeans", "synthetic")),
            "contract",
            False,
            None,
        ),
        paths,
    )


def _fake_runner(request) -> None:
    with request.input_csv.open(encoding="utf-8", newline="") as handle:
        data = list(csv.DictReader(handle))
    with request.jobs_csv.open(encoding="utf-8", newline="") as handle:
        jobs = list(csv.DictReader(handle))
    with request.between_csv.open(encoding="utf-8", newline="") as handle:
        between_specs = list(csv.DictReader(handle))
    with request.within_csv.open(encoding="utf-8", newline="") as handle:
        within_specs = list(csv.DictReader(handle))
    outputs = request.output_dir
    overview = []
    emmeans = []
    between = []
    within = []
    residuals = []
    residual_observations = []
    fixed = []
    anova = []
    sensitivity = []
    sensitivity_anova = []
    sensitivity_between = []
    for job in jobs:
        rows = [
            row
            for row in data
            if all(row[key] == job[key] for key in ("analysis_family", "condition", "band", "network"))
        ]
        cell_mean: dict[tuple[str, str], float] = {}
        raw_cell_mean: dict[tuple[str, str], float] = {}
        for arm in ("active", "control"):
            for session in ("baseline", "followup"):
                values = [float(row["z_pli"]) for row in rows if row["arm"] == arm and row["session"] == session]
                raw_values = [float(row["raw_pli"]) for row in rows if row["arm"] == arm and row["session"] == session]
                cell_mean[(arm, session)] = float(np.mean(values))
                raw_cell_mean[(arm, session)] = float(np.mean(raw_values))
                value = cell_mean[(arm, session)]
                emmeans.append(
                    job
                    | {
                        "arm": arm,
                        "session": session,
                        "emmean": value,
                        "std_error": 0.1,
                        "df": 10,
                        "lower_cl": value - 0.1,
                        "upper_cl": value + 0.1,
                        "raw_emm": math.tanh(value),
                        "raw_lower_cl": math.tanh(value - 0.1),
                        "raw_upper_cl": math.tanh(value + 0.1),
                    }
                )
        overview.append(
            job
            | {
                "n_rows": len(rows),
                "subjects": len({row["subject_id"] for row in rows}),
                "arms": 2,
                "sessions": 2,
                "converged": "true",
                "singular": "false",
                "random_intercept_sd": 0.2,
                "residual_sd": 1.0,
                "icc": 0.038461538,
                "max_abs_gradient": 0,
                "convergence_message": "",
            }
        )
        for term in ("(Intercept)", "arm1", "session1", "arm1:session1"):
            fixed.append(job | {"response": "z_pli", "term": term, "estimate": 0, "std_error": 1, "df": 10, "statistic": 0, "p_value": 1})
            sensitivity.append(job | {"sensitivity": "raw_scale", "term": term, "estimate": 0, "std_error": 1, "df": 10, "statistic": 0, "p_value": 1, "flagged_values": 0})
        for term in ("arm", "session", "arm:session"):
            anova.append(job | {"term": term, "num_df": 1, "den_df": 10, "f_value": 1, "p_value": 0.5})
            sensitivity_anova.append(job | {"sensitivity": "raw_scale", "term": term, "num_df": 1, "den_df": 10, "f_value": 1, "p_value": 0.5, "singular": "false"})
        residuals.append(job | {"n": len(rows), "mean": 0, "sd": 0, "minimum": 0, "maximum": 0, "shapiro_w": "", "shapiro_p": "", "fitted_residual_correlation": ""})
        for row in rows:
            residual_observations.append(
                job
                | {
                    "record_id": row["record_id"],
                    "subject_id": row["subject_id"],
                    "arm": row["arm"],
                    "session": row["session"],
                    "raw_pli": row["raw_pli"],
                    "z_pli": row["z_pli"],
                    "fitted": row["z_pli"],
                    "residual": 0,
                    "standardized_residual": 0,
                }
            )
        for spec in between_specs:
            if spec["analysis_family"] != job["analysis_family"]:
                continue
            estimate = cell_mean[(spec["numerator_arm"], spec["session"])] - cell_mean[(spec["denominator_arm"], spec["session"])]
            between.append(
                job
                | {
                    "contrast": spec["name"],
                    "session": spec["session"],
                    "numerator_arm": spec["numerator_arm"],
                    "denominator_arm": spec["denominator_arm"],
                    "estimate": estimate,
                    "std_error": 0.1,
                    "df": 10,
                    "t_ratio": estimate / 0.1,
                    "p_value": 0.01 + 0.01 * len(between),
                    "residual_sigma": 1,
                    "standardized_estimate": estimate,
                }
            )
            raw_estimate = raw_cell_mean[(spec["numerator_arm"], spec["session"])] - raw_cell_mean[(spec["denominator_arm"], spec["session"])]
            sensitivity_between.append(
                job
                | {
                    "sensitivity": "raw_scale",
                    "contrast": spec["name"],
                    "session": spec["session"],
                    "numerator_arm": spec["numerator_arm"],
                    "denominator_arm": spec["denominator_arm"],
                    "estimate": raw_estimate,
                    "std_error": 0.1,
                    "df": 10,
                    "t_ratio": raw_estimate / 0.1,
                    "p_value": 0.2,
                    "residual_sigma": 1,
                    "standardized_estimate": raw_estimate,
                    "singular": "false",
                    "flagged_values": 0,
                }
            )
        for spec in within_specs:
            if spec["analysis_family"] != job["analysis_family"]:
                continue
            estimate = cell_mean[(spec["arm"], spec["numerator_session"])] - cell_mean[(spec["arm"], spec["denominator_session"])]
            within.append(
                job
                | {
                    "contrast": spec["name"],
                    "arm": spec["arm"],
                    "numerator_session": spec["numerator_session"],
                    "denominator_session": spec["denominator_session"],
                    "estimate": estimate,
                    "std_error": 0.1,
                    "df": 10,
                    "t_ratio": estimate / 0.1,
                    "p_value": 0.03 + 0.01 * len(within),
                    "residual_sigma": 1,
                    "standardized_estimate": estimate,
                }
            )
    _write(outputs / "fixed_effects.csv", fixed, ("analysis_family", "condition", "band", "network", "response", "term", "estimate", "std_error", "df", "statistic", "p_value"))
    _write(outputs / "anova_type3.csv", anova, ("analysis_family", "condition", "band", "network", "term", "num_df", "den_df", "f_value", "p_value"))
    _write(outputs / "estimated_marginal_means.csv", emmeans, ("analysis_family", "condition", "band", "network", "arm", "session", "emmean", "std_error", "df", "lower_cl", "upper_cl", "raw_emm", "raw_lower_cl", "raw_upper_cl"))
    _write(outputs / "between_arm_contrasts.csv", between, ("analysis_family", "condition", "band", "network", "contrast", "session", "numerator_arm", "denominator_arm", "estimate", "std_error", "df", "t_ratio", "p_value", "residual_sigma", "standardized_estimate"))
    _write(outputs / "within_arm_contrasts.csv", within, ("analysis_family", "condition", "band", "network", "contrast", "arm", "numerator_session", "denominator_session", "estimate", "std_error", "df", "t_ratio", "p_value", "residual_sigma", "standardized_estimate"))
    _write(outputs / "model_overview.csv", overview, ("analysis_family", "condition", "band", "network", "n_rows", "subjects", "arms", "sessions", "converged", "singular", "random_intercept_sd", "residual_sd", "icc", "max_abs_gradient", "convergence_message"))
    _write(outputs / "residual_diagnostics.csv", residuals, ("analysis_family", "condition", "band", "network", "n", "mean", "sd", "minimum", "maximum", "shapiro_w", "shapiro_p", "fitted_residual_correlation"))
    _write(outputs / "residual_observations.csv", residual_observations, ("analysis_family", "condition", "band", "network", "record_id", "subject_id", "arm", "session", "raw_pli", "z_pli", "fitted", "residual", "standardized_residual"))
    _write(outputs / "sensitivity_models.csv", sensitivity, ("analysis_family", "condition", "band", "network", "sensitivity", "term", "estimate", "std_error", "df", "statistic", "p_value", "flagged_values"))
    _write(outputs / "sensitivity_anova_type3.csv", sensitivity_anova, ("analysis_family", "condition", "band", "network", "sensitivity", "term", "num_df", "den_df", "f_value", "p_value", "singular"))
    _write(outputs / "sensitivity_between_arm_contrasts.csv", sensitivity_between, ("analysis_family", "condition", "band", "network", "sensitivity", "contrast", "session", "numerator_arm", "denominator_arm", "estimate", "std_error", "df", "t_ratio", "p_value", "residual_sigma", "standardized_estimate", "singular", "flagged_values"))
    empty_specs = {
        "allfit_diagnostics.csv": ("analysis_family", "condition", "band", "network", "optimizer", "success", "singular", "log_likelihood", "max_fixed_effect_difference", "message"),
        "influence_diagnostics.csv": ("analysis_family", "condition", "band", "network", "subject_id", "cooks_distance", "cutoff_4_over_n", "influential", "message"),
        "leave_one_out.csv": ("analysis_family", "condition", "band", "network", "excluded_subject_id", "contrast", "session", "numerator_arm", "denominator_arm", "estimate", "std_error", "df", "t_ratio", "p_value", "full_estimate", "full_t_ratio", "estimate_change", "direction_reversal", "singular", "converged", "convergence_message"),
        "leave_one_out_summary.csv": ("analysis_family", "condition", "band", "network", "contrast", "session", "full_estimate", "full_t_ratio", "minimum_loo_estimate", "maximum_loo_estimate", "minimum_absolute_loo_t", "maximum_absolute_loo_t", "any_direction_reversal", "any_singular"),
    }
    for filename, fields in empty_specs.items():
        _write(outputs / filename, [], fields)


def test_fake_runner_atomic_reuse_and_tamper_detection(tmp_path: Path) -> None:
    project = _project(tmp_path)
    connectivity = _upstream(project)
    first = run_network_models(
        project,
        connectivity,
        statistical_runner=_fake_runner,
        preflight_function=_fake_preflight,
    )
    assert first.reused is False
    assert json.loads((first.run_dir / "network_models" / "validation.json").read_text())["status"] == "passed"
    second = run_network_models(
        project,
        connectivity,
        statistical_runner=_fake_runner,
        preflight_function=_fake_preflight,
    )
    assert second.reused is True
    unexpected = first.run_dir / "network_models" / "r_outputs" / "manifest.json"
    unexpected.write_text("{}\n", encoding="utf-8")
    with pytest.raises(NetworkModelError, match="file set"):
        verify_network_model_run(first.run_dir)
    unexpected.unlink()
    summary = first.run_dir / "network_models" / "network_summaries.csv"
    summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(NetworkModelError, match="SHA-256"):
        verify_network_model_run(first.run_dir)


def test_failed_runner_never_finalizes_a_partial_run(tmp_path: Path) -> None:
    project = _project(tmp_path)
    connectivity = _upstream(project)

    def fail(_request) -> None:
        raise RuntimeError("injected R failure")

    with pytest.raises(RuntimeError, match="injected"):
        run_network_models(
            project,
            connectivity,
            statistical_runner=fail,
            preflight_function=_fake_preflight,
        )
    run_names = [path.name for path in (project.output_root / "runs").iterdir()]
    assert not any(name.startswith("network_models-") or ".staging-" in name for name in run_names)


@pytest.mark.parametrize("tamper", ["condition", "sigma"])
def test_independent_validator_rejects_statistical_metadata_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    project = _project(tmp_path)
    connectivity = _upstream(project)

    def tampered_runner(request) -> None:
        _fake_runner(request)
        path = request.output_dir / "between_arm_contrasts.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            rows = list(reader)
        if tamper == "condition":
            rows[0]["condition"] = "wrong-condition"
        else:
            rows[0]["residual_sigma"] = "2"
            rows[0]["standardized_estimate"] = format(float(rows[0]["estimate"]) / 2, ".17g")
        _write(path, rows, fields)

    with pytest.raises(NetworkModelError, match="condition|residual sigma"):
        run_network_models(
            project,
            connectivity,
            statistical_runner=tampered_runner,
            preflight_function=_fake_preflight,
        )


def test_preflight_preserves_path_spaces_and_combines_process_errors(tmp_path: Path) -> None:
    executable = tmp_path / "R Runtime With Spaces" / "Rscript.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fixture")
    project = _project(tmp_path / "project", executable=executable.as_posix())
    config = parse_network_model_config(project)
    observed: dict[str, object] = {}

    def succeed(args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        Path(args[4]).write_text(
            "kind\tname\tvalue\n"
            "runtime\tR\tR synthetic\n"
            "library\t1\tlibrary with spaces\n"
            "package\tlme4\t2.0.1\n"
            "package\tlmerTest\t3.2.1\n"
            "package\temmeans\t2.0.3\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "preflight ok", "locale warning")

    metadata, files = preflight_r(config, tmp_path / "preflight", process_runner=succeed)
    assert metadata.executable == executable.resolve()
    assert observed["args"][0] == str(executable.resolve())
    assert observed["kwargs"]["shell"] is False
    assert all(path.is_file() for path in files)

    def fail(args, **_kwargs):
        return subprocess.CompletedProcess(args, 9, "actual stdout failure", "stderr warning")

    with pytest.raises(NetworkModelError) as raised:
        preflight_r(config, tmp_path / "failure", process_runner=fail)
    message = str(raised.value)
    assert "exit code 9" in message
    assert "actual stdout failure" in message
    assert "stderr warning" in message


def test_upstream_payload_tamper_is_rejected_before_modeling(tmp_path: Path) -> None:
    import h5py

    project = _project(tmp_path)
    connectivity = _upstream(project)
    target = next((connectivity / "connectivity" / "records").glob("*.h5"))
    with h5py.File(target, "r+") as file:
        file["matrix"][0, 0, 1] += 0.01
    with pytest.raises(NetworkModelError, match="SHA-256|symmetric|payload"):
        run_network_models(
            project,
            connectivity,
            statistical_runner=_fake_runner,
            preflight_function=_fake_preflight,
        )


LOCAL_RSCRIPT = (
    Path(os.environ["NEUROSTAT_TEST_RSCRIPT"])
    if os.environ.get("NEUROSTAT_TEST_RSCRIPT")
    else None
)
LOCAL_R_LIBRARY = (
    Path(os.environ["NEUROSTAT_TEST_R_LIBRARY"])
    if os.environ.get("NEUROSTAT_TEST_R_LIBRARY")
    else None
)


@pytest.mark.skipif(
    LOCAL_RSCRIPT is None
    or LOCAL_R_LIBRARY is None
    or not LOCAL_RSCRIPT.is_file()
    or not LOCAL_R_LIBRARY.is_dir(),
    reason="set NEUROSTAT_TEST_RSCRIPT and NEUROSTAT_TEST_R_LIBRARY for real-R validation",
)
def test_real_r_synthetic_end_to_end(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        executable=LOCAL_RSCRIPT.as_posix(),  # type: ignore[union-attr]
        library_paths=f'["{LOCAL_R_LIBRARY.as_posix()}"]',  # type: ignore[union-attr]
    )
    result = run_network_models(project, _upstream(project))
    assert result.reused is False
    validation = json.loads(
        (result.run_dir / "network_models" / "validation.json").read_text(encoding="utf-8")
    )
    assert validation["status"] == "passed"
    assert validation["between_arm_estimands"] == 2
    assert validation["within_arm_estimands"] == 2
