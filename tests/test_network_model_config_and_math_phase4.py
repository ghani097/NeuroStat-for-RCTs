from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from neurostat.network_models.config import NetworkModelConfigError, parse_network_model_config
from neurostat.network_models.data import (
    NetworkModelError,
    VerifiedConnectivity,
    VerifiedRecord,
    build_network_summaries,
    fisher_boundary_transform,
    within_network_mean,
)
from neurostat.network_models.corrections import adjust_pvalues
from neurostat.project import CleanProject, ProjectRecord, Study, load_project
from neurostat.contracts import RecordKey
from neurostat.connectivity.artifacts import ConnectivityHeader
from neurostat.scouts.artifacts import ScoutHeader


def _section() -> str:
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
        - name: between
          target: between
          contrasts: [active-minus-control-baseline, active-minus-control-followup]
          bands: [alpha]
          networks: [front]
          methods: [BH, holm, bonferroni]
        - name: within
          target: within
          contrasts: [followup-minus-baseline]
          bands: [alpha]
          networks: [front]
          methods: [BH]
  model: {reml: true, optimizer: nloptwrap, maxfun: 100000, arm_contrasts: sum, session_contrasts: poly, df_method: satterthwaite}
  diagnostics: {residuals: true, raw_scale: true, winsorized: false, winsor_robust_z: 3.5, all_fit: false, influence: false, leave_one_out: false}
  runtime: {executable: Rscript, library_paths: [], dependency_contract: builtin, strict_renv: false, timeout_seconds: 600}
  validation: {absolute_tolerance: 1.0e-10, relative_tolerance: 1.0e-8}
"""


def _project(tmp_path: Path, extra: str = "") -> CleanProject:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for index in range(1, 6):
        (tmp_path / f"r{index}.fif").write_bytes(b"x")
    records = [
        ("r1", "p1", "active", "baseline"),
        ("r2", "p1", "active", "followup"),
        ("r3", "p2", "control", "baseline"),
        ("r4", "p2", "control", "followup"),
        ("r5", "drop", "active", "baseline"),
    ]
    rows = "\n".join(
        f"  - {{record_id: {rid}, participant_id: {subject}, arm: {arm}, session: {session}, condition: evoked, data_kind: epochs, format: fif, path: {rid}.fif, analysis_profile: default}}"
        for rid, subject, arm, session in records
    )
    text = f"""schema_version: 1
study: {{id: demo, arms: [active, control], sessions: [baseline, followup], conditions: [evoked, rest]}}
input: {{state: cleaned}}
records:
{rows}
analysis_profiles: {{default: {{opaque: true}}}}
output: {{root: outputs}}
{extra or _section()}
"""
    path = tmp_path / "project.yaml"
    path.write_text(text, encoding="utf-8")
    return load_project(path)


def test_config_is_typed_explicit_and_rejects_correction_gaps(tmp_path: Path) -> None:
    config = parse_network_model_config(_project(tmp_path))
    assert config.families[0].condition == "evoked"
    assert config.families[0].sessions == ("baseline", "followup")
    assert config.diagnostics.winsor_robust_z == 3.5
    bad = _section().replace(
        "contrasts: [followup-minus-baseline]",
        "contrasts: [active-minus-control-baseline]",
    )
    with pytest.raises(NetworkModelConfigError):
        parse_network_model_config(_project(tmp_path / "bad", bad))
    strict_without_lock = _section().replace("strict_renv: false", "strict_renv: true")
    with pytest.raises(NetworkModelConfigError, match="renv_lockfile"):
        parse_network_model_config(_project(tmp_path / "strict", strict_without_lock))
    duplicate_between_name = _section().replace(
        "{name: active-minus-control-followup, session: followup",
        "{name: active-minus-control-baseline, session: followup",
    )
    with pytest.raises(NetworkModelConfigError, match="contrast names must be unique"):
        parse_network_model_config(_project(tmp_path / "duplicate", duplicate_between_name))


def test_exact_network_mean_and_boundary_only_fisher_transform() -> None:
    matrix = np.asarray(
        [
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.8],
            [0.4, 0.8, 0.0],
        ]
    )
    assert within_network_mean(matrix, (0, 1, 2)) == np.asarray([0.2, 0.4, 0.8]).mean()
    assert fisher_boundary_transform(0.25, 512) == (0.25, float(np.arctanh(0.25)), "unchanged_interior")
    adjusted_zero, _, policy_zero = fisher_boundary_transform(0.0, 512)
    adjusted_one, _, policy_one = fisher_boundary_transform(1.0, 512)
    assert adjusted_zero == 1 / 1024
    assert adjusted_one == 1 - 1 / 1024
    assert policy_zero == "exact_zero_to_epsilon"
    assert policy_one == "exact_one_to_one_minus_epsilon"
    with pytest.raises(NetworkModelError):
        fisher_boundary_transform(0.5, 0)


def test_p_adjust_matches_explicit_full_family_formulas() -> None:
    values = [0.01, 0.04, 0.03, 0.002]
    np.testing.assert_allclose(adjust_pvalues(values, "BH"), [0.02, 0.04, 0.04, 0.008])
    np.testing.assert_allclose(adjust_pvalues(values, "holm"), [0.03, 0.06, 0.06, 0.008])
    np.testing.assert_allclose(adjust_pvalues(values, "bonferroni"), [0.04, 0.16, 0.12, 0.008])
    with pytest.raises(NetworkModelError):
        adjust_pvalues([0.1, float("nan")], "BH")


def test_r_runner_uses_registered_influence_generic() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "neurostat"
        / "network_models"
        / "resources"
        / "run_network_models.R"
    ).read_text(encoding="utf-8")
    assert "stats::cooks.distance(influence_object)" in source
    assert "influence.ME::cooks.distance" not in source
    assert "stats::anova(fit, type = 3" in source
    assert "lmerTest::anova" not in source


def test_r_runner_reports_boundary_singularity_without_marking_nonconvergence() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "neurostat"
        / "network_models"
        / "resources"
        / "run_network_models.R"
    ).read_text(encoding="utf-8")
    assert '"^boundary \\\\(singular\\\\) fit:"' in source
    assert (
        "!vapply(raw_messages, is_boundary_singular_message, logical(1))"
        in source
    )
    assert "converged = length(blocking_messages) == 0L && opt_success" in source
    assert "messages <- collapse_messages(raw_messages)" in source


def test_r_runner_leave_one_out_reference_is_session_specific() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "neurostat"
        / "network_models"
        / "resources"
        / "run_network_models.R"
    ).read_text(encoding="utf-8")
    assert "full_between$contrast == value$contrast &" in source
    assert "full_between$session == value$session" in source
    assert (
        "job_loo$contrast == contrast_name & job_loo$session == contrast_session"
        in source
    )


def test_r_runner_allfit_keeps_its_rewritten_call_resolvable() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "neurostat"
        / "network_models"
        / "resources"
        / "run_network_models.R"
    ).read_text(encoding="utf-8")
    assert 'assign(".neurostat_allfit_data", data, envir = .GlobalEnv)' in source
    assert "data = .neurostat_allfit_data" in source
    assert 'finally = rm(".neurostat_allfit_data", envir = .GlobalEnv)' in source


def _verified(project: CleanProject) -> VerifiedConnectivity:
    matrix = np.asarray([[[0.0, 0.2, 0.4], [0.2, 0.0, 0.8], [0.4, 0.8, 0.0]]])
    values: dict[str, VerifiedRecord] = {}
    for record in project.records:
        path = project.project_file.parent / f"{record.record_id}.h5"
        connectivity = ConnectivityHeader(path, record.record_id, "default", "evoked_average_hilbert_pli", ("a-lh", "a-rh", "b-lh"), ("alpha",), record.record_id + "-payload")
        scout = ScoutHeader(path, record.record_id, "default", "fixed_linear_operator", connectivity.labels, 2048.0, 512, record.record_id + "-scout-payload")
        values[record.record_id] = VerifiedRecord(record, path, record.record_id + "-sha", connectivity, path, record.record_id + "-scout-sha", scout, matrix.copy(), {"backend": "evoked_average_hilbert_pli"}, 2048.0)
    return VerifiedConnectivity("upstream", project.project_file.parent, {}, "stable", values)


def test_common_complete_cohort_and_summary_identity(tmp_path: Path) -> None:
    project = _project(tmp_path)
    bundle = build_network_summaries(project, parse_network_model_config(project), _verified(project))
    assert {row["subject_id"] for row in bundle.rows} == {"p1", "p2"}
    assert len(bundle.rows) == 4
    assert all(float(row["raw_pli"]) == pytest.approx((0.2 + 0.4 + 0.8) / 3) for row in bundle.rows)
    excluded = [row for row in bundle.cohort_rows if row["subject_id"] == "drop"][0]
    assert excluded["included"] == "false"
    assert excluded["missing_sessions"] == "followup"
    assert {(row["arm"], row["session"], row["subjects"]) for row in bundle.count_rows} == {
        ("active", "baseline", 1),
        ("active", "followup", 1),
        ("control", "baseline", 1),
        ("control", "followup", 1),
    }
