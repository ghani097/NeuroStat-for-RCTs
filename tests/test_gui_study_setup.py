from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from neurostat.analysis_profiles import parse_analysis_profile
from neurostat.atlas import DK68_LABELS
from neurostat.contracts import ProgressEvent, StageName
from neurostat.gui.controller import WorkflowController
from neurostat.gui.discovery import (
    DiscoveredRecord,
    Layout,
    detect_layout,
    discover_records,
)
from neurostat.gui.networks import PRESET_NETWORKS
from neurostat.gui.study_builder import (
    AnalysisProfilePair,
    StudyForm,
    build_project_document,
    validate_form,
    write_project,
)
from neurostat.network_models.config import parse_network_model_config
from neurostat.project import CleanProject, Study, load_project
from neurostat.workflow import (
    WorkflowServices,
    run_workflow,
)

PACKAGE_ROOT = Path(__file__).parents[1] / "src"


@dataclass(frozen=True, slots=True)
class FakeRun:
    run_id: str
    run_dir: Path


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    return path


def _form(tmp_path: Path, **overrides: Any) -> StudyForm:
    values: dict[str, Any] = {
        "study_id": "demo",
        "base_folder": tmp_path / "inputs",
        "arms": ("active", "comparison"),
        "sessions": ("baseline", "followup"),
        "conditions": ("rest",),
        "output_root": tmp_path / "outputs",
        "already_cleaned": True,
        # This helper supplies fixed-operator resources, so pin the pairing
        # rather than depending on whichever default the form ships with.
        "profile_pair": AnalysisProfilePair.FIXED_EVOKED,
        "fixed_operator": _touch(tmp_path / "operator.mat"),
        "fixed_sensor_labels": ("Fz", "Cz"),
        "fixed_scout_labels": ("frontal-lh", "frontal-rh"),
        "project_file": tmp_path / "project.yaml",
    }
    values.update(overrides)
    return StudyForm(**values)


def _record(tmp_path: Path, **overrides: Any) -> DiscoveredRecord:
    values: dict[str, Any] = {
        "record_id": "subject-1-active-baseline-rest",
        "participant_id": "subject-1",
        "arm": "active",
        "session": "baseline",
        "condition": "rest",
        "data_kind": "raw",
        "format": "fif",
        "path": _touch(tmp_path / "clean.fif"),
    }
    values.update(overrides)
    return DiscoveredRecord(**values)


def _clean_project(tmp_path: Path, *extras: str) -> CleanProject:
    return CleanProject(
        tmp_path / "project.yaml",
        Study("demo", ("active", "comparison"), ("baseline", "followup"), ("rest",)),
        (),
        MappingProxyType({"default": MappingProxyType({})}),
        tmp_path / "outputs",
        MappingProxyType({name: MappingProxyType({}) for name in extras}),
    )


def _services(tmp_path: Path, calls: list[str]) -> WorkflowServices:
    runs = {
        StageName.INGEST: FakeRun("ingest-id", tmp_path / "ingest"),
        StageName.SCOUTS: FakeRun("scouts-id", tmp_path / "scouts"),
    }

    def ingest(_project: object, _progress=None) -> FakeRun:
        calls.append("ingest")
        return runs[StageName.INGEST]

    def scouts(_project: object, ingest_run: object, _progress=None) -> FakeRun:
        assert ingest_run is runs[StageName.INGEST]
        calls.append("scouts")
        return runs[StageName.SCOUTS]

    return WorkflowServices(ingest=ingest, scouts=scouts)


def test_detect_layout_recognises_supported_shapes(tmp_path: Path) -> None:
    group_session = tmp_path / "group-session"
    _touch(group_session / "active" / "baseline" / "Subject 1.set")
    # Files sitting loose at the root match none of the supported shapes.
    flat = tmp_path / "flat"
    _touch(flat / "record.fif")
    empty = tmp_path / "empty"
    empty.mkdir()

    assert detect_layout(group_session) is Layout.GROUP_SESSION
    assert detect_layout(flat) is Layout.UNKNOWN
    assert detect_layout(empty) is Layout.UNKNOWN
    assert detect_layout(tmp_path / "does-not-exist") is Layout.UNKNOWN


def test_detect_layout_uses_sessions_to_split_the_ambiguous_shapes(
    tmp_path: Path,
) -> None:
    """``arm/subject/session`` and ``arm/session/subject`` are the same shape.

    Only the declared session names can tell them apart, so an unhinted probe
    must refuse to guess rather than pick one at random.
    """
    subject_then_session = tmp_path / "uk"
    _touch(subject_then_session / "active" / "Subject 1" / "baseline" / "rec.fif")
    session_then_subject = tmp_path / "v1-output"
    _touch(session_then_subject / "active" / "baseline" / "Subject 1" / "rec.fif")

    assert detect_layout(subject_then_session) is Layout.UNKNOWN
    assert detect_layout(session_then_subject) is Layout.UNKNOWN

    sessions = ("baseline", "followup")
    assert detect_layout(subject_then_session, sessions) is Layout.GROUP_SUBJECT_SESSION
    assert detect_layout(session_then_subject, sessions) is Layout.GROUP_SESSION_SUBJECT


def test_discovery_maps_fields_and_sorts_deterministically(tmp_path: Path) -> None:
    base = tmp_path / "study"
    _touch(base / "comparison" / "followup" / "Beta.mat")
    _touch(base / "active" / "baseline" / "Subject 1.set")
    _touch(base / "active" / "baseline" / "Subject 1_cleaned.set")
    _touch(base / "active" / "followup" / "Alpha.fif.gz")
    _touch(base / "ignored" / "baseline" / "Other.set")

    records = discover_records(
        base,
        ("active", "comparison"),
        ("baseline", "followup"),
        ("rest",),
        Layout.GROUP_SESSION,
        samples_per_epoch=200,
    )

    assert [
        (record.arm, record.session, record.participant_id) for record in records
    ] == [
        ("active", "baseline", "active-subject-1"),
        ("active", "followup", "active-alpha"),
        ("comparison", "followup", "comparison-beta"),
    ]
    assert [record.format for record in records] == ["set", "fif", "fieldtrip_mat"]
    assert records[2].data_kind == "epochs"
    assert records[2].mat_variable == "data_ft"
    assert records[2].samples_per_epoch == 200


def test_discovery_skips_macos_appledouble_sidecars(tmp_path: Path) -> None:
    """Mac-copied data carries a `._name` fork beside every real file.

    They share the data extension, so without filtering each record is
    discovered twice. Found on real study data, not in synthetic fixtures.
    """
    base = tmp_path / "study"
    _touch(base / "active" / "Subject 1" / "baseline" / "Subject 1-data_ft.mat")
    _touch(base / "active" / "Subject 1" / "baseline" / "._Subject 1-data_ft.mat")
    _touch(base / "active" / "Subject 1" / "baseline" / ".DS_Store")

    records = discover_records(
        base,
        ("active", "comparison"),
        ("baseline", "followup"),
        ("rest",),
    )

    assert len(records) == 1
    assert records[0].path.name == "Subject 1-data_ft.mat"


def test_participants_are_namespaced_by_arm(tmp_path: Path) -> None:
    """Subject folders are numbered per group, so `Subject 1` exists in both.

    Without namespacing they collapse into one participant spanning two arms,
    which the project schema forbids.
    """
    base = tmp_path / "study"
    _touch(base / "active" / "Subject 1" / "baseline" / "rec.mat")
    _touch(base / "comparison" / "Subject 1" / "baseline" / "rec.mat")

    records = discover_records(
        base,
        ("active", "comparison"),
        ("baseline", "followup"),
        ("rest",),
    )

    assert len(records) == 2
    participants = {record.participant_id for record in records}
    assert participants == {"active-subject-1", "comparison-subject-1"}
    assert len({record.record_id for record in records}) == 2


def test_legacy_progress_strings_become_progress_events() -> None:
    """v1 calls back with log strings; v2 handlers expect ProgressEvent.

    Handing the raw v1 callback to a v2 handler crashes on the first log line,
    so the controller must adapt between them.
    """
    from neurostat.gui.controller import _legacy_log_adapter

    assert _legacy_log_adapter(None) is None

    events: list[ProgressEvent] = []
    adapter = _legacy_log_adapter(events.append)
    assert adapter is not None
    adapter("Missing folder: nowhere")

    assert len(events) == 1
    assert isinstance(events[0], ProgressEvent)
    assert events[0].stage is StageName.INGEST
    assert events[0].message == "Missing folder: nowhere"


def test_discovery_uses_subject_folder_for_uk_layout(tmp_path: Path) -> None:
    base = tmp_path / "study"
    _touch(base / "active" / "Subject 1" / "baseline" / "rest.set")

    records = discover_records(
        base,
        ("active", "comparison"),
        ("baseline", "followup"),
        ("rest",),
        Layout.GROUP_SUBJECT_SESSION,
    )

    assert records[0].participant_id == "active-subject-1"
    assert records[0].record_id == "active-subject-1-baseline-rest"


def test_discovery_reports_slug_collisions(tmp_path: Path) -> None:
    base = tmp_path / "study"
    _touch(base / "active" / "baseline" / "Subject 1.set")
    _touch(base / "active" / "followup" / "subject-1.set")

    with pytest.raises(ValueError, match="participant_id slug collision"):
        discover_records(
            base,
            ("active", "comparison"),
            ("baseline", "followup"),
            ("rest",),
            Layout.GROUP_SESSION,
        )


def test_build_project_document_round_trips_through_project_and_profiles(
    tmp_path: Path,
) -> None:
    form = _form(tmp_path)
    records = (_record(tmp_path),)
    document = build_project_document(form, records)
    project = write_project(document, form.resolved_project_file)

    loaded = load_project(form.resolved_project_file)
    assert loaded.normalized() == project.normalized()
    profile = parse_analysis_profile(project, "default")
    assert profile.source.backend == "fixed_linear_operator"
    assert profile.connectivity.backend == "evoked_average_hilbert_pli"


def test_network_model_block_validates_through_typed_parser(
    tmp_path: Path,
) -> None:
    form = _form(tmp_path)
    records = (_record(tmp_path),)
    document = build_project_document(form, records, stages=("network_models",))
    project = write_project(document, form.resolved_project_file)

    config = parse_network_model_config(project)

    assert set(config.networks_by_name) == {"SN", "DMN", "CEN"}
    family = config.families[0]
    assert family.condition == "rest"
    assert [item.name for item in family.between_arm_contrasts] == [
        "active-minus-comparison-baseline",
        "active-minus-comparison-followup",
    ]
    assert [item.name for item in family.within_arm_contrasts] == [
        "followup-minus-baseline",
    ]


def test_network_model_contrast_counts_follow_arms_and_sessions(
    tmp_path: Path,
) -> None:
    form = _form(tmp_path, sessions=("baseline", "week4", "week8"))
    document = build_project_document(
        form,
        (_record(tmp_path),),
        stages=("network_models",),
    )
    project = write_project(document, form.resolved_project_file)
    family = parse_network_model_config(project).families[0]

    assert [item.name for item in family.between_arm_contrasts] == [
        "active-minus-comparison-baseline",
        "active-minus-comparison-week4",
        "active-minus-comparison-week8",
    ]
    assert [item.name for item in family.within_arm_contrasts] == [
        "week4-minus-baseline",
        "week8-minus-baseline",
    ]


def test_network_presets_are_dk68_labels_with_at_least_two_members() -> None:
    dk68 = set(DK68_LABELS)

    assert set(PRESET_NETWORKS) == {"SN", "DMN", "CEN"}
    for labels in PRESET_NETWORKS.values():
        assert len(labels) >= 2
        assert set(labels) <= dk68


def test_network_models_requires_a_selected_network(tmp_path: Path) -> None:
    problems = validate_form(
        _form(tmp_path, networks={}),
        (_record(tmp_path),),
        stages=("network_models",),
    )

    assert any("Select at least one network" in problem for problem in problems)


def test_validate_form_requires_exactly_two_arms(tmp_path: Path) -> None:
    problems = validate_form(
        _form(tmp_path, arms=("active", "comparison", "third")),
        (_record(tmp_path),),
    )
    assert any("exactly 2 study arms" in problem for problem in problems)


def test_validate_form_requires_at_least_two_sessions(tmp_path: Path) -> None:
    problems = validate_form(
        _form(tmp_path, sessions=("baseline",)), (_record(tmp_path),)
    )
    assert any("at least 2 sessions" in problem for problem in problems)


def test_validate_form_rejects_participant_in_multiple_arms(tmp_path: Path) -> None:
    records = (
        _record(tmp_path, participant_id="subject-1", arm="active"),
        _record(
            tmp_path,
            record_id="subject-1-comparison-followup-rest",
            participant_id="subject-1",
            arm="comparison",
            session="followup",
            path=_touch(tmp_path / "other.fif"),
        ),
    )
    problems = validate_form(_form(tmp_path), records)
    assert any("more than one arm" in problem for problem in problems)


def test_validate_form_rejects_duplicate_record_id(tmp_path: Path) -> None:
    records = (
        _record(tmp_path),
        _record(
            tmp_path,
            participant_id="subject-2",
            session="followup",
            path=_touch(tmp_path / "other.fif"),
        ),
    )
    problems = validate_form(_form(tmp_path), records)
    assert any("Record IDs must be unique" in problem for problem in problems)


def test_validate_form_rejects_duplicate_participant_tuple(tmp_path: Path) -> None:
    records = (
        _record(tmp_path),
        _record(
            tmp_path,
            record_id="subject-1-active-baseline-rest-copy",
            path=_touch(tmp_path / "other.fif"),
        ),
    )
    problems = validate_form(_form(tmp_path), records)
    assert any("participant/arm/session/condition" in problem for problem in problems)


def test_validate_form_rejects_missing_record_path(tmp_path: Path) -> None:
    problems = validate_form(
        _form(tmp_path),
        (_record(tmp_path, path=tmp_path / "missing.fif"),),
    )
    assert any("path does not exist" in problem for problem in problems)


def test_validate_form_rejects_extension_format_mismatch(tmp_path: Path) -> None:
    problems = validate_form(
        _form(tmp_path),
        (_record(tmp_path, format="set", path=_touch(tmp_path / "clean.fif")),),
    )
    assert any("does not match its file extension" in problem for problem in problems)


def test_validate_form_rejects_zero_discovered_records(tmp_path: Path) -> None:
    problems = validate_form(_form(tmp_path), ())
    assert any("No EEG records" in problem for problem in problems)


def test_run_study_yields_same_stage_results_as_workflow(tmp_path: Path) -> None:
    project = _clean_project(tmp_path)
    source = _touch(tmp_path / "inputs" / "active" / "baseline" / "clean.fif")
    records = (_record(tmp_path, path=source),)
    controller_calls: list[str] = []
    engine_calls: list[str] = []
    controller = WorkflowController(
        services=_services(tmp_path, controller_calls),
        discoverer=lambda *_args: records,
        project_writer=lambda _document, _path: project,
    )

    controlled = controller.run_study(_form(tmp_path), ["scouts"])
    direct = run_workflow(
        project, ["scouts"], services=_services(tmp_path, engine_calls)
    )

    assert [(item.stage, item.state, item.run_id) for item in controlled.outcomes] == [
        (item.stage, item.state, item.run_id) for item in direct.outcomes
    ]
    assert controller_calls == engine_calls


def test_run_study_skips_preprocessing_when_already_cleaned(tmp_path: Path) -> None:
    project = _clean_project(tmp_path)
    source = _touch(tmp_path / "inputs" / "active" / "baseline" / "clean.fif")
    records = (_record(tmp_path, path=source),)

    def fail_preprocessor(_form: StudyForm, _progress=None) -> Path:
        raise AssertionError("preprocessor should not run")

    result = WorkflowController(
        services=_services(tmp_path, []),
        preprocessor=fail_preprocessor,
        discoverer=lambda *_args: records,
        project_writer=lambda _document, _path: project,
    ).run_study(_form(tmp_path, already_cleaned=True), ["ingest"])

    assert [outcome.stage for outcome in result.outcomes] == [StageName.INGEST]


def test_run_study_preprocesses_once_then_discovers_fif_records(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    project = _clean_project(tmp_path)
    calls: list[str] = []
    documents: list[dict[str, Any]] = []

    def preprocessor(form: StudyForm, _progress=None) -> Path:
        calls.append(str(form.base_folder))
        _touch(
            processed_root
            / "active"
            / "baseline"
            / "subject-1"
            / "preprocessed_raw.fif"
        )
        return processed_root

    def writer(document: dict[str, Any], _path: Path) -> CleanProject:
        documents.append(document)
        return project

    WorkflowController(
        services=_services(tmp_path, []),
        preprocessor=preprocessor,
        project_writer=writer,
    ).run_study(_form(tmp_path, already_cleaned=False), ["ingest"])

    assert calls == [str(tmp_path / "inputs")]
    assert documents[0]["records"][0]["format"] == "fif"
    assert documents[0]["records"][0]["path"].endswith("preprocessed_raw.fif")


def _subprocess_environment() -> dict[str, str]:
    existing = os.environ.get("PYTHONPATH")
    pythonpath = str(PACKAGE_ROOT)
    if existing:
        pythonpath = os.pathsep.join((pythonpath, existing))
    return os.environ | {"PYTHONPATH": pythonpath}


@pytest.mark.parametrize(
    "module",
    [
        "neurostat.gui.discovery",
        "neurostat.gui.study_builder",
        "neurostat.gui.legacy_bridge",
        "neurostat.gui.controller",
    ],
)
def test_study_setup_imports_are_dependency_light(module: str) -> None:
    code = (
        "import sys; before = set(sys.modules); "
        f"import {module}; "
        "print('\\n'.join(sorted(set(sys.modules) - before)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
    )
    modules = set(filter(None, completed.stdout.splitlines()))
    prohibited = ("PyQt5", "PyQt6", "numpy", "mne", "matplotlib")
    assert not any(
        loaded == prefix or loaded.startswith(f"{prefix}.")
        for loaded in modules
        for prefix in prohibited
    )


@pytest.mark.skipif(
    os.environ.get("NEUROSTAT_TEST_FSAVERAGE") != "1",
    reason="fsaverage provisioning can require local MNE data or downloads",
)
def test_fsaverage_provisioning_is_opt_in() -> None:
    pytest.importorskip("mne")
    from neurostat.gui.fsaverage import provision_fsaverage

    resources = provision_fsaverage()
    assert resources.subject == "fsaverage"
    assert resources.source_space.is_file()
