from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

from neurostat.contracts import ProgressEvent, StageName, StageResult, StageState
from neurostat.project import CleanProject, Study
from neurostat.workflow import (
    CANONICAL_STAGE_ORDER,
    StageOutcome,
    WorkflowError,
    WorkflowResult,
    WorkflowServices,
    resolve_workflow_stages,
    run_workflow,
)

PACKAGE_ROOT = Path(__file__).parents[1] / "src"


@dataclass(frozen=True, slots=True)
class FakeRun:
    run_id: str
    run_dir: Path
    reused: bool | int = False


def _project(tmp_path: Path, *extras: str) -> CleanProject:
    return CleanProject(
        tmp_path / "project.yaml",
        Study("study", ("active", "control"), ("baseline", "followup"), ("rest",)),
        (),
        MappingProxyType({"default": MappingProxyType({})}),
        tmp_path / "output",
        MappingProxyType({name: MappingProxyType({}) for name in extras}),
    )


def _services(
    tmp_path: Path, calls: list[tuple[str, object | None]]
) -> WorkflowServices:
    runs = {
        "ingest": FakeRun("ingest-id", tmp_path / "ingest", 1),
        "scouts": FakeRun("scouts-id", tmp_path / "scouts", 2),
        "connectivity": FakeRun("connectivity-id", tmp_path / "connectivity", 3),
        "network_models": FakeRun("network-id", tmp_path / "network", True),
        "workspace": FakeRun("workspace-id", tmp_path / "workspace", False),
        "graphvar": FakeRun("graphvar-id", tmp_path / "graphvar", False),
        "artifact": FakeRun("artifact-id", tmp_path / "artifact", False),
        "brainnet": FakeRun("brainnet-id", tmp_path / "brainnet", False),
    }

    def ingest(_project: CleanProject, progress=None) -> FakeRun:
        calls.append(("ingest", None))
        if progress:
            progress(ProgressEvent(StageName.INGEST, "fake ingest"))
        return runs["ingest"]

    def scouts(_project: CleanProject, ingest_run: object, progress=None) -> FakeRun:
        assert ingest_run is runs["ingest"]
        calls.append(("scouts", ingest_run))
        if progress:
            progress(ProgressEvent(StageName.SCOUTS, "fake scouts"))
        return runs["scouts"]

    def connectivity(
        _project: CleanProject, scouts_run: object, progress=None
    ) -> FakeRun:
        assert scouts_run is runs["scouts"]
        calls.append(("connectivity", scouts_run))
        if progress:
            progress(ProgressEvent(StageName.CONNECTIVITY, "fake connectivity"))
        return runs["connectivity"]

    def network_models(
        _project: CleanProject,
        connectivity_run: object,
        progress=None,
    ) -> FakeRun:
        assert connectivity_run == runs["connectivity"].run_dir
        calls.append(("network_models", connectivity_run))
        if progress:
            progress(ProgressEvent(StageName.NETWORK_MODELS, "fake network models"))
        return runs["network_models"]

    def graphvar_build(_project: CleanProject, connectivity_run: object) -> FakeRun:
        assert connectivity_run == runs["connectivity"].run_dir
        calls.append(("graphvar_build", connectivity_run))
        return runs["workspace"]

    def graphvar_analyse(_project: CleanProject, workspace_run: object) -> FakeRun:
        assert workspace_run is runs["workspace"]
        calls.append(("graphvar_analyse", workspace_run))
        return runs["graphvar"]

    def brainnet_build(_project: CleanProject, graphvar_run: object) -> FakeRun:
        assert graphvar_run is runs["graphvar"]
        calls.append(("brainnet_build", graphvar_run))
        return runs["artifact"]

    def brainnet_render(_project: CleanProject, artifact_run: object) -> FakeRun:
        assert artifact_run is runs["artifact"]
        calls.append(("brainnet_render", artifact_run))
        return runs["brainnet"]

    return WorkflowServices(
        ingest=ingest,
        scouts=scouts,
        connectivity=connectivity,
        network_models=network_models,
        graphvar_build=graphvar_build,
        graphvar_analyse=graphvar_analyse,
        brainnet_build=brainnet_build,
        brainnet_render=brainnet_render,
    )


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("ingest", ("ingest",)),
        ("scouts", ("ingest", "scouts")),
        ("connectivity", ("ingest", "scouts", "connectivity")),
        (
            "network_models",
            ("ingest", "scouts", "connectivity", "network_models"),
        ),
        ("graphvar", ("ingest", "scouts", "connectivity", "graphvar")),
        (
            "brainnet",
            ("ingest", "scouts", "connectivity", "graphvar", "brainnet"),
        ),
        (
            "all",
            (
                "ingest",
                "scouts",
                "connectivity",
                "network_models",
                "graphvar",
                "brainnet",
            ),
        ),
    ],
)
def test_dependency_closure_and_deterministic_order(
    requested: str,
    expected: tuple[str, ...],
) -> None:
    assert (
        tuple(stage.value for stage in resolve_workflow_stages([requested])) == expected
    )
    assert tuple(stage.value for stage in CANONICAL_STAGE_ORDER) == (
        "ingest",
        "scouts",
        "connectivity",
        "network_models",
        "graphvar",
        "brainnet",
    )


def test_empty_stage_selection_requires_explicit_all() -> None:
    with pytest.raises(ValueError, match="pass 'all' explicitly"):
        resolve_workflow_stages(())


def test_brainnet_and_network_model_branches_are_separate(tmp_path: Path) -> None:
    calls: list[tuple[str, object | None]] = []
    services = _services(tmp_path, calls)
    brainnet = run_workflow(
        _project(tmp_path, "graphvar", "brainnet"),
        ["brainnet"],
        services=services,
    )
    assert tuple(outcome.stage for outcome in brainnet.outcomes) == (
        StageName.INGEST,
        StageName.SCOUTS,
        StageName.CONNECTIVITY,
        StageName.GRAPHVAR,
        StageName.BRAINNET,
    )
    assert "network_models" not in [name for name, _upstream in calls]

    calls.clear()
    network = run_workflow(
        _project(tmp_path, "network_models"),
        ["network_models"],
        services=services,
    )
    assert tuple(outcome.stage for outcome in network.outcomes) == (
        StageName.INGEST,
        StageName.SCOUTS,
        StageName.CONNECTIVITY,
        StageName.NETWORK_MODELS,
    )
    assert not {
        "graphvar_build",
        "graphvar_analyse",
        "brainnet_build",
        "brainnet_render",
    } & {name for name, _upstream in calls}


def test_upstream_runs_are_threaded_through(tmp_path: Path) -> None:
    calls: list[tuple[str, object | None]] = []
    result = run_workflow(
        _project(tmp_path, "network_models", "graphvar", "brainnet"),
        ["all"],
        services=_services(tmp_path, calls),
    )
    assert result.success is True
    assert [name for name, _upstream in calls] == [
        "ingest",
        "scouts",
        "connectivity",
        "network_models",
        "graphvar_build",
        "graphvar_analyse",
        "brainnet_build",
        "brainnet_render",
    ]
    assert result[StageName.INGEST].reused == 1
    assert result[StageName.NETWORK_MODELS].run_id == "network-id"


def test_failure_preserves_partial_results_and_skips_later(tmp_path: Path) -> None:
    calls: list[tuple[str, object | None]] = []
    services = _services(tmp_path, calls)

    def fail_connectivity(*_args: object, **_kwargs: object) -> object:
        calls.append(("connectivity", None))
        raise RuntimeError("connectivity exploded")

    services = WorkflowServices(
        ingest=services.ingest,
        scouts=services.scouts,
        connectivity=fail_connectivity,
        graphvar_build=services.graphvar_build,
        graphvar_analyse=services.graphvar_analyse,
        brainnet_build=services.brainnet_build,
        brainnet_render=services.brainnet_render,
    )
    with pytest.raises(WorkflowError) as raised:
        run_workflow(
            _project(tmp_path, "graphvar", "brainnet"),
            ["brainnet"],
            services=services,
        )
    assert raised.value.stage is StageName.CONNECTIVITY
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert tuple(
        (outcome.stage, outcome.state) for outcome in raised.value.partial.outcomes
    ) == (
        (StageName.INGEST, StageState.SUCCEEDED),
        (StageName.SCOUTS, StageState.SUCCEEDED),
        (StageName.CONNECTIVITY, StageState.FAILED),
        (StageName.GRAPHVAR, StageState.PENDING),
        (StageName.BRAINNET, StageState.PENDING),
    )
    assert "graphvar_build" not in [name for name, _upstream in calls]


def test_progress_events_are_emitted_in_order(tmp_path: Path) -> None:
    calls: list[tuple[str, object | None]] = []
    events: list[ProgressEvent] = []
    run_workflow(
        _project(tmp_path),
        ["scouts"],
        progress=events.append,
        services=_services(tmp_path, calls),
    )
    assert [(event.stage, event.message) for event in events] == [
        (StageName.INGEST, "ingest started"),
        (StageName.INGEST, "fake ingest"),
        (StageName.INGEST, "ingest succeeded"),
        (StageName.SCOUTS, "scouts started"),
        (StageName.SCOUTS, "fake scouts"),
        (StageName.SCOUTS, "scouts succeeded"),
    ]


def test_missing_config_block_is_clear(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError) as raised:
        run_workflow(
            _project(tmp_path, "graphvar"), ["brainnet"], services=WorkflowServices()
        )
    assert raised.value.stage is StageName.BRAINNET
    assert "brainnet:" in str(raised.value)
    assert "KeyError" not in str(raised.value)
    assert tuple(
        (outcome.stage, outcome.state) for outcome in raised.value.partial.outcomes
    ) == (
        (StageName.INGEST, StageState.PENDING),
        (StageName.SCOUTS, StageState.PENDING),
        (StageName.CONNECTIVITY, StageState.PENDING),
        (StageName.GRAPHVAR, StageState.PENDING),
        (StageName.BRAINNET, StageState.FAILED),
    )


def _subprocess_environment() -> dict[str, str]:
    existing = os.environ.get("PYTHONPATH")
    pythonpath = str(PACKAGE_ROOT)
    if existing:
        pythonpath = os.pathsep.join((pythonpath, existing))
    return os.environ | {"PYTHONPATH": pythonpath}


def _run_import(module: str) -> set[str]:
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
    return set(filter(None, completed.stdout.splitlines()))


def test_cli_run_help_and_workflow_import_are_dependency_light() -> None:
    prohibited = ("numpy", "mne", "matplotlib", "PyQt5")
    modules = _run_import("neurostat.workflow")
    assert not any(
        module == prefix or module.startswith(f"{prefix}.")
        for module in modules
        for prefix in prohibited
    )
    completed = subprocess.run(
        [sys.executable, "-m", "neurostat", "run", "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
    )
    assert "--stage" in completed.stdout
    assert "--json" in completed.stdout


def test_cli_run_json_uses_workflow_result(monkeypatch, capsys, tmp_path: Path) -> None:
    from neurostat import cli

    fake_run = FakeRun("ingest-id", tmp_path / "run", False)
    workflow = WorkflowResult(
        (
            StageOutcome(
                StageResult(StageName.INGEST, StageState.SUCCEEDED),
                fake_run,
            ),
        ),
    )
    observed: dict[str, object] = {}

    def fake_run_workflow(project: object, stages: object) -> WorkflowResult:
        observed["project"] = project
        observed["stages"] = tuple(stages)  # type: ignore[arg-type]
        return workflow

    monkeypatch.setattr("neurostat.workflow.run_workflow", fake_run_workflow)
    assert cli.main(["run", "project.yaml", "--stage", "ingest", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert observed == {"project": "project.yaml", "stages": ("ingest",)}
    assert payload == {
        "stages": [
            {
                "reused": False,
                "run_dir": str(tmp_path / "run"),
                "run_id": "ingest-id",
                "stage": "ingest",
                "state": "succeeded",
            },
        ],
        "success": True,
    }
