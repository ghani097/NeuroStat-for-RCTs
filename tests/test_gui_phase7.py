from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from neurostat.contracts import ProgressEvent, StageName, StageState
from neurostat.gui.controller import ControllerWorkflowError, WorkflowController
from neurostat.workflow import WorkflowServices, run_workflow

PACKAGE_ROOT = Path(__file__).parents[1] / "src"
REPO_ROOT = Path(__file__).parents[1]


@dataclass(frozen=True, slots=True)
class FakeRun:
    run_id: str
    run_dir: Path
    reused: bool | int = False


def _project_file(tmp_path: Path, *, extras: bool = True) -> Path:
    source = tmp_path / "clean.fif"
    source.touch()
    extra_blocks = ""
    if extras:
        extra_blocks = "network_models: {}\ngraphvar: {}\nbrainnet: {}\n"
    project = tmp_path / "project.yaml"
    project.write_text(
        f"""
schema_version: 1
study:
  id: example_study
  arms: [intervention, comparison]
  sessions: [baseline, followup]
  conditions: [rest]
input: {{state: cleaned}}
records:
  - record_id: participant-001-baseline-rest
    participant_id: participant-001
    arm: intervention
    session: baseline
    condition: rest
    data_kind: raw
    format: fif
    path: {source.name}
    analysis_profile: default
    data_unit: V
analysis_profiles:
  default: {{}}
output:
  root: outputs
{extra_blocks}
""".lstrip(),
        encoding="utf-8",
    )
    return project


def _services(
    tmp_path: Path,
    calls: list[str],
    *,
    fail_stage: StageName | None = None,
    thread_ids: list[int] | None = None,
) -> WorkflowServices:
    runs = {
        StageName.INGEST: FakeRun("ingest-id", tmp_path / "ingest", False),
        StageName.SCOUTS: FakeRun("scouts-id", tmp_path / "scouts", True),
        StageName.CONNECTIVITY: FakeRun(
            "connectivity-id", tmp_path / "connectivity", 2
        ),
        StageName.NETWORK_MODELS: FakeRun("network-id", tmp_path / "network", False),
        "workspace": FakeRun("workspace-id", tmp_path / "workspace", False),
        StageName.GRAPHVAR: FakeRun("graphvar-id", tmp_path / "graphvar", False),
        "artifact": FakeRun("artifact-id", tmp_path / "artifact", False),
        StageName.BRAINNET: FakeRun("brainnet-id", tmp_path / "brainnet", False),
    }

    def mark(stage: StageName) -> None:
        calls.append(stage.value)
        if thread_ids is not None:
            thread_ids.append(threading.get_ident())
        if fail_stage is stage:
            raise RuntimeError(f"{stage.value} exploded")

    def ingest(_project: object, progress=None) -> FakeRun:
        mark(StageName.INGEST)
        if progress:
            progress(ProgressEvent(StageName.INGEST, "fake ingest"))
        return runs[StageName.INGEST]

    def scouts(_project: object, ingest_run: object, progress=None) -> FakeRun:
        assert ingest_run is runs[StageName.INGEST]
        mark(StageName.SCOUTS)
        if progress:
            progress(ProgressEvent(StageName.SCOUTS, "fake scouts"))
        return runs[StageName.SCOUTS]

    def connectivity(_project: object, scouts_run: object, progress=None) -> FakeRun:
        assert scouts_run is runs[StageName.SCOUTS]
        mark(StageName.CONNECTIVITY)
        if progress:
            progress(ProgressEvent(StageName.CONNECTIVITY, "fake connectivity"))
        return runs[StageName.CONNECTIVITY]

    def network_models(
        _project: object,
        connectivity_run: object,
        progress=None,
    ) -> FakeRun:
        assert connectivity_run == runs[StageName.CONNECTIVITY].run_dir
        mark(StageName.NETWORK_MODELS)
        if progress:
            progress(ProgressEvent(StageName.NETWORK_MODELS, "fake network"))
        return runs[StageName.NETWORK_MODELS]

    def graphvar_build(_project: object, connectivity_run: object) -> FakeRun:
        assert connectivity_run == runs[StageName.CONNECTIVITY].run_dir
        calls.append("graphvar_build")
        return runs["workspace"]

    def graphvar_analyse(_project: object, workspace_run: object) -> FakeRun:
        assert workspace_run is runs["workspace"]
        mark(StageName.GRAPHVAR)
        return runs[StageName.GRAPHVAR]

    def brainnet_build(_project: object, graphvar_run: object) -> FakeRun:
        assert graphvar_run is runs[StageName.GRAPHVAR]
        calls.append("brainnet_build")
        return runs["artifact"]

    def brainnet_render(_project: object, artifact_run: object) -> FakeRun:
        assert artifact_run is runs["artifact"]
        mark(StageName.BRAINNET)
        return runs[StageName.BRAINNET]

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


def _subprocess_environment() -> dict[str, str]:
    existing = os.environ.get("PYTHONPATH")
    pythonpath = str(PACKAGE_ROOT)
    if existing:
        pythonpath = os.pathsep.join((pythonpath, existing))
    return os.environ | {"PYTHONPATH": pythonpath}


def test_controller_matches_engine_stage_results(tmp_path: Path) -> None:
    project = _project_file(tmp_path)
    controller_calls: list[str] = []
    engine_calls: list[str] = []
    controller = WorkflowController(
        services=_services(tmp_path, controller_calls),
    )
    controlled = controller.run(project, ["brainnet"])
    direct = run_workflow(
        project,
        ["brainnet"],
        services=_services(tmp_path, engine_calls),
    )

    assert [(item.stage, item.state, item.run_id) for item in controlled.outcomes] == [
        (item.stage, item.state, item.run_id) for item in direct.outcomes
    ]
    assert controller_calls == engine_calls


def test_controller_import_is_dependency_light() -> None:
    code = (
        "import sys; before = set(sys.modules); "
        "import neurostat.gui.controller; "
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
        module == prefix or module.startswith(f"{prefix}.")
        for module in modules
        for prefix in prohibited
    )


def test_controller_reports_progress_in_order(tmp_path: Path) -> None:
    project = _project_file(tmp_path, extras=False)
    events: list[ProgressEvent] = []
    outcomes = []
    controller = WorkflowController(services=_services(tmp_path, []))

    controller.run(project, ["scouts"], progress=events.append, outcome=outcomes.append)

    assert [(event.stage, event.message) for event in events] == [
        (StageName.INGEST, "ingest started"),
        (StageName.INGEST, "fake ingest"),
        (StageName.INGEST, "ingest succeeded"),
        (StageName.SCOUTS, "scouts started"),
        (StageName.SCOUTS, "fake scouts"),
        (StageName.SCOUTS, "scouts succeeded"),
    ]
    assert [outcome.stage for outcome in outcomes] == [
        StageName.INGEST,
        StageName.SCOUTS,
    ]


def test_controller_reports_failure_stage_and_partial_results(tmp_path: Path) -> None:
    project = _project_file(tmp_path)
    outcomes = []
    controller = WorkflowController(
        services=_services(tmp_path, [], fail_stage=StageName.CONNECTIVITY),
    )

    with pytest.raises(ControllerWorkflowError) as raised:
        controller.run(project, ["brainnet"], outcome=outcomes.append)

    assert raised.value.stage is StageName.CONNECTIVITY
    assert tuple(
        (outcome.stage, outcome.state) for outcome in raised.value.partial.outcomes
    ) == (
        (StageName.INGEST, StageState.SUCCEEDED),
        (StageName.SCOUTS, StageState.SUCCEEDED),
        (StageName.CONNECTIVITY, StageState.FAILED),
        (StageName.GRAPHVAR, StageState.PENDING),
        (StageName.BRAINNET, StageState.PENDING),
    )
    assert [outcome.stage for outcome in outcomes] == [
        StageName.INGEST,
        StageName.SCOUTS,
        StageName.CONNECTIVITY,
        StageName.GRAPHVAR,
        StageName.BRAINNET,
    ]


def test_controller_empty_stage_selection_is_validation_error(tmp_path: Path) -> None:
    controller = WorkflowController(services=_services(tmp_path, []))
    with pytest.raises(ValueError, match="pass 'all' explicitly"):
        controller.run(_project_file(tmp_path), [])


def _wait_for_window(window: object, qtcore: object) -> None:
    loop = qtcore.QEventLoop()
    timer = qtcore.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    window.run_finished.connect(loop.quit)
    timer.start(5000)
    loop.exec()
    # The timer is still pending only when run_finished quit the loop first.
    # A fired (inactive) timer means the run timed out.
    assert timer.isActive(), window.status.text()
    timer.stop()


def test_offscreen_pyqt6_smoke_runs_faked_workflow(tmp_path: Path, monkeypatch) -> None:
    pyqt6 = pytest.importorskip("PyQt6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets

    from neurostat.gui import app as gui_app

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = gui_app.MainWindow(
        WorkflowController(services=_services(tmp_path, [])),
    )
    window.set_project_path(str(_project_file(tmp_path, extras=False)))
    window.set_selected_stages(("scouts",))
    window.show()
    window.run_button.click()
    _wait_for_window(window, pyqt6.QtCore)

    assert window.results.rowCount() == 2
    assert window.results.item(0, 0).text() == "ingest"
    assert window.results.item(1, 2).text() == "scouts-id"
    window.close()
    application.processEvents()


def test_engine_work_runs_off_ui_thread(tmp_path: Path, monkeypatch) -> None:
    pyqt6 = pytest.importorskip("PyQt6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets

    from neurostat.gui import app as gui_app

    ui_thread = threading.get_ident()
    worker_threads: list[int] = []
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = gui_app.MainWindow(
        WorkflowController(
            services=_services(tmp_path, [], thread_ids=worker_threads),
        ),
    )
    window.set_project_path(str(_project_file(tmp_path, extras=False)))
    window.set_selected_stages(("ingest",))
    window.run_button.click()
    _wait_for_window(window, pyqt6.QtCore)

    assert worker_threads
    assert all(thread_id != ui_thread for thread_id in worker_threads)
    window.close()
    application.processEvents()


def test_build_artifacts_declare_gui_entry_point_and_package(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(dist),
            str(REPO_ROOT),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(dist.glob("*.whl"))
    sdist = next(dist.glob("*.tar.gz"))

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        entry_points = archive.read(
            "neurostat_rcts-2.0.0a1.dist-info/entry_points.txt",
        ).decode()
    assert "src/neurostat/gui/controller.py" not in names
    assert "neurostat/gui/controller.py" in names
    assert "neurostat-gui = neurostat.gui.app:main" in entry_points

    with tarfile.open(sdist) as archive:
        names = set(archive.getnames())
        pyproject = archive.extractfile(
            "neurostat_rcts-2.0.0a1/pyproject.toml",
        )
        assert pyproject is not None
        pyproject_text = pyproject.read().decode()
    assert "neurostat_rcts-2.0.0a1/src/neurostat/gui/controller.py" in names
    assert 'neurostat-gui = "neurostat.gui.app:main"' in pyproject_text
