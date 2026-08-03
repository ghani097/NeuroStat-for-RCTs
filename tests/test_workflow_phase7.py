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


def _synthetic_workflow_project(tmp_path: Path) -> Path:
    import mne
    import numpy as np

    sfreq = 64.0
    samples = 192
    times = np.arange(samples, dtype=np.float64) / sfreq
    data = np.vstack(
        (
            np.sin(2 * np.pi * 10 * times),
            np.sin(2 * np.pi * 10 * times + np.pi / 4),
        )
    )
    raw = mne.io.RawArray(
        data,
        mne.create_info(["Fz", "Cz"], sfreq, "eeg"),
        verbose="ERROR",
    )
    raw.set_montage(
        mne.channels.make_dig_montage(
            ch_pos={"Fz": (0.0, 0.03, 0.08), "Cz": (0.0, 0.0, 0.09)},
            coord_frame="head",
        )
    )
    raw.save(tmp_path / "clean-raw.fif", overwrite=True, verbose="ERROR")

    label_dir = tmp_path / "subjects" / "fsaverage" / "label"
    label_dir.mkdir(parents=True)
    for resource in (
        tmp_path / "trans.fif",
        tmp_path / "source-src.fif",
        tmp_path / "bem-sol.fif",
        label_dir / "lh.aparc.annot",
        label_dir / "rh.aparc.annot",
    ):
        resource.write_bytes(f"synthetic:{resource.name}".encode())

    project = tmp_path / "project.yaml"
    project.write_text(
        """schema_version: 1
study: {id: synthetic-workflow, arms: [active, comparison], sessions: [baseline, followup], conditions: [rest]}
input: {state: cleaned}
records:
  - {record_id: rest-1, participant_id: p1, arm: active, session: baseline, condition: rest, data_kind: raw, format: fif, path: clean-raw.fif, analysis_profile: rest}
analysis_profiles:
  rest:
    source:
      backend: mne_fsaverage
      subject: fsaverage
      subjects_dir: subjects
      trans: trans.fif
      source_space: source-src.fif
      bem: bem-sol.fif
      atlas: aparc
      inverse_method: MNE
      snr: 3.0
      pick_orientation: normal
      label_mode: mean_flip
      chunk_samples: 64
      output_chunk_samples: 64
      depth: 0.8
      loose: 0.2
      mindist_mm: 5.0
      noise_covariance: empirical
      average_reference_projection: true
      coordinate_frame: head
      export_matlab: false
    connectivity:
      backend: continuous_multitaper_pli
      bands: {alpha: [8, 12]}
      window_seconds: 1.0
      tail_policy: drop
      mt_bandwidth: 4
      mt_adaptive: false
      mt_low_bias: true
      n_jobs: 1
output: {root: output}
network_models: {}
graphvar: {}
brainnet: {}
""",
        encoding="utf-8",
    )
    return project


def test_all_runs_real_local_artifact_stages_and_reuses_them(tmp_path: Path) -> None:
    from neurostat.connectivity import run_connectivity
    from neurostat.ingest import ingest_project
    from neurostat.scouts import ScoutStream, run_scouts

    project_path = _synthetic_workflow_project(tmp_path)
    projector_calls = 0
    connectivity_calls = 0
    external_calls: list[str] = []

    class SyntheticProjector:
        def project(self, canonical_path, config):
            import h5py
            import numpy as np

            nonlocal projector_calls
            projector_calls += 1
            with h5py.File(canonical_path, "r") as file:
                times = np.asarray(file["times"][:], dtype=np.float64)
                sfreq = float(file.attrs["sfreq"])
            values = np.vstack((np.sin(times * 5), np.cos(times * 7)))

            def chunks():
                for start in range(0, values.shape[1], config.chunk_samples):
                    yield start, values[:, start : start + config.chunk_samples]

            return ScoutStream(
                ("left", "right"),
                times,
                sfreq,
                chunks(),
                {"backend": "mne_fsaverage", "acceptance_boundary": True},
            )

    class DenseConnectivity:
        def get_data(self, output: str):
            import numpy as np

            assert output == "dense"
            matrix = np.zeros((2, 2, 1), dtype=np.float64)
            matrix[1, 0, 0] = 0.5
            return matrix

    def spectral_connectivity(windows, **_kwargs):
        nonlocal connectivity_calls
        connectivity_calls += 1
        assert len(tuple(windows)) == 3
        return DenseConnectivity()

    projector = SyntheticProjector()

    def ingest(project, progress=None):
        return ingest_project(project, progress)

    def scouts(project, ingest_run, progress=None):
        return run_scouts(
            project,
            ingest_run,
            progress,
            mne_projector=projector,
        )

    def connectivity(project, scouts_run, progress=None):
        return run_connectivity(
            project,
            scouts_run,
            progress,
            connectivity_function=spectral_connectivity,
        )

    def external(name: str, run_id: str):
        def execute(*args):
            upstream = args[1]
            assert upstream is not None
            external_calls.append(name)
            return FakeRun(run_id, tmp_path / run_id)

        return execute

    services = WorkflowServices(
        ingest=ingest,
        scouts=scouts,
        connectivity=connectivity,
        network_models=external("network_models", "network-models"),
        graphvar_build=external("graphvar_build", "graphvar-workspaces"),
        graphvar_analyse=external("graphvar_analyse", "graphvar"),
        brainnet_build=external("brainnet_build", "brainnet-artifacts"),
        brainnet_render=external("brainnet_render", "brainnet-render"),
    )
    progress: list[ProgressEvent] = []

    first = run_workflow(
        project_path,
        ["all"],
        progress=progress.append,
        services=services,
    )
    second = run_workflow(project_path, ["all"], services=services)

    assert tuple(outcome.stage for outcome in first.outcomes) == CANONICAL_STAGE_ORDER
    assert first.success and second.success
    assert first[StageName.INGEST].reused == 0
    assert first[StageName.SCOUTS].reused == 0
    assert first[StageName.CONNECTIVITY].reused == 0
    assert second[StageName.INGEST].reused == 1
    assert second[StageName.SCOUTS].reused == 1
    assert second[StageName.CONNECTIVITY].reused == 1
    assert projector_calls == 1
    assert connectivity_calls == 1
    assert (
        external_calls
        == [
            "network_models",
            "graphvar_build",
            "graphvar_analyse",
            "brainnet_build",
            "brainnet_render",
        ]
        * 2
    )
    assert progress[0] == ProgressEvent(StageName.INGEST, "ingest started")
    assert progress[-1] == ProgressEvent(StageName.BRAINNET, "brainnet succeeded")


@pytest.mark.skipif(
    not os.environ.get("NEUROSTAT_TEST_DS006801_PROJECT"),
    reason="set NEUROSTAT_TEST_DS006801_PROJECT to a prepared public pilot or full project",
)
def test_public_ds006801_runs_through_the_high_level_connectivity_workflow() -> None:
    from neurostat.connectivity import verify_connectivity_artifact
    from neurostat.project import load_project
    from neurostat.stage_io import load_json

    project = load_project(os.environ["NEUROSTAT_TEST_DS006801_PROJECT"])
    assert project.study.id == "openneuro-ds006801-v2-validation"
    assert project.study.arms == ("active", "control")
    assert project.study.sessions == ("pre", "post")
    assert project.study.conditions == ("mixed_rest",)
    assert len(project.records) >= 4
    assert {record.key.arm for record in project.records} == {"active", "control"}
    assert {record.key.session for record in project.records} == {"pre", "post"}

    result = run_workflow(project, ["connectivity"])

    assert tuple(outcome.stage for outcome in result.outcomes) == (
        StageName.INGEST,
        StageName.SCOUTS,
        StageName.CONNECTIVITY,
    )
    assert result.success
    connectivity_run = result[StageName.CONNECTIVITY]
    assert connectivity_run.run_dir is not None
    manifest = load_json(
        connectivity_run.run_dir / "connectivity" / "manifest.json",
        "public validation connectivity manifest",
    )
    assert len(manifest["records"]) == len(project.records)
    for entry in manifest["records"]:
        header = verify_connectivity_artifact(Path(entry["path"]))
        assert header.record_id == entry["record_id"]
        assert header.labels
        assert header.bands == ("theta", "alpha", "beta")


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
