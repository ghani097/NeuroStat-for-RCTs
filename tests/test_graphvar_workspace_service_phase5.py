from __future__ import annotations

import csv
import json
import shutil
import zipfile
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from neurostat.connectivity.artifacts import ConnectivityHeader
from neurostat.contracts import RecordKey
from neurostat.graphvar.runtime import GraphVarRuntimeMetadata, MatlabExecution
from neurostat.graphvar.service import (
    run_graphvar_analysis,
    verify_graphvar_analysis_run,
)
from neurostat.graphvar.workspace import (
    GraphVarError,
    build_graphvar_workspaces,
    verify_graphvar_workspace_run,
)
from neurostat.network_models.data import VerifiedConnectivity, VerifiedRecord
from neurostat.project import CleanProject, ProjectRecord, Study
from neurostat.scouts.artifacts import ScoutHeader
from neurostat.stage_io import sha256_file


def _xlsx(path: Path) -> None:
    rows = []
    for index, label, coords in (
        (1, "L.ROI", (1, 2, 3)),
        (2, "R.ROI", (4, 5, 6)),
    ):
        rows.append(
            f'<row r="{index}"><c r="A{index}"><v>1</v></c>'
            f'<c r="B{index}" t="inlineStr"><is><t>{label}</t></is></c>'
            f'<c r="C{index}"><v>999</v></c>'
            f'<c r="D{index}"><v>{coords[0]}</v></c>'
            f'<c r="E{index}"><v>{coords[1]}</v></c>'
            f'<c r="F{index}"><v>{coords[2]}</v></c></row>'
        )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/'
            'spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships"><sheets><sheet name="Atlas" '
            'sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"><Relationship Id="rId1" '
            'Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/'
            f'spreadsheetml/2006/main"><sheetData>{"".join(rows)}</sheetData></worksheet>',
        )


def _section(tmp_path: Path, *, alpha: float = 0.05) -> MappingProxyType:
    _xlsx(tmp_path / "atlas.xlsx")
    (tmp_path / "viewer.mat").write_bytes(b"viewer")
    executable = tmp_path / "MATLAB With Spaces" / "matlab.exe"
    executable.parent.mkdir(exist_ok=True)
    executable.write_bytes(b"matlab")
    graphvar = tmp_path / "GraphVar"
    (graphvar / "src" / "calc").mkdir(parents=True, exist_ok=True)
    (graphvar / "src" / "calc" / "graphvar_glm.m").write_text(
        "function graphvar_glm\nend\n", encoding="utf-8"
    )
    (graphvar / "start_GraphVar.m").write_text("% Release= 2.03a\n", encoding="utf-8")
    return MappingProxyType(
        {
            "atlas": MappingProxyType(
                {
                    "workbook": "atlas.xlsx",
                    "brain_viewer_config": "viewer.mat",
                    "nodes": (
                        MappingProxyType(
                            {
                                "connectivity_label": "left-dk",
                                "atlas_index_1based": 1,
                                "display_label": "L.ROI",
                                "coordinates": (1, 2, 3),
                            }
                        ),
                        MappingProxyType(
                            {
                                "connectivity_label": "right-dk",
                                "atlas_index_1based": 2,
                                "display_label": "R.ROI",
                                "coordinates": (4, 5, 6),
                            }
                        ),
                    ),
                }
            ),
            "families": (
                MappingProxyType(
                    {
                        "name": "primary",
                        "holm_family": "sep-primary",
                        "condition": "sep",
                        "bands": ("theta",),
                        "session_pairs": (
                            MappingProxyType(
                                {
                                    "name": "pre-post",
                                    "first_session": "pre",
                                    "second_session": "post",
                                }
                            ),
                        ),
                        "effect": MappingProxyType(
                            {
                                "numerator_arm": "active",
                                "denominator_arm": "control",
                            }
                        ),
                        "component_forming_alpha": alpha,
                        "familywise_alpha": 0.05,
                        "permutations": 4,
                        "seed": MappingProxyType(
                            {"policy": "sha256_workspace", "base": 7}
                        ),
                        "matrix_field_template": "OutPLI_{band}",
                        "effect_label": "arm_active-arm_control=0",
                    }
                ),
            ),
            "runtime": MappingProxyType(
                {
                    "executable": "MATLAB With Spaces/matlab.exe",
                    "graphvar_root": "GraphVar",
                    "matlab_release": "2025b",
                    "timeout_seconds": 60,
                }
            ),
        }
    )


def _project(tmp_path: Path, *, alpha: float = 0.05) -> CleanProject:
    rows = []
    for index, (subject, arm, session) in enumerate(
        (
            ("p1", "active", "pre"),
            ("p1", "active", "post"),
            ("p2", "control", "pre"),
            ("p2", "control", "post"),
            ("drop", "active", "pre"),
        ),
        1,
    ):
        path = tmp_path / f"r{index}.fif"
        path.write_bytes(b"x")
        rows.append(
            ProjectRecord(
                f"r{index}",
                RecordKey(subject, arm, session, "sep"),
                "epochs",
                "fif",
                path,
                "default",
            )
        )
    return CleanProject(
        tmp_path / "project.yaml",
        Study("study", ("active", "control"), ("pre", "post"), ("sep",)),
        tuple(rows),
        MappingProxyType({"default": MappingProxyType({"opaque": True})}),
        tmp_path / "outputs",
        MappingProxyType({"graphvar": _section(tmp_path, alpha=alpha)}),
    )


def _upstream(project: CleanProject) -> VerifiedConnectivity:
    matrix = np.asarray([[[0.0, 0.25], [0.25, 0.0]]])
    records = {}
    for record in project.records:
        fake = project.project_file.parent / f"{record.record_id}.h5"
        header = ConnectivityHeader(
            fake,
            record.record_id,
            "default",
            "backend",
            ("left-dk", "right-dk"),
            ("theta",),
            f"payload-{record.record_id}",
        )
        scout = ScoutHeader(
            fake,
            record.record_id,
            "default",
            "fixed_linear_operator",
            header.labels,
            100.0,
            20,
            f"scout-{record.record_id}",
        )
        records[record.record_id] = VerifiedRecord(
            record,
            fake,
            f"sha-{record.record_id}",
            header,
            fake,
            f"scout-sha-{record.record_id}",
            scout,
            matrix.copy(),
            {"backend": "backend"},
            100.0,
        )
    return VerifiedConnectivity(
        "connectivity", project.output_root, {}, "stable", records
    )


def _result(path: Path, *, component: bool = False) -> None:
    import h5py

    effects, nodes, permutations = 1, 2, 4
    b = np.zeros((effects, nodes, nodes))
    f = np.zeros_like(b)
    p = np.zeros_like(b)
    np_value = np.ones_like(b)
    np.fill_diagonal(np_value[0], 0)
    if component:
        b[0, 0, 1] = b[0, 1, 0] = 1
        f[0, 0, 1] = f[0, 1, 0] = 2
        p[0, 0, 1] = p[0, 1, 0] = 0.02
        np_value[0, 0, 1] = np_value[0, 1, 0] = 0.01
    null = np.zeros((permutations, effects, nodes, nodes))
    with h5py.File(path, "w") as file:
        for name, value in (
            ("B", b),
            ("F", f),
            ("P", p),
            ("NP", np_value),
            ("NF", null),
            ("NB", null),
        ):
            file.create_dataset(name, data=value)
        label = "arm_active-arm_control=0"
        chars = file.create_dataset(
            "effect",
            data=np.asarray([ord(char) for char in label], dtype=np.uint16).reshape(
                -1, 1
            ),
        )
        chars.attrs["MATLAB_class"] = np.bytes_("char")
        refs = np.empty((1, 1), dtype=h5py.ref_dtype)
        refs[0, 0] = chars.ref
        file.create_dataset("LAB", data=refs)
        file.create_dataset("J", data=np.asarray([[1.0]]))
        file.create_dataset("DDF", data=np.asarray([[1.0]]))
        file.create_dataset("N", data=np.asarray([[4.0]]))


def _preflight(config, directory: Path) -> GraphVarRuntimeMetadata:
    for name, content in (
        ("request.json", "{}"),
        ("preflight.json", "{}"),
        ("preflight.stdout.log", "ok"),
        ("preflight.stderr.log", ""),
    ):
        (directory / name).write_text(content, encoding="utf-8")
    graphvar_glm = config.runtime.graphvar_root / "src" / "calc" / "graphvar_glm.m"
    start = config.runtime.graphvar_root / "start_GraphVar.m"
    return GraphVarRuntimeMetadata(
        Path(config.runtime.executable).resolve(),
        "2025b",
        "2.03a",
        config.runtime.graphvar_root,
        graphvar_glm,
        sha256_file(graphvar_glm),
        sha256_file(start),
        "runner-sha",
    )


def test_workspace_build_is_relocatable_exact_and_matrix_faithful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    monkeypatch.setattr(
        "neurostat.graphvar.workspace.verify_connectivity_run",
        lambda *_args: _upstream(project),
    )
    run = build_graphvar_workspaces(project, tmp_path / "connectivity")
    assert Path(
        json.loads((run.run_dir / "project.lock.json").read_text(encoding="utf-8"))[
            "config"
        ]["runtime"]["executable"]
    ).is_absolute()
    assert run.reused is False
    workspace = run.workspaces[0]
    with (workspace / "Variables.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    assert len(rows) == 4
    assert [row["session"] for row in rows] == ["pre", "post", "pre", "post"]
    from scipy.io import loadmat

    settings = loadmat(workspace / "Workspace.mat", simplify_cells=True)
    assert Path(str(settings["data_path"])).resolve() == (workspace / "data").resolve()
    relative_files = [str(value) for value in np.atleast_1d(settings["files_rel"])]
    assert relative_files
    assert all(Path(value).name == value for value in relative_files)
    for relative, value in zip(
        relative_files, np.atleast_1d(settings["files"]), strict=True
    ):
        path = Path(str(value)).resolve()
        path.relative_to(workspace.resolve())
        assert path.is_file()
        assert path == (workspace / "data" / "CorrMatrix" / relative).resolve()
    copied = loadmat(Path(str(np.atleast_1d(settings["files"])[0])), simplify_cells=True)
    np.testing.assert_array_equal(
        copied["OutPLI_theta"], np.asarray([[0.0, 0.25], [0.25, 0.0]])
    )
    reused = build_graphvar_workspaces(project, tmp_path / "connectivity")
    assert reused.reused is True
    verify_graphvar_workspace_run(run.run_dir)
    relocated = tmp_path / "relocated" / run.run_dir.name
    relocated.parent.mkdir()
    shutil.move(run.run_dir, relocated)
    relocated_run = verify_graphvar_workspace_run(relocated)
    assert relocated_run.run_id == run.run_id
    assert all(path.is_dir() for path in relocated_run.workspaces)
    result = tmp_path / "relocated-manual.mat"
    _result(result)
    analysis = run_graphvar_analysis(
        project,
        relocated_run,
        result_files={relocated_run.workspaces[0].name: result},
        preflight_function=_preflight,
    )
    analysis_workspace = (
        analysis.run_dir
        / "graphvar"
        / "workspaces"
        / relocated_run.workspaces[0].name
    )
    analysis_settings = loadmat(
        analysis_workspace / "Workspace.mat", simplify_cells=True
    )
    assert Path(str(analysis_settings["data_path"])).resolve() == (
        analysis_workspace / "data"
    ).resolve()
    for relative, value in zip(
        np.atleast_1d(analysis_settings["files_rel"]),
        np.atleast_1d(analysis_settings["files"]),
        strict=True,
    ):
        assert Path(str(relative)).name == str(relative)
        assert Path(str(value)).resolve() == (
            analysis_workspace / "data" / "CorrMatrix" / str(relative)
        ).resolve()
    relocated_workspace = (
        relocated / "workspaces" / relocated_run.workspaces[0].name
    )
    (relocated_workspace / "matrix-audit.csv").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(GraphVarError, match="SHA-256"):
        verify_graphvar_workspace_run(relocated)


def test_manual_import_finalizes_no_component_headers_reuses_and_detects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    upstream = _upstream(project)
    monkeypatch.setattr(
        "neurostat.graphvar.workspace.verify_connectivity_run",
        lambda *_args: upstream,
    )
    prepared = build_graphvar_workspaces(project, tmp_path / "connectivity")
    source = tmp_path / "manual.mat"
    _result(source)
    mapping = {prepared.workspaces[0].name: source}
    run = run_graphvar_analysis(
        project, prepared, result_files=mapping, preflight_function=_preflight
    )
    assert run.reused is False
    components = run.run_dir / "graphvar" / "components-all.csv"
    links = run.run_dir / "graphvar" / "component-member-links-all.csv"
    assert components.read_text(encoding="utf-8").startswith("holm_family,")
    assert links.read_text(encoding="utf-8").startswith("holm_family,")
    status = list(
        csv.DictReader(
            (run.run_dir / "graphvar" / "workspace-status.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    assert status[0]["status"] == "complete_no_component"
    reused = run_graphvar_analysis(
        project, prepared, result_files=mapping, preflight_function=_preflight
    )
    assert reused.reused is True
    components.write_text("tampered", encoding="utf-8")
    with pytest.raises(GraphVarError, match="SHA-256"):
        verify_graphvar_analysis_run(run.run_dir)


def test_failed_attempt_is_cleaned_and_resume_promotes_only_valid_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    monkeypatch.setattr(
        "neurostat.graphvar.workspace.verify_connectivity_run",
        lambda *_args: _upstream(project),
    )
    prepared = build_graphvar_workspaces(project, tmp_path / "connectivity")
    source = tmp_path / "runner.mat"
    _result(source, component=True)
    calls = {"failed": 0, "success": 0}

    def fail(_config, _workspace, _request, output):
        calls["failed"] += 1
        (output / "corr_area_1_Group.mat").write_bytes(b"partial")
        raise RuntimeError("injected runner failure")

    with pytest.raises(RuntimeError, match="injected"):
        run_graphvar_analysis(
            project, prepared, workspace_runner=fail, preflight_function=_preflight
        )

    def succeed(_config, _workspace, _request, output):
        calls["success"] += 1
        target = output / "corr_area_1_Group.mat"
        target.write_bytes(source.read_bytes())
        stdout = output / "matlab.stdout.log"
        stderr = output / "matlab.stderr.log"
        metadata = output / "execution.json"
        stdout.write_text("ok", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        metadata.write_text("{}", encoding="utf-8")
        return MatlabExecution(0, 0, 1, 1, stdout, stderr, metadata, target)

    run = run_graphvar_analysis(
        project, prepared, workspace_runner=succeed, preflight_function=_preflight
    )
    assert calls == {"failed": 1, "success": 1}
    assert run.run_dir.is_dir()
    assert not any(project.output_root.rglob(".neurostat-CorrResults-attempt-*"))


def test_prepared_workspace_run_cannot_be_mixed_with_changed_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    monkeypatch.setattr(
        "neurostat.graphvar.workspace.verify_connectivity_run",
        lambda *_args: _upstream(project),
    )
    prepared = build_graphvar_workspaces(project, tmp_path / "connectivity")
    changed = _project(tmp_path, alpha=0.04)
    result = tmp_path / "manual.mat"
    _result(result)
    with pytest.raises(GraphVarError, match="current project/config/plans"):
        run_graphvar_analysis(
            changed,
            prepared,
            result_files={prepared.workspaces[0].name: result},
            preflight_function=_preflight,
        )
