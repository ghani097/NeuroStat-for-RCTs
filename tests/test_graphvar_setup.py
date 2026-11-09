"""GUI GraphVar/BrainNet setup: node sets, atlas workbook, supported-only BrainNet."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from neurostat.atlas import DK68_LABELS
from neurostat.contracts import StageName, StageState
from neurostat.gui.controller import WorkflowController
from neurostat.gui.discovery import DiscoveredRecord
from neurostat.gui.graphvar_setup import (
    NODE_SET_CUSTOM,
    NODE_SET_DK68,
    NODE_SET_TRIPLE,
    BrainNetSettings,
    GraphVarSettings,
    graph_nodes,
    settings_problems,
    supported_workspaces,
    write_atlas_workbook,
)
from neurostat.gui.study_builder import AnalysisProfilePair, StudyForm, validate_form
from neurostat.project import CleanProject, Study
from neurostat.workflow import WorkflowServices

DAN = (
    "superiorparietal-lh", "superiorparietal-rh", "inferiorparietal-lh", "inferiorparietal-rh",
    "precentral-lh", "precentral-rh", "caudalmiddlefrontal-lh", "caudalmiddlefrontal-rh",
    "supramarginal-lh", "supramarginal-rh", "lateraloccipital-lh", "lateraloccipital-rh",
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def _graphvar(tmp_path: Path, **overrides: Any) -> GraphVarSettings:
    values: dict[str, Any] = {
        "matlab_executable": _touch(tmp_path / "MATLAB" / "R2025b" / "bin" / "matlab.exe"),
        "graphvar_root": (tmp_path / "GraphVar").resolve(),
        "viewer_config": _touch(tmp_path / "viewer.mat"),
    }
    (tmp_path / "GraphVar").mkdir(exist_ok=True)
    values.update(overrides)
    return GraphVarSettings(**values)


def _brainnet(tmp_path: Path) -> BrainNetSettings:
    (tmp_path / "BrainNet").mkdir(exist_ok=True)
    return BrainNetSettings(tmp_path / "BrainNet", _touch(tmp_path / "mesh.nv"), _touch(tmp_path / "plt.mat"), top_k=2)


def test_node_sets_68_triple_and_custom_dan(tmp_path):
    assert len(graph_nodes(_graphvar(tmp_path))) == 68
    triple = graph_nodes(_graphvar(tmp_path, node_set=NODE_SET_TRIPLE))
    assert len(triple) == 28
    assert {node.module for node in triple} == {1, 2, 3}
    assert next(n for n in triple if n.label == "inferiorparietal-lh").module == 3  # DMN
    dan = graph_nodes(_graphvar(tmp_path, node_set=NODE_SET_CUSTOM, custom_name="DAN", custom_labels=DAN))
    assert [n.label for n in dan] == [label for label in DK68_LABELS if label in DAN]
    assert dan[0].display_label == "L.caudalmiddlefrontal"
    assert settings_problems(_graphvar(tmp_path, node_set=NODE_SET_CUSTOM, custom_labels=DAN[:2]), None, False) == [
        "GraphVar needs at least 3 regions; Custom has 2."
    ]


def test_release_is_read_from_the_matlab_path(tmp_path):
    assert _graphvar(tmp_path).release == "2025b"
    settings = _graphvar(tmp_path, matlab_executable=_touch(tmp_path / "bin" / "matlab.exe"))
    assert any("MATLAB release" in problem for problem in settings_problems(settings, None, False))


def test_workbook_passes_graphvar_validation(tmp_path):
    from neurostat.graphvar.config import parse_graphvar_config
    from neurostat.graphvar.workspace import validate_atlas_workbook
    from neurostat.gui.graphvar_setup import build_graphvar_block

    settings = _graphvar(tmp_path, node_set=NODE_SET_CUSTOM, custom_name="DAN", custom_labels=DAN)
    (settings.graphvar_root / "src" / "calc").mkdir(parents=True)
    (settings.graphvar_root / "src" / "calc" / "graphvar_glm.m").write_text("function graphvar_glm\nend\n", encoding="utf-8")
    (settings.graphvar_root / "start_GraphVar.m").write_text("% Release= 2.03a\n", encoding="utf-8")
    workbook = write_atlas_workbook(tmp_path / "atlas.xlsx", graph_nodes(settings))
    assert workbook.read_bytes() == write_atlas_workbook(tmp_path / "again.xlsx", graph_nodes(settings)).read_bytes()
    block = build_graphvar_block(settings, arms=("active", "control"), sessions=("pre", "post", "follow"), conditions=("rest",), bands=("alpha", "beta"), workbook=workbook)
    assert [pair["name"] for pair in block["families"][0]["session_pairs"]] == ["pre-post", "pre-follow"]
    project = CleanProject(
        tmp_path / "project.yaml",
        Study("s", ("active", "control"), ("pre", "post", "follow"), ("rest",)),
        (),
        MappingProxyType({"default": MappingProxyType({})}),
        tmp_path / "out",
        MappingProxyType({"graphvar": _freeze(block)}),
    )
    mapping = validate_atlas_workbook(parse_graphvar_config(project))
    assert len(mapping) == 12 and mapping[0]["display_label"] == "L.caudalmiddlefrontal"


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _components(path: Path, rows: list[tuple[str, bool, int]]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    with (path / "components-all.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["workspace", "primary_node_pfwe_supported", "component_link_count"])
        for workspace, supported, links in rows:
            writer.writerow([workspace, str(supported), links])
    return path.parent


def test_supported_workspaces_keep_order_and_need_top_k_links(tmp_path):
    run = _components(tmp_path / "gv" / "graphvar", [("a", False, 9), ("b", True, 9), ("c", True, 1), ("d", True, 4)])
    assert supported_workspaces(run, 2) == ["b", "d"]


@dataclass(frozen=True, slots=True)
class FakeRun:
    run_id: str
    run_dir: Path
    reused: bool = False


def _controller(tmp_path: Path, supported: list[tuple[str, bool, int]], calls: list[Any]) -> WorkflowController:
    graphvar_run = FakeRun("gv", _components(tmp_path / "gv" / "graphvar", supported))
    fake = {name: FakeRun(name, tmp_path / name) for name in ("ingest", "scouts", "connectivity", "network", "workspace", "artifact", "render")}

    def writer(document: dict[str, Any], path: Path) -> CleanProject:
        calls.append(("write", path.name, sorted(key for key in document if key in {"graphvar", "brainnet"})))
        study = document["study"]
        return CleanProject(
            path,
            Study(study["id"], tuple(study["arms"]), tuple(study["sessions"]), tuple(study["conditions"])),
            (),
            MappingProxyType({"default": MappingProxyType({})}),
            tmp_path / "out",
            MappingProxyType({key: _freeze(document[key]) for key in ("network_models", "graphvar", "brainnet") if key in document}),
        )

    def brainnet_build(project: CleanProject, run: object) -> FakeRun:
        assert run is graphvar_run
        calls.append(("brainnet_build", tuple(project.extras["brainnet"]["selection"]["workspaces"])))
        return fake["artifact"]

    services = WorkflowServices(
        ingest=lambda _p, _progress=None: fake["ingest"],
        scouts=lambda _p, _run, _progress=None: fake["scouts"],
        connectivity=lambda _p, _run, _progress=None: fake["connectivity"],
        network_models=lambda _p, _run, _progress=None: fake["network"],
        graphvar_build=lambda _p, _run: fake["workspace"],
        graphvar_analyse=lambda _p, _run: graphvar_run,
        brainnet_build=brainnet_build,
        brainnet_render=lambda _p, _run: fake["render"],
    )
    record = DiscoveredRecord("r1", "p1", "active", "pre", "rest", "raw", "fif", _touch(tmp_path / "r1.fif"))
    return WorkflowController(services=services, discoverer=lambda *_args: (record,), project_writer=writer)


def _study(tmp_path: Path) -> StudyForm:
    return StudyForm(
        "demo", tmp_path, ("active", "control"), ("pre", "post"), ("rest",), tmp_path / "out",
        already_cleaned=True, profile_pair=AnalysisProfilePair.MNE_FSAVERAGE_CONTINUOUS,
        graphvar=_graphvar(tmp_path, node_set=NODE_SET_TRIPLE), brainnet=_brainnet(tmp_path),
    )


def test_run_study_draws_only_supported_workspaces(tmp_path, monkeypatch):
    import neurostat.gui.controller as controller_module

    monkeypatch.setattr(controller_module, "validate_form", lambda *_args: ())
    calls: list[Any] = []
    controller = _controller(tmp_path, [("rest-graphvar__alpha__pre-post", False, 9), ("rest-graphvar__beta__pre-post", True, 9)], calls)
    result = controller.run_study(_study(tmp_path), ("brainnet",))
    assert result.success
    assert ("brainnet_build", ("rest-graphvar__beta__pre-post",)) in calls
    writes = [call for call in calls if call[0] == "write"]
    assert writes[0][2] == ["graphvar"]  # upstream project never carries brainnet
    assert writes[1][1].endswith(".brainnet.project.yaml") and writes[1][2] == ["brainnet", "graphvar"]
    assert (tmp_path / "out" / "graphvar" / "atlas-triple-network.xlsx").is_file()


def test_run_study_skips_brainnet_when_nothing_is_supported(tmp_path, monkeypatch):
    import neurostat.gui.controller as controller_module

    monkeypatch.setattr(controller_module, "validate_form", lambda *_args: ())
    calls: list[Any] = []
    result = _controller(tmp_path, [("rest-graphvar__alpha__pre-post", False, 9)], calls).run_study(_study(tmp_path), ("all",))
    brainnet = result[StageName.BRAINNET]
    assert brainnet.state is StageState.SKIPPED
    assert "nothing to draw" in brainnet.result.message
    assert not any(call[0] == "brainnet_build" for call in calls)


def test_validate_form_reports_graphvar_problems_and_long_paths(tmp_path, monkeypatch):
    import neurostat.graphvar.service as service

    form = _study(tmp_path)
    record = DiscoveredRecord("r1", "p1", "active", "pre", "rest", "raw", "fif", _touch(tmp_path / "r1.fif"))
    monkeypatch.setattr(service, "path_limit", lambda: None)
    assert not [p for p in validate_form(form, (record,), ("graphvar",)) if "GraphVar" in p]
    monkeypatch.setattr(service, "path_limit", lambda: 40)
    problems = validate_form(form, (record,), ("graphvar",))
    assert any("Windows limit of 40" in problem for problem in problems)
    missing = validate_form(StudyForm("demo", tmp_path, ("active", "control"), ("pre", "post"), ("rest",), tmp_path / "out", already_cleaned=True), (record,), ("graphvar", "brainnet"))
    assert any("GraphVar needs" in problem for problem in missing)


def test_graphvar_run_refuses_over_long_paths(tmp_path, monkeypatch):
    import neurostat.graphvar.service as service

    monkeypatch.setattr(service, "path_limit", lambda: 30)
    with pytest.raises(service.GraphVarError, match="closer to the drive root"):
        service._check_path_lengths(tmp_path, ["rest-graphvar__alpha__pre-post"])
    monkeypatch.setattr(service, "path_limit", lambda: None)
    service._check_path_lengths(tmp_path, ["x" * 400])
