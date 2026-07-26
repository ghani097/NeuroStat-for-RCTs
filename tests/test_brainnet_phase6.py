from __future__ import annotations

import csv
import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from neurostat.brainnet import (
    BrainNetConfigError,
    BrainNetError,
    build_brainnet_artifacts,
    parse_brainnet_config,
    render_brainnet_artifacts,
    verify_brainnet_artifact_run,
    verify_brainnet_render_run,
)
from neurostat.brainnet.artifacts import (
    DESCRIPTIVE_SCOPE,
    TOP_LINK_FIELDS,
    select_components_and_links,
)
from neurostat.brainnet.qa import inspect_png
from neurostat.brainnet.runtime import (
    BrainNetExecution,
    BrainNetRuntimeMetadata,
    preflight_brainnet,
    run_brainnet_image,
)
from neurostat.contracts import RecordKey
from neurostat.graphvar.service import COMPONENT_FIELDS, LINK_FIELDS
from neurostat.graphvar.workspace import _all_files
from neurostat.project import CleanProject, ProjectRecord, Study
from neurostat.stage_io import atomic_json, digest_json, sha256_file


def _section(
    tmp_path: Path, *, workspace: str = "ws-one", top_k: int = 2
) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "surface.nv").write_bytes(b"surface")
    (tmp_path / "viewer.mat").write_bytes(b"viewer")
    root = tmp_path / "BrainNet"
    root.mkdir()
    (root / "BrainNet_MapCfg.m").write_text("% Version 1.52\n", encoding="utf-8")
    (root / "BrainNet.m").write_text("% Version 1.63\n", encoding="utf-8")
    (root / "UpdateLog.txt").write_text("Version 1.7\n", encoding="utf-8")
    return {
        "resources": {"surface": "surface.nv", "viewer_config": "viewer.mat"},
        "selection": {
            "workspaces": [workspace],
            "policy": "largest_primary_node_pfwe",
            "top_k": top_k,
        },
        "nodes": {
            "styles": [
                {
                    "connectivity_label": "dk-a",
                    "display_label": "A",
                    "color": 1,
                    "size": 4,
                },
                {
                    "connectivity_label": "dk-b",
                    "display_label": "B",
                    "color": 1,
                    "size": 4,
                },
                {
                    "connectivity_label": "dk-c",
                    "display_label": "C",
                    "color": 2,
                    "size": 5,
                },
            ]
        },
        "render": {"width_pixels": 8, "height_pixels": 6, "dpi": 300},
        "runtime": {
            "executable": sys.executable,
            "brainnet_root": "BrainNet",
            "matlab_release": "2025b",
            "timeout_seconds": 10,
        },
    }


def _project(tmp_path: Path, section: dict[str, Any] | None = None) -> CleanProject:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "input.fif"
    source.write_bytes(b"x")
    if section is None:
        section = _section(tmp_path)
    return CleanProject(
        tmp_path / "project.yaml",
        Study("study", ("active", "control"), ("pre", "post"), ("sep",)),
        (
            ProjectRecord(
                "r1",
                RecordKey("p1", "active", "pre", "sep"),
                "epochs",
                "fif",
                source,
                "default",
            ),
        ),
        MappingProxyType({"default": MappingProxyType({})}),
        tmp_path / "outputs",
        MappingProxyType({"brainnet": MappingProxyType(section)}),
    )


def _component(
    component_id: str = "ws-one_C01",
    *,
    workspace: str = "ws-one",
    primary_p: float = 0.01,
    primary_supported: bool = True,
    nbs_supported: bool = False,
) -> dict[str, Any]:
    permutations = 99
    primary_exceedances = round(primary_p * (permutations + 1) - 1)
    nbs_exceedances = 49
    return {
        "holm_family": "sep-primary",
        "family": "primary",
        "condition": "sep",
        "workspace": workspace,
        "band": "theta",
        "session_pair": "pre-post",
        "component_id": component_id,
        "effect_label": "arm_active-arm_control=0",
        "effect_orientation_multiplier": 1,
        "component_node_count": 3,
        "component_link_count": 2,
        "member_rois": "A; B; C",
        "component_forming_alpha": 0.05,
        "permutations": permutations,
        "primary_node_exceedances": primary_exceedances,
        "primary_node_pfwe_graphvar_tail": primary_exceedances / permutations,
        "primary_node_pfwe_plus_one": (primary_exceedances + 1) / (permutations + 1),
        "primary_node_pfwe_supported": primary_supported,
        "posthoc_nbs_edge_exceedances": nbs_exceedances,
        "posthoc_nbs_edge_pfwe_graphvar_tail": nbs_exceedances / permutations,
        "posthoc_nbs_edge_pfwe_plus_one": (nbs_exceedances + 1) / (permutations + 1),
        "posthoc_nbs_edge_pfwe_supported": nbs_supported,
        "graphvar_observation_rows": 20,
        "graphvar_numerator_df": 1.0,
        "graphvar_denominator_df": 17.0,
        "inference_scope": (
            "Primary study-wide support is based only on participating-node "
            "component pFWE plus its condition-level Holm correction; edge "
            "extent is a separate post-hoc NBS family."
        ),
        "primary_node_holm_all_candidates": 0.08,
        "primary_studywide_supported": False,
        "posthoc_nbs_edge_holm_all_candidates": 0.9,
        "posthoc_nbs_studywide_supported": False,
    }


def _links(component: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for first, second, b_value, f_value, p_value in (
        (1, 2, 2.0, 4.0, 0.02),
        (1, 3, -3.0, 9.0, 0.01),
    ):
        labels = {1: ("dk-a", "A"), 2: ("dk-b", "B"), 3: ("dk-c", "C")}
        row = {
            "holm_family": component["holm_family"],
            "condition": component["condition"],
            "workspace": component["workspace"],
            "band": component["band"],
            "session_pair": component["session_pair"],
            "component_id": component["component_id"],
            "component_node_count": component["component_node_count"],
            "component_link_count": component["component_link_count"],
            "roi1_result_index_1based": first,
            "roi2_result_index_1based": second,
            "roi1_atlas_index_1based": first + 10,
            "roi2_atlas_index_1based": second + 10,
            "roi1_connectivity_label": labels[first][0],
            "roi2_connectivity_label": labels[second][0],
            "roi1": labels[first][1],
            "roi2": labels[second][1],
            "effect_label": component["effect_label"],
            "paired_change_contrast_b": b_value,
            "f_statistic": f_value,
            "parametric_p_uncorrected": p_value,
            "permutation_p_uncorrected": p_value,
            "effect_direction": "descriptive",
            "inference_scope": (
                "Member link is descriptive and uncorrected; corrected inference "
                "applies only to the complete component."
            ),
            "primary_node_pfwe_plus_one": component["primary_node_pfwe_plus_one"],
            "primary_node_holm_all_candidates": component[
                "primary_node_holm_all_candidates"
            ],
            "primary_studywide_supported": component["primary_studywide_supported"],
            "posthoc_nbs_edge_pfwe_plus_one": component[
                "posthoc_nbs_edge_pfwe_plus_one"
            ],
            "posthoc_nbs_edge_holm_all_candidates": component[
                "posthoc_nbs_edge_holm_all_candidates"
            ],
            "posthoc_nbs_studywide_supported": component[
                "posthoc_nbs_studywide_supported"
            ],
        }
        result.append(row)
    return result


def _csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _graphvar_run(
    tmp_path: Path,
    project: CleanProject,
    *,
    component: dict[str, Any] | None = None,
    components: list[dict[str, Any]] | None = None,
) -> Path:
    if components is None:
        components = [_component() if component is None else component]
    run = tmp_path / "graphvar-source"
    graphvar = run / "graphvar"
    graphvar.mkdir(parents=True)
    node_map = [
        {
            "result_index_1based": index,
            "atlas_index_1based": index + 10,
            "connectivity_label": f"dk-{label.lower()}",
            "display_label": label,
            "coordinates": [float(index), float(index + 1), float(index + 2)],
        }
        for index, label in enumerate(("A", "B", "C"), 1)
    ]
    atomic_json(graphvar / "selected-node-map.json", node_map)
    _csv(graphvar / "components-all.csv", COMPONENT_FIELDS, components)
    _csv(
        graphvar / "component-member-links-all.csv",
        LINK_FIELDS,
        [link for item in components for link in _links(item)],
    )
    atomic_json(graphvar / "status.json", {"state": "succeeded"})
    atomic_json(graphvar / "validation.json", {"status": "passed"})
    identity = {
        "stage": "graphvar",
        "project": project.normalized(),
        "config": {
            "families": [
                {
                    "name": "primary",
                    "holm_family": "sep-primary",
                    "familywise_alpha": 0.05,
                }
            ]
        },
        "source": "test",
    }
    run_id = digest_json(identity)[:12]
    lock = identity | {"run_id": run_id}
    atomic_json(run / "project.lock.json", lock)
    atomic_json(
        graphvar / "manifest.json",
        {
            "schema": "neurostat.graphvar_analysis.v1",
            "run_id": run_id,
            "scientific_identity_sha256": digest_json(identity),
            "artifacts": _all_files(run, {"graphvar/manifest.json"}),
        },
    )
    return run


def _png(
    path: Path,
    width: int = 8,
    height: int = 6,
    dpi: int = 300,
    *,
    constant: bool = False,
) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = []
    for y in range(height):
        values = bytearray()
        for x in range(width):
            value = 255 if constant else (255 if (x + y) % 2 else 0)
            values.extend(
                (
                    value,
                    (x * 41) % 256 if not constant else value,
                    255 - value if not constant else value,
                )
            )
        rows.append(b"\x00" + bytes(values))
    ppm = round(dpi / 0.0254)
    content = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(content)


def test_config_exact_paths_and_style_contract(tmp_path: Path) -> None:
    project = _project(tmp_path)
    config = parse_brainnet_config(project)
    assert config.selection.policy == "largest_primary_node_pfwe"
    assert config.selection.top_k == 2
    assert config.styles[0].color == 1.0
    bad = dict(project.extras["brainnet"])
    bad["unknown"] = True
    with pytest.raises(BrainNetConfigError, match="unknown keys"):
        parse_brainnet_config(_project(tmp_path / "bad", bad))

    single = _section(tmp_path / "single")
    single["nodes"]["styles"] = [single["nodes"]["styles"][0]]
    assert len(parse_brainnet_config(_project(tmp_path / "single", single)).styles) == 1


def test_selection_is_primary_only_and_has_stable_top_link_order(
    tmp_path: Path,
) -> None:
    config = parse_brainnet_config(_project(tmp_path))
    primary = _component()
    links = _links(primary)
    links[0]["permutation_p_uncorrected"] = 0.01
    links[1]["permutation_p_uncorrected"] = 0.01
    selections, selected = select_components_and_links(config, [primary], links)
    assert selections[0]["component_id"] == "ws-one_C01"
    assert [row["roi2"] for row in selected] == ["C", "B"]
    assert [row["display_rank"] for row in selected] == [1, 2]
    assert selected[0]["signed_weight"] == -3.0
    assert selected[0]["inference_scope"] == DESCRIPTIVE_SCOPE

    nbs_only = _component(primary_p=0.5, primary_supported=False, nbs_supported=True)
    with pytest.raises(BrainNetError, match="no primary node-pFWE"):
        select_components_and_links(config, [nbs_only], _links(nbs_only))


def test_build_exact_node_edge_manifest_reuse_and_tamper(tmp_path: Path) -> None:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    run = build_brainnet_artifacts(project, source)
    assert not run.reused
    assert run.workspaces == ("ws-one",)
    node_lines = (run.run_dir / "brainnet" / "network.node").read_text().splitlines()
    assert node_lines[0] == "1\t2\t3\t1\t4\tA"
    edge = [
        [float(value) for value in line.split("\t")]
        for line in (run.run_dir / "brainnet" / "edges" / "ws-one.edge")
        .read_text()
        .splitlines()
    ]
    assert edge == [[0.0, 2.0, -3.0], [2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]
    with (run.run_dir / "brainnet" / "top-links.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))
    assert list(manifest[0]) == list(TOP_LINK_FIELDS)
    assert [row["display_rank"] for row in manifest] == ["1", "2"]
    assert build_brainnet_artifacts(project, source).reused

    edge_path = run.run_dir / "brainnet" / "edges" / "ws-one.edge"
    edge_path.write_text(edge_path.read_text().replace("-3.0000", "-2.0000"))
    with pytest.raises((BrainNetError, RuntimeError), match="SHA-256|hash"):
        verify_brainnet_artifact_run(run.run_dir)


def test_graphvar_schema_identity_and_style_coverage_fail_closed(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    component_file = source / "graphvar" / "components-all.csv"
    text = component_file.read_text()
    component_file.write_text(text.replace("ws-one_C01", "ws-one_C99"))
    with pytest.raises((BrainNetError, RuntimeError), match="SHA-256|hash"):
        build_brainnet_artifacts(project, source)

    project2 = _project(tmp_path / "coverage")
    source2 = _graphvar_run(tmp_path / "coverage", project2)
    section = dict(project2.extras["brainnet"])
    nodes = dict(section["nodes"])
    nodes["styles"] = list(nodes["styles"])[:-1]
    section["nodes"] = nodes
    with pytest.raises(BrainNetError, match="coverage differs"):
        build_brainnet_artifacts(
            _project(tmp_path / "coverage", section),
            source2,
        )


def test_prepared_build_does_not_require_installed_brainnet(tmp_path: Path) -> None:
    section = _section(tmp_path)
    section["runtime"]["brainnet_root"] = "not-installed-yet"
    project = _project(tmp_path, section)
    source = _graphvar_run(tmp_path, project)
    run = build_brainnet_artifacts(project, source)
    assert (run.run_dir / "brainnet" / "network.node").is_file()
    config = parse_brainnet_config(project)
    with pytest.raises(BrainNetError, match="lacks BrainNet_MapCfg"):
        preflight_brainnet(config, tmp_path / "preflight-missing")


def test_png_qa_dimensions_dpi_crc_and_nonblank(tmp_path: Path) -> None:
    path = tmp_path / "valid.png"
    _png(path)
    qa = inspect_png(path, width_pixels=8, height_pixels=6, dpi=300)
    assert qa["channels"] == 3
    assert qa["nonconstant"]
    assert max(qa["channel_standard_deviation"]) > 1
    constant = tmp_path / "constant.png"
    _png(constant, constant=True)
    with pytest.raises(BrainNetError, match="blank or constant"):
        inspect_png(constant, width_pixels=8, height_pixels=6, dpi=300)
    with pytest.raises(BrainNetError, match="dimensions differ"):
        inspect_png(path, width_pixels=9, height_pixels=6, dpi=300)
    damaged = tmp_path / "damaged.png"
    damaged.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(BrainNetError, match="CRC"):
        inspect_png(damaged, width_pixels=8, height_pixels=6, dpi=300)


def _fake_matlab(
    args: Any,
    *,
    env: dict[str, str],
    **_: Any,
) -> subprocess.CompletedProcess[str]:
    request = json.loads(Path(env["NEUROSTAT_BRAINNET_REQUEST"]).read_text())
    if request["mode"] == "preflight":
        atomic_json(
            Path(request["metadata_path"]),
            {
                "state": "passed",
                "matlab_release": "2025b",
                "brainnet_mapcfg": str(
                    Path(request["brainnet_root"]) / "BrainNet_MapCfg.m"
                ),
                "whitebg": str(Path(request["compat_dir"]) / "whitebg.m"),
            },
        )
    else:
        _png(
            Path(request["output"]),
            request["width_pixels"],
            request["height_pixels"],
            request["dpi"],
        )
        atomic_json(
            Path(request["metadata_path"]),
            {
                "state": "succeeded",
                "output": request["output"],
                "width_pixels": request["width_pixels"],
                "height_pixels": request["height_pixels"],
                "dpi": request["dpi"],
                "text_objects": 4,
                "axes_objects": 1,
                "brainnet_mapcfg": str(
                    Path(request["brainnet_root"]) / "BrainNet_MapCfg.m"
                ),
                "whitebg": str(Path(request["compat_dir"]) / "whitebg.m"),
            },
        )
    return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")


def test_runtime_preflight_render_and_false_success(tmp_path: Path) -> None:
    config = parse_brainnet_config(_project(tmp_path))
    preflight = preflight_brainnet(
        config, tmp_path / "preflight", process_runner=_fake_matlab
    )
    assert preflight.mapcfg_version_evidence == "% Version 1.52"
    assert preflight.brainnet_gui_version_evidence == "% Version 1.63"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("network.node", "network.edge"):
        (inputs / name).write_text("fixture", encoding="utf-8")
    compat = tmp_path / "compat"
    compat.mkdir()
    from neurostat.brainnet.runtime import matlab_resources

    (compat / "whitebg.m").write_bytes(matlab_resources()[1].read_bytes())
    execution = run_brainnet_image(
        config,
        surface=config.resources.surface,
        node=inputs / "network.node",
        edge=inputs / "network.edge",
        viewer_config=config.resources.viewer_config,
        compatibility_dir=compat,
        output_dir=tmp_path / "render",
        workspace="ws-one",
        process_runner=_fake_matlab,
    )
    assert execution.returncode == 0
    assert execution.qa["width_pixels"] == 8

    def false_success(args: Any, **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="metadata"):
        run_brainnet_image(
            config,
            surface=config.resources.surface,
            node=inputs / "network.node",
            edge=inputs / "network.edge",
            viewer_config=config.resources.viewer_config,
            compatibility_dir=compat,
            output_dir=tmp_path / "false",
            workspace="ws-one",
            process_runner=false_success,
        )

    def nonzero(args: Any, **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 7, stdout="", stderr="failed")

    with pytest.raises(BrainNetError, match="exit code 7"):
        preflight_brainnet(config, tmp_path / "nonzero", process_runner=nonzero)


def test_render_service_resume_final_reuse_and_tamper(tmp_path: Path) -> None:
    section = _section(tmp_path)
    section["selection"]["workspaces"] = ["ws-one", "ws-two"]
    project = _project(tmp_path, section)
    source = _graphvar_run(
        tmp_path,
        project,
        components=[
            _component("ws-one_C01", workspace="ws-one"),
            _component("ws-two_C01", workspace="ws-two"),
        ],
    )
    prepared = build_brainnet_artifacts(project, source)
    config = parse_brainnet_config(project)

    def fake_preflight(_: Any, output_dir: Path) -> BrainNetRuntimeMetadata:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "preflight.stdout.log").write_text("ok", encoding="utf-8")
        mapcfg = config.runtime.brainnet_root / "BrainNet_MapCfg.m"
        gui = config.runtime.brainnet_root / "BrainNet.m"
        return BrainNetRuntimeMetadata(
            Path(sys.executable).resolve(),
            "2025b",
            config.runtime.brainnet_root,
            mapcfg,
            sha256_file(mapcfg),
            sha256_file(gui),
            sha256_file(config.runtime.brainnet_root / "UpdateLog.txt"),
            "% Version 1.52",
            "% Version 1.63",
            "Version 1.7",
            "a" * 64,
            "b" * 64,
        )

    calls: list[str] = []
    fail_second = True

    def fake_runner(
        _: Any,
        *,
        output_dir: Path,
        workspace: str,
        **__: Any,
    ) -> BrainNetExecution:
        nonlocal fail_second
        calls.append(workspace)
        output_dir.mkdir(parents=True, exist_ok=True)
        if workspace == "ws-two" and fail_second:
            fail_second = False
            raise BrainNetError("injected second-image failure")
        image = output_dir / f"{workspace}.png"
        _png(image)
        stdout = output_dir / "matlab.stdout.log"
        stderr = output_dir / "matlab.stderr.log"
        metadata = output_dir / "execution.json"
        stdout.write_text("ok", encoding="utf-8")
        stderr.write_text("", encoding="utf-8")
        atomic_json(metadata, {"state": "succeeded"})
        return BrainNetExecution(
            0,
            1.0,
            2.0,
            1.0,
            stdout,
            stderr,
            metadata,
            image,
            {},
        )

    with pytest.raises(BrainNetError, match="injected"):
        render_brainnet_artifacts(
            project,
            prepared,
            preflight_function=fake_preflight,
            image_runner=fake_runner,
        )
    assert calls == ["ws-one", "ws-two"]
    final = render_brainnet_artifacts(
        project,
        prepared,
        preflight_function=fake_preflight,
        image_runner=fake_runner,
    )
    assert calls == ["ws-one", "ws-two", "ws-two"]
    assert not final.reused
    assert [path.stem for path in final.images] == ["ws-one", "ws-two"]
    index = json.loads((final.run_dir / "brainnet" / "image-index.json").read_text())
    assert [row["workspace"] for row in index] == ["ws-one", "ws-two"]

    def must_not_render(*_: Any, **__: Any) -> BrainNetExecution:
        raise AssertionError("an exact final run must be reused")

    reused = render_brainnet_artifacts(
        project,
        prepared,
        preflight_function=fake_preflight,
        image_runner=must_not_render,
    )
    assert reused.reused
    image = final.images[0]
    image.write_bytes(image.read_bytes()[:-1] + b"x")
    with pytest.raises((BrainNetError, RuntimeError), match="SHA-256|hash"):
        verify_brainnet_render_run(final.run_dir)


@pytest.mark.skipif(
    not os.environ.get("NEUROSTAT_TEST_V7_GRAPHVAR_RUN")
    or not os.environ.get("NEUROSTAT_TEST_V7_PROJECT")
    or not os.environ.get("NEUROSTAT_TEST_V7_BRAINNET_MANIFEST")
    or not os.environ.get("NEUROSTAT_TEST_V7_BRAINNET_EDGE_DIR"),
    reason="set exact V7 GraphVar run and Phase-6 project for private parity",
)
def test_accepted_v7_top8_private_parity() -> None:
    project_path = Path(os.environ["NEUROSTAT_TEST_V7_PROJECT"])
    graphvar_run = Path(os.environ["NEUROSTAT_TEST_V7_GRAPHVAR_RUN"])
    accepted = Path(os.environ["NEUROSTAT_TEST_V7_BRAINNET_MANIFEST"])
    accepted_edges = Path(os.environ["NEUROSTAT_TEST_V7_BRAINNET_EDGE_DIR"])
    run = build_brainnet_artifacts(project_path, graphvar_run)
    expected_order = (
        "SEP_Theta_Post_Post4W",
        "SEP_Theta_Pre_Post",
        "SEP_Alpha_Pre_Post",
        "SEP_Gamma_Pre_Post",
    )
    assert run.workspaces == expected_order
    with (run.run_dir / "brainnet" / "top-links.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        observed = list(csv.DictReader(handle))
    with accepted.open(encoding="utf-8-sig", newline="") as handle:
        expected = list(csv.DictReader(handle))
    assert len(observed) == len(expected) == 32
    for ours, theirs in zip(observed, expected, strict=True):
        assert ours["workspace"] == theirs["Workspace"]
        assert ours["component_id"] == theirs["ComponentID"]
        assert int(ours["display_rank"]) == int(theirs["DisplayRank"])
        assert ours["roi1_connectivity_label"] == theirs["ROI1"]
        assert ours["roi2_connectivity_label"] == theirs["ROI2"]
        assert float(ours["signed_weight"]) == pytest.approx(
            float(theirs["SignedT"]), abs=1e-10
        )
    for workspace in expected_order:

        def matrix(path: Path) -> list[list[float]]:
            return [
                [float(value) for value in line.split()]
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        ours = matrix(run.run_dir / "brainnet" / "edges" / f"{workspace}.edge")
        theirs = matrix(accepted_edges / f"{workspace}_top8.edge")
        assert len(ours) == len(theirs) == 28
        assert all(len(row) == 28 for row in ours + theirs)
        ours_nonzero = {
            (first, second)
            for first in range(28)
            for second in range(first + 1, 28)
            if ours[first][second] != 0
        }
        theirs_nonzero = {
            (first, second)
            for first in range(28)
            for second in range(first + 1, 28)
            if theirs[first][second] != 0
        }
        assert ours_nonzero == theirs_nonzero
        assert len(ours_nonzero) == 8
        for first in range(28):
            assert ours[first][first] == theirs[first][first] == 0
            for second in range(28):
                assert ours[first][second] == ours[second][first]
                assert theirs[first][second] == theirs[second][first]
                assert ours[first][second] == pytest.approx(
                    theirs[first][second], abs=5e-5
                )
    frozen = json.loads(
        (run.run_dir / "brainnet" / "selected-node-map.json").read_text()
    )
    frozen_labels = {row["connectivity_label"]: row["display_label"] for row in frozen}
    assert frozen_labels["L.MT"] == "L.LMT"
    assert frozen_labels["R.MT"] == "R.LMT"
    node_labels = [
        line.split("\t")[5]
        for line in (run.run_dir / "brainnet" / "network.node")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert "L.LMT" in node_labels and "R.LMT" in node_labels
