from __future__ import annotations

import csv
import json
import os
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

import neurostat.brainnet.artifacts as artifact_service
import neurostat.brainnet.service as render_service
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
            "effect_direction": (
                "Positive: active has a more positive second-minus-first "
                "change than control"
                if b_value > 0
                else "Negative: active has a more negative second-minus-first "
                "change than control"
                if b_value < 0
                else "Zero configured paired-change contrast"
            ),
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
    member_links: list[dict[str, Any]] | None = None,
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
        (
            [link for item in components for link in _links(item)]
            if member_links is None
            else member_links
        ),
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
                    "effect": {
                        "numerator_arm": "active",
                        "denominator_arm": "control",
                    },
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


@pytest.mark.parametrize(
    "workspace",
    (
        ".",
        "..",
        "../escape",
        "nested/workspace",
        r"nested\workspace",
        "C:workspace",
        "CON",
        "NUL.csv",
        "workspace.",
    ),
)
def test_config_rejects_unsafe_workspace_artifact_slugs(
    tmp_path: Path, workspace: str
) -> None:
    section = _section(tmp_path, workspace=workspace)
    with pytest.raises(BrainNetConfigError, match="safe artifact name"):
        parse_brainnet_config(_project(tmp_path, section))


def test_runtime_rejects_unsafe_workspace_before_creating_output(
    tmp_path: Path,
) -> None:
    config = parse_brainnet_config(_project(tmp_path))
    output = tmp_path / "render-output"
    with pytest.raises(BrainNetError, match="single-component artifact slug"):
        run_brainnet_image(
            config,
            surface=tmp_path / "surface",
            node=tmp_path / "node",
            edge=tmp_path / "edge",
            viewer_config=tmp_path / "viewer",
            compatibility_dir=tmp_path / "compatibility",
            output_dir=output,
            workspace="../escape",
        )
    assert not output.exists()


def test_config_rejects_casefold_colliding_workspaces_before_output(
    tmp_path: Path,
) -> None:
    section = _section(tmp_path)
    section["selection"]["workspaces"] = ["ws-one", "WS-ONE"]
    with pytest.raises(BrainNetConfigError, match="case-insensitively unique"):
        parse_brainnet_config(_project(tmp_path, section))
    assert not (tmp_path / "outputs").exists()


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
    repaired = build_brainnet_artifacts(project, source)
    assert verify_brainnet_artifact_run(repaired.run_dir).run_id == run.run_id
    assert list((project.output_root / "runs").glob(f".{run.run_dir.name}.invalid-*"))


def test_artifact_semantic_verifier_rejects_coherently_rehashed_science(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    run = build_brainnet_artifacts(project, source)
    forged = tmp_path / "forged-artifact"
    shutil.copytree(run.run_dir, forged)
    top_json_path = forged / "brainnet" / "top-links.json"
    rows = json.loads(top_json_path.read_text(encoding="utf-8"))
    target = rows[0]
    target.update(
        {
            "paired_change_contrast_b": 4.0,
            "abs_b": 4.0,
            "f_statistic": 16.0,
            "signed_weight": 4.0,
        }
    )
    atomic_json(top_json_path, rows)
    _csv(forged / "brainnet" / "top-links.csv", TOP_LINK_FIELDS, rows)
    edge_path = forged / "brainnet" / "edges" / "ws-one.edge"
    matrix = [
        [float(value) for value in line.split("\t")]
        for line in edge_path.read_text(encoding="utf-8").splitlines()
    ]
    first = target["roi1_result_index_1based"] - 1
    second = target["roi2_result_index_1based"] - 1
    matrix[first][second] = matrix[second][first] = 4.0
    edge_path.write_text(
        "".join(
            "\t".join(format(value, ".4f") for value in row) + "\n" for row in matrix
        ),
        encoding="utf-8",
        newline="\n",
    )
    _rehash_artifact_manifest(forged)
    with pytest.raises(BrainNetError, match="reproducible"):
        verify_brainnet_artifact_run(forged)


def test_nonzero_edge_that_rounds_to_zero_never_publishes(
    tmp_path: Path,
) -> None:
    section = _section(tmp_path, top_k=1)
    project = _project(tmp_path, section)
    component = _component()
    member_links = _links(component)
    member_links[1]["f_statistic"] = 1e-10
    source = _graphvar_run(
        tmp_path,
        project,
        component=component,
        member_links=member_links,
    )
    with pytest.raises(BrainNetError, match="rounds to zero"):
        build_brainnet_artifacts(project, source)
    assert not list((project.output_root / "runs").glob("brainnet-artifacts-*"))

    valid_links = _links(component)
    valid_source = _graphvar_run(
        tmp_path / "valid-source",
        project,
        component=component,
        member_links=valid_links,
    )
    run = build_brainnet_artifacts(project, valid_source)
    assert verify_brainnet_artifact_run(run.run_dir).run_id == run.run_id


def test_artifact_post_publish_verification_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    original = artifact_service.verify_brainnet_artifact_run
    injected = False

    def fail_once(
        run_dir: str | Path, expected_lock: dict[str, Any] | None = None
    ) -> Any:
        nonlocal injected
        path = Path(run_dir)
        if path.name.startswith("brainnet-artifacts-") and not injected:
            injected = True
            raise BrainNetError("injected artifact post-publish failure")
        return original(run_dir, expected_lock)

    monkeypatch.setattr(artifact_service, "verify_brainnet_artifact_run", fail_once)
    with pytest.raises(BrainNetError, match="post-publish"):
        build_brainnet_artifacts(project, source)
    assert injected
    assert not list((project.output_root / "runs").glob("brainnet-artifacts-*"))
    monkeypatch.setattr(artifact_service, "verify_brainnet_artifact_run", original)
    run = build_brainnet_artifacts(project, source)
    assert verify_brainnet_artifact_run(run.run_dir).run_id == run.run_id


@pytest.mark.parametrize(
    "relative",
    (
        "project.lock.json",
        "brainnet/manifest.json",
        "brainnet/source-science/components-all.csv",
        "brainnet/resources/viewer-config.mat",
        "brainnet/matlab/whitebg.m",
    ),
)
def test_artifact_verifier_rejects_same_byte_external_file_links(
    tmp_path: Path, relative: str
) -> None:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    run = build_brainnet_artifacts(project, source)
    forged = tmp_path / "linked-artifact"
    shutil.copytree(run.run_dir, forged)
    _replace_with_external_file_symlink(forged, relative, tmp_path / "external-files")
    with pytest.raises(BrainNetError, match="symlink, junction, or reparse"):
        verify_brainnet_artifact_run(forged)


def test_artifact_builder_quarantines_root_link_without_moving_target(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    run = build_brainnet_artifacts(project, source)
    final = run.run_dir
    external = tmp_path / "external-artifact-final"
    shutil.copytree(final, external)
    external_manifest_sha256 = sha256_file(external / "brainnet" / "manifest.json")
    shutil.rmtree(final)
    os.symlink(external, final, target_is_directory=True)

    rebuilt = build_brainnet_artifacts(project, source)

    quarantined = tuple(
        final.parent.glob(f".{final.name}.invalid-*"),
    )
    try:
        assert rebuilt.run_dir == final
        assert not final.is_symlink()
        assert verify_brainnet_artifact_run(final).run_id == run.run_id
        assert len(quarantined) == 1
        assert quarantined[0].is_symlink()
        assert sha256_file(external / "brainnet" / "manifest.json") == (
            external_manifest_sha256
        )
    finally:
        for linked in quarantined:
            if linked.is_symlink():
                linked.unlink()


@pytest.mark.parametrize("link_kind", ("symlink", "junction"))
def test_artifact_verifier_rejects_directory_links_without_following(
    tmp_path: Path, link_kind: str
) -> None:
    if link_kind == "junction" and os.name != "nt":
        pytest.skip("Windows junctions are only available on Windows")
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    run = build_brainnet_artifacts(project, source)
    forged = tmp_path / f"directory-linked-artifact-{link_kind}"
    shutil.copytree(run.run_dir, forged)
    target = forged / "brainnet" / "source-science"
    external = tmp_path / f"external-source-science-{link_kind}"
    shutil.copytree(target, external)
    shutil.rmtree(target)
    if link_kind == "symlink":
        os.symlink(external, target, target_is_directory=True)
    else:
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(target), str(external)],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        if completed.returncode != 0:
            pytest.skip(f"junction creation is unavailable: {completed.stderr}")
    try:
        with pytest.raises(BrainNetError, match="symlink, junction, or reparse"):
            verify_brainnet_artifact_run(forged)
    finally:
        if os.path.lexists(target):
            if target.is_symlink():
                target.unlink()
            else:
                os.rmdir(target)


def test_artifact_verifier_rejects_forged_traversal_workspace(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    run = build_brainnet_artifacts(project, source)
    forged = tmp_path / "unsafe-workspace-artifact"
    shutil.copytree(run.run_dir, forged)
    _forge_workspace_lock(forged, "../outside")
    with pytest.raises(BrainNetError, match="single-component artifact slugs"):
        verify_brainnet_artifact_run(forged)


def test_artifact_verifier_rejects_forged_casefold_workspace_collision(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    run = build_brainnet_artifacts(project, source)
    forged = tmp_path / "casefold-workspace-artifact"
    shutil.copytree(run.run_dir, forged)
    _forge_workspace_lock(forged, ["ws-one", "WS-ONE"])
    with pytest.raises(BrainNetError, match="case-insensitively unique"):
        verify_brainnet_artifact_run(forged)


def test_graphvar_source_rejects_casefold_workspace_spellings_before_output(
    tmp_path: Path,
) -> None:
    section = _section(tmp_path, workspace="ws-one")
    project = _project(tmp_path, section)
    components = [
        _component("ws-one_C01", workspace="ws-one"),
        _component("WS-ONE_C01", workspace="WS-ONE"),
    ]
    source = _graphvar_run(tmp_path, project, components=components)
    with pytest.raises(BrainNetError, match="case-insensitively unique"):
        build_brainnet_artifacts(project, source)
    assert not list((project.output_root / "runs").glob("brainnet-artifacts-*"))


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


def _service_preflight(config: Any, output_dir: Path) -> BrainNetRuntimeMetadata:
    return preflight_brainnet(config, output_dir, process_runner=_fake_matlab)


def _service_runner(config: Any, **kwargs: Any) -> BrainNetExecution:
    return run_brainnet_image(config, **kwargs, process_runner=_fake_matlab)


def _rendered_fixture(tmp_path: Path) -> tuple[CleanProject, Path, Any]:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    prepared = build_brainnet_artifacts(project, source)
    rendered = render_brainnet_artifacts(
        project,
        prepared,
        preflight_function=_service_preflight,
        image_runner=_service_runner,
    )
    return project, source, rendered


def _rehash_artifact_manifest(run_dir: Path) -> None:
    path = run_dir / "brainnet" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifacts"] = _all_files(run_dir, {"brainnet/manifest.json"})
    atomic_json(path, manifest)


def _rehash_render_manifests(run_dir: Path, workspace: str = "ws-one") -> None:
    image_dir = run_dir / "brainnet" / "renders" / workspace
    inner_path = image_dir / "output-manifest.json"
    inner = json.loads(inner_path.read_text(encoding="utf-8"))
    inner["artifacts"] = _all_files(image_dir, {"output-manifest.json"})
    atomic_json(inner_path, inner)
    _rehash_artifact_manifest(run_dir)


def _replace_with_external_file_symlink(
    run_dir: Path, relative: str, external_root: Path
) -> None:
    target = run_dir / relative
    external = external_root / relative
    external.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, external)
    target.unlink()
    os.symlink(external, target)
    assert target.is_symlink()
    assert target.read_bytes() == external.read_bytes()


def _create_test_link(link: Path, target: Path, link_kind: str) -> None:
    if link_kind == "file_symlink":
        os.symlink(target, link)
    elif link_kind == "directory_symlink":
        os.symlink(target, link, target_is_directory=True)
    elif link_kind == "junction":
        if os.name != "nt":
            pytest.skip("Windows junctions are only available on Windows")
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
        if completed.returncode != 0:
            pytest.skip(f"junction creation is unavailable: {completed.stderr}")
    else:
        raise AssertionError(f"unsupported test link kind: {link_kind}")
    assert os.path.lexists(link)


def _remove_test_link(link: Path) -> None:
    if not os.path.lexists(link):
        return
    if link.is_symlink():
        link.unlink()
    else:
        junction_check = getattr(link, "is_junction", None)
        assert callable(junction_check) and junction_check()
        os.rmdir(link)


def _forge_workspace_lock(run_dir: Path, workspace: str | list[str]) -> None:
    workspaces = [workspace] if isinstance(workspace, str) else workspace
    lock_path = run_dir / "project.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["config"]["selection"]["workspaces"] = workspaces
    if lock["stage"] == "brainnet_artifacts" and len(workspaces) == 1:
        lock["selection"][0]["workspace"] = workspaces[0]
    identity = {key: value for key, value in lock.items() if key != "run_id"}
    scientific_sha = digest_json(identity)
    lock["run_id"] = scientific_sha[:12]
    atomic_json(lock_path, lock)
    manifest_path = run_dir / "brainnet" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = lock["run_id"]
    manifest["scientific_identity_sha256"] = scientific_sha
    manifest["workspaces"] = workspaces
    manifest["artifacts"] = _all_files(run_dir, {"brainnet/manifest.json"})
    atomic_json(manifest_path, manifest)


@pytest.mark.parametrize(
    "link_kind", ("file_symlink", "directory_symlink", "junction")
)
def test_stale_attempt_cleanup_never_touches_internal_link_target(
    tmp_path: Path, link_kind: str
) -> None:
    runs_root = tmp_path / "runs"
    renders = runs_root / "partial" / "brainnet" / "renders"
    renders.mkdir(parents=True)
    attempt = renders / ".ws-one.attempt-linked"
    if link_kind == "file_symlink":
        target = renders / "internal-target.txt"
        target.write_text("keep", encoding="utf-8")
        marker = target
    else:
        target = renders / f"internal-target-{link_kind}"
        target.mkdir()
        marker = target / "marker.txt"
        marker.write_text("keep", encoding="utf-8")
    _create_test_link(attempt, target, link_kind)

    render_service._discard_stale_attempt(attempt, runs_root, "run-id", "ws-one")

    assert not os.path.lexists(attempt)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not list(runs_root.glob(".brainnet-render-run-id-ws-one-attempt.invalid-*"))


@pytest.mark.parametrize("link_kind", ("directory_symlink", "junction"))
def test_stale_attempt_tree_removal_never_follows_child_directory_link(
    tmp_path: Path, link_kind: str
) -> None:
    runs_root = tmp_path / "runs"
    renders = runs_root / "partial" / "brainnet" / "renders"
    renders.mkdir(parents=True)
    target = renders / f"child-target-{link_kind}"
    target.mkdir()
    marker = target / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    attempt = renders / ".ws-one.attempt-owned"
    attempt.mkdir()
    _create_test_link(attempt / "linked-child", target, link_kind)

    render_service._discard_stale_attempt(attempt, runs_root, "run-id", "ws-one")

    assert not os.path.lexists(attempt)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_stale_attempt_cleanup_rechecks_quarantined_root_after_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    renders = runs_root / "partial" / "brainnet" / "renders"
    renders.mkdir(parents=True)
    attempt = renders / ".ws-one.attempt-owned"
    attempt.mkdir()
    (attempt / "owned.txt").write_text("discard", encoding="utf-8")
    target = renders / "swap-target"
    target.mkdir()
    marker = target / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    real_remove = render_service.remove_entry_no_follow

    def swap_then_remove(path: str | Path) -> None:
        quarantined = Path(path)
        shutil.rmtree(quarantined)
        os.symlink(target, quarantined, target_is_directory=True)
        real_remove(quarantined)

    monkeypatch.setattr(render_service, "remove_entry_no_follow", swap_then_remove)
    render_service._discard_stale_attempt(attempt, runs_root, "run-id", "ws-one")

    assert not os.path.lexists(attempt)
    assert marker.read_text(encoding="utf-8") == "keep"


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
        compat = output_dir / "compat"
        compat.mkdir()
        from neurostat.brainnet.runtime import matlab_resources

        (compat / "whitebg.m").write_bytes(matlab_resources()[1].read_bytes())
        (output_dir / "preflight.stdout.log").write_text("ok", encoding="utf-8")
        (output_dir / "preflight.stderr.log").write_text("", encoding="utf-8")
        mapcfg = config.runtime.brainnet_root / "BrainNet_MapCfg.m"
        gui = config.runtime.brainnet_root / "BrainNet.m"
        atomic_json(
            output_dir / "request.json",
            {
                "mode": "preflight",
                "brainnet_root": str(config.runtime.brainnet_root),
                "compat_dir": str(compat),
                "metadata_path": str(output_dir / "preflight.json"),
            },
        )
        atomic_json(
            output_dir / "preflight.json",
            {
                "state": "passed",
                "matlab_release": "2025b",
                "brainnet_mapcfg": str(mapcfg),
                "whitebg": str(compat / "whitebg.m"),
            },
        )
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
            sha256_file(matlab_resources()[0]),
            sha256_file(matlab_resources()[1]),
        )

    calls: list[str] = []
    fail_second = True

    def fake_runner(
        runner_config: Any,
        *,
        surface: Path,
        node: Path,
        edge: Path,
        viewer_config: Path,
        compatibility_dir: Path,
        output_dir: Path,
        workspace: str,
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
        atomic_json(
            output_dir / "request.json",
            {
                "mode": "render",
                "brainnet_root": str(runner_config.runtime.brainnet_root),
                "compat_dir": str(compatibility_dir.resolve()),
                "surface": str(surface.resolve()),
                "node": str(node.resolve()),
                "edge": str(edge.resolve()),
                "viewer_config": str(viewer_config.resolve()),
                "output": str(image.resolve()),
                "metadata_path": str(metadata.resolve()),
                **runner_config.render.normalized(),
            },
        )
        atomic_json(
            metadata,
            {
                "state": "succeeded",
                "output": str(image.resolve()),
                "width_pixels": runner_config.render.width_pixels,
                "height_pixels": runner_config.render.height_pixels,
                "dpi": runner_config.render.dpi,
                "text_objects": 4,
                "axes_objects": 1,
                "brainnet_mapcfg": str(
                    runner_config.runtime.brainnet_root / "BrainNet_MapCfg.m"
                ),
                "whitebg": str(compatibility_dir.resolve() / "whitebg.m"),
            },
        )
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
    repaired = render_brainnet_artifacts(
        project,
        prepared,
        preflight_function=fake_preflight,
        image_runner=fake_runner,
    )
    assert verify_brainnet_render_run(repaired.run_dir).run_id == final.run_id
    assert list((project.output_root / "runs").glob(f".{final.run_dir.name}.invalid-*"))


def test_corrupt_resumed_image_is_quarantined_outside_render_tree(
    tmp_path: Path,
) -> None:
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
    fail_second = True

    def runner(config: Any, **kwargs: Any) -> BrainNetExecution:
        nonlocal fail_second
        if kwargs["workspace"] == "ws-two" and fail_second:
            fail_second = False
            raise BrainNetError("seed second workspace failure")
        return _service_runner(config, **kwargs)

    with pytest.raises(BrainNetError, match="seed second workspace"):
        render_brainnet_artifacts(
            project,
            prepared,
            preflight_function=_service_preflight,
            image_runner=runner,
        )
    runs_root = project.output_root / "runs"
    partial = next(runs_root.glob(".brainnet-render-*.partial"))
    prior_image = partial / "brainnet" / "renders" / "ws-one"
    (prior_image / "unexpected.txt").write_text("corrupt", encoding="utf-8")

    rendered = render_brainnet_artifacts(
        project,
        prepared,
        preflight_function=_service_preflight,
        image_runner=runner,
    )

    quarantined = tuple(
        runs_root.glob(
            f".brainnet-render-{rendered.run_id}-ws-one-image.invalid-*"
        )
    )
    assert len(quarantined) == 1
    assert quarantined[0].parent == runs_root
    assert (quarantined[0] / "unexpected.txt").read_text(encoding="utf-8") == "corrupt"
    assert {
        child.name
        for child in (rendered.run_dir / "brainnet" / "renders").iterdir()
    } == {"ws-one", "ws-two"}
    assert verify_brainnet_render_run(rendered.run_dir).run_id == rendered.run_id


def test_render_provenance_paths_are_portable_and_historical_paths_are_typed(
    tmp_path: Path,
) -> None:
    _, _, rendered = _rendered_fixture(tmp_path)
    provenance_paths = [
        rendered.run_dir / "brainnet" / "runtime" / "preflight-provenance.json",
        rendered.run_dir / "brainnet" / "renders" / "ws-one" / "render-provenance.json",
    ]
    for path in provenance_paths:
        provenance = json.loads(path.read_text(encoding="utf-8"))
        assert provenance["path_semantics"]["historical"].startswith(
            "Exact execution-time"
        )
        assert provenance["path_semantics"]["current"].startswith(
            "Canonical run-root-relative"
        )
        for relative in provenance["current_paths"].values():
            assert not Path(relative).is_absolute()
            assert "\\" not in relative
            current = (rendered.run_dir / relative).resolve()
            current.relative_to(rendered.run_dir)
            assert current.is_file()
        assert Path(provenance["historical_execution_root"]).is_absolute()
        assert not Path(provenance["historical_execution_root"]).exists()
    image_provenance = json.loads(provenance_paths[1].read_text(encoding="utf-8"))
    assert Path(image_provenance["historical_run_root"]).is_absolute()
    assert not Path(image_provenance["historical_run_root"]).exists()


def test_render_semantic_verifier_rejects_coherently_rehashed_carriers(
    tmp_path: Path,
) -> None:
    _, _, rendered = _rendered_fixture(tmp_path)
    mutations = (
        "request",
        "execution",
        "completion",
        "qa",
        "provenance",
        "preflight",
        "preflight_whitebg",
        "index",
        "status",
        "validation",
        "inner_file_set",
        "manifest_source",
    )
    for mutation in mutations:
        forged = tmp_path / f"forged-render-{mutation}"
        shutil.copytree(rendered.run_dir, forged)
        image_dir = forged / "brainnet" / "renders" / "ws-one"
        if mutation == "request":
            path = image_dir / "request.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["width_pixels"] += 1
            atomic_json(path, value)
            provenance_path = image_dir / "render-provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["historical_carriers"]["request"]["sha256"] = sha256_file(path)
            atomic_json(provenance_path, provenance)
        elif mutation == "execution":
            path = image_dir / "execution.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["axes_objects"] = 0
            atomic_json(path, value)
            provenance_path = image_dir / "render-provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["historical_carriers"]["execution"]["sha256"] = sha256_file(path)
            atomic_json(provenance_path, provenance)
        elif mutation == "completion":
            path = image_dir / "completion.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["source_node_sha256"] = "0" * 64
            atomic_json(path, value)
        elif mutation == "qa":
            path = image_dir / "image-qa.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["bytes"] += 1
            atomic_json(path, value)
            index_path = forged / "brainnet" / "image-index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index[0]["bytes"] = value["bytes"]
            atomic_json(index_path, index)
        elif mutation == "provenance":
            path = image_dir / "render-provenance.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["current_paths"]["image"] = str((image_dir / "ws-one.png").resolve())
            atomic_json(path, value)
        elif mutation == "preflight":
            path = (
                forged
                / "brainnet"
                / "runtime"
                / "preflight-historical"
                / "preflight.json"
            )
            value = json.loads(path.read_text(encoding="utf-8"))
            value["matlab_release"] = "forged"
            atomic_json(path, value)
            provenance_path = (
                forged / "brainnet" / "runtime" / "preflight-provenance.json"
            )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["historical_carriers"]["preflight.json"]["sha256"] = sha256_file(
                path
            )
            atomic_json(provenance_path, provenance)
        elif mutation == "preflight_whitebg":
            path = (
                forged
                / "brainnet"
                / "runtime"
                / "preflight-historical"
                / "compat"
                / "whitebg.m"
            )
            path.write_bytes(path.read_bytes() + b"% forged\n")
            provenance_path = (
                forged / "brainnet" / "runtime" / "preflight-provenance.json"
            )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["historical_carriers"]["compat/whitebg.m"]["sha256"] = (
                sha256_file(path)
            )
            atomic_json(provenance_path, provenance)
        elif mutation == "index":
            path = forged / "brainnet" / "image-index.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value[0]["bytes"] += 1
            atomic_json(path, value)
        elif mutation == "status":
            path = forged / "brainnet" / "status.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["completed"] = 0
            atomic_json(path, value)
        elif mutation == "validation":
            path = forged / "brainnet" / "validation.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["images"] = 0
            atomic_json(path, value)
        elif mutation == "inner_file_set":
            (image_dir / "forged.txt").write_text("forged", encoding="utf-8")
        elif mutation == "manifest_source":
            path = forged / "brainnet" / "manifest.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["artifact_run_id"] = "forged"
            atomic_json(path, value)
        _rehash_render_manifests(forged)
        with pytest.raises(BrainNetError):
            verify_brainnet_render_run(forged)


def test_render_verifier_rejects_extra_renders_directory_child(
    tmp_path: Path,
) -> None:
    _, _, rendered = _rendered_fixture(tmp_path)
    forged = tmp_path / "extra-render-child"
    shutil.copytree(rendered.run_dir, forged)
    hidden = forged / "brainnet" / "renders" / ".ws-one.invalid-forged"
    hidden.mkdir()
    (hidden / "marker.txt").write_text("forged", encoding="utf-8")
    _rehash_artifact_manifest(forged)
    with pytest.raises(BrainNetError, match="renders directory entries differ"):
        verify_brainnet_render_run(forged)


@pytest.mark.parametrize(
    "relative",
    (
        "brainnet/source-run/brainnet/resources/surface.nv",
        "brainnet/runtime/source-evidence/BrainNet_MapCfg.m",
    ),
)
def test_render_verifier_rejects_same_byte_external_nested_links(
    tmp_path: Path, relative: str
) -> None:
    _, _, rendered = _rendered_fixture(tmp_path)
    forged = tmp_path / "linked-render"
    shutil.copytree(rendered.run_dir, forged)
    _replace_with_external_file_symlink(
        forged, relative, tmp_path / "external-render-files"
    )
    with pytest.raises(BrainNetError, match="symlink, junction, or reparse"):
        verify_brainnet_render_run(forged)


def test_render_verifier_rejects_forged_reserved_workspace(
    tmp_path: Path,
) -> None:
    _, _, rendered = _rendered_fixture(tmp_path)
    forged = tmp_path / "unsafe-workspace-render"
    shutil.copytree(rendered.run_dir, forged)
    _forge_workspace_lock(forged, "CON")
    with pytest.raises(BrainNetError, match="single-component artifact slugs"):
        verify_brainnet_render_run(forged)


def test_render_verifier_rejects_forged_casefold_workspace_collision(
    tmp_path: Path,
) -> None:
    _, _, rendered = _rendered_fixture(tmp_path)
    forged = tmp_path / "casefold-workspace-render"
    shutil.copytree(rendered.run_dir, forged)
    _forge_workspace_lock(forged, ["ws-one", "WS-ONE"])
    with pytest.raises(BrainNetError, match="case-insensitively unique"):
        verify_brainnet_render_run(forged)


def test_artifact_and_render_finals_remain_valid_after_copy_and_move(
    tmp_path: Path,
) -> None:
    project, source, rendered = _rendered_fixture(tmp_path)
    prepared = build_brainnet_artifacts(project, source)
    relocated_root = tmp_path / "relocated"
    relocated_root.mkdir()
    artifact_copy = relocated_root / "artifact-copy"
    render_copy = relocated_root / "render-copy"
    shutil.copytree(prepared.run_dir, artifact_copy)
    shutil.copytree(rendered.run_dir, render_copy)
    assert verify_brainnet_artifact_run(artifact_copy).run_id == prepared.run_id
    assert verify_brainnet_render_run(render_copy).run_id == rendered.run_id
    moved_artifact = relocated_root / "artifact-moved"
    moved_render = relocated_root / "render-moved"
    shutil.move(artifact_copy, moved_artifact)
    shutil.move(render_copy, moved_render)
    assert verify_brainnet_artifact_run(moved_artifact).run_id == prepared.run_id
    assert verify_brainnet_render_run(moved_render).run_id == rendered.run_id


@pytest.mark.parametrize(
    "link_kind", ("file_symlink", "directory_symlink", "junction")
)
def test_linked_partial_root_is_quarantined_and_reinitialized(
    tmp_path: Path, link_kind: str
) -> None:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    prepared = build_brainnet_artifacts(project, source)

    def fail_runner(*_: Any, **__: Any) -> BrainNetExecution:
        raise BrainNetError("seed partial failure")

    with pytest.raises(BrainNetError, match="seed partial"):
        render_brainnet_artifacts(
            project,
            prepared,
            preflight_function=_service_preflight,
            image_runner=fail_runner,
        )
    runs_root = project.output_root / "runs"
    partial = next(runs_root.glob(".brainnet-render-*.partial"))
    shutil.rmtree(partial)
    if link_kind == "file_symlink":
        target = tmp_path / "partial-target.txt"
        target.write_text("keep", encoding="utf-8")
        marker = target
    else:
        target = tmp_path / f"partial-target-{link_kind}"
        target.mkdir()
        marker = target / "marker.txt"
        marker.write_text("keep", encoding="utf-8")
    _create_test_link(partial, target, link_kind)

    rendered = render_brainnet_artifacts(
        project,
        prepared,
        preflight_function=_service_preflight,
        image_runner=_service_runner,
    )

    quarantined = tuple(
        runs_root.glob(f".brainnet-render-{rendered.run_id}-partial.invalid-*")
    )
    try:
        assert verify_brainnet_render_run(rendered.run_dir).run_id == rendered.run_id
        assert marker.read_text(encoding="utf-8") == "keep"
        assert len(quarantined) == 1
        assert os.path.lexists(quarantined[0])
        assert {
            child.name
            for child in (rendered.run_dir / "brainnet" / "renders").iterdir()
        } == {"ws-one"}
    finally:
        for linked in quarantined:
            _remove_test_link(linked)


def test_image_post_publish_verification_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    prepared = build_brainnet_artifacts(project, source)
    original = render_service._verify_image_directory
    injected = False

    def fail_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal injected
        directory = Path(args[1])
        if (
            kwargs.get("published") is True
            and directory.name == "ws-one"
            and not injected
        ):
            injected = True
            raise BrainNetError("injected image post-publish failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(render_service, "_verify_image_directory", fail_once)
    with pytest.raises(BrainNetError, match="image post-publish"):
        render_brainnet_artifacts(
            project,
            prepared,
            preflight_function=_service_preflight,
            image_runner=_service_runner,
        )
    runs = project.output_root / "runs"
    assert injected
    assert not list(runs.glob("brainnet-render-*"))
    partial = next(runs.glob(".brainnet-render-*.partial"))
    assert not (partial / "brainnet" / "renders" / "ws-one").exists()
    monkeypatch.setattr(render_service, "_verify_image_directory", original)
    run = render_brainnet_artifacts(
        project,
        prepared,
        preflight_function=_service_preflight,
        image_runner=_service_runner,
    )
    assert verify_brainnet_render_run(run.run_dir).run_id == run.run_id


def test_outer_post_publish_verification_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    source = _graphvar_run(tmp_path, project)
    prepared = build_brainnet_artifacts(project, source)
    original = render_service.verify_brainnet_render_run
    injected = False

    def fail_once(
        run_dir: str | Path, expected_lock: dict[str, Any] | None = None
    ) -> Any:
        nonlocal injected
        path = Path(run_dir)
        if path.name.startswith("brainnet-render-") and not injected:
            injected = True
            raise BrainNetError("injected outer post-publish failure")
        return original(run_dir, expected_lock)

    monkeypatch.setattr(render_service, "verify_brainnet_render_run", fail_once)
    with pytest.raises(BrainNetError, match="outer post-publish"):
        render_brainnet_artifacts(
            project,
            prepared,
            preflight_function=_service_preflight,
            image_runner=_service_runner,
        )
    runs = project.output_root / "runs"
    assert injected
    assert not list(runs.glob("brainnet-render-*"))
    assert list(runs.glob(".brainnet-render-*.partial"))
    monkeypatch.setattr(render_service, "verify_brainnet_render_run", original)
    run = render_brainnet_artifacts(
        project,
        prepared,
        preflight_function=_service_preflight,
        image_runner=_service_runner,
    )
    assert verify_brainnet_render_run(run.run_dir).run_id == run.run_id


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
