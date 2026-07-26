from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from neurostat.contracts import RecordKey
from neurostat.graphvar.config import GraphVarConfigError, parse_graphvar_config
from neurostat.graphvar.extract import (
    apply_condition_holm,
    connected_components,
    conservative_upper_tail,
    extract_workspace_result,
    holm_adjust,
)
from neurostat.graphvar.workspace import (
    GraphVarError,
    _family_cohort,
    _verify_hash_set,
    validate_atlas_workbook,
)
from neurostat.project import CleanProject, ProjectRecord, Study
from neurostat.stage_io import sha256_file


def _xlsx(
    path: Path, selected: list[tuple[int, str, tuple[float, float, float]]]
) -> None:
    import zipfile

    rows = []
    selected_by_row = {index: (label, coords) for index, label, coords in selected}
    for index in range(1, max(selected_by_row) + 1):
        label, coords = selected_by_row.get(index, (f"unused-{index}", (0.0, 0.0, 0.0)))
        flag = 1 if index in selected_by_row else 0
        cells = (
            f'<c r="A{index}"><v>{flag}</v></c>'
            f'<c r="B{index}" t="inlineStr"><is><t>{label}</t></is></c>'
            f'<c r="C{index}"><v>999</v></c>'
            f'<c r="D{index}"><v>{coords[0]}</v></c>'
            f'<c r="E{index}"><v>{coords[1]}</v></c>'
            f'<c r="F{index}"><v>{coords[2]}</v></c>'
        )
        rows.append(f'<row r="{index}">{cells}</row>')
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


def _project(tmp_path: Path, graphvar: MappingProxyType | None = None) -> CleanProject:
    source = tmp_path / "input.fif"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"x")
    record = ProjectRecord(
        "r1",
        RecordKey("p1", "active", "pre", "sep"),
        "epochs",
        "fif",
        source,
        "default",
    )
    extras = {} if graphvar is None else {"graphvar": graphvar}
    return CleanProject(
        tmp_path / "project.yaml",
        Study("study", ("active", "control"), ("pre", "post"), ("sep",)),
        (record,),
        MappingProxyType({"default": MappingProxyType({"opaque": True})}),
        tmp_path / "outputs",
        MappingProxyType(extras),
    )


def _graphvar_section(tmp_path: Path) -> MappingProxyType:
    workbook = tmp_path / "atlas.xlsx"
    _xlsx(workbook, [(1, "Left", (1.0, 2.0, 3.0)), (2, "Right", (4.0, 5.0, 6.0))])
    viewer = tmp_path / "viewer.mat"
    viewer.write_bytes(b"viewer")
    root = tmp_path / "GraphVar"
    (root / "src" / "calc").mkdir(parents=True)
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
                                "display_label": "Left",
                                "coordinates": (1.0, 2.0, 3.0),
                            }
                        ),
                        MappingProxyType(
                            {
                                "connectivity_label": "right-dk",
                                "atlas_index_1based": 2,
                                "display_label": "Right",
                                "coordinates": (4.0, 5.0, 6.0),
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
                        "component_forming_alpha": 0.05,
                        "familywise_alpha": 0.05,
                        "permutations": 5000,
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
                    "executable": "matlab",
                    "graphvar_root": "GraphVar",
                    "matlab_release": "2025b",
                    "timeout_seconds": 600,
                }
            ),
        }
    )


def test_config_is_strict_and_has_explicit_holm_and_node_mapping(
    tmp_path: Path,
) -> None:
    section = _graphvar_section(tmp_path)
    config = parse_graphvar_config(_project(tmp_path, section))
    assert config.families[0].holm_family == "sep-primary"
    assert config.atlas.nodes[0].connectivity_label == "left-dk"
    assert config.atlas.nodes[0].display_label == "Left"
    assert config.atlas.nodes[0].coordinates == (1.0, 2.0, 3.0)
    assert validate_atlas_workbook(config)[0]["coordinates"] == [1.0, 2.0, 3.0]

    bad = dict(section)
    bad["unknown"] = True
    with pytest.raises(GraphVarConfigError, match="unknown"):
        parse_graphvar_config(_project(tmp_path / "unknown", MappingProxyType(bad)))

    family = dict(section["families"][0])
    family["effect_label"] = "ambiguous"
    bad_effect = dict(section)
    bad_effect["families"] = (MappingProxyType(family),)
    with pytest.raises(GraphVarConfigError, match="exact reverse"):
        parse_graphvar_config(_project(tmp_path, MappingProxyType(bad_effect)))
    reverse_family = dict(section["families"][0])
    reverse_family["effect_label"] = "arm_control-arm_active=0"
    reverse = dict(section)
    reverse["families"] = (MappingProxyType(reverse_family),)
    assert (
        parse_graphvar_config(_project(tmp_path, MappingProxyType(reverse)))
        .families[0]
        .effect_label
        == "arm_control-arm_active=0"
    )


def _hdf5_result(path: Path) -> None:
    import h5py

    effects, nodes, permutations = 1, 4, 4
    b = np.zeros((effects, nodes, nodes))
    f = np.zeros_like(b)
    p = np.zeros_like(b)
    np_value = np.ones_like(b)
    np.fill_diagonal(np_value[0], 0.0)
    for first, second, effect_value in ((0, 1, 2.0), (1, 2, -3.0)):
        b[0, first, second] = b[0, second, first] = effect_value
        f[0, first, second] = f[0, second, first] = effect_value**2
        p[0, first, second] = p[0, second, first] = 0.02
        np_value[0, first, second] = np_value[0, second, first] = 0.05
    nf = np.zeros((permutations, effects, nodes, nodes))
    nb = np.zeros_like(nf)
    for first in range(nodes - 1):
        for second in range(first + 1, nodes):
            values = np.asarray([4.0, 3.0, 2.0, 1.0])
            nf[:, 0, first, second] = nf[:, 0, second, first] = values
    with h5py.File(path, "w") as file:
        for name, value in (
            ("B", b),
            ("F", f),
            ("P", p),
            ("NP", np_value),
            ("NF", nf),
            ("NB", nb),
        ):
            file.create_dataset(name, data=value)
        label = "arm_control-arm_active=0"
        chars = file.create_dataset(
            "label",
            data=np.asarray([ord(char) for char in label], dtype=np.uint16).reshape(
                -1, 1
            ),
        )
        chars.attrs["MATLAB_class"] = np.bytes_("char")
        references = np.empty((effects, 1), dtype=h5py.ref_dtype)
        references[0, 0] = chars.ref
        file.create_dataset("LAB", data=references)
        file.create_dataset("J", data=np.asarray([[1.0]]))
        file.create_dataset("DDF", data=np.asarray([[7.0]]))
        file.create_dataset("N", data=np.asarray([[8.0]]))


def test_hdf5_axes_refs_orientation_components_null_and_plus_one(
    tmp_path: Path,
) -> None:
    result = tmp_path / "result.mat"
    _hdf5_result(result)
    request = {
        "workspace_id": "sep__theta__pre-post",
        "holm_family": "sep",
        "family": "primary",
        "condition": "sep",
        "band": "theta",
        "pair": "pre-post",
        "numerator_arm": "active",
        "denominator_arm": "control",
        "effect_label": "arm_control-arm_active=0",
        "component_forming_alpha": 0.25,
        "familywise_alpha": 0.5,
        "permutations": 4,
        "observations": 8,
    }
    node_map = tuple(
        {
            "atlas_index_1based": index + 1,
            "connectivity_label": f"dk-{index + 1}",
            "display_label": f"ROI-{index + 1}",
        }
        for index in range(4)
    )
    extracted = extract_workspace_result(result, request, node_map)
    assert extracted.status["candidate_components"] == 1
    component = extracted.components[0]
    assert component["component_node_count"] == 3
    assert component["component_link_count"] == 2
    assert component["primary_node_pfwe_graphvar_tail"] == 0.25
    assert component["primary_node_pfwe_plus_one"] == 0.4
    assert component["posthoc_nbs_edge_pfwe_plus_one"] == 0.4
    assert extracted.links[0]["paired_change_contrast_b"] == -2.0
    assert component["effect_orientation_multiplier"] == -1
    assert extracted.null_node_max.tolist() == [4, 1, 1, 1]
    assert extracted.null_edge_max.tolist() == [6, 0, 0, 0]


def test_conservative_ties_components_and_all_candidate_separate_holm() -> None:
    np.testing.assert_allclose(
        conservative_upper_tail(np.asarray([3.0, 3.0, 2.0, 1.0])),
        [0.5, 0.5, 0.75, 1.0],
    )
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 1] = mask[1, 0] = True
    mask[1, 2] = mask[2, 1] = True
    mask[3, 4] = mask[4, 3] = True
    assert connected_components(mask) == ((0, 1, 2), (3, 4))
    np.testing.assert_allclose(holm_adjust([0.002, 0.1, 0.04]), [0.006, 0.1, 0.08])
    rows = [
        {
            "holm_family": "sep",
            "primary_node_pfwe_plus_one": 0.002,
            "primary_node_pfwe_supported": True,
            "posthoc_nbs_edge_pfwe_plus_one": 0.02,
            "posthoc_nbs_edge_pfwe_supported": True,
        },
        {
            "holm_family": "sep",
            "primary_node_pfwe_plus_one": 0.04,
            "primary_node_pfwe_supported": True,
            "posthoc_nbs_edge_pfwe_plus_one": 0.001,
            "posthoc_nbs_edge_pfwe_supported": True,
        },
    ]
    apply_condition_holm(rows, {"sep": 0.05})
    assert [row["primary_node_holm_all_candidates"] for row in rows] == [0.004, 0.04]
    assert [row["posthoc_nbs_edge_holm_all_candidates"] for row in rows] == [
        0.02,
        0.002,
    ]


def test_common_cohort_audits_wholly_missing_condition_and_manifest_nested_files(
    tmp_path: Path,
) -> None:
    paths = []
    rows = []
    for index, (subject, arm, session, condition) in enumerate(
        (
            ("p1", "active", "pre", "sep"),
            ("p1", "active", "post", "sep"),
            ("p2", "control", "pre", "sep"),
            ("p2", "control", "post", "sep"),
            ("p3", "active", "pre", "rest"),
        ),
        1,
    ):
        path = tmp_path / f"r{index}.fif"
        path.write_bytes(b"x")
        paths.append(path)
        rows.append(
            ProjectRecord(
                f"r{index}",
                RecordKey(subject, arm, session, condition),
                "epochs",
                "fif",
                path,
                "default",
            )
        )
    project = CleanProject(
        tmp_path / "project.yaml",
        Study("study", ("active", "control"), ("pre", "post"), ("sep", "rest")),
        tuple(rows),
        MappingProxyType({"default": MappingProxyType({})}),
        tmp_path / "outputs",
        MappingProxyType({}),
    )
    from neurostat.graphvar.config import GraphVarFamily, SessionPair

    family = GraphVarFamily(
        "primary",
        "sep",
        "sep",
        ("theta",),
        (SessionPair("pre-post", "pre", "post"),),
        "active",
        "control",
        0.05,
        0.05,
        4,
        "sha256_workspace",
        1,
        "OutPLI_{band}",
        "arm_active-arm_control=0",
    )
    included, _, audit = _family_cohort(project, family)
    assert included == ("p1", "p2")
    missing = next(row for row in audit if row["subject"] == "p3")
    assert missing["included"] == "false"
    assert missing["missing_sessions"] == "pre;post"

    root = tmp_path / "hashes"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "manifest.json").write_text("nested", encoding="utf-8")
    expected = {"nested/manifest.json": sha256_file(root / "nested" / "manifest.json")}
    _verify_hash_set(root, expected, set(), "fixture")
    (root / "nested" / "manifest.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(GraphVarError, match="SHA-256"):
        _verify_hash_set(root, expected, set(), "fixture")


@pytest.mark.skipif(
    not os.environ.get("NEUROSTAT_TEST_V7_GRAPHVAR_RESULTS")
    or not os.environ.get("NEUROSTAT_TEST_V7_ACCEPTED_OUTPUTS"),
    reason="set V7 GraphVar result/output roots for accepted local parity",
)
def test_accepted_v7_all_candidate_parity() -> None:
    import csv

    result_root = Path(os.environ["NEUROSTAT_TEST_V7_GRAPHVAR_RESULTS"])
    accepted_root = Path(os.environ["NEUROSTAT_TEST_V7_ACCEPTED_OUTPUTS"])
    with (accepted_root / "v7_graphvar_component_member_links_all.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        accepted_links = list(csv.DictReader(handle))
    mapping_by_index: dict[int, dict[str, object]] = {}
    for row in accepted_links:
        for side in ("1", "2"):
            result_index = int(row[f"ROI{side}_ResultIndex"])
            mapping_by_index[result_index] = {
                "atlas_index_1based": int(row[f"ROI{side}_AtlasIndex1Based"]),
                "connectivity_label": row[f"ROI{side}"],
                "display_label": row[f"ROI{side}"],
            }
    node_map = tuple(mapping_by_index[index] for index in sorted(mapping_by_index))
    with (accepted_root / "v7_graphvar_components_all.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        accepted_components = list(csv.DictReader(handle))
    extracted_rows: list[dict[str, object]] = []
    extracted_links: list[dict[str, object]] = []
    for result in sorted(result_root.rglob("corr_area_1_Group.mat")):
        workspace = result.parents[2].name
        parts = workspace.split("_")
        band = parts[1]
        pair = "_".join(parts[2:])
        accepted_workspace = [
            row for row in accepted_components if row["Workspace"] == workspace
        ]
        request = {
            "workspace_id": workspace,
            "holm_family": "SEP",
            "family": "SEP",
            "condition": "SEP",
            "band": band,
            "pair": pair,
            "numerator_arm": "Chiro",
            "denominator_arm": "Control",
            "effect_label": "Group_Chiro-Group_Control=0",
            "effect_orientation_multiplier": 1,
            "component_forming_alpha": 0.05,
            "familywise_alpha": 0.05,
            "permutations": 5000,
            "observations": 122,
        }
        extracted = extract_workspace_result(result, request, node_map)
        assert len(extracted.components) == len(accepted_workspace)
        extracted_rows.extend(dict(row) for row in extracted.components)
        extracted_links.extend(dict(row) for row in extracted.links)
    apply_condition_holm(extracted_rows, {"SEP": 0.05})
    assert len(extracted_rows) == 24
    assert sum(bool(row["primary_node_pfwe_supported"]) for row in extracted_rows) == 4
    assert sum(bool(row["primary_studywide_supported"]) for row in extracted_rows) == 1
    assert (
        sum(bool(row["posthoc_nbs_edge_pfwe_supported"]) for row in extracted_rows) == 4
    )
    assert (
        sum(bool(row["posthoc_nbs_studywide_supported"]) for row in extracted_rows) == 4
    )
    accepted_by_id = {row["ComponentID"]: row for row in accepted_components}
    maxima = {
        "node": 0.0,
        "nbs": 0.0,
        "holm": 0.0,
        "b": 0.0,
        "f": 0.0,
        "p": 0.0,
        "np": 0.0,
    }
    for row in extracted_rows:
        accepted = accepted_by_id[row["component_id"]]
        assert row["component_node_count"] == int(accepted["ComponentNodeCount"])
        assert row["component_link_count"] == int(accepted["ComponentLinkCount"])
        maxima["node"] = max(
            maxima["node"],
            abs(
                float(row["primary_node_pfwe_plus_one"])
                - float(accepted["ComponentP_FWE_PlusOne"])
            ),
        )
        maxima["nbs"] = max(
            maxima["nbs"],
            abs(
                float(row["posthoc_nbs_edge_pfwe_plus_one"])
                - float(accepted["NBS_Extent_P_FWE_PlusOne"])
            ),
        )
        maxima["holm"] = max(
            maxima["holm"],
            abs(
                float(row["primary_node_holm_all_candidates"])
                - float(accepted["GlobalHolmP"])
            ),
        )
    accepted_links_by_key = {
        (
            row["ComponentID"],
            int(row["ROI1_ResultIndex"]),
            int(row["ROI2_ResultIndex"]),
        ): row
        for row in accepted_links
    }
    for row in extracted_links:
        accepted = accepted_links_by_key[
            (
                row["component_id"],
                row["roi1_result_index_1based"],
                row["roi2_result_index_1based"],
            )
        ]
        for key, ours, theirs in (
            ("b", "paired_change_contrast_b", "PairedChangeContrast_B"),
            ("f", "f_statistic", "FStatistic"),
            ("p", "parametric_p_uncorrected", "ParametricP_Uncorrected"),
            ("np", "permutation_p_uncorrected", "PermutationP_Uncorrected"),
        ):
            maxima[key] = max(
                maxima[key], abs(float(row[ours]) - float(accepted[theirs]))
            )
    assert max(maxima.values()) < 1e-12
    theta = next(
        row for row in extracted_rows if row["workspace"] == "SEP_Theta_Post_Post4W"
    )
    assert theta["primary_node_pfwe_plus_one"] == pytest.approx(0.001999600079984)
    assert theta["primary_node_holm_all_candidates"] == pytest.approx(
        0.0479904019196161
    )
