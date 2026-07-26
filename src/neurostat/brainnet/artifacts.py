"""Validated BrainNet node/edge artifacts from a finalized GraphVar run."""

from __future__ import annotations

import csv
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..graphvar import GraphVarAnalysisRun, verify_graphvar_analysis_run
from ..graphvar.service import COMPONENT_FIELDS, LINK_FIELDS
from ..graphvar.workspace import _all_files, _verify_hash_set
from ..project import CleanProject, load_project
from ..stage_io import atomic_json, canonical_json, digest_json, load_json, sha256_file
from .config import BrainNetConfig, parse_brainnet_config

SCHEMA = "neurostat.brainnet_artifacts.v1"
TOP_LINK_FIELDS = (
    "workspace",
    "component_id",
    "display_rank",
    "roi1_result_index_1based",
    "roi2_result_index_1based",
    "roi1_connectivity_label",
    "roi2_connectivity_label",
    "roi1",
    "roi2",
    "signed_weight",
    "permutation_p_uncorrected",
    "paired_change_contrast_b",
    "abs_b",
    "f_statistic",
    "component_node_count",
    "component_link_count",
    "primary_node_pfwe_plus_one",
    "primary_node_holm_all_candidates",
    "posthoc_nbs_edge_pfwe_plus_one",
    "posthoc_nbs_edge_holm_all_candidates",
    "inference_scope",
)
DESCRIPTIVE_SCOPE = (
    "Displayed member link is descriptive and uncorrected; corrected inference "
    "applies only to the complete component. NBS is a separate post-hoc sensitivity."
)


class BrainNetError(RuntimeError):
    """Raised when a BrainNet source, artifact, runtime, or image fails closed."""


@dataclass(frozen=True, slots=True)
class BrainNetArtifactRun:
    run_id: str
    run_dir: Path
    manifest: Path
    reused: bool
    workspaces: tuple[str, ...]
    artifacts: tuple[Path, ...]


def _read_csv(
    path: Path, expected: tuple[str, ...], description: str
) -> list[dict[str, str]]:
    if not path.is_file():
        raise BrainNetError(f"{description} is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected):
            raise BrainNetError(
                f"{description} schema differs; expected {list(expected)!r}, "
                f"got {reader.fieldnames!r}"
            )
        rows = list(reader)
    if any(
        set(row) != set(expected) or any(value is None for value in row.values())
        for row in rows
    ):
        raise BrainNetError(f"{description} contains malformed rows")
    return rows


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BrainNetError(f"{name} must be nonblank")
    return value.strip()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise BrainNetError(f"{name} must be an integer >= {minimum}")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise BrainNetError(f"{name} must be an integer >= {minimum}") from exc
    if str(number) != str(value).strip() or number < minimum:
        raise BrainNetError(f"{name} must be an integer >= {minimum}")
    return number


def _float(value: Any, name: str, *, probability: bool = False) -> float:
    if isinstance(value, bool):
        raise BrainNetError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BrainNetError(f"{name} must be finite") from exc
    if not math.isfinite(result) or (probability and not 0 <= result <= 1):
        qualifier = "a probability" if probability else "finite"
        raise BrainNetError(f"{name} must be {qualifier}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise BrainNetError(f"{name} must be exactly True or False")


def _node_map(path: Path) -> tuple[dict[str, Any], ...]:
    value = load_json(path, "GraphVar selected node map")
    if not isinstance(value, list) or len(value) < 2:
        raise BrainNetError(
            "GraphVar selected node map must contain at least two nodes"
        )
    expected = {
        "result_index_1based",
        "atlas_index_1based",
        "connectivity_label",
        "display_label",
        "coordinates",
    }
    nodes = []
    for position, item in enumerate(value, 1):
        if not isinstance(item, dict) or set(item) != expected:
            raise BrainNetError(f"selected node {position} has an invalid schema")
        item = cast(dict[str, Any], item)
        result_index = _integer(item["result_index_1based"], "result index", 1)
        atlas_index = _integer(item["atlas_index_1based"], "atlas index", 1)
        if result_index != position:
            raise BrainNetError(
                "selected node result indices must be contiguous and ordered"
            )
        coordinates = item["coordinates"]
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 3
            or any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                for number in coordinates
            )
        ):
            raise BrainNetError(
                "every BrainNet node needs three exact finite atlas coordinates"
            )
        nodes.append(
            {
                "result_index_1based": result_index,
                "atlas_index_1based": atlas_index,
                "connectivity_label": _text(
                    item["connectivity_label"], "node connectivity label"
                ),
                "display_label": _text(item["display_label"], "node display label"),
                "coordinates": [float(number) for number in coordinates],
            }
        )
    for key in ("atlas_index_1based", "connectivity_label", "display_label"):
        if len({node[key] for node in nodes}) != len(nodes):
            raise BrainNetError(f"selected node {key} values must be unique")
    return tuple(nodes)


def _alpha_by_family(graphvar_lock: dict[str, Any]) -> dict[tuple[str, str], float]:
    try:
        families = graphvar_lock["config"]["families"]
    except (KeyError, TypeError) as exc:
        raise BrainNetError("GraphVar lock lacks normalized family config") from exc
    if not isinstance(families, list) or not families:
        raise BrainNetError("GraphVar lock family config is invalid")
    result: dict[tuple[str, str], float] = {}
    for family in families:
        if not isinstance(family, dict):
            raise BrainNetError("GraphVar lock family entry is invalid")
        key = (
            _text(family.get("name"), "family name"),
            _text(family.get("holm_family"), "Holm family"),
        )
        alpha = _float(
            family.get("familywise_alpha"), "familywise alpha", probability=True
        )
        previous = result.setdefault(key, alpha)
        if previous != alpha:
            raise BrainNetError("GraphVar family alpha is ambiguous")
    return result


def load_verified_graphvar_tables(
    graphvar: GraphVarAnalysisRun | str | Path,
) -> tuple[
    GraphVarAnalysisRun,
    tuple[dict[str, Any], ...],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Verify the whole source run, then strictly type and cross-check its tables."""
    source = (
        verify_graphvar_analysis_run(graphvar.run_dir)
        if isinstance(graphvar, GraphVarAnalysisRun)
        else verify_graphvar_analysis_run(graphvar)
    )
    graphvar_dir = source.run_dir / "graphvar"
    nodes = _node_map(graphvar_dir / "selected-node-map.json")
    node_by_index = {node["result_index_1based"]: node for node in nodes}
    component_raw = _read_csv(
        graphvar_dir / "components-all.csv", COMPONENT_FIELDS, "GraphVar components"
    )
    link_raw = _read_csv(
        graphvar_dir / "component-member-links-all.csv",
        LINK_FIELDS,
        "GraphVar component member links",
    )
    lock = load_json(source.run_dir / "project.lock.json", "GraphVar analysis lock")
    alpha_lookup = _alpha_by_family(lock)
    components: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(component_raw, 1):
        component_id = _text(raw["component_id"], f"component {position} id")
        if component_id in by_id:
            raise BrainNetError("GraphVar component identifiers must be unique")
        family_key = (
            _text(raw["family"], "component family"),
            _text(raw["holm_family"], "component Holm family"),
        )
        if family_key not in alpha_lookup:
            raise BrainNetError("component family is absent from the GraphVar lock")
        alpha = alpha_lookup[family_key]
        row: dict[str, Any] = dict(raw)
        row.update(
            holm_family=family_key[1],
            family=family_key[0],
            condition=_text(raw["condition"], "component condition"),
            component_id=component_id,
            workspace=_text(raw["workspace"], "component workspace"),
            band=_text(raw["band"], "component band"),
            session_pair=_text(raw["session_pair"], "component session pair"),
            effect_label=_text(raw["effect_label"], "component effect label"),
            effect_orientation_multiplier=_integer(
                raw["effect_orientation_multiplier"],
                "component effect orientation",
                -1,
            ),
            component_node_count=_integer(
                raw["component_node_count"], "component node count", 2
            ),
            component_link_count=_integer(
                raw["component_link_count"], "component link count", 1
            ),
            member_rois=_text(raw["member_rois"], "component member ROIs"),
            component_forming_alpha=_float(
                raw["component_forming_alpha"],
                "component-forming alpha",
                probability=True,
            ),
            permutations=_integer(raw["permutations"], "component permutations", 1),
            primary_node_exceedances=_integer(
                raw["primary_node_exceedances"],
                "primary node exceedances",
            ),
            primary_node_pfwe_graphvar_tail=_float(
                raw["primary_node_pfwe_graphvar_tail"],
                "primary GraphVar-tail pFWE",
                probability=True,
            ),
            primary_node_pfwe_plus_one=_float(
                raw["primary_node_pfwe_plus_one"],
                "primary node pFWE",
                probability=True,
            ),
            primary_node_pfwe_supported=_boolean(
                raw["primary_node_pfwe_supported"], "primary pFWE flag"
            ),
            primary_node_holm_all_candidates=_float(
                raw["primary_node_holm_all_candidates"],
                "primary Holm p",
                probability=True,
            ),
            primary_studywide_supported=_boolean(
                raw["primary_studywide_supported"], "primary study-wide flag"
            ),
            posthoc_nbs_edge_pfwe_plus_one=_float(
                raw["posthoc_nbs_edge_pfwe_plus_one"],
                "post-hoc NBS pFWE",
                probability=True,
            ),
            posthoc_nbs_edge_exceedances=_integer(
                raw["posthoc_nbs_edge_exceedances"],
                "post-hoc NBS exceedances",
            ),
            posthoc_nbs_edge_pfwe_graphvar_tail=_float(
                raw["posthoc_nbs_edge_pfwe_graphvar_tail"],
                "post-hoc NBS GraphVar-tail pFWE",
                probability=True,
            ),
            posthoc_nbs_edge_pfwe_supported=_boolean(
                raw["posthoc_nbs_edge_pfwe_supported"], "post-hoc NBS pFWE flag"
            ),
            posthoc_nbs_edge_holm_all_candidates=_float(
                raw["posthoc_nbs_edge_holm_all_candidates"],
                "post-hoc NBS Holm p",
                probability=True,
            ),
            posthoc_nbs_studywide_supported=_boolean(
                raw["posthoc_nbs_studywide_supported"],
                "post-hoc NBS study-wide flag",
            ),
            graphvar_observation_rows=_integer(
                raw["graphvar_observation_rows"],
                "GraphVar observation rows",
                1,
            ),
            graphvar_numerator_df=_float(
                raw["graphvar_numerator_df"], "GraphVar numerator DF"
            ),
            graphvar_denominator_df=_float(
                raw["graphvar_denominator_df"], "GraphVar denominator DF"
            ),
            inference_scope=_text(raw["inference_scope"], "component inference scope"),
        )
        if row["effect_orientation_multiplier"] not in {-1, 1}:
            raise BrainNetError("component effect orientation must be -1 or 1")
        if (
            not 0 < row["component_forming_alpha"] < 1
            or row["graphvar_numerator_df"] <= 0
            or row["graphvar_denominator_df"] <= 0
            or row["primary_node_exceedances"] > row["permutations"]
            or row["posthoc_nbs_edge_exceedances"] > row["permutations"]
        ):
            raise BrainNetError("component numeric metadata is outside its domain")
        if (
            not math.isclose(
                row["primary_node_pfwe_graphvar_tail"],
                row["primary_node_exceedances"] / row["permutations"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                row["primary_node_pfwe_plus_one"],
                (row["primary_node_exceedances"] + 1) / (row["permutations"] + 1),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                row["posthoc_nbs_edge_pfwe_graphvar_tail"],
                row["posthoc_nbs_edge_exceedances"] / row["permutations"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                row["posthoc_nbs_edge_pfwe_plus_one"],
                (row["posthoc_nbs_edge_exceedances"] + 1) / (row["permutations"] + 1),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise BrainNetError("component permutation p-values are inconsistent")
        if row["component_link_count"] > (
            row["component_node_count"] * (row["component_node_count"] - 1) // 2
        ):
            raise BrainNetError("component link count exceeds its possible topology")
        if row["primary_node_pfwe_supported"] != (
            row["primary_node_pfwe_plus_one"] <= alpha
        ) or row["primary_studywide_supported"] != (
            row["primary_node_pfwe_supported"]
            and row["primary_node_holm_all_candidates"] <= alpha
        ):
            raise BrainNetError("primary component inference flags are inconsistent")
        if row["posthoc_nbs_edge_pfwe_supported"] != (
            row["posthoc_nbs_edge_pfwe_plus_one"] <= alpha
        ) or row["posthoc_nbs_studywide_supported"] != (
            row["posthoc_nbs_edge_pfwe_supported"]
            and row["posthoc_nbs_edge_holm_all_candidates"] <= alpha
        ):
            raise BrainNetError("post-hoc NBS component flags are inconsistent")
        by_id[component_id] = row
        components.append(row)

    links: list[dict[str, Any]] = []
    links_by_component: dict[str, list[dict[str, Any]]] = {}
    identities: set[tuple[str, int, int]] = set()
    for position, raw in enumerate(link_raw, 1):
        component_id = _text(raw["component_id"], f"link {position} component")
        component = by_id.get(component_id)
        if component is None:
            raise BrainNetError("member link has no unique component")
        first = _integer(raw["roi1_result_index_1based"], "ROI1 result index", 1)
        second = _integer(raw["roi2_result_index_1based"], "ROI2 result index", 1)
        if first >= second:
            raise BrainNetError("member links must use ordered distinct result indices")
        identity = (component_id, first, second)
        if identity in identities:
            raise BrainNetError("duplicate component member link")
        identities.add(identity)
        first_node, second_node = node_by_index.get(first), node_by_index.get(second)
        if first_node is None or second_node is None:
            raise BrainNetError("member link result index is outside the node map")
        exact = (
            (
                _integer(raw["roi1_atlas_index_1based"], "ROI1 atlas index", 1),
                first_node["atlas_index_1based"],
            ),
            (
                _integer(raw["roi2_atlas_index_1based"], "ROI2 atlas index", 1),
                second_node["atlas_index_1based"],
            ),
            (
                _text(raw["roi1_connectivity_label"], "ROI1 connectivity label"),
                first_node["connectivity_label"],
            ),
            (
                _text(raw["roi2_connectivity_label"], "ROI2 connectivity label"),
                second_node["connectivity_label"],
            ),
            (_text(raw["roi1"], "ROI1 display label"), first_node["display_label"]),
            (_text(raw["roi2"], "ROI2 display label"), second_node["display_label"]),
        )
        if any(observed != expected for observed, expected in exact):
            raise BrainNetError(
                "member link labels/indices differ from selected node map"
            )
        row: dict[str, Any] = dict(raw)
        row.update(
            component_id=component_id,
            workspace=_text(raw["workspace"], "link workspace"),
            roi1_result_index_1based=first,
            roi2_result_index_1based=second,
            paired_change_contrast_b=_float(raw["paired_change_contrast_b"], "link B"),
            f_statistic=_float(raw["f_statistic"], "link F"),
            parametric_p_uncorrected=_float(
                raw["parametric_p_uncorrected"],
                "link parametric p",
                probability=True,
            ),
            permutation_p_uncorrected=_float(
                raw["permutation_p_uncorrected"],
                "link permutation p",
                probability=True,
            ),
        )
        if row["f_statistic"] < 0:
            raise BrainNetError("member link F must be nonnegative")
        if (
            row["workspace"] != component["workspace"]
            or _text(raw["holm_family"], "link Holm family") != component["holm_family"]
            or _text(raw["condition"], "link condition") != component["condition"]
            or _text(raw["band"], "link band") != component["band"]
            or _text(raw["session_pair"], "link session pair")
            != component["session_pair"]
            or _text(raw["effect_label"], "link effect label")
            != component["effect_label"]
            or _integer(raw["component_node_count"], "link component node count", 2)
            != component["component_node_count"]
            or _integer(raw["component_link_count"], "link component link count", 1)
            != component["component_link_count"]
        ):
            raise BrainNetError("member link component metadata is inconsistent")
        if _text(raw["inference_scope"], "link inference scope") != (
            "Member link is descriptive and uncorrected; corrected inference "
            "applies only to the complete component."
        ) or "Primary study-wide support" not in _text(
            component["inference_scope"], "component inference scope"
        ):
            raise BrainNetError("component/link inference scope is inconsistent")
        for name in (
            "primary_node_pfwe_plus_one",
            "primary_node_holm_all_candidates",
            "posthoc_nbs_edge_pfwe_plus_one",
            "posthoc_nbs_edge_holm_all_candidates",
        ):
            if _float(raw[name], f"link {name}", probability=True) != component[name]:
                raise BrainNetError(
                    "member link inference values differ from component"
                )
        for name in (
            "primary_studywide_supported",
            "posthoc_nbs_studywide_supported",
        ):
            if _boolean(raw[name], f"link {name}") != component[name]:
                raise BrainNetError("member link inference flags differ from component")
        links.append(row)
        links_by_component.setdefault(component_id, []).append(row)

    for component in components:
        members = links_by_component.get(component["component_id"], [])
        if len(members) != component["component_link_count"]:
            raise BrainNetError("component member-link cardinality is inconsistent")
        endpoint_labels = {
            str(link[side]) for link in members for side in ("roi1", "roi2")
        }
        declared = {
            part.strip()
            for part in str(component["member_rois"]).split(";")
            if part.strip()
        }
        if (
            len(endpoint_labels) != component["component_node_count"]
            or endpoint_labels != declared
        ):
            raise BrainNetError("component node membership is inconsistent")
    return source, nodes, components, links


def select_components_and_links(
    config: BrainNetConfig,
    components: list[dict[str, Any]],
    links: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select requested primary components and deterministic descriptive links."""
    links_by_component: dict[str, list[dict[str, Any]]] = {}
    for row in links:
        links_by_component.setdefault(row["component_id"], []).append(row)
    selections = []
    top_links = []
    for workspace in config.selection.workspaces:
        candidates = [
            row
            for row in components
            if row["workspace"] == workspace and row["primary_node_pfwe_supported"]
        ]
        candidates.sort(
            key=lambda row: (
                -row["component_node_count"],
                row["primary_node_pfwe_plus_one"],
                row["component_id"],
            )
        )
        if not candidates:
            raise BrainNetError(
                f"requested workspace has no primary node-pFWE-supported component: {workspace}"
            )
        selected = candidates[0]
        member_links = list(links_by_component.get(selected["component_id"], ()))
        member_links.sort(
            key=lambda row: (
                row["permutation_p_uncorrected"],
                -abs(row["paired_change_contrast_b"]),
                row["roi1"],
                row["roi2"],
            )
        )
        if len(member_links) < config.selection.top_k:
            raise BrainNetError(
                f"selected component has fewer than top_k member links: {workspace}"
            )
        selected_links = member_links[: config.selection.top_k]
        selections.append(
            {
                "workspace": workspace,
                "component_id": selected["component_id"],
                "policy": config.selection.policy,
                "component_node_count": selected["component_node_count"],
                "component_link_count": selected["component_link_count"],
                "primary_node_pfwe_plus_one": selected["primary_node_pfwe_plus_one"],
                "primary_node_holm_all_candidates": selected[
                    "primary_node_holm_all_candidates"
                ],
                "primary_node_pfwe_supported": True,
                "primary_studywide_supported": selected["primary_studywide_supported"],
                "posthoc_nbs_edge_pfwe_plus_one": selected[
                    "posthoc_nbs_edge_pfwe_plus_one"
                ],
                "posthoc_nbs_edge_holm_all_candidates": selected[
                    "posthoc_nbs_edge_holm_all_candidates"
                ],
                "selection_basis": (
                    "largest primary node-pFWE-supported component; NBS does not "
                    "qualify selection and global Holm is not required for display"
                ),
            }
        )
        for rank, link in enumerate(selected_links, 1):
            b_value = link["paired_change_contrast_b"]
            f_value = link["f_statistic"]
            weight = (1.0 if b_value > 0 else -1.0 if b_value < 0 else 0.0) * math.sqrt(
                f_value
            )
            if weight == 0:
                raise BrainNetError(
                    f"selected descriptive link has zero signed weight: "
                    f"{workspace} rank {rank}"
                )
            top_links.append(
                {
                    "workspace": workspace,
                    "component_id": selected["component_id"],
                    "display_rank": rank,
                    "roi1_result_index_1based": link["roi1_result_index_1based"],
                    "roi2_result_index_1based": link["roi2_result_index_1based"],
                    "roi1_connectivity_label": link["roi1_connectivity_label"],
                    "roi2_connectivity_label": link["roi2_connectivity_label"],
                    "roi1": link["roi1"],
                    "roi2": link["roi2"],
                    "signed_weight": weight,
                    "permutation_p_uncorrected": link["permutation_p_uncorrected"],
                    "paired_change_contrast_b": b_value,
                    "abs_b": abs(b_value),
                    "f_statistic": f_value,
                    "component_node_count": selected["component_node_count"],
                    "component_link_count": selected["component_link_count"],
                    "primary_node_pfwe_plus_one": selected[
                        "primary_node_pfwe_plus_one"
                    ],
                    "primary_node_holm_all_candidates": selected[
                        "primary_node_holm_all_candidates"
                    ],
                    "posthoc_nbs_edge_pfwe_plus_one": selected[
                        "posthoc_nbs_edge_pfwe_plus_one"
                    ],
                    "posthoc_nbs_edge_holm_all_candidates": selected[
                        "posthoc_nbs_edge_holm_all_candidates"
                    ],
                    "inference_scope": DESCRIPTIVE_SCOPE,
                }
            )
    return selections, top_links


def _write_node(
    path: Path, nodes: tuple[dict[str, Any], ...], config: BrainNetConfig
) -> None:
    styles = {
        (style.connectivity_label, style.display_label): style
        for style in config.styles
    }
    expected = {(node["connectivity_label"], node["display_label"]) for node in nodes}
    if set(styles) != expected:
        missing = sorted(expected - set(styles))
        extra = sorted(set(styles) - expected)
        raise BrainNetError(
            f"node style coverage differs from frozen node map; missing={missing!r}, extra={extra!r}"
        )
    lines = []
    for node in nodes:
        style = styles[(node["connectivity_label"], node["display_label"])]
        label = node["display_label"]
        if any(character in label for character in "\t\r\n"):
            raise BrainNetError("BrainNet node labels cannot contain tabs/newlines")
        x, y, z = node["coordinates"]
        lines.append(
            "\t".join(
                (
                    format(x, ".10g"),
                    format(y, ".10g"),
                    format(z, ".10g"),
                    format(style.color, ".10g"),
                    format(style.size, ".10g"),
                    label,
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _matrix(
    node_count: int, rows: list[dict[str, Any]], top_k: int
) -> list[list[float]]:
    matrix = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    identities = set()
    for row in rows:
        first = row["roi1_result_index_1based"] - 1
        second = row["roi2_result_index_1based"] - 1
        identity = (first, second)
        if identity in identities or first == second:
            raise BrainNetError("selected top links contain duplicate/self edges")
        identities.add(identity)
        weight = row["signed_weight"]
        if not math.isfinite(weight):
            raise BrainNetError("selected edge weight is nonfinite")
        matrix[first][second] = matrix[second][first] = weight
    if len(identities) != top_k:
        raise BrainNetError("edge matrix does not contain exactly top_k unique edges")
    return matrix


def _write_edge(path: Path, matrix: list[list[float]]) -> None:
    path.write_text(
        "".join(
            "\t".join(format(value, ".4f") for value in row) + "\n" for row in matrix
        ),
        encoding="utf-8",
        newline="\n",
    )


def _write_top_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOP_LINK_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (format(value, ".10g") if isinstance(value, float) else value)
                    for key, value in row.items()
                }
            )


def _without_brainnet(project: Any) -> Any:
    if not isinstance(project, dict):
        return project
    result = dict(project)
    result.pop("brainnet", None)
    return result


def verify_brainnet_artifact_run(
    run_dir: str | Path, expected_lock: dict[str, Any] | None = None
) -> BrainNetArtifactRun:
    run_dir = Path(run_dir).resolve()
    lock = load_json(run_dir / "project.lock.json", "BrainNet artifact lock")
    manifest = load_json(
        run_dir / "brainnet" / "manifest.json", "BrainNet artifact manifest"
    )
    if expected_lock is not None and canonical_json(lock) != canonical_json(
        expected_lock
    ):
        raise BrainNetError("existing BrainNet artifact lock differs from request")
    if (
        not isinstance(lock, dict)
        or lock.get("stage") != "brainnet_artifacts"
        or not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or manifest.get("run_id") != lock.get("run_id")
        or manifest.get("scientific_identity_sha256")
        != digest_json({key: value for key, value in lock.items() if key != "run_id"})
        or manifest.get("graphvar_run_id") != lock.get("graphvar", {}).get("run_id")
        or manifest.get("graphvar_manifest_sha256")
        != lock.get("graphvar", {}).get("manifest_sha256")
    ):
        raise BrainNetError("BrainNet artifact identity is invalid")
    try:
        _verify_hash_set(
            run_dir,
            manifest.get("artifacts"),
            {"brainnet/manifest.json"},
            "BrainNet artifact",
        )
    except RuntimeError as exc:
        raise BrainNetError(str(exc)) from exc
    status = load_json(run_dir / "brainnet" / "status.json", "BrainNet status")
    validation = load_json(
        run_dir / "brainnet" / "validation.json", "BrainNet validation"
    )
    if status.get("state") != "prepared" or validation.get("status") != "passed":
        raise BrainNetError("BrainNet artifact run is not prepared and validated")
    workspaces = manifest.get("workspaces")
    if (
        not isinstance(workspaces, list)
        or not workspaces
        or len(set(workspaces)) != len(workspaces)
        or any(not isinstance(item, str) or not item for item in workspaces)
        or workspaces != lock.get("config", {}).get("selection", {}).get("workspaces")
    ):
        raise BrainNetError("BrainNet manifest workspace order is invalid")
    packaged_matlab = lock.get("packaged_matlab")
    if not isinstance(packaged_matlab, dict) or set(packaged_matlab) != {
        "neurostat_brainnet_entry.m",
        "whitebg.m",
    }:
        raise BrainNetError("BrainNet packaged MATLAB provenance is invalid")
    for name, expected_hash in packaged_matlab.items():
        resource = run_dir / "brainnet" / "matlab" / name
        if (
            not isinstance(expected_hash, str)
            or not resource.is_file()
            or sha256_file(resource) != expected_hash
        ):
            raise BrainNetError("BrainNet packaged MATLAB resource differs")
    for name, config_key in (
        ("surface.nv", "surface_sha256"),
        ("viewer-config.mat", "viewer_config_sha256"),
    ):
        expected_hash = lock.get("config", {}).get("resources", {}).get(config_key)
        resource = run_dir / "brainnet" / "resources" / name
        if (
            not isinstance(expected_hash, str)
            or not resource.is_file()
            or sha256_file(resource) != expected_hash
        ):
            raise BrainNetError("BrainNet copied viewer resource differs")
    nodes = _parse_node(run_dir / "brainnet" / "network.node")
    frozen_nodes = _node_map(run_dir / "brainnet" / "selected-node-map.json")
    styles = lock.get("config", {}).get("nodes", {}).get("styles", [])
    if not isinstance(styles, list):
        raise BrainNetError("BrainNet lock node styles are invalid")
    style_by_identity = {
        (
            _text(style.get("connectivity_label"), "style connectivity label"),
            _text(style.get("display_label"), "style display label"),
        ): style
        for style in styles
        if isinstance(style, dict)
    }
    if len(nodes) != len(frozen_nodes) or len(style_by_identity) != len(frozen_nodes):
        raise BrainNetError("BrainNet node file/map/style cardinalities differ")
    for exported, frozen in zip(nodes, frozen_nodes, strict=True):
        style = style_by_identity.get(
            (frozen["connectivity_label"], frozen["display_label"])
        )
        if style is None:
            raise BrainNetError("BrainNet node style does not cover frozen mapping")
        expected = (
            *frozen["coordinates"],
            _float(style.get("color"), "style color"),
            _float(style.get("size"), "style size"),
        )
        if exported[5] != frozen["display_label"] or any(
            not math.isclose(observed, wanted, rel_tol=1e-9, abs_tol=1e-9)
            for observed, wanted in zip(exported[:5], expected, strict=True)
        ):
            raise BrainNetError("BrainNet node file differs from frozen map/styles")
    top_rows = _read_csv(
        run_dir / "brainnet" / "top-links.csv",
        TOP_LINK_FIELDS,
        "BrainNet top-link manifest",
    )
    top_k = _integer(lock["config"]["selection"]["top_k"], "top_k", 1)
    if len(top_rows) != len(workspaces) * top_k:
        raise BrainNetError("BrainNet top-link manifest cardinality is invalid")
    top_json = load_json(
        run_dir / "brainnet" / "top-links.json", "BrainNet JSON top-link manifest"
    )
    if not isinstance(top_json, list) or len(top_json) != len(top_rows):
        raise BrainNetError("BrainNet CSV/JSON top-link cardinalities differ")
    for csv_row, json_row in zip(top_rows, top_json, strict=True):
        if not isinstance(json_row, dict) or set(json_row) != set(TOP_LINK_FIELDS):
            raise BrainNetError("BrainNet JSON top-link schema is invalid")
        json_row = cast(dict[str, Any], json_row)
        for field in TOP_LINK_FIELDS:
            if field in {
                "display_rank",
                "roi1_result_index_1based",
                "roi2_result_index_1based",
                "component_node_count",
                "component_link_count",
            }:
                equal = _integer(csv_row[field], field) == _integer(
                    json_row[field], field
                )
            elif field in {
                "signed_weight",
                "permutation_p_uncorrected",
                "paired_change_contrast_b",
                "abs_b",
                "f_statistic",
                "primary_node_pfwe_plus_one",
                "primary_node_holm_all_candidates",
                "posthoc_nbs_edge_pfwe_plus_one",
                "posthoc_nbs_edge_holm_all_candidates",
            }:
                equal = math.isclose(
                    _float(csv_row[field], field),
                    _float(json_row[field], field),
                    rel_tol=5e-10,
                    abs_tol=5e-12,
                )
            else:
                equal = csv_row[field] == json_row[field]
            if not equal:
                raise BrainNetError("BrainNet CSV/JSON top-link values differ")
    selected_components = load_json(
        run_dir / "brainnet" / "selected-components.json",
        "BrainNet selected components",
    )
    if (
        not isinstance(selected_components, list)
        or [
            row.get("workspace") for row in selected_components if isinstance(row, dict)
        ]
        != workspaces
        or any(
            not isinstance(row, dict)
            or row.get("policy")
            != lock.get("config", {}).get("selection", {}).get("policy")
            or row.get("primary_node_pfwe_supported") is not True
            for row in selected_components
        )
    ):
        raise BrainNetError("BrainNet selected-component manifest is invalid")
    for workspace in workspaces:
        rows = [row for row in top_rows if row["workspace"] == workspace]
        selected = next(
            row for row in selected_components if row["workspace"] == workspace
        )
        if len({row["component_id"] for row in rows}) != 1 or rows[0][
            "component_id"
        ] != selected.get("component_id"):
            raise BrainNetError("BrainNet top links map to the wrong component")
        if [int(row["display_rank"]) for row in rows] != list(range(1, top_k + 1)):
            raise BrainNetError("BrainNet display ranks are not contiguous")
        edge = _parse_edge(
            run_dir / "brainnet" / "edges" / f"{workspace}.edge",
            len(nodes),
            top_k,
        )
        expected_edges: set[tuple[int, int]] = set()
        for row in rows:
            first = (
                _integer(row["roi1_result_index_1based"], "top-link ROI1 index", 1) - 1
            )
            second = (
                _integer(row["roi2_result_index_1based"], "top-link ROI2 index", 1) - 1
            )
            if (
                first >= second
                or first >= len(nodes)
                or second >= len(nodes)
                or (first, second) in expected_edges
            ):
                raise BrainNetError("BrainNet top-link endpoint identity is invalid")
            expected_edges.add((first, second))
            weight = _float(row["signed_weight"], "top-link signed weight")
            if weight == 0 or edge[first][second] != float(format(weight, ".4f")):
                raise BrainNetError(
                    "BrainNet edge weight differs from its top-link manifest"
                )
        observed_edges = {
            (first, second)
            for first in range(len(nodes))
            for second in range(first + 1, len(nodes))
            if edge[first][second] != 0
        }
        if observed_edges != expected_edges:
            raise BrainNetError("BrainNet edge topology differs from top-link manifest")
    artifacts = tuple(run_dir / relative for relative in sorted(manifest["artifacts"]))
    return BrainNetArtifactRun(
        str(lock["run_id"]),
        run_dir,
        run_dir / "brainnet" / "manifest.json",
        True,
        tuple(workspaces),
        artifacts,
    )


def _parse_node(path: Path) -> list[tuple[float, float, float, float, float, str]]:
    if not path.is_file():
        raise BrainNetError(f"BrainNet node file is missing: {path}")
    rows: list[tuple[float, float, float, float, float, str]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("\t")
        if len(parts) != 6 or not parts[5]:
            raise BrainNetError(f"BrainNet node row {number} is malformed")
        first, second, third, fourth, fifth = (
            _float(value, f"node row {number}") for value in parts[:5]
        )
        rows.append((first, second, third, fourth, fifth, parts[5]))
    if len(rows) < 2 or len({row[5] for row in rows}) != len(rows):
        raise BrainNetError("BrainNet node labels/cardinality are invalid")
    return rows


def _parse_edge(path: Path, node_count: int, top_k: int) -> list[list[float]]:
    if not path.is_file():
        raise BrainNetError(f"BrainNet edge file is missing: {path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split("\t")
        if len(parts) != node_count:
            raise BrainNetError(f"BrainNet edge row {number} has wrong width")
        rows.append([_float(value, f"edge row {number}") for value in parts])
    if len(rows) != node_count:
        raise BrainNetError("BrainNet edge matrix is not square")
    nonzero = 0
    for first in range(node_count):
        if rows[first][first] != 0:
            raise BrainNetError("BrainNet edge diagonal must be zero")
        for second in range(first + 1, node_count):
            if rows[first][second] != rows[second][first]:
                raise BrainNetError("BrainNet edge matrix must be symmetric")
            nonzero += rows[first][second] != 0
    if nonzero != top_k:
        raise BrainNetError("BrainNet edge matrix does not have exactly top_k edges")
    return rows


def build_brainnet_artifacts(
    project: CleanProject | str | Path,
    graphvar_run: GraphVarAnalysisRun | str | Path,
) -> BrainNetArtifactRun:
    """Build immutable, atlas-derived BrainNet inputs from verified GraphVar output."""
    project = (
        load_project(project) if not isinstance(project, CleanProject) else project
    )
    config = parse_brainnet_config(project)
    if (
        sha256_file(config.resources.surface) != config.resources.surface_sha256
        or sha256_file(config.resources.viewer_config)
        != config.resources.viewer_config_sha256
    ):
        raise BrainNetError("BrainNet resource changed after config parsing")
    source, nodes, components, links = load_verified_graphvar_tables(graphvar_run)
    source_lock = load_json(source.run_dir / "project.lock.json", "GraphVar lock")
    if canonical_json(_without_brainnet(source_lock.get("project"))) != canonical_json(
        _without_brainnet(project.normalized())
    ):
        raise BrainNetError("GraphVar source run does not match the current project")
    selections, top_links = select_components_and_links(config, components, links)
    from .runtime import matlab_resources

    resources = matlab_resources()
    identity = {
        "stage": "brainnet_artifacts",
        "project": project.normalized(),
        "config": config.normalized(),
        "graphvar": {
            "run_id": source.run_id,
            "manifest_sha256": sha256_file(source.manifest),
            "components_sha256": sha256_file(
                source.run_dir / "graphvar" / "components-all.csv"
            ),
            "links_sha256": sha256_file(
                source.run_dir / "graphvar" / "component-member-links-all.csv"
            ),
            "node_map_sha256": sha256_file(
                source.run_dir / "graphvar" / "selected-node-map.json"
            ),
        },
        "selection": selections,
        "packaged_matlab": {
            resource.name: sha256_file(resource) for resource in resources
        },
    }
    scientific_sha = digest_json(identity)
    run_id = scientific_sha[:12]
    lock = identity | {"run_id": run_id}
    runs_root = project.output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    final = (runs_root / f"brainnet-artifacts-{run_id}").resolve()
    if final.exists():
        return verify_brainnet_artifact_run(final, lock)
    staging = Path(
        tempfile.mkdtemp(prefix=f".brainnet-artifacts-{run_id}.staging-", dir=runs_root)
    ).resolve()
    try:
        brainnet = staging / "brainnet"
        (brainnet / "edges").mkdir(parents=True)
        (brainnet / "resources").mkdir()
        atomic_json(staging / "project.lock.json", lock)
        shutil.copy2(config.resources.surface, brainnet / "resources" / "surface.nv")
        shutil.copy2(
            config.resources.viewer_config,
            brainnet / "resources" / "viewer-config.mat",
        )
        shutil.copy2(
            source.run_dir / "graphvar" / "selected-node-map.json",
            brainnet / "selected-node-map.json",
        )
        _write_node(brainnet / "network.node", nodes, config)
        for workspace in config.selection.workspaces:
            workspace_rows = [row for row in top_links if row["workspace"] == workspace]
            matrix = _matrix(len(nodes), workspace_rows, config.selection.top_k)
            _write_edge(brainnet / "edges" / f"{workspace}.edge", matrix)
        _write_top_csv(brainnet / "top-links.csv", top_links)
        atomic_json(brainnet / "top-links.json", top_links)
        atomic_json(brainnet / "selected-components.json", selections)
        matlab = brainnet / "matlab"
        matlab.mkdir()
        for resource in resources:
            shutil.copy2(resource, matlab / resource.name)
        validation = {
            "status": "passed",
            "state": "prepared",
            "node_count": len(nodes),
            "workspace_count": len(config.selection.workspaces),
            "top_k": config.selection.top_k,
            "unique_edges_per_workspace": config.selection.top_k,
            "source_components_validated": len(components),
            "source_member_links_validated": len(links),
            "primary_selection_policy": config.selection.policy,
            "nbs_qualifies_selection": False,
            "links_are_corrected_inference": False,
        }
        atomic_json(brainnet / "validation.json", validation)
        atomic_json(
            brainnet / "status.json",
            {"state": "prepared", "rendered": 0, "total": len(selections)},
        )
        artifacts = _all_files(staging, {"brainnet/manifest.json"})
        atomic_json(
            brainnet / "manifest.json",
            {
                "schema": SCHEMA,
                "run_id": run_id,
                "scientific_identity_sha256": scientific_sha,
                "graphvar_run_id": source.run_id,
                "graphvar_manifest_sha256": sha256_file(source.manifest),
                "workspaces": list(config.selection.workspaces),
                "artifacts": artifacts,
            },
        )
        if final.exists():
            raise BrainNetError(
                f"BrainNet artifact final appeared concurrently: {final}"
            )
        os.replace(staging, final)
        verified = verify_brainnet_artifact_run(final, lock)
        return BrainNetArtifactRun(
            verified.run_id,
            verified.run_dir,
            verified.manifest,
            False,
            verified.workspaces,
            verified.artifacts,
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
