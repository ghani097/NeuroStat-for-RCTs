"""Validated BrainNet node/edge artifacts from a finalized GraphVar run."""

from __future__ import annotations

import csv
import io
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
from .containment import (
    RunContainmentError,
    artifact_slug_key,
    audit_run_tree,
    is_safe_artifact_slug,
    quarantine_entry,
)

SCHEMA = "neurostat.brainnet_artifacts.v1"
SOURCE_SCIENCE_SCHEMA = "neurostat.brainnet_source_science.v1"
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


def _workspace(value: Any, name: str, spellings: dict[str, str] | None = None) -> str:
    result = _text(value, name)
    if not is_safe_artifact_slug(result):
        raise BrainNetError(f"{name} must be a safe single-component artifact slug")
    if spellings is not None:
        key = artifact_slug_key(result)
        prior = spellings.setdefault(key, result)
        if prior != result:
            raise BrainNetError(
                "GraphVar workspace spellings must be case-insensitively unique"
            )
    return result


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


def _families_by_identity(
    graphvar_lock: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    try:
        families = graphvar_lock["config"]["families"]
    except (KeyError, TypeError) as exc:
        raise BrainNetError("GraphVar lock lacks normalized family config") from exc
    if not isinstance(families, list) or not families:
        raise BrainNetError("GraphVar lock family config is invalid")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for family in families:
        if not isinstance(family, dict):
            raise BrainNetError("GraphVar lock family entry is invalid")
        key = (
            _text(family.get("name"), "family name"),
            _text(family.get("holm_family"), "Holm family"),
        )
        effect = family.get("effect")
        if isinstance(effect, dict) and set(effect) == {
            "numerator_arm",
            "denominator_arm",
        }:
            numerator = effect.get("numerator_arm")
            denominator = effect.get("denominator_arm")
        elif "effect" not in family and {
            "numerator_arm",
            "denominator_arm",
        } <= set(family):
            numerator = family.get("numerator_arm")
            denominator = family.get("denominator_arm")
        else:
            raise BrainNetError("GraphVar family effect contract is invalid")
        normalized = dict(family)
        normalized["familywise_alpha"] = _float(
            family.get("familywise_alpha"), "familywise alpha", probability=True
        )
        normalized["effect"] = {
            "numerator_arm": _text(numerator, "family numerator arm"),
            "denominator_arm": _text(denominator, "family denominator arm"),
        }
        if key in result:
            raise BrainNetError("GraphVar family identity is duplicated")
        result[key] = normalized
    return result


def _validate_graphvar_tables(
    *,
    node_map_path: Path,
    components_path: Path,
    links_path: Path,
    graphvar_lock: dict[str, Any],
) -> tuple[
    tuple[dict[str, Any], ...],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Strictly type and cross-check portable GraphVar source-science tables."""
    nodes = _node_map(node_map_path)
    node_by_index = {node["result_index_1based"]: node for node in nodes}
    component_raw = _read_csv(components_path, COMPONENT_FIELDS, "GraphVar components")
    link_raw = _read_csv(
        links_path,
        LINK_FIELDS,
        "GraphVar component member links",
    )
    family_lookup = _families_by_identity(graphvar_lock)
    components: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    workspace_spellings: dict[str, str] = {}
    for position, raw in enumerate(component_raw, 1):
        component_id = _text(raw["component_id"], f"component {position} id")
        if component_id in by_id:
            raise BrainNetError("GraphVar component identifiers must be unique")
        family_key = (
            _text(raw["family"], "component family"),
            _text(raw["holm_family"], "component Holm family"),
        )
        family = family_lookup.get(family_key)
        if family is None:
            raise BrainNetError("component family is absent from the GraphVar lock")
        alpha = family["familywise_alpha"]
        row: dict[str, Any] = dict(raw)
        row.update(
            holm_family=family_key[1],
            family=family_key[0],
            condition=_text(raw["condition"], "component condition"),
            component_id=component_id,
            workspace=_workspace(
                raw["workspace"], "component workspace", workspace_spellings
            ),
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
            ("condition" in family and row["condition"] != family["condition"])
            or (
                "bands" in family
                and (
                    not isinstance(family["bands"], list)
                    or row["band"] not in family["bands"]
                )
            )
            or (
                "session_pairs" in family
                and (
                    not isinstance(family["session_pairs"], list)
                    or row["session_pair"]
                    not in {
                        pair.get("name")
                        for pair in family["session_pairs"]
                        if isinstance(pair, dict)
                    }
                )
            )
            or (
                "effect_label" in family
                and row["effect_label"] != family["effect_label"]
            )
            or (
                "component_forming_alpha" in family
                and row["component_forming_alpha"]
                != _float(
                    family["component_forming_alpha"],
                    "family component-forming alpha",
                    probability=True,
                )
            )
            or (
                "permutations" in family
                and row["permutations"]
                != _integer(family["permutations"], "family permutations", 1)
            )
        ):
            raise BrainNetError("component metadata differs from its GraphVar family")
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
            workspace=_workspace(
                raw["workspace"], "link workspace", workspace_spellings
            ),
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
        numerator = family_lookup[(component["family"], component["holm_family"])][
            "effect"
        ]["numerator_arm"]
        denominator = family_lookup[(component["family"], component["holm_family"])][
            "effect"
        ]["denominator_arm"]
        expected_direction = (
            f"Positive: {numerator} has a more positive second-minus-first "
            f"change than {denominator}"
            if row["paired_change_contrast_b"] > 0
            else f"Negative: {numerator} has a more negative second-minus-first "
            f"change than {denominator}"
            if row["paired_change_contrast_b"] < 0
            else "Zero configured paired-change contrast"
        )
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
            or _text(raw["effect_direction"], "link effect direction")
            != expected_direction
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
    return nodes, components, links


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
    graphvar_lock = load_json(
        source.run_dir / "project.lock.json", "GraphVar analysis lock"
    )
    if not isinstance(graphvar_lock, dict):
        raise BrainNetError("GraphVar analysis lock is invalid")
    nodes, components, links = _validate_graphvar_tables(
        node_map_path=graphvar_dir / "selected-node-map.json",
        components_path=graphvar_dir / "components-all.csv",
        links_path=graphvar_dir / "component-member-links-all.csv",
        graphvar_lock=graphvar_lock,
    )
    return source, nodes, components, links


def _selection_values(
    selection: dict[str, Any],
) -> tuple[tuple[str, ...], str, int]:
    if not isinstance(selection, dict) or set(selection) != {
        "workspaces",
        "policy",
        "top_k",
    }:
        raise BrainNetError("BrainNet selection contract is invalid")
    raw_workspaces = selection["workspaces"]
    if (
        not isinstance(raw_workspaces, list)
        or not raw_workspaces
        or any(not is_safe_artifact_slug(item) for item in raw_workspaces)
        or len({artifact_slug_key(item) for item in raw_workspaces})
        != len(raw_workspaces)
    ):
        raise BrainNetError(
            "BrainNet selection workspaces must be safe, case-insensitively unique "
            "single-component artifact slugs"
        )
    policy = _text(selection["policy"], "BrainNet selection policy")
    if policy != "largest_primary_node_pfwe":
        raise BrainNetError("BrainNet selection policy is unsupported")
    return tuple(raw_workspaces), policy, _integer(selection["top_k"], "top_k", 1)


def _select_components_and_links(
    selection: dict[str, Any],
    components: list[dict[str, Any]],
    links: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workspaces, policy, top_k = _selection_values(selection)
    links_by_component: dict[str, list[dict[str, Any]]] = {}
    for row in links:
        links_by_component.setdefault(row["component_id"], []).append(row)
    selections = []
    top_links = []
    for workspace in workspaces:
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
        if len(member_links) < top_k:
            raise BrainNetError(
                f"selected component has fewer than top_k member links: {workspace}"
            )
        selected_links = member_links[:top_k]
        selections.append(
            {
                "workspace": workspace,
                "component_id": selected["component_id"],
                "policy": policy,
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


def select_components_and_links(
    config: BrainNetConfig,
    components: list[dict[str, Any]],
    links: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select requested primary components and deterministic descriptive links."""
    return _select_components_and_links(
        config.selection.normalized(), components, links
    )


def _node_text(
    nodes: tuple[dict[str, Any], ...], styles_raw: list[dict[str, Any]]
) -> str:
    styles: dict[tuple[str, str], dict[str, Any]] = {}
    for position, style in enumerate(styles_raw, 1):
        if not isinstance(style, dict) or set(style) != {
            "connectivity_label",
            "display_label",
            "color",
            "size",
        }:
            raise BrainNetError(f"BrainNet node style {position} is invalid")
        identity = (
            _text(style["connectivity_label"], "style connectivity label"),
            _text(style["display_label"], "style display label"),
        )
        color = _float(style["color"], "style color")
        size = _float(style["size"], "style size")
        if color <= 0 or size <= 0:
            raise BrainNetError("BrainNet node style color/size must be positive")
        if identity in styles:
            raise BrainNetError("BrainNet node style identity is duplicated")
        styles[identity] = style | {"color": color, "size": size}
    if len({identity[0] for identity in styles}) != len(styles) or len(
        {identity[1] for identity in styles}
    ) != len(styles):
        raise BrainNetError("BrainNet node style labels must be independently unique")
    expected = {(node["connectivity_label"], node["display_label"]) for node in nodes}
    if set(styles) != expected:
        missing = sorted(expected - set(styles))
        extra = sorted(set(styles) - expected)
        raise BrainNetError(
            f"node style coverage differs from frozen node map; missing={missing!r}, extra={extra!r}"
        )
    lines: list[str] = []
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
                    format(_float(style["color"], "style color"), ".10g"),
                    format(_float(style["size"], "style size"), ".10g"),
                    label,
                )
            )
        )
    return "\n".join(lines) + "\n"


def _write_node(
    path: Path, nodes: tuple[dict[str, Any], ...], config: BrainNetConfig
) -> None:
    path.write_text(
        _node_text(nodes, [style.normalized() for style in config.styles]),
        encoding="utf-8",
        newline="\n",
    )


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


def _edge_text(matrix: list[list[float]]) -> str:
    rows: list[str] = []
    for row in matrix:
        formatted: list[str] = []
        for value in row:
            text = format(value, ".4f")
            if value != 0 and float(text) == 0:
                raise BrainNetError(
                    "nonzero BrainNet edge weight rounds to zero at .edge precision"
                )
            formatted.append(text)
        rows.append("\t".join(formatted))
    return "\n".join(rows) + "\n"


def _write_edge(path: Path, matrix: list[list[float]]) -> None:
    path.write_text(_edge_text(matrix), encoding="utf-8", newline="\n")


def _top_csv_text(rows: list[dict[str, Any]]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=TOP_LINK_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(TOP_LINK_FIELDS):
            raise BrainNetError("BrainNet top-link row schema is invalid")
        writer.writerow(
            {
                key: (format(value, ".10g") if isinstance(value, float) else value)
                for key, value in row.items()
            }
        )
    return handle.getvalue()


def _write_top_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(_top_csv_text(rows), encoding="utf-8", newline="\n")


def _without_brainnet(project: Any) -> Any:
    if not isinstance(project, dict):
        return project
    result = dict(project)
    result.pop("brainnet", None)
    return result


def verify_brainnet_artifact_run(
    run_dir: str | Path, expected_lock: dict[str, Any] | None = None
) -> BrainNetArtifactRun:
    """Recompute every prepared artifact from its portable GraphVar evidence."""
    try:
        run_dir = audit_run_tree(run_dir)
    except RunContainmentError as exc:
        raise BrainNetError(str(exc)) from exc
    lock = load_json(run_dir / "project.lock.json", "BrainNet artifact lock")
    manifest = load_json(
        run_dir / "brainnet" / "manifest.json", "BrainNet artifact manifest"
    )
    if expected_lock is not None and canonical_json(lock) != canonical_json(
        expected_lock
    ):
        raise BrainNetError("existing BrainNet artifact lock differs from request")
    if not isinstance(lock, dict) or set(lock) != {
        "stage",
        "project",
        "config",
        "graphvar",
        "source_science",
        "selection",
        "packaged_matlab",
        "run_id",
    }:
        raise BrainNetError("BrainNet artifact lock schema is invalid")
    identity = {key: value for key, value in lock.items() if key != "run_id"}
    scientific_sha = digest_json(identity)
    if (
        lock.get("stage") != "brainnet_artifacts"
        or lock.get("run_id") != scientific_sha[:12]
        or not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or manifest.get("run_id") != lock.get("run_id")
        or manifest.get("scientific_identity_sha256") != scientific_sha
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
    config = lock.get("config")
    graphvar = lock.get("graphvar")
    science = lock.get("source_science")
    if (
        not isinstance(config, dict)
        or not isinstance(graphvar, dict)
        or set(graphvar)
        != {
            "run_id",
            "manifest_sha256",
            "components_sha256",
            "links_sha256",
            "node_map_sha256",
        }
        or not isinstance(science, dict)
        or set(science)
        != {
            "schema",
            "graphvar_run_id",
            "graphvar_manifest_sha256",
            "families",
            "artifacts",
        }
        or science.get("schema") != SOURCE_SCIENCE_SCHEMA
        or science.get("graphvar_run_id") != graphvar.get("run_id")
        or science.get("graphvar_manifest_sha256") != graphvar.get("manifest_sha256")
    ):
        raise BrainNetError("BrainNet portable source-science contract is invalid")
    science_artifacts = science.get("artifacts")
    expected_science_names = {
        "selected-node-map.json",
        "components-all.csv",
        "component-member-links-all.csv",
    }
    if (
        not isinstance(science_artifacts, dict)
        or set(science_artifacts) != expected_science_names
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in science_artifacts.values()
        )
        or science_artifacts["selected-node-map.json"]
        != graphvar.get("node_map_sha256")
        or science_artifacts["components-all.csv"] != graphvar.get("components_sha256")
        or science_artifacts["component-member-links-all.csv"]
        != graphvar.get("links_sha256")
    ):
        raise BrainNetError("BrainNet source-science hashes are invalid")
    science_dir = run_dir / "brainnet" / "source-science"
    recorded_science = load_json(
        science_dir / "analysis-contract.json",
        "BrainNet portable source-science contract",
    )
    if canonical_json(recorded_science) != canonical_json(science):
        raise BrainNetError("BrainNet source-science contract differs from its lock")
    for name, expected_hash in science_artifacts.items():
        path = science_dir / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise BrainNetError(f"BrainNet source-science input differs: {name}")
    families = science.get("families")
    nodes, components, links = _validate_graphvar_tables(
        node_map_path=science_dir / "selected-node-map.json",
        components_path=science_dir / "components-all.csv",
        links_path=science_dir / "component-member-links-all.csv",
        graphvar_lock={"config": {"families": families}},
    )
    frozen_map = run_dir / "brainnet" / "selected-node-map.json"
    if (
        not frozen_map.is_file()
        or frozen_map.read_bytes()
        != (science_dir / "selected-node-map.json").read_bytes()
    ):
        raise BrainNetError("BrainNet frozen node map differs from source science")

    selection_config = config.get("selection")
    if not isinstance(selection_config, dict):
        raise BrainNetError("BrainNet normalized selection config is invalid")
    workspaces, policy, top_k = _selection_values(selection_config)
    selections, top_links = _select_components_and_links(
        selection_config, components, links
    )
    if canonical_json(lock.get("selection")) != canonical_json(selections):
        raise BrainNetError("BrainNet locked component selection is not reproducible")
    selected_path = run_dir / "brainnet" / "selected-components.json"
    top_json_path = run_dir / "brainnet" / "top-links.json"
    if canonical_json(
        load_json(selected_path, "BrainNet selected components")
    ) != canonical_json(selections):
        raise BrainNetError("BrainNet selected-component manifest is not reproducible")
    if canonical_json(
        load_json(top_json_path, "BrainNet JSON top-link manifest")
    ) != canonical_json(top_links):
        raise BrainNetError("BrainNet JSON top links are not reproducible")
    top_csv_path = run_dir / "brainnet" / "top-links.csv"
    if top_csv_path.read_text(encoding="utf-8") != _top_csv_text(top_links):
        raise BrainNetError("BrainNet CSV top links are not reproducible")

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
    nodes_config = config.get("nodes")
    styles = nodes_config.get("styles") if isinstance(nodes_config, dict) else None
    if not isinstance(styles, list) or any(
        not isinstance(style, dict) for style in styles
    ):
        raise BrainNetError("BrainNet lock node styles are invalid")
    node_path = run_dir / "brainnet" / "network.node"
    if node_path.read_text(encoding="utf-8") != _node_text(nodes, cast(Any, styles)):
        raise BrainNetError("BrainNet node file is not reproducible")
    _parse_node(node_path)
    for workspace in workspaces:
        rows = [row for row in top_links if row["workspace"] == workspace]
        matrix = _matrix(len(nodes), rows, top_k)
        edge_path = run_dir / "brainnet" / "edges" / f"{workspace}.edge"
        if edge_path.read_text(encoding="utf-8") != _edge_text(matrix):
            raise BrainNetError(f"BrainNet edge file is not reproducible: {workspace}")
        _parse_edge(edge_path, len(nodes), top_k)

    expected_status = {
        "state": "prepared",
        "rendered": 0,
        "total": len(selections),
    }
    expected_validation = {
        "status": "passed",
        "state": "prepared",
        "node_count": len(nodes),
        "workspace_count": len(workspaces),
        "top_k": top_k,
        "unique_edges_per_workspace": top_k,
        "source_components_validated": len(components),
        "source_member_links_validated": len(links),
        "primary_selection_policy": policy,
        "nbs_qualifies_selection": False,
        "links_are_corrected_inference": False,
    }
    status = load_json(run_dir / "brainnet" / "status.json", "BrainNet status")
    validation = load_json(
        run_dir / "brainnet" / "validation.json", "BrainNet validation"
    )
    if canonical_json(status) != canonical_json(expected_status):
        raise BrainNetError("BrainNet prepared status is not reproducible")
    if canonical_json(validation) != canonical_json(expected_validation):
        raise BrainNetError("BrainNet prepared validation is not reproducible")

    expected_manifest = {
        "schema": SCHEMA,
        "run_id": lock["run_id"],
        "scientific_identity_sha256": scientific_sha,
        "graphvar_run_id": graphvar["run_id"],
        "graphvar_manifest_sha256": graphvar["manifest_sha256"],
        "workspaces": list(workspaces),
        "artifacts": _all_files(run_dir, {"brainnet/manifest.json"}),
    }
    if canonical_json(manifest) != canonical_json(expected_manifest):
        raise BrainNetError("BrainNet artifact manifest fields are not reproducible")
    artifacts = tuple(run_dir / relative for relative in sorted(manifest["artifacts"]))
    return BrainNetArtifactRun(
        str(lock["run_id"]),
        run_dir,
        run_dir / "brainnet" / "manifest.json",
        True,
        workspaces,
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
    if not isinstance(source_lock, dict):
        raise BrainNetError("GraphVar lock is invalid")
    if canonical_json(_without_brainnet(source_lock.get("project"))) != canonical_json(
        _without_brainnet(project.normalized())
    ):
        raise BrainNetError("GraphVar source run does not match the current project")
    selections, top_links = select_components_and_links(config, components, links)
    from .runtime import matlab_resources

    resources = matlab_resources()
    graphvar_dir = source.run_dir / "graphvar"
    graphvar_identity = {
        "run_id": source.run_id,
        "manifest_sha256": sha256_file(source.manifest),
        "components_sha256": sha256_file(graphvar_dir / "components-all.csv"),
        "links_sha256": sha256_file(graphvar_dir / "component-member-links-all.csv"),
        "node_map_sha256": sha256_file(graphvar_dir / "selected-node-map.json"),
    }
    source_science = {
        "schema": SOURCE_SCIENCE_SCHEMA,
        "graphvar_run_id": source.run_id,
        "graphvar_manifest_sha256": graphvar_identity["manifest_sha256"],
        "families": source_lock.get("config", {}).get("families"),
        "artifacts": {
            "selected-node-map.json": graphvar_identity["node_map_sha256"],
            "components-all.csv": graphvar_identity["components_sha256"],
            "component-member-links-all.csv": graphvar_identity["links_sha256"],
        },
    }
    _families_by_identity({"config": {"families": source_science["families"]}})
    identity = {
        "stage": "brainnet_artifacts",
        "project": project.normalized(),
        "config": config.normalized(),
        "graphvar": graphvar_identity,
        "source_science": source_science,
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
    # Keep the final leaf unresolved so a pre-existing root link is audited and
    # quarantined as the link itself, never as its external target.
    runs_root = runs_root.resolve()
    final = runs_root / f"brainnet-artifacts-{run_id}"
    if os.path.lexists(final):
        try:
            return verify_brainnet_artifact_run(final, lock)
        except (BrainNetError, RuntimeError, OSError) as verification_error:
            try:
                quarantine_entry(
                    final,
                    runs_root,
                    f"artifact-final-{run_id}",
                )
            except OSError:
                raise verification_error from None
    staging = Path(
        tempfile.mkdtemp(prefix=f".brainnet-artifacts-{run_id}.staging-", dir=runs_root)
    )
    try:
        brainnet = staging / "brainnet"
        (brainnet / "edges").mkdir(parents=True)
        (brainnet / "resources").mkdir()
        science_dir = brainnet / "source-science"
        science_dir.mkdir()
        atomic_json(staging / "project.lock.json", lock)
        shutil.copy2(config.resources.surface, brainnet / "resources" / "surface.nv")
        shutil.copy2(
            config.resources.viewer_config,
            brainnet / "resources" / "viewer-config.mat",
        )
        shutil.copy2(
            graphvar_dir / "selected-node-map.json",
            brainnet / "selected-node-map.json",
        )
        for name in (
            "selected-node-map.json",
            "components-all.csv",
            "component-member-links-all.csv",
        ):
            shutil.copy2(graphvar_dir / name, science_dir / name)
        atomic_json(science_dir / "analysis-contract.json", source_science)
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
        verify_brainnet_artifact_run(staging, lock)
        if os.path.lexists(final):
            raise BrainNetError(
                f"BrainNet artifact final appeared concurrently: {final}"
            )
        os.replace(staging, final)
        try:
            verified = verify_brainnet_artifact_run(final, lock)
        except BaseException:
            if os.path.lexists(final) and not os.path.lexists(staging):
                os.replace(final, staging)
            raise
        return BrainNetArtifactRun(
            verified.run_id,
            verified.run_dir,
            verified.manifest,
            False,
            verified.workspaces,
            verified.artifacts,
        )
    except BaseException:
        try:
            quarantine_entry(
                staging,
                runs_root,
                f"artifact-staging-{run_id}",
            )
        except OSError:
            pass
        raise
