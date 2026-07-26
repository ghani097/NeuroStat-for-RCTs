"""Strict, dependency-light GraphVar project configuration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..project import CleanProject, ProjectValidationError
from ..stage_io import sha256_file

_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_ROOT = {"atlas", "families", "runtime"}
_ATLAS = {"workbook", "brain_viewer_config", "nodes"}
_NODE = {"connectivity_label", "atlas_index_1based", "display_label", "coordinates"}
_FAMILY = {
    "name",
    "holm_family",
    "condition",
    "bands",
    "session_pairs",
    "effect",
    "component_forming_alpha",
    "familywise_alpha",
    "permutations",
    "seed",
    "matrix_field_template",
    "effect_label",
}
_PAIR = {"name", "first_session", "second_session"}
_EFFECT = {"numerator_arm", "denominator_arm"}
_SEED = {"policy", "base"}
_RUNTIME = {
    "executable",
    "graphvar_root",
    "matlab_release",
    "timeout_seconds",
}


class GraphVarConfigError(ProjectValidationError):
    """Raised before execution when a GraphVar contract is ambiguous."""


def _map(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GraphVarConfigError(f"{name} must be a mapping")
    return value


def _exact(
    value: Mapping[str, Any],
    keys: set[str],
    name: str,
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    unknown = set(value) - keys
    missing = (keys - optional) - set(value)
    if unknown or missing:
        detail = []
        if unknown:
            detail.append(f"unknown keys: {', '.join(sorted(unknown))}")
        if missing:
            detail.append(f"missing keys: {', '.join(sorted(missing))}")
        raise GraphVarConfigError(f"{name} has {'; '.join(detail)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphVarConfigError(f"{name} must be a nonblank string")
    return value.strip()


def _name(value: Any, name: str) -> str:
    result = _string(value, name)
    if not _SAFE.fullmatch(result):
        raise GraphVarConfigError(f"{name} must be a safe artifact name")
    return result


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise GraphVarConfigError(f"{name} must be a nonempty list")
    result = tuple(_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise GraphVarConfigError(f"{name} must not contain duplicates")
    return result


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise GraphVarConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraphVarConfigError(f"{name} must be a finite number in (0,1)")
    result = float(value)
    if not math.isfinite(result) or not 0 < result < 1:
        raise GraphVarConfigError(f"{name} must be a finite number in (0,1)")
    return result


def _resource(project: CleanProject, value: Any, name: str) -> tuple[Path, str]:
    path = (project.project_file.parent / _string(value, name)).resolve()
    if not path.is_file():
        raise GraphVarConfigError(f"{name} is not a file: {path}")
    return path, sha256_file(path)


@dataclass(frozen=True, slots=True)
class AtlasNode:
    connectivity_label: str
    atlas_index_1based: int
    display_label: str
    coordinates: tuple[float, float, float] | None

    def normalized(self) -> dict[str, Any]:
        return {
            "connectivity_label": self.connectivity_label,
            "atlas_index_1based": self.atlas_index_1based,
            "display_label": self.display_label,
            "coordinates": None if self.coordinates is None else list(self.coordinates),
        }


@dataclass(frozen=True, slots=True)
class AtlasSettings:
    workbook: Path
    workbook_sha256: str
    brain_viewer_config: Path
    brain_viewer_config_sha256: str
    nodes: tuple[AtlasNode, ...]

    def normalized(self) -> dict[str, Any]:
        return {
            "workbook": str(self.workbook),
            "workbook_sha256": self.workbook_sha256,
            "brain_viewer_config": str(self.brain_viewer_config),
            "brain_viewer_config_sha256": self.brain_viewer_config_sha256,
            "nodes": [node.normalized() for node in self.nodes],
        }


@dataclass(frozen=True, slots=True)
class SessionPair:
    name: str
    first_session: str
    second_session: str

    def normalized(self) -> dict[str, str]:
        return {
            "name": self.name,
            "first_session": self.first_session,
            "second_session": self.second_session,
        }


@dataclass(frozen=True, slots=True)
class GraphVarFamily:
    name: str
    holm_family: str
    condition: str
    bands: tuple[str, ...]
    session_pairs: tuple[SessionPair, ...]
    numerator_arm: str
    denominator_arm: str
    component_forming_alpha: float
    familywise_alpha: float
    permutations: int
    seed_policy: str
    seed_base: int
    matrix_field_template: str
    effect_label: str

    def normalized(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "holm_family": self.holm_family,
            "condition": self.condition,
            "bands": list(self.bands),
            "session_pairs": [pair.normalized() for pair in self.session_pairs],
            "effect": {
                "numerator_arm": self.numerator_arm,
                "denominator_arm": self.denominator_arm,
            },
            "component_forming_alpha": self.component_forming_alpha,
            "familywise_alpha": self.familywise_alpha,
            "permutations": self.permutations,
            "seed": {"policy": self.seed_policy, "base": self.seed_base},
            "matrix_field_template": self.matrix_field_template,
            "effect_label": self.effect_label,
        }


@dataclass(frozen=True, slots=True)
class GraphVarRuntime:
    executable: str
    graphvar_root: Path
    matlab_release: str
    timeout_seconds: int

    def normalized(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "graphvar_root": str(self.graphvar_root),
            "matlab_release": self.matlab_release,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class GraphVarConfig:
    atlas: AtlasSettings
    families: tuple[GraphVarFamily, ...]
    runtime: GraphVarRuntime

    def normalized(self) -> dict[str, Any]:
        return {
            "atlas": self.atlas.normalized(),
            "families": [family.normalized() for family in self.families],
            "runtime": self.runtime.normalized(),
        }


def parse_graphvar_config(project: CleanProject) -> GraphVarConfig:
    raw = _map(project.extras.get("graphvar"), "graphvar")
    _exact(raw, _ROOT, "graphvar")
    atlas_raw = _map(raw["atlas"], "graphvar.atlas")
    _exact(atlas_raw, _ATLAS, "graphvar.atlas")
    workbook, workbook_sha = _resource(
        project, atlas_raw["workbook"], "graphvar.atlas.workbook"
    )
    viewer, viewer_sha = _resource(
        project, atlas_raw["brain_viewer_config"], "graphvar.atlas.brain_viewer_config"
    )
    nodes_raw = atlas_raw["nodes"]
    if not isinstance(nodes_raw, (list, tuple)) or len(nodes_raw) < 2:
        raise GraphVarConfigError(
            "graphvar.atlas.nodes must contain at least two nodes"
        )
    nodes: list[AtlasNode] = []
    for index, item in enumerate(nodes_raw, 1):
        node = _map(item, f"graphvar.atlas.nodes[{index}]")
        _exact(node, _NODE, f"graphvar.atlas.nodes[{index}]", {"coordinates"})
        coords = node.get("coordinates")
        if coords is not None:
            if (
                not isinstance(coords, (list, tuple))
                or len(coords) != 3
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not math.isfinite(float(x))
                    for x in coords
                )
            ):
                raise GraphVarConfigError(
                    f"graphvar.atlas.nodes[{index}].coordinates must be three finite numbers"
                )
            coordinates = (float(coords[0]), float(coords[1]), float(coords[2]))
        else:
            coordinates = None
        nodes.append(
            AtlasNode(
                _string(node["connectivity_label"], "connectivity_label"),
                _integer(node["atlas_index_1based"], "atlas_index_1based"),
                _string(node["display_label"], "display_label"),
                coordinates,
            )
        )
    if len({node.connectivity_label for node in nodes}) != len(nodes):
        raise GraphVarConfigError("atlas connectivity_label values must be unique")
    if len({node.atlas_index_1based for node in nodes}) != len(nodes):
        raise GraphVarConfigError("atlas indices must be unique")
    if len({node.display_label for node in nodes}) != len(nodes):
        raise GraphVarConfigError("atlas display_label values must be unique")
    if tuple(sorted(node.atlas_index_1based for node in nodes)) != tuple(
        node.atlas_index_1based for node in nodes
    ):
        raise GraphVarConfigError(
            "atlas nodes must be ordered by increasing 1-based atlas index"
        )

    families_raw = raw["families"]
    if not isinstance(families_raw, (list, tuple)) or not families_raw:
        raise GraphVarConfigError("graphvar.families must be a nonempty list")
    families: list[GraphVarFamily] = []
    planned: set[tuple[str, str, str, str]] = set()
    scientific_plans: set[tuple[str, str, str, str]] = set()
    holm_condition: dict[str, str] = {}
    holm_semantics: dict[str, tuple[float, float]] = {}
    for family_index, item in enumerate(families_raw, 1):
        value = _map(item, f"graphvar.families[{family_index}]")
        _exact(value, _FAMILY, f"graphvar.families[{family_index}]")
        name = _name(value["name"], "family.name")
        holm_family = _name(value["holm_family"], "family.holm_family")
        condition = _string(value["condition"], "family.condition")
        if condition not in project.study.conditions:
            raise GraphVarConfigError(
                f"family {name} references undeclared condition {condition!r}"
            )
        prior_condition = holm_condition.setdefault(holm_family, condition)
        if prior_condition != condition:
            raise GraphVarConfigError(
                f"Holm family {holm_family!r} spans multiple conditions"
            )
        bands = _strings(value["bands"], "family.bands")
        effect = _map(value["effect"], "family.effect")
        _exact(effect, _EFFECT, "family.effect")
        numerator = _string(effect["numerator_arm"], "numerator_arm")
        denominator = _string(effect["denominator_arm"], "denominator_arm")
        if {numerator, denominator} != set(project.study.arms):
            raise GraphVarConfigError(
                "family effect must orient the project's exact two arms"
            )
        seed = _map(value["seed"], "family.seed")
        _exact(seed, _SEED, "family.seed")
        if seed["policy"] != "sha256_workspace":
            raise GraphVarConfigError("family.seed.policy must be sha256_workspace")
        pairs_raw = value["session_pairs"]
        if not isinstance(pairs_raw, (list, tuple)) or not pairs_raw:
            raise GraphVarConfigError("family.session_pairs must be nonempty")
        pairs: list[SessionPair] = []
        for pair_index, pair_item in enumerate(pairs_raw, 1):
            pair = _map(pair_item, f"session_pairs[{pair_index}]")
            _exact(pair, _PAIR, f"session_pairs[{pair_index}]")
            first = _string(pair["first_session"], "first_session")
            second = _string(pair["second_session"], "second_session")
            if (
                first == second
                or first not in project.study.sessions
                or second not in project.study.sessions
            ):
                raise GraphVarConfigError(
                    "session pair must contain two distinct declared sessions"
                )
            pair_name = _name(pair["name"], "session_pair.name")
            pairs.append(SessionPair(pair_name, first, second))
            for band in bands:
                key = (holm_family, condition, band, pair_name)
                if key in planned:
                    raise GraphVarConfigError(
                        f"ambiguous duplicate planned workspace in Holm family: {key}"
                    )
                planned.add(key)
                scientific_key = (condition, band, first, second)
                if scientific_key in scientific_plans:
                    raise GraphVarConfigError(
                        f"duplicate scientific analysis across families: {scientific_key}"
                    )
                scientific_plans.add(scientific_key)
        if len({pair.name for pair in pairs}) != len(pairs):
            raise GraphVarConfigError(f"family {name} has duplicate session-pair names")
        field = _string(value["matrix_field_template"], "matrix_field_template")
        if field.count("{band}") != 1:
            raise GraphVarConfigError(
                "matrix_field_template must contain {band} exactly once"
            )
        component_alpha = _probability(
            value["component_forming_alpha"], "component_forming_alpha"
        )
        familywise_alpha = _probability(value["familywise_alpha"], "familywise_alpha")
        semantics = (component_alpha, familywise_alpha)
        prior_semantics = holm_semantics.setdefault(holm_family, semantics)
        if prior_semantics != semantics:
            raise GraphVarConfigError(
                f"Holm family {holm_family!r} mixes component/familywise alpha semantics"
            )
        effect_label = _string(value["effect_label"], "effect_label")
        permitted_effect_labels = {
            f"arm_{numerator}-arm_{denominator}=0",
            f"arm_{denominator}-arm_{numerator}=0",
        }
        if effect_label not in permitted_effect_labels:
            raise GraphVarConfigError(
                "effect_label must be exactly the configured arm numerator-minus-"
                "denominator GraphVar label or its exact reverse"
            )
        families.append(
            GraphVarFamily(
                name,
                holm_family,
                condition,
                bands,
                tuple(pairs),
                numerator,
                denominator,
                component_alpha,
                familywise_alpha,
                _integer(value["permutations"], "permutations"),
                "sha256_workspace",
                _integer(value["seed"]["base"], "seed.base", 0),
                field,
                effect_label,
            )
        )
    if len({family.name for family in families}) != len(families):
        raise GraphVarConfigError("family names must be unique")

    runtime_raw = _map(raw["runtime"], "graphvar.runtime")
    _exact(runtime_raw, _RUNTIME, "graphvar.runtime")
    graphvar_root = (
        project.project_file.parent
        / _string(runtime_raw["graphvar_root"], "graphvar_root")
    ).resolve()
    if not graphvar_root.is_dir():
        raise GraphVarConfigError(
            f"graphvar.runtime.graphvar_root is not a directory: {graphvar_root}"
        )
    executable = _string(runtime_raw["executable"], "runtime.executable")
    if not Path(executable).is_absolute() and ("/" in executable or "\\" in executable):
        executable = str((project.project_file.parent / executable).resolve())
    return GraphVarConfig(
        AtlasSettings(workbook, workbook_sha, viewer, viewer_sha, tuple(nodes)),
        tuple(families),
        GraphVarRuntime(
            executable,
            graphvar_root,
            _string(runtime_raw["matlab_release"], "runtime.matlab_release"),
            _integer(runtime_raw["timeout_seconds"], "runtime.timeout_seconds"),
        ),
    )
