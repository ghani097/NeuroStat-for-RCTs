"""Strict, dependency-light BrainNet project configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..project import CleanProject, ProjectValidationError
from ..stage_io import sha256_file
from .containment import is_safe_artifact_slug

_ROOT = {"resources", "selection", "nodes", "render", "runtime"}
_RESOURCES = {"surface", "viewer_config"}
_SELECTION = {"workspaces", "policy", "top_k"}
_NODES = {"styles"}
_STYLE = {"connectivity_label", "display_label", "color", "size"}
_RENDER = {"width_pixels", "height_pixels", "dpi"}
_RUNTIME = {
    "executable",
    "brainnet_root",
    "matlab_release",
    "timeout_seconds",
}


class BrainNetConfigError(ProjectValidationError):
    """Raised before artifact generation when BrainNet config is ambiguous."""


def _map(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BrainNetConfigError(f"{name} must be a mapping")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown keys: {', '.join(sorted(unknown))}")
        if missing:
            details.append(f"missing keys: {', '.join(sorted(missing))}")
        raise BrainNetConfigError(f"{name} has {'; '.join(details)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BrainNetConfigError(f"{name} must be a nonblank string")
    return value.strip()


def _safe(value: Any, name: str) -> str:
    result = _string(value, name)
    if not is_safe_artifact_slug(result):
        raise BrainNetConfigError(f"{name} must be a safe artifact name")
    return result


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise BrainNetConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BrainNetConfigError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise BrainNetConfigError(f"{name} must be a finite positive number")
    return result


def _path(project: CleanProject, value: Any, name: str, *, directory: bool) -> Path:
    raw = _string(value, name)
    candidate = Path(raw)
    path = (
        candidate.resolve()
        if candidate.is_absolute()
        else (project.project_file.parent / candidate).resolve()
    )
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise BrainNetConfigError(f"{name} is not a {kind}: {path}")
    return path


@dataclass(frozen=True, slots=True)
class BrainNetResources:
    surface: Path
    surface_sha256: str
    viewer_config: Path
    viewer_config_sha256: str

    def normalized(self) -> dict[str, Any]:
        return {
            "surface": str(self.surface),
            "surface_sha256": self.surface_sha256,
            "viewer_config": str(self.viewer_config),
            "viewer_config_sha256": self.viewer_config_sha256,
        }


@dataclass(frozen=True, slots=True)
class NodeStyle:
    connectivity_label: str
    display_label: str
    color: float
    size: float

    def normalized(self) -> dict[str, Any]:
        return {
            "connectivity_label": self.connectivity_label,
            "display_label": self.display_label,
            "color": self.color,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class BrainNetSelection:
    workspaces: tuple[str, ...]
    policy: str
    top_k: int

    def normalized(self) -> dict[str, Any]:
        return {
            "workspaces": list(self.workspaces),
            "policy": self.policy,
            "top_k": self.top_k,
        }


@dataclass(frozen=True, slots=True)
class BrainNetRender:
    width_pixels: int
    height_pixels: int
    dpi: int

    def normalized(self) -> dict[str, int]:
        return {
            "width_pixels": self.width_pixels,
            "height_pixels": self.height_pixels,
            "dpi": self.dpi,
        }


@dataclass(frozen=True, slots=True)
class BrainNetRuntime:
    executable: str
    brainnet_root: Path
    matlab_release: str
    timeout_seconds: int

    def normalized(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "brainnet_root": str(self.brainnet_root),
            "matlab_release": self.matlab_release,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class BrainNetConfig:
    resources: BrainNetResources
    selection: BrainNetSelection
    styles: tuple[NodeStyle, ...]
    render: BrainNetRender
    runtime: BrainNetRuntime

    def normalized(self) -> dict[str, Any]:
        return {
            "resources": self.resources.normalized(),
            "selection": self.selection.normalized(),
            "nodes": {"styles": [style.normalized() for style in self.styles]},
            "render": self.render.normalized(),
            "runtime": self.runtime.normalized(),
        }


def parse_brainnet_config(project: CleanProject) -> BrainNetConfig:
    """Parse the exact BrainNet schema and bind every file resource by hash."""
    raw = _map(project.extras.get("brainnet"), "brainnet")
    _exact(raw, _ROOT, "brainnet")

    resources_raw = _map(raw["resources"], "brainnet.resources")
    _exact(resources_raw, _RESOURCES, "brainnet.resources")
    surface = _path(
        project, resources_raw["surface"], "brainnet.resources.surface", directory=False
    )
    viewer_config = _path(
        project,
        resources_raw["viewer_config"],
        "brainnet.resources.viewer_config",
        directory=False,
    )

    selection_raw = _map(raw["selection"], "brainnet.selection")
    _exact(selection_raw, _SELECTION, "brainnet.selection")
    workspace_raw = selection_raw["workspaces"]
    if not isinstance(workspace_raw, (list, tuple)) or not workspace_raw:
        raise BrainNetConfigError(
            "brainnet.selection.workspaces must be a nonempty ordered list"
        )
    workspaces = tuple(
        _safe(item, f"brainnet.selection.workspaces[{index}]")
        for index, item in enumerate(workspace_raw, 1)
    )
    if len(set(workspaces)) != len(workspaces):
        raise BrainNetConfigError(
            "brainnet.selection.workspaces must not contain duplicates"
        )
    policy = _string(selection_raw["policy"], "brainnet.selection.policy")
    if policy != "largest_primary_node_pfwe":
        raise BrainNetConfigError(
            "brainnet.selection.policy must be largest_primary_node_pfwe"
        )

    nodes_raw = _map(raw["nodes"], "brainnet.nodes")
    _exact(nodes_raw, _NODES, "brainnet.nodes")
    styles_raw = nodes_raw["styles"]
    if not isinstance(styles_raw, (list, tuple)) or not styles_raw:
        raise BrainNetConfigError(
            "brainnet.nodes.styles must be a nonempty exact node-style list"
        )
    styles = []
    for index, item in enumerate(styles_raw, 1):
        style = _map(item, f"brainnet.nodes.styles[{index}]")
        _exact(style, _STYLE, f"brainnet.nodes.styles[{index}]")
        styles.append(
            NodeStyle(
                _string(style["connectivity_label"], "style.connectivity_label"),
                _string(style["display_label"], "style.display_label"),
                _positive(style["color"], "style.color"),
                _positive(style["size"], "style.size"),
            )
        )
    if len({style.connectivity_label for style in styles}) != len(styles):
        raise BrainNetConfigError("node style connectivity labels must be unique")
    if len({style.display_label for style in styles}) != len(styles):
        raise BrainNetConfigError("node style display labels must be unique")

    render_raw = _map(raw["render"], "brainnet.render")
    _exact(render_raw, _RENDER, "brainnet.render")
    runtime_raw = _map(raw["runtime"], "brainnet.runtime")
    _exact(runtime_raw, _RUNTIME, "brainnet.runtime")
    root_raw = _string(runtime_raw["brainnet_root"], "brainnet.runtime.brainnet_root")
    root_candidate = Path(root_raw)
    root = (
        root_candidate.resolve()
        if root_candidate.is_absolute()
        else (project.project_file.parent / root_candidate).resolve()
    )
    executable = _string(runtime_raw["executable"], "brainnet.runtime.executable")
    executable_path = Path(executable)
    if not executable_path.is_absolute() and ("/" in executable or "\\" in executable):
        executable = str((project.project_file.parent / executable_path).resolve())

    return BrainNetConfig(
        BrainNetResources(
            surface,
            sha256_file(surface),
            viewer_config,
            sha256_file(viewer_config),
        ),
        BrainNetSelection(
            workspaces,
            policy,
            _integer(selection_raw["top_k"], "brainnet.selection.top_k"),
        ),
        tuple(styles),
        BrainNetRender(
            _integer(render_raw["width_pixels"], "brainnet.render.width_pixels"),
            _integer(render_raw["height_pixels"], "brainnet.render.height_pixels"),
            _integer(render_raw["dpi"], "brainnet.render.dpi"),
        ),
        BrainNetRuntime(
            executable,
            root,
            _string(runtime_raw["matlab_release"], "brainnet.runtime.matlab_release"),
            _integer(
                runtime_raw["timeout_seconds"], "brainnet.runtime.timeout_seconds"
            ),
        ),
    )
