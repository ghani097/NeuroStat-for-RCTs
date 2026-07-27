"""Resumable BrainNet rendering and immutable rendered-run verification."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..graphvar.workspace import _all_files, _verify_hash_set
from ..project import CleanProject, load_project
from ..stage_io import atomic_json, canonical_json, digest_json, load_json, sha256_file
from .artifacts import (
    BrainNetArtifactRun,
    BrainNetError,
    verify_brainnet_artifact_run,
)
from .config import BrainNetConfig, parse_brainnet_config
from .containment import (
    RunContainmentError,
    audit_run_tree,
    is_safe_artifact_slug,
)
from .qa import inspect_png
from .runtime import (
    BrainNetExecution,
    BrainNetRuntimeMetadata,
    preflight_brainnet,
    run_brainnet_image,
)

SCHEMA = "neurostat.brainnet_render.v1"
IMAGE_SCHEMA = "neurostat.brainnet_image.v1"
PREFLIGHT_PROVENANCE_SCHEMA = "neurostat.brainnet_preflight_provenance.v1"
IMAGE_PROVENANCE_SCHEMA = "neurostat.brainnet_image_provenance.v1"
HISTORICAL_PATH_SEMANTICS = (
    "Exact execution-time absolute paths retained as historical evidence; "
    "they are not asserted to exist after atomic publication."
)
CURRENT_PATH_SEMANTICS = (
    "Canonical run-root-relative carrier paths for the published immutable run."
)
PREFLIGHT_CARRIERS = {
    "request.json",
    "preflight.json",
    "preflight.stdout.log",
    "preflight.stderr.log",
    "compat/whitebg.m",
}
IMAGE_FILES_TEMPLATE = {
    "request.json",
    "execution.json",
    "matlab.stdout.log",
    "matlab.stderr.log",
    "image-qa.json",
    "completion.json",
    "render-provenance.json",
    "output-manifest.json",
}
VERSION_SCOPE = (
    "Evidence is recorded per source file; no directory-wide BrainNet "
    "version is inferred because bundled files may declare different versions."
)
Preflight = Callable[[BrainNetConfig, Path], BrainNetRuntimeMetadata]
ImageRunner = Callable[..., BrainNetExecution]


@dataclass(frozen=True, slots=True)
class BrainNetRenderRun:
    run_id: str
    run_dir: Path
    manifest: Path
    reused: bool
    images: tuple[Path, ...]
    artifacts: tuple[Path, ...]


def _same_path(first: Any, second: Any) -> bool:
    if not isinstance(first, str) or not isinstance(second, str):
        return False
    return os.path.normcase(os.path.normpath(first)) == os.path.normcase(
        os.path.normpath(second)
    )


def _version_line(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.search(r"\bversion\b", line, flags=re.IGNORECASE):
            return line.strip()[:300]
    return None


def _historical_child(root: str, *parts: str) -> str:
    return str(Path(root).joinpath(*parts))


def _resolve_current(run_root: Path, relative: Any, description: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or "\\" in relative
    ):
        raise BrainNetError(f"{description} must be a canonical run-relative path")
    path = (run_root / relative).resolve()
    try:
        path.relative_to(run_root.resolve())
    except ValueError:
        raise BrainNetError(f"{description} escapes the BrainNet render run") from None
    return path


def _preflight_bundle(
    config: BrainNetConfig,
    runs_root: Path,
    preflight_function: Preflight | None,
) -> tuple[BrainNetRuntimeMetadata, dict[str, bytes], str]:
    temporary = Path(
        tempfile.mkdtemp(prefix=".brainnet-preflight-", dir=runs_root)
    ).resolve()
    try:
        metadata = (preflight_function or preflight_brainnet)(config, temporary)
        carriers = {
            path.relative_to(temporary).as_posix(): path.read_bytes()
            for path in sorted(temporary.rglob("*"), key=lambda item: item.as_posix())
            if path.is_file()
        }
        if set(carriers) != PREFLIGHT_CARRIERS:
            missing = sorted(PREFLIGHT_CARRIERS - set(carriers))
            extra = sorted(set(carriers) - PREFLIGHT_CARRIERS)
            raise BrainNetError(
                "BrainNet preflight exact carrier set differs; "
                f"missing={missing}, extra={extra}"
            )
        return metadata, carriers, str(temporary)
    finally:
        shutil.rmtree(temporary)


def _preflight_current_paths(runtime: dict[str, Any]) -> dict[str, str]:
    result = {
        "request": "brainnet/runtime/preflight-historical/request.json",
        "metadata": "brainnet/runtime/preflight-historical/preflight.json",
        "stdout": "brainnet/runtime/preflight-historical/preflight.stdout.log",
        "stderr": "brainnet/runtime/preflight-historical/preflight.stderr.log",
        "preflight_whitebg": ("brainnet/runtime/preflight-historical/compat/whitebg.m"),
        "mapcfg_source": ("brainnet/runtime/source-evidence/BrainNet_MapCfg.m"),
        "runner_source": (
            "brainnet/source-run/brainnet/matlab/neurostat_brainnet_entry.m"
        ),
        "compatibility_source": ("brainnet/source-run/brainnet/matlab/whitebg.m"),
    }
    if runtime.get("brainnet_gui_sha256") is not None:
        result["brainnet_gui_source"] = "brainnet/runtime/source-evidence/BrainNet.m"
    if runtime.get("update_log_sha256") is not None:
        result["update_log_source"] = "brainnet/runtime/source-evidence/UpdateLog.txt"
    return result


def _write_preflight_evidence(
    run_root: Path,
    runtime: BrainNetRuntimeMetadata,
    carriers: dict[str, bytes],
    historical_root: str,
) -> None:
    runtime_dir = run_root / "brainnet" / "runtime"
    historical_dir = runtime_dir / "preflight-historical"
    historical_dir.mkdir(parents=True)
    for relative, content in carriers.items():
        destination = historical_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    source_dir = runtime_dir / "source-evidence"
    source_dir.mkdir()
    shutil.copy2(runtime.mapcfg, source_dir / "BrainNet_MapCfg.m")
    gui = runtime.brainnet_root / "BrainNet.m"
    update_log = runtime.brainnet_root / "UpdateLog.txt"
    if runtime.brainnet_gui_sha256 is not None:
        if not gui.is_file() or sha256_file(gui) != runtime.brainnet_gui_sha256:
            raise BrainNetError("BrainNet GUI source changed after preflight")
        shutil.copy2(gui, source_dir / "BrainNet.m")
    if runtime.update_log_sha256 is not None:
        if (
            not update_log.is_file()
            or sha256_file(update_log) != runtime.update_log_sha256
        ):
            raise BrainNetError("BrainNet update log changed after preflight")
        shutil.copy2(update_log, source_dir / "UpdateLog.txt")
    historical_carriers = {
        relative: {
            "current_relative_path": (
                f"brainnet/runtime/preflight-historical/{relative}"
            ),
            "sha256": sha256_file(historical_dir / relative),
        }
        for relative in sorted(carriers)
    }
    atomic_json(
        runtime_dir / "preflight-provenance.json",
        {
            "schema": PREFLIGHT_PROVENANCE_SCHEMA,
            "path_semantics": {
                "historical": HISTORICAL_PATH_SEMANTICS,
                "current": CURRENT_PATH_SEMANTICS,
            },
            "historical_execution_root": historical_root,
            "historical_carriers": historical_carriers,
            "current_paths": _preflight_current_paths(runtime.normalized()),
        },
    )


def _verify_preflight_evidence(run_root: Path, lock: dict[str, Any]) -> None:
    runtime = lock.get("runtime")
    if not isinstance(runtime, dict):
        raise BrainNetError("BrainNet runtime lock is invalid")
    provenance = load_json(
        run_root / "brainnet" / "runtime" / "preflight-provenance.json",
        "BrainNet preflight provenance",
    )
    if not isinstance(provenance, dict) or set(provenance) != {
        "schema",
        "path_semantics",
        "historical_execution_root",
        "historical_carriers",
        "current_paths",
    }:
        raise BrainNetError("BrainNet preflight provenance schema is invalid")
    if provenance.get("schema") != PREFLIGHT_PROVENANCE_SCHEMA or provenance.get(
        "path_semantics"
    ) != {
        "historical": HISTORICAL_PATH_SEMANTICS,
        "current": CURRENT_PATH_SEMANTICS,
    }:
        raise BrainNetError("BrainNet preflight path semantics are invalid")
    historical_root = provenance.get("historical_execution_root")
    if not isinstance(historical_root, str) or not Path(historical_root).is_absolute():
        raise BrainNetError("BrainNet historical preflight root is invalid")
    current_paths = provenance.get("current_paths")
    expected_current = _preflight_current_paths(runtime)
    if current_paths != expected_current:
        raise BrainNetError("BrainNet preflight current carrier paths are invalid")
    for name, relative in expected_current.items():
        path = _resolve_current(
            run_root, relative, f"BrainNet preflight current path {name}"
        )
        if not path.is_file():
            raise BrainNetError(
                f"BrainNet preflight current carrier is missing: {name}"
            )
    historical_carriers = provenance.get("historical_carriers")
    if (
        not isinstance(historical_carriers, dict)
        or set(historical_carriers) != PREFLIGHT_CARRIERS
    ):
        raise BrainNetError("BrainNet historical preflight carriers are invalid")
    for relative in sorted(PREFLIGHT_CARRIERS):
        record = historical_carriers[relative]
        expected_relative = f"brainnet/runtime/preflight-historical/{relative}"
        if (
            not isinstance(record, dict)
            or set(record) != {"current_relative_path", "sha256"}
            or record.get("current_relative_path") != expected_relative
        ):
            raise BrainNetError(
                "BrainNet historical preflight carrier record is invalid"
            )
        carrier = _resolve_current(
            run_root,
            record["current_relative_path"],
            "BrainNet historical preflight carrier",
        )
        if not carrier.is_file() or record.get("sha256") != sha256_file(carrier):
            raise BrainNetError("BrainNet historical preflight carrier hash differs")

    request = load_json(
        run_root / expected_current["request"], "BrainNet preflight request"
    )
    metadata = load_json(
        run_root / expected_current["metadata"], "BrainNet preflight metadata"
    )
    if not isinstance(request, dict) or set(request) != {
        "mode",
        "brainnet_root",
        "compat_dir",
        "metadata_path",
    }:
        raise BrainNetError("BrainNet preflight request schema is invalid")
    if (
        request.get("mode") != "preflight"
        or not _same_path(request.get("brainnet_root"), runtime.get("brainnet_root"))
        or not _same_path(
            request.get("compat_dir"),
            _historical_child(historical_root, "compat"),
        )
        or not _same_path(
            request.get("metadata_path"),
            _historical_child(historical_root, "preflight.json"),
        )
    ):
        raise BrainNetError("BrainNet historical preflight request is inconsistent")
    if not isinstance(metadata, dict) or set(metadata) != {
        "state",
        "matlab_release",
        "brainnet_mapcfg",
        "whitebg",
    }:
        raise BrainNetError("BrainNet preflight metadata schema is invalid")
    if (
        metadata.get("state") != "passed"
        or metadata.get("matlab_release") != runtime.get("matlab_release")
        or not _same_path(metadata.get("brainnet_mapcfg"), runtime.get("mapcfg"))
        or not _same_path(
            metadata.get("whitebg"),
            _historical_child(historical_root, "compat", "whitebg.m"),
        )
    ):
        raise BrainNetError("BrainNet historical preflight metadata is inconsistent")
    source_hashes = {
        "mapcfg_source": "mapcfg_sha256",
        "runner_source": "runner_sha256",
        "compatibility_source": "whitebg_sha256",
    }
    if runtime.get("brainnet_gui_sha256") is not None:
        source_hashes["brainnet_gui_source"] = "brainnet_gui_sha256"
    if runtime.get("update_log_sha256") is not None:
        source_hashes["update_log_source"] = "update_log_sha256"
    for path_key, hash_key in source_hashes.items():
        source_path = _resolve_current(
            run_root, expected_current[path_key], f"BrainNet {path_key}"
        )
        if sha256_file(source_path) != runtime.get(hash_key):
            raise BrainNetError(f"BrainNet runtime source evidence differs: {path_key}")
    if sha256_file(
        _resolve_current(
            run_root,
            expected_current["preflight_whitebg"],
            "BrainNet historical preflight whitebg",
        )
    ) != runtime.get("whitebg_sha256"):
        raise BrainNetError("BrainNet preflight whitebg differs from its runtime lock")
    version_sources = {
        "mapcfg_version_evidence": "mapcfg_source",
        "brainnet_gui_version_evidence": "brainnet_gui_source",
        "update_log_version_evidence": "update_log_source",
    }
    for evidence_key, source_key in version_sources.items():
        expected = (
            _version_line(
                _resolve_current(
                    run_root,
                    expected_current[source_key],
                    f"BrainNet {source_key}",
                )
            )
            if source_key in expected_current
            else None
        )
        if runtime.get(evidence_key) != expected:
            raise BrainNetError(
                f"BrainNet runtime version evidence differs: {evidence_key}"
            )


def _image_current_paths(workspace: str) -> dict[str, str]:
    root = f"brainnet/renders/{workspace}"
    source = "brainnet/source-run/brainnet"
    return {
        "request": f"{root}/request.json",
        "execution": f"{root}/execution.json",
        "stdout": f"{root}/matlab.stdout.log",
        "stderr": f"{root}/matlab.stderr.log",
        "image": f"{root}/{workspace}.png",
        "image_qa": f"{root}/image-qa.json",
        "completion": f"{root}/completion.json",
        "provenance": f"{root}/render-provenance.json",
        "output_manifest": f"{root}/output-manifest.json",
        "surface": f"{source}/resources/surface.nv",
        "node": f"{source}/network.node",
        "edge": f"{source}/edges/{workspace}.edge",
        "viewer_config": f"{source}/resources/viewer-config.mat",
        "compatibility_shim": f"{source}/matlab/whitebg.m",
    }


def _write_image_provenance(attempt: Path, run_root: Path, workspace: str) -> None:
    paths = _image_current_paths(workspace)
    carriers = {
        name: {
            "current_relative_path": paths[name],
            "sha256": sha256_file(attempt / filename),
        }
        for name, filename in {
            "request": "request.json",
            "execution": "execution.json",
            "stdout": "matlab.stdout.log",
            "stderr": "matlab.stderr.log",
            "image": f"{workspace}.png",
        }.items()
    }
    atomic_json(
        attempt / "render-provenance.json",
        {
            "schema": IMAGE_PROVENANCE_SCHEMA,
            "workspace": workspace,
            "path_semantics": {
                "historical": HISTORICAL_PATH_SEMANTICS,
                "current": CURRENT_PATH_SEMANTICS,
            },
            "historical_execution_root": str(attempt),
            "historical_run_root": str(run_root),
            "historical_carriers": carriers,
            "current_paths": paths,
        },
    )


def _render_values(lock: dict[str, Any]) -> tuple[int, int, int]:
    config = lock.get("config")
    render = config.get("render") if isinstance(config, dict) else None
    if not isinstance(render, dict) or set(render) != {
        "width_pixels",
        "height_pixels",
        "dpi",
    }:
        raise BrainNetError("BrainNet render config contract is invalid")
    values: list[int] = []
    for name in ("width_pixels", "height_pixels", "dpi"):
        value = render[name]
        if type(value) is not int or value < 1:
            raise BrainNetError(f"BrainNet render {name} is invalid")
        values.append(value)
    return values[0], values[1], values[2]


def _image_index_entry(workspace: str, qa: dict[str, Any]) -> dict[str, Any]:
    return {
        "workspace": workspace,
        "relative_path": f"brainnet/renders/{workspace}/{workspace}.png",
        "sha256": qa["sha256"],
        "bytes": qa["bytes"],
        "width_pixels": qa["width_pixels"],
        "height_pixels": qa["height_pixels"],
        "dpi_x": qa["dpi_x"],
        "dpi_y": qa["dpi_y"],
        "channels": qa["channels"],
        "nonconstant": qa["nonconstant"],
    }


def _verify_image_directory(
    run_root: Path,
    directory: Path,
    lock: dict[str, Any],
    workspace: str,
    *,
    source_node_sha256: str,
    source_edge_sha256: str,
    published: bool,
) -> dict[str, Any]:
    try:
        directory = audit_run_tree(directory)
    except RunContainmentError as exc:
        raise BrainNetError(str(exc)) from exc
    manifest = load_json(
        directory / "output-manifest.json", "BrainNet image output manifest"
    )
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "artifacts"}
        or manifest.get("schema") != IMAGE_SCHEMA
    ):
        raise BrainNetError(f"BrainNet image manifest schema is invalid: {workspace}")
    try:
        _verify_hash_set(
            directory,
            manifest.get("artifacts"),
            {"output-manifest.json"},
            f"BrainNet image output {workspace}",
        )
    except RuntimeError as exc:
        raise BrainNetError(str(exc)) from exc
    expected_file_set = IMAGE_FILES_TEMPLATE | {f"{workspace}.png"}
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_file_set:
        raise BrainNetError(f"BrainNet image exact file set differs: {workspace}")
    expected_manifest = {
        "schema": IMAGE_SCHEMA,
        "artifacts": _all_files(directory, {"output-manifest.json"}),
    }
    if canonical_json(manifest) != canonical_json(expected_manifest):
        raise BrainNetError(f"BrainNet image manifest differs: {workspace}")

    provenance = load_json(
        directory / "render-provenance.json", "BrainNet render provenance"
    )
    if not isinstance(provenance, dict) or set(provenance) != {
        "schema",
        "workspace",
        "path_semantics",
        "historical_execution_root",
        "historical_run_root",
        "historical_carriers",
        "current_paths",
    }:
        raise BrainNetError(f"BrainNet render provenance schema differs: {workspace}")
    if (
        provenance.get("schema") != IMAGE_PROVENANCE_SCHEMA
        or provenance.get("workspace") != workspace
        or provenance.get("path_semantics")
        != {
            "historical": HISTORICAL_PATH_SEMANTICS,
            "current": CURRENT_PATH_SEMANTICS,
        }
    ):
        raise BrainNetError(f"BrainNet render provenance identity differs: {workspace}")
    historical_execution_root = provenance.get("historical_execution_root")
    historical_run_root = provenance.get("historical_run_root")
    if (
        not isinstance(historical_execution_root, str)
        or not Path(historical_execution_root).is_absolute()
        or not isinstance(historical_run_root, str)
        or not Path(historical_run_root).is_absolute()
    ):
        raise BrainNetError(f"BrainNet historical path roots are invalid: {workspace}")
    current_paths = provenance.get("current_paths")
    expected_paths = _image_current_paths(workspace)
    if current_paths != expected_paths:
        raise BrainNetError(
            f"BrainNet current render carrier paths differ: {workspace}"
        )
    output_keys = {
        "request",
        "execution",
        "stdout",
        "stderr",
        "image",
        "image_qa",
        "completion",
        "provenance",
        "output_manifest",
    }
    for name, relative in expected_paths.items():
        current = _resolve_current(
            run_root, relative, f"BrainNet current render path {name}"
        )
        if (name not in output_keys or published) and not current.is_file():
            raise BrainNetError(
                f"BrainNet current render carrier is missing: {workspace}/{name}"
            )
        if (
            name in output_keys
            and published
            and current.parent.resolve() != directory.resolve()
        ):
            raise BrainNetError(
                f"BrainNet published carrier directory differs: {workspace}/{name}"
            )

    historical_carriers = provenance.get("historical_carriers")
    historical_names = {"request", "execution", "stdout", "stderr", "image"}
    if (
        not isinstance(historical_carriers, dict)
        or set(historical_carriers) != historical_names
    ):
        raise BrainNetError(f"BrainNet historical render carriers differ: {workspace}")
    local_names = {
        "request": "request.json",
        "execution": "execution.json",
        "stdout": "matlab.stdout.log",
        "stderr": "matlab.stderr.log",
        "image": f"{workspace}.png",
    }
    for name, filename in local_names.items():
        record = historical_carriers[name]
        if (
            not isinstance(record, dict)
            or set(record) != {"current_relative_path", "sha256"}
            or record.get("current_relative_path") != expected_paths[name]
            or record.get("sha256") != sha256_file(directory / filename)
        ):
            raise BrainNetError(
                f"BrainNet historical render carrier differs: {workspace}/{name}"
            )

    width, height, dpi = _render_values(lock)
    runtime = lock.get("runtime")
    if not isinstance(runtime, dict):
        raise BrainNetError("BrainNet runtime lock is invalid")
    request = load_json(directory / "request.json", "BrainNet render request")
    request_keys = {
        "mode",
        "brainnet_root",
        "compat_dir",
        "surface",
        "node",
        "edge",
        "viewer_config",
        "output",
        "metadata_path",
        "width_pixels",
        "height_pixels",
        "dpi",
    }
    if not isinstance(request, dict) or set(request) != request_keys:
        raise BrainNetError(f"BrainNet render request schema differs: {workspace}")
    historical_inputs = {
        "compat_dir": _historical_child(
            historical_run_root,
            "brainnet",
            "source-run",
            "brainnet",
            "matlab",
        ),
        "surface": _historical_child(
            historical_run_root,
            "brainnet",
            "source-run",
            "brainnet",
            "resources",
            "surface.nv",
        ),
        "node": _historical_child(
            historical_run_root,
            "brainnet",
            "source-run",
            "brainnet",
            "network.node",
        ),
        "edge": _historical_child(
            historical_run_root,
            "brainnet",
            "source-run",
            "brainnet",
            "edges",
            f"{workspace}.edge",
        ),
        "viewer_config": _historical_child(
            historical_run_root,
            "brainnet",
            "source-run",
            "brainnet",
            "resources",
            "viewer-config.mat",
        ),
        "output": _historical_child(historical_execution_root, f"{workspace}.png"),
        "metadata_path": _historical_child(historical_execution_root, "execution.json"),
    }
    if (
        request.get("mode") != "render"
        or not _same_path(request.get("brainnet_root"), runtime.get("brainnet_root"))
        or request.get("width_pixels") != width
        or request.get("height_pixels") != height
        or request.get("dpi") != dpi
        or any(
            not _same_path(request.get(name), expected)
            for name, expected in historical_inputs.items()
        )
    ):
        raise BrainNetError(f"BrainNet historical render request differs: {workspace}")

    execution = load_json(directory / "execution.json", "BrainNet execution metadata")
    execution_keys = {
        "state",
        "output",
        "width_pixels",
        "height_pixels",
        "dpi",
        "text_objects",
        "axes_objects",
        "brainnet_mapcfg",
        "whitebg",
    }
    if not isinstance(execution, dict) or set(execution) != execution_keys:
        raise BrainNetError(f"BrainNet execution metadata schema differs: {workspace}")
    text_objects = execution.get("text_objects")
    axes_objects = execution.get("axes_objects")
    if (
        execution.get("state") != "succeeded"
        or not _same_path(execution.get("output"), request.get("output"))
        or execution.get("width_pixels") != width
        or execution.get("height_pixels") != height
        or execution.get("dpi") != dpi
        or type(text_objects) is not int
        or text_objects < 0
        or type(axes_objects) is not int
        or axes_objects < 1
        or not _same_path(
            execution.get("brainnet_mapcfg"),
            str(Path(cast(str, runtime.get("brainnet_root"))) / "BrainNet_MapCfg.m"),
        )
        or not _same_path(
            execution.get("whitebg"),
            _historical_child(cast(str, request.get("compat_dir")), "whitebg.m"),
        )
    ):
        raise BrainNetError(
            f"BrainNet historical execution metadata differs: {workspace}"
        )

    image = directory / f"{workspace}.png"
    qa = inspect_png(image, width_pixels=width, height_pixels=height, dpi=dpi)
    qa["sha256"] = sha256_file(image)
    qa.pop("sha256_pending", None)
    recorded_qa = load_json(directory / "image-qa.json", "BrainNet image QA")
    if canonical_json(recorded_qa) != canonical_json(qa):
        raise BrainNetError(f"recorded BrainNet image QA differs: {workspace}")
    expected_completion = {
        "state": "succeeded",
        "workspace": workspace,
        "image_sha256": qa["sha256"],
        "source_edge_sha256": source_edge_sha256,
        "source_node_sha256": source_node_sha256,
    }
    completion = load_json(directory / "completion.json", "BrainNet image completion")
    if canonical_json(completion) != canonical_json(expected_completion):
        raise BrainNetError(f"BrainNet image completion differs: {workspace}")
    return _image_index_entry(workspace, qa)


def _completion_valid(
    run_root: Path,
    directory: Path,
    lock: dict[str, Any],
    workspace: str,
    *,
    source_node_sha256: str,
    source_edge_sha256: str,
) -> dict[str, Any] | None:
    if not os.path.lexists(directory):
        return None
    if not (directory / "completion.json").is_file():
        raise BrainNetError(f"incomplete BrainNet image directory exists: {workspace}")
    return _verify_image_directory(
        run_root,
        directory,
        lock,
        workspace,
        source_node_sha256=source_node_sha256,
        source_edge_sha256=source_edge_sha256,
        published=True,
    )


def _workspaces(lock: dict[str, Any]) -> tuple[str, ...]:
    config = lock.get("config")
    selection = config.get("selection") if isinstance(config, dict) else None
    raw = selection.get("workspaces") if isinstance(selection, dict) else None
    if (
        not isinstance(raw, list)
        or not raw
        or any(not is_safe_artifact_slug(item) for item in raw)
        or len(set(raw)) != len(raw)
    ):
        raise BrainNetError(
            "BrainNet render workspaces must be safe single-component artifact slugs"
        )
    return tuple(raw)


def verify_brainnet_render_run(
    run_dir: str | Path, expected_lock: dict[str, Any] | None = None
) -> BrainNetRenderRun:
    """Reverify runtime evidence, every image carrier, and aggregate contracts."""
    try:
        run_dir = audit_run_tree(run_dir)
    except RunContainmentError as exc:
        raise BrainNetError(str(exc)) from exc
    lock = load_json(run_dir / "project.lock.json", "BrainNet render lock")
    manifest = load_json(
        run_dir / "brainnet" / "manifest.json", "BrainNet render manifest"
    )
    if expected_lock is not None and canonical_json(lock) != canonical_json(
        expected_lock
    ):
        raise BrainNetError("existing BrainNet render lock differs from request")
    if not isinstance(lock, dict) or set(lock) != {
        "stage",
        "project",
        "config",
        "artifact_run_id",
        "artifact_manifest_sha256",
        "runtime",
        "run_id",
    }:
        raise BrainNetError("BrainNet render lock schema is invalid")
    identity = {key: value for key, value in lock.items() if key != "run_id"}
    scientific_sha = digest_json(identity)
    if (
        lock.get("stage") != "brainnet_render"
        or lock.get("run_id") != scientific_sha[:12]
        or not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or manifest.get("run_id") != lock.get("run_id")
        or manifest.get("scientific_identity_sha256") != scientific_sha
    ):
        raise BrainNetError("BrainNet render identity is invalid")
    runtime = lock.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "executable",
        "matlab_release",
        "brainnet_root",
        "mapcfg",
        "mapcfg_sha256",
        "brainnet_gui_sha256",
        "update_log_sha256",
        "mapcfg_version_evidence",
        "brainnet_gui_version_evidence",
        "update_log_version_evidence",
        "version_scope",
        "runner_sha256",
        "whitebg_sha256",
        "path_semantics",
    }:
        raise BrainNetError("BrainNet runtime provenance schema is invalid")
    if runtime.get("path_semantics") != "historical_execution_environment":
        raise BrainNetError("BrainNet runtime absolute paths are not typed historical")
    if (
        runtime.get("version_scope") != VERSION_SCOPE
        or not isinstance(runtime.get("executable"), str)
        or not Path(runtime["executable"]).is_absolute()
        or not isinstance(runtime.get("brainnet_root"), str)
        or not Path(runtime["brainnet_root"]).is_absolute()
        or not _same_path(
            runtime.get("mapcfg"),
            str(Path(runtime["brainnet_root"]) / "BrainNet_MapCfg.m"),
        )
    ):
        raise BrainNetError("BrainNet historical runtime environment is invalid")
    _render_values(lock)
    workspaces = _workspaces(lock)
    try:
        _verify_hash_set(
            run_dir,
            manifest.get("artifacts") if isinstance(manifest, dict) else None,
            {"brainnet/manifest.json"},
            "BrainNet render",
        )
    except RuntimeError as exc:
        raise BrainNetError(str(exc)) from exc

    copied_source = verify_brainnet_artifact_run(run_dir / "brainnet" / "source-run")
    if (
        copied_source.run_id != lock.get("artifact_run_id")
        or sha256_file(copied_source.manifest) != lock.get("artifact_manifest_sha256")
        or copied_source.workspaces != workspaces
    ):
        raise BrainNetError("BrainNet rendered source identity is invalid")
    _verify_preflight_evidence(run_dir, lock)

    source = copied_source.run_dir / "brainnet"
    computed_index: list[dict[str, Any]] = []
    images: list[Path] = []
    for workspace in workspaces:
        source_node_hash = sha256_file(source / "network.node")
        source_edge_hash = sha256_file(source / "edges" / f"{workspace}.edge")
        directory = run_dir / "brainnet" / "renders" / workspace
        computed_index.append(
            _verify_image_directory(
                run_dir,
                directory,
                lock,
                workspace,
                source_node_sha256=source_node_hash,
                source_edge_sha256=source_edge_hash,
                published=True,
            )
        )
        images.append(directory / f"{workspace}.png")
    index = load_json(run_dir / "brainnet" / "image-index.json", "BrainNet image index")
    if canonical_json(index) != canonical_json(computed_index):
        raise BrainNetError("BrainNet image index is not reproducible")
    width, height, dpi = _render_values(lock)
    expected_status = {
        "state": "rendered",
        "completed": len(workspaces),
        "total": len(workspaces),
    }
    expected_validation = {
        "status": "passed",
        "state": "rendered",
        "images": len(workspaces),
        "exact_width_pixels": width,
        "exact_height_pixels": height,
        "requested_dpi": dpi,
        "dpi_tolerance": 1.0,
        "all_nonconstant": all(item["nonconstant"] for item in computed_index),
        "source_artifact_run_id": copied_source.run_id,
        "source_artifact_manifest_sha256": sha256_file(copied_source.manifest),
    }
    status = load_json(run_dir / "brainnet" / "status.json", "BrainNet render status")
    validation = load_json(
        run_dir / "brainnet" / "validation.json", "BrainNet render validation"
    )
    if canonical_json(status) != canonical_json(expected_status):
        raise BrainNetError("BrainNet render status is not reproducible")
    if canonical_json(validation) != canonical_json(expected_validation):
        raise BrainNetError("BrainNet render validation is not reproducible")
    expected_manifest = {
        "schema": SCHEMA,
        "run_id": lock["run_id"],
        "scientific_identity_sha256": scientific_sha,
        "artifact_run_id": copied_source.run_id,
        "artifact_manifest_sha256": sha256_file(copied_source.manifest),
        "workspaces": list(workspaces),
        "artifacts": _all_files(run_dir, {"brainnet/manifest.json"}),
    }
    if canonical_json(manifest) != canonical_json(expected_manifest):
        raise BrainNetError("BrainNet render manifest fields are not reproducible")
    artifacts = tuple(run_dir / relative for relative in sorted(manifest["artifacts"]))
    return BrainNetRenderRun(
        str(lock["run_id"]),
        run_dir,
        run_dir / "brainnet" / "manifest.json",
        True,
        tuple(images),
        artifacts,
    )


def _quarantine(path: Path) -> Path:
    quarantine = path.with_name(f".{path.name}.invalid-{uuid.uuid4().hex}")
    os.replace(path, quarantine)
    return quarantine


def render_brainnet_artifacts(
    project: CleanProject | str | Path,
    artifact_run: BrainNetArtifactRun | str | Path,
    *,
    preflight_function: Preflight | None = None,
    image_runner: ImageRunner | None = None,
) -> BrainNetRenderRun:
    """Render every prepared workspace in its own resumable disposable attempt."""
    project = (
        load_project(project) if not isinstance(project, CleanProject) else project
    )
    config = parse_brainnet_config(project)
    prepared = (
        verify_brainnet_artifact_run(artifact_run.run_dir)
        if isinstance(artifact_run, BrainNetArtifactRun)
        else verify_brainnet_artifact_run(artifact_run)
    )
    prepared_lock = load_json(
        prepared.run_dir / "project.lock.json", "BrainNet artifact lock"
    )
    if canonical_json(prepared_lock.get("project")) != canonical_json(
        project.normalized()
    ) or canonical_json(prepared_lock.get("config")) != canonical_json(
        config.normalized()
    ):
        raise BrainNetError(
            "prepared BrainNet artifact run does not match current project/config"
        )
    runs_root = project.output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    runtime, carriers, historical_preflight_root = _preflight_bundle(
        config, runs_root, preflight_function
    )
    identity = {
        "stage": "brainnet_render",
        "project": project.normalized(),
        "config": config.normalized(),
        "artifact_run_id": prepared.run_id,
        "artifact_manifest_sha256": sha256_file(prepared.manifest),
        "runtime": runtime.normalized(),
    }
    scientific_sha = digest_json(identity)
    run_id = scientific_sha[:12]
    lock = identity | {"run_id": run_id}
    # Do not resolve an existing final/partial leaf: containment must inspect
    # and quarantine the link itself rather than operating on its target.
    runs_root = runs_root.resolve()
    final = runs_root / f"brainnet-render-{run_id}"
    if os.path.lexists(final):
        try:
            return verify_brainnet_render_run(final, lock)
        except (BrainNetError, RuntimeError, OSError):
            _quarantine(final)
    partial = runs_root / f".brainnet-render-{run_id}.partial"
    brainnet = partial / "brainnet"
    if os.path.lexists(partial):
        try:
            partial = audit_run_tree(partial)
        except RunContainmentError as exc:
            raise BrainNetError(str(exc)) from exc
        existing = load_json(partial / "project.lock.json", "partial BrainNet lock")
        if canonical_json(existing) != canonical_json(lock):
            raise BrainNetError("partial BrainNet render lock differs from request")
    else:
        initializing = Path(
            tempfile.mkdtemp(
                prefix=f".brainnet-render-{run_id}.initializing-", dir=runs_root
            )
        ).resolve()
        try:
            initialized = initializing / "brainnet"
            initialized.mkdir()
            atomic_json(initializing / "project.lock.json", lock)
            shutil.copytree(prepared.run_dir, initialized / "source-run")
            _write_preflight_evidence(
                initializing,
                runtime,
                carriers,
                historical_preflight_root,
            )
            try:
                audit_run_tree(initializing)
            except RunContainmentError as exc:
                raise BrainNetError(str(exc)) from exc
            _verify_preflight_evidence(initializing, lock)
            (initialized / "renders").mkdir()
            atomic_json(
                initialized / "status.json",
                {
                    "state": "rendering",
                    "completed": 0,
                    "total": len(prepared.workspaces),
                },
            )
            os.replace(initializing, partial)
        except BaseException:
            if initializing.exists():
                shutil.rmtree(initializing)
            raise
        brainnet = partial / "brainnet"
    source_run = brainnet / "source-run"
    source = source_run / "brainnet"
    runner = image_runner or run_brainnet_image
    try:
        index: list[dict[str, Any]] = []
        for workspace in prepared.workspaces:
            copied_source = verify_brainnet_artifact_run(source_run, prepared_lock)
            if copied_source.run_id != prepared.run_id or sha256_file(
                copied_source.manifest
            ) != sha256_file(prepared.manifest):
                raise BrainNetError(
                    "copied BrainNet source differs from prepared artifact run"
                )
            if (
                not runtime.mapcfg.is_file()
                or sha256_file(runtime.mapcfg) != runtime.mapcfg_sha256
                or (
                    sha256_file(config.runtime.brainnet_root / "BrainNet.m")
                    if (config.runtime.brainnet_root / "BrainNet.m").is_file()
                    else None
                )
                != runtime.brainnet_gui_sha256
                or (
                    sha256_file(config.runtime.brainnet_root / "UpdateLog.txt")
                    if (config.runtime.brainnet_root / "UpdateLog.txt").is_file()
                    else None
                )
                != runtime.update_log_sha256
            ):
                raise BrainNetError(
                    "BrainNet source changed after preflight and before rendering"
                )
            source_node_hash = sha256_file(source / "network.node")
            source_edge_hash = sha256_file(source / "edges" / f"{workspace}.edge")
            renders = brainnet / "renders"
            final_image_dir = renders / workspace
            try:
                completed = _completion_valid(
                    partial,
                    final_image_dir,
                    lock,
                    workspace,
                    source_node_sha256=source_node_hash,
                    source_edge_sha256=source_edge_hash,
                )
            except (BrainNetError, RuntimeError, OSError):
                if os.path.lexists(final_image_dir):
                    _quarantine(final_image_dir)
                    completed = None
                else:
                    raise
            if completed is not None:
                index.append(completed)
                continue
            for attempt in renders.glob(f".{workspace}.attempt-*"):
                resolved = attempt.resolve()
                try:
                    resolved.relative_to(renders.resolve())
                except ValueError:
                    raise BrainNetError("unsafe BrainNet attempt path") from None
                if not resolved.is_dir():
                    raise BrainNetError("BrainNet attempt path is not a directory")
                shutil.rmtree(resolved)
            attempt = Path(
                tempfile.mkdtemp(prefix=f".{workspace}.attempt-", dir=renders)
            ).resolve()
            execution = runner(
                config,
                surface=source / "resources" / "surface.nv",
                node=source / "network.node",
                edge=source / "edges" / f"{workspace}.edge",
                viewer_config=source / "resources" / "viewer-config.mat",
                compatibility_dir=source / "matlab",
                output_dir=attempt,
                workspace=workspace,
            )
            if execution.returncode != 0:
                raise BrainNetError(f"BrainNet runner returned nonzero: {workspace}")
            qa = inspect_png(
                attempt / f"{workspace}.png",
                width_pixels=config.render.width_pixels,
                height_pixels=config.render.height_pixels,
                dpi=config.render.dpi,
            )
            qa["sha256"] = sha256_file(attempt / f"{workspace}.png")
            qa.pop("sha256_pending", None)
            atomic_json(attempt / "image-qa.json", qa)
            atomic_json(
                attempt / "completion.json",
                {
                    "state": "succeeded",
                    "workspace": workspace,
                    "image_sha256": qa["sha256"],
                    "source_edge_sha256": source_edge_hash,
                    "source_node_sha256": source_node_hash,
                },
            )
            _write_image_provenance(attempt, partial, workspace)
            atomic_json(
                attempt / "output-manifest.json",
                {
                    "schema": IMAGE_SCHEMA,
                    "artifacts": _all_files(attempt, {"output-manifest.json"}),
                },
            )
            _verify_image_directory(
                partial,
                attempt,
                lock,
                workspace,
                source_node_sha256=source_node_hash,
                source_edge_sha256=source_edge_hash,
                published=False,
            )
            os.replace(attempt, final_image_dir)
            try:
                completed = _verify_image_directory(
                    partial,
                    final_image_dir,
                    lock,
                    workspace,
                    source_node_sha256=source_node_hash,
                    source_edge_sha256=source_edge_hash,
                    published=True,
                )
            except BaseException:
                if os.path.lexists(final_image_dir) and not os.path.lexists(attempt):
                    os.replace(final_image_dir, attempt)
                raise
            index.append(completed)
            atomic_json(
                brainnet / "status.json",
                {
                    "state": "rendering",
                    "completed": len(index),
                    "total": len(prepared.workspaces),
                },
            )
        atomic_json(brainnet / "image-index.json", index)
        atomic_json(
            brainnet / "validation.json",
            {
                "status": "passed",
                "state": "rendered",
                "images": len(index),
                "exact_width_pixels": config.render.width_pixels,
                "exact_height_pixels": config.render.height_pixels,
                "requested_dpi": config.render.dpi,
                "dpi_tolerance": 1.0,
                "all_nonconstant": all(item["nonconstant"] for item in index),
                "source_artifact_run_id": prepared.run_id,
                "source_artifact_manifest_sha256": sha256_file(prepared.manifest),
            },
        )
        atomic_json(
            brainnet / "status.json",
            {
                "state": "rendered",
                "completed": len(index),
                "total": len(prepared.workspaces),
            },
        )
        artifacts = _all_files(partial, {"brainnet/manifest.json"})
        atomic_json(
            brainnet / "manifest.json",
            {
                "schema": SCHEMA,
                "run_id": run_id,
                "scientific_identity_sha256": scientific_sha,
                "artifact_run_id": prepared.run_id,
                "artifact_manifest_sha256": sha256_file(prepared.manifest),
                "workspaces": list(prepared.workspaces),
                "artifacts": artifacts,
            },
        )
        verify_brainnet_render_run(partial, lock)
        if os.path.lexists(final):
            raise BrainNetError(f"BrainNet render final appeared concurrently: {final}")
        os.replace(partial, final)
        try:
            verified = verify_brainnet_render_run(final, lock)
        except BaseException:
            if os.path.lexists(final) and not os.path.lexists(partial):
                os.replace(final, partial)
            raise
        return BrainNetRenderRun(
            verified.run_id,
            verified.run_dir,
            verified.manifest,
            False,
            verified.images,
            verified.artifacts,
        )
    except BaseException as exc:
        if os.path.lexists(partial):
            atomic_json(
                partial / "brainnet" / "status.json",
                {
                    "state": "failed_nonfinal",
                    "completed": sum(
                        (
                            partial
                            / "brainnet"
                            / "renders"
                            / workspace
                            / "completion.json"
                        ).is_file()
                        for workspace in prepared.workspaces
                    ),
                    "total": len(prepared.workspaces),
                    "error": str(exc),
                },
            )
        raise
