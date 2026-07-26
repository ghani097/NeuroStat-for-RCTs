"""Resumable BrainNet rendering and immutable rendered-run verification."""

from __future__ import annotations

import os
import shutil
import tempfile
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
from .qa import inspect_png
from .runtime import (
    BrainNetExecution,
    BrainNetRuntimeMetadata,
    preflight_brainnet,
    run_brainnet_image,
)

SCHEMA = "neurostat.brainnet_render.v1"
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


def _preflight_bundle(
    config: BrainNetConfig,
    runs_root: Path,
    preflight_function: Preflight | None,
) -> tuple[BrainNetRuntimeMetadata, dict[str, bytes]]:
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
        return metadata, carriers
    finally:
        shutil.rmtree(temporary)


def _completion_valid(
    directory: Path,
    config: BrainNetConfig,
    workspace: str,
    *,
    source_node_sha256: str,
    source_edge_sha256: str,
) -> dict[str, Any] | None:
    completion_path = directory / "completion.json"
    if not completion_path.is_file():
        return None
    manifest = load_json(
        directory / "output-manifest.json", "BrainNet image output manifest"
    )
    try:
        _verify_hash_set(
            directory,
            manifest.get("artifacts"),
            {"output-manifest.json"},
            f"BrainNet image output {workspace}",
        )
    except RuntimeError as exc:
        raise BrainNetError(str(exc)) from exc
    completion = load_json(completion_path, "BrainNet image completion")
    image = directory / f"{workspace}.png"
    if (
        completion.get("state") != "succeeded"
        or completion.get("workspace") != workspace
        or completion.get("image_sha256")
        != (sha256_file(image) if image.is_file() else None)
        or completion.get("source_node_sha256") != source_node_sha256
        or completion.get("source_edge_sha256") != source_edge_sha256
    ):
        raise BrainNetError(f"completed BrainNet image was modified: {workspace}")
    qa = inspect_png(
        image,
        width_pixels=config.render.width_pixels,
        height_pixels=config.render.height_pixels,
        dpi=config.render.dpi,
    )
    qa["sha256"] = sha256_file(image)
    qa.pop("sha256_pending", None)
    recorded_qa = load_json(directory / "image-qa.json", "BrainNet image QA")
    if canonical_json(recorded_qa) != canonical_json(qa):
        raise BrainNetError(f"recorded BrainNet image QA differs: {workspace}")
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


def verify_brainnet_render_run(
    run_dir: str | Path, expected_lock: dict[str, Any] | None = None
) -> BrainNetRenderRun:
    run_dir = Path(run_dir).resolve()
    lock = load_json(run_dir / "project.lock.json", "BrainNet render lock")
    manifest = load_json(
        run_dir / "brainnet" / "manifest.json", "BrainNet render manifest"
    )
    if expected_lock is not None and canonical_json(lock) != canonical_json(
        expected_lock
    ):
        raise BrainNetError("existing BrainNet render lock differs from request")
    if (
        not isinstance(lock, dict)
        or lock.get("stage") != "brainnet_render"
        or not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or manifest.get("run_id") != lock.get("run_id")
        or manifest.get("scientific_identity_sha256")
        != digest_json({key: value for key, value in lock.items() if key != "run_id"})
    ):
        raise BrainNetError("BrainNet render identity is invalid")
    try:
        _verify_hash_set(
            run_dir,
            manifest.get("artifacts"),
            {"brainnet/manifest.json"},
            "BrainNet render",
        )
    except RuntimeError as exc:
        raise BrainNetError(str(exc)) from exc
    status = load_json(run_dir / "brainnet" / "status.json", "BrainNet render status")
    validation = load_json(
        run_dir / "brainnet" / "validation.json", "BrainNet render validation"
    )
    workspaces = manifest.get("workspaces")
    if (
        status.get("state") != "rendered"
        or validation.get("status") != "passed"
        or not isinstance(workspaces, list)
        or not workspaces
        or len(set(workspaces)) != len(workspaces)
        or workspaces != lock.get("config", {}).get("selection", {}).get("workspaces")
    ):
        raise BrainNetError("BrainNet render is not complete and validated")
    copied_source = verify_brainnet_artifact_run(run_dir / "brainnet" / "source-run")
    if (
        copied_source.run_id != lock.get("artifact_run_id")
        or sha256_file(copied_source.manifest) != lock.get("artifact_manifest_sha256")
        or list(copied_source.workspaces) != workspaces
    ):
        raise BrainNetError("BrainNet rendered source identity is invalid")
    render = lock.get("config", {}).get("render", {})
    index = load_json(run_dir / "brainnet" / "image-index.json", "BrainNet image index")
    if not isinstance(index, list) or any(not isinstance(item, dict) for item in index):
        raise BrainNetError("BrainNet image index/order is invalid")
    typed_index = cast(list[dict[str, Any]], index)
    if [item.get("workspace") for item in typed_index] != workspaces:
        raise BrainNetError("BrainNet image index/order is invalid")
    images = []
    for item, workspace in zip(typed_index, workspaces, strict=True):
        relative = item.get("relative_path")
        if relative != f"brainnet/renders/{workspace}/{workspace}.png":
            raise BrainNetError("BrainNet image index path is invalid")
        image = (run_dir / relative).resolve()
        try:
            image.relative_to(run_dir)
        except ValueError:
            raise BrainNetError("BrainNet image index escapes its run") from None
        qa = inspect_png(
            image,
            width_pixels=int(render["width_pixels"]),
            height_pixels=int(render["height_pixels"]),
            dpi=int(render["dpi"]),
        )
        if item.get("sha256") != sha256_file(image) or item.get("bytes") != qa["bytes"]:
            raise BrainNetError("BrainNet image index hash/size differs")
        images.append(image)
    artifacts = tuple(run_dir / relative for relative in sorted(manifest["artifacts"]))
    return BrainNetRenderRun(
        str(lock["run_id"]),
        run_dir,
        run_dir / "brainnet" / "manifest.json",
        True,
        tuple(images),
        artifacts,
    )


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
    runtime, carriers = _preflight_bundle(config, runs_root, preflight_function)
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
    final = (runs_root / f"brainnet-render-{run_id}").resolve()
    if final.exists():
        return verify_brainnet_render_run(final, lock)
    partial = (runs_root / f".brainnet-render-{run_id}.partial").resolve()
    brainnet = partial / "brainnet"
    if partial.exists():
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
            runtime_dir = initialized / "runtime"
            runtime_dir.mkdir()
            for relative, content in carriers.items():
                destination = runtime_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
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
        index = []
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
            completed = _completion_valid(
                final_image_dir,
                config,
                workspace,
                source_node_sha256=source_node_hash,
                source_edge_sha256=source_edge_hash,
            )
            if completed is not None:
                index.append(completed)
                continue
            if final_image_dir.exists():
                raise BrainNetError(
                    f"unverified BrainNet image output will not be overwritten: "
                    f"{final_image_dir}"
                )
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
            atomic_json(
                attempt / "output-manifest.json",
                {
                    "schema": "neurostat.brainnet_image.v1",
                    "artifacts": _all_files(attempt, {"output-manifest.json"}),
                },
            )
            os.replace(attempt, final_image_dir)
            completed = _completion_valid(
                final_image_dir,
                config,
                workspace,
                source_node_sha256=source_node_hash,
                source_edge_sha256=source_edge_hash,
            )
            assert completed is not None
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
        if final.exists():
            raise BrainNetError(f"BrainNet render final appeared concurrently: {final}")
        os.replace(partial, final)
        brainnet = final / "brainnet"
        verified = verify_brainnet_render_run(final, lock)
        return BrainNetRenderRun(
            verified.run_id,
            verified.run_dir,
            verified.manifest,
            False,
            verified.images,
            verified.artifacts,
        )
    except BaseException as exc:
        if partial.exists():
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
