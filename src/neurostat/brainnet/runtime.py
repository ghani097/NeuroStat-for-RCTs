"""MATLAB/BrainNet preflight and noninteractive per-image render boundary."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..stage_io import atomic_json, load_json, sha256_file
from .artifacts import BrainNetError
from .config import BrainNetConfig
from .containment import is_safe_artifact_slug
from .qa import inspect_png

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class BrainNetRuntimeMetadata:
    executable: Path
    matlab_release: str
    brainnet_root: Path
    mapcfg: Path
    mapcfg_sha256: str
    brainnet_gui_sha256: str | None
    update_log_sha256: str | None
    mapcfg_version_evidence: str | None
    brainnet_gui_version_evidence: str | None
    update_log_version_evidence: str | None
    runner_sha256: str
    whitebg_sha256: str

    def normalized(self) -> dict[str, Any]:
        return {
            "path_semantics": "historical_execution_environment",
            "executable": str(self.executable),
            "matlab_release": self.matlab_release,
            "brainnet_root": str(self.brainnet_root),
            "mapcfg": str(self.mapcfg),
            "mapcfg_sha256": self.mapcfg_sha256,
            "brainnet_gui_sha256": self.brainnet_gui_sha256,
            "update_log_sha256": self.update_log_sha256,
            "mapcfg_version_evidence": self.mapcfg_version_evidence,
            "brainnet_gui_version_evidence": self.brainnet_gui_version_evidence,
            "update_log_version_evidence": self.update_log_version_evidence,
            "version_scope": (
                "Evidence is recorded per source file; no directory-wide BrainNet "
                "version is inferred because bundled files may declare different versions."
            ),
            "runner_sha256": self.runner_sha256,
            "whitebg_sha256": self.whitebg_sha256,
        }


@dataclass(frozen=True, slots=True)
class BrainNetExecution:
    returncode: int
    started_unix: float
    finished_unix: float
    elapsed_seconds: float
    stdout: Path
    stderr: Path
    metadata: Path
    image: Path
    qa: dict[str, Any]


def matlab_resources() -> tuple[Path, Path]:
    root = Path(__file__).parent / "matlab"
    resources = (
        root / "neurostat_brainnet_entry.m",
        root / "compat" / "whitebg.m",
    )
    if any(not path.is_file() for path in resources):
        raise BrainNetError("packaged BrainNet MATLAB resources are missing")
    return resources


def resolve_matlab(config: BrainNetConfig) -> Path:
    requested = config.runtime.executable
    if Path(requested).is_absolute() or "/" in requested or "\\" in requested:
        candidate = Path(requested).resolve()
    else:
        found = shutil.which(requested)
        if found is None:
            raise BrainNetError(
                f"MATLAB executable {requested!r} was not found; configure "
                "brainnet.runtime.executable"
            )
        candidate = Path(found).resolve()
    if not candidate.is_file():
        raise BrainNetError(f"MATLAB executable is not a file: {candidate}")
    return candidate


def _invoke(
    config: BrainNetConfig,
    request_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    *,
    process_runner: ProcessRunner,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["NEUROSTAT_BRAINNET_REQUEST"] = str(request_path.resolve())
    environment["NEUROSTAT_BRAINNET_ENTRY"] = str(matlab_resources()[0].resolve())
    args: Sequence[str] = (
        str(resolve_matlab(config)),
        "-batch",
        "run(getenv('NEUROSTAT_BRAINNET_ENTRY'))",
    )
    try:
        completed = process_runner(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.runtime.timeout_seconds,
            env=environment,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        raise BrainNetError(
            f"MATLAB/BrainNet exceeded {config.runtime.timeout_seconds} seconds"
        ) from exc
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "no MATLAB output").strip()
        raise BrainNetError(
            f"MATLAB/BrainNet failed with exit code {completed.returncode}: {details}"
        )
    return completed


def _version_line(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if re.search(r"\bversion\b", line, flags=re.IGNORECASE):
            return line.strip()[:300]
    return None


def preflight_brainnet(
    config: BrainNetConfig,
    output_dir: Path,
    *,
    process_runner: ProcessRunner = subprocess.run,
) -> BrainNetRuntimeMetadata:
    """Require exact MATLAB release, MapCfg resolution, and compatibility shim."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mapcfg = (config.runtime.brainnet_root / "BrainNet_MapCfg.m").resolve()
    gui = (config.runtime.brainnet_root / "BrainNet.m").resolve()
    update_log = (config.runtime.brainnet_root / "UpdateLog.txt").resolve()
    if not mapcfg.is_file():
        raise BrainNetError("BrainNet root lacks BrainNet_MapCfg.m")
    compat = output_dir / "compat"
    compat.mkdir(exist_ok=True)
    shutil.copy2(matlab_resources()[1], compat / "whitebg.m")
    metadata_path = output_dir / "preflight.json"
    request_path = output_dir / "request.json"
    atomic_json(
        request_path,
        {
            "mode": "preflight",
            "brainnet_root": str(config.runtime.brainnet_root),
            "compat_dir": str(compat),
            "metadata_path": str(metadata_path),
        },
    )
    _invoke(
        config,
        request_path,
        output_dir / "preflight.stdout.log",
        output_dir / "preflight.stderr.log",
        process_runner=process_runner,
    )
    metadata = load_json(metadata_path, "MATLAB/BrainNet preflight metadata")
    expected_mapcfg = str(mapcfg).replace("\\", "/")
    expected_whitebg = str((compat / "whitebg.m").resolve()).replace("\\", "/")
    if (
        metadata.get("state") != "passed"
        or metadata.get("matlab_release") != config.runtime.matlab_release
        or str(metadata.get("brainnet_mapcfg", "")).replace("\\", "/").casefold()
        != expected_mapcfg.casefold()
        or str(metadata.get("whitebg", "")).replace("\\", "/").casefold()
        != expected_whitebg.casefold()
    ):
        raise BrainNetError(
            "MATLAB release, BrainNet_MapCfg, or whitebg resolution differs"
        )
    return BrainNetRuntimeMetadata(
        resolve_matlab(config),
        config.runtime.matlab_release,
        config.runtime.brainnet_root,
        mapcfg,
        sha256_file(mapcfg),
        sha256_file(gui) if gui.is_file() else None,
        sha256_file(update_log) if update_log.is_file() else None,
        _version_line(mapcfg),
        _version_line(gui),
        _version_line(update_log),
        sha256_file(matlab_resources()[0]),
        sha256_file(matlab_resources()[1]),
    )


def run_brainnet_image(
    config: BrainNetConfig,
    *,
    surface: Path,
    node: Path,
    edge: Path,
    viewer_config: Path,
    compatibility_dir: Path,
    output_dir: Path,
    workspace: str,
    process_runner: ProcessRunner = subprocess.run,
) -> BrainNetExecution:
    """Render one image and reject process, metadata, or pixel false success."""
    if not is_safe_artifact_slug(workspace):
        raise BrainNetError(
            "BrainNet render workspace must be a safe single-component artifact slug"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mapcfg = (config.runtime.brainnet_root / "BrainNet_MapCfg.m").resolve()
    gui = (config.runtime.brainnet_root / "BrainNet.m").resolve()
    if not mapcfg.is_file():
        raise BrainNetError("BrainNet_MapCfg.m disappeared after preflight")
    mapcfg_hash_before = sha256_file(mapcfg)
    gui_hash_before = sha256_file(gui) if gui.is_file() else None
    image = output_dir / f"{workspace}.png"
    metadata_path = output_dir / "execution.json"
    request_path = output_dir / "request.json"
    atomic_json(
        request_path,
        {
            "mode": "render",
            "brainnet_root": str(config.runtime.brainnet_root),
            "compat_dir": str(compatibility_dir.resolve()),
            "surface": str(surface.resolve()),
            "node": str(node.resolve()),
            "edge": str(edge.resolve()),
            "viewer_config": str(viewer_config.resolve()),
            "output": str(image.resolve()),
            "metadata_path": str(metadata_path.resolve()),
            **config.render.normalized(),
        },
    )
    started = time.time()
    completed = _invoke(
        config,
        request_path,
        output_dir / "matlab.stdout.log",
        output_dir / "matlab.stderr.log",
        process_runner=process_runner,
    )
    finished = time.time()
    metadata = load_json(metadata_path, "BrainNet render metadata")
    expected_mapcfg = str(mapcfg).replace("\\", "/")
    expected_whitebg = str((compatibility_dir / "whitebg.m").resolve()).replace(
        "\\", "/"
    )
    if (
        metadata.get("state") != "succeeded"
        or metadata.get("output") != str(image.resolve())
        or metadata.get("width_pixels") != config.render.width_pixels
        or metadata.get("height_pixels") != config.render.height_pixels
        or metadata.get("dpi") != config.render.dpi
        or not isinstance(metadata.get("text_objects"), int)
        or not isinstance(metadata.get("axes_objects"), int)
        or str(metadata.get("brainnet_mapcfg", "")).replace("\\", "/").casefold()
        != expected_mapcfg.casefold()
        or str(metadata.get("whitebg", "")).replace("\\", "/").casefold()
        != expected_whitebg.casefold()
    ):
        raise BrainNetError(
            "MATLAB exited zero without exact render completion metadata"
        )
    if (
        sha256_file(mapcfg) != mapcfg_hash_before
        or (sha256_file(gui) if gui.is_file() else None) != gui_hash_before
    ):
        raise BrainNetError("BrainNet installation changed during image rendering")
    qa = inspect_png(
        image,
        width_pixels=config.render.width_pixels,
        height_pixels=config.render.height_pixels,
        dpi=config.render.dpi,
    )
    qa["sha256"] = sha256_file(image)
    qa.pop("sha256_pending", None)
    atomic_json(output_dir / "image-qa.json", qa)
    return BrainNetExecution(
        completed.returncode,
        started,
        finished,
        finished - started,
        output_dir / "matlab.stdout.log",
        output_dir / "matlab.stderr.log",
        metadata_path,
        image,
        qa,
    )
