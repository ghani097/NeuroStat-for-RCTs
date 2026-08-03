"""MATLAB/GraphVar 2.03a preflight and noninteractive process boundary."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..stage_io import atomic_json, load_json, sha256_file
from .config import GraphVarConfig
from .workspace import GraphVarError

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class GraphVarRuntimeMetadata:
    executable: Path
    matlab_release: str
    graphvar_release: str
    graphvar_root: Path
    graphvar_glm: Path
    graphvar_glm_sha256: str
    start_graphvar_sha256: str
    runner_sha256: str

    def normalized(self) -> dict[str, Any]:
        return {
            "executable": str(self.executable),
            "matlab_release": self.matlab_release,
            "graphvar_release": self.graphvar_release,
            "graphvar_root": str(self.graphvar_root),
            "graphvar_glm": str(self.graphvar_glm),
            "graphvar_glm_sha256": self.graphvar_glm_sha256,
            "start_graphvar_sha256": self.start_graphvar_sha256,
            "runner_sha256": self.runner_sha256,
        }


@dataclass(frozen=True, slots=True)
class MatlabExecution:
    returncode: int
    started_unix: float
    finished_unix: float
    elapsed_seconds: float
    stdout: Path
    stderr: Path
    metadata: Path
    result: Path


def _resource() -> Path:
    path = Path(__file__).parent / "matlab" / "neurostat_graphvar_entry.m"
    if not path.is_file():
        raise GraphVarError(f"packaged GraphVar MATLAB runner is missing: {path}")
    return path


def resolve_matlab(config: GraphVarConfig) -> Path:
    requested = config.runtime.executable
    if Path(requested).is_absolute() or "/" in requested or "\\" in requested:
        candidate = Path(requested).resolve()
    else:
        found = shutil.which(requested)
        if found is None:
            raise GraphVarError(
                f"MATLAB executable {requested!r} was not found; configure graphvar.runtime.executable"
            )
        candidate = Path(found).resolve()
    if not candidate.is_file():
        raise GraphVarError(f"MATLAB executable is not a file: {candidate}")
    return candidate


def _detail(completed: subprocess.CompletedProcess[str]) -> str:
    values = []
    if completed.stderr and completed.stderr.strip():
        values.append(f"stderr:\n{completed.stderr.strip()}")
    if completed.stdout and completed.stdout.strip():
        values.append(f"stdout:\n{completed.stdout.strip()}")
    return "\n".join(values) if values else "no MATLAB output"


def _invoke(
    config: GraphVarConfig,
    request_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    *,
    process_runner: ProcessRunner,
) -> subprocess.CompletedProcess[str]:
    executable = resolve_matlab(config)
    environment = dict(os.environ)
    environment["NEUROSTAT_GRAPHVAR_REQUEST"] = str(request_path.resolve())
    environment["NEUROSTAT_GRAPHVAR_ENTRY"] = str(_resource().resolve())
    args: Sequence[str] = (
        str(executable),
        "-batch",
        "run(getenv('NEUROSTAT_GRAPHVAR_ENTRY'))",
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
        timeout_stdout = exc.stdout or ""
        timeout_stderr = exc.stderr or ""
        if isinstance(timeout_stdout, bytes):
            timeout_stdout = timeout_stdout.decode("utf-8", errors="replace")
        if isinstance(timeout_stderr, bytes):
            timeout_stderr = timeout_stderr.decode("utf-8", errors="replace")
        stdout_path.write_text(timeout_stdout, encoding="utf-8")
        stderr_path.write_text(timeout_stderr, encoding="utf-8")
        raise GraphVarError(
            f"MATLAB/GraphVar exceeded {config.runtime.timeout_seconds} seconds"
        ) from exc
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise GraphVarError(
            f"MATLAB/GraphVar failed with exit code {completed.returncode}: {_detail(completed)}"
        )
    return completed


def preflight_graphvar(
    config: GraphVarConfig,
    output_dir: Path,
    *,
    process_runner: ProcessRunner = subprocess.run,
) -> GraphVarRuntimeMetadata:
    """Verify exact MATLAB release, GraphVar release/source resolution, and hashes."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    graphvar_glm = (
        config.runtime.graphvar_root / "src" / "calc" / "graphvar_glm.m"
    ).resolve()
    start_graphvar = (config.runtime.graphvar_root / "start_GraphVar.m").resolve()
    if not graphvar_glm.is_file() or not start_graphvar.is_file():
        raise GraphVarError(
            "GraphVar root lacks start_GraphVar.m or src/calc/graphvar_glm.m"
        )
    start_text = start_graphvar.read_text(encoding="utf-8", errors="replace")
    if "Release= 2.03a" not in start_text:
        raise GraphVarError("configured GraphVar installation is not release 2.03a")
    metadata_path = output_dir / "preflight.json"
    request_path = output_dir / "request.json"
    atomic_json(
        request_path,
        {
            "mode": "preflight",
            "graphvar_root": str(config.runtime.graphvar_root),
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
    metadata = load_json(metadata_path, "MATLAB/GraphVar preflight metadata")
    expected_glm = str(graphvar_glm).replace("\\", "/")
    resolved_glm = str(metadata.get("graphvar_glm", "")).replace("\\", "/")
    if (
        metadata.get("matlab_release") != config.runtime.matlab_release
        or metadata.get("graphvar_release") != "2.03a"
        or resolved_glm.casefold() != expected_glm.casefold()
        or metadata.get("statistics_toolbox") is not True
        or metadata.get("required_functions")
        != ["jsondecode", "readcell", "rng", "regress", "fcdf"]
    ):
        raise GraphVarError(
            "MATLAB release, GraphVar 2.03a release, or exact graphvar_glm.m resolution differs"
        )
    observed_glm_hash = sha256_file(graphvar_glm)
    observed_start_hash = sha256_file(start_graphvar)
    if (
        metadata.get("graphvar_glm_sha256") != observed_glm_hash
        or metadata.get("start_graphvar_sha256") != observed_start_hash
    ):
        raise GraphVarError(
            "MATLAB preflight source hashes differ from Python verification"
        )
    return GraphVarRuntimeMetadata(
        resolve_matlab(config),
        config.runtime.matlab_release,
        "2.03a",
        config.runtime.graphvar_root,
        graphvar_glm,
        observed_glm_hash,
        observed_start_hash,
        sha256_file(_resource()),
    )


def run_graphvar_workspace(
    config: GraphVarConfig,
    workspace: Path,
    request: dict[str, Any],
    output_dir: Path,
    *,
    process_runner: ProcessRunner = subprocess.run,
) -> MatlabExecution:
    """Execute one request and reject false success, missing, or malformed results."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = output_dir / "corr_area_1_Group.mat"
    metadata = output_dir / "execution.json"
    request_path = output_dir / "request.json"
    carrier = {
        "mode": "run",
        "graphvar_root": str(config.runtime.graphvar_root),
        "workspace": str(workspace.resolve()),
        "result_path": str(result.resolve()),
        "metadata_path": str(metadata.resolve()),
        "seed": int(request["seed"]),
        "permutations": int(request["permutations"]),
        "atlas_indices_1based": [
            node.atlas_index_1based for node in config.atlas.nodes
        ],
        "expected_effect_label": request["effect_label"],
    }
    atomic_json(request_path, carrier)
    started = time.time()
    completed = _invoke(
        config,
        request_path,
        output_dir / "matlab.stdout.log",
        output_dir / "matlab.stderr.log",
        process_runner=process_runner,
    )
    finished = time.time()
    if not result.is_file() or result.stat().st_size == 0:
        raise GraphVarError(
            "MATLAB exited successfully without a nonempty GraphVar result"
        )
    run_metadata = load_json(metadata, "GraphVar execution metadata")
    if (
        run_metadata.get("state") != "succeeded"
        or run_metadata.get("seed") != carrier["seed"]
        or run_metadata.get("permutations") != carrier["permutations"]
        or run_metadata.get("result_sha256") != sha256_file(result)
    ):
        raise GraphVarError(
            "GraphVar execution metadata/result hash is missing or inconsistent"
        )
    return MatlabExecution(
        completed.returncode,
        started,
        finished,
        finished - started,
        output_dir / "matlab.stdout.log",
        output_dir / "matlab.stderr.log",
        metadata,
        result,
    )
