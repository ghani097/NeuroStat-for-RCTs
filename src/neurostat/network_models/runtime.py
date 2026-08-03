"""Noninteractive R runtime preflight and statistical-runner boundary."""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..stage_io import sha256_file
from .config import NetworkModelConfig, RuntimeSettings
from .data import NetworkModelError


@dataclass(frozen=True, slots=True)
class RuntimeMetadata:
    executable: Path
    r_version: str
    library_paths: tuple[str, ...]
    packages: tuple[tuple[str, str], ...]
    contract_sha256: str
    strict_renv: bool
    renv_lockfile_sha256: str | None

    def normalized(self) -> dict[str, Any]:
        return {
            "executable": str(self.executable),
            "r_version": self.r_version,
            "library_paths": list(self.library_paths),
            "packages": [list(item) for item in self.packages],
            "contract_sha256": self.contract_sha256,
            "strict_renv": self.strict_renv,
            "renv_lockfile_sha256": self.renv_lockfile_sha256,
        }


@dataclass(frozen=True, slots=True)
class ModelRunRequest:
    executable: Path
    script: Path
    input_csv: Path
    jobs_csv: Path
    levels_csv: Path
    between_csv: Path
    within_csv: Path
    settings_tsv: Path
    output_dir: Path
    stdout_log: Path
    stderr_log: Path
    timeout_seconds: int
    environment: dict[str, str]


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
StatisticalRunner = Callable[[ModelRunRequest], None]


def _process_detail(stdout: str | None, stderr: str | None) -> str:
    parts = []
    if stderr and stderr.strip():
        parts.append(f"stderr:\n{stderr.strip()}")
    if stdout and stdout.strip():
        parts.append(f"stdout:\n{stdout.strip()}")
    return "\n".join(parts) if parts else "no R output"


def _resources() -> Path:
    return Path(__file__).parent / "resources"


def resolve_rscript(settings: RuntimeSettings) -> Path:
    requested = settings.executable
    if Path(requested).is_absolute() or "/" in requested or "\\" in requested:
        candidate = Path(requested).resolve()
    else:
        found = shutil.which(requested)
        if found is None:
            raise NetworkModelError(
                f"Rscript executable {requested!r} was not found; configure network_models.runtime.executable"
            )
        candidate = Path(found).resolve()
    if not candidate.is_file():
        raise NetworkModelError(f"Rscript executable is not a file: {candidate}")
    return candidate


def r_environment(settings: RuntimeSettings) -> dict[str, str]:
    environment = dict(os.environ)
    if settings.library_paths:
        environment["R_LIBS_USER"] = os.pathsep.join(str(path) for path in settings.library_paths)
    return environment


def _read_dependency_contract(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["package", "comparison", "version"]:
            raise NetworkModelError(
                "R dependency contract header must be: package, comparison, version (tab-separated)"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise NetworkModelError("R dependency contract is empty")
    packages: set[str] = set()
    for row in rows:
        if (
            not row["package"].strip()
            or row["comparison"] not in {">=", "=="}
            or not row["version"].strip()
            or row["package"] in packages
        ):
            raise NetworkModelError("R dependency contract has invalid or duplicate package rows")
        packages.add(row["package"])
    return rows


def _effective_contract(config: NetworkModelConfig, path: Path) -> None:
    rows = _read_dependency_contract(config.runtime.dependency_contract)
    required = {row["package"] for row in rows}
    if config.diagnostics.influence and "influence.ME" not in required:
        rows.append({"package": "influence.ME", "comparison": ">=", "version": "0.9.9"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["package", "comparison", "version"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def preflight_r(
    config: NetworkModelConfig,
    work_dir: Path,
    *,
    process_runner: ProcessRunner = subprocess.run,
) -> tuple[RuntimeMetadata, tuple[Path, ...]]:
    """Resolve R safely, validate packages/lockfile, and capture runtime identity."""
    work_dir.mkdir(parents=True, exist_ok=True)
    executable = resolve_rscript(config.runtime)
    effective_contract = work_dir / "effective-r-package-contract.tsv"
    _effective_contract(config, effective_contract)
    metadata_path = work_dir / "r-runtime.tsv"
    stdout_path = work_dir / "r-preflight.stdout.log"
    stderr_path = work_dir / "r-preflight.stderr.log"
    script = _resources() / "preflight.R"
    args: Sequence[str] = (
        str(executable),
        "--vanilla",
        str(script),
        str(effective_contract),
        str(metadata_path),
        "true" if config.runtime.strict_renv else "false",
        "" if config.runtime.renv_lockfile is None else str(config.runtime.renv_lockfile),
    )
    try:
        completed = process_runner(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.runtime.timeout_seconds,
            env=r_environment(config.runtime),
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NetworkModelError(
            f"R preflight exceeded {config.runtime.timeout_seconds} seconds"
        ) from exc
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise NetworkModelError(
            f"R runtime preflight failed with exit code {completed.returncode}: "
            f"{_process_detail(completed.stdout, completed.stderr)}"
        )
    if not metadata_path.is_file():
        raise NetworkModelError("R preflight succeeded without producing runtime metadata")
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["kind", "name", "value"]:
            raise NetworkModelError("R preflight runtime metadata has an invalid header")
        rows = [dict(row) for row in reader]
    r_versions = [row["value"] for row in rows if row["kind"] == "runtime" and row["name"] == "R"]
    if len(r_versions) != 1:
        raise NetworkModelError("R preflight did not report exactly one R version")
    libraries = tuple(row["value"] for row in rows if row["kind"] == "library")
    packages = tuple(
        sorted(
            (row["name"], row["value"])
            for row in rows
            if row["kind"] == "package"
        )
    )
    expected_packages = {row["package"] for row in _read_dependency_contract(effective_contract)}
    if {name for name, _ in packages} != expected_packages:
        raise NetworkModelError("R preflight package metadata differs from the effective contract")
    metadata = RuntimeMetadata(
        executable,
        r_versions[0],
        libraries,
        packages,
        sha256_file(effective_contract),
        config.runtime.strict_renv,
        config.runtime.renv_lockfile_sha256,
    )
    return metadata, (effective_contract, metadata_path, stdout_path, stderr_path)


def run_r_models(request: ModelRunRequest, *, process_runner: ProcessRunner = subprocess.run) -> None:
    """Invoke the packaged R runner without a shell, preserving all process logs."""
    request.output_dir.mkdir(parents=True, exist_ok=True)
    args: Sequence[str] = (
        str(request.executable),
        "--vanilla",
        str(request.script),
        str(request.input_csv),
        str(request.jobs_csv),
        str(request.levels_csv),
        str(request.between_csv),
        str(request.within_csv),
        str(request.settings_tsv),
        str(request.output_dir),
    )
    try:
        completed = process_runner(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=request.timeout_seconds,
            env=request.environment,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NetworkModelError(
            f"R model runner exceeded {request.timeout_seconds} seconds"
        ) from exc
    request.stdout_log.write_text(completed.stdout or "", encoding="utf-8")
    request.stderr_log.write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise NetworkModelError(
            f"R model runner failed with exit code {completed.returncode}: "
            f"{_process_detail(completed.stdout, completed.stderr)}"
        )
