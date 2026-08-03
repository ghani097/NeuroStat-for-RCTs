"""Atomic, content-addressed network summaries and mixed-model stage."""
from __future__ import annotations

import csv
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..contracts import ProgressEvent, StageName
from ..project import CleanProject, load_project
from ..stage_io import atomic_json, canonical_json, digest_json, load_json, sha256_file
from .config import NetworkModelConfig, parse_network_model_config
from .corrections import apply_correction_families
from .data import (
    NetworkModelError,
    build_network_summaries,
    verify_connectivity_run,
    write_csv,
)
from .runtime import (
    ModelRunRequest,
    RuntimeMetadata,
    StatisticalRunner,
    preflight_r,
    r_environment,
    run_r_models,
)
from .validation import read_csv_rows, validate_statistical_exports

SUMMARY_FIELDS = (
    "analysis_family",
    "condition",
    "record_id",
    "record_sha256",
    "subject_id",
    "arm",
    "session",
    "band",
    "network",
    "raw_pli",
    "adjusted_pli",
    "z_pli",
    "sample_count",
    "sample_count_policy",
    "boundary_policy",
)
COHORT_FIELDS = (
    "analysis_family",
    "condition",
    "subject_id",
    "arm",
    "included",
    "reason",
    "available_sessions",
    "missing_sessions",
)
COUNT_FIELDS = ("analysis_family", "condition", "arm", "session", "subjects")
CORRECTION_FIELDS = (
    "analysis_family",
    "correction_family",
    "target",
    "method",
    "family_size",
    "condition",
    "band",
    "network",
    "term",
    "contrast",
    "arm",
    "session",
    "raw_p_value",
    "adjusted_p_value",
)


@dataclass(frozen=True, slots=True)
class NetworkModelRun:
    run_id: str
    run_dir: Path
    manifest: Path
    reused: bool
    artifacts: tuple[Path, ...]


PreflightFunction = Callable[[NetworkModelConfig, Path], tuple[RuntimeMetadata, tuple[Path, ...]]]


def _resource(name: str) -> Path:
    path = Path(__file__).parent / "resources" / name
    if not path.is_file():
        raise NetworkModelError(f"packaged network-model resource is missing: {path}")
    return path


def _write_carriers(root: Path, project: CleanProject, config: NetworkModelConfig) -> dict[str, Path]:
    jobs = [
        {
            "analysis_family": family.name,
            "condition": family.condition,
            "band": band,
            "network": network,
        }
        for family in config.families
        for band in family.bands
        for network in family.networks
    ]
    levels = [
        {
            "analysis_family": family.name,
            "dimension": dimension,
            "position": position,
            "value": value,
        }
        for family in config.families
        for dimension, values in (("arm", project.study.arms), ("session", family.sessions))
        for position, value in enumerate(values, 1)
    ]
    between = [
        {"analysis_family": family.name, **contrast.normalized()}
        for family in config.families
        for contrast in family.between_arm_contrasts
    ]
    within = [
        {
            "analysis_family": family.name,
            "name": contrast.name,
            "arm": arm,
            "numerator_session": contrast.numerator_session,
            "denominator_session": contrast.denominator_session,
        }
        for family in config.families
        for contrast in family.within_arm_contrasts
        for arm in contrast.arms
    ]
    paths = {
        "jobs": root / "model-jobs.csv",
        "levels": root / "factor-levels.csv",
        "between": root / "between-contrast-spec.csv",
        "within": root / "within-contrast-spec.csv",
        "settings": root / "model-settings.tsv",
    }
    write_csv(paths["jobs"], jobs, ("analysis_family", "condition", "band", "network"))
    write_csv(paths["levels"], levels, ("analysis_family", "dimension", "position", "value"))
    write_csv(
        paths["between"],
        between,
        ("analysis_family", "name", "session", "numerator_arm", "denominator_arm"),
    )
    write_csv(
        paths["within"],
        within,
        ("analysis_family", "name", "arm", "numerator_session", "denominator_session"),
    )
    settings = {
        "reml": str(config.model.reml).lower(),
        "optimizer": config.model.optimizer,
        "maxfun": str(config.model.maxfun),
        "residuals": str(config.diagnostics.residuals).lower(),
        "raw_scale": str(config.diagnostics.raw_scale).lower(),
        "winsorized": str(config.diagnostics.winsorized).lower(),
        "winsor_robust_z": format(config.diagnostics.winsor_robust_z, ".17g"),
        "all_fit": str(config.diagnostics.all_fit).lower(),
        "influence": str(config.diagnostics.influence).lower(),
        "leave_one_out": str(config.diagnostics.leave_one_out).lower(),
    }
    with paths["settings"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("name", "value"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows({"name": name, "value": value} for name, value in settings.items())
    return paths


def _all_files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "network_models/manifest.json":
            continue
        result[relative] = sha256_file(path)
    return result


def verify_network_model_run(run_dir: str | Path, expected_lock: dict[str, Any] | None = None) -> NetworkModelRun:
    """Verify the immutable lock, exact file set, and every frozen SHA-256."""
    run_dir = Path(run_dir).resolve()
    lock_path = run_dir / "project.lock.json"
    manifest_path = run_dir / "network_models" / "manifest.json"
    lock = load_json(lock_path, "network-model project lock")
    manifest = load_json(manifest_path, "network-model manifest")
    if expected_lock is not None and canonical_json(lock) != canonical_json(expected_lock):
        raise NetworkModelError("existing network-model run lock differs from the requested scientific identity")
    if (
        not isinstance(lock, dict)
        or lock.get("stage") != "network_models"
        or not isinstance(manifest, dict)
        or manifest.get("schema") != "neurostat.network_models_manifest.v1"
        or manifest.get("run_id") != lock.get("run_id")
        or manifest.get("scientific_identity_sha256") != digest_json(
            {key: value for key, value in lock.items() if key != "run_id"}
        )
        or not isinstance(manifest.get("artifacts"), dict)
    ):
        raise NetworkModelError("network-model lock or manifest identity is invalid")
    expected_artifacts = manifest["artifacts"]
    actual_artifacts = _all_files(run_dir)
    if set(actual_artifacts) != set(expected_artifacts):
        raise NetworkModelError("network-model run file set differs from its immutable manifest")
    for relative, expected_sha in expected_artifacts.items():
        candidate = (run_dir / relative).resolve()
        try:
            candidate.relative_to(run_dir)
        except ValueError:
            raise NetworkModelError(f"unsafe artifact path in network-model manifest: {relative}") from None
        if not isinstance(expected_sha, str) or actual_artifacts[relative] != expected_sha:
            raise NetworkModelError(f"network-model artifact SHA-256 mismatch: {relative}")
    status = load_json(run_dir / "network_models" / "status.json", "network-model status")
    validation = load_json(
        run_dir / "network_models" / "validation.json",
        "network-model independent validation",
    )
    if status.get("state") != "succeeded" or validation.get("status") != "passed":
        raise NetworkModelError("network-model run is not successfully validated")
    artifact_paths = tuple(run_dir / relative for relative in sorted(expected_artifacts))
    return NetworkModelRun(str(lock["run_id"]), run_dir, manifest_path, True, artifact_paths)


def run_network_models(
    project: CleanProject | str | Path,
    connectivity_run: str | Path,
    progress: Callable[[ProgressEvent], None] | None = None,
    *,
    statistical_runner: StatisticalRunner | None = None,
    preflight_function: PreflightFunction | None = None,
) -> NetworkModelRun:
    """Run exact summaries, R mixed models, corrections, and independent QA."""
    project = load_project(project) if not isinstance(project, CleanProject) else project
    config = parse_network_model_config(project)
    upstream = verify_connectivity_run(project, connectivity_run)
    project.output_root.mkdir(parents=True, exist_ok=True)
    runs_root = project.output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(ProgressEvent(StageName.NETWORK_MODELS, "validated connectivity and linked scout run", 1, 5))

    with tempfile.TemporaryDirectory(prefix=".network-model-preflight-", dir=runs_root) as preflight_name:
        preflight_dir = Path(preflight_name)
        if preflight_function is None:
            runtime_metadata, preflight_files = preflight_r(config, preflight_dir)
        else:
            runtime_metadata, preflight_files = preflight_function(config, preflight_dir)
        resources = {
            name: {"sha256": sha256_file(_resource(name))}
            for name in ("preflight.R", "run_network_models.R", "r-package-contract.tsv")
        }
        identity = {
            "stage": "network_models",
            "project": project.normalized(),
            "config": config.normalized(),
            "upstream": {
                "stage": "connectivity",
                "run_id": upstream.run_id,
                "scientific_identity_sha256": upstream.stable_identity_sha256,
                "manifest": upstream.stable_identity,
            },
            "runtime": runtime_metadata.normalized(),
            "resources": resources,
        }
        scientific_sha = digest_json(identity)
        run_id = scientific_sha[:12]
        lock = identity | {"run_id": run_id}
        final_dir = (runs_root / f"network_models-{run_id}").resolve()
        if final_dir.exists():
            result = verify_network_model_run(final_dir, lock)
            if progress:
                progress(ProgressEvent(StageName.NETWORK_MODELS, "reused verified network-model run", 5, 5))
            return result

        staging = Path(tempfile.mkdtemp(prefix=f".network_models-{run_id}.staging-", dir=runs_root)).resolve()
        try:
            stage_dir = staging / "network_models"
            inputs_dir = stage_dir / "inputs"
            resources_dir = stage_dir / "resources"
            runtime_dir = stage_dir / "runtime"
            r_output_dir = stage_dir / "r_outputs"
            logs_dir = stage_dir / "logs"
            for directory in (inputs_dir, resources_dir, runtime_dir, r_output_dir, logs_dir):
                directory.mkdir(parents=True, exist_ok=True)
            atomic_json(staging / "project.lock.json", lock)
            atomic_json(stage_dir / "config.json", config.normalized())
            atomic_json(stage_dir / "status.json", {"state": "running", "completed": 1, "total": 5})
            for resource_name in resources:
                shutil.copy2(_resource(resource_name), resources_dir / resource_name)
            for source in preflight_files:
                if not source.is_file():
                    raise NetworkModelError(f"preflight did not produce declared file: {source}")
                shutil.copy2(source, runtime_dir / source.name)

            summaries = build_network_summaries(project, config, upstream)
            summary_path = stage_dir / "network_summaries.csv"
            cohort_path = stage_dir / "cohort_inclusion.csv"
            count_path = stage_dir / "cohort_counts.csv"
            write_csv(summary_path, summaries.rows, SUMMARY_FIELDS)
            write_csv(cohort_path, summaries.cohort_rows, COHORT_FIELDS)
            write_csv(count_path, summaries.count_rows, COUNT_FIELDS)
            carriers = _write_carriers(inputs_dir, project, config)
            atomic_json(stage_dir / "status.json", {"state": "running", "completed": 2, "total": 5})
            if progress:
                progress(ProgressEvent(StageName.NETWORK_MODELS, "built complete-cohort network summaries", 2, 5))

            request = ModelRunRequest(
                runtime_metadata.executable,
                resources_dir / "run_network_models.R",
                summary_path,
                carriers["jobs"],
                carriers["levels"],
                carriers["between"],
                carriers["within"],
                carriers["settings"],
                r_output_dir,
                logs_dir / "r-models.stdout.log",
                logs_dir / "r-models.stderr.log",
                config.runtime.timeout_seconds,
                r_environment(config.runtime),
            )
            if statistical_runner is None:
                run_r_models(request)
            else:
                statistical_runner(request)
            # A custom runner must still honor the same logging contract.
            for log in (request.stdout_log, request.stderr_log):
                if not log.exists():
                    log.write_text("", encoding="utf-8")
            atomic_json(stage_dir / "status.json", {"state": "running", "completed": 3, "total": 5})
            if progress:
                progress(ProgressEvent(StageName.NETWORK_MODELS, "completed parameterized mixed models", 3, 5))

            between_rows = read_csv_rows(r_output_dir / "between_arm_contrasts.csv")
            within_rows = read_csv_rows(r_output_dir / "within_arm_contrasts.csv")
            anova_rows = read_csv_rows(r_output_dir / "anova_type3.csv")
            correction_rows = apply_correction_families(
                config,
                anova_rows,
                between_rows,
                within_rows,
            )
            correction_path = stage_dir / "multiple_testing.csv"
            write_csv(correction_path, correction_rows, CORRECTION_FIELDS)
            validation = validate_statistical_exports(
                project,
                config,
                upstream,
                summaries.rows,
                summaries.count_rows,
                r_output_dir,
                correction_rows,
            )
            atomic_json(stage_dir / "validation.json", validation)
            atomic_json(stage_dir / "status.json", {"state": "running", "completed": 4, "total": 5})
            if progress:
                progress(ProgressEvent(StageName.NETWORK_MODELS, "passed independent estimand validation", 4, 5))

            atomic_json(
                stage_dir / "status.json",
                {
                    "state": "succeeded",
                    "completed": 5,
                    "total": 5,
                    "model_jobs": validation["model_jobs"],
                },
            )
            artifacts = _all_files(staging)
            atomic_json(
                stage_dir / "manifest.json",
                {
                    "schema": "neurostat.network_models_manifest.v1",
                    "run_id": run_id,
                    "upstream_run_id": upstream.run_id,
                    "scientific_identity_sha256": scientific_sha,
                    "artifacts": artifacts,
                },
            )
            if final_dir.exists():
                raise NetworkModelError(f"network-model run appeared concurrently: {final_dir}")
            os.replace(staging, final_dir)
            result = verify_network_model_run(final_dir, lock)
            if progress:
                progress(ProgressEvent(StageName.NETWORK_MODELS, "finalized immutable network-model run", 5, 5))
            return NetworkModelRun(result.run_id, result.run_dir, result.manifest, False, result.artifacts)
        except BaseException:
            # The staging directory is private to this invocation and has never
            # been exposed as a completed run.
            if staging.exists():
                try:
                    staging.relative_to(runs_root.resolve())
                except ValueError:
                    raise NetworkModelError(f"refusing to remove unsafe staging path: {staging}") from None
                shutil.rmtree(staging)
            raise
