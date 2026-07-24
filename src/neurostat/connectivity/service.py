"""Deterministic connectivity stage over canonical scout artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..analysis_profiles import (
    ContinuousMultitaperConnectivity,
    EvokedHilbertConnectivity,
    project_analysis_profiles,
)
from ..contracts import ProgressEvent, StageName
from ..project import CleanProject, load_project
from ..scouts.artifacts import verify_scout_artifact
from ..scouts.service import ScoutRun
from ..stage_io import atomic_json, canonical_json, digest_json, load_json, sha256_file
from .artifacts import ConnectivityHeader, verify_connectivity_artifact, write_connectivity_artifact
from .methods import (
    ConnectivityError,
    continuous_multitaper_pli,
    evoked_average_hilbert_pli,
)


@dataclass(frozen=True, slots=True)
class ConnectivityRun:
    run_id: str
    run_dir: Path
    manifest: Path
    reused: int
    written: int
    records: tuple[Path, ...]


def _scout_records(project: CleanProject, run_dir: Path, profiles: dict) -> tuple[str, str, dict[str, dict[str, Any]]]:
    run_dir = run_dir.resolve()
    lock_path = run_dir / "project.lock.json"
    manifest_path = run_dir / "scouts" / "manifest.json"
    lock = load_json(lock_path, "scout project lock")
    manifest = load_json(manifest_path, "scout manifest")
    expected_profiles = {name: profile.normalized() for name, profile in profiles.items()}
    if (
        not isinstance(lock, dict)
        or lock.get("stage") != "scouts"
        or canonical_json(lock.get("project")) != canonical_json(project.normalized())
        or canonical_json(lock.get("profiles")) != canonical_json(expected_profiles)
    ):
        raise ConnectivityError("scout run belongs to a different normalized project or profile resources")
    if not isinstance(manifest, dict) or manifest.get("schema") != "neurostat.scout_manifest.v1":
        raise ConnectivityError("invalid scout manifest schema")
    entries = manifest.get("records")
    if not isinstance(entries, list) or {item.get("record_id") for item in entries if isinstance(item, dict)} != {record.record_id for record in project.records}:
        raise ConnectivityError("scout manifest records do not exactly match the project")
    by_id = {item["record_id"]: item for item in entries}
    for record in project.records:
        entry = by_id[record.record_id]
        expected_path = (run_dir / "scouts" / "records" / f"{record.record_id}.h5").resolve()
        try:
            actual = Path(entry["path"]).resolve()
        except (KeyError, TypeError):
            raise ConnectivityError(f"{record.record_id}: invalid scout artifact path") from None
        if actual != expected_path or not actual.is_file():
            raise ConnectivityError(f"{record.record_id}: scout artifact is outside or missing from the scout run")
        digest = sha256_file(actual)
        if entry.get("sha256") != digest:
            raise ConnectivityError(f"{record.record_id}: scout SHA-256 differs from the scout manifest")
        profile = profiles[record.analysis_profile]
        header = verify_scout_artifact(
            actual,
            {
                "record_id": record.record_id,
                "participant_id": record.key.participant_id,
                "arm": record.key.arm,
                "session": record.key.session,
                "condition": record.key.condition,
                "analysis_profile": record.analysis_profile,
                "source_backend": profile.source.backend,
                "profile_digest": profile.digest,
            },
        )
        if (
            entry.get("payload_digest") != header.payload_digest
            or entry.get("dimensions") != [len(header.labels), header.samples]
            or entry.get("labels") != list(header.labels)
        ):
            raise ConnectivityError(f"{record.record_id}: scout manifest metadata differs from its artifact")
        entry["header"] = header
    stable_manifest = {
        "schema": manifest["schema"],
        "run_id": manifest["run_id"],
        "upstream_run_id": manifest.get("upstream_run_id"),
        "records": [
            {key: value for key, value in item.items() if key not in {"reused", "header"}}
            for item in entries
        ],
    }
    return str(manifest["run_id"]), digest_json(stable_manifest), by_id


def run_connectivity(
    project: CleanProject | str | Path,
    scouts_run: ScoutRun | str | Path,
    progress: Callable[[ProgressEvent], None] | None = None,
    *,
    connectivity_function: Callable[..., Any] | None = None,
) -> ConnectivityRun:
    """Create or validate one deterministic connectivity run."""
    project = load_project(project) if not isinstance(project, CleanProject) else project
    profiles = project_analysis_profiles(project)
    upstream_dir = scouts_run.run_dir if isinstance(scouts_run, ScoutRun) else Path(scouts_run)
    upstream_id, upstream_manifest_sha, scout_entries = _scout_records(project, upstream_dir, profiles)
    frozen = {
        "stage": "connectivity",
        "project": project.normalized(),
        "profiles": {name: profile.normalized() for name, profile in profiles.items()},
        "upstream": {
            "stage": "scouts",
            "run_id": upstream_id,
            "manifest_content_sha256": upstream_manifest_sha,
            "records": {record_id: entry["sha256"] for record_id, entry in sorted(scout_entries.items())},
        },
    }
    run_id = digest_json(frozen)[:12]
    run_dir = project.output_root / "runs" / f"connectivity-{run_id}"
    lock_path = run_dir / "project.lock.json"
    manifest_path = run_dir / "connectivity" / "manifest.json"
    status_path = run_dir / "connectivity" / "status.json"
    if lock_path.exists() and canonical_json(load_json(lock_path, "connectivity project lock")) != canonical_json(frozen):
        raise ConnectivityError(f"existing lock differs for deterministic connectivity run {run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not lock_path.exists():
        atomic_json(lock_path, frozen)
    atomic_json(status_path, {"state": "running", "total": len(project.records), "completed": 0})
    prior_manifest = load_json(manifest_path, "connectivity manifest") if manifest_path.exists() else None
    prior_by_id = {
        item["record_id"]: item
        for item in prior_manifest.get("records", [])
        if isinstance(prior_manifest, dict) and isinstance(item, dict) and "record_id" in item
    } if prior_manifest else {}
    entries: list[dict[str, Any]] = []
    paths: list[Path] = []
    reused = written = 0
    try:
        for completed, record in enumerate(project.records, 1):
            profile = profiles[record.analysis_profile]
            config = profile.connectivity
            scout_entry = scout_entries[record.record_id]
            scout_path = Path(scout_entry["path"]).resolve()
            scout_header = scout_entry["header"]
            target = run_dir / "connectivity" / "records" / f"{record.record_id}.h5"
            expected = {
                "record_id": record.record_id,
                "participant_id": record.key.participant_id,
                "arm": record.key.arm,
                "session": record.key.session,
                "condition": record.key.condition,
                "analysis_profile": record.analysis_profile,
                "connectivity_backend": config.backend,
                "sfreq": scout_header.sfreq,
                "scout_sha256": scout_entry["sha256"],
                "scout_payload_digest": scout_header.payload_digest,
                "profile_digest": profile.digest,
            }
            if target.exists():
                try:
                    header = verify_connectivity_artifact(target, expected)
                    digest = sha256_file(target)
                    prior = prior_by_id.get(record.record_id)
                    if prior is not None and prior.get("sha256") != digest:
                        raise ConnectivityError("connectivity SHA-256 differs from the frozen manifest")
                except Exception as exc:
                    raise ConnectivityError(f"refusing to overwrite mismatched or corrupt artifact {target}: {exc}") from exc
                reused += 1
            else:
                if isinstance(config, EvokedHilbertConnectivity):
                    import h5py

                    with h5py.File(scout_path, "r") as file:
                        values = evoked_average_hilbert_pli(file["data"], scout_header.sfreq, config)
                elif isinstance(config, ContinuousMultitaperConnectivity):
                    values = continuous_multitaper_pli(
                        scout_path,
                        scout_header.labels,
                        scout_header.sfreq,
                        scout_header.samples,
                        config,
                        connectivity_function=connectivity_function,
                    )
                else:  # pragma: no cover - exhaustive profile contract
                    raise ConnectivityError(f"unsupported connectivity backend: {config}")
                digest, header = write_connectivity_artifact(
                    target,
                    values,
                    scout_header.labels,
                    config.bands,
                    expected,
                )
                written += 1
            entry = {
                "record_id": record.record_id,
                "path": str(target),
                "sha256": digest,
                "payload_digest": header.payload_digest,
                "dimensions": [len(header.bands), len(header.labels), len(header.labels)],
                "labels": list(header.labels),
                "bands": list(header.bands),
                "analysis_profile": record.analysis_profile,
                "connectivity_backend": header.connectivity_backend,
                "scout_sha256": scout_entry["sha256"],
            }
            entries.append(entry)
            paths.append(target)
            atomic_json(status_path, {"state": "running", "total": len(project.records), "completed": completed})
            if progress:
                progress(ProgressEvent(StageName.CONNECTIVITY, f"generated connectivity for {record.record_id}", completed, len(project.records)))
        atomic_json(manifest_path, {"schema": "neurostat.connectivity_manifest.v1", "run_id": run_id, "upstream_run_id": upstream_id, "records": entries})
        atomic_json(status_path, {"state": "succeeded", "total": len(project.records), "completed": len(project.records), "reused": reused, "written": written})
    except BaseException as exc:
        atomic_json(status_path, {"state": "failed", "total": len(project.records), "completed": len(entries), "error": str(exc)})
        raise
    return ConnectivityRun(run_id, run_dir, manifest_path, reused, written, tuple(paths))
