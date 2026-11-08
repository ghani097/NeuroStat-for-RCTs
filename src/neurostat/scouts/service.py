"""Deterministic source-scout stage over canonical clean recordings."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..io.formats import RAW_READER_FORMATS
from ..analysis_profiles import (
    FixedLinearSource,
    MNEFsaverageSource,
    project_analysis_profiles,
)
from ..contracts import ProgressEvent, StageName
from ..ingest.service import IngestRun, source_bundle
from ..project import CleanProject, load_project
from ..stage_io import atomic_json, canonical_json, digest_json, load_json, sha256_file
from .artifacts import (
    ScoutError,
    verify_matlab_export,
    verify_scout_artifact,
    write_matlab_export,
    write_scout_artifact,
)
from .fixed import project_fixed
from .mne_backend import DefaultMNEProjector, MNEProjector


@dataclass(frozen=True, slots=True)
class ScoutRun:
    run_id: str
    run_dir: Path
    manifest: Path
    reused: int
    written: int
    records: tuple[Path, ...]


def _ingest_records(project: CleanProject, run_dir: Path) -> tuple[str, str, dict[str, dict]]:
    run_dir = run_dir.resolve()
    lock_path = run_dir / "project.lock.json"
    manifest_path = run_dir / "ingest" / "manifest.json"
    lock = load_json(lock_path, "ingest project lock")
    manifest = load_json(manifest_path, "ingest manifest")
    if not isinstance(lock, dict) or canonical_json(lock.get("project")) != canonical_json(project.normalized()):
        raise ScoutError("ingest run belongs to a different normalized project")
    if not isinstance(manifest, dict) or manifest.get("schema") != "neurostat.ingest_manifest.v1":
        raise ScoutError("invalid ingest manifest schema")
    entries = manifest.get("records")
    if not isinstance(entries, list) or {item.get("record_id") for item in entries if isinstance(item, dict)} != {record.record_id for record in project.records}:
        raise ScoutError("ingest manifest records do not exactly match the project")
    by_id = {item["record_id"]: item for item in entries}
    frozen_bundles = lock.get("source_bundles")
    if not isinstance(frozen_bundles, dict):
        raise ScoutError("ingest lock has no source bundles")
    import h5py

    for record in project.records:
        bundle = source_bundle(record)
        if frozen_bundles.get(record.record_id, {}).get("digest") != bundle.digest:
            raise ScoutError(f"{record.record_id}: clean source hash differs from the ingest lock")
        entry = by_id[record.record_id]
        expected = (run_dir / "ingest" / "records" / f"{record.record_id}.h5").resolve()
        try:
            actual = Path(entry["path"]).resolve()
        except (KeyError, TypeError):
            raise ScoutError(f"{record.record_id}: invalid canonical artifact path") from None
        if actual != expected or not actual.is_file():
            raise ScoutError(f"{record.record_id}: canonical artifact is outside or missing from the ingest run")
        if entry.get("sha256") != sha256_file(actual):
            raise ScoutError(f"{record.record_id}: canonical SHA-256 differs from the ingest manifest")
        try:
            with h5py.File(actual, "r") as file:
                if (
                    file.attrs.get("schema") != "neurostat.clean_recording.v1"
                    or file.attrs.get("record_id") != record.record_id
                    or file.attrs.get("source_bundle_digest") != bundle.digest
                    or file.attrs.get("analysis_profile") != record.analysis_profile
                    or tuple(file["data"].shape) != tuple(entry.get("dimensions", ()))
                    or "payload_digest" not in file.attrs
                ):
                    raise ScoutError(f"{record.record_id}: canonical metadata differs from the ingest manifest")
                entry["payload_digest"] = str(file.attrs["payload_digest"])
        except OSError as exc:
            raise ScoutError(f"{record.record_id}: corrupt canonical artifact") from exc
    stable_manifest = {
        "schema": manifest["schema"],
        "run_id": manifest["run_id"],
        "records": [
            {key: value for key, value in item.items() if key != "reused"}
            for item in entries
        ],
    }
    return str(manifest["run_id"]), digest_json(stable_manifest), by_id


def run_scouts(
    project: CleanProject | str | Path,
    ingest_run: IngestRun | str | Path,
    progress: Callable[[ProgressEvent], None] | None = None,
    *,
    mne_projector: MNEProjector | None = None,
) -> ScoutRun:
    """Create or validate one deterministic scout run."""
    project = load_project(project) if not isinstance(project, CleanProject) else project
    profiles = project_analysis_profiles(project)
    upstream_dir = ingest_run.run_dir if isinstance(ingest_run, IngestRun) else Path(ingest_run)
    upstream_id, upstream_manifest_sha, ingest_entries = _ingest_records(project, upstream_dir)
    frozen = {
        "stage": "scouts",
        "project": project.normalized(),
        "profiles": {name: profile.normalized() for name, profile in profiles.items()},
        "upstream": {
            "stage": "ingest",
            "run_id": upstream_id,
            "manifest_content_sha256": upstream_manifest_sha,
            "records": {record_id: entry["sha256"] for record_id, entry in sorted(ingest_entries.items())},
        },
    }
    run_id = digest_json(frozen)[:12]
    run_dir = project.output_root / "runs" / f"scouts-{run_id}"
    lock_path = run_dir / "project.lock.json"
    manifest_path = run_dir / "scouts" / "manifest.json"
    status_path = run_dir / "scouts" / "status.json"
    if lock_path.exists() and canonical_json(load_json(lock_path, "scout project lock")) != canonical_json(frozen):
        raise ScoutError(f"existing lock differs for deterministic scout run {run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not lock_path.exists():
        atomic_json(lock_path, frozen)
    atomic_json(status_path, {"state": "running", "total": len(project.records), "completed": 0})
    prior_manifest = load_json(manifest_path, "scout manifest") if manifest_path.exists() else None
    prior_by_id = {
        item["record_id"]: item
        for item in prior_manifest.get("records", [])
        if isinstance(prior_manifest, dict) and isinstance(item, dict) and "record_id" in item
    } if prior_manifest else {}
    projector = mne_projector or DefaultMNEProjector()
    entries: list[dict] = []
    paths: list[Path] = []
    reused = written = 0
    try:
        for completed, record in enumerate(project.records, 1):
            profile = profiles[record.analysis_profile]
            config = profile.source
            if isinstance(config, FixedLinearSource) and record.data_kind != "epochs":
                raise ScoutError(f"{record.record_id}: fixed_linear_operator requires data_kind epochs")
            if isinstance(config, MNEFsaverageSource):
                if record.data_kind != "raw":
                    raise ScoutError(f"{record.record_id}: mne_fsaverage requires data_kind raw")
                if record.format not in RAW_READER_FORMATS:
                    raise ScoutError(
                        f"{record.record_id}: mne_fsaverage requires an MNE-ingested continuous recording "
                        "because FieldTrip coordinate units are not canonicalized"
                    )
            canonical_entry = ingest_entries[record.record_id]
            canonical_path = Path(canonical_entry["path"]).resolve()
            target = run_dir / "scouts" / "records" / f"{record.record_id}.h5"
            expected = {
                "record_id": record.record_id,
                "participant_id": record.key.participant_id,
                "arm": record.key.arm,
                "session": record.key.session,
                "condition": record.key.condition,
                "analysis_profile": record.analysis_profile,
                "source_backend": config.backend,
                "canonical_sha256": canonical_entry["sha256"],
                "canonical_payload_digest": canonical_entry["payload_digest"],
                "profile_digest": profile.digest,
            }
            if target.exists():
                try:
                    header = verify_scout_artifact(target, expected)
                    prior = prior_by_id.get(record.record_id)
                    digest = sha256_file(target)
                    if prior is not None and prior.get("sha256") != digest:
                        raise ScoutError("scout SHA-256 differs from the frozen manifest")
                except Exception as exc:
                    raise ScoutError(f"refusing to overwrite mismatched or corrupt artifact {target}: {exc}") from exc
                reused += 1
            else:
                if isinstance(config, FixedLinearSource):
                    stream = project_fixed(canonical_path, config)
                elif isinstance(config, MNEFsaverageSource):
                    stream = projector.project(canonical_path, config)
                else:  # pragma: no cover - exhaustive profile contract
                    raise ScoutError(f"unsupported source backend: {config}")
                digest, header = write_scout_artifact(
                    target,
                    stream,
                    expected,
                    chunk_samples=config.output_chunk_samples,
                )
                written += 1
            matlab_entry = None
            if config.export_matlab:
                matlab_path = run_dir / "scouts" / "matlab" / f"{record.record_id}.mat"
                prior_matlab = prior_by_id.get(record.record_id, {}).get("matlab")
                if matlab_path.exists():
                    verify_matlab_export(matlab_path, target)
                    matlab_sha = sha256_file(matlab_path)
                    if prior_matlab and prior_matlab.get("sha256") != matlab_sha:
                        raise ScoutError("MATLAB export SHA-256 differs from the frozen manifest")
                else:
                    matlab_sha = write_matlab_export(matlab_path, target)
                matlab_entry = {"path": str(matlab_path), "sha256": matlab_sha}
            entry = {
                "record_id": record.record_id,
                "path": str(target),
                "sha256": digest,
                "payload_digest": header.payload_digest,
                "dimensions": [len(header.labels), header.samples],
                "labels": list(header.labels),
                "sfreq": header.sfreq,
                "analysis_profile": record.analysis_profile,
                "source_backend": header.source_backend,
                "canonical_sha256": canonical_entry["sha256"],
            }
            if matlab_entry:
                entry["matlab"] = matlab_entry
            entries.append(entry)
            paths.append(target)
            atomic_json(status_path, {"state": "running", "total": len(project.records), "completed": completed})
            if progress:
                progress(ProgressEvent(StageName.SCOUTS, f"generated scouts for {record.record_id}", completed, len(project.records)))
        atomic_json(manifest_path, {"schema": "neurostat.scout_manifest.v1", "run_id": run_id, "upstream_run_id": upstream_id, "records": entries})
        atomic_json(status_path, {"state": "succeeded", "total": len(project.records), "completed": len(project.records), "reused": reused, "written": written})
    except BaseException as exc:
        atomic_json(status_path, {"state": "failed", "total": len(project.records), "completed": len(entries), "error": str(exc)})
        raise
    return ScoutRun(run_id, run_dir, manifest_path, reused, written, tuple(paths))
