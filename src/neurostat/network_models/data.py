"""Verified connectivity input, exact network summaries, and cohort contracts."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis_profiles import project_analysis_profiles
from ..connectivity.artifacts import ConnectivityHeader, verify_connectivity_artifact
from ..connectivity.methods import ConnectivityError
from ..project import CleanProject, ProjectRecord
from ..scouts.artifacts import ScoutHeader, verify_scout_artifact
from ..stage_io import canonical_json, digest_json, load_json, sha256_file
from .config import AnalysisFamily, NetworkModelConfig


class NetworkModelError(RuntimeError):
    """Raised when input, analysis, runtime, or output integrity is unsafe."""


@dataclass(frozen=True, slots=True)
class VerifiedRecord:
    project_record: ProjectRecord
    connectivity_path: Path
    connectivity_sha256: str
    connectivity_header: ConnectivityHeader
    scout_path: Path
    scout_sha256: str
    scout_header: ScoutHeader
    matrix: np.ndarray
    method_provenance: dict[str, Any]
    sfreq: float


@dataclass(frozen=True, slots=True)
class VerifiedConnectivity:
    run_id: str
    run_dir: Path
    stable_identity: dict[str, Any]
    stable_identity_sha256: str
    records: dict[str, VerifiedRecord]


@dataclass(frozen=True, slots=True)
class SummaryBundle:
    rows: tuple[dict[str, Any], ...]
    cohort_rows: tuple[dict[str, Any], ...]
    count_rows: tuple[dict[str, Any], ...]


def _expected_profiles(project: CleanProject) -> tuple[dict[str, Any], dict[str, Any]]:
    profiles = project_analysis_profiles(project)
    return profiles, {name: profile.normalized() for name, profile in profiles.items()}


def _manifest_entries(manifest: Any, schema: str, expected_ids: set[str], description: str) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, dict) or manifest.get("schema") != schema or not isinstance(manifest.get("records"), list):
        raise NetworkModelError(f"invalid {description} manifest schema")
    entries = manifest["records"]
    if any(not isinstance(item, dict) or not isinstance(item.get("record_id"), str) for item in entries):
        raise NetworkModelError(f"invalid {description} manifest records")
    if len({item["record_id"] for item in entries}) != len(entries):
        raise NetworkModelError(f"duplicate record ids in {description} manifest")
    by_id = {item["record_id"]: item for item in entries}
    if set(by_id) != expected_ids:
        raise NetworkModelError(f"{description} manifest records do not exactly match the project")
    return by_id


def _exact_artifact_path(run_dir: Path, stage: str, record_id: str, entry: dict[str, Any]) -> Path:
    expected = (run_dir / stage / "records" / f"{record_id}.h5").resolve()
    try:
        actual = Path(entry["path"]).resolve()
    except (KeyError, TypeError):
        raise NetworkModelError(f"{record_id}: invalid {stage} artifact path") from None
    if actual != expected or not actual.is_file():
        raise NetworkModelError(f"{record_id}: {stage} artifact is outside or missing from its frozen run")
    return actual


def verify_connectivity_run(project: CleanProject, run_dir: str | Path) -> VerifiedConnectivity:
    """Validate the connectivity and its linked scout run down to payload hashes."""
    run_dir = Path(run_dir).resolve()
    profiles, normalized_profiles = _expected_profiles(project)
    lock = load_json(run_dir / "project.lock.json", "connectivity project lock")
    manifest = load_json(run_dir / "connectivity" / "manifest.json", "connectivity manifest")
    status = load_json(run_dir / "connectivity" / "status.json", "connectivity status")
    if (
        not isinstance(lock, dict)
        or lock.get("stage") != "connectivity"
        or canonical_json(lock.get("project")) != canonical_json(project.normalized())
        or canonical_json(lock.get("profiles")) != canonical_json(normalized_profiles)
        or not isinstance(lock.get("upstream"), dict)
        or lock["upstream"].get("stage") != "scouts"
    ):
        raise NetworkModelError("connectivity run belongs to a different normalized project or profile resources")
    if not isinstance(status, dict) or status.get("state") != "succeeded":
        raise NetworkModelError("connectivity run is not complete")
    expected_ids = {record.record_id for record in project.records}
    connectivity_entries = _manifest_entries(
        manifest,
        "neurostat.connectivity_manifest.v1",
        expected_ids,
        "connectivity",
    )
    run_id = manifest.get("run_id")
    upstream_id = manifest.get("upstream_run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(upstream_id, str)
        or upstream_id != lock["upstream"].get("run_id")
    ):
        raise NetworkModelError("connectivity run/upstream identity is inconsistent")

    scout_dir = (project.output_root / "runs" / f"scouts-{upstream_id}").resolve()
    scout_lock = load_json(scout_dir / "project.lock.json", "scout project lock")
    scout_manifest = load_json(scout_dir / "scouts" / "manifest.json", "scout manifest")
    scout_status = load_json(scout_dir / "scouts" / "status.json", "scout status")
    if (
        not isinstance(scout_lock, dict)
        or scout_lock.get("stage") != "scouts"
        or canonical_json(scout_lock.get("project")) != canonical_json(project.normalized())
        or canonical_json(scout_lock.get("profiles")) != canonical_json(normalized_profiles)
        or not isinstance(scout_status, dict)
        or scout_status.get("state") != "succeeded"
    ):
        raise NetworkModelError("linked scout run belongs to a different project/profile or is incomplete")
    scout_entries = _manifest_entries(scout_manifest, "neurostat.scout_manifest.v1", expected_ids, "scout")
    stable_scout = {
        "schema": scout_manifest["schema"],
        "run_id": scout_manifest.get("run_id"),
        "upstream_run_id": scout_manifest.get("upstream_run_id"),
        "records": [
            {key: value for key, value in item.items() if key != "reused"}
            for item in scout_manifest["records"]
        ],
    }
    if lock["upstream"].get("manifest_content_sha256") != digest_json(stable_scout):
        raise NetworkModelError("linked scout scientific manifest differs from the connectivity lock")

    verified: dict[str, VerifiedRecord] = {}
    for record in project.records:
        profile = profiles[record.analysis_profile]
        connectivity_entry = connectivity_entries[record.record_id]
        connectivity_path = _exact_artifact_path(run_dir, "connectivity", record.record_id, connectivity_entry)
        connectivity_sha = sha256_file(connectivity_path)
        if connectivity_entry.get("sha256") != connectivity_sha:
            raise NetworkModelError(f"{record.record_id}: connectivity SHA-256 differs from its manifest")
        expected = {
            "record_id": record.record_id,
            "participant_id": record.key.participant_id,
            "arm": record.key.arm,
            "session": record.key.session,
            "condition": record.key.condition,
            "analysis_profile": record.analysis_profile,
            "connectivity_backend": profile.connectivity.backend,
            "profile_digest": profile.digest,
        }
        try:
            connectivity_header = verify_connectivity_artifact(connectivity_path, expected)
        except ConnectivityError as exc:
            raise NetworkModelError(str(exc)) from exc
        if (
            connectivity_entry.get("payload_digest") != connectivity_header.payload_digest
            or connectivity_entry.get("labels") != list(connectivity_header.labels)
            or connectivity_entry.get("bands") != list(connectivity_header.bands)
        ):
            raise NetworkModelError(f"{record.record_id}: connectivity manifest metadata differs from the artifact")

        scout_entry = scout_entries[record.record_id]
        scout_path = _exact_artifact_path(scout_dir, "scouts", record.record_id, scout_entry)
        scout_sha = sha256_file(scout_path)
        if scout_entry.get("sha256") != scout_sha or lock["upstream"].get("records", {}).get(record.record_id) != scout_sha:
            raise NetworkModelError(f"{record.record_id}: scout SHA-256 differs from its linked manifests")
        scout_header = verify_scout_artifact(
            scout_path,
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
            scout_entry.get("payload_digest") != scout_header.payload_digest
            or scout_entry.get("dimensions") != [len(scout_header.labels), scout_header.samples]
            or scout_entry.get("labels") != list(scout_header.labels)
            or tuple(scout_header.labels) != tuple(connectivity_header.labels)
            or connectivity_entry.get("scout_sha256") != scout_sha
        ):
            raise NetworkModelError(f"{record.record_id}: linked scout metadata/labels differ from connectivity")
        connectivity_header = verify_connectivity_artifact(
            connectivity_path,
            expected
            | {
                "scout_sha256": scout_sha,
                "scout_payload_digest": scout_header.payload_digest,
                "sfreq": scout_header.sfreq,
            },
        )
        import h5py

        with h5py.File(connectivity_path, "r") as file:
            matrix = np.asarray(file["matrix"][:], dtype=np.float64)
            try:
                provenance = json.loads(file.attrs["method_provenance"])
            except (TypeError, json.JSONDecodeError):
                raise NetworkModelError(f"{record.record_id}: invalid method provenance") from None
            sfreq = float(file.attrs["sfreq"])
        if sfreq != scout_header.sfreq:
            raise NetworkModelError(f"{record.record_id}: connectivity sampling rate differs from linked scouts")
        verified[record.record_id] = VerifiedRecord(
            record,
            connectivity_path,
            connectivity_sha,
            connectivity_header,
            scout_path,
            scout_sha,
            scout_header,
            matrix,
            provenance,
            sfreq,
        )
    stable_manifest = {
        "schema": manifest["schema"],
        "run_id": run_id,
        "upstream_run_id": upstream_id,
        "records": [
            {key: value for key, value in item.items() if key != "reused"}
            for item in manifest["records"]
        ],
        "verified_scout_observations": {
            record_id: {
                "sha256": item.scout_sha256,
                "payload_digest": item.scout_header.payload_digest,
                "samples": item.scout_header.samples,
            }
            for record_id, item in sorted(verified.items())
        },
    }
    stable_sha = digest_json(stable_manifest)
    return VerifiedConnectivity(str(run_id), run_dir, stable_manifest, stable_sha, verified)


def within_network_mean(matrix: np.ndarray, indexes: tuple[int, ...]) -> float:
    """Return the exact upper-triangle within-network mean."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise NetworkModelError("network matrix must be square")
    if len(indexes) < 2 or len(set(indexes)) != len(indexes):
        raise NetworkModelError("network indexes must contain at least two unique nodes")
    if min(indexes) < 0 or max(indexes) >= value.shape[0]:
        raise NetworkModelError("network indexes are outside the connectivity matrix")
    submatrix = value[np.ix_(indexes, indexes)]
    result = float(submatrix[np.triu_indices(len(indexes), 1)].mean())
    if not math.isfinite(result) or result < 0 or result > 1:
        raise NetworkModelError("network mean is outside [0, 1]")
    return result


def fisher_boundary_transform(raw_pli: float, sample_count: int) -> tuple[float, float, str]:
    """Apply the manuscript boundary epsilon only at exact PLI 0 or 1."""
    if isinstance(raw_pli, bool) or not math.isfinite(raw_pli) or not 0 <= raw_pli <= 1:
        raise NetworkModelError("raw PLI must be finite and inside [0, 1]")
    if type(sample_count) is not int or sample_count <= 0:
        raise NetworkModelError("sample_count must be a positive integer")
    epsilon = 1.0 / (2.0 * sample_count)
    if raw_pli == 0.0:
        adjusted, policy = epsilon, "exact_zero_to_epsilon"
    elif raw_pli == 1.0:
        adjusted, policy = 1.0 - epsilon, "exact_one_to_one_minus_epsilon"
    else:
        adjusted, policy = raw_pli, "unchanged_interior"
    return adjusted, float(np.arctanh(adjusted)), policy


def _observation_count(item: VerifiedRecord, family: AnalysisFamily) -> tuple[int, str]:
    backend = item.connectivity_header.connectivity_backend
    if family.boundary_observation_count == "scout_samples":
        if backend != "evoked_average_hilbert_pli":
            raise NetworkModelError(
                f"{family.name}: scout_samples boundary count is only valid for all-sample evoked Hilbert PLI"
            )
        return item.scout_header.samples, "verified_upstream_scout_samples"
    if backend != "continuous_multitaper_pli":
        raise NetworkModelError(
            f"{family.name}: connectivity_windows boundary count is only valid for continuous multitaper PLI"
        )
    count = item.method_provenance.get("window_count")
    window_samples = item.method_provenance.get("window_samples")
    dropped = item.method_provenance.get("dropped_tail_samples")
    if (
        type(count) is not int
        or count <= 0
        or type(window_samples) is not int
        or window_samples <= 0
        or type(dropped) is not int
        or dropped < 0
        or count * window_samples + dropped != item.scout_header.samples
    ):
        raise NetworkModelError(f"{item.project_record.record_id}: ambiguous connectivity-window observation count")
    return count, "verified_complete_connectivity_windows"


def _family_cohort(project: CleanProject, family: AnalysisFamily) -> tuple[list[dict[str, Any]], dict[str, ProjectRecord]]:
    participants: dict[str, tuple[str, list[ProjectRecord]]] = {}
    for record in project.records:
        existing = participants.setdefault(record.key.participant_id, (record.key.arm, []))
        if existing[0] != record.key.arm:
            raise NetworkModelError(f"participant {record.key.participant_id!r} has conflicting arm metadata")
        existing[1].append(record)
    included_records: dict[str, ProjectRecord] = {}
    cohort_rows: list[dict[str, Any]] = []
    included_by_arm = {arm: 0 for arm in project.study.arms}
    for subject in sorted(participants):
        arm, records = participants[subject]
        selected = [
            record
            for record in records
            if record.key.condition == family.condition and record.key.session in family.sessions
        ]
        by_session: dict[str, list[ProjectRecord]] = {session: [] for session in family.sessions}
        for record in selected:
            by_session[record.key.session].append(record)
        duplicates = [session for session, values in by_session.items() if len(values) > 1]
        if duplicates:
            raise NetworkModelError(
                f"{family.name}: participant {subject!r} has duplicate records for {', '.join(duplicates)}"
            )
        missing = [session for session, values in by_session.items() if not values]
        included = not missing
        if included:
            included_by_arm[arm] += 1
            for session in family.sessions:
                record = by_session[session][0]
                included_records[record.record_id] = record
        cohort_rows.append(
            {
                "analysis_family": family.name,
                "condition": family.condition,
                "subject_id": subject,
                "arm": arm,
                "included": "true" if included else "false",
                "reason": "complete" if included else "missing_required_sessions",
                "available_sessions": "|".join(session for session in family.sessions if by_session[session]),
                "missing_sessions": "|".join(missing),
            }
        )
    missing_arms = [arm for arm, count in included_by_arm.items() if count == 0]
    if missing_arms:
        raise NetworkModelError(
            f"{family.name}: complete cohort has no participant(s) in configured arm(s): {', '.join(missing_arms)}"
        )
    return cohort_rows, included_records


def build_network_summaries(
    project: CleanProject,
    config: NetworkModelConfig,
    upstream: VerifiedConnectivity,
) -> SummaryBundle:
    """Build one common-cohort long table per explicit condition family."""
    rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    networks = config.networks_by_name
    for family in config.families:
        family_cohort, records = _family_cohort(project, family)
        cohort_rows.extend(family_cohort)
        included_subjects = {
            row["subject_id"] for row in family_cohort if row["included"] == "true"
        }
        for arm in project.study.arms:
            for session in family.sessions:
                count_rows.append(
                    {
                        "analysis_family": family.name,
                        "condition": family.condition,
                        "arm": arm,
                        "session": session,
                        "subjects": sum(
                            row["included"] == "true" and row["arm"] == arm
                            for row in family_cohort
                        ),
                    }
                )
        reference_labels: tuple[str, ...] | None = None
        for record in sorted(
            records.values(),
            key=lambda item: (
                item.key.arm,
                item.key.participant_id,
                family.sessions.index(item.key.session),
                item.record_id,
            ),
        ):
            item = upstream.records[record.record_id]
            labels = item.connectivity_header.labels
            if reference_labels is None:
                reference_labels = labels
            elif labels != reference_labels:
                raise NetworkModelError(f"{family.name}: connectivity atlas labels differ across records")
            band_indexes = {name: index for index, name in enumerate(item.connectivity_header.bands)}
            missing_bands = set(family.bands) - set(band_indexes)
            if missing_bands:
                raise NetworkModelError(
                    f"{record.record_id}: requested bands are absent: {', '.join(sorted(missing_bands))}"
                )
            label_indexes = {label: index for index, label in enumerate(labels)}
            sample_count, sample_policy = _observation_count(item, family)
            for network_name in family.networks:
                definition = networks[network_name]
                missing_labels = set(definition.labels) - set(label_indexes)
                if missing_labels:
                    raise NetworkModelError(
                        f"{record.record_id}: network {network_name!r} has labels absent from the atlas: "
                        f"{', '.join(sorted(missing_labels))}"
                    )
                indexes = tuple(label_indexes[label] for label in definition.labels)
                for band in family.bands:
                    raw = within_network_mean(item.matrix[band_indexes[band]], indexes)
                    adjusted, fisher_z, boundary_policy = fisher_boundary_transform(raw, sample_count)
                    rows.append(
                        {
                            "analysis_family": family.name,
                            "condition": family.condition,
                            "record_id": record.record_id,
                            "record_sha256": item.connectivity_sha256,
                            "subject_id": record.key.participant_id,
                            "arm": record.key.arm,
                            "session": record.key.session,
                            "band": band,
                            "network": network_name,
                            "raw_pli": format(raw, ".17g"),
                            "adjusted_pli": format(adjusted, ".17g"),
                            "z_pli": format(fisher_z, ".17g"),
                            "sample_count": sample_count,
                            "sample_count_policy": sample_policy,
                            "boundary_policy": boundary_policy,
                        }
                    )
        if not included_subjects:
            raise NetworkModelError(f"{family.name}: common complete cohort is empty")
    return SummaryBundle(tuple(rows), tuple(cohort_rows), tuple(count_rows))


def write_csv(path: Path, rows: tuple[dict[str, Any], ...] | list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
