#!/usr/bin/env python
"""Replicate a published LEMON aging effect with NeuroStat V2.

The confirmatory endpoint is lower global source-space alpha PLI in older
(55-80 years) than younger (20-35 years) adults during eyes-closed rest. The
target and cohort-selection rule are fixed before NeuroStat outcomes are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_NAME = "MPI Leipzig Mind-Brain-Body (LEMON)"
DATASET_DOI = "10.1038/sdata.2018.308"
DATASET_PAGE = "http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/MPI_LEMON.html"
# NITRC's legacy fcon_1000 host has a certificate for a different hostname.
# Use its published HTTP index only for discovery; EEG payloads use pinned HTTPS S3.
DOWNLOAD_PAGE = "http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/downloads/download_EEG.html"
BEHAVIOURAL_URL = "https://fcp-indi.s3.amazonaws.com/data/Projects/INDI/MPI-LEMON/Compressed_tar/Behavioural_Data_MPILMBB_LEMON.tar.gz"
PREPROCESSED_ROOT = "https://fcp-indi.s3.amazonaws.com/data/Projects/INDI/MPI-LEMON/Compressed_tar/EEG_MPILMBB_LEMON/EEG_Preprocessed_BIDS_ID/EEG_Preprocessed"
DISCOVER_DOI = "10.1038/s41597-023-02525-0"
DISCOVER_CODE_COMMIT = "b92295d5b2dd0a496c3340fc95ddfa54e40236e9"
DISCOVER_FIGURE_SCRIPT = "examples/fig5_lemon_young_old.m"
SELECTION_SALT = "neurostat-v2-lemon-replication-v1"
AGE_GROUPS = {
    "young": {"20-25", "25-30", "30-35"},
    "old": {"55-60", "60-65", "65-70", "70-75", "75-80"},
}
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
PERMUTATIONS = 100_000
RANDOM_SEED = 20260804
MIN_DURATION_SECONDS = 400.0


@dataclass(frozen=True, slots=True)
class Participant:
    participant_id: str
    age_bin: str
    age_group: str
    sex: str


@dataclass(frozen=True, slots=True)
class SourceSpec:
    participant_id: str
    url: str
    version_id: str
    size: int
    etag: str

    @property
    def pinned_url(self) -> str:
        return f"{self.url}?{urllib.parse.urlencode({'versionId': self.version_id})}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree-uninstalled"


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "NeuroStat-V2-validation/1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _metadata_csv(archive: Path) -> bytes:
    expected = "Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.getmember(expected)
        if not member.isfile():
            raise RuntimeError("LEMON metadata member is not a regular file")
        handle = bundle.extractfile(member)
        if handle is None:
            raise RuntimeError("cannot read LEMON metadata member")
        return handle.read()


def _parse_participants(content: bytes, available_ids: set[str]) -> list[Participant]:
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig")))
    next(reader)
    participants: list[Participant] = []
    for row in reader:
        if len(row) < 3 or row[0] not in available_ids:
            continue
        age_group = next(
            (name for name, bins in AGE_GROUPS.items() if row[2] in bins), None
        )
        if age_group is None or row[1] not in {"1", "2"}:
            continue
        participants.append(
            Participant(row[0], row[2], age_group, {"1": "female", "2": "male"}[row[1]])
        )
    if len({item.participant_id for item in participants}) != len(participants):
        raise RuntimeError("duplicate eligible participant IDs in LEMON metadata")
    return sorted(participants, key=lambda item: item.participant_id)


def _available_preprocessed_ids(download_page: bytes) -> set[str]:
    import re

    pattern = rb"EEG_Preprocessed_BIDS_ID/EEG_Preprocessed/(sub-[0-9]+)\.tar\.gz"
    values = {match.decode("ascii") for match in re.findall(pattern, download_page)}
    if len(values) < 200:
        raise RuntimeError(
            f"expected at least 200 public preprocessed LEMON archives; found {len(values)}"
        )
    return values


def _rank(participant_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{participant_id}".encode()).hexdigest()


def select_cohort(
    participants: Iterable[Participant], cohort: str
) -> list[Participant]:
    values = list(participants)
    if cohort == "full":
        return sorted(values, key=lambda item: item.participant_id)
    per_stratum = 1 if cohort == "pilot" else 15
    selected: list[Participant] = []
    for age_group in ("young", "old"):
        for sex in ("female", "male"):
            stratum = sorted(
                (
                    item
                    for item in values
                    if item.age_group == age_group and item.sex == sex
                ),
                key=lambda item: (_rank(item.participant_id), item.participant_id),
            )
            if len(stratum) < per_stratum:
                raise RuntimeError(
                    f"insufficient participants in {age_group}/{sex} stratum"
                )
            selected.extend(stratum[:per_stratum])
    return sorted(selected, key=lambda item: item.participant_id)


def _head_source(participant_id: str) -> SourceSpec:
    url = f"{PREPROCESSED_ROOT}/{participant_id}.tar.gz"
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "NeuroStat-V2-validation/1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        version_id = response.headers.get("x-amz-version-id")
        size = response.headers.get("Content-Length")
        etag = response.headers.get("ETag")
    if not version_id or not size or not etag:
        raise RuntimeError(
            f"public archive does not expose immutable S3 metadata: {url}"
        )
    return SourceSpec(participant_id, url, version_id, int(size), etag.strip('"'))


def _head_source_with_retry(participant_id: str) -> SourceSpec:
    for attempt in range(3):
        try:
            return _head_source(participant_id)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or attempt == 2:
                raise
        except urllib.error.URLError:
            if attempt == 2:
                raise
        time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable retry state")


def freeze_metadata(root: Path, cohort: str, workers: int = 8) -> Path:
    provenance = root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    behavioural = provenance / "Behavioural_Data_MPILMBB_LEMON.tar.gz"
    if not behavioural.is_file():
        behavioural.write_bytes(_read_url(BEHAVIOURAL_URL))
    page = _read_url(DOWNLOAD_PAGE)
    page_path = provenance / "download_EEG.html"
    page_path.write_bytes(page)
    metadata = _metadata_csv(behavioural)
    metadata_path = provenance / "participants_metadata.csv"
    metadata_path.write_bytes(metadata)
    participants = _parse_participants(metadata, _available_preprocessed_ids(page))
    available_specs: dict[str, SourceSpec] = {}
    unavailable_ids: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_head_source_with_retry, item.participant_id): item
            for item in participants
        }
        for future in as_completed(futures):
            participant = futures[future]
            try:
                available_specs[participant.participant_id] = future.result()
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                unavailable_ids.append(participant.participant_id)
    eligible = [item for item in participants if item.participant_id in available_specs]
    selected = select_cohort(eligible, cohort)

    manifest_path = provenance / f"source_manifest_{cohort}.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("participant_ids") != [
            item.participant_id for item in selected
        ]:
            raise RuntimeError(
                "existing frozen manifest does not match deterministic cohort"
            )
        return manifest_path

    specs = [available_specs[item.participant_id] for item in selected]
    for spec in specs:
        print(
            f"pinned {spec.participant_id} ({spec.size / 1024**2:.1f} MiB)",
            flush=True,
        )
    specs.sort(key=lambda item: item.participant_id)
    _atomic_json(
        manifest_path,
        {
            "schema": "neurostat.lemon_source_manifest.v1",
            "dataset": DATASET_NAME,
            "dataset_doi": DATASET_DOI,
            "dataset_page": DATASET_PAGE,
            "download_page": DOWNLOAD_PAGE,
            "download_page_sha256": _sha256(page_path),
            "behavioural_archive_url": BEHAVIOURAL_URL,
            "behavioural_archive_sha256": _sha256(behavioural),
            "participant_metadata_sha256": _sha256(metadata_path),
            "cohort": cohort,
            "selection_salt": SELECTION_SALT,
            "selection_rule": "SHA-256 rank within age-group x sex strata after outcome-independent archive availability; 15 per stratum for replication",
            "officially_listed_but_unavailable_ids": sorted(unavailable_ids),
            "participant_ids": [item.participant_id for item in selected],
            "participants": [asdict(item) for item in selected],
            "sources": [asdict(item) for item in specs],
        },
    )
    return manifest_path


def _download(spec: SourceSpec, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    if (
        destination.is_file()
        and destination.stat().st_size == spec.size
        and sidecar.is_file()
    ):
        recorded = sidecar.read_text(encoding="ascii").strip()
        if recorded == _sha256(destination):
            return {
                "participant_id": spec.participant_id,
                "size": spec.size,
                "sha256": recorded,
                "reused": True,
            }
    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.is_file() else 0
    if start > spec.size:
        partial.unlink()
        start = 0
    headers = {"User-Agent": "NeuroStat-V2-validation/1"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(spec.pinned_url, headers=headers)
    response = urllib.request.urlopen(request, timeout=180)
    if start and response.status != 206:
        response.close()
        partial.unlink(missing_ok=True)
        start = 0
        response = urllib.request.urlopen(
            urllib.request.Request(
                spec.pinned_url, headers={"User-Agent": headers["User-Agent"]}
            ),
            timeout=180,
        )
    with response, partial.open("ab" if start else "wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
    if partial.stat().st_size != spec.size:
        raise RuntimeError(f"size mismatch for {spec.participant_id}")
    digest = _sha256(partial)
    os.replace(partial, destination)
    sidecar.write_text(digest + "\n", encoding="ascii")
    return {
        "participant_id": spec.participant_id,
        "size": spec.size,
        "sha256": digest,
        "reused": False,
    }


def download_dataset(root: Path, cohort: str, workers: int = 6) -> Path:
    manifest_path = freeze_metadata(root, cohort, workers)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_root = root / "raw" / cohort
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for source in manifest["sources"]:
            spec = SourceSpec(**source)
            target = raw_root / f"{spec.participant_id}.tar.gz"
            futures[pool.submit(_download, spec, target)] = spec.participant_id
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"downloaded/verified {result['participant_id']}", flush=True)
    results.sort(key=lambda item: item["participant_id"])
    target = root / "provenance" / f"download_manifest_{cohort}.json"
    _atomic_json(
        target,
        {"schema": "neurostat.lemon_download.v1", "cohort": cohort, "files": results},
    )
    return target


def _extract_ec(
    archive: Path, destination: Path, participant_id: str
) -> tuple[Path, Path]:
    wanted = {
        f"{participant_id}/{participant_id}_EC.set": destination
        / participant_id
        / f"{participant_id}_EC.set",
        f"{participant_id}/{participant_id}_EC.fdt": destination
        / participant_id
        / f"{participant_id}_EC.fdt",
    }
    with tarfile.open(archive, "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        for name, target in wanted.items():
            path = PurePosixPath(name)
            member = members.get(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or member is None
                or not member.isfile()
            ):
                raise RuntimeError(f"unsafe or missing expected archive member: {name}")
            handle = bundle.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read archive member: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp")
            with handle, temporary.open("wb") as output:
                shutil.copyfileobj(handle, output, length=1024 * 1024)
            os.replace(temporary, target)
    return wanted[f"{participant_id}/{participant_id}_EC.set"], wanted[
        f"{participant_id}/{participant_id}_EC.fdt"
    ]


def _alpha_peak_frequency(raw: Any) -> tuple[float, float]:
    from scipy.signal import find_peaks, welch

    data = raw.get_data(picks="eeg")
    sfreq = float(raw.info["sfreq"])
    frequencies, psd = welch(
        data,
        fs=sfreq,
        window="hann",
        nperseg=round(10 * sfreq),
        noverlap=round(5 * sfreq),
        detrend="constant",
        axis=-1,
        scaling="density",
    )
    average = psd.mean(axis=0)
    mask = (frequencies >= 8.0) & (frequencies <= 12.9)
    band_frequency = frequencies[mask]
    band_power = average[mask]
    peaks, _ = find_peaks(band_power)
    local_max = (
        math.nan
        if not len(peaks)
        else float(band_frequency[peaks[np.argmax(band_power[peaks])]])
    )
    centre = float(np.sum(band_frequency * band_power) / np.sum(band_power))
    return local_max, centre


def _ensure_eeglab_companion_alias(set_path: Path, extracted_fdt: Path) -> Path:
    from scipy.io import loadmat

    eeg = loadmat(set_path, squeeze_me=True, struct_as_record=False).get("EEG")
    reference = getattr(eeg, "data", None)
    if not isinstance(reference, str) or not reference.lower().endswith(".fdt"):
        raise RuntimeError(
            f"{set_path.name} does not declare an external .fdt companion"
        )
    reference_path = PurePosixPath(reference.replace("\\", "/"))
    if reference_path.is_absolute() or len(reference_path.parts) != 1:
        raise RuntimeError(
            f"unsafe EEGLAB companion reference in {set_path.name}: {reference}"
        )
    alias = set_path.parent / reference_path.name
    if alias == extracted_fdt:
        return alias
    if alias.exists():
        if not alias.is_file() or _sha256(alias) != _sha256(extracted_fdt):
            raise RuntimeError(f"conflicting EEGLAB companion alias: {alias}")
        return alias
    os.link(extracted_fdt, alias)
    return alias


def prepare_dataset(root: Path, cohort: str) -> Path:
    import mne

    source_manifest = json.loads(
        (root / "provenance" / f"source_manifest_{cohort}.json").read_text(
            encoding="utf-8"
        )
    )
    participants = {
        item["participant_id"]: item for item in source_manifest["participants"]
    }
    extracted = root / "extracted" / cohort
    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for participant_id in source_manifest["participant_ids"]:
        archive = root / "raw" / cohort / f"{participant_id}.tar.gz"
        set_path = extracted / participant_id / f"{participant_id}_EC.set"
        fdt_path = set_path.with_suffix(".fdt")
        if not set_path.is_file() or not fdt_path.is_file():
            set_path, fdt_path = _extract_ec(archive, extracted, participant_id)
        companion_path = _ensure_eeglab_companion_alias(set_path, fdt_path)
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
        eeg_count = len(raw.copy().pick("eeg").ch_names)
        duration = raw.n_times / float(raw.info["sfreq"])
        if (
            eeg_count < 50
            or not raw.info.get("dig")
            or duration < MIN_DURATION_SECONDS
            or not np.isfinite(raw.get_data(picks="eeg")).all()
        ):
            raise RuntimeError(
                f"{participant_id} failed the frozen EEG eligibility checks"
            )
        apf_local, apf_cog = _alpha_peak_frequency(raw)
        metadata = participants[participant_id]
        rows.append(
            {
                "record_id": f"{participant_id.removeprefix('sub-')}-eyes-closed",
                "participant_id": participant_id,
                "arm": metadata["age_group"],
                "session": "eyes_closed",
                "condition": "rest",
                "data_kind": "raw",
                "format": "set",
                "path": str(set_path.resolve()),
                "analysis_profile": "rest",
                "data_unit": "V",
            }
        )
        qc_rows.append(
            {
                **metadata,
                "channels": len(raw.ch_names),
                "eeg_channels": eeg_count,
                "sfreq_hz": float(raw.info["sfreq"]),
                "samples": raw.n_times,
                "duration_seconds": duration,
                "set_sha256": _sha256(set_path),
                "fdt_sha256": _sha256(fdt_path),
                "declared_fdt_name": companion_path.name,
                "alpha_peak_local_max_hz": apf_local,
                "alpha_peak_centre_of_gravity_hz": apf_cog,
            }
        )
        print(
            f"prepared {participant_id} ({metadata['age_group']}, {metadata['sex']})",
            flush=True,
        )
    prepared = root / "prepared" / cohort
    prepared.mkdir(parents=True, exist_ok=True)
    for name, values in (("records.csv", rows), ("input_qc_and_apf.csv", qc_rows)):
        with (prepared / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    return prepared / "records.csv"


def _relative_to_project(project_dir: Path, path: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), project_dir.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_project(root: Path, cohort: str) -> Path:
    import mne
    import yaml

    from neurostat.gui.fsaverage import provision_fsaverage
    from neurostat.project import load_project

    source = root / "prepared" / cohort / "records.csv"
    if not source.is_file():
        raise RuntimeError(f"missing prepared records: {source}")
    project_dir = root / "projects" / cohort
    project_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(source.open(encoding="utf-8", newline="")))
    for row in rows:
        row["path"] = _relative_to_project(project_dir, Path(row["path"]))
    with (project_dir / "records.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    resources = provision_fsaverage(progress=print)
    source_space = root / "resources" / "fsaverage-ico3-src.fif"
    if not source_space.is_file():
        source_space.parent.mkdir(parents=True, exist_ok=True)
        src = mne.setup_source_space(
            resources.subject,
            spacing="ico3",
            subjects_dir=resources.subjects_dir,
            add_dist=False,
            verbose="ERROR",
        )
        mne.write_source_spaces(source_space, src, overwrite=True, verbose="ERROR")
    project = {
        "schema_version": 1,
        "study": {
            "id": "lemon-discover-eeg-v2-replication",
            "arms": ["young", "old"],
            "sessions": ["eyes_closed", "not_observed_contract_slot"],
            "conditions": ["rest"],
        },
        "input": {"state": "cleaned"},
        "records": {"manifest": "records.csv"},
        "analysis_profiles": {
            "rest": {
                "source": {
                    "backend": "mne_fsaverage",
                    "subject": resources.subject,
                    "subjects_dir": _relative_to_project(
                        project_dir, resources.subjects_dir
                    ),
                    "trans": _relative_to_project(project_dir, resources.trans),
                    "source_space": _relative_to_project(project_dir, source_space),
                    "bem": _relative_to_project(project_dir, resources.bem),
                    "atlas": "aparc",
                    "inverse_method": "MNE",
                    "snr": 3.0,
                    "pick_orientation": "normal",
                    "label_mode": "mean_flip",
                    "chunk_samples": 8192,
                    "output_chunk_samples": 4096,
                    "depth": 0.8,
                    "loose": 0.2,
                    "mindist_mm": 5.0,
                    "noise_covariance": "empirical",
                    "average_reference_projection": True,
                    "coordinate_frame": "head",
                    "export_matlab": False,
                },
                "connectivity": {
                    "backend": "continuous_multitaper_pli",
                    "bands": {name: list(edges) for name, edges in BANDS.items()},
                    "window_seconds": 2.0,
                    "tail_policy": "drop",
                    "mt_bandwidth": 2.0,
                    "mt_adaptive": False,
                    "mt_low_bias": True,
                    "n_jobs": 1,
                },
            }
        },
        "output": {"root": "output"},
    }
    project_path = project_dir / "project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    load_project(project_path)
    source_manifest = json.loads(
        (root / "provenance" / f"source_manifest_{cohort}.json").read_text(
            encoding="utf-8"
        )
    )
    _atomic_json(
        project_dir / "validation_protocol.json",
        {
            "schema": "neurostat.lemon_replication_protocol.v1",
            "frozen_before_outcomes": True,
            "dataset": {
                "name": DATASET_NAME,
                "doi": DATASET_DOI,
                "page": DATASET_PAGE,
                "condition": "eyes closed",
                "public_derivative": "preprocessed EEGLAB .set/.fdt",
            },
            "published_target": {
                "paper_doi": DISCOVER_DOI,
                "code_commit": DISCOVER_CODE_COMMIT,
                "code_path": DISCOVER_FIGURE_SCRIPT,
                "finding": "older adults have reduced source-space phase connectivity, predominantly in alpha, and lower alpha peak frequency",
            },
            "cohort": cohort,
            "session_contract_note": (
                "V2's RCT project schema requires at least two declared sessions. "
                "LEMON is cross-sectional, so not_observed_contract_slot is declared "
                "but has no records and is never analyzed. EEG is not duplicated."
            ),
            "selection_salt": SELECTION_SALT,
            "participants": source_manifest["participants"],
            "input_eligibility": {
                "minimum_eeg_channels": 50,
                "channel_geometry_required": True,
                "finite_eeg_required": True,
                "minimum_duration_seconds": MIN_DURATION_SECONDS,
                "minimum_complete_connectivity_windows": int(
                    MIN_DURATION_SECONDS / 2.0
                ),
                "amendment": (
                    "Minimum duration changed from the draft 420 s guard to 400 s "
                    "before replication connectivity was run, after an otherwise valid "
                    "official recording was found to last 413.392 s."
                ),
            },
            "primary_endpoint": "mean upper-triangle DK68 alpha PLI",
            "primary_contrast": "old minus young",
            "primary_hypothesis": "old < young",
            "primary_test": {
                "method": "one-sided label permutation",
                "permutations": PERMUTATIONS,
                "seed": RANDOM_SEED,
                "alpha": 0.05,
                "pass": "effect < 0 and p < 0.05",
            },
            "supporting_endpoints": [
                "fraction of negative alpha edge effects > 0.5",
                "alpha Cohen d is more negative than other bands",
                "sensor-space alpha peak frequency old < young",
            ],
            "source_model": {
                "template": "fsaverage",
                "spacing": "ico3",
                "atlas": "aparc_DK68",
                "inverse": "MNE",
                "snr": 3.0,
            },
            "connectivity": {
                "measure": "PLI",
                "backend": "mne-connectivity multitaper",
                "bands_hz": BANDS,
                "window_seconds": 2.0,
                "bandwidth_hz": 2.0,
            },
            "replication_scope": "directional/conceptual, not exact numerical reproduction; atlas, inverse method, and phase metric differ from DISCOVER-EEG",
            "software": {
                name: _package_version(name)
                for name in (
                    "neurostat-rcts",
                    "numpy",
                    "scipy",
                    "mne",
                    "mne-connectivity",
                )
            },
        },
    )
    return project_path


def run_v2(project_path: Path) -> Path:
    from neurostat.workflow import run_workflow

    started = time.time()
    result = run_workflow(
        project_path,
        ["connectivity"],
        progress=lambda event: print(
            f"[{event.stage.value}] {event.message}", flush=True
        ),
    )
    target = project_path.parent / "workflow_result.json"
    _atomic_json(
        target,
        {
            "schema": "neurostat.lemon_workflow_result.v1",
            "started_unix": started,
            "finished_unix": time.time(),
            "success": result.success,
            "stages": [
                {
                    "stage": item.stage.value,
                    "state": item.state.value,
                    "run_id": item.run_id,
                    "run_dir": str(item.run_dir) if item.run_dir else None,
                    "reused": item.reused,
                }
                for item in result.outcomes
            ],
        },
    )
    if not result.success:
        raise RuntimeError(f"V2 workflow failed; inspect {target}")
    return target


def _cohens_d(old: np.ndarray, young: np.ndarray) -> float:
    pooled = math.sqrt(
        ((len(old) - 1) * old.var(ddof=1) + (len(young) - 1) * young.var(ddof=1))
        / (len(old) + len(young) - 2)
    )
    return float((old.mean() - young.mean()) / pooled) if pooled else math.nan


def _permutation_p(
    old: np.ndarray,
    young: np.ndarray,
    *,
    permutations: int = PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> float:
    observed = float(old.mean() - young.mean())
    combined = np.concatenate([old, young])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        shuffled = rng.permutation(combined)
        difference = float(shuffled[: len(old)].mean() - shuffled[len(old) :].mean())
        count += difference <= observed + 1e-15
    return (count + 1) / (permutations + 1)


def summarize(project_path: Path) -> Path:
    import h5py
    import matplotlib
    from scipy import stats

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neurostat.connectivity import verify_connectivity_artifact
    from neurostat.project import load_project

    workflow = json.loads(
        (project_path.parent / "workflow_result.json").read_text(encoding="utf-8")
    )
    stage = next(item for item in workflow["stages"] if item["stage"] == "connectivity")
    manifest_path = Path(stage["run_dir"]) / "connectivity" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {item["record_id"]: Path(item["path"]) for item in manifest["records"]}
    project = load_project(project_path)
    rows: list[dict[str, Any]] = []
    matrices: dict[str, dict[str, np.ndarray]] = {}
    labels: tuple[str, ...] | None = None
    for record in project.records:
        path = artifacts[record.record_id]
        header = verify_connectivity_artifact(path)
        labels = header.labels
        with h5py.File(path, "r") as handle:
            values = np.asarray(handle["matrix"][:], dtype=float)
        triangle = np.triu_indices(len(header.labels), 1)
        matrices[record.key.participant_id] = {}
        for index, band in enumerate(header.bands):
            matrices[record.key.participant_id][band] = values[index]
            rows.append(
                {
                    "participant_id": record.key.participant_id,
                    "age_group": record.key.arm,
                    "band": band,
                    "global_mean_pli": float(values[index][triangle].mean()),
                    "artifact_sha256": _sha256(path),
                }
            )
    if labels is None:
        raise RuntimeError("no connectivity artifacts")

    results_dir = project_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "subject_band_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries: list[dict[str, Any]] = []
    for band_index, band in enumerate(BANDS):
        old = np.asarray(
            [
                row["global_mean_pli"]
                for row in rows
                if row["band"] == band and row["age_group"] == "old"
            ]
        )
        young = np.asarray(
            [
                row["global_mean_pli"]
                for row in rows
                if row["band"] == band and row["age_group"] == "young"
            ]
        )
        welch = stats.ttest_ind(old, young, equal_var=False)
        summaries.append(
            {
                "band": band,
                "old_n": len(old),
                "young_n": len(young),
                "old_mean": float(old.mean()),
                "old_sd": float(old.std(ddof=1)),
                "young_mean": float(young.mean()),
                "young_sd": float(young.std(ddof=1)),
                "old_minus_young": float(old.mean() - young.mean()),
                "cohens_d": _cohens_d(old, young),
                "welch_t": float(welch.statistic),
                "welch_two_sided_p": float(welch.pvalue),
                "one_sided_permutation_p": _permutation_p(
                    old, young, seed=RANDOM_SEED + band_index
                ),
            }
        )
    with (results_dir / "band_group_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    alpha_old = np.stack(
        [
            matrices[item]["alpha"]
            for item in matrices
            if next(row["age_group"] for row in rows if row["participant_id"] == item)
            == "old"
        ]
    )
    alpha_young = np.stack(
        [
            matrices[item]["alpha"]
            for item in matrices
            if next(row["age_group"] for row in rows if row["participant_id"] == item)
            == "young"
        ]
    )
    old_mean, young_mean = alpha_old.mean(axis=0), alpha_young.mean(axis=0)
    pooled = np.sqrt(
        (
            (len(alpha_old) - 1) * alpha_old.var(axis=0, ddof=1)
            + (len(alpha_young) - 1) * alpha_young.var(axis=0, ddof=1)
        )
        / (len(alpha_old) + len(alpha_young) - 2)
    )
    edge_d = np.divide(
        old_mean - young_mean, pooled, out=np.zeros_like(pooled), where=pooled > 0
    )
    triangle = np.triu_indices(len(labels), 1)
    negative_fraction = float(np.mean(edge_d[triangle] < 0))
    with (results_dir / "alpha_edge_effect_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["parcel", *labels])
        for label, values in zip(labels, edge_d):
            writer.writerow([label, *values])

    apf_path = (
        project_path.parents[2]
        / "prepared"
        / project_path.parent.name
        / "input_qc_and_apf.csv"
    )
    apf_rows = list(csv.DictReader(apf_path.open(encoding="utf-8", newline="")))
    apf_old = np.asarray(
        [
            float(row["alpha_peak_local_max_hz"])
            for row in apf_rows
            if row["age_group"] == "old" and row["alpha_peak_local_max_hz"]
        ]
    )
    apf_young = np.asarray(
        [
            float(row["alpha_peak_local_max_hz"])
            for row in apf_rows
            if row["age_group"] == "young" and row["alpha_peak_local_max_hz"]
        ]
    )
    apf_p = _permutation_p(apf_old, apf_young, seed=RANDOM_SEED + 99)
    apf_summary = {
        "old_n": len(apf_old),
        "young_n": len(apf_young),
        "old_mean_hz": float(apf_old.mean()),
        "young_mean_hz": float(apf_young.mean()),
        "old_minus_young_hz": float(apf_old.mean() - apf_young.mean()),
        "cohens_d": _cohens_d(apf_old, apf_young),
        "one_sided_permutation_p": apf_p,
    }
    alpha_summary = next(item for item in summaries if item["band"] == "alpha")
    criteria = {
        "primary_alpha_global": bool(
            alpha_summary["old_minus_young"] < 0
            and alpha_summary["one_sided_permutation_p"] < 0.05
        ),
        "support_alpha_edge_direction": negative_fraction > 0.5,
        "support_alpha_strongest_band": alpha_summary["cohens_d"]
        == min(item["cohens_d"] for item in summaries),
        "positive_control_alpha_peak": bool(
            apf_summary["old_minus_young_hz"] < 0 and apf_p < 0.05
        ),
    }
    result = {
        "schema": "neurostat.lemon_replication_results.v1",
        "dataset_doi": DATASET_DOI,
        "published_target_doi": DISCOVER_DOI,
        "records_verified": len(artifacts),
        "participants": len(matrices),
        "bands": summaries,
        "alpha_edge_negative_fraction": negative_fraction,
        "alpha_peak_frequency": apf_summary,
        "criteria": criteria,
        "primary_replication_passed": criteria["primary_alpha_global"],
        "interpretation": "Directional conceptual replication only; NeuroStat V2 and DISCOVER-EEG use different atlases, inverse solvers, and phase-connectivity estimators.",
        "workflow_duration_seconds": workflow["finished_unix"]
        - workflow["started_unix"],
        "connectivity_manifest_sha256": _sha256(manifest_path),
        "source_data_included": False,
    }
    _atomic_json(results_dir / "validation_results.json", result)

    colours = {"young": "#147d92", "old": "#c6533d"}
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.2))
    alpha_rows = [row for row in rows if row["band"] == "alpha"]
    for index, group in enumerate(("young", "old")):
        values = np.asarray(
            [row["global_mean_pli"] for row in alpha_rows if row["age_group"] == group]
        )
        jitter = np.random.default_rng(RANDOM_SEED + index).uniform(
            -0.09, 0.09, len(values)
        )
        axes[0, 0].scatter(
            np.full(len(values), index) + jitter,
            values,
            color=colours[group],
            alpha=0.72,
            s=26,
        )
        axes[0, 0].boxplot(
            values,
            positions=[index],
            widths=0.38,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": colours[group]},
            medianprops={"color": "#202020"},
        )
    axes[0, 0].set_xticks([0, 1], ["Young\n20-35", "Old\n55-80"])
    axes[0, 0].set_ylabel("Global mean DK68 alpha PLI")
    axes[0, 0].set_title(
        f"Primary replication: old < young\nd = {alpha_summary['cohens_d']:.2f}, one-sided p = {alpha_summary['one_sided_permutation_p']:.4f}"
    )

    names = [item["band"].title() for item in summaries]
    effects = [item["cohens_d"] for item in summaries]
    bar_colours = ["#9aa1a6", "#c6533d", "#9aa1a6", "#9aa1a6"]
    axes[0, 1].bar(names, effects, color=bar_colours)
    axes[0, 1].axhline(0, color="#202020", linewidth=0.8)
    axes[0, 1].set_ylabel("Cohen's d (old - young)")
    axes[0, 1].set_title(
        "Band specificity\nNegative values match the published direction"
    )

    for index, (group, values) in enumerate((("young", apf_young), ("old", apf_old))):
        jitter = np.random.default_rng(RANDOM_SEED + 10 + index).uniform(
            -0.09, 0.09, len(values)
        )
        axes[1, 0].scatter(
            np.full(len(values), index) + jitter,
            values,
            color=colours[group],
            alpha=0.72,
            s=26,
        )
        axes[1, 0].boxplot(
            values,
            positions=[index],
            widths=0.38,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": colours[group]},
            medianprops={"color": "#202020"},
        )
    axes[1, 0].set_xticks([0, 1], ["Young", "Old"])
    axes[1, 0].set_ylabel("Alpha peak frequency (Hz)")
    axes[1, 0].set_title(
        f"Independent positive control\nd = {apf_summary['cohens_d']:.2f}, one-sided p = {apf_p:.4f}"
    )

    vmax = max(0.25, float(np.nanpercentile(np.abs(edge_d[triangle]), 98)))
    image = axes[1, 1].imshow(
        edge_d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"
    )
    axes[1, 1].set_title(
        f"Alpha edge effects (old - young)\n{negative_fraction:.1%} of edges are negative"
    )
    axes[1, 1].set_xlabel("DK68 parcel")
    axes[1, 1].set_ylabel("DK68 parcel")
    fig.colorbar(image, ax=axes[1, 1], label="Cohen's d", shrink=0.82)
    for axis in axes.flat:
        axis.grid(False)
    status = "PASS" if criteria["primary_alpha_global"] else "NOT REPLICATED"
    fig.suptitle(
        f"NeuroStat V2 public validation: DISCOVER-EEG / LEMON aging finding [{status}]",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Prespecified eyes-closed contrast. Published target: reduced phase connectivity in older adults, predominantly alpha.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(
        results_dir / "lemon_v2_replication_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    return results_dir / "validation_results.json"


def published_metric_sensitivity(project_path: Path) -> Path:
    """Post-primary check using DISCOVER-EEG's debiased wPLI metric."""
    import h5py
    import matplotlib
    from mne_connectivity import spectral_connectivity_epochs

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neurostat.project import load_project

    workflow = json.loads(
        (project_path.parent / "workflow_result.json").read_text(encoding="utf-8")
    )
    scout_stage = next(item for item in workflow["stages"] if item["stage"] == "scouts")
    manifest_path = Path(scout_stage["run_dir"]) / "scouts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    project = load_project(project_path)
    groups = {record.record_id: record.key.arm for record in project.records}
    results_dir = project_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_path = results_dir / "published_metric_subject_results.csv"
    rows: list[dict[str, Any]] = []
    group_matrices: dict[str, dict[str, list[np.ndarray]]] = {
        group: {band: [] for band in BANDS} for group in ("old", "young")
    }
    matrix_cache = results_dir / "published_metric_matrices.npz"
    cached: dict[str, np.ndarray] = {}
    if matrix_cache.is_file():
        with np.load(matrix_cache) as values:
            cached = {name: values[name] for name in values.files}
    else:
        for entry in manifest["records"]:
            path = Path(entry["path"])
            window_samples = round(2.0 * float(entry["sfreq"]))

            def windows(
                artifact_path: Path = path,
                samples_per_window: int = window_samples,
            ) -> Iterable[np.ndarray]:
                with h5py.File(artifact_path, "r") as handle:
                    data = handle["data"]
                    count = int(data.shape[1]) // samples_per_window
                    for index in range(count):
                        start = index * samples_per_window
                        yield np.asarray(
                            data[:, start : start + samples_per_window],
                            dtype=np.float64,
                        )

            result = spectral_connectivity_epochs(
                windows(),
                names=entry["labels"],
                method="wpli2_debiased",
                mode="multitaper",
                sfreq=float(entry["sfreq"]),
                fmin=tuple(edges[0] for edges in BANDS.values()),
                fmax=tuple(edges[1] for edges in BANDS.values()),
                faverage=True,
                mt_bandwidth=2.0,
                mt_adaptive=False,
                mt_low_bias=True,
                n_jobs=1,
                verbose=False,
            ).get_data(output="dense")
            matrices = np.moveaxis(result + result.transpose(1, 0, 2), -1, 0)
            for matrix in matrices:
                np.fill_diagonal(matrix, 0.0)
            cached[entry["record_id"]] = matrices
            print(
                f"computed published-metric sensitivity for {entry['record_id']}",
                flush=True,
            )
        np.savez_compressed(matrix_cache, **cast(dict[str, Any], cached))

    for entry in manifest["records"]:
        record_id = entry["record_id"]
        participant_id = "sub-" + record_id.split("-", 1)[0]
        group = groups[record_id]
        matrices = cached[record_id]
        triangle = np.triu_indices(matrices.shape[1], 1)
        for index, band in enumerate(BANDS):
            matrix = matrices[index]
            group_matrices[group][band].append(matrix)
            rows.append(
                {
                    "participant_id": participant_id,
                    "age_group": group,
                    "band": band,
                    "global_mean_debiased_wpli": float(matrix[triangle].mean()),
                }
            )
    with cache_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries: list[dict[str, Any]] = []
    for index, band in enumerate(BANDS):
        old = np.asarray(
            [
                row["global_mean_debiased_wpli"]
                for row in rows
                if row["band"] == band and row["age_group"] == "old"
            ]
        )
        young = np.asarray(
            [
                row["global_mean_debiased_wpli"]
                for row in rows
                if row["band"] == band and row["age_group"] == "young"
            ]
        )
        summaries.append(
            {
                "band": band,
                "old_mean": float(old.mean()),
                "young_mean": float(young.mean()),
                "old_minus_young": float(old.mean() - young.mean()),
                "cohens_d": _cohens_d(old, young),
                "one_sided_permutation_p": _permutation_p(
                    old, young, seed=RANDOM_SEED + 200 + index
                ),
            }
        )
    old_alpha = np.stack(group_matrices["old"]["alpha"])
    young_alpha = np.stack(group_matrices["young"]["alpha"])
    pooled = np.sqrt(
        (
            (len(old_alpha) - 1) * old_alpha.var(axis=0, ddof=1)
            + (len(young_alpha) - 1) * young_alpha.var(axis=0, ddof=1)
        )
        / (len(old_alpha) + len(young_alpha) - 2)
    )
    edge_d = np.divide(
        old_alpha.mean(axis=0) - young_alpha.mean(axis=0),
        pooled,
        out=np.zeros_like(pooled),
        where=pooled > 0,
    )
    triangle = np.triu_indices(edge_d.shape[0], 1)
    negative_fraction = float(np.mean(edge_d[triangle] < 0))
    alpha = next(item for item in summaries if item["band"] == "alpha")
    payload = {
        "schema": "neurostat.lemon_published_metric_sensitivity.v1",
        "status": "post-primary exploratory sensitivity; does not alter the failed PLI primary endpoint",
        "method": "mne-connectivity wpli2_debiased on unchanged V2 DK68 scout artifacts",
        "published_method": "DISCOVER-EEG used source-space debiased wPLI with a different atlas and inverse solver",
        "bands": summaries,
        "alpha_edge_negative_fraction": negative_fraction,
        "directional_alpha_pass": bool(
            alpha["old_minus_young"] < 0 and alpha["one_sided_permutation_p"] < 0.05
        ),
        "scout_manifest_sha256": _sha256(manifest_path),
    }
    _atomic_json(results_dir / "published_metric_sensitivity.json", payload)

    colours = {"young": "#147d92", "old": "#c6533d"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for group_index, group in enumerate(("young", "old")):
        values = np.asarray(
            [
                row["global_mean_debiased_wpli"]
                for row in rows
                if row["band"] == "alpha" and row["age_group"] == group
            ]
        )
        jitter = np.random.default_rng(RANDOM_SEED + 300 + group_index).uniform(
            -0.09, 0.09, len(values)
        )
        axes[0].scatter(
            np.full(len(values), group_index) + jitter,
            values,
            color=colours[group],
            alpha=0.72,
            s=25,
        )
        axes[0].boxplot(
            values,
            positions=[group_index],
            widths=0.38,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": colours[group]},
            medianprops={"color": "#202020"},
        )
    axes[0].set_xticks([0, 1], ["Young", "Old"])
    axes[0].set_ylabel("Global mean DK68 debiased wPLI")
    axes[0].set_title(
        f"Published-metric sensitivity\nd = {alpha['cohens_d']:.2f}, "
        f"one-sided p = {alpha['one_sided_permutation_p']:.4f}"
    )
    axes[1].bar(
        [item["band"].title() for item in summaries],
        [item["cohens_d"] for item in summaries],
        color=["#9aa1a6", "#c6533d", "#9aa1a6", "#9aa1a6"],
    )
    axes[1].axhline(0, color="#202020", linewidth=0.8)
    axes[1].set_ylabel("Cohen's d (old - young)")
    axes[1].set_title(
        f"Debiased wPLI band effects\n{negative_fraction:.1%} of alpha edges negative"
    )
    fig.suptitle(
        "Post-primary metric sensitivity on unchanged NeuroStat V2 scouts",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(
        results_dir / "published_metric_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    return results_dir / "published_metric_sensitivity.json"


def publish_results(project_path: Path, destination: Path | None = None) -> Path:
    if project_path.parent.name != "replication":
        raise RuntimeError("only the prespecified replication cohort can be published")
    source = project_path.parent / "results"
    target = (
        destination
        or PROJECT_ROOT / "scripts" / "validation" / "lemon_discover_eeg" / "results"
    )
    target.mkdir(parents=True, exist_ok=True)
    names = (
        "subject_band_results.csv",
        "band_group_summary.csv",
        "alpha_edge_effect_matrix.csv",
        "lemon_v2_replication_dashboard.png",
        "validation_results.json",
    )
    optional = (
        "published_metric_subject_results.csv",
        "published_metric_sensitivity.json",
        "published_metric_sensitivity.png",
    )
    copied = list(names)
    for name in (*names, *optional):
        if not (source / name).is_file():
            continue
        shutil.copy2(source / name, target / name)
        if name in optional:
            copied.append(name)
    shutil.copy2(
        project_path.parent / "validation_protocol.json",
        target / "validation_protocol.json",
    )
    report = json.loads(
        (target / "validation_results.json").read_text(encoding="utf-8")
    )
    alpha = next(item for item in report["bands"] if item["band"] == "alpha")
    apf = report["alpha_peak_frequency"]
    sensitivity_path = target / "published_metric_sensitivity.json"
    sensitivity_text = ""
    if sensitivity_path.is_file():
        sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
        sensitivity_alpha = next(
            item for item in sensitivity["bands"] if item["band"] == "alpha"
        )
        sensitivity_text = (
            "\n\nThe post-primary debiased-wPLI sensitivity also did not reproduce a "
            "global age decrease: older minus younger was "
            f"{sensitivity_alpha['old_minus_young']:.6f} "
            f"(d = {sensitivity_alpha['cohens_d']:.3f}; one-sided p = "
            f"{sensitivity_alpha['one_sided_permutation_p']:.6f}). "
            "This sensitivity does not replace the failed primary endpoint."
        )
    status = "PASSED" if report["primary_replication_passed"] else "DID NOT PASS"
    (target / "README.md").write_text(
        f"# LEMON / DISCOVER-EEG Replication\n\n![Validation dashboard](lemon_v2_replication_dashboard.png)\n\n"
        f"**Prespecified primary criterion: {status}.** Older minus younger global alpha PLI "
        f"was {alpha['old_minus_young']:.6f} (Cohen's d = {alpha['cohens_d']:.3f}; "
        f"one-sided permutation p = {alpha['one_sided_permutation_p']:.6f}). "
        "The effect was opposite to the frozen old-lower direction.\n\n"
        "**Independent positive control: PASSED.** Older minus younger alpha peak frequency "
        f"was {apf['old_minus_young_hz']:.3f} Hz (d = {apf['cohens_d']:.3f}; "
        f"one-sided permutation p = {apf['one_sided_permutation_p']:.6f})."
        f"{sensitivity_text}\n\n"
        "DISCOVER-EEG reported selected edgewise Bayesian effects rather than this frozen "
        "global-mean endpoint. This is a directional conceptual replication, not an exact "
        "numerical reproduction. "
        "Raw EEG and intermediate source artifacts are excluded from Git; the aggregate tables, "
        "figure, protocol, and hashes are included here.\n",
        encoding="utf-8",
        newline="\n",
    )
    published = [*copied, "validation_protocol.json", "README.md"]
    _atomic_json(
        target / "publication_manifest.json",
        {
            "schema": "neurostat.lemon_publication_bundle.v1",
            "dataset_doi": DATASET_DOI,
            "published_target_doi": DISCOVER_DOI,
            "files": {name: _sha256(target / name) for name in published},
        },
    )
    return target


def _default_root() -> Path:
    configured = os.environ.get("NEUROSTAT_LEMON_VALIDATION_DIR")
    return (
        Path(configured).expanduser()
        if configured
        else PROJECT_ROOT.parent / "NeuroStat-v2-public-validation" / "lemon"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "metadata",
            "download",
            "prepare",
            "project",
            "run",
            "summarize",
            "sensitivity",
            "publish",
            "all",
        ),
    )
    parser.add_argument(
        "--cohort", choices=("pilot", "replication", "full"), default="pilot"
    )
    parser.add_argument("--data-root", type=Path, default=_default_root())
    parser.add_argument("--download-workers", type=int, default=6)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    project_path = root / "projects" / args.cohort / "project.yaml"
    if args.action in {"metadata", "all"}:
        freeze_metadata(root, args.cohort, args.download_workers)
    if args.action in {"download", "all"}:
        download_dataset(root, args.cohort, args.download_workers)
    if args.action in {"prepare", "all"}:
        prepare_dataset(root, args.cohort)
    if args.action in {"project", "all"}:
        project_path = build_project(root, args.cohort)
    if args.action in {"run", "all"}:
        run_v2(project_path)
    if args.action in {"summarize", "all"}:
        summarize(project_path)
    if args.action == "sensitivity":
        published_metric_sensitivity(project_path)
    if args.action == "publish" or (
        args.action == "all" and args.cohort == "replication"
    ):
        publish_results(project_path)
    print(f"completed {args.action} for {args.cohort}: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
