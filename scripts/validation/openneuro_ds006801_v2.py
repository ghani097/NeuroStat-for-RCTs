#!/usr/bin/env python
"""Reproducible public-data validation of NeuroStat v2 on OpenNeuro ds006801.

The fixed analysis compares every available active-learning participant with
every control participant at pre and post. A two-participant pilot exercises the
same code and project contract before the full 13-participant run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_ID = "ds006801"
DATASET_VERSION = "1.0.0"
DATASET_DOI = "10.18112/openneuro.ds006801.v1.0.0"
DATASET_SNAPSHOT = "c4e91b19132fed86ee24a42ec5105bf14f961011"
S3_ROOT = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}"
GITHUB_RAW_ROOT = (
    "https://raw.githubusercontent.com/OpenNeuroDatasets/"
    f"{DATASET_ID}/{DATASET_SNAPSHOT}"
)
GROUPS: dict[str, tuple[str, ...]] = {
    "active": ("sub-01", "sub-05", "sub-07", "sub-10", "sub-15", "sub-16"),
    "control": ("sub-02", "sub-09", "sub-12", "sub-14", "sub-17", "sub-18", "sub-22"),
}
PILOT_GROUPS: dict[str, tuple[str, ...]] = {
    "active": ("sub-01",),
    "control": ("sub-02",),
}
SESSIONS = ("pre", "post")
SOURCE_EXTENSIONS = ("vhdr", "vmrk", "eeg")
ANNEX_POINTER = re.compile(
    r"SHA256E-s(?P<size>\d+)--(?P<sha>[0-9a-f]{64})\.[A-Za-z0-9]+"
)
ARTIFACT_LABELS = {
    "muscle artifact",
    "eye blink",
    "heart beat",
    "line noise",
    "channel noise",
}


@dataclass(frozen=True, slots=True)
class SourceSpec:
    relative_path: str
    size: int
    sha256: str


def _cohort(name: str) -> dict[str, tuple[str, ...]]:
    return PILOT_GROUPS if name == "pilot" else GROUPS


def _record_stem(subject: str, session: str) -> str:
    return f"{subject}_ses-{session}_task-rest_eeg"


def _relative_source(subject: str, session: str, extension: str) -> str:
    return f"{subject}/ses-{session}/eeg/{_record_stem(subject, session)}.{extension}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(value), handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


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


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree-uninstalled"


def _read_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def _source_spec(relative_path: str) -> SourceSpec:
    pointer = _read_url(f"{GITHUB_RAW_ROOT}/{relative_path}").decode("ascii").strip()
    match = ANNEX_POINTER.search(pointer)
    if match is None:
        raise RuntimeError(
            f"pinned OpenNeuro path is not a SHA-256 annex pointer: {relative_path}"
        )
    return SourceSpec(relative_path, int(match.group("size")), match.group("sha"))


def _quarantine(path: Path, reason: str) -> Path:
    stamp = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{time.time_ns()}"
    target = path.with_name(f"{path.name}.invalid-{reason}-{stamp}")
    os.replace(path, target)
    return target


def _download(spec: SourceSpec, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual = _sha256(destination)
        if destination.stat().st_size == spec.size and actual == spec.sha256:
            return {
                "path": str(destination),
                "size": spec.size,
                "sha256": actual,
                "reused": True,
            }
        _quarantine(destination, "checksum")

    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.is_file() else 0
    if start > spec.size:
        _quarantine(partial, "oversize")
        start = 0
    elif start == spec.size:
        actual_sha = _sha256(partial)
        if actual_sha == spec.sha256:
            os.replace(partial, destination)
            return {
                "path": str(destination),
                "size": spec.size,
                "sha256": actual_sha,
                "reused": False,
            }
        _quarantine(partial, "checksum")
        start = 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    request = urllib.request.Request(f"{S3_ROOT}/{spec.relative_path}", headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"download failed for {spec.relative_path}: HTTP {exc.code}"
        ) from exc
    if start and response.status != 206:
        response.close()
        _quarantine(partial, "range-ignored")
        start = 0
        response = urllib.request.urlopen(
            f"{S3_ROOT}/{spec.relative_path}", timeout=120
        )
    mode = "ab" if start else "wb"
    with response, partial.open(mode) as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
    actual_size = partial.stat().st_size
    actual_sha = _sha256(partial)
    if actual_size != spec.size or actual_sha != spec.sha256:
        retained = _quarantine(partial, "checksum")
        raise RuntimeError(
            f"download integrity failure for {spec.relative_path}; retained at {retained}"
        )
    os.replace(partial, destination)
    return {
        "path": str(destination),
        "size": actual_size,
        "sha256": actual_sha,
        "reused": False,
    }


def download_dataset(root: Path, cohort_name: str) -> Path:
    raw_root = root / "raw" / DATASET_ID
    provenance = root / "provenance"
    metadata: dict[str, str] = {}
    for name in (
        "README",
        "dataset_description.json",
        "participants.tsv",
        "participants.json",
    ):
        content = _read_url(f"{GITHUB_RAW_ROOT}/{name}")
        target = raw_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        metadata[name] = hashlib.sha256(content).hexdigest()

    participants = (raw_root / "participants.tsv").read_text(encoding="utf-8")
    for arm, subjects in _cohort(cohort_name).items():
        for subject in subjects:
            if f"{subject}\t{arm.title()}" not in participants:
                raise RuntimeError(
                    f"pinned participants.tsv does not confirm {subject} as {arm}"
                )

    entries: list[dict[str, Any]] = []
    for arm, subjects in _cohort(cohort_name).items():
        for subject in subjects:
            for session in SESSIONS:
                for extension in SOURCE_EXTENSIONS:
                    relative = _relative_source(subject, session, extension)
                    spec = _source_spec(relative)
                    result = _download(spec, raw_root / relative)
                    entries.append(
                        {"arm": arm, "subject": subject, "session": session, **result}
                    )
                    print(f"downloaded/verified {relative}", flush=True)
    manifest = provenance / f"download_manifest_{cohort_name}.json"
    _atomic_json(
        manifest,
        {
            "schema": "neurostat.public_download.v1",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "dataset_doi": DATASET_DOI,
            "snapshot": DATASET_SNAPSHOT,
            "cohort": cohort_name,
            "groups": _cohort(cohort_name),
            "metadata_sha256": metadata,
            "files": entries,
        },
    )
    return manifest


def _detect_bad_channels(raw: Any, threshold: float = 6.0) -> list[str]:
    values = np.log(np.maximum(np.var(raw.get_data(), axis=1), np.finfo(float).tiny))
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    if mad == 0:
        return []
    robust_z = 0.67448975 * (values - centre) / mad
    return [
        raw.ch_names[index] for index in np.flatnonzero(np.abs(robust_z) > threshold)
    ]


def _matches_cleaning_contract(qc: dict[str, Any]) -> bool:
    if qc.get("schema") != "neurostat.public_cleaning_qc.v1":
        return False
    cleaning = qc.get("cleaning", {})
    expected = {
        "resample_hz": 200.0,
        "bandpass_hz": [1.0, 45.0],
        "reference": "average_applied",
        "bad_channel_method": "log-variance robust z > 6",
        "ica_method": "extended_infomax",
        "ica_components": 25,
        "ica_random_state": 42,
        "window_seconds": 4.0,
        "window_peak_to_peak_limit_uv": 150.0,
        "window_absolute_peak_limit_uv": 150.0,
        "window_log_rms_robust_z_limit": 8.0,
    }
    return all(cleaning.get(key) == value for key, value in expected.items())


def _clean_recording(source: Path, destination: Path) -> dict[str, Any]:
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    raw = mne.io.read_raw_brainvision(source, preload=True, verbose="ERROR")
    raw.pick("eeg")
    original = {
        "channels": len(raw.ch_names),
        "sfreq": float(raw.info["sfreq"]),
        "samples": int(raw.n_times),
        "duration_seconds": float(raw.times[-1]),
    }
    if len(raw.ch_names) != 31 or not raw.info.get("dig"):
        raise RuntimeError(f"unexpected channel geometry in {source}")
    raw.resample(200.0, npad="auto", verbose="ERROR")
    raw.filter(1.0, 45.0, fir_design="firwin", verbose="ERROR")
    bad_channels = _detect_bad_channels(raw)
    if len(bad_channels) > 3:
        raise RuntimeError(
            f"more than three robust-variance bad channels in {source}: {bad_channels}"
        )
    if bad_channels:
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    ica = ICA(
        n_components=25,
        method="infomax",
        fit_params={"extended": True},
        random_state=42,
        max_iter="auto",
    )
    ica.fit(raw, reject={"eeg": 200e-6}, reject_by_annotation=True, verbose="ERROR")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The provided Raw instance is not filtered between 1 and 100 Hz.*",
            category=RuntimeWarning,
        )
        classified = label_components(raw, ica, method="iclabel")
    labels = [str(value) for value in classified["labels"]]
    confidence = np.asarray(classified["y_pred_proba"], dtype=float)
    if len(labels) != ica.n_components_ or confidence.shape != (ica.n_components_,):
        raise RuntimeError(f"unexpected ICLabel output for {source}")
    excluded = [index for index, label in enumerate(labels) if label in ARTIFACT_LABELS]
    if not excluded or len(excluded) >= ica.n_components_:
        raise RuntimeError(
            f"implausible ICLabel exclusion count for {source}: {len(excluded)}"
        )
    ica.exclude = excluded
    ica.apply(raw, verbose="ERROR")

    window_samples = round(4.0 * raw.info["sfreq"])
    total_windows = raw.n_times // window_samples
    usable = total_windows * window_samples
    windows = raw.get_data()[:, :usable].reshape(
        len(raw.ch_names), total_windows, window_samples
    )
    peak_to_peak = np.ptp(windows, axis=2).max(axis=0)
    absolute_peak = np.abs(windows).max(axis=(0, 2))
    rms = np.sqrt(np.mean(np.square(windows), axis=2))
    log_rms = np.log(np.maximum(rms, np.finfo(float).tiny))
    centre = np.median(log_rms, axis=1, keepdims=True)
    mad = np.median(np.abs(log_rms - centre), axis=1, keepdims=True)
    robust_z = np.divide(
        0.67448975 * (log_rms - centre),
        mad,
        out=np.zeros_like(log_rms),
        where=mad > 0,
    )
    keep = (
        (peak_to_peak <= 150e-6)
        & (absolute_peak <= 150e-6)
        & (np.max(np.abs(robust_z), axis=0) <= 8.0)
    )
    retained = int(keep.sum())
    retained_fraction = retained / total_windows
    if retained < 60 or retained_fraction < 0.50:
        raise RuntimeError(
            f"insufficient artifact-free data in {source}: {retained}/{total_windows} windows"
        )
    clean_data = windows[:, keep, :].reshape(
        len(raw.ch_names), retained * window_samples
    )
    if not np.isfinite(clean_data).all():
        raise RuntimeError(f"non-finite cleaned data in {source}")
    info = raw.info.copy()
    info["bads"] = []
    cleaned = mne.io.RawArray(clean_data, info, verbose="ERROR")
    cleaned.set_meas_date(None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.save(destination, overwrite=True, verbose="ERROR")
    return {
        "schema": "neurostat.public_cleaning_qc.v1",
        "source": str(source),
        "source_files": {
            extension: {
                "path": str(source.with_suffix(f".{extension}")),
                "sha256": _sha256(source.with_suffix(f".{extension}")),
            }
            for extension in SOURCE_EXTENSIONS
        },
        "original": original,
        "cleaning": {
            "resample_hz": 200.0,
            "bandpass_hz": [1.0, 45.0],
            "reference": "average_applied",
            "bad_channel_method": "log-variance robust z > 6",
            "bad_channels_interpolated": bad_channels,
            "ica_method": "extended_infomax",
            "ica_components": int(ica.n_components_),
            "ica_random_state": 42,
            "iclabel_backend": "onnxruntime_or_pytorch",
            "iclabel_frequency_caveat": (
                "The public recordings are natively low-pass filtered at 50 Hz, so "
                "ICLabel's 1-100 Hz training recommendation cannot be met."
            ),
            "iclabel_labels": labels,
            "iclabel_predicted_confidence": confidence.tolist(),
            "ica_excluded_indices": excluded,
            "ica_excluded_labels": [labels[index] for index in excluded],
            "window_seconds": 4.0,
            "window_peak_to_peak_limit_uv": 150.0,
            "window_absolute_peak_limit_uv": 150.0,
            "window_log_rms_robust_z_limit": 8.0,
            "total_windows": total_windows,
            "retained_windows": retained,
            "retained_fraction": retained_fraction,
        },
        "output": {
            "path": str(destination),
            "sha256": _sha256(destination),
            "channels": len(cleaned.ch_names),
            "sfreq": float(cleaned.info["sfreq"]),
            "samples": int(cleaned.n_times),
            "duration_seconds": float(cleaned.times[-1]),
        },
    }


def prepare_dataset(root: Path, cohort_name: str) -> Path:
    raw_root = root / "raw" / DATASET_ID
    prepared_root = root / "prepared" / cohort_name
    rows: list[dict[str, str]] = []
    for arm, subjects in _cohort(cohort_name).items():
        for subject in subjects:
            for session in SESSIONS:
                stem = _record_stem(subject, session)
                source = raw_root / _relative_source(subject, session, "vhdr")
                if not source.is_file():
                    raise RuntimeError(f"missing downloaded source: {source}")
                destination = (
                    prepared_root / arm / subject / session / f"{stem}_cleaned_raw.fif"
                )
                qc_path = destination.with_suffix(".qc.json")
                reuse = False
                if destination.is_file() and qc_path.is_file():
                    qc = json.loads(qc_path.read_text(encoding="utf-8"))
                    expected_sources = {
                        extension: _sha256(source.with_suffix(f".{extension}"))
                        for extension in SOURCE_EXTENSIONS
                    }
                    recorded_sources = {
                        extension: item.get("sha256")
                        for extension, item in qc.get("source_files", {}).items()
                    }
                    reuse = (
                        _matches_cleaning_contract(qc)
                        and qc.get("output", {}).get("sha256") == _sha256(destination)
                        and recorded_sources == expected_sources
                    )
                if not reuse:
                    qc = _clean_recording(source, destination)
                    _atomic_json(qc_path, qc)
                print(f"prepared {subject} {session} ({arm})", flush=True)
                rows.append(
                    {
                        "record_id": f"{subject.removeprefix('sub-')}-{session}-mixed-rest",
                        "participant_id": subject,
                        "arm": arm,
                        "session": session,
                        "condition": "mixed_rest",
                        "data_kind": "raw",
                        "format": "fif",
                        "path": str(destination),
                        "analysis_profile": "rest",
                        "data_unit": "V",
                    }
                )
    manifest = prepared_root / "records.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def _relative_to_project(project_dir: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return Path(os.path.relpath(resolved, project_dir.resolve())).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_project(root: Path, cohort_name: str) -> Path:
    import mne
    import yaml

    from neurostat.gui.fsaverage import provision_fsaverage
    from neurostat.project import load_project

    records_source = root / "prepared" / cohort_name / "records.csv"
    if not records_source.is_file():
        raise RuntimeError(f"missing prepared record manifest: {records_source}")
    project_dir = root / "projects" / cohort_name
    project_dir.mkdir(parents=True, exist_ok=True)
    records_target = project_dir / "records.csv"
    rows = list(csv.DictReader(records_source.open(encoding="utf-8", newline="")))
    for row in rows:
        row["path"] = _relative_to_project(project_dir, Path(row["path"]))
    with records_target.open("w", encoding="utf-8", newline="") as handle:
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
            "id": "openneuro-ds006801-v2-validation",
            "arms": ["active", "control"],
            "sessions": ["pre", "post"],
            "conditions": ["mixed_rest"],
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
                    "bands": {
                        "theta": [4.0, 8.0],
                        "alpha": [8.0, 13.0],
                        "beta": [13.0, 30.0],
                    },
                    "window_seconds": 4.0,
                    "tail_policy": "drop",
                    "mt_bandwidth": 4.0,
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
    _atomic_json(
        project_dir / "validation_protocol.json",
        {
            "schema": "neurostat.ds006801_protocol.v1",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "dataset_doi": DATASET_DOI,
            "snapshot": DATASET_SNAPSHOT,
            "cohort": cohort_name,
            "groups": _cohort(cohort_name),
            "cohort_selection": (
                "All participants assigned to the active-learning or control arm "
                "with both pre and post recordings in the pinned release."
            ),
            "sessions": SESSIONS,
            "condition": "mixed_rest",
            "condition_definition": (
                "The complete recording combines two eyes-open and two eyes-closed "
                "rest blocks. Block markers are absent, so no block-level condition "
                "labels are inferred."
            ),
            "preprocessing": {
                "native_sampling_hz": 500.0,
                "native_lowpass_hz": 50.0,
                "resample_hz": 200.0,
                "bandpass_hz": [1.0, 45.0],
                "reference": "average_applied",
                "bad_channel_rule": "absolute log-variance robust z > 6",
                "maximum_interpolated_channels": 3,
                "ica_method": "extended_infomax",
                "ica_components": 25,
                "ica_random_state": 42,
                "ica_fit_reject_peak_to_peak_uv": 200.0,
                "iclabel_excluded_classes": sorted(ARTIFACT_LABELS),
                "iclabel_frequency_caveat": (
                    "Native 50 Hz low-pass prevents meeting ICLabel's 1-100 Hz "
                    "training recommendation."
                ),
                "qc_window_seconds": 4.0,
                "qc_peak_to_peak_limit_uv": 150.0,
                "qc_absolute_peak_limit_uv": 150.0,
                "qc_log_rms_robust_z_limit": 8.0,
                "minimum_retained_windows": 60,
                "minimum_retained_fraction": 0.50,
            },
            "source_model": {
                "template": "fsaverage",
                "spacing": "ico3",
                "atlas": "aparc_DK68",
                "inverse_method": "MNE",
                "snr": 3.0,
                "orientation": "normal",
                "label_mode": "mean_flip",
                "noise_covariance": "empirical",
            },
            "connectivity": {
                "method": "continuous_multitaper_pli",
                "bands_hz": {
                    "theta": [4.0, 8.0],
                    "alpha": [8.0, 13.0],
                    "beta": [13.0, 30.0],
                },
                "window_seconds": 4.0,
                "tail_policy": "drop",
                "multitaper_bandwidth_hz": 4.0,
            },
            "group_summary": {
                "measure": "mean upper-triangle DK68 PLI",
                "contrast": "active pre/post change minus control pre/post change",
                "inference": "two-sided exact label permutation conditional on arm sizes",
                "multiplicity": "Benjamini-Hochberg FDR across three bands",
                "effect_size": "pooled-SD Cohen's d on participant change scores",
            },
            "software": {
                name: _package_version(name)
                for name in (
                    "neurostat-rcts",
                    "numpy",
                    "scipy",
                    "mne",
                    "mne-icalabel",
                    "onnxruntime",
                    "mne-connectivity",
                )
            },
            "primary_validation_claim": "public two-arm longitudinal V2 workflow feasibility and artifact integrity",
            "inferential_status": "exploratory; no directional connectivity hypothesis was preregistered",
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
    payload = {
        "schema": "neurostat.ds006801_workflow_result.v1",
        "project": str(project_path.resolve()),
        "started_unix": started,
        "finished_unix": time.time(),
        "success": result.success,
        "stages": [
            {
                "stage": outcome.stage.value,
                "state": outcome.state.value,
                "run_id": outcome.run_id,
                "run_dir": str(outcome.run_dir) if outcome.run_dir else None,
                "reused": outcome.reused,
            }
            for outcome in result.outcomes
        ],
    }
    target = project_path.parent / "workflow_result.json"
    _atomic_json(target, payload)
    if not result.success:
        raise RuntimeError(
            f"V2 workflow did not complete successfully; inspect {target}"
        )
    return target


def _exact_permutation_p(
    changes: np.ndarray, groups: np.ndarray, observed: float
) -> float:
    active_count = int(np.sum(groups == "active"))
    values = []
    for active_indices in itertools.combinations(range(len(changes)), active_count):
        mask = np.zeros(len(changes), dtype=bool)
        mask[list(active_indices)] = True
        values.append(float(changes[mask].mean() - changes[~mask].mean()))
    values_array = np.asarray(values)
    return float(np.mean(np.abs(values_array) >= abs(observed) - 1e-15))


def _fdr_bh(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(values) - reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def summarize(project_path: Path) -> Path:
    import h5py
    import matplotlib
    from scipy import stats

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neurostat.connectivity import verify_connectivity_artifact
    from neurostat.project import load_project

    workflow_path = project_path.parent / "workflow_result.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    connectivity = next(
        item for item in workflow["stages"] if item["stage"] == "connectivity"
    )
    manifest_path = Path(connectivity["run_dir"]) / "connectivity" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {
        entry["record_id"]: Path(entry["path"]) for entry in manifest["records"]
    }
    project = load_project(project_path)
    rows: list[dict[str, Any]] = []
    for record in project.records:
        path = artifacts[record.record_id]
        header = verify_connectivity_artifact(path)
        with h5py.File(path, "r") as handle:
            matrices = np.asarray(handle["matrix"][:], dtype=float)
        triangle = np.triu_indices(len(header.labels), k=1)
        for index, band in enumerate(header.bands):
            rows.append(
                {
                    "participant_id": record.key.participant_id,
                    "arm": record.key.arm,
                    "session": record.key.session,
                    "band": band,
                    "global_mean_pli": float(matrices[index][triangle].mean()),
                    "artifact_sha256": _sha256(path),
                }
            )

    results_dir = project_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    subject_csv = results_dir / "subject_band_results.csv"
    with subject_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries: list[dict[str, Any]] = []
    bands = sorted({row["band"] for row in rows})
    for band in bands:
        band_rows = [row for row in rows if row["band"] == band]
        by_key = {
            (row["participant_id"], row["session"]): row["global_mean_pli"]
            for row in band_rows
        }
        participants = sorted({row["participant_id"] for row in band_rows})
        groups = np.asarray(
            [
                next(
                    row["arm"] for row in band_rows if row["participant_id"] == subject
                )
                for subject in participants
            ]
        )
        changes = np.asarray(
            [
                by_key[(subject, "post")] - by_key[(subject, "pre")]
                for subject in participants
            ]
        )
        active = changes[groups == "active"]
        control = changes[groups == "control"]
        active_pre = np.asarray(
            [
                by_key[(subject, "pre")]
                for subject in participants
                if groups[participants.index(subject)] == "active"
            ]
        )
        active_post = np.asarray(
            [
                by_key[(subject, "post")]
                for subject in participants
                if groups[participants.index(subject)] == "active"
            ]
        )
        control_pre = np.asarray(
            [
                by_key[(subject, "pre")]
                for subject in participants
                if groups[participants.index(subject)] == "control"
            ]
        )
        control_post = np.asarray(
            [
                by_key[(subject, "post")]
                for subject in participants
                if groups[participants.index(subject)] == "control"
            ]
        )
        observed = float(active.mean() - control.mean())
        inferential = len(active) >= 2 and len(control) >= 2
        t_stat = p_welch = effect = p_exact = math.nan
        if inferential:
            t_stat, p_welch = (
                float(value)
                for value in stats.ttest_ind(active, control, equal_var=False)
            )
            pooled = math.sqrt(
                (
                    (len(active) - 1) * active.var(ddof=1)
                    + (len(control) - 1) * control.var(ddof=1)
                )
                / (len(active) + len(control) - 2)
            )
            effect = observed / pooled if pooled > 0 else math.nan
            p_exact = _exact_permutation_p(changes, groups, observed)
        summaries.append(
            {
                "band": band,
                "active_n": len(active),
                "control_n": len(control),
                "active_pre_mean": float(active_pre.mean()),
                "active_pre_sd": float(active_pre.std(ddof=1))
                if len(active_pre) > 1
                else math.nan,
                "active_post_mean": float(active_post.mean()),
                "active_post_sd": float(active_post.std(ddof=1))
                if len(active_post) > 1
                else math.nan,
                "control_pre_mean": float(control_pre.mean()),
                "control_pre_sd": float(control_pre.std(ddof=1))
                if len(control_pre) > 1
                else math.nan,
                "control_post_mean": float(control_post.mean()),
                "control_post_sd": float(control_post.std(ddof=1))
                if len(control_post) > 1
                else math.nan,
                "active_change_mean": float(active.mean()),
                "active_change_sd": float(active.std(ddof=1))
                if len(active) > 1
                else math.nan,
                "control_change_mean": float(control.mean()),
                "control_change_sd": float(control.std(ddof=1))
                if len(control) > 1
                else math.nan,
                "difference_in_change": observed,
                "welch_t": t_stat,
                "welch_p": p_welch,
                "cohens_d": effect,
                "exact_permutation_p": p_exact,
                "inferential_ready": inferential,
            }
        )

    inferential_indices = [
        index for index, item in enumerate(summaries) if item["inferential_ready"]
    ]
    if inferential_indices:
        adjusted = _fdr_bh(
            [summaries[index]["exact_permutation_p"] for index in inferential_indices]
        )
        for index, value in zip(inferential_indices, adjusted):
            summaries[index]["exact_permutation_fdr_bh"] = value
    for item in summaries:
        item.setdefault("exact_permutation_fdr_bh", math.nan)

    summary_csv = results_dir / "band_change_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    qc_files = sorted(
        (project_path.parents[2] / "prepared" / project_path.parent.name).rglob(
            "*.qc.json"
        )
    )
    qc_values = [json.loads(path.read_text(encoding="utf-8")) for path in qc_files]
    retained_fractions = [item["cleaning"]["retained_fraction"] for item in qc_values]
    interpolated_records = sum(
        bool(item["cleaning"]["bad_channels_interpolated"]) for item in qc_values
    )
    _atomic_json(
        results_dir / "validation_results.json",
        {
            "schema": "neurostat.ds006801_results.v1",
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "dataset_doi": DATASET_DOI,
            "dataset_snapshot": DATASET_SNAPSHOT,
            "project": str(project_path.resolve()),
            "project_sha256": _sha256(project_path),
            "connectivity_manifest": str(manifest_path.resolve()),
            "connectivity_manifest_sha256": _sha256(manifest_path),
            "records_verified": len(artifacts),
            "participants": len({row["participant_id"] for row in rows}),
            "workflow_duration_seconds": workflow["finished_unix"]
            - workflow["started_unix"],
            "stage_runs": workflow["stages"],
            "preprocessing_qc": {
                "records": len(qc_values),
                "minimum_retained_fraction": min(retained_fractions),
                "maximum_retained_fraction": max(retained_fractions),
                "records_with_interpolated_channels": interpolated_records,
            },
            "results": summaries,
            "interpretation": "Exploratory global PLI change estimates; pipeline validation is based on reproducibility, completion, and artifact integrity, not a directional efficacy claim.",
        },
    )

    fig, axes = plt.subplots(
        1, len(bands), figsize=(4.1 * len(bands), 4.2), sharey=True
    )
    if len(bands) == 1:
        axes = [axes]
    colours = {"active": "#087f8c", "control": "#b24c63"}
    for axis, band in zip(axes, bands):
        band_rows = [row for row in rows if row["band"] == band]
        for arm, xbase in (("active", 0), ("control", 3)):
            subjects = sorted(
                {row["participant_id"] for row in band_rows if row["arm"] == arm}
            )
            for subject in subjects:
                values = [
                    next(
                        row["global_mean_pli"]
                        for row in band_rows
                        if row["participant_id"] == subject
                        and row["session"] == session
                    )
                    for session in SESSIONS
                ]
                axis.plot(
                    [xbase, xbase + 1],
                    values,
                    color=colours[arm],
                    alpha=0.65,
                    linewidth=1.2,
                )
                axis.scatter(
                    [xbase, xbase + 1], values, color=colours[arm], s=18, zorder=3
                )
        summary = next(item for item in summaries if item["band"] == band)
        p_value = summary["exact_permutation_p"]
        annotation = f"Difference in change = {summary['difference_in_change']:.3f}"
        if math.isfinite(p_value):
            annotation += f" | exact p = {p_value:.3f}"
        axis.set_xticks(
            [0, 1, 3, 4],
            ["Active\npre", "Active\npost", "Control\npre", "Control\npost"],
        )
        axis.set_title(f"{band.title()}\n{annotation}", fontsize=10.5)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axes[0].set_ylabel("Global mean source PLI")
    fig.suptitle(
        "OpenNeuro ds006801: exploratory NeuroStat v2 validation", fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(results_dir / "global_pli_pre_post.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return results_dir / "validation_results.json"


def publish_results(project_path: Path, destination: Path | None = None) -> Path:
    if project_path.parent.name != "full":
        raise RuntimeError("only the fixed full cohort can be published")
    source = project_path.parent / "results"
    if not (source / "validation_results.json").is_file():
        raise RuntimeError(f"missing summarized full-cohort results: {source}")
    target = destination or (
        PROJECT_ROOT / "scripts" / "validation" / "openneuro_ds006801" / "results"
    )
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "subject_band_results.csv",
        "band_change_summary.csv",
        "global_pli_pre_post.png",
    ):
        shutil.copy2(source / name, target / name)

    portable = json.loads(
        (source / "validation_results.json").read_text(encoding="utf-8")
    )
    portable["project"] = "external/projects/full/project.yaml"
    connectivity_run = next(
        item for item in portable["stage_runs"] if item["stage"] == "connectivity"
    )
    portable["connectivity_manifest"] = (
        "external/projects/full/output/runs/"
        f"connectivity-{connectivity_run['run_id']}/connectivity/manifest.json"
    )
    for stage in portable["stage_runs"]:
        stage["run_dir"] = (
            "external/projects/full/output/runs/" + Path(stage["run_dir"]).name
        )
    portable["source_data_included"] = False
    _atomic_json(target / "validation_results.json", portable)
    shutil.copy2(
        project_path.parent / "validation_protocol.json",
        target / "validation_protocol.json",
    )
    published = [
        "band_change_summary.csv",
        "global_pli_pre_post.png",
        "subject_band_results.csv",
        "validation_protocol.json",
        "validation_results.json",
    ]
    _atomic_json(
        target / "publication_manifest.json",
        {
            "schema": "neurostat.ds006801_publication_bundle.v1",
            "dataset_doi": DATASET_DOI,
            "files": {name: _sha256(target / name) for name in published},
        },
    )
    return target


def _default_root() -> Path:
    configured = os.environ.get("NEUROSTAT_PUBLIC_VALIDATION_DIR")
    if configured:
        return Path(configured).expanduser()
    return PROJECT_ROOT.parent / "NeuroStat-v2-public-validation" / DATASET_ID


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "download",
            "prepare",
            "project",
            "run",
            "summarize",
            "publish",
            "all",
        ),
    )
    parser.add_argument("--cohort", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--data-root", type=Path, default=_default_root())
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    project_path = root / "projects" / args.cohort / "project.yaml"
    if args.action in {"download", "all"}:
        download_dataset(root, args.cohort)
    if args.action in {"prepare", "all"}:
        prepare_dataset(root, args.cohort)
    if args.action in {"project", "all"}:
        project_path = build_project(root, args.cohort)
    if args.action in {"run", "all"}:
        if not project_path.is_file():
            raise RuntimeError(f"missing project: {project_path}")
        run_v2(project_path)
    if args.action in {"summarize", "all"}:
        if not project_path.is_file():
            raise RuntimeError(f"missing project: {project_path}")
        summarize(project_path)
    if args.action == "publish" or (args.action == "all" and args.cohort == "full"):
        if not project_path.is_file():
            raise RuntimeError(f"missing project: {project_path}")
        publish_results(project_path)
    print(f"completed {args.action} for {args.cohort}: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
