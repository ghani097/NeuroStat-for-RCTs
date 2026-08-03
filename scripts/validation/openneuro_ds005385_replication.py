#!/usr/bin/env python
"""Download and validate NeuroStat on longitudinal OpenNeuro ds005385 EEG.

Raw EEG and BIDS sidecars are required to live outside the Git worktree. Only
checksummed manifests and compact result/provenance files are written under
``scripts/validation/provenance``.

The pre-registered pilot uses continuous pre-cognitive eyes-open and
eyes-closed recordings from baseline and approximately five-year follow-up.
It applies the same sensor-space NeuroStat validation path used for EEGBCI:
Traditional preprocessing, overlapping epochs, pipeline ``compute_pli``, and
paired posterior-alpha statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pli_pipeline import PreprocessingConfig, compute_pli, preprocess_raw

LOGGER = logging.getLogger("ds005385_replication")
DATASET_ID = "ds005385"
DATASET_VERSION = "1.0.3"
DATASET_DOI = "10.18112/openneuro.ds005385.v1.0.3"
S3_BASE = "https://s3.amazonaws.com/openneuro.org"
DEFAULT_SUBJECTS = (
    "001",
    "028",
    "033",
    "038",
    "044",
    "046",
    "047",
    "050",
    "053",
    "054",
    "059",
    "073",
)
SESSIONS = ("1", "2")
STATES = ("EyesOpen", "EyesClosed")
ACQUISITION = "pre"
ALPHA_BAND = (8.0, 13.0)
EPOCH_SECONDS = 2.0
EPOCH_OVERLAP = 0.50
RANDOM_SEED = 42
POSTERIOR_CHANNELS = (
    "O1",
    "Oz",
    "O2",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "P3",
    "Pz",
    "P4",
    "P7",
    "P8",
)
DEFAULT_PROVENANCE_DIR = (
    Path(__file__).resolve().parent / "provenance" / "ds005385_v1.0.3"
)

PREPROC_CONFIG = PreprocessingConfig(
    l_freq=1.0,
    h_freq=40.0,
    notch_freqs=(50.0,),
    ica_method="fastica",
    ica_n_components=15,
    random_state=RANDOM_SEED,
    use_ged=False,
    use_asr=False,
    use_iclabel=True,
    iclabel_brain_threshold=0.50,
    bad_channel_threshold=4.0,
    interpolate_bad_channels=True,
    ica_reject_threshold=200e-6,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_data_root(data_root: Path) -> Path:
    resolved = data_root.expanduser().resolve()
    if _is_relative_to(resolved, PROJECT_ROOT.resolve()):
        raise ValueError(
            f"Data root must be outside the Git worktree ({PROJECT_ROOT}); got {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _object_keys(subjects: Sequence[str]) -> list[str]:
    keys = [
        f"{DATASET_ID}/dataset_description.json",
        f"{DATASET_ID}/CHANGES",
        f"{DATASET_ID}/README.md",
        f"{DATASET_ID}/participants.tsv",
        f"{DATASET_ID}/participants.json",
    ]
    for subject in subjects:
        keys.extend(
            [
                f"{DATASET_ID}/sub-{subject}/sub-{subject}_sessions.tsv",
                f"{DATASET_ID}/sub-{subject}/sub-{subject}_sessions.json",
            ]
        )
        for session in SESSIONS:
            for state in STATES:
                stem = f"sub-{subject}_ses-{session}_task-{state}_acq-{ACQUISITION}"
                prefix = f"{DATASET_ID}/sub-{subject}/ses-{session}/eeg/{stem}"
                keys.extend(
                    [
                        f"{prefix}_eeg.edf",
                        f"{prefix}_eeg.json",
                        f"{prefix}_channels.tsv",
                        f"{prefix}_events.tsv",
                        f"{prefix}_events.json",
                    ]
                )
    return keys


def _download_one(key: str, dataset_root: Path, retries: int = 3) -> dict[str, Any]:
    relative = Path(key).relative_to(DATASET_ID)
    destination = dataset_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{S3_BASE}/{key}"

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        part = destination.with_suffix(destination.suffix + ".part")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "NeuroStat/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                expected = int(response.headers.get("Content-Length", 0))
                etag = response.headers.get("ETag", "").strip('"')
                modified = response.headers.get("Last-Modified", "")
                if (
                    destination.exists()
                    and expected
                    and destination.stat().st_size == expected
                ):
                    return {
                        "object_key": key,
                        "relative_path": relative.as_posix(),
                        "bytes": destination.stat().st_size,
                        "etag": etag,
                        "last_modified": modified,
                        "sha256": _sha256(destination),
                        "downloaded": False,
                    }
                digest = hashlib.sha256()
                written = 0
                with part.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                if expected and written != expected:
                    raise OSError(f"short download for {key}: {written} != {expected}")
                os.replace(part, destination)
                return {
                    "object_key": key,
                    "relative_path": relative.as_posix(),
                    "bytes": written,
                    "etag": etag,
                    "last_modified": modified,
                    "sha256": digest.hexdigest(),
                    "downloaded": True,
                }
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
            if part.exists():
                part.unlink()
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"failed to download {key}: {last_error}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _verify_selection(
    dataset_root: Path, subjects: Sequence[str]
) -> list[dict[str, Any]]:
    participants = pd.read_csv(dataset_root / "participants.tsv", sep="\t")
    participants["participant_id"] = participants["participant_id"].astype(str)
    selected = participants[
        participants["participant_id"].isin([f"sub-{s}" for s in subjects])
    ].copy()
    if len(selected) != len(subjects):
        found = set(selected["participant_id"])
        missing = [s for s in subjects if f"sub-{s}" not in found]
        raise ValueError(f"subjects missing from participants.tsv: {missing}")
    if not selected["session2"].eq("yes").all():
        bad = selected.loc[~selected["session2"].eq("yes"), "participant_id"].tolist()
        raise ValueError(f"subjects without follow-up: {bad}")
    return selected.sort_values("participant_id").to_dict(orient="records")


def download_subset(
    data_root: Path,
    provenance_dir: Path,
    subjects: Sequence[str],
    workers: int,
) -> dict[str, Any]:
    dataset_root = _validate_data_root(data_root)
    keys = _object_keys(subjects)
    LOGGER.info("Downloading/verifying %d files into %s", len(keys), dataset_root)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_to_key = {
            pool.submit(_download_one, key, dataset_root): key for key in keys
        }
        for completed, future in enumerate(as_completed(future_to_key), start=1):
            records.append(future.result())
            if completed % 10 == 0 or completed == len(keys):
                LOGGER.info("Download manifest progress: %d/%d", completed, len(keys))

    description = json.loads(
        (dataset_root / "dataset_description.json").read_text(encoding="utf-8")
    )
    actual_doi = str(description.get("DatasetDOI", "")).removeprefix("doi:")
    if actual_doi != DATASET_DOI:
        raise ValueError(
            f"dataset DOI mismatch: expected {DATASET_DOI}, got {actual_doi}"
        )
    selected_metadata = _verify_selection(dataset_root, subjects)
    records.sort(key=lambda item: item["object_key"])
    manifest = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "dataset": {
            "accession": DATASET_ID,
            "version": DATASET_VERSION,
            "doi": DATASET_DOI,
            "license": description.get("License"),
            "title": description.get("Name"),
            "source": S3_BASE,
        },
        "subjects": list(subjects),
        "subject_metadata": selected_metadata,
        "file_count": len(records),
        "edf_count": sum(
            item["relative_path"].endswith("_eeg.edf") for item in records
        ),
        "total_bytes": sum(int(item["bytes"]) for item in records),
        "files": records,
    }
    _write_json(dataset_root / "download_manifest.json", manifest)
    _write_json(provenance_dir / "download_manifest.json", manifest)
    return manifest


def _recording_path(dataset_root: Path, subject: str, session: str, state: str) -> Path:
    stem = f"sub-{subject}_ses-{session}_task-{state}_acq-{ACQUISITION}_eeg.edf"
    return dataset_root / f"sub-{subject}" / f"ses-{session}" / "eeg" / stem


def _verify_download_manifest(
    dataset_root: Path,
    expected_paths: Sequence[Path],
    subjects: Sequence[str],
) -> bool:
    manifest_path = dataset_root / "download_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"download manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset", {}).get("doi") != DATASET_DOI:
        raise ValueError(
            "download manifest dataset DOI does not match the pinned version"
        )
    if tuple(manifest.get("subjects", [])) != tuple(subjects):
        raise ValueError("download manifest subject list does not match this analysis")
    by_path = {item["relative_path"]: item for item in manifest.get("files", [])}
    for path in expected_paths:
        relative = path.relative_to(dataset_root).as_posix()
        record = by_path.get(relative)
        if record is None:
            raise ValueError(f"recording absent from download manifest: {relative}")
        actual_hash = _sha256(path)
        if actual_hash != record.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {relative}")
    return True


def _overlapping_epochs(raw: mne.io.Raw) -> mne.EpochsArray:
    picks = mne.pick_types(raw.info, eeg=True, meg=False)
    names = [raw.ch_names[index] for index in picks]
    data = raw.get_data(picks=picks)
    samples_per_epoch = round(EPOCH_SECONDS * raw.info["sfreq"])
    step = round(samples_per_epoch * (1.0 - EPOCH_OVERLAP))
    starts = range(0, data.shape[1] - samples_per_epoch + 1, step)
    epochs = np.asarray(
        [data[:, start : start + samples_per_epoch] for start in starts]
    )
    info = mne.create_info(names, raw.info["sfreq"], ch_types="eeg")
    return mne.EpochsArray(epochs, info, verbose=False)


def _posterior_alpha_pli(raw: mne.io.Raw) -> tuple[float, int, list[str]]:
    epochs = _overlapping_epochs(raw)
    pli = compute_pli(epochs, band=ALPHA_BAND, method="pli", use_gpu=False)
    found = [name for name in POSTERIOR_CHANNELS if name in epochs.ch_names]
    if len(found) != len(POSTERIOR_CHANNELS):
        missing = sorted(set(POSTERIOR_CHANNELS) - set(found))
        raise ValueError(f"posterior channels missing: {missing}")
    indices = [epochs.ch_names.index(name) for name in found]
    matrix = pli[np.ix_(indices, indices)]
    value = float(matrix[np.triu_indices_from(matrix, k=1)].mean())
    return value, len(epochs), found


def _process_recording(path: Path) -> dict[str, Any]:
    started = time.time()
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    original_sfreq = float(raw.info["sfreq"])
    original_duration = float(raw.times[-1])
    raw.resample(250.0, npad="auto", verbose=False)
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1020"),
        on_missing="ignore",
        verbose=False,
    )
    # ICLabel expects an applied common-average reference. The pipeline later
    # adds its own reference projection, matching the EEGBCI validation path.
    raw.set_eeg_reference("average", projection=False, verbose=False)
    _, cleaned, _, info = preprocess_raw(raw, PREPROC_CONFIG)
    value, n_epochs, posterior = _posterior_alpha_pli(cleaned)
    return {
        "posterior_alpha_pli": value,
        "epochs": n_epochs,
        "posterior_channels": len(posterior),
        "eeg_channels": len(mne.pick_types(cleaned.info, eeg=True, meg=False)),
        "original_sfreq_hz": original_sfreq,
        "analysis_sfreq_hz": float(cleaned.info["sfreq"]),
        "duration_s": original_duration,
        "ica_excluded": int(info.get("ica_excluded", 0)),
        "iclabel_applied": bool(info.get("iclabel_applied", False)),
        "bad_channels": ";".join(info.get("bad_channels", [])),
        "elapsed_s": time.time() - started,
    }


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    difference = a - b
    standard_deviation = difference.std(ddof=1)
    return (
        float(difference.mean() / standard_deviation) if standard_deviation > 0 else 0.0
    )


def _paired_state_stats(frame: pd.DataFrame) -> dict[str, Any]:
    pivot = frame.pivot(index="subject", columns="state", values="posterior_alpha_pli")
    ec = pivot["EyesClosed"].to_numpy(float)
    eo = pivot["EyesOpen"].to_numpy(float)
    difference = ec - eo
    t_stat, p_value = stats.ttest_rel(ec, eo)
    try:
        w_stat, w_p = stats.wilcoxon(ec, eo)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    return {
        "n": len(pivot),
        "eyes_open_mean": float(eo.mean()),
        "eyes_open_sd": float(eo.std(ddof=1)),
        "eyes_closed_mean": float(ec.mean()),
        "eyes_closed_sd": float(ec.std(ddof=1)),
        "difference_mean": float(difference.mean()),
        "difference_sd": float(difference.std(ddof=1)),
        "paired_t": float(t_stat),
        "paired_t_p": float(p_value),
        "wilcoxon_w": float(w_stat),
        "wilcoxon_p": float(w_p),
        "cohens_d": _cohens_d_paired(ec, eo),
        "positive_subjects": int((difference > 0).sum()),
    }


def _icc_a1(values: np.ndarray) -> float:
    """Two-way absolute-agreement, single-measure ICC(A,1)."""
    values = np.asarray(values, dtype=float)
    n_subjects, n_raters = values.shape
    grand = values.mean()
    subject_means = values.mean(axis=1)
    rater_means = values.mean(axis=0)
    ss_subject = n_raters * np.square(subject_means - grand).sum()
    ss_rater = n_subjects * np.square(rater_means - grand).sum()
    residual = values - subject_means[:, None] - rater_means[None, :] + grand
    ss_error = np.square(residual).sum()
    ms_subject = ss_subject / (n_subjects - 1)
    ms_rater = ss_rater / (n_raters - 1)
    ms_error = ss_error / ((n_subjects - 1) * (n_raters - 1))
    denominator = (
        ms_subject
        + (n_raters - 1) * ms_error
        + n_raters * (ms_rater - ms_error) / n_subjects
    )
    return float((ms_subject - ms_error) / denominator) if denominator else np.nan


def _bootstrap_icc(values: np.ndarray, repeats: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    n_subjects = len(values)
    estimates = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sample = rng.integers(0, n_subjects, size=n_subjects)
        estimates[index] = _icc_a1(values[sample])
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return np.nan, np.nan
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def _longitudinal_stats(subject_frame: pd.DataFrame, metric: str) -> dict[str, Any]:
    pivot = subject_frame.pivot(index="subject", columns="session", values=metric)
    session1 = pivot["1"].to_numpy(float)
    session2 = pivot["2"].to_numpy(float)
    pearson = stats.pearsonr(session1, session2)
    spearman = stats.spearmanr(session1, session2)
    values = np.column_stack([session1, session2])
    icc = _icc_a1(values)
    icc_low, icc_high = _bootstrap_icc(values)
    change_t = stats.ttest_rel(session2, session1)
    return {
        "n": len(pivot),
        "session1_mean": float(session1.mean()),
        "session2_mean": float(session2.mean()),
        "mean_change": float((session2 - session1).mean()),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "icc_a1": icc,
        "icc_a1_bootstrap_95_ci": [icc_low, icc_high],
        "session_change_t": float(change_t.statistic),
        "session_change_p": float(change_t.pvalue),
    }


def _aggregate_results(
    frame: pd.DataFrame,
    subjects: Sequence[str],
    download_integrity_pass: bool,
) -> dict[str, Any]:
    session_stats = {
        session: _paired_state_stats(frame[frame["session"] == session])
        for session in SESSIONS
    }
    subject_state = (
        frame.pivot_table(
            index=["subject", "session"],
            columns="state",
            values="posterior_alpha_pli",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    subject_state["EC_minus_EO"] = (
        subject_state["EyesClosed"] - subject_state["EyesOpen"]
    )
    averaged = subject_state.groupby("subject", as_index=False)[
        ["EyesClosed", "EyesOpen"]
    ].mean()
    averaged_long = averaged.melt(
        id_vars="subject",
        value_vars=["EyesClosed", "EyesOpen"],
        var_name="state",
        value_name="posterior_alpha_pli",
    )
    primary = _paired_state_stats(averaged_long)

    longitudinal = {
        metric: _longitudinal_stats(subject_state, metric)
        for metric in ("EyesOpen", "EyesClosed", "EC_minus_EO")
    }
    integrity_pass = (
        download_integrity_pass
        and len(frame) == len(subjects) * len(SESSIONS) * len(STATES)
        and frame["iclabel_applied"].all()
        and frame["posterior_alpha_pli"].notna().all()
        and frame["eeg_channels"].eq(64).all()
    )
    primary_pass = (
        primary["difference_mean"] > 0
        and primary["paired_t_p"] < 0.05
        and primary["cohens_d"] >= 0.50
    )
    session_pass = all(
        item["difference_mean"] > 0 and item["positive_subjects"] >= 8
        for item in session_stats.values()
    )
    gate = {
        "integrity_pass": bool(integrity_pass),
        "primary_effect_pass": bool(primary_pass),
        "session_consistency_pass": bool(session_pass),
    }
    gate["overall"] = "PASS" if all(gate.values()) else "FAIL"
    return {
        "pre_registered_gate": gate,
        "primary_across_session_average": primary,
        "state_effect_by_session": session_stats,
        "exploratory_longitudinal": longitudinal,
    }


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "mne",
        "mne-connectivity",
        "mne-icalabel",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
    }


def run_analysis(
    data_root: Path,
    provenance_dir: Path,
    subjects: Sequence[str],
) -> dict[str, Any]:
    dataset_root = _validate_data_root(data_root)
    expected = [
        _recording_path(dataset_root, s, session, state)
        for s in subjects
        for session in SESSIONS
        for state in STATES
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} recordings missing; run without --run-only first"
        )
    LOGGER.info("Verifying SHA-256 for %d EDF recordings", len(expected))
    download_integrity_pass = _verify_download_manifest(
        dataset_root, expected, subjects
    )

    rows: list[dict[str, Any]] = []
    started = time.time()
    for subject in subjects:
        for session in SESSIONS:
            for state in STATES:
                path = _recording_path(dataset_root, subject, session, state)
                LOGGER.info("Processing sub-%s ses-%s %s", subject, session, state)
                result = _process_recording(path)
                result.update({"subject": subject, "session": session, "state": state})
                rows.append(result)
                LOGGER.info(
                    "  alpha PLI=%.4f, ICA excluded=%d, elapsed=%.1fs",
                    result["posterior_alpha_pli"],
                    result["ica_excluded"],
                    result["elapsed_s"],
                )

    frame = pd.DataFrame(rows).sort_values(["subject", "session", "state"])
    summary = _aggregate_results(frame, subjects, download_integrity_pass)
    summary.update(
        {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "dataset": {
                "accession": DATASET_ID,
                "version": DATASET_VERSION,
                "doi": DATASET_DOI,
            },
            "subjects": list(subjects),
            "recordings_processed": len(frame),
            "elapsed_minutes": (time.time() - started) / 60.0,
            "analysis_parameters": {
                "preprocessing": "Traditional (bad-channel interpolation + FastICA/ICLabel; ASR disabled)",
                "resample_hz": 250.0,
                "bandpass_hz": [1.0, 40.0],
                "notch_hz": [50.0],
                "ica_components": 15,
                "iclabel_brain_threshold": 0.50,
                "alpha_band_hz": list(ALPHA_BAND),
                "epoch_seconds": EPOCH_SECONDS,
                "epoch_overlap": EPOCH_OVERLAP,
                "posterior_channels": list(POSTERIOR_CHANNELS),
                "random_seed": RANDOM_SEED,
            },
            "environment": _environment(),
        }
    )
    provenance_dir.mkdir(parents=True, exist_ok=True)
    csv_path = provenance_dir / "subject_recording_results.csv"
    frame.to_csv(csv_path, index=False, float_format="%.8g")
    summary["artifacts"] = {
        "subject_recording_results_csv_sha256": _sha256(csv_path),
        "analysis_script_sha256": _sha256(Path(__file__)),
        "pipeline_sha256": _sha256(PROJECT_ROOT / "pli_pipeline.py"),
        "pre_registration_sha256": _sha256(
            Path(__file__).with_name("ds005385_replication_plan.json")
        ),
    }
    _write_json(provenance_dir / "replication_results.json", summary)
    LOGGER.info("Pre-registered result: %s", summary["pre_registered_gate"]["overall"])
    return summary


def _normalise_subjects(values: Iterable[str]) -> tuple[str, ...]:
    subjects = tuple(str(value).removeprefix("sub-").zfill(3) for value in values)
    if len(subjects) != len(set(subjects)):
        raise ValueError("duplicate subject identifiers")
    return subjects


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="External directory for ds005385 files (must be outside this Git worktree).",
    )
    parser.add_argument(
        "--provenance-dir",
        type=Path,
        default=DEFAULT_PROVENANCE_DIR,
        help="Directory for compact tracked manifests/results.",
    )
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    parser.add_argument("--workers", type=int, default=6)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true")
    mode.add_argument("--run-only", action="store_true")
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING")
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    mne.set_log_level("WARNING")
    subjects = _normalise_subjects(args.subjects)
    data_root = _validate_data_root(args.data_root)
    provenance_dir = args.provenance_dir.expanduser().resolve()
    if not args.run_only:
        manifest = download_subset(data_root, provenance_dir, subjects, args.workers)
        LOGGER.info(
            "Verified %d files (%.2f GiB)",
            manifest["file_count"],
            manifest["total_bytes"] / 1024**3,
        )
    if not args.download_only:
        summary = run_analysis(data_root, provenance_dir, subjects)
        primary = summary["primary_across_session_average"]
        print(
            json.dumps(
                {
                    "gate": summary["pre_registered_gate"]["overall"],
                    "n": primary["n"],
                    "mean_EC_minus_EO": primary["difference_mean"],
                    "paired_t_p": primary["paired_t_p"],
                    "cohens_d": primary["cohens_d"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
