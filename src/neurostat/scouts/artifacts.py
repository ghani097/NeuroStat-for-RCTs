"""Canonical variable-length scout artifact writer and validator."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..stage_io import canonical_json, sha256_file

SCHEMA = "neurostat.scouts.v1"


class ScoutError(RuntimeError):
    """Raised for unsafe source-stage inputs or artifacts."""


@dataclass(frozen=True, slots=True)
class ScoutStream:
    """Metadata plus ordered ``(start_sample, nodes x samples)`` chunks."""

    labels: tuple[str, ...]
    times: np.ndarray
    sfreq: float
    chunks: Iterable[tuple[int, np.ndarray]]
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScoutHeader:
    path: Path
    record_id: str
    analysis_profile: str
    source_backend: str
    labels: tuple[str, ...]
    sfreq: float
    samples: int
    payload_digest: str


def _decode_labels(values: np.ndarray) -> tuple[str, ...]:
    return tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values)


def _metadata(file: Any) -> dict[str, Any]:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in file.attrs.items()
        if key != "payload_digest"
    }


def _payload_digest(file: Any, metadata: dict[str, Any]) -> str:
    data = file["data"]
    labels = _decode_labels(file["labels"][:])
    times = np.asarray(file["times"][:], dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(canonical_json(list(labels)).encode("utf-8"))
    digest.update(canonical_json({"data_dtype": str(data.dtype), "data_shape": list(data.shape)}).encode("utf-8"))
    digest.update(canonical_json({"times_dtype": str(times.dtype), "times_shape": list(times.shape)}).encode("utf-8"))
    digest.update(np.ascontiguousarray(times).tobytes())
    step = int(data.chunks[-1]) if data.chunks else max(1, data.shape[-1])
    for start in range(0, data.shape[-1], step):
        block = np.asarray(data[:, start : start + step], dtype=np.float64)
        digest.update(canonical_json([start, block.shape[-1]]).encode("utf-8"))
        digest.update(np.ascontiguousarray(block).tobytes())
    digest.update(canonical_json(metadata).encode("utf-8"))
    return digest.hexdigest()


def verify_scout_artifact(path: Path, expected: dict[str, Any] | None = None) -> ScoutHeader:
    """Validate schema, complete payload integrity, and optional frozen attributes."""
    import h5py

    required = {
        "schema",
        "record_id",
        "participant_id",
        "arm",
        "session",
        "condition",
        "analysis_profile",
        "source_backend",
        "sfreq",
        "dimensions",
        "canonical_sha256",
        "canonical_payload_digest",
        "profile_digest",
        "source_provenance",
        "payload_digest",
    }
    try:
        with h5py.File(path, "r") as file:
            if not required.issubset(file.attrs) or not {"data", "labels", "times"}.issubset(file):
                raise ScoutError(f"invalid scout artifact: {path}")
            if file.attrs["schema"] != SCHEMA:
                raise ScoutError(f"invalid scout schema: {path}")
            data = file["data"]
            labels = _decode_labels(file["labels"][:])
            times = np.asarray(file["times"][:], dtype=np.float64)
            shape = tuple(data.shape)
            if (
                len(shape) != 2
                or not shape[0]
                or not shape[1]
                or len(labels) != shape[0]
                or len(set(labels)) != len(labels)
                or any(not label.strip() for label in labels)
                or times.shape != (shape[1],)
                or not np.isfinite(times).all()
                or (len(times) > 1 and not np.all(np.diff(times) > 0))
                or json.loads(file.attrs["dimensions"]) != list(shape)
                or not np.isfinite(float(file.attrs["sfreq"]))
                or float(file.attrs["sfreq"]) <= 0
                or data.chunks is None
                or data.chunks[0] != shape[0]
                or not (0 < data.chunks[1] <= shape[1])
            ):
                raise ScoutError(f"invalid scout dimensions, labels, times, or chunking: {path}")
            step = int(data.chunks[-1])
            for start in range(0, shape[1], step):
                if not np.isfinite(data[:, start : start + step]).all():
                    raise ScoutError(f"non-finite scout values: {path}")
            metadata = _metadata(file)
            if expected:
                for key, value in expected.items():
                    if key not in metadata or metadata[key] != value:
                        raise ScoutError(f"scout artifact {key} differs from the frozen run: {path}")
            actual = _payload_digest(file, metadata)
            if actual != file.attrs["payload_digest"]:
                raise ScoutError(f"scout payload digest mismatch: {path}")
            return ScoutHeader(
                path=path,
                record_id=str(file.attrs["record_id"]),
                analysis_profile=str(file.attrs["analysis_profile"]),
                source_backend=str(file.attrs["source_backend"]),
                labels=labels,
                sfreq=float(file.attrs["sfreq"]),
                samples=shape[1],
                payload_digest=str(file.attrs["payload_digest"]),
            )
    except OSError as exc:
        raise ScoutError(f"corrupt scout artifact: {path}") from exc


def write_scout_artifact(
    path: Path,
    stream: ScoutStream,
    attributes: dict[str, Any],
    *,
    chunk_samples: int,
) -> tuple[str, ScoutHeader]:
    """Atomically materialize a bounded scout stream."""
    import h5py

    labels = tuple(stream.labels)
    times = np.asarray(stream.times, dtype=np.float64)
    if (
        not labels
        or len(set(labels)) != len(labels)
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or (len(times) > 1 and not np.all(np.diff(times) > 0))
        or isinstance(stream.sfreq, bool)
        or not np.isfinite(stream.sfreq)
        or stream.sfreq <= 0
        or type(chunk_samples) is not int
        or chunk_samples <= 0
    ):
        raise ScoutError("invalid scout stream metadata")
    shape = (len(labels), len(times))
    metadata = dict(attributes)
    metadata.update(
        {
            "schema": SCHEMA,
            "sfreq": float(stream.sfreq),
            "dimensions": canonical_json(list(shape)),
            "source_provenance": canonical_json(stream.provenance),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as file:
            dtype = h5py.string_dtype("utf-8")
            file.create_dataset("labels", data=np.asarray(labels, dtype=dtype), dtype=dtype)
            file.create_dataset("times", data=times)
            dataset = file.create_dataset(
                "data",
                shape=shape,
                dtype=np.float64,
                chunks=(shape[0], min(shape[1], chunk_samples)),
                compression="gzip",
                shuffle=True,
            )
            position = 0
            for start, raw in stream.chunks:
                block = np.asarray(raw, dtype=np.float64)
                if (
                    type(start) is not int
                    or start != position
                    or block.ndim != 2
                    or block.shape[0] != shape[0]
                    or not block.shape[1]
                    or start + block.shape[1] > shape[1]
                    or not np.isfinite(block).all()
                ):
                    raise ScoutError("scout chunks must be finite, nonempty, contiguous, and match labels/times")
                dataset[:, start : start + block.shape[1]] = block
                position += block.shape[1]
            if position != shape[1]:
                raise ScoutError(f"scout stream ended at sample {position}; expected {shape[1]}")
            for key, value in metadata.items():
                file.attrs[key] = value
            file.attrs["payload_digest"] = _payload_digest(file, metadata)
        expected = {key: value for key, value in metadata.items() if key not in {"schema", "dimensions", "sfreq", "source_provenance"}}
        verify_scout_artifact(Path(temporary), expected)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    header = verify_scout_artifact(path, expected)
    return sha256_file(path), header


def write_matlab_export(path: Path, scout_path: Path) -> str:
    """Atomically export a canonical scout artifact using generic MAT keys."""
    import h5py
    from scipy.io import savemat

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.mat")
    if temporary.exists():
        raise ScoutError(f"stale partial MATLAB export exists: {temporary}")
    with h5py.File(scout_path, "r") as file:
        values = {
            "Value": np.asarray(file["data"][:], dtype=np.float64),
            "Time": np.asarray(file["times"][:], dtype=np.float64),
            "Labels": np.asarray(_decode_labels(file["labels"][:]), dtype=object),
            "SamplingRateHz": float(file.attrs["sfreq"]),
            "RecordID": str(file.attrs["record_id"]),
            "Schema": SCHEMA,
        }
    try:
        savemat(temporary, values, appendmat=False, do_compression=True, long_field_names=True, oned_as="row")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    verify_matlab_export(path, scout_path)
    return sha256_file(path)


def verify_matlab_export(path: Path, scout_path: Path) -> None:
    import h5py
    from scipy.io import loadmat

    try:
        values = loadmat(path, squeeze_me=True)
    except (OSError, ValueError) as exc:
        raise ScoutError(f"invalid MATLAB scout export: {path}") from exc
    with h5py.File(scout_path, "r") as file:
        required = {"Value", "Time", "Labels", "SamplingRateHz", "RecordID", "Schema"}
        if not required.issubset(values):
            raise ScoutError(f"MATLAB scout export is missing required values: {path}")
        if "Value" not in values or np.asarray(values["Value"]).shape != file["data"].shape:
            raise ScoutError(f"MATLAB scout export shape mismatch: {path}")
        if not np.array_equal(np.asarray(values["Value"], dtype=np.float64), file["data"][:]):
            raise ScoutError(f"MATLAB scout export values mismatch: {path}")
        if not np.array_equal(np.asarray(values.get("Time"), dtype=np.float64).ravel(), file["times"][:]):
            raise ScoutError(f"MATLAB scout export times mismatch: {path}")
        raw_labels = np.asarray(values["Labels"], dtype=object).ravel()
        labels = tuple(
            str(item.item() if isinstance(item, np.ndarray) and item.size == 1 else item).strip()
            for item in raw_labels
        )
        expected_labels = _decode_labels(file["labels"][:])
        if labels != expected_labels:
            raise ScoutError(f"MATLAB scout export labels mismatch: {path}")
        if float(np.asarray(values["SamplingRateHz"]).squeeze()) != float(file.attrs["sfreq"]):
            raise ScoutError(f"MATLAB scout export sampling rate mismatch: {path}")
        if str(np.asarray(values["RecordID"]).squeeze()) != str(file.attrs["record_id"]):
            raise ScoutError(f"MATLAB scout export record mismatch: {path}")
        if str(np.asarray(values["Schema"]).squeeze()) != SCHEMA:
            raise ScoutError(f"MATLAB scout export schema mismatch: {path}")
