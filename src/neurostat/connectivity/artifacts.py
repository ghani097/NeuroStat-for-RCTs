"""Canonical connectivity artifact writer and complete integrity validator."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis_profiles import Band
from ..stage_io import canonical_json, sha256_file
from .methods import ConnectivityError, ConnectivityValues, validate_pli_matrix

SCHEMA = "neurostat.connectivity.v1"


@dataclass(frozen=True, slots=True)
class ConnectivityHeader:
    path: Path
    record_id: str
    analysis_profile: str
    connectivity_backend: str
    labels: tuple[str, ...]
    bands: tuple[str, ...]
    payload_digest: str


def _strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values)


def _metadata(file: Any) -> dict[str, Any]:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in file.attrs.items()
        if key != "payload_digest"
    }


def _payload_digest(file: Any, metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in ("labels", "band_names"):
        digest.update(canonical_json(list(_strings(file[name][:]))).encode("utf-8"))
    for name in ("band_edges_hz", "matrix", "sos"):
        if name not in file:
            digest.update(f"{name}:none".encode("utf-8"))
            continue
        array = np.ascontiguousarray(file[name][:])
        digest.update(canonical_json([name, str(array.dtype), list(array.shape)]).encode("utf-8"))
        digest.update(array.tobytes())
    digest.update(canonical_json(metadata).encode("utf-8"))
    return digest.hexdigest()


def verify_connectivity_artifact(path: Path, expected: dict[str, Any] | None = None) -> ConnectivityHeader:
    import h5py

    required = {
        "schema",
        "record_id",
        "participant_id",
        "arm",
        "session",
        "condition",
        "analysis_profile",
        "connectivity_backend",
        "sfreq",
        "dimensions",
        "scout_sha256",
        "scout_payload_digest",
        "profile_digest",
        "method_provenance",
        "payload_digest",
    }
    try:
        with h5py.File(path, "r") as file:
            if not required.issubset(file.attrs) or not {"matrix", "labels", "band_names", "band_edges_hz"}.issubset(file):
                raise ConnectivityError(f"invalid connectivity artifact: {path}")
            matrix = np.asarray(file["matrix"][:], dtype=np.float64)
            labels = _strings(file["labels"][:])
            bands = _strings(file["band_names"][:])
            edges = np.asarray(file["band_edges_hz"][:], dtype=np.float64)
            if (
                file.attrs["schema"] != SCHEMA
                or matrix.shape != (len(bands), len(labels), len(labels))
                or edges.shape != (len(bands), 2)
                or len(set(labels)) != len(labels)
                or len(set(bands)) != len(bands)
                or any(not value.strip() for value in (*labels, *bands))
                or not np.isfinite(edges).all()
                or not np.all(edges[:, 0] > 0)
                or not np.all(edges[:, 1] > edges[:, 0])
                or json.loads(file.attrs["dimensions"]) != list(matrix.shape)
            ):
                raise ConnectivityError(f"invalid connectivity dimensions, labels, or bands: {path}")
            try:
                provenance = json.loads(file.attrs["method_provenance"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ConnectivityError(f"invalid connectivity method provenance: {path}") from exc
            if not isinstance(provenance, dict) or provenance.get("backend") != file.attrs["connectivity_backend"]:
                raise ConnectivityError(f"connectivity method provenance/backend mismatch: {path}")
            if file.attrs["connectivity_backend"] == "evoked_average_hilbert_pli":
                if "sos" not in file:
                    raise ConnectivityError(f"evoked connectivity artifact has no SOS coefficients: {path}")
                sos = np.asarray(file["sos"][:], dtype=np.float64)
                if sos.shape != (len(bands), 4, 6) or not np.isfinite(sos).all():
                    raise ConnectivityError(f"invalid evoked connectivity SOS coefficients: {path}")
            elif "sos" in file:
                raise ConnectivityError(f"continuous connectivity artifact unexpectedly contains SOS coefficients: {path}")
            for value in matrix:
                validate_pli_matrix(value)
            metadata = _metadata(file)
            if expected:
                for key, value in expected.items():
                    if key not in metadata or metadata[key] != value:
                        raise ConnectivityError(f"connectivity artifact {key} differs from the frozen run: {path}")
            actual = _payload_digest(file, metadata)
            if actual != file.attrs["payload_digest"]:
                raise ConnectivityError(f"connectivity payload digest mismatch: {path}")
            return ConnectivityHeader(
                path=path,
                record_id=str(file.attrs["record_id"]),
                analysis_profile=str(file.attrs["analysis_profile"]),
                connectivity_backend=str(file.attrs["connectivity_backend"]),
                labels=labels,
                bands=bands,
                payload_digest=str(file.attrs["payload_digest"]),
            )
    except OSError as exc:
        raise ConnectivityError(f"corrupt connectivity artifact: {path}") from exc


def write_connectivity_artifact(
    path: Path,
    values: ConnectivityValues,
    labels: tuple[str, ...],
    bands: tuple[Band, ...],
    attributes: dict[str, Any],
) -> tuple[str, ConnectivityHeader]:
    import h5py

    matrices = np.asarray(values.matrices, dtype=np.float64)
    expected_shape = (len(bands), len(labels), len(labels))
    if matrices.shape != expected_shape:
        raise ConnectivityError(f"connectivity output shape {matrices.shape} differs from {expected_shape}")
    for matrix in matrices:
        validate_pli_matrix(matrix)
    metadata = dict(attributes)
    metadata.update(
        {
            "schema": SCHEMA,
            "dimensions": canonical_json(list(expected_shape)),
            "method_provenance": canonical_json(values.provenance),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as file:
            dtype = h5py.string_dtype("utf-8")
            file.create_dataset("labels", data=np.asarray(labels, dtype=dtype), dtype=dtype)
            file.create_dataset("band_names", data=np.asarray([band.name for band in bands], dtype=dtype), dtype=dtype)
            file.create_dataset("band_edges_hz", data=np.asarray([[band.low_hz, band.high_hz] for band in bands], dtype=np.float64))
            file.create_dataset("matrix", data=matrices, compression="gzip", shuffle=True, chunks=(1, len(labels), len(labels)))
            if values.sos is not None:
                file.create_dataset("sos", data=np.asarray(values.sos, dtype=np.float64))
            for key, value in metadata.items():
                file.attrs[key] = value
            file.attrs["payload_digest"] = _payload_digest(file, metadata)
        verify_connectivity_artifact(Path(temporary), attributes)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    header = verify_connectivity_artifact(path, attributes)
    return sha256_file(path), header
