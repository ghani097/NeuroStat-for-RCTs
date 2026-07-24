"""Adapters and resumable artifact writer for already-clean EEG recordings."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..contracts import ProgressEvent, RecordKey, StageName
from ..project import CleanProject, ProjectRecord, load_project

SCHEMA = "neurostat.clean_recording.v1"
CLEANING_MARKER = "externally_cleaned_bypass"


class IngestError(RuntimeError): pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle: json.dump(value, handle, sort_keys=True, indent=2); handle.write("\n")
        os.replace(name, path)
    except BaseException:
        try: os.unlink(name)
        except FileNotFoundError: pass
        raise


@dataclass(frozen=True, slots=True)
class SourceBundle:
    files: tuple[Path, ...]
    hashes: tuple[tuple[str, str], ...]
    digest: str


@dataclass(frozen=True, slots=True)
class CleanRecording:
    record_id: str
    key: RecordKey
    data_kind: str
    format: str
    analysis_profile: str
    sfreq: float
    channel_names: tuple[str, ...]
    data: np.ndarray
    times: np.ndarray
    events: np.ndarray | None
    channel_positions: np.ndarray | None
    source_unit: str
    canonical_unit: str
    scale: float
    source_bundle: SourceBundle

    def __post_init__(self) -> None:
        data, times = np.array(self.data, dtype=np.float64, copy=True), np.array(self.times, dtype=np.float64, copy=True)
        if data.ndim != 3 or not data.shape[0] or not data.shape[1] or not data.shape[2]: raise IngestError("canonical data must have nonempty (epochs, channels, samples) shape")
        if len(self.channel_names) != data.shape[1] or any(not isinstance(x, str) or not x.strip() for x in self.channel_names) or len(set(self.channel_names)) != len(self.channel_names): raise IngestError("channel labels must be unique nonblank strings")
        if not np.isfinite(self.sfreq) or self.sfreq <= 0 or times.shape != (data.shape[2],) or not np.isfinite(times).all() or (len(times) > 1 and not np.all(np.diff(times) > 0)): raise IngestError("sfreq/times are invalid")
        if not np.isfinite(data).all(): raise IngestError("recording data contains non-finite values")
        events = None if self.events is None else np.array(self.events, copy=True)
        positions = None if self.channel_positions is None else np.array(self.channel_positions, dtype=np.float64, copy=True)
        if events is not None and (events.ndim == 0 or events.shape[0] != data.shape[0]): raise IngestError("events must have one row per epoch")
        if positions is not None and (positions.shape != (data.shape[1], 3) or not np.isfinite(positions).all()): raise IngestError("channel_positions has invalid shape")
        data.setflags(write=False); times.setflags(write=False)
        if events is not None: events.setflags(write=False)
        if positions is not None: positions.setflags(write=False)
        object.__setattr__(self, "data", data); object.__setattr__(self, "times", times); object.__setattr__(self, "events", events); object.__setattr__(self, "channel_positions", positions)


@dataclass(frozen=True, slots=True)
class IngestRun:
    run_id: str
    run_dir: Path
    manifest: Path
    reused: int
    written: int
    records: tuple[Path, ...]


def source_bundle(record: ProjectRecord) -> SourceBundle:
    files = [record.path]
    if record.format == "set":
        # EEGLAB headers commonly hold a relative .fdt name; only include a real companion.
        text = record.path.read_bytes()[:1024 * 1024].decode("latin1", "ignore")
        candidates = [record.path.with_suffix(".fdt")]
        try:
            from scipy.io import loadmat
            eeg = loadmat(record.path, squeeze_me=True, struct_as_record=False).get("EEG")
            data_ref = getattr(eeg, "data", None)
            if isinstance(data_ref, str) and data_ref.lower().endswith(".fdt"): candidates.insert(0, record.path.parent / data_ref)
        except (OSError, ValueError, NotImplementedError):
            pass
        candidates += [record.path.parent / match for match in re.findall(r"['\"]([^'\"]+\.fdt)['\"]", text, flags=re.I)]
        for candidate in candidates:
            if candidate.is_file(): files.append(candidate.resolve())
    unique = tuple(sorted(set(path.resolve() for path in files), key=lambda p: str(p).casefold()))
    hashes = tuple((str(path), _sha256(path)) for path in unique)
    return SourceBundle(unique, hashes, hashlib.sha256(_json(hashes).encode()).hexdigest())


def _mne(record: ProjectRecord, bundle: SourceBundle) -> CleanRecording:
    import mne
    if record.data_unit not in (None, "", "V"): raise IngestError(f"{record.record_id}: MNE SET/FIF data are volts; data_unit must be V or blank")
    if record.format == "set": obj = mne.io.read_raw_eeglab(record.path, preload=True, verbose="ERROR") if record.data_kind == "raw" else mne.read_epochs_eeglab(record.path, preload=True, verbose="ERROR")
    else: obj = mne.io.read_raw_fif(record.path, preload=True, verbose="ERROR") if record.data_kind == "raw" else mne.read_epochs(record.path, preload=True, verbose="ERROR")
    data = obj.get_data()
    if record.data_kind == "raw": data = data[None, :, :]
    times = np.asarray(obj.times, dtype=float)
    positions = None
    try:
        positions = np.asarray([ch["loc"][:3] for ch in obj.info["chs"]], dtype=float)
        if not np.isfinite(positions).all(): positions = None
    except (KeyError, TypeError): pass
    events = np.asarray(obj.events) if record.data_kind == "epochs" and getattr(obj, "events", None) is not None else None
    return CleanRecording(record.record_id, record.key, record.data_kind, record.format, record.analysis_profile, float(obj.info["sfreq"]), tuple(obj.ch_names), data, times, events, positions, "V", "V", 1.0, bundle)


def _h5_value(file: Any, value: Any) -> Any:
    """Dereference MATLAB v7.3 scalar/cell arrays, undoing MATLAB axes."""
    import h5py
    if isinstance(value, h5py.Reference): return _h5_value(file, file[value])
    if isinstance(value, h5py.Group): return {name: _h5_value(file, value[name]) for name in value.keys()}
    if isinstance(value, h5py.Dataset):
        raw = value[()]
        if raw.dtype == h5py.ref_dtype:
            return [_h5_value(file, ref) for ref in raw.ravel(order="F")]
        matlab_class = value.attrs.get("MATLAB_class", b"")
        if matlab_class == b"char" or matlab_class == "char":
            return "".join(chr(int(code)) for code in np.asarray(raw).ravel(order="F") if int(code))
        return raw.T if raw.ndim >= 2 else raw
    return value


def _fieldtrip(record: ProjectRecord, bundle: SourceBundle) -> CleanRecording:
    path, variable = record.path, record.mat_variable or "data_ft"
    raw: dict[str, Any]
    try:
        from scipy.io import loadmat
        classic = loadmat(path, squeeze_me=True, struct_as_record=False)
        if variable not in classic: raise IngestError(f"{record.record_id}: MATLAB variable {variable!r} not found")
        value = classic[variable]
        raw = {name: getattr(value, name) for name in dir(value) if not name.startswith("_")}
    except (NotImplementedError, ValueError, OSError):
        import h5py
        with h5py.File(path, "r") as file:
            if variable not in file: raise IngestError(f"{record.record_id}: MATLAB variable {variable!r} not found")
            group = file[variable]
            raw = {name: _h5_value(file, group[name]) for name in group.keys()}
            for name in ("pnts",):
                if name in file: raw[name] = _h5_value(file, file[name])
    labels = raw.get("label")
    if labels is None: raise IngestError(f"{record.record_id}: FieldTrip label is required")
    labels = tuple(str(x.decode() if isinstance(x, bytes) else x).strip() for x in np.asarray(labels, dtype=object).ravel())
    fsample = float(np.asarray(raw.get("fsample")).squeeze())
    trials0 = raw.get("trial")
    if trials0 is None: raise IngestError(f"{record.record_id}: FieldTrip trial is required")
    if isinstance(trials0, list):
        trials = trials0
    elif isinstance(trials0, np.ndarray) and trials0.dtype == object:
        trials = list(trials0.ravel())
    else:
        trials = [trials0]
    arrays: list[np.ndarray] = []
    for trial in trials:
        a = np.asarray(trial, dtype=float)
        if a.ndim != 2: raise IngestError("FieldTrip trials must be 2-D")
        if a.shape[0] == len(labels): pass
        elif a.shape[1] == len(labels): a = a.T
        else: raise IngestError("FieldTrip trial dimensions do not match labels")
        arrays.append(a)
    if record.data_kind == "raw":
        if len(arrays) != 1: raise IngestError("FieldTrip raw requires exactly one trial")
    elif len(arrays) == 1:
        n = record.samples_per_epoch or (int(np.asarray(raw["pnts"]).squeeze()) if "pnts" in raw else None)
        if n:
            if arrays[0].shape[1] % n: raise IngestError("FieldTrip concatenated trial is not divisible by samples_per_epoch")
            arrays = [arrays[0][:, i:i+n] for i in range(0, arrays[0].shape[1], n)]
    if len({a.shape for a in arrays}) != 1: raise IngestError("FieldTrip trials must have identical shapes")
    times0 = raw.get("time")
    times = np.arange(arrays[0].shape[1], dtype=float) / fsample
    if times0 is not None:
        candidate = times0[0] if isinstance(times0, list) else np.asarray(times0)
        candidate = np.asarray(candidate, dtype=float).ravel()
        if candidate.shape == times.shape: times = candidate
    positions = None
    elec = raw.get("elec")
    if elec is not None:
        pos = getattr(elec, "chanpos", None) if not isinstance(elec, dict) else elec.get("chanpos")
        if pos is None and isinstance(elec, dict): pos = elec.get("pnt")
        if pos is not None:
            pos = np.asarray(pos, dtype=float)
            if pos.shape == (len(labels), 3): positions = pos
    events = raw.get("trialinfo")
    events = np.asarray(events) if events is not None else None
    if events is not None and len(arrays) == 1 and events.ndim == 1: events = events[None, :]
    unit = record.data_unit or "native"; scales = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "native": 1.0}; scale = scales[unit]
    return CleanRecording(record.record_id, record.key, record.data_kind, record.format, record.analysis_profile, fsample, labels, np.stack(arrays) * scale, times, events, positions, unit, "V" if unit != "native" else "native", scale, bundle)


def read_record(record: ProjectRecord, bundle: SourceBundle) -> CleanRecording:
    return _mne(record, bundle) if record.format in {"set", "fif"} else _fieldtrip(record, bundle)


def _write_record(path: Path, recording: CleanRecording) -> str:
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True); fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent); os.close(fd)
    try:
        with h5py.File(name, "w") as file:
            nbytes = recording.data.nbytes; chunks = (1, recording.data.shape[1], min(recording.data.shape[2], max(1, 1024 * 1024 // max(8 * recording.data.shape[1], 1))))
            file.create_dataset("data", data=recording.data, compression="gzip", shuffle=True, chunks=chunks)
            dtype = h5py.string_dtype("utf-8"); file.create_dataset("channel_names", data=np.asarray(recording.channel_names, dtype=dtype), dtype=dtype); file.create_dataset("times", data=recording.times)
            if recording.events is not None: file.create_dataset("events", data=recording.events)
            if recording.channel_positions is not None: file.create_dataset("channel_positions", data=recording.channel_positions)
            attrs = {"schema": SCHEMA, "record_id": recording.record_id, "participant_id": recording.key.participant_id, "arm": recording.key.arm, "session": recording.key.session, "condition": recording.key.condition, "format": recording.format, "data_kind": recording.data_kind, "analysis_profile": recording.analysis_profile, "sfreq": recording.sfreq, "dimensions": _json(list(recording.data.shape)), "source_bundle_digest": recording.source_bundle.digest, "source_file_hashes": _json(recording.source_bundle.hashes), "source_unit": recording.source_unit, "canonical_unit": recording.canonical_unit, "scale": recording.scale, "cleaning": CLEANING_MARKER}
            for key, value in attrs.items(): file.attrs[key] = value
        os.replace(name, path)
    except BaseException:
        try: os.unlink(name)
        except FileNotFoundError: pass
        raise
    _verify(path, recording.source_bundle.digest, recording.data.shape)
    return _sha256(path)


def _verify(path: Path, source_digest: str, shape: tuple[int, ...]) -> None:
    import h5py
    try:
        with h5py.File(path, "r") as file:
            if file.attrs.get("schema") != SCHEMA or tuple(file["data"].shape) != tuple(shape) or file.attrs.get("source_bundle_digest") != source_digest: raise IngestError(f"invalid canonical artifact: {path}")
    except OSError as exc: raise IngestError(f"corrupt canonical artifact: {path}") from exc


def ingest_project(project: CleanProject | str | Path, progress: Callable[[ProgressEvent], None] | None = None) -> IngestRun:
    project = load_project(project) if not isinstance(project, CleanProject) else project
    bundles = {r.record_id: source_bundle(r) for r in project.records}
    frozen = {"project": project.normalized(), "source_bundles": {rid: {"digest": b.digest, "hashes": b.hashes} for rid, b in bundles.items()}}
    run_id = hashlib.sha256(_json(frozen).encode()).hexdigest()[:12]; run_dir = project.output_root / "runs" / f"ingest-{run_id}"; lock = run_dir / "project.lock.json"; manifest_path = run_dir / "ingest" / "manifest.json"; status = run_dir / "ingest" / "status.json"
    if lock.exists() and _json(json.loads(lock.read_text(encoding="utf-8"))) != _json(frozen): raise IngestError(f"existing lock differs for deterministic run {run_id}; remove only after inspection: {lock}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not lock.exists(): _atomic_json(lock, frozen)
    _atomic_json(status, {"state": "running", "total": len(project.records), "completed": 0})
    entries=[]; paths=[]; reused=written=0
    try:
        for i, record in enumerate(project.records, 1):
            target = run_dir / "ingest" / "records" / f"{record.record_id}.h5"; bundle = bundles[record.record_id]
            if target.exists():
                try:
                    prior = next((item for item in json.loads(manifest_path.read_text(encoding="utf-8"))["records"] if item["record_id"] == record.record_id), None) if manifest_path.exists() else None
                    import h5py
                    with h5py.File(target, "r") as file: shape = tuple(file["data"].shape)
                    _verify(target, bundle.digest, shape)
                    if prior is not None and prior["sha256"] != _sha256(target): raise IngestError("canonical SHA-256 differs from frozen manifest")
                except Exception as exc: raise IngestError(f"refusing to overwrite mismatched or corrupt artifact {target}: {exc}") from exc
                digest = _sha256(target); reused += 1
                was_reused = True
            else:
                recording = read_record(record, bundle); digest = _write_record(target, recording); written += 1
                was_reused = False
            # reopen dimensions avoids trusting a reader value for reused artifacts.
            import h5py
            with h5py.File(target, "r") as file: dimensions = list(file["data"].shape)
            entries.append({"record_id": record.record_id, "path": str(target), "sha256": digest, "dimensions": _json(dimensions), "source_bundle_digest": bundle.digest, "source_hashes": list(bundle.hashes), "reused": was_reused})
            paths.append(target); _atomic_json(status, {"state": "running", "total": len(project.records), "completed": i})
            if progress: progress(ProgressEvent(StageName.INGEST, f"ingested {record.record_id}", i, len(project.records)))
        _atomic_json(manifest_path, {"schema": "neurostat.ingest_manifest.v1", "run_id": run_id, "records": entries}); _atomic_json(status, {"state": "succeeded", "total": len(project.records), "completed": len(project.records), "reused": reused, "written": written})
    except BaseException as exc:
        _atomic_json(status, {"state": "failed", "total": len(project.records), "completed": len(entries), "error": str(exc)}); raise
    return IngestRun(run_id, run_dir, manifest_path, reused, written, tuple(paths))
