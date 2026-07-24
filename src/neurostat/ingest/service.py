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


def _payload_digest(data: np.ndarray, labels: tuple[str, ...], times: np.ndarray, events: np.ndarray | None, positions: np.ndarray | None, metadata: dict[str, Any]) -> str:
    """Hash every canonical value, not merely HDF5's structural metadata."""
    digest = hashlib.sha256()
    def add_array(value: np.ndarray | None) -> None:
        if value is None:
            digest.update(b"none\0"); return
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode()); digest.update(_json(list(array.shape)).encode()); digest.update(array.tobytes())
    digest.update(_json(list(labels)).encode("utf-8")); add_array(data); add_array(times); add_array(events); add_array(positions)
    digest.update(_json(metadata).encode("utf-8"))
    return digest.hexdigest()


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


def _h5_scalar_or_char(file: Any, value: Any) -> Any:
    """Read the scalar/reference/UTF-16 shapes used by MATLAB v7.3 headers."""
    import h5py
    if isinstance(value, h5py.Reference): return _h5_scalar_or_char(file, file[value])
    if isinstance(value, h5py.Dataset):
        raw = value[()]
        if raw.dtype == h5py.ref_dtype and raw.size == 1: return _h5_scalar_or_char(file, raw.flat[0])
        matlab_class = value.attrs.get("MATLAB_class", b"")
        if matlab_class == b"char" or matlab_class == "char": return "".join(chr(int(code)) for code in np.asarray(raw).ravel(order="F") if int(code))
        if isinstance(raw, bytes): return raw.decode("utf-8", "strict")
        if isinstance(raw, np.ndarray) and raw.size == 1: return raw.item()
        return raw
    if isinstance(value, h5py.Group):
        for name in ("value", "data"):
            if name in value: return _h5_scalar_or_char(file, value[name])
    return value


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
        names = tuple(self.channel_names)
        if data.ndim != 3 or not data.shape[0] or not data.shape[1] or not data.shape[2]: raise IngestError("canonical data must have nonempty (epochs, channels, samples) shape")
        if len(names) != data.shape[1] or any(not isinstance(x, str) or not x.strip() for x in names) or len(set(names)) != len(names): raise IngestError("channel labels must be unique nonblank strings")
        if not np.isfinite(self.sfreq) or self.sfreq <= 0 or times.shape != (data.shape[2],) or not np.isfinite(times).all() or (len(times) > 1 and not np.all(np.diff(times) > 0)): raise IngestError("sfreq/times are invalid")
        if not np.isfinite(data).all(): raise IngestError("recording data contains non-finite values")
        events = None if self.events is None else np.array(self.events, copy=True)
        positions = None if self.channel_positions is None else np.array(self.channel_positions, dtype=np.float64, copy=True)
        if events is not None and (not np.issubdtype(events.dtype, np.number) or events.ndim == 0 or events.shape[0] != data.shape[0] or not np.isfinite(events).all()): raise IngestError("events must be finite numeric rows aligned to epochs")
        if positions is not None and (positions.shape != (data.shape[1], 3) or not np.isfinite(positions).all()): raise IngestError("channel_positions has invalid shape")
        data.setflags(write=False); times.setflags(write=False)
        if events is not None: events.setflags(write=False)
        if positions is not None: positions.setflags(write=False)
        object.__setattr__(self, "channel_names", names); object.__setattr__(self, "data", data); object.__setattr__(self, "times", times); object.__setattr__(self, "events", events); object.__setattr__(self, "channel_positions", positions)


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
        # Prefer the actual EEGLAB EEG.data field; same-stem is only a fallback.
        parsed = False
        try:
            from scipy.io import loadmat
            eeg = loadmat(record.path, squeeze_me=True, struct_as_record=False).get("EEG")
            if eeg is not None and hasattr(eeg, "data"):
                parsed = True; data_ref = eeg.data
                if isinstance(data_ref, str) and data_ref.lower().endswith(".fdt"):
                    companion = (record.path.parent / data_ref).resolve()
                    if not companion.is_file(): raise IngestError(f"{record.record_id}: EEGLAB header references missing companion .fdt: {companion}")
                    files.append(companion)
        except (OSError, ValueError, NotImplementedError):
            parsed = False
        if not parsed:
            try:
                import h5py
                with h5py.File(record.path, "r") as file:
                    if "EEG" in file and "data" in file["EEG"]:
                        parsed = True
                        data_ref = _h5_scalar_or_char(file, file["EEG"]["data"])
                        if isinstance(data_ref, str) and data_ref.lower().endswith(".fdt"):
                            companion = (record.path.parent / data_ref).resolve()
                            if not companion.is_file(): raise IngestError(f"{record.record_id}: EEGLAB header references missing companion .fdt: {companion}")
                            files.append(companion)
            except OSError:
                pass
        if not parsed:
            text = record.path.read_bytes()[:1024 * 1024].decode("latin1", "ignore")
            refs = re.findall(r"['\"]([^'\"]+\.fdt)['\"]", text, flags=re.I)
            candidates = [record.path.parent / ref for ref in refs] or [record.path.with_suffix(".fdt")]
            for candidate in candidates:
                if candidate.is_file(): files.append(candidate.resolve())
    unique = tuple(sorted(set(path.resolve() for path in files), key=lambda p: str(p).casefold()))
    hashes = tuple((str(path), _sha256(path)) for path in unique)
    return SourceBundle(unique, hashes, hashlib.sha256(_json(hashes).encode()).hexdigest())


def _mne(record: ProjectRecord, bundle: SourceBundle) -> CleanRecording:
    import mne
    if record.data_unit not in (None, "", "V"): raise IngestError(f"{record.record_id}: MNE SET/FIF data are volts; data_unit must be V or blank")
    if record.format == "set": obj = mne.io.read_raw_eeglab(record.path, preload=True, verbose="ERROR") if record.data_kind == "raw" else mne.read_epochs_eeglab(record.path, verbose="ERROR")
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
        if isinstance(raw, bytes): return raw
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
    if raw.get("fsample") is None: raise IngestError(f"{record.record_id}: FieldTrip fsample is required")
    try: fsample = float(np.asarray(raw["fsample"]).squeeze())
    except (TypeError, ValueError): raise IngestError(f"{record.record_id}: FieldTrip fsample is invalid") from None
    if not np.isfinite(fsample) or fsample <= 0: raise IngestError(f"{record.record_id}: FieldTrip fsample must be positive and finite")
    trials0 = raw.get("trial")
    if trials0 is None: raise IngestError(f"{record.record_id}: FieldTrip trial is required")
    if isinstance(trials0, list):
        trials = trials0
    elif isinstance(trials0, np.ndarray) and trials0.dtype == object:
        numeric = np.asarray(trials0, dtype=float)
        trials = [numeric] if numeric.ndim == 2 and len(labels) in numeric.shape else (list(numeric) if numeric.ndim == 3 and len(labels) in numeric.shape[1:] else list(trials0.ravel()))
    else:
        trials = [trials0]
    arrays: list[np.ndarray] = []
    for trial in trials:
        a = np.asarray(trial, dtype=float)
        if a.ndim != 2: raise IngestError("FieldTrip trials must be 2-D")
        if a.shape[0] != len(labels): raise IngestError("FieldTrip trials must be logical channels x samples (axis 0 matches labels)")
        arrays.append(a)
    split_samples: int | None = None
    if record.data_kind == "raw":
        if len(arrays) != 1: raise IngestError("FieldTrip raw requires exactly one trial")
    elif len(arrays) == 1:
        n = record.samples_per_epoch
        if n is None and "pnts" in raw:
            pnts_value = np.asarray(raw["pnts"])
            if isinstance(raw["pnts"], (bool, np.bool_)) or pnts_value.ndim != 0:
                raise IngestError("FieldTrip pnts must be a scalar positive finite integer")
            try: candidate = float(pnts_value.item())
            except (TypeError, ValueError): raise IngestError("FieldTrip pnts must be a numeric scalar") from None
            if not np.isfinite(candidate) or candidate <= 0 or not candidate.is_integer(): raise IngestError("FieldTrip pnts must be a positive finite integer")
            n = int(candidate)
        if n:
            if arrays[0].shape[1] % n: raise IngestError("FieldTrip concatenated trial is not divisible by samples_per_epoch")
            split_samples = n
            arrays = [arrays[0][:, i:i+n] for i in range(0, arrays[0].shape[1], n)]
    if len({a.shape for a in arrays}) != 1: raise IngestError("FieldTrip trials must have identical shapes")
    times0 = raw.get("time")
    times = np.arange(arrays[0].shape[1], dtype=float) / fsample
    if times0 is not None:
        if isinstance(times0, list): time_trials = times0
        elif isinstance(times0, np.ndarray) and times0.dtype == object:
            numeric_times = np.asarray(times0, dtype=float)
            time_trials = [numeric_times] if numeric_times.ndim == 1 else (list(numeric_times) if numeric_times.ndim == 2 and numeric_times.shape[0] == len(arrays) else list(times0.ravel()))
        else: time_trials = [times0]
        candidates = [np.asarray(candidate, dtype=float).ravel() for candidate in time_trials]
        if split_samples is not None and len(candidates) == 1 and candidates[0].size == split_samples * len(arrays):
            # A concatenated FieldTrip trial can have one continuous time vector;
            # canonical epochs share the first retained epoch's sample grid.
            candidates = [candidates[0][:split_samples]] * len(arrays)
        if len(candidates) == 1 and len(arrays) > 1: candidates *= len(arrays)
        if len(candidates) != len(arrays) or any(candidate.shape != times.shape for candidate in candidates) or any(not np.array_equal(candidate, candidates[0]) for candidate in candidates[1:]): raise IngestError("FieldTrip trial time vectors must be identical and match samples")
        times = candidates[0]
    positions = None
    elec = raw.get("elec")
    if elec is not None:
        pos = getattr(elec, "chanpos", None) if not isinstance(elec, dict) else elec.get("chanpos")
        if pos is None: pos = getattr(elec, "pnt", None) if not isinstance(elec, dict) else elec.get("pnt")
        elec_labels = getattr(elec, "label", None) if not isinstance(elec, dict) else elec.get("label")
        if pos is not None:
            if elec_labels is None: raise IngestError("FieldTrip electrode positions require elec.label mapping")
            electrode_names = tuple(str(x.decode() if isinstance(x, bytes) else x).strip() for x in np.asarray(elec_labels, dtype=object).ravel())
            pos = np.asarray(pos, dtype=float)
            if pos.shape != (len(electrode_names), 3) or not np.isfinite(pos).all() or len(set(electrode_names)) != len(electrode_names) or set(electrode_names) != set(labels): raise IngestError("FieldTrip electrode labels/positions do not unambiguously match data labels")
            positions = np.asarray([pos[electrode_names.index(label)] for label in labels], dtype=float)
    events = raw.get("trialinfo")
    if events is not None:
        events = np.asarray(events)
        if not np.issubdtype(events.dtype, np.number): raise IngestError("FieldTrip trialinfo must be numeric")
        if events.ndim == 0: events = events.reshape(1, 1)
        elif events.ndim == 1: events = events.reshape(1, -1) if len(arrays) == 1 else (events.reshape(-1, 1) if events.size == len(arrays) else (_ for _ in ()).throw(IngestError("FieldTrip trialinfo does not align to epochs")))
        elif events.ndim != 2: raise IngestError("FieldTrip trialinfo must be scalar, vector, or matrix")
        if events.shape[0] != len(arrays) or not np.isfinite(events).all(): raise IngestError("FieldTrip trialinfo must be finite and row-aligned to epochs")
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
            attrs["payload_digest"] = _payload_digest(recording.data, recording.channel_names, recording.times, recording.events, recording.channel_positions, attrs)
            for key, value in attrs.items(): file.attrs[key] = value
        _verify(Path(name), recording.source_bundle.digest, recording.data.shape)
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
            required = {"schema", "record_id", "participant_id", "arm", "session", "condition", "format", "data_kind", "analysis_profile", "sfreq", "dimensions", "source_bundle_digest", "source_file_hashes", "source_unit", "canonical_unit", "scale", "cleaning", "payload_digest"}
            if not required.issubset(file.attrs) or not {"data", "channel_names", "times"}.issubset(file): raise IngestError(f"invalid canonical artifact: {path}")
            if file.attrs["schema"] != SCHEMA or tuple(file["data"].shape) != tuple(shape) or file.attrs["source_bundle_digest"] != source_digest or file.attrs["cleaning"] != CLEANING_MARKER or json.loads(file.attrs["dimensions"]) != list(shape): raise IngestError(f"invalid canonical artifact: {path}")
            data, times = file["data"][:], file["times"][:]
            labels = tuple(x.decode() if isinstance(x, bytes) else str(x) for x in file["channel_names"][:])
            events = file["events"][:] if "events" in file else None; positions = file["channel_positions"][:] if "channel_positions" in file else None
            metadata = {key: file.attrs[key] for key in required - {"payload_digest"}}
            expected = _payload_digest(data, labels, times, events, positions, metadata)
            if expected != file.attrs["payload_digest"]: raise IngestError(f"canonical payload digest mismatch: {path}")
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
            entries.append({"record_id": record.record_id, "path": str(target), "sha256": digest, "dimensions": dimensions, "source_bundle_digest": bundle.digest, "source_hashes": list(bundle.hashes), "reused": was_reused})
            paths.append(target); _atomic_json(status, {"state": "running", "total": len(project.records), "completed": i})
            if progress: progress(ProgressEvent(StageName.INGEST, f"ingested {record.record_id}", i, len(project.records)))
        _atomic_json(manifest_path, {"schema": "neurostat.ingest_manifest.v1", "run_id": run_id, "records": entries}); _atomic_json(status, {"state": "succeeded", "total": len(project.records), "completed": len(project.records), "reused": reused, "written": written})
    except BaseException as exc:
        _atomic_json(status, {"state": "failed", "total": len(project.records), "completed": len(entries), "error": str(exc)}); raise
    return IngestRun(run_id, run_dir, manifest_path, reused, written, tuple(paths))
