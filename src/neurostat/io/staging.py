"""Stage any supported raw recording as EEGLAB ``.set`` for the v1 preprocessor.

v1 (``pli_pipeline``) is frozen and reads only ``<base>/<arm>/<session>/*.set``.
Staging converts every other format (and any non-flat layout) into that shape
under ``staging_root``, writing a provenance sidecar beside each file so a
rerun with the same source and options is skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .formats import NEEDS_STAGING, data_stem
from .reader import READER_VERSION, load_recording

PROVENANCE_SCHEMA = "neurostat.staged_raw.v1"


class StagingError(RuntimeError):
    """Raised when a recording cannot be staged."""


class StageableRecord(Protocol):
    participant_id: str
    arm: str
    session: str
    format: str
    path: Path


@dataclass(frozen=True, slots=True)
class StagedFile:
    source: Path
    staged: Path
    reused: bool
    summary: Mapping[str, Any]


def needs_staging(records: Iterable[StageableRecord], base_folder: Path) -> bool:
    """True unless every record is already a ``.set`` at ``<arm>/<session>/``."""
    base = Path(base_folder).resolve()
    for record in records:
        if record.format in NEEDS_STAGING:
            return True
        if Path(record.path).resolve().parent != base / record.arm / record.session:
            return True
    return False


def stage_as_set(
    records: Iterable[StageableRecord],
    staging_root: str | Path,
    *,
    reader_options: Mapping[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[StagedFile, ...]:
    """Convert ``records`` into ``staging_root/<arm>/<session>/<name>.set``."""
    root = Path(staging_root)
    options = dict(reader_options or {})
    staged: list[StagedFile] = []
    taken: dict[Path, Path] = {}
    for record in records:
        source = Path(record.path).resolve()
        target = root / record.arm / record.session / f"{_staged_stem(record)}.set"
        if target in taken:
            raise StagingError(f"{source} and {taken[target]} would both stage to {target}")
        taken[target] = source
        staged.append(_stage_one(record, source, target, _options_for(record.format, options), progress))
    return tuple(staged)


def preview_records(
    records: Iterable[StageableRecord],
    *,
    reader_options: Mapping[str, Any] | None = None,
    per_arm: int = 1,
) -> list[tuple[StageableRecord, Mapping[str, Any] | None, str | None]]:
    """Read the first ``per_arm`` records of each arm; return summaries or errors.

    Used by the GUI "Check files" button so a wrong channel map, rate or unit
    shows up before a long run starts.
    """
    options = dict(reader_options or {})
    seen: dict[str, int] = {}
    results: list[tuple[StageableRecord, Mapping[str, Any] | None, str | None]] = []
    for record in records:
        if seen.get(record.arm, 0) >= per_arm:
            continue
        seen[record.arm] = seen.get(record.arm, 0) + 1
        try:
            loaded = load_recording(record.path, record.format, preload=False, **_options_for(record.format, options))
        except Exception as exc:  # reported to the user, never raised from a preview
            results.append((record, None, str(exc)))
        else:
            results.append((record, loaded.summary(), None))
    return results


def _options_for(fmt: str, options: Mapping[str, Any]) -> dict[str, Any]:
    chosen = {"channel_map": options.get("channel_map") or "auto"}
    if fmt == "xdf" and options.get("stream"):
        chosen["stream"] = options["stream"]
    if fmt == "wibci_csv" and options.get("sampling_rate"):
        chosen["sampling_rate"] = float(options["sampling_rate"])
    if fmt in {"xdf", "wibci_csv"} and options.get("data_unit"):
        chosen["data_unit"] = options["data_unit"]
    return chosen


def _stage_one(record: StageableRecord, source: Path, target: Path, options: dict[str, Any], progress) -> StagedFile:
    sidecar = target.with_suffix(".provenance.json")
    digest = _sha256(source)
    expected = {"schema": PROVENANCE_SCHEMA, "reader_version": READER_VERSION, "source": str(source), "source_sha256": digest, "format": record.format, "options": _jsonable(options)}
    if target.is_file() and sidecar.is_file():
        try:
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if {key: previous.get(key) for key in expected} == expected:
            if progress:
                progress(f"Staged copy is current: {target.name}")
            return StagedFile(source, target, True, previous.get("summary", {}))
    if progress:
        progress(f"Converting {source.name} ({record.format}) to {target.name}")
    try:
        loaded = load_recording(source, record.format, **options)
    except ValueError as exc:
        raise StagingError(str(exc)) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.partial.set")
    try:
        loaded.raw.export(temporary, fmt="eeglab", overwrite=True, verbose="ERROR")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    summary = loaded.summary()
    _atomic_json(sidecar, {**expected, "summary": summary})
    for note in summary["notes"]:
        if progress:
            progress(f"{source.name}: {note}")
    return StagedFile(source, target, False, summary)


def _staged_stem(record: StageableRecord) -> str:
    stem = data_stem(record.path)
    # Keep v1's file-name-based subject and condition inference working: the
    # stem stays the original name, prefixed by the subject folder when the
    # layout had one (otherwise identically named files would collide).
    arm_slug = _slug(record.arm)
    if record.participant_id == _slug(f"{record.arm}-{stem}"):
        return stem
    subject = record.participant_id[len(arm_slug) + 1:] if record.participant_id.startswith(f"{arm_slug}-") else record.participant_id
    return f"{subject}_{stem}"


def _slug(value: str) -> str:
    from neurostat.gui.discovery import slugify

    return slugify(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(name, path)


__all__ = ["PROVENANCE_SCHEMA", "StagedFile", "StagingError", "needs_staging", "preview_records", "stage_as_set"]
