"""One front door for reading any supported continuous EEG file."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .channel_maps import apply_channel_map, resolve_channel_map
from .formats import FORMATS, RAW_READER_FORMATS, format_for_path

READER_VERSION = "neurostat.io.v1"
_MONTAGE = "standard_1005"


class FormatReadError(ValueError):
    """Raised when a file cannot be read as the requested format."""


@dataclass(frozen=True, slots=True)
class LoadedRecording:
    """A continuous recording in volts plus what the reader did to it."""

    raw: Any  # mne.io.BaseRaw
    format: str
    channel_map: str
    dropped_channels: tuple[str, ...] = ()
    unmatched_labels: tuple[str, ...] = ()
    missing_fraction: float = 0.0
    notes: tuple[str, ...] = field(default=())

    def summary(self) -> dict[str, Any]:
        raw = self.raw
        return {
            "format": self.format,
            "channels": len(raw.ch_names),
            "sfreq": float(raw.info["sfreq"]),
            "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
            "channel_map": self.channel_map,
            "dropped_channels": list(self.dropped_channels),
            "unmatched_labels": list(self.unmatched_labels),
            "missing_fraction": self.missing_fraction,
            "events": len(raw.annotations),
            "notes": list(self.notes),
        }


def load_recording(
    path: str | Path,
    fmt: str | None = None,
    *,
    channel_map: str | Path | Mapping[str, str] | None = "auto",
    data_unit: str | None = None,
    stream: str | None = None,
    sampling_rate: float | None = None,
    keep_non_eeg: bool = False,
    preload: bool = True,
) -> LoadedRecording:
    """Read ``path`` into an MNE Raw in volts.

    Steps: read with the format's reader, apply the channel map, turn trigger
    channels into annotations, drop non-EEG channels (unless
    ``keep_non_eeg``), then apply the standard 10-05 montage where labels match.
    """
    source = Path(path)
    if not source.is_file():
        raise FormatReadError(f"file not found: {source}")
    fmt = fmt or format_for_path(source)
    if fmt is None:
        raise FormatReadError(f"{source.name}: unrecognised EEG format")
    if fmt not in RAW_READER_FORMATS:
        known = ", ".join(sorted(RAW_READER_FORMATS))
        raise FormatReadError(f"{source.name}: format {fmt!r} cannot be read as continuous data (use one of {known})")
    if data_unit not in (None, "", "V") and fmt not in {"wibci_csv", "xdf"}:
        raise FormatReadError(f"{source.name}: data_unit applies only to WiBCI CSV and XDF; {FORMATS[fmt].label} readers already return volts")

    notes: list[str] = []
    missing = 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw, extra_notes, missing = _read(source, fmt, data_unit, stream, sampling_rate, preload)
    notes.extend(extra_notes)
    notes.extend(str(item.message) for item in caught if "annotation" not in str(item.message).lower())

    description, mapping = resolve_channel_map(channel_map, raw.ch_names)
    dropped = apply_channel_map(raw, mapping)
    _stim_to_annotations(raw, fmt)
    if not keep_non_eeg:
        others = [name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind != "eeg"]
        if others:
            raw.drop_channels(others)
            dropped = [*dropped, *others]
    if not raw.ch_names:
        raise FormatReadError(f"{source.name}: no EEG channels left after the channel map")
    unmatched = _apply_montage(raw)
    if unmatched:
        notes.append(f"{len(unmatched)} channel(s) have no standard position: {', '.join(unmatched[:8])}")
    return LoadedRecording(raw, fmt, description, tuple(dropped), tuple(unmatched), float(missing), tuple(notes))


def read_raw_any(path: str | Path, fmt: str | None = None, **options: Any):
    """Return only the MNE Raw from :func:`load_recording`."""
    return load_recording(path, fmt, **options).raw


def companion_files(path: str | Path, fmt: str) -> tuple[Path, ...]:
    """Files beyond ``path`` that hold part of the recording (for provenance)."""
    source = Path(path)
    if fmt != "brainvision":
        return ()
    text = source.read_text(encoding="latin1", errors="replace")
    found: list[Path] = []
    for key in ("DataFile", "MarkerFile"):
        match = re.search(rf"^\s*{key}\s*=\s*(.+?)\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            candidate = (source.parent / match.group(1)).resolve()
            if not candidate.is_file():
                raise FormatReadError(f"{source.name}: {key} {match.group(1)!r} is missing")
            found.append(candidate)
    return tuple(found)


def _read(source: Path, fmt: str, data_unit: str | None, stream: str | None, sampling_rate: float | None, preload: bool):
    import mne

    if sampling_rate is not None and fmt != "wibci_csv":
        raise FormatReadError(f"{source.name}: sampling_rate applies only to WiBCI CSV; other formats store their rate")
    try:
        if fmt == "bdf":
            return mne.io.read_raw_bdf(source, preload=preload, verbose="ERROR"), [], 0.0
        if fmt == "edf":
            return mne.io.read_raw_edf(source, preload=preload, verbose="ERROR"), [], 0.0
        if fmt == "brainvision":
            return mne.io.read_raw_brainvision(source, preload=preload, verbose="ERROR"), [], 0.0
        if fmt == "set":
            return mne.io.read_raw_eeglab(source, preload=preload, verbose="ERROR"), [], 0.0
        if fmt == "fif":
            return mne.io.read_raw_fif(source, preload=preload, verbose="ERROR"), [], 0.0
        if fmt == "poly5":
            from .poly5 import read_raw_poly5

            return read_raw_poly5(source), [], 0.0
        if fmt == "xdf":
            from .xdf import read_raw_xdf

            raw, notes = read_raw_xdf(source, stream, data_unit)
            return raw, notes, 0.0
        if fmt == "wibci_csv":
            from .wibci_csv import read_raw_wibci

            raw, missing = read_raw_wibci(source, sampling_rate)
            if data_unit not in (None, "", "uV"):
                raise FormatReadError(f"{source.name}: WiBCI CSV columns are labelled uV; data_unit must be uV or blank")
            notes = [f"{missing:.2%} of packets were missing and filled by pchip interpolation"] if missing else []
            return raw, notes, missing
    except FormatReadError:
        raise
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise FormatReadError(f"{source.name}: could not read as {FORMATS[fmt].label}: {exc}") from exc
    raise FormatReadError(f"{source.name}: no reader for {fmt!r}")


def _stim_to_annotations(raw, fmt: str) -> None:
    import mne

    stim = [name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "stim"]
    if not stim:
        return
    raw.load_data()
    onsets: list[float] = []
    descriptions: list[str] = []
    for name in stim:
        values = raw.get_data(picks=[name])[0]
        if fmt == "bdf":
            # BioSemi keeps triggers in the low 16 bits; the high byte is status.
            values = (np.round(values).astype(np.int64) & 0xFFFF).astype(float)
        # An event is any change to a non-idle value; idle is the commonest value
        # (0 for most amplifiers, all-bits-high for TMSi digital inputs).
        idle = _mode(values)
        changes = np.flatnonzero(np.diff(values) != 0) + 1
        starts = changes[values[changes] != idle]
        if values[0] != idle:
            starts = np.concatenate(([0], starts))
        for sample in starts:
            onsets.append(float(raw.times[sample]))
            descriptions.append(f"{name}/{int(values[sample])}")
    if onsets:
        order = np.argsort(onsets, kind="stable")
        added = mne.Annotations(
            [onsets[i] for i in order],
            [0.0] * len(order),
            [descriptions[i] for i in order],
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(raw.annotations + added)


def _mode(values: np.ndarray) -> float:
    unique, counts = np.unique(values, return_counts=True)
    return float(unique[np.argmax(counts)])


def _apply_montage(raw) -> list[str]:
    import mne

    if _has_positions(raw):
        return []
    best = None
    for name in (_MONTAGE, "biosemi128", "biosemi256"):
        montage = mne.channels.make_standard_montage(name)
        known = {label.lower() for label in montage.ch_names}
        matched = sum(label.lower() in known for label in raw.ch_names)
        if best is None or matched > best[1]:
            best = (montage, matched, known)
    montage, matched, known = best
    unmatched = [label for label in raw.ch_names if label.lower() not in known]
    if matched:
        raw.set_montage(montage, match_case=False, on_missing="ignore", verbose="ERROR")
    return unmatched


def _has_positions(raw) -> bool:
    locations = np.array([channel["loc"][:3] for channel in raw.info["chs"]])
    return bool(np.isfinite(locations).all() and np.any(np.abs(locations) > 0, axis=1).all())


__all__ = [
    "FormatReadError",
    "LoadedRecording",
    "READER_VERSION",
    "companion_files",
    "load_recording",
    "read_raw_any",
]
