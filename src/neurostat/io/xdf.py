"""Reader for Lab Streaming Layer ``.xdf`` recordings (optional ``pyxdf``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_UNIT_SCALES = {
    "v": 1.0, "volt": 1.0, "volts": 1.0,
    "mv": 1e-3, "millivolt": 1e-3, "millivolts": 1e-3,
    "uv": 1e-6, "microvolt": 1e-6, "microvolts": 1e-6,
    "nv": 1e-9, "nanovolt": 1e-9, "nanovolts": 1e-9,
}
_MAX_RATE_DRIFT = 0.01  # 1 % between nominal and effective rate


class XDFError(ValueError):
    """Raised when an XDF file has no usable EEG stream."""


def _pyxdf():
    try:
        import pyxdf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise XDFError(
            "reading .xdf needs pyxdf; install it with "
            "`pip install neurostat-rcts[formats]` or `pip install pyxdf`"
        ) from exc
    return pyxdf


def list_streams(path: str | Path) -> list[dict[str, Any]]:
    """Return name/type/channel count/rate for each stream, for GUI pickers."""
    pyxdf = _pyxdf()
    return [
        {
            "stream_id": item["stream_id"],
            "name": item.get("name", ""),
            "type": item.get("type", ""),
            "channel_count": int(item.get("channel_count", 0)),
            "nominal_srate": float(item.get("nominal_srate", 0.0)),
        }
        for item in pyxdf.resolve_streams(str(path))
    ]


def read_raw_xdf(path: str | Path, stream: str | None = None, data_unit: str | None = None):
    """Return ``(raw, notes)`` for the selected EEG stream.

    ``stream`` matches a stream name (or a numeric stream id). Without it the
    single stream of type ``EEG`` is used. Every marker stream becomes
    annotations, timed against the first EEG sample.
    """
    import mne

    pyxdf = _pyxdf()
    streams, _header = pyxdf.load_xdf(str(path), dejitter_timestamps=True)
    eeg = _select(streams, stream, Path(path).name)
    info = eeg["info"]
    series = np.asarray(eeg["time_series"], dtype=np.float64)
    stamps = np.asarray(eeg["time_stamps"], dtype=np.float64)
    if series.ndim != 2 or series.shape[0] < 2:
        raise XDFError(f"{Path(path).name}: EEG stream has no samples")
    notes: list[str] = []
    nominal = float(info["nominal_srate"][0])
    effective = float(info.get("effective_srate", 0.0) or (len(stamps) - 1) / (stamps[-1] - stamps[0]))
    if nominal <= 0:
        nominal = effective
        notes.append(f"stream has no nominal rate; using effective {effective:.3f} Hz")
    elif effective and abs(effective - nominal) / nominal > _MAX_RATE_DRIFT:
        notes.append(f"effective rate {effective:.3f} Hz differs from nominal {nominal:g} Hz by over 1 %")
    labels, units = _channels(info, series.shape[1])
    scales = []
    for label, unit in zip(labels, units):
        chosen = data_unit or unit
        key = (chosen or "").replace("\xb5", "u").replace("μ", "u").lower()
        if key not in _UNIT_SCALES:
            notes.append(f"{label}: unit {chosen or 'missing'!r} is unknown; assuming microvolts")
            key = "uv"
        scales.append(_UNIT_SCALES[key])
    data = series.T * np.asarray(scales)[:, None]
    raw = mne.io.RawArray(data, mne.create_info(labels, nominal, ch_types="eeg"), verbose="ERROR")
    onsets: list[float] = []
    descriptions: list[str] = []
    for item in streams:
        if item is eeg or str(item["info"]["type"][0]).lower() not in {"markers", "marker", "events"}:
            continue
        for time, value in zip(item["time_stamps"], item["time_series"]):
            onset = float(time - stamps[0])
            if 0.0 <= onset <= raw.times[-1]:
                onsets.append(onset)
                descriptions.append(str(value[0] if np.ndim(value) else value))
    if onsets:
        raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), descriptions))
    return raw, notes


def _select(streams: list[dict[str, Any]], stream: str | None, name: str) -> dict[str, Any]:
    if stream is not None:
        for item in streams:
            info = item["info"]
            if str(info["name"][0]) == stream or str(info.get("stream_id", "")) == stream:
                return item
        raise XDFError(f"{name}: no stream named {stream!r}")
    eeg = [item for item in streams if str(item["info"]["type"][0]).upper() == "EEG"]
    if len(eeg) != 1:
        found = ", ".join(f"{s['info']['name'][0]} ({s['info']['type'][0]})" for s in streams)
        raise XDFError(f"{name}: expected one EEG stream, found {len(eeg)}; choose one of: {found}")
    return eeg[0]


def _channels(info: dict[str, Any], count: int) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    units: list[str] = []
    try:
        for channel in info["desc"][0]["channels"][0]["channel"]:
            labels.append(str(channel.get("label", [""])[0]).strip())
            units.append(str(channel.get("unit", [""])[0]).strip())
    except (KeyError, IndexError, TypeError):
        labels, units = [], []
    if len(labels) != count or any(not label for label in labels) or len(set(labels)) != count:
        labels = [f"Ch{index}" for index in range(1, count + 1)]
        units = units if len(units) == count else [""] * count
    return labels, units


__all__ = ["XDFError", "list_streams", "read_raw_xdf"]
