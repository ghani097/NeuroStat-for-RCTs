"""Pure-numpy reader for TMSi Poly5 v2.03 recordings.

Ported from ``poly5_to_set.py`` (ANN data-discrepancy scripts), which produced
the ANN ``.set`` files. The layout is a 217-byte file header, one 136-byte
descriptor per signal, then blocks of an 86-byte block header followed by
interleaved float32 samples. 32-bit channels are described by (Lo)/(Hi)
descriptor pairs, so only every second descriptor carries a channel.
"""

from __future__ import annotations

import datetime as dt
import re
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"POLY SAMPLE FILEversion 2.03"
_HEADER = 217
_DESCRIPTOR = 136
_BLOCK_HEADER = 86
_TRIGGER_NAMES = {"digi", "triggers", "trigger", "status", "digital"}


class Poly5Error(ValueError):
    """Raised for files this reader does not understand."""


@dataclass(frozen=True, slots=True)
class Poly5Data:
    data: np.ndarray  # (channels, samples) float32 in file units
    names: tuple[str, ...]
    units: tuple[str, ...]
    sfreq: float
    start: dt.datetime | None
    blocks_expected: int
    blocks_read: int


def read_poly5(path: str | Path) -> Poly5Data:
    source = Path(path)
    with source.open("rb") as handle:
        header = handle.read(_HEADER)
        if not header.startswith(b"POLY SAMPLE FILE"):
            raise Poly5Error(f"{source.name}: not a Poly5 file")
        if not header.startswith(MAGIC):
            version = header[16:31].decode("latin1", "replace").strip()
            raise Poly5Error(f"{source.name}: Poly5 {version!r} is not supported (need version 2.03)")
        sfreq = struct.unpack("<h", header[114:116])[0]
        n_signals = struct.unpack("<h", header[119:121])[0]
        year, month, day, _weekday, hour, minute, second = struct.unpack("<7h", header[129:143])
        n_blocks = struct.unpack("<i", header[143:147])[0]
        per_block, bytes_per_block, delta = struct.unpack("<hhh", header[147:153])
        if delta:
            raise Poly5Error(f"{source.name}: delta-compressed Poly5 is not supported")
        if n_signals % 2:
            raise Poly5Error(f"{source.name}: odd signal count {n_signals}; only 32-bit float Poly5 is supported")
        names: list[str] = []
        units: list[str] = []
        for index in range(n_signals):
            descriptor = handle.read(_DESCRIPTOR)
            if index % 2 == 0:
                names.append(descriptor[1:1 + descriptor[0]].decode("latin1").replace("(Lo) ", "").strip())
                units.append(descriptor[46:46 + descriptor[45]].decode("latin1").strip())
        n_channels = n_signals // 2
        if bytes_per_block != per_block * n_channels * 4:
            raise Poly5Error(f"{source.name}: block size {bytes_per_block} does not match {n_channels} float32 channels")
        payload = np.frombuffer(handle.read(), dtype=np.uint8)
    block = _BLOCK_HEADER + bytes_per_block
    complete = len(payload) // block
    if complete == 0:
        raise Poly5Error(f"{source.name}: no complete sample blocks")
    if complete < n_blocks:
        warnings.warn(f"{source.name}: {complete}/{n_blocks} complete blocks (file is truncated)", stacklevel=2)
    blocks = payload[: complete * block].reshape(complete, block)[:, _BLOCK_HEADER:]
    data = blocks.copy().view("<f4").reshape(complete * per_block, n_channels).T
    try:
        start = dt.datetime(year, month, day, hour, minute, second, tzinfo=dt.timezone.utc)
    except ValueError:
        start = None
    return Poly5Data(data, tuple(names), tuple(units), float(sfreq), start, n_blocks, complete)


def read_raw_poly5(path: str | Path):
    """Return an ``mne.io.RawArray`` in volts with typed channels.

    ``ExG``/``UNI`` channels are EEG, trigger channels are stim and everything
    else (BIP, AUX, saw-tooth counters) is misc. Channel naming to 10-20 labels
    happens later through the channel map.
    """
    import mne

    poly = read_poly5(path)
    types = [_channel_type(name) for name in poly.names]
    data = poly.data.astype(np.float64)
    for row, (kind, unit) in enumerate(zip(types, poly.units)):
        if kind == "eeg":
            data[row] *= _unit_scale(unit)
    info = mne.create_info(list(_unique(poly.names)), poly.sfreq, ch_types=types)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    if poly.start is not None:
        raw.set_meas_date(poly.start)
    return raw


def _unit_scale(unit: str) -> float:
    normalised = unit.replace("\xb5", "u").replace("μ", "u").lower()
    return {"uv": 1e-6, "mv": 1e-3, "v": 1.0, "nv": 1e-9}.get(normalised, 1e-6)


def _channel_type(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith(("exg", "uni")) or _looks_10_20(name):
        return "eeg"
    if lowered in _TRIGGER_NAMES or lowered.startswith("digi"):
        return "stim"
    return "misc"


def _looks_10_20(name: str) -> bool:
    return bool(re.fullmatch(r"(Fp|AF|F|FC|FT|C|T|TP|CP|P|PO|O|I|N)(z|\d{1,2})h?", name, re.IGNORECASE))


def _unique(names: tuple[str, ...]):
    seen: dict[str, int] = {}
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        yield name if count == 0 else f"{name}-{count}"


__all__ = ["MAGIC", "Poly5Data", "Poly5Error", "read_poly5", "read_raw_poly5"]
