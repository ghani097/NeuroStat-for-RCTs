"""Tiny writers that build synthetic files in each input format for tests.

They write only what the readers need, so tests do not depend on edfio, pybv
or pyxdf writers.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_edf(path: Path, data_uv: np.ndarray, labels: list[str], sfreq: int, *, bdf: bool = False, status: np.ndarray | None = None) -> None:
    """Write EDF (int16) or BDF (int24) with one-second records."""
    n_channels, n_samples = data_uv.shape
    n_records = n_samples // sfreq
    signals = [(label, data_uv[i], "uV") for i, label in enumerate(labels)]
    if status is not None:
        signals.append(("Status", status.astype(float), "Boolean"))
    if bdf:
        dmin, dmax, width = -8388608, 8388607, 3
    else:
        dmin, dmax, width = -32768, 32767, 2
    phys = 3276.8 if not bdf else 262144.0

    def field(value: object, size: int) -> bytes:
        return str(value).encode("ascii").ljust(size)[:size]

    ns = len(signals)
    header = (b"\xffBIOSEMI" if bdf else field("0", 8)) + field("X X X X", 80) + field("Startdate X X X X", 80)
    header += field("01.01.24", 8) + field("10.00.00", 8) + field(256 * (ns + 1), 8)
    header += field("24BIT" if bdf else "", 44) + field(n_records, 8) + field(1, 8) + field(ns, 4)
    columns = {
        "label": [field(s[0], 16) for s in signals],
        "transducer": [field("", 80) for _ in signals],
        "dimension": [field(s[2], 8) for s in signals],
        "pmin": [field(dmin if s[2] == "Boolean" else -phys, 8) for s in signals],
        "pmax": [field(dmax if s[2] == "Boolean" else phys, 8) for s in signals],
        "dmin": [field(dmin, 8) for _ in signals],
        "dmax": [field(dmax, 8) for _ in signals],
        "prefilter": [field("", 80) for _ in signals],
        "samples": [field(sfreq, 8) for _ in signals],
        "reserved": [field("", 32) for _ in signals],
    }
    for values in columns.values():
        header += b"".join(values)
    digital = []
    for _label, values, unit in signals:
        if unit == "Boolean":
            digital.append(np.asarray(values, dtype=np.int64))
        else:
            gain = (dmax - dmin) / (2 * phys)
            digital.append(np.round((np.asarray(values) + phys) * gain + dmin).astype(np.int64))
    body = bytearray()
    for record in range(n_records):
        for samples in digital:
            chunk = samples[record * sfreq:(record + 1) * sfreq]
            if bdf:
                raw = (chunk & 0xFFFFFF).astype("<u4").tobytes()
                body += b"".join(raw[i:i + 3] for i in range(0, len(raw), 4))
            else:
                body += chunk.astype("<i2").tobytes()
    path.write_bytes(header + bytes(body))


def write_brainvision(path: Path, data_uv: np.ndarray, labels: list[str], sfreq: float, markers: list[tuple[int, str]]) -> None:
    stem = path.stem
    channels = "\n".join(f"Ch{i}={label},,1,\xb5V" for i, label in enumerate(labels, 1))
    path.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\n"
        f"Codepage=Latin1\nDataFile={stem}.eeg\nMarkerFile={stem}.vmrk\nDataFormat=BINARY\n"
        f"DataOrientation=MULTIPLEXED\nNumberOfChannels={len(labels)}\nSamplingInterval={1e6 / sfreq:g}\n"
        "[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\n" + channels + "\n",
        encoding="latin1",
    )
    marks = "\n".join(f"Mk{i}=Stimulus,{name},{sample + 1},1,0" for i, (sample, name) in enumerate(markers, 2))
    (path.parent / f"{stem}.vmrk").write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nCodepage=Latin1\n"
        f"DataFile={stem}.eeg\n[Marker Infos]\nMk1=New Segment,,1,1,0\n{marks}\n",
        encoding="latin1",
    )
    (path.parent / f"{stem}.eeg").write_bytes(data_uv.T.astype("<f4").tobytes())


def write_poly5(path: Path, data: np.ndarray, names: list[str], units: list[str], sfreq: int, per_block: int = 16) -> None:
    n_channels, n_samples = data.shape
    n_blocks = n_samples // per_block
    header = bytearray(217)
    header[:28] = b"POLY SAMPLE FILEversion 2.03"
    header[114:116] = struct.pack("<h", sfreq)
    header[119:121] = struct.pack("<h", 2 * n_channels)
    header[129:143] = struct.pack("<7h", 2024, 4, 10, 3, 14, 10, 56)
    header[143:147] = struct.pack("<i", n_blocks)
    header[147:153] = struct.pack("<hhh", per_block, per_block * n_channels * 4, 0)
    descriptors = bytearray()
    for name, unit in zip(names, units):
        for half in ("(Lo) ", "(Hi) "):
            item = bytearray(136)
            label = (half + name).encode("latin1")
            item[0] = len(label)
            item[1:1 + len(label)] = label
            encoded = unit.encode("latin1")
            item[45] = len(encoded)
            item[46:46 + len(encoded)] = encoded
            descriptors += item
    blocks = bytearray()
    for block in range(n_blocks):
        blocks += bytes(86)
        blocks += data[:, block * per_block:(block + 1) * per_block].T.astype("<f4").tobytes()
    path.write_bytes(bytes(header) + bytes(descriptors) + bytes(blocks))


def _varlen(value: int) -> bytes:
    if value < 256:
        return b"\x01" + struct.pack("<B", value)
    if value < 2**32:
        return b"\x04" + struct.pack("<I", value)
    return b"\x08" + struct.pack("<Q", value)


def _chunk(tag: int, content: bytes) -> bytes:
    return _varlen(len(content) + 2) + struct.pack("<H", tag) + content


def write_xdf(path: Path, data_uv: np.ndarray, labels: list[str], sfreq: float, markers: list[tuple[float, str]], unit: str = "microvolts") -> None:
    channels = "".join(f"<channel><label>{label}</label><unit>{unit}</unit></channel>" for label in labels)
    eeg_info = (
        "<?xml version=\"1.0\"?><info><name>Amp</name><type>EEG</type>"
        f"<channel_count>{len(labels)}</channel_count><nominal_srate>{sfreq}</nominal_srate>"
        f"<channel_format>float32</channel_format><created_at>0</created_at><desc><channels>{channels}</channels></desc></info>"
    )
    marker_info = (
        "<?xml version=\"1.0\"?><info><name>Task</name><type>Markers</type><channel_count>1</channel_count>"
        "<nominal_srate>0</nominal_srate><channel_format>string</channel_format><created_at>0</created_at></info>"
    )
    out = bytearray(b"XDF:")
    out += _chunk(1, b"<?xml version=\"1.0\"?><info><version>1.0</version></info>")
    out += _chunk(2, struct.pack("<I", 1) + eeg_info.encode())
    out += _chunk(2, struct.pack("<I", 2) + marker_info.encode())
    start = 100.0
    samples = bytearray(struct.pack("<I", 1)) + _varlen(data_uv.shape[1])
    for index in range(data_uv.shape[1]):
        samples += b"\x08" + struct.pack("<d", start + index / sfreq) + data_uv[:, index].astype("<f4").tobytes()
    out += _chunk(3, bytes(samples))
    marker_samples = bytearray(struct.pack("<I", 2)) + _varlen(len(markers))
    for onset, text in markers:
        encoded = text.encode()
        marker_samples += b"\x08" + struct.pack("<d", start + onset) + _varlen(len(encoded)) + encoded
    out += _chunk(3, bytes(marker_samples))
    for stream_id, count, last in ((1, data_uv.shape[1], start + (data_uv.shape[1] - 1) / sfreq), (2, len(markers), start)):
        footer = f"<?xml version=\"1.0\"?><info><first_timestamp>{start}</first_timestamp><last_timestamp>{last}</last_timestamp><sample_count>{count}</sample_count></info>"
        out += _chunk(6, struct.pack("<I", stream_id) + footer.encode())
    path.write_bytes(bytes(out))


def write_wibci_csv(path: Path, data_uv: np.ndarray, labels: list[str], *, drop_rows: tuple[int, ...] = (), events: dict[int, int] | None = None) -> None:
    """Encode absolute microvolts the way the WiBCI app does (deltas after row 1)."""
    n_samples = data_uv.shape[1]
    header = ["Packet No", "App TimeStamp", "Sensor TimeStamp", *(f"{label} (uV)" for label in labels), "ref (uV)", "AccX(mg)", "AccY(mg)", "AccZ(mg)", "Event A", "Event B"]
    events = events or {}
    keep = [i for i in range(n_samples) if i not in set(drop_rows)]
    rows = []
    previous = None
    for i in keep:
        eeg = data_uv[:, i]
        event = float(events.get(i, 0))
        absolute = np.concatenate(([1000 + i, 4000.0 * i, 4000.0 * i], eeg, [0.0, 0, 0, 0, event, 0]))
        if previous is None:
            row = absolute
        else:
            row = absolute - previous
            row[0] -= 1
            row[1:3] -= 4000.0
            row[3:3 + len(labels) + 1] /= 0.02235
        previous = absolute
        rows.append(row)
    lines = [",".join(header)]
    for index, row in enumerate(rows):
        cells = [repr(float(value)) for value in row]
        if index and row[0] == 0:
            cells[0] = ""
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
