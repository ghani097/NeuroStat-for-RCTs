"""Reader for WiBCI headset CSV exports.

The rules follow ``WIBCI_CSVtoSET_QC/MAIN_Pipeline/loadWiBCIData_UG.m`` and
``wibci2eeglab.m``:

* row 1 holds absolute values and every later row holds differences, so the
  table is decoded with a cumulative sum;
* before summing, later rows get +1 on the packet number, +4000 us on both
  timestamps, and EEG deltas are scaled by 0.02235 uV per count;
* blank cells are zero;
* packet-number gaps are filled with NaN rows and then pchip-interpolated;
* ``Event A``/``Event B`` values of 100 or more are software events and 1-99
  are hardware events, each taken at its rising edge.

The sampling rate is 250 Hz unless the caller overrides it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SFREQ = 250.0
EEG_DELTA_SCALE = 0.02235
TIMESTAMP_STEP_US = 4000.0
_EEG_COLUMN = re.compile(r"^\s*(.+?)\s*\(uV\)\s*$", re.IGNORECASE)
_REFERENCE_LABELS = {"ref", "reference"}


class WiBCIError(ValueError):
    """Raised for CSV files that are not WiBCI exports."""


@dataclass(frozen=True, slots=True)
class WiBCIData:
    eeg: np.ndarray  # (channels, samples) microvolts
    labels: tuple[str, ...]
    events: tuple[tuple[int, str], ...]  # (sample, description)
    missing_fraction: float
    n_packets: int


def read_wibci(path: str | Path, *, fill_missing: bool = True) -> WiBCIData:
    import pandas as pd

    source = Path(path)
    table = pd.read_csv(source, skipinitialspace=True)
    columns = [str(name) for name in table.columns]
    if not columns or columns[0].strip().lower() != "packet no":
        raise WiBCIError(f"{source.name}: first column is not 'Packet No'")
    eeg_columns = [
        index for index, name in enumerate(columns) if _EEG_COLUMN.match(name)
    ]
    if not eeg_columns:
        raise WiBCIError(f"{source.name}: no '<label> (uV)' columns")
    values = table.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    values[np.isnan(values)] = 0.0
    if len(values) > 1:
        values[1:, 0] += 1.0
        values[1:, 1:3] += TIMESTAMP_STEP_US
        values[1:, eeg_columns] *= EEG_DELTA_SCALE
    decoded = np.cumsum(values, axis=0)
    decoded, n_missing = _insert_gaps(decoded, _event_columns(columns))
    if n_missing and fill_missing:
        decoded = _pchip_fill(decoded)
    labels = [_EEG_COLUMN.match(columns[index]).group(1) for index in eeg_columns]
    keep = [i for i, label in enumerate(labels) if label.lower() not in _REFERENCE_LABELS]
    eeg = decoded[:, [eeg_columns[i] for i in keep]].T
    events = []
    for column in _event_columns(columns):
        events.extend(_rising_events(decoded[:, column], columns[column].strip()))
    events.sort()
    return WiBCIData(
        eeg,
        tuple(labels[i] for i in keep),
        tuple(events),
        n_missing / len(decoded) if len(decoded) else 0.0,
        len(decoded),
    )


def read_raw_wibci(path: str | Path, sfreq: float | None = None):
    """Return an ``mne.io.RawArray`` in volts with events as annotations."""
    import mne

    rate = float(sfreq or DEFAULT_SFREQ)
    wibci = read_wibci(path)
    if not np.isfinite(wibci.eeg).all():
        raise WiBCIError(f"{Path(path).name}: data still has gaps after filling")
    info = mne.create_info(list(wibci.labels), rate, ch_types="eeg")
    raw = mne.io.RawArray(wibci.eeg * 1e-6, info, verbose="ERROR")
    if wibci.events:
        onsets = [sample / rate for sample, _ in wibci.events]
        raw.set_annotations(
            mne.Annotations(onsets, [0.0] * len(onsets), [label for _, label in wibci.events])
        )
    raw.info["description"] = f"WiBCI CSV; missing packets {wibci.missing_fraction:.4%}"
    return raw, wibci.missing_fraction


def _event_columns(columns: list[str]) -> list[int]:
    return [i for i, name in enumerate(columns) if name.strip().lower().startswith("event ")]


def _insert_gaps(decoded: np.ndarray, event_columns: list[int]) -> tuple[np.ndarray, int]:
    packets = decoded[:, 0]
    gaps = np.diff(packets) - 1
    positions = np.flatnonzero(gaps > 0)
    if positions.size == 0:
        return decoded, 0
    n_missing = int(gaps[positions].sum())
    output = np.full((len(decoded) + n_missing, decoded.shape[1]), np.nan)
    source = target = 0
    for position in positions:
        count = position - source + 1
        output[target:target + count] = decoded[source:position + 1]
        target += count
        source = position + 1
        size = int(gaps[position])
        output[target:target + size, 0] = packets[position] + np.arange(1, size + 1)
        output[target:target + size, event_columns] = 0.0
        target += size
    output[target:] = decoded[source:]
    return output, n_missing


def _pchip_fill(decoded: np.ndarray) -> np.ndarray:
    from scipy.interpolate import PchipInterpolator

    filled = decoded.copy()
    index = np.arange(len(filled))
    for column in range(filled.shape[1]):
        values = filled[:, column]
        missing = np.isnan(values)
        if missing.any() and (~missing).sum() >= 2:
            values[missing] = PchipInterpolator(index[~missing], values[~missing])(index[missing])
    return filled


def _rising_events(values: np.ndarray, column: str) -> list[tuple[int, str]]:
    channel = column.split()[-1]
    events: list[tuple[int, str]] = []
    for kind, mask in (("soft", values >= 100), ("hard", (values >= 1) & (values < 100))):
        onsets = np.flatnonzero(np.diff(np.concatenate(([False], mask)).astype(int)) == 1)
        events.extend((int(i), f"{kind}{channel}/{int(round(values[i]))}") for i in onsets)
    return events


__all__ = ["DEFAULT_SFREQ", "WiBCIData", "WiBCIError", "read_raw_wibci", "read_wibci"]
