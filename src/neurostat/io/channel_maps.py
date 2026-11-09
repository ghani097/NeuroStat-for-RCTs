"""Channel-label maps that turn amplifier labels into 10-20 names.

A map is a mapping ``{file_label: new_label}``. A new label of ``""`` drops the
channel. ``resolve_channel_map`` accepts a preset name, a TSV/CSV path with two
columns, ``"none"``, or ``"auto"`` (pick a preset from the labels themselves).
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

# TMSi REFA 64-channel cap order for ExG1..ExG64 (UK and ANN studies). M1/M2 are
# the recording reference and are dropped, as in the published pipeline.
TMSI64 = (
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "M1",
    "T7", "C3", "Cz", "C4", "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
    "P8", "POz", "O1", "Oz", "O2", "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FC3",
    "FCz", "FC4", "C5", "C1", "C2", "C6", "CP3", "CPz", "CP4", "P5", "P1", "P2", "P6", "PO5",
    "PO3", "PO4", "PO6", "FT7", "FT8", "TP7", "TP8", "PO7", "PO8",
)
TMSI64_DROP = frozenset({"M1", "M2"})

PRESETS = ("auto", "none", "tmsi64", "biosemi32", "biosemi64")
_BIOSEMI_LABEL = re.compile(r"^[A-H]([1-9]|[12][0-9]|3[0-2])$")
_PREFIX = re.compile(r"^(EEG|EXG)[\s_-]+", re.IGNORECASE)
_SUFFIX = re.compile(r"[\s_-]+(REF|LE|AR|AVG|M1|M2|A1|A2|CAR)$", re.IGNORECASE)


class ChannelMapError(ValueError):
    """Raised when a channel map cannot be resolved or does not fit the file."""


def tmsi64_map() -> dict[str, str]:
    return {
        f"ExG{index}": ("" if label in TMSI64_DROP else label)
        for index, label in enumerate(TMSI64, start=1)
    }


def biosemi_map(n_channels: int) -> dict[str, str]:
    import mne

    names = mne.channels.make_standard_montage(f"biosemi{n_channels}").ch_names
    letters = "ABCDEFGH"
    return {
        f"{letters[index // 32]}{index % 32 + 1}": name
        for index, name in enumerate(names)
    }


def load_map_file(path: str | Path) -> dict[str, str]:
    source = Path(path)
    if not source.is_file():
        raise ChannelMapError(f"channel map file not found: {source}")
    delimiter = "\t" if source.suffix.lower() in {".tsv", ".txt"} else ","
    mapping: dict[str, str] = {}
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter=delimiter), 1):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if row_number == 1 and row[0].strip().lower() in {"from", "old", "file_label"}:
                continue
            old = row[0].strip()
            new = row[1].strip() if len(row) > 1 else ""
            if old in mapping:
                raise ChannelMapError(f"{source}: label {old!r} appears twice")
            mapping[old] = new
    if not mapping:
        raise ChannelMapError(f"{source}: no channel rows")
    return mapping


def auto_map(labels: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Pick a preset from the file's labels; return ``(name, mapping)``."""
    present = set(labels)
    if all(f"ExG{index}" in present for index in range(1, 65)):
        return "tmsi64", tmsi64_map()
    biosemi = [label for label in labels if _BIOSEMI_LABEL.match(label)]
    if len(biosemi) > 64:
        # 128/256 caps keep their A1..H32 names; the BioSemi montage knows them.
        return "none", {}
    if len(biosemi) >= 32:
        n_channels = 64 if len(biosemi) >= 64 else 32
        return f"biosemi{n_channels}", biosemi_map(n_channels)
    cleaned = {label: _clean_label(label) for label in labels}
    changed = {old: new for old, new in cleaned.items() if new != old}
    if changed and len(set(cleaned.values())) == len(cleaned):
        return "strip-prefixes", changed
    return "none", {}


def resolve_channel_map(
    spec: str | Path | Mapping[str, str] | None,
    labels: Sequence[str],
) -> tuple[str, dict[str, str]]:
    """Return ``(description, mapping)`` for a preset, a file, or a mapping."""
    if spec is None or spec == "auto":
        return auto_map(labels)
    if isinstance(spec, Mapping):
        return "custom", dict(spec)
    text = str(spec)
    if text == "none":
        return "none", {}
    if text == "tmsi64":
        return "tmsi64", tmsi64_map()
    if text in {"biosemi32", "biosemi64"}:
        return text, biosemi_map(int(text.removeprefix("biosemi")))
    if text in PRESETS:
        raise ChannelMapError(f"unhandled channel map preset {text!r}")
    return f"file:{Path(text).name}", load_map_file(text)


def apply_channel_map(raw, mapping: Mapping[str, str]) -> list[str]:
    """Rename and drop channels in place; return the dropped labels."""
    if not mapping:
        return []
    present = set(raw.ch_names)
    drop = [old for old, new in mapping.items() if old in present and not new]
    rename = {old: new for old, new in mapping.items() if old in present and new}
    if not rename and not drop:
        raise ChannelMapError(
            "channel map matches none of the file's labels "
            f"(file starts with {raw.ch_names[:4]})"
        )
    if drop:
        raw.drop_channels(drop)
    final = [rename.get(name, name) for name in raw.ch_names]
    duplicates = sorted({name for name in final if final.count(name) > 1})
    if duplicates:
        raise ChannelMapError(f"channel map creates duplicate labels: {duplicates}")
    if rename:
        raw.rename_channels(rename)
    return drop


def _clean_label(label: str) -> str:
    return _SUFFIX.sub("", _PREFIX.sub("", label.strip())).strip()


__all__ = [
    "ChannelMapError",
    "PRESETS",
    "TMSI64",
    "apply_channel_map",
    "auto_map",
    "biosemi_map",
    "load_map_file",
    "resolve_channel_map",
    "tmsi64_map",
]
