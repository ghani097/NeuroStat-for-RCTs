"""Registry of EEG input formats, importable without MNE or any reader.

Project validation and GUI discovery use this module to recognise files. The
readers themselves live in sibling modules and are imported only when a file is
actually read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FormatSpec:
    """One supported input format."""

    key: str
    label: str
    extensions: tuple[str, ...]
    data_kinds: tuple[str, ...]
    # Formats whose header must be sniffed because the extension is generic.
    sniffed: bool = False


FORMATS: dict[str, FormatSpec] = {
    spec.key: spec
    for spec in (
        FormatSpec("set", "EEGLAB .set", (".set",), ("raw", "epochs")),
        FormatSpec("fif", "MNE .fif", (".fif.gz", ".fif"), ("raw", "epochs")),
        FormatSpec("fieldtrip_mat", "FieldTrip .mat", (".mat",), ("raw", "epochs")),
        FormatSpec("bdf", "BioSemi .bdf", (".bdf",), ("raw",)),
        FormatSpec("edf", "EDF/EDF+ .edf", (".edf",), ("raw",)),
        FormatSpec("brainvision", "BrainVision .vhdr", (".vhdr",), ("raw",)),
        FormatSpec("poly5", "TMSi .Poly5", (".poly5",), ("raw",)),
        FormatSpec("xdf", "LSL .xdf", (".xdf", ".xdfz"), ("raw",)),
        FormatSpec("wibci_csv", "WiBCI .csv", (".csv",), ("raw",), sniffed=True),
    )
}

#: Formats the original v2 contract accepted; kept for code that needs them.
CLEANED_LEGACY_FORMATS = frozenset({"set", "fif", "fieldtrip_mat"})
#: Formats read through MNE into continuous ``mne.io.Raw`` objects.
RAW_READER_FORMATS = frozenset(set(FORMATS) - {"fieldtrip_mat"})
#: Formats the v1 preprocessor cannot read directly and that must be staged.
NEEDS_STAGING = frozenset(RAW_READER_FORMATS - {"set"})

_WIBCI_HEADER = re.compile(r"^\s*Packet No\s*,.*\(uV\)", re.IGNORECASE)


def extensions_for(fmt: str) -> tuple[str, ...]:
    return FORMATS[fmt].extensions


def data_stem(path: str | Path) -> str:
    """Return the file name without its (possibly double) data extension."""
    name = Path(path).name
    lowered = name.lower()
    for spec in FORMATS.values():
        for extension in spec.extensions:
            if lowered.endswith(extension):
                return name[: -len(extension)]
    return Path(name).stem


def format_for_path(path: str | Path, *, sniff: bool = True) -> str | None:
    """Map a file to a format key, or ``None`` when it is not EEG input.

    ``.csv`` is too generic to trust by extension, so it is accepted only when
    the header looks like a WiBCI export. Pass ``sniff=False`` to skip opening
    the file (the extension match is then returned as is).
    """
    name = Path(path).name.lower()
    for spec in FORMATS.values():
        if any(name.endswith(extension) for extension in spec.extensions):
            if spec.sniffed and sniff and not _looks_like(spec.key, Path(path)):
                return None
            return spec.key
    return None


def _looks_like(fmt: str, path: Path) -> bool:
    if fmt != "wibci_csv":
        return True
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            return bool(_WIBCI_HEADER.match(handle.readline()))
    except OSError:
        return False


__all__ = [
    "CLEANED_LEGACY_FORMATS",
    "FORMATS",
    "FormatSpec",
    "NEEDS_STAGING",
    "RAW_READER_FORMATS",
    "data_stem",
    "extensions_for",
    "format_for_path",
]
