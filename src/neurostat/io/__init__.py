"""Multi-format EEG input: format registry, readers, channel maps and staging.

Importing this package is cheap: readers load MNE, pandas or pyxdf only when a
file is read.
"""

from .formats import FORMATS, RAW_READER_FORMATS, data_stem, format_for_path

__all__ = ["FORMATS", "RAW_READER_FORMATS", "data_stem", "format_for_path", "load_recording", "read_raw_any"]


def __getattr__(name: str):
    if name in {"load_recording", "read_raw_any", "LoadedRecording", "FormatReadError"}:
        from . import reader

        return getattr(reader, name)
    raise AttributeError(name)
