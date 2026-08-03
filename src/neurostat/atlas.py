"""Stable cortical label contracts used by source-space stages."""
from __future__ import annotations

_DK34 = (
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "frontalpole",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "insula",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "paracentral",
    "parahippocampal",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "temporalpole",
    "transversetemporal",
)

# Accepted v1/GraphVar order: left/right are adjacent for each aparc parcel.
DK68_LABELS = tuple(label for name in _DK34 for label in (f"{name}-lh", f"{name}-rh"))

__all__ = ["DK68_LABELS"]
