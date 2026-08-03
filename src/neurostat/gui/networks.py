"""Network presets exposed by the GUI.

The salience-network preset intentionally preserves v1's indices exactly:
it includes bankssts and insula, and omits anterior cingulate labels. That is
an unusual salience-network definition, but preserving the legacy contract keeps
phase-7 GUI output comparable with v1 until the user chooses otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping

from neurostat.atlas import DK68_LABELS

V1_NETWORK_INDICES: Mapping[str, tuple[int, ...]] = {
    "SN": (1, 2, 19, 20),
    "DMN": (15, 16, 21, 22, 29, 30, 31, 32, 35, 36, 47, 48, 51, 52, 53, 54),
    "CEN": (5, 6, 55, 56, 57, 58, 59, 60),
}


def labels_from_v1_indices(indices: tuple[int, ...]) -> tuple[str, ...]:
    """Return DK68 labels for v1's documented 1-based atlas indices."""
    return tuple(DK68_LABELS[index - 1] for index in indices)


PRESET_NETWORKS: Mapping[str, tuple[str, ...]] = {
    name: labels_from_v1_indices(indices)
    for name, indices in V1_NETWORK_INDICES.items()
}


def default_networks() -> dict[str, tuple[str, ...]]:
    """Return the v1-compatible default network selection."""
    return {name: tuple(labels) for name, labels in PRESET_NETWORKS.items()}


__all__ = [
    "PRESET_NETWORKS",
    "V1_NETWORK_INDICES",
    "default_networks",
    "labels_from_v1_indices",
]
