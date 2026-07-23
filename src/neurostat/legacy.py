"""Explicit, lazy access to the unmodified v1 pipeline.

Importing this module is safe for v2 CLI and GUI clients. Call
``load_legacy_pipeline`` only when a source checkout needs the legacy
``pli_pipeline`` module and its scientific dependencies.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def load_legacy_pipeline() -> ModuleType:
    """Import and return the top-level v1 :mod:`pli_pipeline` module lazily."""
    return import_module("pli_pipeline")
