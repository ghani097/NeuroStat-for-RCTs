"""Filesystem containment and safe artifact-name boundaries for BrainNet."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path, PureWindowsPath
from typing import Any

_SAFE_ARTIFACT_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_WINDOWS_RESERVED = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class RunContainmentError(RuntimeError):
    """Raised before content reads when a run tree is not self-contained."""


def is_safe_artifact_slug(value: Any) -> bool:
    """Return whether a value is one portable, unambiguous path component."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 255
        or value in {".", ".."}
        or value.endswith((".", " "))
        or "/" in value
        or "\\" in value
        or not _SAFE_ARTIFACT_SLUG.fullmatch(value)
    ):
        return False
    windows = PureWindowsPath(value)
    if windows.drive or windows.root or Path(value).is_absolute():
        return False
    return value.split(".", 1)[0].casefold() not in _WINDOWS_RESERVED


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        int(getattr(metadata, "st_file_attributes", 0)),
    )


def _ordinary_lstat(path: Path) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise RunContainmentError(
            f"BrainNet run carrier cannot be inspected safely: {path}"
        ) from exc
    junction_check = getattr(path, "is_junction", None)
    try:
        is_junction = bool(junction_check()) if callable(junction_check) else False
    except OSError as exc:
        raise RunContainmentError(
            f"BrainNet run carrier junction state is unreadable: {path}"
        ) from exc
    is_reparse = bool(int(getattr(metadata, "st_file_attributes", 0)) & _REPARSE_POINT)
    if stat.S_ISLNK(metadata.st_mode) or is_junction or is_reparse:
        raise RunContainmentError(
            f"BrainNet run carrier is a symlink, junction, or reparse point: {path}"
        )
    if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
        raise RunContainmentError(
            f"BrainNet run carrier is not an ordinary file or directory: {path}"
        )
    return metadata


def _same_carrier(
    path: Path, before: os.stat_result, description: str
) -> os.stat_result:
    after = _ordinary_lstat(path)
    if _identity(before) != _identity(after):
        raise RunContainmentError(
            f"BrainNet run carrier changed during containment audit: {description}"
        )
    return after


def audit_run_tree(run_dir: str | Path) -> Path:
    """Audit without following child links, then return the resolved run root."""
    requested = Path(os.path.abspath(os.fspath(run_dir)))
    root_before = _ordinary_lstat(requested)
    if not stat.S_ISDIR(root_before.st_mode):
        raise RunContainmentError(
            f"BrainNet run root is not an ordinary directory: {requested}"
        )
    try:
        root = requested.resolve(strict=True)
    except OSError as exc:
        raise RunContainmentError(
            f"BrainNet run root cannot be resolved safely: {requested}"
        ) from exc
    _same_carrier(requested, root_before, "run root")
    root_resolved = _ordinary_lstat(root)
    if _identity(root_before) != _identity(root_resolved):
        raise RunContainmentError(
            "BrainNet run root changed identity while it was resolved"
        )

    pending = [root]
    while pending:
        directory = pending.pop()
        directory_before = _ordinary_lstat(directory)
        if not stat.S_ISDIR(directory_before.st_mode):
            raise RunContainmentError(
                f"BrainNet run directory changed type during audit: {directory}"
            )
        try:
            scanner = os.scandir(directory)
        except OSError as exc:
            raise RunContainmentError(
                f"BrainNet run directory cannot be enumerated safely: {directory}"
            ) from exc
        with scanner:
            _same_carrier(directory, directory_before, str(directory))
            for entry in scanner:
                candidate = directory / entry.name
                before = _ordinary_lstat(candidate)
                try:
                    resolved = candidate.resolve(strict=True)
                    resolved.relative_to(root)
                except (OSError, ValueError) as exc:
                    raise RunContainmentError(
                        f"BrainNet run carrier resolves outside its run: {candidate}"
                    ) from exc
                _same_carrier(candidate, before, str(candidate))
                resolved_metadata = _ordinary_lstat(resolved)
                if _identity(before) != _identity(resolved_metadata):
                    raise RunContainmentError(
                        "BrainNet run carrier changed identity while it was resolved: "
                        f"{candidate}"
                    )
                if stat.S_ISDIR(before.st_mode):
                    pending.append(candidate)
        _same_carrier(directory, directory_before, str(directory))
    return root
