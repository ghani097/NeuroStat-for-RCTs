"""Qt-free study layout detection and EEG file discovery."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Layout(str, Enum):
    """Supported on-disk study layouts."""

    GROUP_SESSION = "group_session"
    GROUP_SUBJECT_SESSION = "group_subject_session"
    GROUP_SESSION_SUBJECT = "group_session_subject"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DiscoveredRecord:
    """One input file discovered from a GUI study setup folder."""

    record_id: str
    participant_id: str
    arm: str
    session: str
    condition: str
    data_kind: str
    format: str
    path: Path
    mat_variable: str | None = None
    samples_per_epoch: int | None = None


_PROBED_LAYOUTS = (
    Layout.GROUP_SESSION,
    Layout.GROUP_SUBJECT_SESSION,
    Layout.GROUP_SESSION_SUBJECT,
)
_SLUG_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SLUG_CHUNK = re.compile(r"[^A-Za-z0-9_.-]+")
_FORMAT_EXTENSIONS = {
    "set": (".set",),
    "fif": (".fif", ".fif.gz"),
    "fieldtrip_mat": (".mat",),
}


def detect_layout(base: str | Path, sessions: Iterable[str] | None = None) -> Layout:
    """Detect a supported study layout by probing for supported EEG files.

    ``<arm>/<subject>/<session>`` and ``<arm>/<session>/<subject>`` are the same
    shape on disk, so probing alone cannot separate them. When the caller knows
    which session names to expect, the layout whose session level actually
    carries those names wins.
    """
    root = Path(base)
    if not root.is_dir():
        return Layout.UNKNOWN
    candidates = {layout: _candidates_for(root, layout) for layout in _PROBED_LAYOUTS}
    detected = [layout for layout, found in candidates.items() if found]
    if len(detected) == 1:
        return detected[0]
    if not detected:
        return Layout.UNKNOWN

    declared = {session.casefold() for session in sessions or ()}
    if not declared:
        return Layout.UNKNOWN
    scores = {
        layout: sum(
            1 for item in candidates[layout] if item.session.casefold() in declared
        )
        for layout in detected
    }
    best = max(scores.values())
    winners = [layout for layout, score in scores.items() if score == best]
    if best > 0 and len(winners) == 1:
        return winners[0]
    return Layout.UNKNOWN


def discover_records(
    base: str | Path,
    arms: Iterable[str],
    sessions: Iterable[str],
    conditions: Iterable[str],
    layout: Layout | None = None,
    *,
    mat_variable: str = "data_ft",
    fieldtrip_data_kind: str = "epochs",
    samples_per_epoch: int | None = None,
) -> tuple[DiscoveredRecord, ...]:
    """Return records from a declared study layout in deterministic order."""
    root = Path(base)
    declared_arms = tuple(arms)
    declared_sessions = tuple(sessions)
    declared_conditions = tuple(conditions)
    selected_layout = (
        detect_layout(root, declared_sessions) if layout is None else layout
    )
    if selected_layout is Layout.UNKNOWN or not root.is_dir():
        return ()
    candidates = _candidates_for(root, selected_layout)
    if not candidates:
        return ()

    participant_slugs: dict[str, str] = {}
    record_slugs: dict[str, str] = {}
    records: list[DiscoveredRecord] = []
    for candidate in sorted(candidates, key=lambda item: _sort_key(root, item.path)):
        if (
            candidate.arm not in declared_arms
            or candidate.session not in declared_sessions
        ):
            continue
        condition = _infer_condition(candidate.path, declared_conditions)
        if condition is None:
            continue
        fmt = format_for_path(candidate.path)
        if fmt is None:
            continue
        # Namespace the participant by arm. Subject folders are numbered per
        # group, so `GroupY/Subject 1` and `GroupZ/Subject 1` are different
        # people; merging them would break the rule that one participant maps
        # to exactly one arm.
        participant_source = f"{candidate.arm}-{candidate.participant_source}"
        participant_id = _slug_unique(
            participant_source,
            participant_slugs,
            "participant_id",
        )
        record_source = f"{participant_source}-{candidate.session}-{condition}"
        record_id = _slug_unique(record_source, record_slugs, "record_id")
        is_fieldtrip = fmt == "fieldtrip_mat"
        records.append(
            DiscoveredRecord(
                record_id=record_id,
                participant_id=participant_id,
                arm=candidate.arm,
                session=candidate.session,
                condition=condition,
                data_kind=fieldtrip_data_kind if is_fieldtrip else "raw",
                format=fmt,
                path=candidate.path,
                mat_variable=mat_variable if is_fieldtrip else None,
                samples_per_epoch=samples_per_epoch if is_fieldtrip else None,
            ),
        )
    return tuple(records)


def slugify(value: str) -> str:
    """Return a deterministic artifact-safe slug."""
    text = _SLUG_CHUNK.sub("-", value.strip().lower())
    text = re.sub(r"-{2,}", "-", text).strip("._-")
    if not text:
        text = "item"
    if not text[0].isalnum():
        text = f"id-{text}"
    if _SLUG_SAFE.fullmatch(text):
        return text
    return re.sub(r"[^A-Za-z0-9_.-]", "-", text)


def format_for_path(path: str | Path) -> str | None:
    """Map a supported EEG file extension to the v2 project format."""
    name = Path(path).name.lower()
    for fmt, extensions in _FORMAT_EXTENSIONS.items():
        if any(name.endswith(extension) for extension in extensions):
            return fmt
    return None


@dataclass(frozen=True, slots=True)
class _Candidate:
    path: Path
    arm: str
    session: str
    participant_source: str


def _candidates_for(root: Path, layout: Layout) -> tuple[_Candidate, ...]:
    if layout is Layout.GROUP_SESSION:
        return _group_session_files(root)
    if layout is Layout.GROUP_SUBJECT_SESSION:
        return _group_subject_session_files(root)
    if layout is Layout.GROUP_SESSION_SUBJECT:
        return _group_session_subject_files(root)
    return ()


def _supported_file(path: Path) -> bool:
    return (
        path.is_file()
        and not _is_sidecar(path)
        and format_for_path(path) is not None
        and not _is_cleaned(path)
    )


def _is_sidecar(path: Path) -> bool:
    """Reject dotfiles and macOS AppleDouble resource forks.

    Data copied via macOS carries a ``._name`` sidecar beside every real file.
    They share the data extension, so without this every record is discovered
    twice.
    """
    return path.name.startswith(".")


def _is_cleaned(path: Path) -> bool:
    return _data_stem(path).lower().endswith("_cleaned")


def _data_stem(path: Path) -> str:
    name = path.name
    lowered = name.lower()
    for extension in (".fif.gz", ".set", ".fif", ".mat"):
        if lowered.endswith(extension):
            return name[: -len(extension)]
    return path.stem


def _group_session_files(root: Path) -> tuple[_Candidate, ...]:
    records: list[_Candidate] = []
    for arm_dir in _dirs(root):
        for session_dir in _dirs(arm_dir):
            for path in _files(session_dir):
                if _supported_file(path):
                    records.append(
                        _Candidate(
                            path, arm_dir.name, session_dir.name, _data_stem(path)
                        ),
                    )
    return tuple(records)


def _group_subject_session_files(root: Path) -> tuple[_Candidate, ...]:
    records: list[_Candidate] = []
    for arm_dir in _dirs(root):
        for subject_dir in _dirs(arm_dir):
            for session_dir in _dirs(subject_dir):
                for path in _files(session_dir):
                    if _supported_file(path):
                        records.append(
                            _Candidate(
                                path,
                                arm_dir.name,
                                session_dir.name,
                                subject_dir.name,
                            ),
                        )
    return tuple(records)


def _group_session_subject_files(root: Path) -> tuple[_Candidate, ...]:
    records: list[_Candidate] = []
    for arm_dir in _dirs(root):
        for session_dir in _dirs(arm_dir):
            for subject_dir in _dirs(session_dir):
                for path in _files(subject_dir):
                    if _supported_file(path):
                        records.append(
                            _Candidate(
                                path,
                                arm_dir.name,
                                session_dir.name,
                                subject_dir.name,
                            ),
                        )
    return tuple(records)


def _dirs(path: Path) -> tuple[Path, ...]:
    return tuple(child for child in path.iterdir() if child.is_dir())


def _files(path: Path) -> tuple[Path, ...]:
    return tuple(child for child in path.iterdir() if child.is_file())


def _infer_condition(path: Path, conditions: tuple[str, ...]) -> str | None:
    if len(conditions) == 1:
        return conditions[0]
    stem_slug = slugify(_data_stem(path))
    padded = f"-{stem_slug}-"
    matches = [
        condition for condition in conditions if f"-{slugify(condition)}-" in padded
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _slug_unique(source: str, seen: dict[str, str], field_name: str) -> str:
    slug = slugify(source)
    previous = seen.setdefault(slug, source)
    if previous != source:
        raise ValueError(
            f"{field_name} slug collision: {previous!r} and {source!r} both become {slug!r}",
        )
    return slug


def _sort_key(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix().lower()
    except ValueError:
        return path.as_posix().lower()


__all__ = [
    "DiscoveredRecord",
    "Layout",
    "detect_layout",
    "discover_records",
    "format_for_path",
    "slugify",
]
