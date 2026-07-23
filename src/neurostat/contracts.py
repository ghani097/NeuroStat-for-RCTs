"""Dependency-light contracts shared by future NeuroStat v2 stages.

The contracts in this module intentionally contain no scientific or GUI
imports, so CLI and GUI callers can share the same stage boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol


class StageName(str, Enum):
    """Public names for the six v2 pipeline stages."""

    INGEST = "ingest"
    SCOUTS = "scouts"
    CONNECTIVITY = "connectivity"
    NETWORK_MODELS = "network_models"
    GRAPHVAR = "graphvar"
    BRAINNET = "brainnet"


class StageState(str, Enum):
    """Lifecycle states reported by a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


def _require_nonblank(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")


def _require_path(value: Path, field_name: str) -> None:
    if not isinstance(value, Path):
        raise TypeError(f"{field_name} must be a pathlib.Path")


@dataclass(frozen=True, slots=True)
class RecordKey:
    """Stable identifier for one participant's study record."""

    participant_id: str
    arm: str
    session: str
    condition: str

    def __post_init__(self) -> None:
        for field_name in ("participant_id", "arm", "session", "condition"):
            _require_nonblank(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Reference to a stage artifact stored on disk."""

    kind: str
    path: Path

    def __post_init__(self) -> None:
        _require_nonblank(self.kind, "kind")
        _require_path(self.path, "path")


@dataclass(frozen=True, slots=True)
class StageResult:
    """Immutable outcome reported after one pipeline stage completes."""

    stage: StageName
    state: StageState
    artifacts: tuple[ArtifactRef, ...] = ()
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, StageName):
            raise TypeError("stage must be a StageName")
        if not isinstance(self.state, StageState):
            raise TypeError("state must be a StageState")
        artifacts = tuple(self.artifacts)
        if not all(isinstance(artifact, ArtifactRef) for artifact in artifacts):
            raise TypeError("artifacts must contain ArtifactRef values")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be a string or None")
        object.__setattr__(self, "artifacts", artifacts)


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """A progress update emitted while a stage is running."""

    stage: StageName
    message: str
    completed: int | None = None
    total: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, StageName):
            raise TypeError("stage must be a StageName")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        for field_name in ("completed", "total"):
            value = getattr(self, field_name)
            if value is not None and type(value) is not int:
                raise TypeError(f"{field_name} must be an int or None")
        if self.completed is not None and self.completed < 0:
            raise ValueError("completed must be nonnegative")
        if self.total is not None and self.total < 0:
            raise ValueError("total must be nonnegative")
        if self.completed is not None and self.total is not None and self.completed > self.total:
            raise ValueError("completed cannot exceed total")


class ProgressCallback(Protocol):
    """Callable protocol used by stages to publish progress events."""

    def __call__(self, event: ProgressEvent) -> None:
        """Handle one progress event."""


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Filesystem locations that identify a NeuroStat project and its outputs."""

    project_file: Path
    output_root: Path

    def __post_init__(self) -> None:
        _require_path(self.project_file, "project_file")
        _require_path(self.output_root, "output_root")
