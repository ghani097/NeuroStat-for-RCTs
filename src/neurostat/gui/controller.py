"""Qt-free controller layer for the NeuroStat v2 GUI."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from neurostat.contracts import ProgressCallback, ProgressEvent, StageName
from neurostat.project import CleanProject
from neurostat.workflow import (
    StageOutcome,
    WorkflowError,
    WorkflowResult,
    WorkflowServices,
    WorkflowStage,
    resolve_workflow_stages,
    run_workflow,
)

from .discovery import DiscoveredRecord, discover_records
from .legacy_bridge import run_v1_preprocessing
from .study_builder import (
    StudyForm,
    build_project_document,
    validate_form,
    write_project,
)

ProjectInput = CleanProject | str | Path
StageSelection = Iterable[WorkflowStage | str]
ProgressHandler = Callable[[ProgressEvent], None]
OutcomeHandler = Callable[[StageOutcome], None]
WorkflowRunner = Callable[
    [ProjectInput, Iterable[WorkflowStage | str]],
    WorkflowResult,
]
LegacyLogHandler = Callable[[str], None]
StudyPreprocessor = Callable[[StudyForm, LegacyLogHandler | None], Path]
StudyDiscoverer = Callable[
    [Path, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    tuple[DiscoveredRecord, ...],
]
ProjectWriter = Callable[[dict[str, Any], Path], CleanProject]


def _legacy_log_adapter(
    progress: ProgressHandler | None,
) -> LegacyLogHandler | None:
    """Adapt v1's plain-string progress callback to v2 progress events.

    v1 calls ``progress_cb`` with log lines, while v2 handlers expect a
    ``ProgressEvent``. Preprocessing feeds ingest, so its messages are reported
    against that stage.
    """
    if progress is None:
        return None

    def handle(message: str) -> None:
        progress(ProgressEvent(StageName.INGEST, str(message)))

    return handle


@dataclass(frozen=True, slots=True)
class ControllerWorkflowError(RuntimeError):
    """Workflow failure with the engine's stage name and partial result."""

    stage: WorkflowStage
    partial: WorkflowResult
    message: str

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)


class WorkflowController:
    """Resolve and run v2 workflow requests without importing Qt."""

    def __init__(
        self,
        *,
        services: WorkflowServices | None = None,
        runner: WorkflowRunner | None = None,
        preprocessor: StudyPreprocessor | None = None,
        discoverer: StudyDiscoverer | None = None,
        project_writer: ProjectWriter | None = None,
    ) -> None:
        self._services = services
        self._runner = runner
        self._preprocessor = preprocessor or run_v1_preprocessing
        self._discoverer = discoverer or discover_records
        self._project_writer = project_writer or write_project

    def resolve_plan(self, stages: StageSelection) -> tuple[WorkflowStage, ...]:
        """Return the ordered dependency closure for the requested stages."""
        return resolve_workflow_stages(tuple(stages))

    def run(
        self,
        project: ProjectInput,
        stages: StageSelection,
        *,
        progress: ProgressHandler | None = None,
        outcome: OutcomeHandler | None = None,
    ) -> WorkflowResult:
        """Run the workflow and forward progress and stage outcomes."""
        requested = tuple(stages)
        self.resolve_plan(requested)
        try:
            result = self._run_engine(project, requested, progress)
        except WorkflowError as exc:
            for item in exc.partial.outcomes:
                if outcome is not None:
                    outcome(item)
            raise ControllerWorkflowError(exc.stage, exc.partial, str(exc)) from exc
        for item in result.outcomes:
            if outcome is not None:
                outcome(item)
        return result

    def run_study(
        self,
        form: StudyForm,
        stages: StageSelection,
        *,
        progress: ProgressHandler | None = None,
        outcome: OutcomeHandler | None = None,
    ) -> WorkflowResult:
        """Generate a project from a study form, then run the workflow."""
        base_folder = Path(form.base_folder)
        if not form.already_cleaned:
            base_folder = self._preprocessor(form, _legacy_log_adapter(progress))
        records = self._discoverer(
            base_folder,
            form.arms,
            form.sessions,
            form.conditions,
        )
        problems = validate_form(form, records)
        if problems:
            message = "Study setup is not valid:\n- " + "\n- ".join(problems)
            raise ValueError(message)
        document = build_project_document(form, records)
        project = self._project_writer(document, form.resolved_project_file)
        return self.run(project, stages, progress=progress, outcome=outcome)

    def _run_engine(
        self,
        project: ProjectInput,
        stages: tuple[WorkflowStage | str, ...],
        progress: ProgressHandler | None,
    ) -> WorkflowResult:
        if self._runner is not None:
            return self._runner(project, stages)
        # ProgressHandler is a plain one-argument callable; the engine declares the
        # equivalent ProgressCallback protocol, whose parameter is named.
        return run_workflow(
            project,
            stages,
            progress=cast(ProgressCallback | None, progress),
            services=self._services,
        )


__all__ = [
    "ControllerWorkflowError",
    "OutcomeHandler",
    "ProgressHandler",
    "ProjectInput",
    "ProjectWriter",
    "StageSelection",
    "StudyDiscoverer",
    "StudyPreprocessor",
    "WorkflowController",
]
