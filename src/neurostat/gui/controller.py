"""Qt-free controller layer for the NeuroStat v2 GUI."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from neurostat.contracts import ProgressCallback, ProgressEvent
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

ProjectInput = CleanProject | str | Path
StageSelection = Iterable[WorkflowStage | str]
ProgressHandler = Callable[[ProgressEvent], None]
OutcomeHandler = Callable[[StageOutcome], None]
WorkflowRunner = Callable[
    [ProjectInput, Iterable[WorkflowStage | str]],
    WorkflowResult,
]


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
    ) -> None:
        self._services = services
        self._runner = runner

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
    "StageSelection",
    "WorkflowController",
]
