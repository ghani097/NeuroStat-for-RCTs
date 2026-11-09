"""Qt-free controller layer for the NeuroStat v2 GUI."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from neurostat.contracts import (
    ProgressCallback,
    ProgressEvent,
    StageName,
    StageResult,
    StageState,
)
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
from .graphvar_setup import build_brainnet_block, supported_workspaces
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
        requested = tuple(stages)
        plan = self.resolve_plan(requested)
        base_folder = Path(form.base_folder)
        if not form.already_cleaned:
            base_folder = self._preprocessor(form, _legacy_log_adapter(progress))
        records = self._discoverer(
            base_folder,
            form.arms,
            form.sessions,
            form.conditions,
        )
        problems = validate_form(form, records, plan)
        if problems:
            message = "Study setup is not valid:\n- " + "\n- ".join(problems)
            raise ValueError(message)
        document = build_project_document(form, records, stages=plan)
        project = self._project_writer(document, form.resolved_project_file)
        if StageName.BRAINNET not in plan or self._runner is not None:
            return self.run(project, requested, progress=progress, outcome=outcome)
        # BrainNet can only draw workspaces whose component passed node pFWE,
        # which is known only after GraphVar. Run up to GraphVar on the main
        # project, then add the brainnet block in a sibling project file (the
        # GraphVar identity ignores that block, so nothing upstream reruns).
        upstream = tuple(stage for stage in plan if stage is not StageName.BRAINNET)
        result = self.run(project, upstream, progress=progress, outcome=outcome)
        return self._run_brainnet(form, document, result, progress, outcome)

    def _run_brainnet(
        self,
        form: StudyForm,
        document: dict[str, Any],
        result: WorkflowResult,
        progress: ProgressHandler | None,
        outcome: OutcomeHandler | None,
    ) -> WorkflowResult:
        if form.brainnet is None or form.graphvar is None:
            raise ValueError("BrainNet needs GraphVar and BrainNet settings.")
        graphvar_run = result[StageName.GRAPHVAR].run
        graphvar_dir = result[StageName.GRAPHVAR].run_dir
        if graphvar_run is None or graphvar_dir is None:
            raise ValueError("GraphVar did not produce a run for BrainNet to draw.")
        workspaces = supported_workspaces(graphvar_dir, form.brainnet.top_k)
        if not workspaces:
            message = (
                "BrainNet skipped: no GraphVar component passed the node-pFWE test "
                f"with at least {form.brainnet.top_k} links, so there is nothing to draw."
            )
            if progress is not None:
                progress(ProgressEvent(StageName.BRAINNET, message))
            skipped = StageOutcome(
                StageResult(StageName.BRAINNET, StageState.SKIPPED, message=message)
            )
            if outcome is not None:
                outcome(skipped)
            return WorkflowResult(result.outcomes + (skipped,))
        if progress is not None:
            progress(
                ProgressEvent(
                    StageName.BRAINNET,
                    "drawing supported components: " + ", ".join(workspaces),
                )
            )
        brainnet_document = dict(document)
        brainnet_document["brainnet"] = build_brainnet_block(
            form.brainnet, form.graphvar, workspaces
        )
        project_file = form.resolved_project_file
        brainnet_project = self._project_writer(
            brainnet_document,
            project_file.with_name(project_file.name.replace(".project.yaml", "") + ".brainnet.project.yaml"),
        )
        services = self._services
        try:
            if services is not None and services.brainnet_build is not None:
                artifact_run = services.brainnet_build(brainnet_project, graphvar_run)
            else:
                from neurostat.brainnet import build_brainnet_artifacts

                artifact_run = build_brainnet_artifacts(brainnet_project, cast(Any, graphvar_run))
            if services is not None and services.brainnet_render is not None:
                rendered = services.brainnet_render(brainnet_project, artifact_run)
            else:
                from neurostat.brainnet import render_brainnet_artifacts

                rendered = render_brainnet_artifacts(brainnet_project, cast(Any, artifact_run))
        except (RuntimeError, ValueError, OSError) as exc:
            failed = StageOutcome(StageResult(StageName.BRAINNET, StageState.FAILED, message=str(exc)))
            partial = WorkflowResult(result.outcomes + (failed,))
            if outcome is not None:
                outcome(failed)
            raise ControllerWorkflowError(StageName.BRAINNET, partial, str(exc)) from exc
        done = StageOutcome(StageResult(StageName.BRAINNET, StageState.SUCCEEDED), rendered)
        if progress is not None:
            progress(ProgressEvent(StageName.BRAINNET, "brainnet succeeded"))
        if outcome is not None:
            outcome(done)
        return WorkflowResult(result.outcomes + (done,))

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
