"""Dependency-light orchestration for high-level NeuroStat workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..contracts import (
    ArtifactRef,
    ProgressCallback,
    ProgressEvent,
    StageName,
    StageResult,
    StageState,
)
from ..project import CleanProject, load_project

WorkflowStage = StageName
StageCallable = Callable[..., object]

CANONICAL_STAGE_ORDER: tuple[StageName, ...] = (
    StageName.INGEST,
    StageName.SCOUTS,
    StageName.CONNECTIVITY,
    StageName.NETWORK_MODELS,
    StageName.GRAPHVAR,
    StageName.BRAINNET,
)

_DEPENDENCIES: Mapping[StageName, tuple[StageName, ...]] = {
    StageName.INGEST: (StageName.INGEST,),
    StageName.SCOUTS: (StageName.INGEST, StageName.SCOUTS),
    StageName.CONNECTIVITY: (
        StageName.INGEST,
        StageName.SCOUTS,
        StageName.CONNECTIVITY,
    ),
    StageName.NETWORK_MODELS: (
        StageName.INGEST,
        StageName.SCOUTS,
        StageName.CONNECTIVITY,
        StageName.NETWORK_MODELS,
    ),
    StageName.GRAPHVAR: (
        StageName.INGEST,
        StageName.SCOUTS,
        StageName.CONNECTIVITY,
        StageName.GRAPHVAR,
    ),
    StageName.BRAINNET: (
        StageName.INGEST,
        StageName.SCOUTS,
        StageName.CONNECTIVITY,
        StageName.GRAPHVAR,
        StageName.BRAINNET,
    ),
}

_CONFIG_BLOCKS: Mapping[StageName, str] = {
    StageName.NETWORK_MODELS: "network_models",
    StageName.GRAPHVAR: "graphvar",
    StageName.BRAINNET: "brainnet",
}


@dataclass(frozen=True, slots=True)
class WorkflowServices:
    """Optional service callables for fast orchestration tests."""

    ingest: StageCallable | None = None
    scouts: StageCallable | None = None
    connectivity: StageCallable | None = None
    network_models: StageCallable | None = None
    graphvar_build: StageCallable | None = None
    graphvar_analyse: StageCallable | None = None
    brainnet_build: StageCallable | None = None
    brainnet_render: StageCallable | None = None


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """One workflow stage result plus the concrete run object, if produced."""

    result: StageResult
    run: object | None = None

    @property
    def stage(self) -> StageName:
        return self.result.stage

    @property
    def state(self) -> StageState:
        return self.result.state

    @property
    def run_id(self) -> str | None:
        value = getattr(self.run, "run_id", None)
        return value if isinstance(value, str) else None

    @property
    def run_dir(self) -> Path | None:
        value = getattr(self.run, "run_dir", None)
        if value is None:
            return None
        return value if isinstance(value, Path) else Path(value)

    @property
    def reused(self) -> object | None:
        return getattr(self.run, "reused", None)


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Ordered workflow outcomes with stage lookup and aggregate status."""

    outcomes: tuple[StageOutcome, ...]

    @property
    def by_stage(self) -> Mapping[StageName, StageOutcome]:
        return {outcome.stage: outcome for outcome in self.outcomes}

    @property
    def success(self) -> bool:
        return bool(self.outcomes) and all(
            outcome.state is StageState.SUCCEEDED for outcome in self.outcomes
        )

    def __getitem__(self, stage: StageName | str) -> StageOutcome:
        return self.by_stage[_coerce_stage(stage)]


class WorkflowError(RuntimeError):
    """Raised when a workflow stage fails after preserving partial results."""

    def __init__(self, stage: StageName, partial: WorkflowResult, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.partial = partial


def _coerce_stage(value: StageName | str) -> StageName:
    if isinstance(value, StageName):
        return value
    if value == "all":
        raise ValueError("'all' is a workflow selector, not a concrete stage")
    try:
        return StageName(value)
    except ValueError:
        allowed = ", ".join(stage.value for stage in CANONICAL_STAGE_ORDER)
        raise ValueError(
            f"unknown workflow stage {value!r}; expected one of: {allowed}, all"
        ) from None


def resolve_workflow_stages(stages: Iterable[StageName | str]) -> tuple[StageName, ...]:
    """Resolve requested stages into the ordered dependency closure."""
    requested_values = tuple(stages)
    if not requested_values:
        raise ValueError(
            "empty workflow stage selection; pass 'all' explicitly to run every stage"
        )
    if any(value == "all" for value in requested_values):
        return CANONICAL_STAGE_ORDER
    selected: set[StageName] = set()
    for value in requested_values:
        stage = _coerce_stage(value)
        selected.update(_DEPENDENCIES[stage])
    return tuple(stage for stage in CANONICAL_STAGE_ORDER if stage in selected)


plan_workflow_stages = resolve_workflow_stages


def run_workflow(
    project: CleanProject | str | Path,
    stages: Iterable[StageName | str],
    *,
    progress: ProgressCallback | None = None,
    services: WorkflowServices | None = None,
) -> WorkflowResult:
    """Run the requested high-level workflow using one shared orchestration path."""
    clean_project = (
        project if isinstance(project, CleanProject) else load_project(project)
    )
    plan = resolve_workflow_stages(stages)
    service_set = services or WorkflowServices()
    missing = _first_missing_config(clean_project, plan)
    if missing is not None:
        stage, block = missing
        message = (
            f"workflow stage {stage.value} requires a project {block}: block; "
            f"add {block}: to the project YAML or omit that stage"
        )
        partial = _failure_result(plan, (), stage, message)
        raise WorkflowError(stage, partial, message)

    outcomes: list[StageOutcome] = []
    runs: dict[StageName, object] = {}
    for index, stage in enumerate(plan):
        _emit(progress, stage, f"{stage.value} started")
        try:
            run = _execute_stage(stage, clean_project, runs, service_set, progress)
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            _emit(progress, stage, f"{stage.value} failed")
            partial = _failure_result(plan, tuple(outcomes), stage, message, index)
            raise WorkflowError(stage, partial, message) from exc
        outcome = _outcome(stage, StageState.SUCCEEDED, run, "succeeded")
        outcomes.append(outcome)
        runs[stage] = run
        _emit(progress, stage, f"{stage.value} succeeded")
    return WorkflowResult(tuple(outcomes))


def _first_missing_config(
    project: CleanProject,
    plan: tuple[StageName, ...],
) -> tuple[StageName, str] | None:
    for stage in plan:
        block = _CONFIG_BLOCKS.get(stage)
        if block is not None and block not in project.extras:
            return stage, block
    return None


def _failure_result(
    plan: tuple[StageName, ...],
    completed: tuple[StageOutcome, ...],
    failed_stage: StageName,
    message: str,
    failed_index: int | None = None,
) -> WorkflowResult:
    if failed_index is None:
        failed_index = plan.index(failed_stage)
    outcomes = list(completed)
    while len(outcomes) < failed_index:
        outcomes.append(_outcome(plan[len(outcomes)], StageState.PENDING))
    outcomes.append(_outcome(failed_stage, StageState.FAILED, message=message))
    outcomes.extend(
        _outcome(stage, StageState.PENDING) for stage in plan[failed_index + 1 :]
    )
    return WorkflowResult(tuple(outcomes))


def _execute_stage(
    stage: StageName,
    project: CleanProject,
    runs: Mapping[StageName, object],
    services: WorkflowServices,
    progress: ProgressCallback | None,
) -> object:
    if stage is StageName.INGEST:
        if services.ingest is not None:
            return services.ingest(project, progress)
        from ..ingest import ingest_project

        return ingest_project(project, progress)
    if stage is StageName.SCOUTS:
        if services.scouts is not None:
            return services.scouts(project, runs[StageName.INGEST], progress)
        from ..scouts import run_scouts

        return run_scouts(project, cast(Any, runs[StageName.INGEST]), progress)
    if stage is StageName.CONNECTIVITY:
        if services.connectivity is not None:
            return services.connectivity(project, runs[StageName.SCOUTS], progress)
        from ..connectivity import run_connectivity

        return run_connectivity(project, cast(Any, runs[StageName.SCOUTS]), progress)
    if stage is StageName.NETWORK_MODELS:
        if services.network_models is not None:
            return services.network_models(
                project,
                _run_dir(runs[StageName.CONNECTIVITY]),
                progress,
            )
        from ..network_models import run_network_models

        return run_network_models(
            project,
            _run_dir(runs[StageName.CONNECTIVITY]),
            progress,
        )
    if stage is StageName.GRAPHVAR:
        if services.graphvar_build is not None:
            build = services.graphvar_build
        else:
            from ..graphvar import build_graphvar_workspaces as build
        workspace_run = build(project, _run_dir(runs[StageName.CONNECTIVITY]))
        if services.graphvar_analyse is not None:
            return services.graphvar_analyse(project, workspace_run)
        from ..graphvar import run_graphvar_analysis

        return run_graphvar_analysis(project, cast(Any, workspace_run))
    if stage is StageName.BRAINNET:
        if services.brainnet_build is not None:
            artifact_run = services.brainnet_build(project, runs[StageName.GRAPHVAR])
        else:
            from ..brainnet import build_brainnet_artifacts

            artifact_run = build_brainnet_artifacts(
                project,
                cast(Any, runs[StageName.GRAPHVAR]),
            )
        if services.brainnet_render is not None:
            return services.brainnet_render(project, artifact_run)
        from ..brainnet import render_brainnet_artifacts

        return render_brainnet_artifacts(project, cast(Any, artifact_run))
    raise AssertionError(f"unhandled workflow stage: {stage}")


def _run_dir(run: object) -> Path:
    value = getattr(run, "run_dir", None)
    if value is None:
        raise RuntimeError("upstream run object has no run_dir")
    return value if isinstance(value, Path) else Path(value)


def _outcome(
    stage: StageName,
    state: StageState,
    run: object | None = None,
    message: str | None = None,
) -> StageOutcome:
    run_dir = getattr(run, "run_dir", None)
    artifacts = ()
    if run_dir is not None:
        artifacts = (ArtifactRef("run_dir", _run_dir(run)),)
    return StageOutcome(StageResult(stage, state, artifacts, message), run)


def _emit(
    progress: ProgressCallback | None,
    stage: StageName,
    message: str,
) -> None:
    if progress is not None:
        progress(ProgressEvent(stage, message))


__all__ = [
    "CANONICAL_STAGE_ORDER",
    "StageOutcome",
    "WorkflowError",
    "WorkflowResult",
    "WorkflowServices",
    "WorkflowStage",
    "plan_workflow_stages",
    "resolve_workflow_stages",
    "run_workflow",
]
