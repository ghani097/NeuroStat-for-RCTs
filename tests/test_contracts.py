"""Tests for the dependency-light v2 contract boundary."""

from pathlib import Path

import pytest

from neurostat.contracts import ArtifactRef, ProgressEvent, ProjectPaths, RecordKey, StageName, StageResult, StageState


def test_stage_name_values_are_the_public_pipeline() -> None:
    assert [stage.value for stage in StageName] == ["ingest", "scouts", "connectivity", "network_models", "graphvar", "brainnet"]


def test_stage_state_values_are_the_public_lifecycle() -> None:
    assert [state.value for state in StageState] == ["pending", "running", "succeeded", "failed", "skipped"]


@pytest.mark.parametrize("field_name", ["participant_id", "arm", "session", "condition"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_record_key_rejects_blank_identifiers(field_name: str, value: object) -> None:
    values: dict[str, object] = {"participant_id": "P01", "arm": "control", "session": "baseline", "condition": "eyes_closed"}
    values[field_name] = value
    with pytest.raises(ValueError, match=field_name):
        RecordKey(**values)  # type: ignore[arg-type]


def test_record_key_is_frozen_and_slotted() -> None:
    record = RecordKey("P01", "control", "baseline", "eyes_closed")
    with pytest.raises(AttributeError):
        record.arm = "treatment"  # type: ignore[misc]
    assert not hasattr(record, "__dict__")


@pytest.mark.parametrize("kind", ["", "  ", None])
def test_artifact_ref_rejects_blank_kind(kind: object) -> None:
    with pytest.raises(ValueError, match="kind"):
        ArtifactRef(kind, Path("output.csv"))  # type: ignore[arg-type]


def test_contract_paths_are_pathlib_paths() -> None:
    artifact = ArtifactRef("connectivity", Path("output.csv"))
    paths = ProjectPaths(Path("project.yaml"), Path("results"))
    assert artifact.path == Path("output.csv")
    assert paths.output_root == Path("results")
    with pytest.raises(TypeError, match="project_file"):
        ProjectPaths("project.yaml", Path("results"))  # type: ignore[arg-type]


def test_stage_result_converts_artifacts_to_an_immutable_tuple() -> None:
    refs = [ArtifactRef("matrix", Path("matrix.csv"))]
    result = StageResult(StageName.CONNECTIVITY, StageState.SUCCEEDED, refs)
    refs.append(ArtifactRef("plot", Path("plot.png")))
    assert result.artifacts == (ArtifactRef("matrix", Path("matrix.csv")),)


def test_stage_result_rejects_a_non_string_message() -> None:
    with pytest.raises(TypeError, match="message"):
        StageResult(StageName.CONNECTIVITY, StageState.SUCCEEDED, message=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(("completed", "total", "message"), [(-1, None, "completed"), (None, -1, "total"), (3, 2, "cannot exceed")])
def test_progress_event_rejects_invalid_counts(completed: int | None, total: int | None, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProgressEvent(StageName.INGEST, "reading", completed, total)


@pytest.mark.parametrize(("completed", "total"), [(1.5, None), (True, None), (None, 1.5), (None, False)])
def test_progress_event_rejects_non_integer_counts(completed: object, total: object) -> None:
    with pytest.raises(TypeError, match="int"):
        ProgressEvent(StageName.INGEST, "reading", completed, total)  # type: ignore[arg-type]


def test_progress_event_rejects_a_non_string_message() -> None:
    with pytest.raises(TypeError, match="message"):
        ProgressEvent(StageName.INGEST, 1)  # type: ignore[arg-type]


def test_progress_event_allows_unknown_or_valid_progress() -> None:
    assert ProgressEvent(StageName.INGEST, "reading") == ProgressEvent(StageName.INGEST, "reading")
    assert ProgressEvent(StageName.INGEST, "reading", 0, 0).completed == 0
    assert ProgressEvent(StageName.INGEST, "reading", 2, 2).total == 2
