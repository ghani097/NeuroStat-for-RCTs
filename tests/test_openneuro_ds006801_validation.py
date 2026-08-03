from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validation import openneuro_ds006801_v2 as validation


def test_public_validation_cohort_is_fixed_and_complete() -> None:
    assert validation.DATASET_DOI == "10.18112/openneuro.ds006801.v1.0.0"
    assert validation.DATASET_SNAPSHOT == "c4e91b19132fed86ee24a42ec5105bf14f961011"
    assert validation.GROUPS == {
        "active": ("sub-01", "sub-05", "sub-07", "sub-10", "sub-15", "sub-16"),
        "control": (
            "sub-02",
            "sub-09",
            "sub-12",
            "sub-14",
            "sub-17",
            "sub-18",
            "sub-22",
        ),
    }
    assert validation.PILOT_GROUPS == {
        "active": ("sub-01",),
        "control": ("sub-02",),
    }


def test_source_paths_match_the_pinned_bids_layout() -> None:
    assert validation._relative_source("sub-01", "pre", "vhdr") == (
        "sub-01/ses-pre/eeg/sub-01_ses-pre_task-rest_eeg.vhdr"
    )
    assert validation._record_stem("sub-22", "post") == (
        "sub-22_ses-post_task-rest_eeg"
    )


def test_exact_permutation_p_enumerates_every_group_assignment() -> None:
    changes = np.asarray([2.0, 3.0, -1.0, 0.0])
    groups = np.asarray(["active", "active", "control", "control"])

    assert validation._exact_permutation_p(changes, groups, observed=3.0) == 2 / 6


def test_fdr_bh_is_monotone_in_rank_and_restores_input_order() -> None:
    adjusted = validation._fdr_bh([0.04, 0.01, 0.03])

    assert adjusted == [0.04, 0.03, 0.04]


def test_atomic_json_replaces_nonfinite_values_with_null(tmp_path: Path) -> None:
    target = tmp_path / "result.json"

    validation._atomic_json(target, {"finite": 1.0, "missing": float("nan")})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "finite": 1.0,
        "missing": None,
    }


def test_cleaned_output_reuse_requires_the_fixed_cleaning_contract() -> None:
    cleaning = {
        "resample_hz": 200.0,
        "bandpass_hz": [1.0, 45.0],
        "reference": "average_applied",
        "bad_channel_method": "log-variance robust z > 6",
        "ica_method": "extended_infomax",
        "ica_components": 25,
        "ica_random_state": 42,
        "window_seconds": 4.0,
        "window_peak_to_peak_limit_uv": 150.0,
        "window_absolute_peak_limit_uv": 150.0,
        "window_log_rms_robust_z_limit": 8.0,
    }
    qc = {"schema": "neurostat.public_cleaning_qc.v1", "cleaning": cleaning}

    assert validation._matches_cleaning_contract(qc)
    cleaning["window_seconds"] = 2.0
    assert not validation._matches_cleaning_contract(qc)


def test_run_v2_records_and_raises_on_workflow_failure(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project.yaml"
    project.write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "neurostat.workflow.run_workflow",
        lambda *_args, **_kwargs: SimpleNamespace(success=False, outcomes=()),
    )

    with pytest.raises(RuntimeError, match="did not complete successfully"):
        validation.run_v2(project)

    result = json.loads((tmp_path / "workflow_result.json").read_text())
    assert result["success"] is False


def test_complete_partial_download_is_finalized_without_network(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "record.eeg"
    partial = destination.with_suffix(".eeg.part")
    partial.write_bytes(b"complete")
    spec = validation.SourceSpec(
        "sub-01/ses-pre/eeg/record.eeg",
        partial.stat().st_size,
        validation._sha256(partial),
    )
    monkeypatch.setattr(
        validation.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )

    result = validation._download(spec, destination)

    assert destination.read_bytes() == b"complete"
    assert not partial.exists()
    assert result["sha256"] == spec.sha256


def test_project_paths_fall_back_to_absolute_across_windows_volumes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        validation.os.path, "relpath", lambda *_args: (_ for _ in ()).throw(ValueError)
    )

    result = validation._relative_to_project(Path("project"), Path("resource"))

    assert Path(result).is_absolute()


def test_tracked_publication_bundle_matches_its_manifest() -> None:
    root = (
        validation.PROJECT_ROOT
        / "scripts"
        / "validation"
        / "openneuro_ds006801"
        / "results"
    )
    manifest = json.loads((root / "publication_manifest.json").read_text())

    assert manifest["dataset_doi"] == validation.DATASET_DOI
    for name, expected in manifest["files"].items():
        assert validation._sha256(root / name) == expected
    results = json.loads((root / "validation_results.json").read_text())
    assert results["records_verified"] == 26
    assert results["source_data_included"] is False
    assert "E:\\" not in json.dumps(results)
