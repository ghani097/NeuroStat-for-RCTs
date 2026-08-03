from __future__ import annotations

import json

import numpy as np

from scripts.validation import physionet_eoec_v2 as validation


def test_physionet_confirmation_cohort_is_independent_and_fixed() -> None:
    assert validation.DISCOVERY_SUBJECTS == tuple(range(1, 21))
    assert validation.CONFIRMATION_SUBJECTS == tuple(range(21, 110))
    assert not set(validation.DISCOVERY_SUBJECTS) & set(
        validation.CONFIRMATION_SUBJECTS
    )
    assert validation.DATASET_DOI == "10.13026/C28G6P"
    assert validation.PUBLISHED_DOI == "10.1038/s41598-024-68532-2"
    assert validation.BANDS["alpha"] == (8.0, 13.0)
    assert len(validation.POSTERIOR_LABELS) == 14


def test_one_sided_sign_flip_detects_consistent_eye_closure_increase() -> None:
    differences = np.asarray([0.02, 0.03, 0.015, 0.04, 0.025, 0.018])

    p_value = validation._sign_flip_p(differences, seed=4)

    assert p_value < 0.05
    assert validation._paired_d(differences) > 1.0


def test_tracked_confirmation_bundle_matches_manifest_and_passes() -> None:
    root = (
        validation.PROJECT_ROOT
        / "scripts"
        / "validation"
        / "physionet_eoec_v2"
        / "results"
    )
    manifest = json.loads((root / "publication_manifest.json").read_text())

    assert manifest["dataset_doi"] == validation.DATASET_DOI
    assert manifest["published_target_doi"] == validation.PUBLISHED_DOI
    for name, expected in manifest["files"].items():
        assert validation.shared._sha256(root / name) == expected
    results = json.loads((root / "validation_results.json").read_text())
    assert results["primary_replication_passed"] is True
    assert results["participants"] == 89
    assert results["records_verified"] == 178
    assert results["primary"]["difference_ec_minus_eo"] > 0
    assert results["primary"]["one_sided_sign_flip_p"] < 0.05
    assert results["source_data_included"] is False
    assert "E:\\" not in json.dumps(results)
    attributes = (validation.PROJECT_ROOT / ".gitattributes").read_text()
    assert "physionet_eoec_v2/results/*.json -text" in attributes
    assert "physionet_eoec_v2/results/*.md -text" in attributes
