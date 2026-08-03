from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from scripts.validation import lemon_discover_eeg_v2 as validation


def _participants() -> list[validation.Participant]:
    values = []
    number = 1
    for age_group, age_bin in (("young", "20-25"), ("old", "60-65")):
        for sex in ("female", "male"):
            for _ in range(20):
                values.append(
                    validation.Participant(f"sub-{number:06d}", age_bin, age_group, sex)
                )
                number += 1
    return values


def test_replication_cohort_is_deterministic_balanced_and_outcome_independent() -> None:
    selected = validation.select_cohort(_participants(), "replication")

    assert len(selected) == 60
    assert len({item.participant_id for item in selected}) == 60
    assert {
        (age, sex): sum(item.age_group == age and item.sex == sex for item in selected)
        for age in ("young", "old")
        for sex in ("female", "male")
    } == {
        ("young", "female"): 15,
        ("young", "male"): 15,
        ("old", "female"): 15,
        ("old", "male"): 15,
    }
    assert selected == validation.select_cohort(
        reversed(_participants()), "replication"
    )
    assert validation.SELECTION_SALT == "neurostat-v2-lemon-replication-v1"


def test_published_target_and_primary_hypothesis_are_frozen() -> None:
    assert validation.DATASET_DOI == "10.1038/sdata.2018.308"
    assert validation.DISCOVER_DOI == "10.1038/s41597-023-02525-0"
    assert validation.DISCOVER_CODE_COMMIT == "b92295d5b2dd0a496c3340fc95ddfa54e40236e9"
    assert validation.AGE_GROUPS == {
        "young": {"20-25", "25-30", "30-35"},
        "old": {"55-60", "60-65", "65-70", "70-75", "75-80"},
    }
    assert validation.BANDS["alpha"] == (8.0, 13.0)
    assert validation.PERMUTATIONS == 100_000
    assert validation.RANDOM_SEED == 20260804
    assert validation.MIN_DURATION_SECONDS == 400.0


def test_safe_extraction_accepts_only_expected_regular_ec_members(
    tmp_path: Path,
) -> None:
    participant = "sub-032301"
    archive = tmp_path / "record.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for suffix, content in (("set", b"set-data"), ("fdt", b"fdt-data")):
            value = io.BytesIO(content)
            member = tarfile.TarInfo(f"{participant}/{participant}_EC.{suffix}")
            member.size = len(content)
            bundle.addfile(member, value)

    set_path, fdt_path = validation._extract_ec(
        archive, tmp_path / "extracted", participant
    )

    assert set_path.read_bytes() == b"set-data"
    assert fdt_path.read_bytes() == b"fdt-data"


def test_safe_extraction_rejects_link_in_place_of_eeg_data(tmp_path: Path) -> None:
    participant = "sub-032301"
    archive = tmp_path / "record.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        set_member = tarfile.TarInfo(f"{participant}/{participant}_EC.set")
        set_member.size = 3
        bundle.addfile(set_member, io.BytesIO(b"set"))
        link = tarfile.TarInfo(f"{participant}/{participant}_EC.fdt")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        bundle.addfile(link)

    with pytest.raises(RuntimeError, match="unsafe or missing"):
        validation._extract_ec(archive, tmp_path / "extracted", participant)


def test_one_sided_permutation_tracks_the_prespecified_old_lower_direction() -> None:
    old = np.asarray([0.08, 0.09, 0.10, 0.11])
    young = np.asarray([0.31, 0.32, 0.33, 0.34])

    lower_p = validation._permutation_p(old, young, permutations=10_000, seed=9)
    reversed_p = validation._permutation_p(young, old, permutations=10_000, seed=9)

    assert lower_p < 0.05
    assert reversed_p > 0.95
    assert validation._cohens_d(old, young) < 0


def test_tracked_lemon_bundle_matches_manifest_and_reports_failure() -> None:
    root = (
        validation.PROJECT_ROOT
        / "scripts"
        / "validation"
        / "lemon_discover_eeg"
        / "results"
    )
    manifest = json.loads((root / "publication_manifest.json").read_text())

    assert manifest["dataset_doi"] == validation.DATASET_DOI
    for name, expected in manifest["files"].items():
        assert validation._sha256(root / name) == expected
    results = json.loads((root / "validation_results.json").read_text())
    sensitivity = json.loads((root / "published_metric_sensitivity.json").read_text())
    assert results["primary_replication_passed"] is False
    assert results["alpha_peak_frequency"]["one_sided_permutation_p"] < 0.05
    assert sensitivity["directional_alpha_pass"] is False
    attributes = (validation.PROJECT_ROOT / ".gitattributes").read_text()
    assert "lemon_discover_eeg/results/*.json -text" in attributes
    assert "lemon_discover_eeg/results/*.md -text" in attributes
