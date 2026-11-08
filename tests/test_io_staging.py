"""Staging for v1 preprocessing and v2 ingest of the new input formats."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import mne
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from io_writers import write_edf, write_poly5, write_wibci_csv  # noqa: E402

from neurostat.gui.discovery import discover_records  # noqa: E402
from neurostat.gui.legacy_bridge import stage_raw_input  # noqa: E402
from neurostat.gui.study_builder import StudyForm, build_project_document, recommended_profile_pair  # noqa: E402
from neurostat.ingest import ingest_project  # noqa: E402
from neurostat.io.staging import needs_staging, stage_as_set  # noqa: E402
from neurostat.project import ProjectValidationError, load_project  # noqa: E402

LABELS = ["Fp1", "Fz", "Cz", "Pz", "O1", "O2"]
SFREQ = 128


def _data(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 20.0, (len(LABELS), 4 * SFREQ))


def _bdf_study(root: Path) -> Path:
    """<arm>/<subject>/<session>/rest.bdf: same file name for every subject."""
    for a, arm in enumerate(("Active", "Control")):
        for s in (1, 2):
            for t, session in enumerate(("pre", "post")):
                folder = root / arm / f"S{s}" / session
                folder.mkdir(parents=True)
                write_edf(folder / "rest.bdf", _data(a * 10 + s * 2 + t), LABELS, SFREQ, bdf=True, status=np.zeros(4 * SFREQ, dtype=int))
    return root


def _form(base: Path, output: Path, **extra) -> StudyForm:
    return StudyForm("multi", base, ("Active", "Control"), ("pre", "post"), ("rest",), output, already_cleaned=False, **extra)


def test_flat_set_study_is_not_staged(tmp_path):
    folder = tmp_path / "study" / "Active" / "pre"
    folder.mkdir(parents=True)
    raw = mne.io.RawArray(_data(1) * 1e-6, mne.create_info(LABELS, SFREQ, "eeg"), verbose="ERROR")
    raw.export(folder / "S1.set", fmt="eeglab", verbose="ERROR")
    records = discover_records(tmp_path / "study", ["Active", "Control"], ["pre", "post"], ["rest"])
    assert records and not needs_staging(records, tmp_path / "study")
    assert stage_raw_input(_form(tmp_path / "study", tmp_path / "out")) == tmp_path / "study"


def test_bdf_subject_layout_is_staged_flat_with_provenance(tmp_path):
    base = _bdf_study(tmp_path / "study")
    form = _form(base, tmp_path / "out")
    root = stage_raw_input(form)
    assert root == form.resolved_staging_root
    staged = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.set"))
    assert staged == [f"{arm}/{session}/s{s}_rest.set" for arm in ("Active", "Control") for session in ("post", "pre") for s in (1, 2)]
    first = root / "Active" / "pre" / "s1_rest.set"
    raw = mne.io.read_raw_eeglab(first, verbose="ERROR")
    original = mne.io.read_raw_bdf(base / "Active" / "S1" / "pre" / "rest.bdf", verbose="ERROR").pick(LABELS)
    np.testing.assert_allclose(raw.get_data(), original.get_data(), atol=1e-9)
    provenance = json.loads(first.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert provenance["format"] == "bdf" and len(provenance["source_sha256"]) == 64
    assert provenance["summary"]["channels"] == len(LABELS)
    messages: list[str] = []
    stage_raw_input(form, messages.append)
    assert sum("is current" in message for message in messages) == 8


def test_staging_refuses_collisions(tmp_path):
    base = tmp_path / "study" / "Active" / "pre"
    base.mkdir(parents=True)
    write_edf(base / "S1.edf", _data(1), LABELS, SFREQ)
    write_edf(base / "S1.bdf", _data(2), LABELS, SFREQ, bdf=True)
    records = discover_records(tmp_path / "study", ["Active", "Control"], ["pre", "post"], ["rest"])
    with pytest.raises(Exception, match="slug collision|would both stage"):
        stage_as_set(records, tmp_path / "staged")


def test_cleaned_bdf_and_poly5_ingest_with_positions(tmp_path):
    base = tmp_path / "study"
    for arm, session, maker in (("Active", "pre", "bdf"), ("Active", "post", "bdf"), ("Control", "pre", "poly5"), ("Control", "post", "poly5")):
        folder = base / arm / session
        folder.mkdir(parents=True)
        if maker == "bdf":
            write_edf(folder / "P1.bdf", _data(3), LABELS, SFREQ, bdf=True, status=np.zeros(4 * SFREQ, dtype=int))
        else:
            names = [f"ExG{i}" for i in range(1, 65)]
            write_poly5(folder / "P2.Poly5", np.random.default_rng(4).normal(0, 20, (64, 4 * SFREQ)).astype(np.float32), names, ["uV"] * 64, SFREQ)
    form = StudyForm("multi", base, ("Active", "Control"), ("pre", "post"), ("rest",), tmp_path / "out", already_cleaned=True)
    records = discover_records(base, form.arms, form.sessions, form.conditions)
    assert {r.format for r in records} == {"bdf", "poly5"}
    assert recommended_profile_pair(records) is not None
    document = build_project_document(form, records)
    document["analysis_profiles"] = {"default": {}}  # ingest does not need the source profile
    form.resolved_project_file.parent.mkdir(parents=True)
    form.resolved_project_file.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    project = load_project(form.resolved_project_file)
    run = ingest_project(project)
    shapes = {}
    for path in run.records:
        with h5py.File(path, "r") as file:
            shapes[file.attrs["format"]] = file["data"].shape
            assert "channel_positions" in file
    assert shapes == {"bdf": (1, len(LABELS), 4 * SFREQ), "poly5": (1, 62, 4 * SFREQ)}
    assert ingest_project(project).reused == 4


def _project(tmp_path: Path, record: str) -> Path:
    path = tmp_path / "p.yaml"
    path.write_text(f"""schema_version: 1
study: {{id: s, arms: [a, b], sessions: [pre, post], conditions: [rest]}}
input: {{state: cleaned}}
records: [{{record_id: r1, participant_id: p1, arm: a, session: pre, condition: rest, analysis_profile: default, {record}}}]
analysis_profiles: {{default: {{}}}}
output: {{root: out}}
""", encoding="utf-8")
    return path


def test_project_validation_of_reader_options(tmp_path):
    write_edf(tmp_path / "r.edf", _data(1), LABELS, SFREQ)
    write_wibci_csv(tmp_path / "w.csv", _data(1)[:3], LABELS[:3])
    project = load_project(_project(tmp_path, "data_kind: raw, format: wibci_csv, path: w.csv, sampling_rate: 500, channel_map: none"))
    assert project.records[0].reader_options()["sampling_rate"] == 500.0
    assert project.normalized()["records"][0]["sampling_rate"] == 500.0
    bad = [
        ("data_kind: epochs, format: edf, path: r.edf", "supports data_kind raw"),
        ("data_kind: raw, format: edf, path: r.edf, stream: EEG", "only valid for XDF"),
        ("data_kind: raw, format: edf, path: r.edf, sampling_rate: 100", "only valid for WiBCI"),
        ("data_kind: raw, format: edf, path: r.edf, data_unit: uV", "data_unit must be"),
        ("data_kind: raw, format: edf, path: r.edf, channel_map: missing.tsv", "channel_map must be"),
        ("data_kind: raw, format: wibci_csv, path: r.edf", "extension"),
    ]
    for record, message in bad:
        with pytest.raises(ProjectValidationError, match=message):
            load_project(_project(tmp_path, record))


def test_existing_projects_keep_their_normalized_shape(tmp_path):
    write_edf(tmp_path / "r.edf", _data(1), LABELS, SFREQ)
    normalized = load_project(_project(tmp_path, "data_kind: raw, format: edf, path: r.edf")).normalized()
    assert not {"channel_map", "stream", "sampling_rate"} & set(normalized["records"][0])
