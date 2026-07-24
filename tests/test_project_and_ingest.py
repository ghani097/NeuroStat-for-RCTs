from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
import pytest
import h5py

from neurostat.ingest import ingest_project
from neurostat.project import ProjectValidationError, load_project


def _project(tmp_path: Path, source: Path) -> Path:
    project = tmp_path / "project.yaml"
    project.write_text(f"""schema_version: 1
study: {{id: demo, arms: [active, comparison], sessions: [baseline, followup], conditions: [rest]}}
input: {{state: cleaned}}
records:
  - {{record_id: demo-1, participant_id: p1, arm: active, session: baseline, condition: rest, data_kind: raw, format: fif, path: {source.name}, analysis_profile: default, data_unit: V}}
analysis_profiles: {{default: {{band: alpha}}}}
output: {{root: output}}
""", encoding="utf-8")
    return project


def _fif(path: Path) -> None:
    raw = mne.io.RawArray(np.arange(24, dtype=float).reshape(2, 12), mne.create_info(["Cz", "Pz"], 10.0, "eeg"), verbose="ERROR")
    raw.save(path, overwrite=True, verbose="ERROR")


def test_fif_ingestion_is_deterministic_and_resumable(tmp_path: Path) -> None:
    source = tmp_path / "clean_raw.fif"; _fif(source); project = _project(tmp_path, source)
    loaded = load_project(project)
    assert loaded.analysis_profiles["default"]["band"] == "alpha"
    with pytest.raises(TypeError): loaded.analysis_profiles["default"]["band"] = "beta"  # type: ignore[index]
    first = ingest_project(loaded); second = ingest_project(loaded)
    assert first.run_id == second.run_id and second.reused == 1
    manifest = json.loads(second.manifest.read_text(encoding="utf-8"))
    assert manifest["records"][0]["dimensions"] == "[1,2,12]"


def test_duplicate_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"; path.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(ProjectValidationError, match="duplicate YAML"):
        load_project(path)


def test_invalid_cleaning_state_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "clean_raw.fif"; _fif(source); project = _project(tmp_path, source)
    project.write_text(project.read_text().replace("state: cleaned", "state: raw"), encoding="utf-8")
    with pytest.raises(ProjectValidationError, match="never cleans"):
        load_project(project)


def test_v73_fieldtrip_concatenated_trial_is_split(tmp_path: Path) -> None:
    source = tmp_path / "fieldtrip.mat"
    with h5py.File(source, "w") as file:
        group = file.create_group("data_ft")
        trial = file.create_dataset("trial_data", data=np.arange(16, dtype=float).reshape(2, 8))
        trial_refs = group.create_dataset("trial", (1, 1), dtype=h5py.ref_dtype); trial_refs[0, 0] = trial.ref
        labels = []
        for text in ("Cz", "Pz"):
            label = file.create_dataset(f"label_{text}", data=np.asarray([ord(x) for x in text], dtype=np.uint16)); label.attrs["MATLAB_class"] = np.bytes_("char"); labels.append(label.ref)
        label_refs = group.create_dataset("label", (2, 1), dtype=h5py.ref_dtype); label_refs[:, 0] = labels
        group.create_dataset("fsample", data=10.0); file.create_dataset("pnts", data=4)
    project = tmp_path / "project.yaml"
    project.write_text("""schema_version: 1
study: {id: demo, arms: [active, comparison], sessions: [baseline, followup], conditions: [rest]}
input: {state: cleaned}
records: [{record_id: ft-1, participant_id: p1, arm: active, session: baseline, condition: rest, data_kind: epochs, format: fieldtrip_mat, path: fieldtrip.mat, analysis_profile: default, data_unit: uV}]
analysis_profiles: {default: {}}
output: {root: output}
""", encoding="utf-8")
    run = ingest_project(project)
    with h5py.File(run.records[0], "r") as file:
        assert file["data"].shape == (2, 2, 4)
        np.testing.assert_allclose(file["data"][0], np.asarray([[0, 1, 2, 3], [8, 9, 10, 11]]) * 1e-6)
