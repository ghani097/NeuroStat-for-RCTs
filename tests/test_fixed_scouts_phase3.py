from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import mne
import numpy as np
import pytest
from scipy.io import loadmat, savemat

from neurostat.ingest import ingest_project
from neurostat.connectivity import ConnectivityError, run_connectivity
from neurostat.contracts import StageName
from neurostat.scouts import ScoutError, run_scouts
from neurostat.scouts.fixed import load_operator
from neurostat.analysis_profiles import parse_analysis_profile
from neurostat.project import load_project


def _epochs(path: Path, data: np.ndarray, sfreq: float = 128.0) -> None:
    info = mne.create_info(["A", "B"], sfreq, "eeg")
    events = np.column_stack((np.arange(len(data)), np.zeros(len(data), int), np.ones(len(data), int)))
    epochs = mne.EpochsArray(data, info, events=events, tmin=0.0, verbose="ERROR")
    epochs.save(path, overwrite=True, verbose="ERROR")


def _project(tmp_path: Path, *, export: bool = False, epoch_chunk_size: int = 2) -> Path:
    path = tmp_path / "project.yaml"
    path.write_text(
        f"""schema_version: 1
study: {{id: fixed-demo, arms: [active, comparison], sessions: [baseline, followup], conditions: [sep]}}
input: {{state: cleaned}}
records:
  - {{record_id: sep-1, participant_id: p1, arm: active, session: baseline, condition: sep, data_kind: epochs, format: fif, path: clean-epo.fif, analysis_profile: sep}}
analysis_profiles:
  sep:
    source:
      backend: fixed_linear_operator
      operator: operator.mat
      sensor_labels: [B, A]
      scout_labels: [roi-one, roi-two]
      epoch_chunk_size: {epoch_chunk_size}
      output_chunk_samples: 37
      export_matlab: {str(export).lower()}
    connectivity:
      backend: evoked_average_hilbert_pli
      filter_order: 4
      bands: {{alpha: [7.5, 12.5]}}
output: {{root: output}}
""",
        encoding="utf-8",
    )
    return path


def _setup(tmp_path: Path, *, export: bool = False) -> tuple[Path, np.ndarray, np.ndarray]:
    samples = 257
    base = np.arange(5 * 2 * samples, dtype=np.float64).reshape(5, 2, samples) / 1000.0
    _epochs(tmp_path / "clean-epo.fif", base)
    operator = np.asarray([[2.0, -1.0], [0.5, 3.0]], dtype=np.float64)
    savemat(tmp_path / "operator.mat", {"SensorToScoutOperator": operator})
    return _project(tmp_path, export=export), base, operator


def test_fixed_projection_streams_epoch_blocks_and_reorders_nonmonotone_labels(tmp_path: Path) -> None:
    project_file, input_data, operator = _setup(tmp_path)
    ingest = ingest_project(project_file)
    run = run_scouts(project_file, ingest)
    with h5py.File(ingest.records[0], "r") as canonical:
        expected_sensor_average = canonical["data"][:].mean(axis=0, dtype=np.float64)[[1, 0], :]
    expected = operator @ expected_sensor_average
    with h5py.File(run.records[0], "r") as file:
        np.testing.assert_allclose(file["data"][:], expected, rtol=0.0, atol=1e-15)
        assert file["data"].shape == (2, 257)
        assert file["data"].chunks == (2, 37)
        assert list(file["labels"].asstr()[:]) == ["roi-one", "roi-two"]
        provenance = json.loads(file.attrs["source_provenance"])
        assert provenance["epoch_chunk_size"] == 2
        assert provenance["retained_epoch_count"] == 5
        assert provenance["hidden_preprocessing"] is False


def test_fixed_run_is_resumable_and_corruption_is_refused(tmp_path: Path) -> None:
    project_file, _, _ = _setup(tmp_path)
    ingest = ingest_project(project_file)
    first = run_scouts(project_file, ingest)
    second = run_scouts(project_file, ingest)
    assert first.run_id == second.run_id and second.reused == 1 and second.written == 0
    with h5py.File(first.records[0], "r+") as file:
        file["data"][0, 0] += 1
    with pytest.raises(ScoutError, match="payload digest"):
        run_scouts(project_file, ingest)


def test_operator_content_changes_run_identity(tmp_path: Path) -> None:
    project_file, _, operator = _setup(tmp_path)
    ingest = ingest_project(project_file)
    first = run_scouts(project_file, ingest)
    savemat(tmp_path / "operator.mat", {"SensorToScoutOperator": operator * 2})
    second = run_scouts(project_file, ingest)
    assert first.run_id != second.run_id


def test_optional_matlab_export_is_exact_and_resumable(tmp_path: Path) -> None:
    project_file, _, _ = _setup(tmp_path, export=True)
    ingest = ingest_project(project_file)
    first = run_scouts(project_file, ingest)
    manifest = json.loads(first.manifest.read_text(encoding="utf-8"))
    export = Path(manifest["records"][0]["matlab"]["path"])
    values = loadmat(export, squeeze_me=True)
    with h5py.File(first.records[0], "r") as file:
        np.testing.assert_array_equal(values["Value"], file["data"][:])
        np.testing.assert_array_equal(np.asarray(values["Time"]).ravel(), file["times"][:])
    second = run_scouts(project_file, ingest)
    assert second.reused == 1


def test_operator_metadata_disagreement_and_wrong_orientation_are_rejected(tmp_path: Path) -> None:
    project_file, _, operator = _setup(tmp_path)
    project = load_project(project_file)
    config = parse_analysis_profile(project, "sep").source
    savemat(
        tmp_path / "operator.mat",
        {
            "SensorToScoutOperator": operator,
            "SensorLabels": np.asarray(["A", "B"], dtype=object),
        },
    )
    changed = parse_analysis_profile(load_project(project_file), "sep").source
    with pytest.raises(ScoutError, match="metadata disagrees"):
        load_operator(changed)
    # A non-square 3-scout x 2-sensor contract must not accept a classic
    # MAT matrix saved in the transposed 2x3 orientation.
    text = project_file.read_text(encoding="utf-8").replace(
        "scout_labels: [roi-one, roi-two]",
        "scout_labels: [roi-one, roi-two, roi-three]",
    )
    project_file.write_text(text, encoding="utf-8")
    savemat(tmp_path / "operator.mat", {"SensorToScoutOperator": np.ones((2, 3))})
    changed = parse_analysis_profile(load_project(project_file), "sep").source
    with pytest.raises(ScoutError, match="operator shape"):
        load_operator(changed)


def test_v73_operator_axes_are_undone_only_for_matlab_hdf5(tmp_path: Path) -> None:
    project_file, _, operator = _setup(tmp_path)
    with h5py.File(tmp_path / "operator.mat", "w") as file:
        dataset = file.create_dataset("SensorToScoutOperator", data=operator.T)
        dataset.attrs["MATLAB_class"] = np.bytes_("double")
    config = parse_analysis_profile(load_project(project_file), "sep").source
    np.testing.assert_array_equal(load_operator(config), operator)


def test_fixed_ingest_scout_connectivity_pipeline_is_real_resumable_and_integrity_checked(tmp_path: Path) -> None:
    project_file, _, _ = _setup(tmp_path)
    events = []
    ingest = ingest_project(project_file)
    scouts = run_scouts(project_file, ingest)
    first = run_connectivity(project_file, scouts, events.append)
    assert events[-1].stage is StageName.CONNECTIVITY
    with h5py.File(first.records[0], "r") as file:
        assert file["matrix"].shape == (1, 2, 2)
        assert file["sos"].shape == (1, 4, 6)
        assert file["matrix"][0, 0, 0] == file["matrix"][0, 1, 1] == 0.0
        assert np.array_equal(file["matrix"][0], file["matrix"][0].T)
    second = run_connectivity(project_file, scouts)
    assert second.run_id == first.run_id and second.reused == 1
    with h5py.File(first.records[0], "r+") as file:
        file["matrix"][0, 1, 0] += 0.01
    with pytest.raises(ConnectivityError, match="symmetric|payload digest"):
        run_connectivity(project_file, scouts)


def test_cli_fixed_pipeline_json_and_invalid_profile_error(tmp_path: Path) -> None:
    project_file, _, _ = _setup(tmp_path)
    root = Path(__file__).parents[1]
    env = os.environ | {"PYTHONPATH": str(root / "src")}

    def command(*args: str, check: bool = True):
        return subprocess.run(
            [sys.executable, "-m", "neurostat", *args],
            text=True,
            capture_output=True,
            env=env,
            check=check,
        )

    ingest = json.loads(command("ingest", str(project_file), "--json").stdout)
    scouts = json.loads(command("scouts", str(project_file), "--ingest-run", ingest["run_dir"], "--json").stdout)
    connectivity = json.loads(
        command("connectivity", str(project_file), "--scouts-run", scouts["run_dir"], "--json").stdout
    )
    assert ingest["records"] == scouts["records"] == connectivity["records"] == 1
    missing = command(
        "scouts",
        str(project_file),
        "--ingest-run",
        str(tmp_path / "missing-run"),
        check=False,
    )
    wrong_stage = command(
        "connectivity",
        str(project_file),
        "--scouts-run",
        ingest["run_dir"],
        check=False,
    )
    assert missing.returncode == 2 and "missing" in missing.stderr
    assert wrong_stage.returncode == 2 and "missing" in wrong_stage.stderr
    project_file.write_text(
        project_file.read_text(encoding="utf-8").replace("filter_order: 4", "filter_order: true"),
        encoding="utf-8",
    )
    failed = command("scouts", str(project_file), "--ingest-run", ingest["run_dir"], check=False)
    assert failed.returncode == 2 and failed.stderr.startswith("error:")
