from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from neurostat.ingest import ingest_project
from neurostat.ingest.service import IngestError


def _project(tmp_path: Path, source: Path, kind: str, **options: str) -> Path:
    extras = ", ".join(f"{key}: {value}" for key, value in options.items())
    project = tmp_path / "project.yaml"
    project.write_text(f"""schema_version: 1
study: {{id: fieldtrip, arms: [active, comparison], sessions: [baseline, followup], conditions: [rest]}}
input: {{state: cleaned}}
records: [{{record_id: ft-1, participant_id: p1, arm: active, session: baseline, condition: rest, data_kind: {kind}, format: fieldtrip_mat, path: {source.name}, analysis_profile: default, data_unit: uV{', ' + extras if extras else ''}}}]
analysis_profiles: {{default: {{}}}}
output: {{root: output}}
""", encoding="utf-8")
    return project


def _classic(path: Path, trials: list[np.ndarray], times: list[np.ndarray] | None = None) -> None:
    data = {"label": np.array(["Cz", "Pz"], dtype=object), "trial": np.array(trials, dtype=object), "fsample": 20.0, "trialinfo": np.arange(len(trials))[:, None], "elec": {"label": np.array(["Cz", "Pz"], dtype=object), "pnt": np.array([[0., 0., 1.], [0., 1., 0.]])}}
    if times is not None: data["time"] = np.array(times, dtype=object)
    savemat(path, {"data_ft": data})


def test_classic_fieldtrip_raw_and_epochs_metadata(tmp_path: Path) -> None:
    raw_file = tmp_path / "raw.mat"; trial = np.arange(12, dtype=float).reshape(2, 6); _classic(raw_file, [trial], [np.arange(6) / 20])
    raw_run = ingest_project(_project(tmp_path, raw_file, "raw"))
    with h5py.File(raw_run.records[0], "r") as file:
        np.testing.assert_allclose(file["data"][0], trial * 1e-6); np.testing.assert_allclose(file["times"][:], np.arange(6) / 20); assert file["channel_positions"].shape == (2, 3)
    epochs_file = tmp_path / "epochs.mat"; trials = [trial, trial + 20]; _classic(epochs_file, trials, [np.arange(6) / 20, np.arange(6) / 20])
    epochs_run = ingest_project(_project(tmp_path, epochs_file, "epochs"))
    with h5py.File(epochs_run.records[0], "r") as file:
        assert file["data"].shape == (2, 2, 6); assert file["events"].shape[0] == 2


def _v73(path: Path, *, concatenated: bool = False, mismatched_time: bool = False) -> None:
    with h5py.File(path, "w") as file:
        group = file.create_group("data_ft")
        values = [np.arange(8, dtype=float).reshape(4, 2), np.arange(8, 16, dtype=float).reshape(4, 2)]
        if concatenated: values = [np.vstack(values)]
        refs = group.create_dataset("trial", (len(values), 1), dtype=h5py.ref_dtype)
        time_refs = None if concatenated else group.create_dataset("time", (len(values), 1), dtype=h5py.ref_dtype)
        for i, value in enumerate(values):
            trial = file.create_dataset(f"trial_{i}", data=value)  # physical MATLAB storage: samples x channels
            refs[i, 0] = trial.ref
            if time_refs is not None:
                time = file.create_dataset(f"time_{i}", data=np.arange(value.shape[0]) / 20 + (0.1 if mismatched_time and i else 0))
                time_refs[i, 0] = time.ref
        labels = []
        for name in ("Cz", "Pz"):
            label = file.create_dataset(f"label_{name}", data=np.asarray([ord(char) for char in name], dtype=np.uint16)); label.attrs["MATLAB_class"] = np.bytes_("char"); labels.append(label.ref)
        cell = group.create_dataset("label", (2, 1), dtype=h5py.ref_dtype); cell[:, 0] = labels
        group.create_dataset("fsample", data=20.0)
        if not concatenated: group.create_dataset("trialinfo", data=np.arange(len(values)).reshape(1, -1))
        elec = group.create_group("elec"); elec.create_dataset("pnt", data=np.array([[0., 0., 1.], [0., 1., 0.]]).T)
        elec_labels = elec.create_dataset("label", (2, 1), dtype=h5py.ref_dtype); elec_labels[:, 0] = labels
        if concatenated: file.create_dataset("pnts", data=4)


def test_v73_multiple_cells_orientation_time_and_positions(tmp_path: Path) -> None:
    source = tmp_path / "standard.mat"; _v73(source)
    run = ingest_project(_project(tmp_path, source, "epochs"))
    with h5py.File(run.records[0], "r") as file:
        assert tuple(file["channel_names"].asstr()[:]) == ("Cz", "Pz")
        np.testing.assert_allclose(file["data"][0], np.array([[0, 2, 4, 6], [1, 3, 5, 7]]) * 1e-6)
        assert file["channel_positions"].shape == (2, 3)


def test_v73_concatenated_pnts_and_time_mismatch_rejection(tmp_path: Path) -> None:
    source = tmp_path / "concatenated.mat"; _v73(source, concatenated=True)
    run = ingest_project(_project(tmp_path, source, "epochs"))
    with h5py.File(run.records[0], "r") as file: assert file["data"].shape == (2, 2, 4)
    bad = tmp_path / "bad.mat"; _v73(bad, mismatched_time=True)
    with pytest.raises(IngestError, match="time vectors"):
        ingest_project(_project(tmp_path, bad, "epochs"))


def test_fieldtrip_square_orientation_electrode_mapping_and_trialinfo(tmp_path: Path) -> None:
    source = tmp_path / "mapped.mat"
    savemat(source, {"data_ft": {"label": np.array(["Cz", "Pz"], dtype=object), "trial": np.eye(2) * 7, "fsample": 10., "trialinfo": np.array([4]), "elec": {"label": np.array(["Pz", "Cz"], dtype=object), "chanpos": np.array([[2., 2., 2.], [1., 1., 1.]])}}})
    run = ingest_project(_project(tmp_path, source, "raw"))
    with h5py.File(run.records[0], "r") as file:
        np.testing.assert_allclose(file["data"][0], np.eye(2) * 7e-6)
        np.testing.assert_allclose(file["channel_positions"][:], [[1, 1, 1], [2, 2, 2]])
        assert file["events"].shape == (1, 1)


@pytest.mark.parametrize("pnts", [0, -1, 2.5])
def test_fieldtrip_invalid_pnts_and_logical_orientation_rejected(tmp_path: Path, pnts: float) -> None:
    source = tmp_path / f"pnts-{pnts}.mat"; _v73(source, concatenated=True)
    with h5py.File(source, "r+") as file: del file["pnts"]; file.create_dataset("pnts", data=pnts)
    with pytest.raises(IngestError, match="pnts"):
        ingest_project(_project(tmp_path, source, "epochs"))
    wrong = tmp_path / "wrong.mat"; savemat(wrong, {"data_ft": {"label": np.array(["Cz", "Pz"], dtype=object), "trial": np.arange(6.).reshape(3, 2), "fsample": 10.}})
    with pytest.raises(IngestError, match="axis 0"):
        ingest_project(_project(tmp_path, wrong, "raw"))
