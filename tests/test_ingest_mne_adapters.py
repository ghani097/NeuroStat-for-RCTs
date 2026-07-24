from __future__ import annotations

from pathlib import Path

import h5py
import mne
import numpy as np
import pytest

from neurostat.ingest import ingest_project


def _write_project(tmp_path: Path, source: Path, kind: str, fmt: str) -> Path:
    project = tmp_path / "project.yaml"
    project.write_text(f"""schema_version: 1
study: {{id: adapters, arms: [active, comparison], sessions: [baseline, followup], conditions: [rest]}}
input: {{state: cleaned}}
records: [{{record_id: source-1, participant_id: p1, arm: active, session: baseline, condition: rest, data_kind: {kind}, format: {fmt}, path: {source.name}, analysis_profile: default, data_unit: V}}]
analysis_profiles: {{default: {{}}}}
output: {{root: output}}
""", encoding="utf-8")
    return project


def _raw() -> mne.io.BaseRaw:
    return mne.io.RawArray(np.arange(24, dtype=float).reshape(2, 12) * 1e-6, mne.create_info(["Cz", "Pz"], 20.0, "eeg"), verbose="ERROR")


def _epochs() -> mne.BaseEpochs:
    return mne.EpochsArray(np.arange(48, dtype=float).reshape(2, 2, 12) * 1e-6, mne.create_info(["Cz", "Pz"], 20.0, "eeg"), events=np.array([[0, 0, 1], [12, 0, 1]]), tmin=-0.1, verbose="ERROR")


@pytest.mark.parametrize(("kind", "fmt"), [("raw", "fif"), ("epochs", "fif"), ("raw", "set"), ("epochs", "set")])
def test_real_mne_round_trips(tmp_path: Path, kind: str, fmt: str) -> None:
    source = tmp_path / f"source.{fmt}"
    object = _raw() if kind == "raw" else _epochs()
    if fmt == "fif": object.save(source, overwrite=True, verbose="ERROR")
    elif kind == "raw": mne.export.export_raw(source, object, fmt="eeglab", overwrite=True)
    else: mne.export.export_epochs(source, object, fmt="eeglab", overwrite=True)
    run = ingest_project(_write_project(tmp_path, source, kind, fmt))
    with h5py.File(run.records[0], "r") as canonical:
        actual = canonical["data"][:]
        expected = object.get_data()
        if kind == "raw": expected = expected[None, :, :]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
        assert tuple(canonical["channel_names"].asstr()[:]) == ("Cz", "Pz")
        assert canonical.attrs["cleaning"] == "externally_cleaned_bypass"


def test_mne_adapter_never_calls_cleaning_methods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.fif"; _raw().save(source, overwrite=True, verbose="ERROR")
    def forbidden(*args, **kwargs): raise AssertionError("cleaning method invoked")
    monkeypatch.setattr(mne.io.BaseRaw, "filter", forbidden)
    monkeypatch.setattr(mne.io.BaseRaw, "resample", forbidden)
    monkeypatch.setattr(mne.io.BaseRaw, "set_eeg_reference", forbidden)
    ingest_project(_write_project(tmp_path, source, "raw", "fif"))
