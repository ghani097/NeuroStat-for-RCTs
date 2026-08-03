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
from scipy.io import savemat

from neurostat.contracts import StageName
from neurostat.ingest import ingest_project
from neurostat.ingest.service import CleanRecording, IngestError, SourceBundle, source_bundle
from neurostat.contracts import RecordKey
from neurostat.project import load_project


def _source(path: Path, samples: int = 12) -> None:
    mne.io.RawArray(np.arange(2 * samples, dtype=float).reshape(2, samples), mne.create_info(["Cz", "Pz"], 10., "eeg"), verbose="ERROR").save(path, overwrite=True, verbose="ERROR")


def _project(tmp_path: Path, source: Path) -> Path:
    path = tmp_path / "project.yaml"
    path.write_text(f"""schema_version: 1
study: {{id: integrity, arms: [active, comparison], sessions: [baseline, followup], conditions: [rest]}}
input: {{state: cleaned}}
records: [{{record_id: item-1, participant_id: p1, arm: active, session: baseline, condition: rest, data_kind: raw, format: fif, path: {source.name}, analysis_profile: default, data_unit: V}}]
analysis_profiles: {{default: {{}}}}
output: {{root: output}}
""", encoding="utf-8")
    return path


def test_determinism_progress_payload_and_partial_reuse(tmp_path: Path) -> None:
    source = tmp_path / "source.fif"; _source(source); project = _project(tmp_path, source); events = []
    first = ingest_project(project, events.append); second = ingest_project(project)
    assert first.run_id == second.run_id and second.reused == 1 and events[-1].stage is StageName.INGEST
    manifest = json.loads(first.manifest.read_text()); assert manifest["records"][0]["dimensions"] == [1, 2, 12]
    first.manifest.unlink()  # Simulate an interrupted run after its atomically complete artifact.
    third = ingest_project(project); assert third.reused == 1
    with h5py.File(third.records[0], "r+") as file: file["data"][0, 0, 0] += 1
    third.manifest.unlink()
    with pytest.raises(IngestError, match="payload digest"):
        ingest_project(project)
    _source(source, 14)
    changed = ingest_project(project); assert changed.run_id != first.run_id


def test_lock_mismatch_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "source.fif"; _source(source); project = _project(tmp_path, source); run = ingest_project(project)
    lock = run.run_dir / "project.lock.json"; lock.write_text("{}", encoding="utf-8")
    with pytest.raises(IngestError, match="existing lock differs"): ingest_project(project)


def test_manifest_hash_and_atomic_failure_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.fif"; _source(source); project = _project(tmp_path, source); run = ingest_project(project)
    with h5py.File(run.records[0], "r+") as file: file["data"][0, 0, 0] += 1
    with pytest.raises(IngestError, match="payload digest|SHA-256"):
        ingest_project(project)
    other = tmp_path / "other"; other.mkdir(); other_source = other / "source.fif"; _source(other_source); other_project = _project(other, other_source)
    import neurostat.ingest.service as service
    monkeypatch.setattr(service, "_verify", lambda *args: (_ for _ in ()).throw(IngestError("injected")))
    with pytest.raises(IngestError, match="injected"): ingest_project(other_project)
    assert not list((other / "output").rglob("*.tmp")) and not list((other / "output").rglob("*.h5"))
    assert json.loads(next((other / "output").rglob("status.json")).read_text())["state"] == "failed"


def test_complete_manifest_rejects_nonpayload_file_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.fif"; _source(source); project = _project(tmp_path, source); run = ingest_project(project)
    with h5py.File(run.records[0], "r+") as file: file.attrs["untracked_note"] = "changed bytes but valid payload"
    with pytest.raises(IngestError, match="SHA-256"):
        ingest_project(project)


def test_set_companion_provenance_uses_actual_header_reference(tmp_path: Path) -> None:
    external = tmp_path / "actual.fdt"; external.write_bytes(b"first")
    stray = tmp_path / "embedded.fdt"; stray.write_bytes(b"stray")
    external_set = tmp_path / "external.set"; savemat(external_set, {"EEG": {"data": "actual.fdt"}})
    embedded_set = tmp_path / "embedded.set"; savemat(embedded_set, {"EEG": {"data": np.zeros((2, 2))}})
    def project_for(path: Path) -> Path:
        text = _project(tmp_path, tmp_path / "source.fif").read_text().replace("format: fif", "format: set").replace("source.fif", path.name).replace("data_unit: V", "data_unit: V")
        out = tmp_path / (path.stem + ".yaml"); out.write_text(text, encoding="utf-8"); return out
    external_bundle = source_bundle(load_project(project_for(external_set)).records[0]); assert external.resolve() in external_bundle.files
    before = external_bundle.digest; external.write_bytes(b"second"); assert source_bundle(load_project(project_for(external_set)).records[0]).digest != before
    assert stray.resolve() not in source_bundle(load_project(project_for(embedded_set)).records[0]).files
    missing = tmp_path / "missing.set"; savemat(missing, {"EEG": {"data": "gone.fdt"}})
    with pytest.raises(IngestError, match="missing companion"):
        source_bundle(load_project(project_for(missing)).records[0])


def test_v73_set_companion_reference_is_hashed(tmp_path: Path) -> None:
    companion = tmp_path / "nonstem-data.fdt"; companion.write_bytes(b"one")
    header = tmp_path / "header.set"
    with h5py.File(header, "w") as file:
        eeg = file.create_group("EEG")
        text = file.create_dataset("external_name", data=np.asarray([ord(char) for char in companion.name], dtype=np.uint16)); text.attrs["MATLAB_class"] = np.bytes_("char")
        ref = eeg.create_dataset("data", (1, 1), dtype=h5py.ref_dtype); ref[0, 0] = text.ref
    source = tmp_path / "source.fif"; _source(source)
    text = _project(tmp_path, source).read_text().replace("format: fif", "format: set").replace("source.fif", "header.set")
    project = tmp_path / "v73-set.yaml"; project.write_text(text, encoding="utf-8")
    before = source_bundle(load_project(project).records[0]); assert companion.resolve() in before.files
    companion.write_bytes(b"two")
    assert source_bundle(load_project(project).records[0]).digest != before.digest


def test_cli_validate_ingest_json_and_error(tmp_path: Path) -> None:
    source = tmp_path / "source.fif"; _source(source); project = _project(tmp_path, source)
    env = os.environ | {"PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    valid = subprocess.run([sys.executable, "-m", "neurostat", "validate", str(project), "--json"], text=True, capture_output=True, env=env, check=True)
    assert json.loads(valid.stdout)["valid"] is True
    text_mode = subprocess.run([sys.executable, "-m", "neurostat", "validate", str(project)], text=True, capture_output=True, env=env, check=True)
    assert text_mode.stdout == "valid\n"
    light = subprocess.run([sys.executable, "-c", "from neurostat.cli import main; import sys; main([\"validate\", sys.argv[1]]); print(' '.join(sys.modules))", str(project)], text=True, capture_output=True, env=env, check=True)
    assert not any(name in light.stdout.split() or any(item.startswith(name + ".") for item in light.stdout.split()) for name in ("mne", "matplotlib", "PyQt5", "pli_pipeline"))
    ingested = subprocess.run([sys.executable, "-m", "neurostat", "ingest", str(project), "--json"], text=True, capture_output=True, env=env, check=True)
    assert json.loads(ingested.stdout)["records"] == 1
    bad = subprocess.run([sys.executable, "-m", "neurostat", "validate", str(tmp_path / "missing.yaml")], text=True, capture_output=True, env=env)
    assert bad.returncode == 2 and bad.stderr.startswith("error:")


def test_clean_recording_owns_readonly_arrays() -> None:
    data = np.zeros((1, 2, 3)); times = np.arange(3.); events = np.array([[1]]); positions = np.zeros((2, 3))
    record = CleanRecording("r", RecordKey("p", "a", "s", "c"), "raw", "fif", "profile", 1., ["Cz", "Pz"], data, times, events, positions, "V", "V", 1., SourceBundle((), (), "digest"))
    data[0, 0, 0] = 9; events[0, 0] = 9
    assert record.channel_names == ("Cz", "Pz") and record.data[0, 0, 0] == 0 and record.events[0, 0] == 1
    with pytest.raises(ValueError): record.times[0] = 3


def test_long_signal_is_chunked_without_value_changes(tmp_path: Path) -> None:
    source = tmp_path / "source.fif"; _source(source, 25001)
    run = ingest_project(_project(tmp_path, source))
    with h5py.File(run.records[0], "r") as file:
        assert file["data"].chunks is not None and 0 < file["data"].chunks[-1] <= 25001
        np.testing.assert_array_equal(file["data"][0, :, -3:], np.array([[24998, 24999, 25000], [49999, 50000, 50001]], dtype=float))
