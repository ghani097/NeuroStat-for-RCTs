from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import h5py
import mne
import numpy as np
import pytest
from scipy.io import savemat

from neurostat.atlas import DK68_LABELS
from neurostat.connectivity import run_connectivity
from neurostat.contracts import StageName
from neurostat.ingest import ingest_project
from neurostat.scouts import ScoutStream, run_scouts


def _raw(path: Path, samples: int = 135, sfreq: float = 64.0) -> None:
    data = np.vstack(
        [
            np.sin(2 * np.pi * 10 * np.arange(samples) / sfreq),
            np.sin(2 * np.pi * 10 * np.arange(samples) / sfreq + np.pi / 5),
        ]
    )
    raw = mne.io.RawArray(data, mne.create_info(["A", "B"], sfreq, "eeg"), verbose="ERROR")
    raw.set_montage(
        mne.channels.make_dig_montage(
            ch_pos={"A": np.asarray([0.0, 0.03, 0.08]), "B": np.asarray([0.02, -0.02, 0.075])},
            coord_frame="head",
        )
    )
    raw.save(path, overwrite=True, verbose="ERROR")


def _resources(tmp_path: Path) -> None:
    subjects = tmp_path / "subjects" / "fsaverage" / "label"
    subjects.mkdir(parents=True)
    for path in (
        tmp_path / "trans.fif",
        tmp_path / "source-src.fif",
        tmp_path / "bem-sol.fif",
        subjects / "lh.aparc.annot",
        subjects / "rh.aparc.annot",
    ):
        path.write_bytes(("resource:" + path.name).encode())


def _project(tmp_path: Path) -> Path:
    _raw(tmp_path / "clean-raw.fif")
    _resources(tmp_path)
    path = tmp_path / "project.yaml"
    path.write_text(
        """schema_version: 1
study: {id: mne-demo, arms: [active, comparison], sessions: [baseline, followup], conditions: [rest]}
input: {state: cleaned}
records:
  - {record_id: rest-1, participant_id: p1, arm: active, session: baseline, condition: rest, data_kind: raw, format: fif, path: clean-raw.fif, analysis_profile: rest}
analysis_profiles:
  rest:
    source:
      backend: mne_fsaverage
      subject: fsaverage
      subjects_dir: subjects
      trans: trans.fif
      source_space: source-src.fif
      bem: bem-sol.fif
      atlas: aparc
      inverse_method: MNE
      snr: 3.0
      pick_orientation: normal
      label_mode: mean_flip
      chunk_samples: 40
      output_chunk_samples: 29
      depth: 0.8
      loose: 0.2
      mindist_mm: 5.0
      noise_covariance: empirical
      average_reference_projection: true
      coordinate_frame: head
      export_matlab: false
    connectivity:
      backend: continuous_multitaper_pli
      bands: {alpha: [8, 12]}
      window_seconds: 1.0
      tail_policy: drop
      mt_bandwidth: 4
      mt_adaptive: false
      mt_low_bias: true
      n_jobs: 1
output: {root: output}
""",
        encoding="utf-8",
    )
    return path


class FakeMNEProjector:
    def __init__(self) -> None:
        self.config = None
        self.calls = 0

    def project(self, canonical_path: Path, config) -> ScoutStream:
        self.config = config
        self.calls += 1
        with h5py.File(canonical_path, "r") as file:
            sfreq = float(file.attrs["sfreq"])
            times = file["times"][:]
        samples = len(times)
        base = np.arange(samples, dtype=np.float64)
        values = np.vstack([np.sin(base / 7 + index / 20) for index in range(68)])

        def chunks():
            for start in range(0, samples, config.chunk_samples):
                yield start, values[:, start : start + config.chunk_samples]

        return ScoutStream(
            DK68_LABELS,
            times,
            sfreq,
            chunks(),
            {
                "backend": "mne_fsaverage",
                "label_order": list(DK68_LABELS),
                "source_chunk_samples": config.chunk_samples,
                "test_seam": True,
            },
        )


class DenseResult:
    def __init__(self, nodes: int) -> None:
        self.nodes = nodes

    def get_data(self, output: str) -> np.ndarray:
        value = np.zeros((self.nodes, self.nodes, 1), dtype=np.float64)
        value[1, 0, 0] = 0.25
        return value


def test_mne_orchestration_seam_streams_dk68_and_continuous_windows(tmp_path: Path) -> None:
    project = _project(tmp_path)
    ingest = ingest_project(project)
    projector = FakeMNEProjector()
    scout_events = []
    scouts = run_scouts(project, ingest, scout_events.append, mne_projector=projector)
    assert projector.calls == 1
    assert projector.config.chunk_samples == 40
    assert scout_events[-1].stage is StageName.SCOUTS
    with h5py.File(scouts.records[0], "r") as file:
        assert tuple(file["labels"].asstr()[:]) == DK68_LABELS
        assert file["data"].shape == (68, 135)
        assert file["data"].chunks == (68, 29)

    observed: dict[str, object] = {}

    def fake_connectivity(windows, **kwargs):
        observed["iterator"] = iter(windows) is windows
        observed["shapes"] = [window.shape for window in windows]
        observed["kwargs"] = kwargs
        return DenseResult(68)

    events = []
    connectivity = run_connectivity(project, scouts, events.append, connectivity_function=fake_connectivity)
    assert observed["iterator"] is True
    assert observed["shapes"] == [(68, 64), (68, 64)]
    assert events[-1].stage is StageName.CONNECTIVITY
    with h5py.File(connectivity.records[0], "r") as file:
        assert file["matrix"].shape == (1, 68, 68)
        assert file["matrix"][0, 0, 1] == file["matrix"][0, 1, 0] == 0.25
        provenance = json.loads(file.attrs["method_provenance"])
        assert provenance["dropped_tail_samples"] == 7
    ingest_again = ingest_project(project)
    scouts_again = run_scouts(project, ingest_again, mne_projector=projector)
    assert ingest_again.run_id == ingest.run_id
    assert scouts_again.run_id == scouts.run_id and scouts_again.reused == 1
    connectivity_again = run_connectivity(project, scouts_again, connectivity_function=fake_connectivity)
    assert connectivity_again.run_id == connectivity.run_id and connectivity_again.reused == 1
    assert projector.calls == 1


def test_mne_resource_content_is_part_of_scout_run_identity(tmp_path: Path) -> None:
    project = _project(tmp_path)
    ingest = ingest_project(project)
    projector = FakeMNEProjector()
    first = run_scouts(project, ingest, mne_projector=projector)
    (tmp_path / "trans.fif").write_bytes(b"changed transformation resource")
    second = run_scouts(project, ingest, mne_projector=projector)
    assert first.run_id != second.run_id


@pytest.mark.parametrize(
    ("old", "new", "needle"),
    [
        ("atlas: aparc", "atlas: invented", "atlas"),
        ("noise_covariance: empirical", "noise_covariance: magic", "noise_covariance"),
        ("average_reference_projection: true", "average_reference_projection: false", "must be true"),
        ("snr: 3.0", "snr: true", "finite number"),
        ("chunk_samples: 40", "chunk_samples: 1.5", "integer"),
    ],
)
def test_mne_profile_rejects_unsupported_or_ambiguous_parameters(
    tmp_path: Path, old: str, new: str, needle: str
) -> None:
    from neurostat.analysis_profiles import AnalysisProfileError, parse_analysis_profile
    from neurostat.project import load_project

    project = _project(tmp_path)
    project.write_text(project.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(AnalysisProfileError, match=needle):
        parse_analysis_profile(load_project(project), "rest")


@pytest.mark.parametrize("unsafe_mode", ["pca_flip", "max"])
def test_mne_profile_defaults_to_mean_flip_and_rejects_cross_time_modes(
    tmp_path: Path, unsafe_mode: str
) -> None:
    from neurostat.analysis_profiles import AnalysisProfileError, parse_analysis_profile
    from neurostat.project import load_project

    project = _project(tmp_path)
    text = project.read_text(encoding="utf-8")
    project.write_text(text.replace("      label_mode: mean_flip\n", ""), encoding="utf-8")
    assert parse_analysis_profile(load_project(project), "rest").source.label_mode == "mean_flip"
    project.write_text(text.replace("label_mode: mean_flip", f"label_mode: {unsafe_mode}"), encoding="utf-8")
    with pytest.raises(AnalysisProfileError, match="label_mode"):
        parse_analysis_profile(load_project(project), "rest")


def test_fieldtrip_raw_is_rejected_before_mne_projector_due_to_coordinate_units(tmp_path: Path) -> None:
    project = _project(tmp_path)
    samples = 135
    time = np.arange(samples, dtype=np.float64) / 64.0
    savemat(
        tmp_path / "clean-fieldtrip.mat",
        {
            "data_ft": {
                "label": np.asarray(["A", "B"], dtype=object),
                "fsample": 64.0,
                "trial": np.vstack((np.sin(time), np.cos(time))),
                "time": time,
                # Deliberately millimetre-like values: Phase 3 cannot safely
                # infer or normalize FieldTrip coordinate units.
                "elec": {
                    "label": np.asarray(["A", "B"], dtype=object),
                    "chanpos": np.asarray([[0.0, 30.0, 80.0], [20.0, -20.0, 75.0]]),
                },
            }
        },
    )
    text = project.read_text(encoding="utf-8")
    text = text.replace("format: fif, path: clean-raw.fif", "format: fieldtrip_mat, path: clean-fieldtrip.mat")
    project.write_text(text, encoding="utf-8")
    ingest = ingest_project(project)
    projector = FakeMNEProjector()
    with pytest.raises(RuntimeError, match="MNE-ingested"):
        run_scouts(project, ingest, mne_projector=projector)
    assert projector.calls == 0


def test_cli_help_lists_source_and_connectivity_without_importing_science() -> None:
    root = Path(__file__).parents[1]
    env = os.environ | {"PYTHONPATH": str(root / "src")}
    command = (
        "from neurostat.cli import _build_parser; import sys; "
        "print(_build_parser().format_help()); print('MODULES=' + ' '.join(sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", command], text=True, capture_output=True, env=env, check=True)
    help_text, modules = result.stdout.split("MODULES=", 1)
    assert (
        "{validate,ingest,scouts,connectivity,network-models,graphvar,brainnet}"
        in help_text
    )
    assert not any(name in modules.split() for name in ("mne", "scipy", "h5py", "matplotlib", "pli_pipeline"))


def test_streamed_covariance_matches_numpy() -> None:
    from neurostat.scouts.mne_backend import _streamed_covariance

    rng = np.random.default_rng(123)
    data = rng.normal(size=(3, 137))
    actual = _streamed_covariance(data[np.newaxis, :, :], 3, 137, 29)
    np.testing.assert_allclose(actual, np.cov(data, ddof=1), rtol=1e-13, atol=1e-14)


def test_default_mne_backend_forwards_scientific_parameters_reorders_atlas_and_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    from neurostat.analysis_profiles import parse_analysis_profile
    from neurostat.project import load_project
    from neurostat.scouts.mne_backend import DefaultMNEProjector
    import mne.minimum_norm

    project = _project(tmp_path)
    ingest = ingest_project(project)
    config = parse_analysis_profile(load_project(project), "rest").source
    with pytest.raises(RuntimeError, match="only mean or mean_flip"):
        DefaultMNEProjector().project(ingest.records[0], replace(config, label_mode="pca_flip"))
    calls: dict[str, object] = {"apply": [], "extract": []}

    class Label:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(mne, "read_source_spaces", lambda path, **kwargs: "source-space")
    monkeypatch.setattr(mne, "read_bem_solution", lambda path, **kwargs: "bem")

    def forward(info, **kwargs):
        calls["forward"] = kwargs
        return "forward"

    monkeypatch.setattr(mne, "make_forward_solution", forward)
    monkeypatch.setattr(
        mne,
        "read_labels_from_annot",
        lambda *args, **kwargs: [Label(name) for name in reversed(DK68_LABELS)]
        + [Label("unknown-lh"), Label("corpuscallosum-rh")],
    )

    def inverse(info, forward, covariance, **kwargs):
        calls["inverse"] = kwargs
        calls["covariance"] = covariance.data.copy()
        return {"src": "inverse-source"}

    monkeypatch.setattr(mne.minimum_norm, "make_inverse_operator", inverse)

    def apply(raw, inverse, **kwargs):
        calls["apply"].append(kwargs)
        return {"samples": raw.n_times}

    monkeypatch.setattr(mne.minimum_norm, "apply_inverse_raw", apply)

    def extract(stc, labels, **kwargs):
        calls["extract"].append((tuple(label.name for label in labels), kwargs))
        return np.repeat(np.arange(68, dtype=np.float64)[:, None], stc["samples"], axis=1)

    monkeypatch.setattr(mne, "extract_label_time_course", extract)
    stream = DefaultMNEProjector().project(ingest.records[0], config)
    chunks = list(stream.chunks)
    assert [value.shape for _, value in chunks] == [(68, 40), (68, 40), (68, 40), (68, 15)]
    assert [start for start, _ in chunks] == [0, 40, 80, 120]
    assert calls["forward"]["trans"] == str(config.trans)
    assert calls["forward"]["src"] == "source-space"
    assert calls["forward"]["bem"] == "bem"
    assert calls["forward"]["mindist"] == config.mindist_mm
    assert calls["inverse"] == {"loose": 0.2, "depth": 0.8, "verbose": "ERROR"}
    assert all(call["lambda2"] == config.lambda2 and call["method"] == "MNE" for call in calls["apply"])
    assert all(call["pick_ori"] == "normal" and call["buffer_size"] == 40 for call in calls["apply"])
    assert all(names == DK68_LABELS for names, _ in calls["extract"])
    assert all(kwargs["mode"] == "mean_flip" and kwargs["src"] == "inverse-source" for _, kwargs in calls["extract"])

    calls["apply"].clear()
    none_stream = DefaultMNEProjector().project(ingest.records[0], replace(config, pick_orientation="none"))
    next(iter(none_stream.chunks))
    assert calls["apply"][0]["pick_ori"] is None
