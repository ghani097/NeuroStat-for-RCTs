from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.signal import butter, hilbert, sosfiltfilt

from neurostat.analysis_profiles import (
    Band,
    ContinuousMultitaperConnectivity,
    EvokedHilbertConnectivity,
)
from neurostat.connectivity import (
    ConnectivityError,
    continuous_multitaper_pli,
    evoked_average_hilbert_pli,
    iter_scout_windows,
    scout_window_count,
)
from neurostat.scouts.artifacts import ScoutStream, write_scout_artifact


V7_BANDS = (
    Band("theta", 3.5, 7.5),
    Band("alpha", 7.5, 12.5),
    Band("beta", 12.5, 30.0),
    Band("gamma", 30.0, 40.0),
)


def _scout_artifact(tmp_path: Path, data: np.ndarray, sfreq: float = 100.0) -> Path:
    path = tmp_path / "scouts.h5"
    labels = tuple(f"node-{index}" for index in range(data.shape[0]))
    times = np.arange(data.shape[1], dtype=np.float64) / sfreq
    stream = ScoutStream(labels, times, sfreq, ((0, data),), {"backend": "test"})
    write_scout_artifact(
        path,
        stream,
        {
            "record_id": "r1",
            "participant_id": "p1",
            "arm": "active",
            "session": "baseline",
            "condition": "rest",
            "analysis_profile": "test",
            "source_backend": "test",
            "canonical_sha256": "canonical",
            "canonical_payload_digest": "payload",
            "profile_digest": "profile",
        },
        chunk_samples=31,
    )
    return path


def test_evoked_v7_formula_sos_and_all_samples_match_direct_scipy() -> None:
    sfreq = 2048.0
    samples = 512
    time = np.arange(samples) / sfreq
    scouts = np.vstack(
        [
            np.sin(2 * np.pi * 10 * time),
            np.sin(2 * np.pi * 10 * time + np.pi / 4),
            np.sin(2 * np.pi * 22 * time + np.pi / 7),
        ]
    )
    config = EvokedHilbertConnectivity(V7_BANDS, 4)
    result = evoked_average_hilbert_pli(scouts, sfreq, config)
    for index, band in enumerate(V7_BANDS):
        expected_sos = butter(4, [band.low_hz, band.high_hz], btype="bandpass", fs=sfreq, output="sos")
        filtered = sosfiltfilt(expected_sos, scouts, axis=-1)
        phase = np.angle(hilbert(filtered, axis=-1))
        expected = np.abs(np.mean(np.sign(np.sin(phase[:, None, :] - phase[None, :, :])), axis=-1))
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_array_equal(result.sos[index], expected_sos)
        np.testing.assert_array_equal(result.matrices[index], expected)
    assert result.provenance["bandpass_realized_order"] == 8
    assert "no temporal subdivision" in result.provenance["sample_policy"]


def test_evoked_pli_invariants_and_nyquist_rejection() -> None:
    rng = np.random.default_rng(42)
    result = evoked_average_hilbert_pli(
        rng.normal(size=(5, 300)),
        100.0,
        EvokedHilbertConnectivity((Band("alpha", 8.0, 12.0),), 4),
    )
    matrix = result.matrices[0]
    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), np.zeros(5))
    assert matrix.min() >= 0 and matrix.max() <= 1
    with pytest.raises(ConnectivityError, match="Nyquist"):
        evoked_average_hilbert_pli(rng.normal(size=(2, 100)), 20.0, EvokedHilbertConnectivity((Band("bad", 8.0, 12.0),), 4))


def test_window_iterator_streams_complete_windows_and_drops_tail(tmp_path: Path) -> None:
    values = np.arange(2 * 23, dtype=np.float64).reshape(2, 23)
    path = _scout_artifact(tmp_path, values, sfreq=10.0)
    iterator = iter_scout_windows(path, 10, "drop")
    assert not isinstance(iterator, (list, tuple))
    windows = list(iterator)
    assert len(windows) == 2
    np.testing.assert_array_equal(windows[0], values[:, :10])
    np.testing.assert_array_equal(windows[1], values[:, 10:20])
    assert scout_window_count(23, 10) == (2, 3)
    with pytest.raises(ConnectivityError, match="zero complete"):
        scout_window_count(9, 10)


def test_continuous_backend_passes_a_generator_and_normalizes_dense_triangle(tmp_path: Path) -> None:
    values = np.arange(2 * 23, dtype=np.float64).reshape(2, 23)
    path = _scout_artifact(tmp_path, values, sfreq=10.0)
    observed: dict[str, object] = {}

    class Result:
        def get_data(self, output: str):
            assert output == "dense"
            dense = np.zeros((2, 2, 1), dtype=np.float64)
            dense[1, 0, 0] = 0.75
            return dense

    def fake(data, **kwargs):
        observed["is_iterator"] = iter(data) is data
        observed["windows"] = [block.copy() for block in data]
        observed["kwargs"] = kwargs
        return Result()

    config = ContinuousMultitaperConnectivity(
        (Band("slow", 1.0, 2.0),),
        window_seconds=1.0,
        tail_policy="drop",
        mt_bandwidth=None,
        mt_adaptive=True,
        mt_low_bias=True,
        n_jobs=1,
    )
    result = continuous_multitaper_pli(path, ("node-0", "node-1"), 10.0, 23, config, connectivity_function=fake)
    assert observed["is_iterator"] is True
    assert len(observed["windows"]) == 2
    assert observed["kwargs"]["method"] == "pli"
    assert observed["kwargs"]["mode"] == "multitaper"
    assert observed["kwargs"]["faverage"] is True
    np.testing.assert_array_equal(result.matrices[0], np.asarray([[0.0, 0.75], [0.75, 0.0]]))
    assert result.provenance["dropped_tail_samples"] == 3


def test_continuous_multitaper_agrees_with_direct_mne_connectivity(tmp_path: Path) -> None:
    from mne_connectivity import spectral_connectivity_epochs

    rng = np.random.default_rng(7)
    sfreq = 64.0
    values = rng.normal(size=(3, 64 * 6 + 7))
    path = _scout_artifact(tmp_path, values, sfreq)
    config = ContinuousMultitaperConnectivity(
        (Band("alpha", 8.0, 12.0),),
        window_seconds=1.0,
        tail_policy="drop",
        mt_bandwidth=4.0,
        mt_adaptive=False,
        mt_low_bias=True,
        n_jobs=1,
    )
    streamed = continuous_multitaper_pli(path, ("node-0", "node-1", "node-2"), sfreq, values.shape[1], config)
    direct = spectral_connectivity_epochs(
        [values[:, start : start + 64] for start in range(0, 64 * 6, 64)],
        names=["node-0", "node-1", "node-2"],
        method="pli",
        mode="multitaper",
        sfreq=sfreq,
        fmin=(8.0,),
        fmax=(12.0,),
        faverage=True,
        mt_bandwidth=4.0,
        mt_adaptive=False,
        mt_low_bias=True,
        n_jobs=1,
        verbose=False,
    ).get_data(output="dense")[:, :, 0]
    direct = np.maximum(direct, direct.T)
    np.fill_diagonal(direct, 0.0)
    np.testing.assert_allclose(streamed.matrices[0], direct, rtol=0.0, atol=0.0)


def test_scout_artifact_chunking_and_payload_corruption_detection(tmp_path: Path) -> None:
    from neurostat.scouts import ScoutError, verify_scout_artifact

    path = _scout_artifact(tmp_path, np.ones((2, 101)), sfreq=100.0)
    header = verify_scout_artifact(path)
    assert header.samples == 101
    with h5py.File(path, "r") as file:
        assert file["data"].chunks == (2, 31)
        assert json.loads(file.attrs["dimensions"]) == [2, 101]
    with h5py.File(path, "r+") as file:
        file["times"][0] += 0.01
    with pytest.raises(ScoutError, match="times|payload digest"):
        verify_scout_artifact(path)
