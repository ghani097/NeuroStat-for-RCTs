"""Scientifically explicit PLI implementations for the two supported routes."""
from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from ..analysis_profiles import Band, ContinuousMultitaperConnectivity, EvokedHilbertConnectivity


class ConnectivityError(RuntimeError):
    """Raised for invalid connectivity input, computation, or artifacts."""


@dataclass(frozen=True, slots=True)
class ConnectivityValues:
    matrices: np.ndarray
    sos: np.ndarray | None
    provenance: dict[str, Any]


def validate_pli_matrix(matrix: np.ndarray, *, tolerance: float = 1e-12) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not value.shape[0]:
        raise ConnectivityError(f"PLI matrix must be nonempty and square; got {value.shape}")
    if not np.isfinite(value).all():
        raise ConnectivityError("PLI matrix contains non-finite values")
    if np.max(np.abs(value - value.T)) > tolerance:
        raise ConnectivityError("PLI matrix is not symmetric")
    if value.min() < -tolerance or value.max() > 1.0 + tolerance:
        raise ConnectivityError("PLI matrix values are outside [0, 1]")
    if np.max(np.abs(np.diag(value))) > tolerance:
        raise ConnectivityError("PLI matrix diagonal is not zero")
    result = np.clip(value, 0.0, 1.0)
    np.fill_diagonal(result, 0.0)
    return result


def _validate_bands(bands: Sequence[Band], sfreq: float) -> None:
    nyquist = sfreq / 2.0
    if isinstance(sfreq, bool) or not np.isfinite(sfreq) or sfreq <= 0:
        raise ConnectivityError("sampling frequency must be positive and finite")
    for band in bands:
        if band.high_hz >= nyquist:
            raise ConnectivityError(f"band {band.name!r} high edge must be below Nyquist ({nyquist:g} Hz)")


def evoked_average_hilbert_pli(
    scouts: np.ndarray,
    sfreq: float,
    config: EvokedHilbertConnectivity,
) -> ConnectivityValues:
    """Run the exact V7 Butterworth/Hilbert/all-samples PLI formula."""
    import scipy
    from scipy.signal import butter, hilbert, sosfiltfilt

    values = np.asarray(scouts, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or not np.isfinite(values).all():
        raise ConnectivityError("evoked scouts must be a finite nodes x samples matrix")
    _validate_bands(config.bands, sfreq)
    matrices: list[np.ndarray] = []
    designs: list[np.ndarray] = []
    band_provenance: dict[str, Any] = {}
    for band in config.bands:
        sos = butter(
            config.filter_order,
            [band.low_hz, band.high_hz],
            btype="bandpass",
            fs=sfreq,
            output="sos",
        )
        filtered = sosfiltfilt(sos, values, axis=-1)
        phase = np.angle(hilbert(filtered, axis=-1))
        differences = phase[:, np.newaxis, :] - phase[np.newaxis, :, :]
        matrix = np.abs(np.mean(np.sign(np.sin(differences)), axis=-1))
        np.fill_diagonal(matrix, 0.0)
        matrices.append(validate_pli_matrix(matrix))
        designs.append(np.asarray(sos, dtype=np.float64))
        band_provenance[band.name] = {
            "low_hz": band.low_hz,
            "high_hz": band.high_hz,
            "second_order_sections": np.asarray(sos, dtype=np.float64).tolist(),
        }
    return ConnectivityValues(
        matrices=np.stack(matrices),
        sos=np.stack(designs),
        provenance={
            "backend": config.backend,
            "formula": "abs(mean(sign(sin(phase_i - phase_j)), axis=all_retained_samples))",
            "analytic_signal": "scipy.signal.hilbert(axis=-1)",
            "filter_implementation": "scipy.signal.butter(output='sos') + scipy.signal.sosfiltfilt(axis=-1)",
            "butterworth_prototype_order": config.filter_order,
            "bandpass_realized_order": 2 * config.filter_order,
            "phase_treatment": "zero-phase forward/backward filtering",
            "sample_policy": "one matrix across every retained scout sample; no temporal subdivision",
            "diagonal_policy": "exactly zero",
            "bands": band_provenance,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "python_version": platform.python_version(),
        },
    )


def scout_window_count(samples: int, window_samples: int, tail_policy: str = "drop") -> tuple[int, int]:
    if type(samples) is not int or samples <= 0 or type(window_samples) is not int or window_samples <= 0:
        raise ConnectivityError("sample and window counts must be positive integers")
    if tail_policy != "drop":
        raise ConnectivityError("only the explicit drop tail policy is supported")
    count, tail = divmod(samples, window_samples)
    if count == 0:
        raise ConnectivityError("continuous scouts contain zero complete connectivity windows")
    return count, tail


def iter_scout_windows(path: Path, window_samples: int, tail_policy: str = "drop") -> Iterator[np.ndarray]:
    """Yield one nodes x window array at a time directly from HDF5."""
    import h5py

    with h5py.File(path, "r") as file:
        data = file["data"]
        count, _ = scout_window_count(int(data.shape[-1]), window_samples, tail_policy)
        for index in range(count):
            start = index * window_samples
            yield np.asarray(data[:, start : start + window_samples], dtype=np.float64)


def _dense_bands(raw: Any, nodes: int, bands: int) -> np.ndarray:
    dense = raw.get_data(output="dense") if hasattr(raw, "get_data") else raw[0]
    value = np.asarray(dense, dtype=np.float64)
    if value.ndim == 2 and bands == 1:
        value = value[:, :, np.newaxis]
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.shape == (nodes, nodes, bands):
        value = np.moveaxis(value, -1, 0)
    elif value.shape != (bands, nodes, nodes):
        raise ConnectivityError(f"unexpected dense multitaper PLI shape {value.shape}")
    result = np.empty_like(value)
    for index, matrix in enumerate(value):
        # mne-connectivity dense bivariate results occupy one triangle.
        symmetric = np.maximum(matrix, matrix.T)
        np.fill_diagonal(symmetric, 0.0)
        result[index] = validate_pli_matrix(symmetric)
    return result


def continuous_multitaper_pli(
    scout_path: Path,
    labels: tuple[str, ...],
    sfreq: float,
    samples: int,
    config: ContinuousMultitaperConnectivity,
    *,
    connectivity_function: Callable[..., Any] | None = None,
) -> ConnectivityValues:
    """Stream non-overlapping scout windows into mne-connectivity."""
    _validate_bands(config.bands, sfreq)
    exact_window = config.window_seconds * sfreq
    window_samples = int(round(exact_window))
    if window_samples <= 0 or not np.isclose(exact_window, window_samples, rtol=0.0, atol=1e-9):
        raise ConnectivityError("window_seconds * sfreq must be a positive integer sample count")
    window_count, dropped_tail = scout_window_count(samples, window_samples, config.tail_policy)
    if connectivity_function is None:
        import mne_connectivity
        from mne_connectivity import spectral_connectivity_epochs

        connectivity_function = spectral_connectivity_epochs
        connectivity_version = mne_connectivity.__version__
    else:
        connectivity_version = "injected"
    raw = connectivity_function(
        iter_scout_windows(scout_path, window_samples, config.tail_policy),
        names=list(labels),
        method="pli",
        mode="multitaper",
        sfreq=sfreq,
        fmin=tuple(band.low_hz for band in config.bands),
        fmax=tuple(band.high_hz for band in config.bands),
        faverage=True,
        mt_bandwidth=config.mt_bandwidth,
        mt_adaptive=config.mt_adaptive,
        mt_low_bias=config.mt_low_bias,
        n_jobs=config.n_jobs,
        verbose=False,
    )
    matrices = _dense_bands(raw, len(labels), len(config.bands))
    return ConnectivityValues(
        matrices=matrices,
        sos=None,
        provenance={
            "backend": config.backend,
            "implementation": "mne_connectivity.spectral_connectivity_epochs",
            "method": "pli",
            "mode": "multitaper",
            "sfreq": sfreq,
            "faverage": True,
            "window_seconds_requested": config.window_seconds,
            "window_samples": window_samples,
            "window_count": window_count,
            "tail_policy": config.tail_policy,
            "dropped_tail_samples": dropped_tail,
            "mt_bandwidth": config.mt_bandwidth,
            "mt_adaptive": config.mt_adaptive,
            "mt_low_bias": config.mt_low_bias,
            "n_jobs": config.n_jobs,
            "diagonal_policy": "exactly zero after symmetric dense normalization",
            "mne_connectivity_version": connectivity_version,
            "numpy_version": np.__version__,
        },
    )
