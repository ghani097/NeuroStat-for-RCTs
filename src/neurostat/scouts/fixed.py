"""Fixed sensor-to-scout projection, including the manuscript V7 route."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..analysis_profiles import FixedLinearSource
from ..stage_io import digest_json
from .artifacts import ScoutError, ScoutStream


def _labels(value: Any) -> tuple[str, ...]:
    array = np.asarray(value, dtype=object)
    result: list[str] = []
    for item in array.ravel(order="F"):
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.item()
        if isinstance(item, bytes):
            text = item.decode("utf-8")
        elif isinstance(item, np.ndarray) and np.issubdtype(item.dtype, np.integer):
            text = "".join(chr(int(code)) for code in item.ravel(order="F") if int(code))
        else:
            text = str(item)
        result.append(text.strip())
    return tuple(result)


def _classic_operator(path: Path, config: FixedLinearSource) -> tuple[np.ndarray, tuple[str, ...] | None, tuple[str, ...] | None]:
    from scipy.io import loadmat

    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    if config.operator_variable not in content:
        raise ScoutError(f"operator variable {config.operator_variable!r} not found: {path}")
    matrix = np.asarray(content[config.operator_variable], dtype=np.float64)
    sensor = _labels(content[config.operator_sensor_labels_variable]) if config.operator_sensor_labels_variable in content else None
    scout = _labels(content[config.operator_scout_labels_variable]) if config.operator_scout_labels_variable in content else None
    return matrix, sensor, scout


def _hdf5_text(file: Any, value: Any) -> tuple[str, ...]:
    import h5py

    if isinstance(value, h5py.Dataset):
        raw = value[()]
        if raw.dtype == h5py.ref_dtype:
            texts: list[str] = []
            for reference in raw.ravel(order="F"):
                texts.extend(_hdf5_text(file, file[reference]))
            return tuple(texts)
        matlab_class = value.attrs.get("MATLAB_class", b"")
        if matlab_class in {b"char", "char"}:
            return ("".join(chr(int(code)) for code in np.asarray(raw).ravel(order="F") if int(code)).strip(),)
        return _labels(raw)
    return _labels(value)


def _hdf5_operator(path: Path, config: FixedLinearSource) -> tuple[np.ndarray, tuple[str, ...] | None, tuple[str, ...] | None]:
    import h5py

    with h5py.File(path, "r") as file:
        if config.operator_variable not in file:
            raise ScoutError(f"operator variable {config.operator_variable!r} not found: {path}")
        dataset = file[config.operator_variable]
        matrix = np.asarray(dataset[()], dtype=np.float64)
        # MATLAB stores numeric array axes reversed in v7.3/HDF5 files.
        if "MATLAB_class" in dataset.attrs or "#refs#" in file:
            matrix = matrix.T
        sensor = _hdf5_text(file, file[config.operator_sensor_labels_variable]) if config.operator_sensor_labels_variable in file else None
        scout = _hdf5_text(file, file[config.operator_scout_labels_variable]) if config.operator_scout_labels_variable in file else None
    return matrix, sensor, scout


def load_operator(config: FixedLinearSource) -> np.ndarray:
    """Load, orient, and validate a classic or v7.3 MAT operator."""
    try:
        matrix, sensor_metadata, scout_metadata = _classic_operator(config.operator, config)
    except (NotImplementedError, ValueError, OSError):
        matrix, sensor_metadata, scout_metadata = _hdf5_operator(config.operator, config)
    expected = (len(config.scout_labels), len(config.sensor_labels))
    if matrix.shape != expected:
        raise ScoutError(f"operator shape {matrix.shape} does not match scouts x sensors {expected}")
    if not np.isfinite(matrix).all():
        raise ScoutError("operator contains non-finite values")
    if sensor_metadata is not None and sensor_metadata != config.sensor_labels:
        raise ScoutError("operator sensor-label metadata disagrees with the configured ordered contract")
    if scout_metadata is not None and scout_metadata != config.scout_labels:
        raise ScoutError("operator scout-label metadata disagrees with the configured ordered contract")
    return np.asarray(matrix, dtype=np.float64)


def project_fixed(canonical_path: Path, config: FixedLinearSource) -> ScoutStream:
    """Average retained epochs in float64, reorder sensors, and project once."""
    import h5py

    operator = load_operator(config)
    with h5py.File(canonical_path, "r") as file:
        data = file["data"]
        canonical_labels = tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in file["channel_names"][:])
        if len(canonical_labels) != len(set(canonical_labels)):
            raise ScoutError("canonical channel labels are duplicated")
        if set(canonical_labels) != set(config.sensor_labels) or len(canonical_labels) != len(config.sensor_labels):
            missing = sorted(set(config.sensor_labels) - set(canonical_labels))
            extra = sorted(set(canonical_labels) - set(config.sensor_labels))
            raise ScoutError(f"canonical/operator sensor labels do not match exactly (missing={missing}, extra={extra})")
        order = tuple(canonical_labels.index(label) for label in config.sensor_labels)
        if str(file.attrs["data_kind"]) == "raw" and data.shape[0] != 1:
            raise ScoutError("canonical raw data must contain exactly one epoch")
        # Read all channels per bounded epoch block, then reorder in NumPy.
        # This supports non-monotone label contracts without h5py fancy-index
        # restrictions and matches the accepted V7 float64 accumulation.
        sensor_sum = np.zeros((len(config.sensor_labels), data.shape[-1]), dtype=np.float64)
        for start in range(0, data.shape[0], config.epoch_chunk_size):
            block = np.asarray(data[start : start + config.epoch_chunk_size, :, :], dtype=np.float64)
            sensor_sum += block[:, order, :].sum(axis=0, dtype=np.float64)
        sensor_average = sensor_sum / int(data.shape[0])
        times = np.asarray(file["times"][:], dtype=np.float64)
        sfreq = float(file.attrs["sfreq"])
        epoch_count = int(data.shape[0])
    values = np.asarray(operator @ sensor_average, dtype=np.float64)
    contract = {
        "sensor_labels": list(config.sensor_labels),
        "scout_labels": list(config.scout_labels),
        "operator_sha256": config.operator_sha256,
        "operator_variable": config.operator_variable,
    }
    provenance = {
        "backend": config.backend,
        "operator": str(config.operator),
        "operator_sha256": config.operator_sha256,
        "operator_variable": config.operator_variable,
        "label_contract_sha256": digest_json(contract),
        "sensor_labels": list(config.sensor_labels),
        "scout_labels": list(config.scout_labels),
        "epoch_reduction": "bounded epoch blocks: sum(axis=0,dtype=float64), then divide by retained epoch count",
        "epoch_chunk_size": config.epoch_chunk_size,
        "retained_epoch_count": epoch_count,
        "matrix_multiplication": "operator(scouts x sensors) @ sensor_average(sensors x samples)",
        "hidden_preprocessing": False,
    }
    return ScoutStream(
        labels=config.scout_labels,
        times=times,
        sfreq=sfreq,
        chunks=((0, values),),
        provenance=provenance,
    )
