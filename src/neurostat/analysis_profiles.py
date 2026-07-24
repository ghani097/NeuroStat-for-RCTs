"""Strict, dependency-light source and connectivity profile contracts."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .project import CleanProject, ProjectValidationError

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_FIXED_SOURCE = {
    "backend",
    "operator",
    "operator_variable",
    "operator_sensor_labels_variable",
    "operator_scout_labels_variable",
    "sensor_labels",
    "scout_labels",
    "epoch_reduction",
    "epoch_chunk_size",
    "output_chunk_samples",
    "export_matlab",
}
_MNE_SOURCE = {
    "backend",
    "subject",
    "subjects_dir",
    "trans",
    "source_space",
    "bem",
    "atlas",
    "inverse_method",
    "snr",
    "pick_orientation",
    "label_mode",
    "chunk_samples",
    "output_chunk_samples",
    "depth",
    "loose",
    "mindist_mm",
    "noise_covariance",
    "average_reference_projection",
    "coordinate_frame",
    "export_matlab",
}
_HILBERT_CONNECTIVITY = {"backend", "bands", "filter_order"}
_MULTITAPER_CONNECTIVITY = {
    "backend",
    "bands",
    "window_seconds",
    "tail_policy",
    "mt_bandwidth",
    "mt_adaptive",
    "mt_low_bias",
    "n_jobs",
}


class AnalysisProfileError(ProjectValidationError):
    """Raised when a profile cannot safely drive scientific analysis."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisProfileError(f"{name} must be a mapping")
    return value


def _unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise AnalysisProfileError(f"{name} has unknown keys: {', '.join(sorted(extra))}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisProfileError(f"{name} must be a nonblank string")
    return value.strip()


def _choice(value: Any, name: str, choices: set[str]) -> str:
    result = _string(value, name)
    if result not in choices:
        raise AnalysisProfileError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise AnalysisProfileError(f"{name} must be a boolean")
    return value


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise AnalysisProfileError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisProfileError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisProfileError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise AnalysisProfileError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise AnalysisProfileError(f"{name} must be <= {maximum}")
    return result


def _labels(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise AnalysisProfileError(f"{name} must be a nonempty list")
    result = tuple(_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise AnalysisProfileError(f"{name} must not contain duplicates")
    return result


def _resource(base: Path, value: Any, name: str, *, directory: bool = False) -> Path:
    path = (base / _string(value, name)).resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise AnalysisProfileError(f"{name} is not a {kind}: {path}")
    return path


def _bands(value: Any, name: str) -> tuple["Band", ...]:
    raw = _mapping(value, name)
    if not raw:
        raise AnalysisProfileError(f"{name} must not be empty")
    result: list[Band] = []
    for band_name, limits in raw.items():
        clean_name = _string(band_name, f"{name} name")
        if not _NAME.fullmatch(clean_name):
            raise AnalysisProfileError(f"{name} name {clean_name!r} is not a safe artifact label")
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            raise AnalysisProfileError(f"{name}.{clean_name} must be [low, high]")
        low = _number(limits[0], f"{name}.{clean_name}[0]", minimum=0.0)
        high = _number(limits[1], f"{name}.{clean_name}[1]", minimum=0.0)
        if low <= 0 or high <= low:
            raise AnalysisProfileError(f"{name}.{clean_name} needs 0 < low < high")
        result.append(Band(clean_name, low, high))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Band:
    name: str
    low_hz: float
    high_hz: float

    def normalized(self) -> dict[str, Any]:
        return {"name": self.name, "low_hz": self.low_hz, "high_hz": self.high_hz}


@dataclass(frozen=True, slots=True)
class FixedLinearSource:
    operator: Path
    operator_sha256: str
    operator_variable: str
    operator_sensor_labels_variable: str
    operator_scout_labels_variable: str
    sensor_labels: tuple[str, ...]
    scout_labels: tuple[str, ...]
    epoch_chunk_size: int
    output_chunk_samples: int
    export_matlab: bool
    backend: str = "fixed_linear_operator"
    epoch_reduction: str = "mean"

    def normalized(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "operator": str(self.operator),
            "operator_sha256": self.operator_sha256,
            "operator_variable": self.operator_variable,
            "operator_sensor_labels_variable": self.operator_sensor_labels_variable,
            "operator_scout_labels_variable": self.operator_scout_labels_variable,
            "sensor_labels": list(self.sensor_labels),
            "scout_labels": list(self.scout_labels),
            "epoch_reduction": self.epoch_reduction,
            "epoch_chunk_size": self.epoch_chunk_size,
            "output_chunk_samples": self.output_chunk_samples,
            "export_matlab": self.export_matlab,
        }


@dataclass(frozen=True, slots=True)
class MNEFsaverageSource:
    subject: str
    subjects_dir: Path
    trans: Path
    source_space: Path
    bem: Path
    resource_hashes: tuple[tuple[str, str], ...]
    inverse_method: str
    snr: float
    pick_orientation: str
    label_mode: str
    chunk_samples: int
    output_chunk_samples: int
    depth: float
    loose: float
    mindist_mm: float
    average_reference_projection: bool
    coordinate_frame: str
    export_matlab: bool
    backend: str = "mne_fsaverage"
    atlas: str = "aparc"
    noise_covariance: str = "empirical"

    @property
    def lambda2(self) -> float:
        return 1.0 / (self.snr * self.snr)

    def normalized(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "subject": self.subject,
            "subjects_dir": str(self.subjects_dir),
            "trans": str(self.trans),
            "source_space": str(self.source_space),
            "bem": str(self.bem),
            "resource_hashes": [list(item) for item in self.resource_hashes],
            "atlas": self.atlas,
            "inverse_method": self.inverse_method,
            "snr": self.snr,
            "lambda2": self.lambda2,
            "pick_orientation": self.pick_orientation,
            "label_mode": self.label_mode,
            "chunk_samples": self.chunk_samples,
            "output_chunk_samples": self.output_chunk_samples,
            "depth": self.depth,
            "loose": self.loose,
            "mindist_mm": self.mindist_mm,
            "noise_covariance": self.noise_covariance,
            "average_reference_projection": self.average_reference_projection,
            "coordinate_frame": self.coordinate_frame,
            "export_matlab": self.export_matlab,
        }


@dataclass(frozen=True, slots=True)
class EvokedHilbertConnectivity:
    bands: tuple[Band, ...]
    filter_order: int = 4
    backend: str = "evoked_average_hilbert_pli"

    def normalized(self) -> dict[str, Any]:
        return {"backend": self.backend, "bands": [band.normalized() for band in self.bands], "filter_order": self.filter_order}


@dataclass(frozen=True, slots=True)
class ContinuousMultitaperConnectivity:
    bands: tuple[Band, ...]
    window_seconds: float
    tail_policy: str
    mt_bandwidth: float | None
    mt_adaptive: bool
    mt_low_bias: bool
    n_jobs: int
    backend: str = "continuous_multitaper_pli"

    def normalized(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "bands": [band.normalized() for band in self.bands],
            "window_seconds": self.window_seconds,
            "tail_policy": self.tail_policy,
            "mt_bandwidth": self.mt_bandwidth,
            "mt_adaptive": self.mt_adaptive,
            "mt_low_bias": self.mt_low_bias,
            "n_jobs": self.n_jobs,
        }


SourceProfile = FixedLinearSource | MNEFsaverageSource
ConnectivityProfile = EvokedHilbertConnectivity | ContinuousMultitaperConnectivity


@dataclass(frozen=True, slots=True)
class AnalysisProfile:
    name: str
    source: SourceProfile
    connectivity: ConnectivityProfile

    def normalized(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source.normalized(), "connectivity": self.connectivity.normalized()}

    @property
    def digest(self) -> str:
        value = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_analysis_profile(project: CleanProject, name: str) -> AnalysisProfile:
    """Resolve and validate one profile without importing MNE or SciPy."""
    if name not in project.analysis_profiles:
        raise AnalysisProfileError(f"undeclared analysis profile {name!r}")
    raw = _mapping(project.analysis_profiles[name], f"analysis_profiles.{name}")
    if set(raw) != {"source", "connectivity"}:
        raise AnalysisProfileError(f"analysis_profiles.{name} must contain exactly source and connectivity")
    source_raw = _mapping(raw["source"], f"analysis_profiles.{name}.source")
    connectivity_raw = _mapping(raw["connectivity"], f"analysis_profiles.{name}.connectivity")
    source_backend = _choice(
        source_raw.get("backend"),
        f"analysis_profiles.{name}.source.backend",
        {"fixed_linear_operator", "mne_fsaverage"},
    )
    connectivity_backend = _choice(
        connectivity_raw.get("backend"),
        f"analysis_profiles.{name}.connectivity.backend",
        {"evoked_average_hilbert_pli", "continuous_multitaper_pli"},
    )
    pair = (source_backend, connectivity_backend)
    allowed_pairs = {
        ("fixed_linear_operator", "evoked_average_hilbert_pli"),
        ("mne_fsaverage", "continuous_multitaper_pli"),
    }
    if pair not in allowed_pairs:
        raise AnalysisProfileError(f"unsupported source/connectivity combination: {source_backend} + {connectivity_backend}")

    prefix = f"analysis_profiles.{name}"
    base = project.project_file.parent
    if source_backend == "fixed_linear_operator":
        _unknown(source_raw, _FIXED_SOURCE, f"{prefix}.source")
        operator = _resource(base, source_raw.get("operator"), f"{prefix}.source.operator")
        reduction = _choice(source_raw.get("epoch_reduction", "mean"), f"{prefix}.source.epoch_reduction", {"mean"})
        source: SourceProfile = FixedLinearSource(
            operator=operator,
            operator_sha256=_sha256(operator),
            operator_variable=_string(source_raw.get("operator_variable", "SensorToScoutOperator"), f"{prefix}.source.operator_variable"),
            operator_sensor_labels_variable=_string(source_raw.get("operator_sensor_labels_variable", "SensorLabels"), f"{prefix}.source.operator_sensor_labels_variable"),
            operator_scout_labels_variable=_string(source_raw.get("operator_scout_labels_variable", "ScoutLabels"), f"{prefix}.source.operator_scout_labels_variable"),
            sensor_labels=_labels(source_raw.get("sensor_labels"), f"{prefix}.source.sensor_labels"),
            scout_labels=_labels(source_raw.get("scout_labels"), f"{prefix}.source.scout_labels"),
            epoch_chunk_size=_integer(source_raw.get("epoch_chunk_size", 32), f"{prefix}.source.epoch_chunk_size"),
            epoch_reduction=reduction,
            output_chunk_samples=_integer(source_raw.get("output_chunk_samples", 4096), f"{prefix}.source.output_chunk_samples"),
            export_matlab=_boolean(source_raw.get("export_matlab", False), f"{prefix}.source.export_matlab"),
        )
    else:
        _unknown(source_raw, _MNE_SOURCE, f"{prefix}.source")
        subjects_dir = _resource(base, source_raw.get("subjects_dir"), f"{prefix}.source.subjects_dir", directory=True)
        subject = _string(source_raw.get("subject", "fsaverage"), f"{prefix}.source.subject")
        trans = _resource(base, source_raw.get("trans"), f"{prefix}.source.trans")
        source_space = _resource(base, source_raw.get("source_space"), f"{prefix}.source.source_space")
        bem = _resource(base, source_raw.get("bem"), f"{prefix}.source.bem")
        annotation_files = (
            subjects_dir / subject / "label" / "lh.aparc.annot",
            subjects_dir / subject / "label" / "rh.aparc.annot",
        )
        missing_annotations = [path for path in annotation_files if not path.is_file()]
        if missing_annotations:
            raise AnalysisProfileError(f"{prefix}.source atlas annotations are missing: {missing_annotations[0]}")
        resources = tuple((str(path), _sha256(path)) for path in (trans, source_space, bem, *annotation_files))
        atlas = _choice(source_raw.get("atlas", "aparc"), f"{prefix}.source.atlas", {"aparc"})
        noise_covariance = _choice(
            source_raw.get("noise_covariance", "empirical"),
            f"{prefix}.source.noise_covariance",
            {"empirical"},
        )
        average_reference_projection = _boolean(
            source_raw.get("average_reference_projection", True),
            f"{prefix}.source.average_reference_projection",
        )
        if not average_reference_projection:
            raise AnalysisProfileError(f"{prefix}.source.average_reference_projection must be true for the MNE inverse backend")
        source = MNEFsaverageSource(
            subject=subject,
            subjects_dir=subjects_dir,
            trans=trans,
            source_space=source_space,
            bem=bem,
            resource_hashes=resources,
            inverse_method=_choice(source_raw.get("inverse_method", "MNE"), f"{prefix}.source.inverse_method", {"MNE", "dSPM", "sLORETA", "eLORETA"}),
            snr=_number(source_raw.get("snr", 3.0), f"{prefix}.source.snr", minimum=1e-12),
            pick_orientation=_choice(source_raw.get("pick_orientation", "normal"), f"{prefix}.source.pick_orientation", {"normal", "none"}),
            label_mode=_choice(source_raw.get("label_mode", "pca_flip"), f"{prefix}.source.label_mode", {"mean", "mean_flip", "pca_flip", "max"}),
            chunk_samples=_integer(source_raw.get("chunk_samples", 16384), f"{prefix}.source.chunk_samples"),
            output_chunk_samples=_integer(source_raw.get("output_chunk_samples", 4096), f"{prefix}.source.output_chunk_samples"),
            depth=_number(source_raw.get("depth", 0.8), f"{prefix}.source.depth", minimum=0.0, maximum=1.0),
            loose=_number(source_raw.get("loose", 0.2), f"{prefix}.source.loose", minimum=0.0, maximum=1.0),
            mindist_mm=_number(source_raw.get("mindist_mm", 5.0), f"{prefix}.source.mindist_mm", minimum=0.0),
            average_reference_projection=average_reference_projection,
            coordinate_frame=_choice(source_raw.get("coordinate_frame", "head"), f"{prefix}.source.coordinate_frame", {"head"}),
            export_matlab=_boolean(source_raw.get("export_matlab", False), f"{prefix}.source.export_matlab"),
            atlas=atlas,
            noise_covariance=noise_covariance,
        )

    bands = _bands(connectivity_raw.get("bands"), f"{prefix}.connectivity.bands")
    if connectivity_backend == "evoked_average_hilbert_pli":
        _unknown(connectivity_raw, _HILBERT_CONNECTIVITY, f"{prefix}.connectivity")
        order = _integer(connectivity_raw.get("filter_order", 4), f"{prefix}.connectivity.filter_order")
        if order != 4:
            raise AnalysisProfileError(f"{prefix}.connectivity.filter_order must be 4 for V7-compatible Hilbert PLI")
        connectivity: ConnectivityProfile = EvokedHilbertConnectivity(bands, order)
    else:
        _unknown(connectivity_raw, _MULTITAPER_CONNECTIVITY, f"{prefix}.connectivity")
        bandwidth_raw = connectivity_raw.get("mt_bandwidth")
        bandwidth = None if bandwidth_raw is None else _number(bandwidth_raw, f"{prefix}.connectivity.mt_bandwidth", minimum=1e-12)
        connectivity = ContinuousMultitaperConnectivity(
            bands=bands,
            window_seconds=_number(connectivity_raw.get("window_seconds", 4.0), f"{prefix}.connectivity.window_seconds", minimum=1e-12),
            tail_policy=_choice(connectivity_raw.get("tail_policy", "drop"), f"{prefix}.connectivity.tail_policy", {"drop"}),
            mt_bandwidth=bandwidth,
            mt_adaptive=_boolean(connectivity_raw.get("mt_adaptive", True), f"{prefix}.connectivity.mt_adaptive"),
            mt_low_bias=_boolean(connectivity_raw.get("mt_low_bias", True), f"{prefix}.connectivity.mt_low_bias"),
            n_jobs=_integer(connectivity_raw.get("n_jobs", 1), f"{prefix}.connectivity.n_jobs"),
        )
    return AnalysisProfile(name, source, connectivity)


def project_analysis_profiles(project: CleanProject) -> dict[str, AnalysisProfile]:
    """Validate every profile referenced by at least one record."""
    names = sorted({record.analysis_profile for record in project.records})
    return {name: parse_analysis_profile(project, name) for name in names}
