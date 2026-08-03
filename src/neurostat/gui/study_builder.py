"""Build and validate v2 project YAML from Qt-free GUI study forms."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from neurostat.analysis_profiles import parse_analysis_profile
from neurostat.project import CleanProject, load_project

from .discovery import DiscoveredRecord, format_for_path, slugify
from .networks import default_networks


class AnalysisProfilePair(str, Enum):
    """Legal source/connectivity backend pairings exposed as one choice."""

    FIXED_EVOKED = "fixed_linear_operator+evoked_average_hilbert_pli"
    MNE_FSAVERAGE_CONTINUOUS = "mne_fsaverage+continuous_multitaper_pli"


BandLimits = Mapping[str, tuple[float, float]]
T = TypeVar("T")


def default_frequency_bands() -> dict[str, tuple[float, float]]:
    """Return v1-compatible frequency bands."""
    return {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }


@dataclass(frozen=True, slots=True)
class StudyForm:
    """All non-Qt study setup values collected by the future GUI view."""

    study_id: str
    base_folder: Path
    arms: tuple[str, ...]
    sessions: tuple[str, ...]
    conditions: tuple[str, ...]
    output_root: Path
    already_cleaned: bool
    frequency_bands: BandLimits = field(default_factory=default_frequency_bands)
    # Default to the MNE route, because that is what v1 did: it fetched
    # fsaverage automatically and needed nothing from the user. The fixed
    # operator route requires a precomputed sensor-to-scout .mat that most
    # studies do not have.
    profile_pair: AnalysisProfilePair = AnalysisProfilePair.MNE_FSAVERAGE_CONTINUOUS
    fixed_operator: Path | None = None
    fixed_sensor_labels: tuple[str, ...] = ()
    fixed_scout_labels: tuple[str, ...] = ()
    fixed_epoch_chunk_size: int = 32
    fixed_output_chunk_samples: int = 4096
    fixed_export_matlab: bool = False
    mne_subject: str = "fsaverage"
    mne_subjects_dir: Path | None = None
    mne_trans: Path | None = None
    mne_source_space: Path | None = None
    mne_bem: Path | None = None
    mne_window_seconds: float = 4.0
    mne_n_jobs: int = 1
    preprocessing_output_root: Path | None = None
    preprocessing_settings: Mapping[str, Any] = field(default_factory=dict)
    networks: Mapping[str, tuple[str, ...]] = field(default_factory=default_networks)
    project_file: Path | None = None

    @property
    def resolved_project_file(self) -> Path:
        """Return where the generated project YAML should be written."""
        if self.project_file is not None:
            return self.project_file
        return self.output_root / f"{slugify(self.study_id)}.project.yaml"

    @property
    def resolved_preprocessing_output_root(self) -> Path:
        """Return where v1 preprocessing should write cleaned FIF files."""
        if self.preprocessing_output_root is not None:
            return self.preprocessing_output_root
        return self.output_root / "preprocessed"


def validate_form(
    form: StudyForm,
    records: tuple[DiscoveredRecord, ...],
    stages: Iterable[Any] = (),
) -> tuple[str, ...]:
    """Return actionable setup problems before project YAML is written."""
    problems: list[str] = []
    if len(form.arms) != 2:
        problems.append("Enter exactly 2 study arms.")
    if len(form.sessions) < 2:
        problems.append("Enter at least 2 sessions.")
    if not records:
        problems.append("No EEG records were discovered in the selected study folder.")
    problems.extend(_profile_pair_problems(form, records))
    if _requests_stage(stages, "network_models"):
        if not form.networks:
            problems.append(
                "Select at least one network before running network models."
            )
        for name, labels in form.networks.items():
            if len(labels) < 2:
                problems.append(
                    f"Network {name!r} must contain at least 2 labels.",
                )

    participant_arms: dict[str, str] = {}
    for record in records:
        previous = participant_arms.setdefault(record.participant_id, record.arm)
        if previous != record.arm:
            problems.append(
                f"Participant {record.participant_id!r} appears under more than "
                "one arm.",
            )
            break

    duplicate_record_ids = _duplicates(record.record_id for record in records)
    if duplicate_record_ids:
        problems.append(
            "Record IDs must be unique; duplicates: " + ", ".join(duplicate_record_ids),
        )

    duplicate_keys = _duplicates(
        (
            record.participant_id,
            record.arm,
            record.session,
            record.condition,
        )
        for record in records
    )
    if duplicate_keys:
        rendered = ["/".join(str(value) for value in key) for key in duplicate_keys]
        problems.append(
            "Each participant/arm/session/condition can appear once; duplicates: "
            + ", ".join(rendered),
        )

    declared = (set(form.arms), set(form.sessions), set(form.conditions))
    for record in records:
        if record.arm not in declared[0]:
            problems.append(
                f"Record {record.record_id!r} uses undeclared arm {record.arm!r}."
            )
        if record.session not in declared[1]:
            problems.append(
                f"Record {record.record_id!r} uses undeclared session "
                f"{record.session!r}.",
            )
        if record.condition not in declared[2]:
            problems.append(
                f"Record {record.record_id!r} uses undeclared condition "
                f"{record.condition!r}.",
            )
        if not record.path.is_file():
            problems.append(
                f"Record {record.record_id!r} path does not exist: {record.path}.",
            )
        actual_format = format_for_path(record.path)
        if actual_format != record.format:
            problems.append(
                f"Record {record.record_id!r} format {record.format!r} does not "
                "match its file extension.",
            )
    return tuple(problems)


def recommended_profile_pair(
    records: Iterable[DiscoveredRecord],
) -> AnalysisProfilePair | None:
    """Return the only source pairing the discovered data can actually use.

    The scout stage is strict: `mne_fsaverage` needs continuous raw SET/FIF,
    because FieldTrip coordinate units are not canonicalized, while
    `fixed_linear_operator` needs retained epochs. Data that is all one shape has
    exactly one legal pairing; mixed data has none.
    """
    kinds = {record.data_kind for record in records}
    formats = {record.format for record in records}
    if not kinds:
        return None
    if kinds == {"raw"} and formats <= {"set", "fif"}:
        return AnalysisProfilePair.MNE_FSAVERAGE_CONTINUOUS
    if kinds == {"epochs"}:
        return AnalysisProfilePair.FIXED_EVOKED
    return None


def _profile_pair_problems(
    form: StudyForm,
    records: tuple[DiscoveredRecord, ...],
) -> tuple[str, ...]:
    """Catch a source backend the data cannot satisfy, before anything runs.

    Without this the run ingests every file first and only fails at the scout
    stage, which on a real study is minutes of wasted work.
    """
    if not records:
        return ()
    problems: list[str] = []
    if form.profile_pair is AnalysisProfilePair.MNE_FSAVERAGE_CONTINUOUS:
        epoched = sorted({r.record_id for r in records if r.data_kind != "raw"})
        wrong_format = sorted(
            {r.record_id for r in records if r.format not in {"set", "fif"}}
        )
        if epoched or wrong_format:
            problems.append(
                "The MNE fsaverage source needs continuous raw SET or FIF data, but "
                f"{len(epoched or wrong_format)} discovered record(s) are not "
                "(for example "
                f"{(epoched or wrong_format)[0]}). Switch Source/Connectivity to the "
                "fixed linear operator pairing and supply a sensor-to-scout operator, "
                "or ingest raw SET/FIF instead."
            )
    elif form.profile_pair is AnalysisProfilePair.FIXED_EVOKED:
        continuous = sorted({r.record_id for r in records if r.data_kind != "epochs"})
        if continuous:
            problems.append(
                "The fixed linear operator source needs epoched data, but "
                f"{len(continuous)} discovered record(s) are continuous (for example "
                f"{continuous[0]}). Switch Source/Connectivity to the MNE fsaverage "
                "pairing."
            )
        if form.fixed_operator is None:
            problems.append(
                "The fixed linear operator source needs a sensor-to-scout operator "
                ".mat file. Set it under Analysis."
            )
    return tuple(problems)


def build_project_document(
    form: StudyForm,
    records: tuple[DiscoveredRecord, ...],
    stages: Iterable[Any] = (),
) -> dict[str, Any]:
    """Build the v2 project mapping for a validated study setup."""
    document: dict[str, Any] = {
        "schema_version": 1,
        "study": {
            "id": form.study_id,
            "arms": list(form.arms),
            "sessions": list(form.sessions),
            "conditions": list(form.conditions),
        },
        "input": {"state": "cleaned"},
        "analysis_profiles": {"default": _analysis_profile(form)},
        "records": [_record_document(record) for record in records],
        "output": {"root": str(form.output_root)},
    }
    if _requests_stage(stages, "network_models"):
        document["network_models"] = build_network_models_block(form, form.networks)
    return document


def build_network_models_block(
    form: StudyForm,
    networks: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    """Build the strict network-model stage block from study-form values."""
    network_names = list(networks)
    bands = list(form.frequency_bands)
    families = [
        _network_model_family(form, condition, bands, network_names)
        for condition in form.conditions
    ]
    return {
        "networks": {name: list(labels) for name, labels in networks.items()},
        "families": families,
        "model": {
            "reml": True,
            "optimizer": "nloptwrap",
            "maxfun": 100000,
            "arm_contrasts": "sum",
            "session_contrasts": "poly",
            "df_method": "satterthwaite",
        },
        "diagnostics": {
            "residuals": True,
            "raw_scale": True,
            "winsorized": True,
            "winsor_robust_z": 3.5,
            "all_fit": False,
            "influence": False,
            "leave_one_out": False,
        },
        "runtime": {
            "executable": "Rscript",
            "library_paths": [],
            "dependency_contract": "builtin",
            "strict_renv": False,
            "timeout_seconds": 3600,
        },
        "validation": {
            "absolute_tolerance": 1.0e-10,
            "relative_tolerance": 1.0e-8,
        },
    }


def write_project(document: Mapping[str, Any], path: str | Path) -> CleanProject:
    """Write YAML, load it through project validation, then validate profiles."""
    project_path = Path(path)
    project_path.parent.mkdir(parents=True, exist_ok=True)
    project_path.write_text(
        yaml.safe_dump(dict(document), sort_keys=False),
        encoding="utf-8",
    )
    project = load_project(project_path)
    for name in project.analysis_profiles:
        parse_analysis_profile(project, name)
    return project


def _network_model_family(
    form: StudyForm,
    condition: str,
    bands: list[str],
    network_names: list[str],
) -> dict[str, Any]:
    arms = form.arms
    first_session = form.sessions[0]
    between_names = [
        f"{arms[0]}-minus-{arms[1]}-{session}" for session in form.sessions
    ]
    within_names = [f"{session}-minus-{first_session}" for session in form.sessions[1:]]
    return {
        "name": f"{slugify(condition)}-primary",
        "condition": condition,
        "bands": bands,
        "networks": network_names,
        "sessions": list(form.sessions),
        "boundary_observation_count": "scout_samples",
        "between_arm_contrasts": [
            {
                "name": name,
                "session": session,
                "numerator_arm": arms[0],
                "denominator_arm": arms[1],
            }
            for name, session in zip(between_names, form.sessions, strict=True)
        ],
        "within_arm_contrasts": [
            {
                "name": name,
                "arms": list(arms),
                "numerator_session": session,
                "denominator_session": first_session,
            }
            for name, session in zip(within_names, form.sessions[1:], strict=True)
        ],
        "correction_families": [
            {
                "name": "all-arm-tests",
                "target": "anova",
                "terms": ["arm"],
                "bands": bands,
                "networks": network_names,
                "methods": ["BH", "holm"],
            },
            {
                "name": "all-session-tests",
                "target": "anova",
                "terms": ["session"],
                "bands": bands,
                "networks": network_names,
                "methods": ["BH", "holm"],
            },
            {
                "name": "all-interaction-tests",
                "target": "anova",
                "terms": ["arm:session"],
                "bands": bands,
                "networks": network_names,
                "methods": ["BH", "holm"],
            },
            {
                "name": "all-primary-between",
                "target": "between",
                "contrasts": between_names,
                "bands": bands,
                "networks": network_names,
                "methods": ["BH", "holm", "bonferroni"],
            },
            {
                "name": "all-primary-within",
                "target": "within",
                "contrasts": within_names,
                "bands": bands,
                "networks": network_names,
                "methods": ["BH", "holm", "bonferroni"],
            },
        ],
    }


def _analysis_profile(form: StudyForm) -> dict[str, Any]:
    if form.profile_pair is AnalysisProfilePair.FIXED_EVOKED:
        return {
            "source": {
                "backend": "fixed_linear_operator",
                "operator": str(form.fixed_operator or Path("sensor-to-scout.mat")),
                "operator_variable": "SensorToScoutOperator",
                "sensor_labels": list(form.fixed_sensor_labels),
                "scout_labels": list(form.fixed_scout_labels),
                "epoch_reduction": "mean",
                "epoch_chunk_size": form.fixed_epoch_chunk_size,
                "output_chunk_samples": form.fixed_output_chunk_samples,
                "export_matlab": form.fixed_export_matlab,
            },
            "connectivity": {
                "backend": "evoked_average_hilbert_pli",
                "filter_order": 4,
                "bands": _bands_document(form.frequency_bands),
            },
        }
    return {
        "source": {
            "backend": "mne_fsaverage",
            "subject": form.mne_subject,
            "subjects_dir": str(form.mne_subjects_dir or Path("subjects")),
            "trans": str(form.mne_trans or Path("fsaverage-trans.fif")),
            "source_space": str(
                form.mne_source_space
                or Path("subjects") / "fsaverage" / "bem" / "fsaverage-ico5-src.fif",
            ),
            "bem": str(
                form.mne_bem
                or Path("subjects")
                / "fsaverage"
                / "bem"
                / "fsaverage-5120-5120-5120-bem-sol.fif",
            ),
            "atlas": "aparc",
            "inverse_method": "MNE",
            "snr": 3.0,
            "pick_orientation": "normal",
            "label_mode": "mean_flip",
            "chunk_samples": 16384,
            "output_chunk_samples": 4096,
            "depth": 0.8,
            "loose": 0.2,
            "mindist_mm": 5.0,
            "noise_covariance": "empirical",
            "average_reference_projection": True,
            "coordinate_frame": "head",
            "export_matlab": False,
        },
        "connectivity": {
            "backend": "continuous_multitaper_pli",
            "bands": _bands_document(form.frequency_bands),
            "window_seconds": form.mne_window_seconds,
            "tail_policy": "drop",
            "mt_bandwidth": None,
            "mt_adaptive": True,
            "mt_low_bias": True,
            "n_jobs": form.mne_n_jobs,
        },
    }


def _record_document(record: DiscoveredRecord) -> dict[str, Any]:
    item: dict[str, Any] = {
        "record_id": record.record_id,
        "participant_id": record.participant_id,
        "arm": record.arm,
        "session": record.session,
        "condition": record.condition,
        "data_kind": record.data_kind,
        "format": record.format,
        "path": str(record.path),
        "analysis_profile": "default",
    }
    if record.mat_variable is not None:
        item["mat_variable"] = record.mat_variable
    if record.samples_per_epoch is not None:
        item["samples_per_epoch"] = record.samples_per_epoch
    return item


def _bands_document(bands: BandLimits) -> dict[str, list[float]]:
    return {name: [float(low), float(high)] for name, (low, high) in bands.items()}


def _duplicates(values: Iterable[T]) -> list[T]:
    counts = Counter(values)
    return [value for value, count in counts.items() if count > 1]


def _requests_stage(stages: Iterable[Any], stage_name: str) -> bool:
    for stage in stages:
        value = getattr(stage, "value", stage)
        if value == "all" or value == stage_name:
            return True
    return False


__all__ = [
    "AnalysisProfilePair",
    "StudyForm",
    "build_network_models_block",
    "build_project_document",
    "default_frequency_bands",
    "validate_form",
    "write_project",
]
