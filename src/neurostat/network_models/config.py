"""Strict, dependency-light configuration for the network-model stage."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..project import CleanProject, ProjectValidationError
from ..stage_io import sha256_file

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_ROOT_KEYS = {"networks", "families", "model", "diagnostics", "runtime", "validation"}
_FAMILY_KEYS = {
    "name",
    "condition",
    "bands",
    "networks",
    "sessions",
    "boundary_observation_count",
    "between_arm_contrasts",
    "within_arm_contrasts",
    "correction_families",
}
_BETWEEN_KEYS = {"name", "session", "numerator_arm", "denominator_arm"}
_WITHIN_KEYS = {"name", "arms", "numerator_session", "denominator_session"}
_CORRECTION_KEYS = {"name", "target", "contrasts", "terms", "bands", "networks", "methods"}
_MODEL_KEYS = {"reml", "optimizer", "maxfun", "arm_contrasts", "session_contrasts", "df_method"}
_DIAGNOSTIC_KEYS = {
    "residuals",
    "raw_scale",
    "winsorized",
    "winsor_robust_z",
    "all_fit",
    "influence",
    "leave_one_out",
}
_RUNTIME_KEYS = {
    "executable",
    "library_paths",
    "dependency_contract",
    "strict_renv",
    "renv_lockfile",
    "timeout_seconds",
}
_VALIDATION_KEYS = {"absolute_tolerance", "relative_tolerance"}


class NetworkModelConfigError(ProjectValidationError):
    """Raised before any statistical process when a model contract is unsafe."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NetworkModelConfigError(f"{name} must be a mapping")
    return value


def _unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise NetworkModelConfigError(f"{name} has unknown keys: {', '.join(sorted(extra))}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NetworkModelConfigError(f"{name} must be a nonblank string")
    return value.strip()


def _name(value: Any, name: str) -> str:
    result = _string(value, name)
    if not _NAME.fullmatch(result):
        raise NetworkModelConfigError(f"{name} must be a safe artifact name")
    return result


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise NetworkModelConfigError(f"{name} must be a boolean")
    return value


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise NetworkModelConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NetworkModelConfigError(f"{name} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise NetworkModelConfigError(f"{name} must be a finite number >= {minimum}")
    return result


def _strings(value: Any, name: str, *, minimum: int = 1) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        raise NetworkModelConfigError(f"{name} must contain at least {minimum} value(s)")
    result = tuple(_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise NetworkModelConfigError(f"{name} must not contain duplicates")
    return result


def _choice(value: Any, name: str, choices: set[str]) -> str:
    result = _string(value, name)
    if result not in choices:
        raise NetworkModelConfigError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return result


@dataclass(frozen=True, slots=True)
class NetworkDefinition:
    name: str
    labels: tuple[str, ...]

    def normalized(self) -> dict[str, Any]:
        return {"name": self.name, "labels": list(self.labels)}


@dataclass(frozen=True, slots=True)
class BetweenArmContrast:
    name: str
    session: str
    numerator_arm: str
    denominator_arm: str

    def normalized(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "session": self.session,
            "numerator_arm": self.numerator_arm,
            "denominator_arm": self.denominator_arm,
        }


@dataclass(frozen=True, slots=True)
class WithinArmContrast:
    name: str
    arms: tuple[str, ...]
    numerator_session: str
    denominator_session: str

    def normalized(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arms": list(self.arms),
            "numerator_session": self.numerator_session,
            "denominator_session": self.denominator_session,
        }


@dataclass(frozen=True, slots=True)
class CorrectionFamily:
    name: str
    target: str
    contrasts: tuple[str, ...]
    terms: tuple[str, ...]
    bands: tuple[str, ...]
    networks: tuple[str, ...]
    methods: tuple[str, ...]

    def normalized(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "contrasts": list(self.contrasts),
            "terms": list(self.terms),
            "bands": list(self.bands),
            "networks": list(self.networks),
            "methods": list(self.methods),
        }


@dataclass(frozen=True, slots=True)
class AnalysisFamily:
    name: str
    condition: str
    bands: tuple[str, ...]
    networks: tuple[str, ...]
    sessions: tuple[str, ...]
    boundary_observation_count: str
    between_arm_contrasts: tuple[BetweenArmContrast, ...]
    within_arm_contrasts: tuple[WithinArmContrast, ...]
    correction_families: tuple[CorrectionFamily, ...]

    def normalized(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "condition": self.condition,
            "bands": list(self.bands),
            "networks": list(self.networks),
            "sessions": list(self.sessions),
            "boundary_observation_count": self.boundary_observation_count,
            "between_arm_contrasts": [item.normalized() for item in self.between_arm_contrasts],
            "within_arm_contrasts": [item.normalized() for item in self.within_arm_contrasts],
            "correction_families": [item.normalized() for item in self.correction_families],
        }


@dataclass(frozen=True, slots=True)
class ModelSettings:
    reml: bool
    optimizer: str
    maxfun: int
    arm_contrasts: str
    session_contrasts: str
    df_method: str

    def normalized(self) -> dict[str, Any]:
        return {
            "reml": self.reml,
            "optimizer": self.optimizer,
            "maxfun": self.maxfun,
            "arm_contrasts": self.arm_contrasts,
            "session_contrasts": self.session_contrasts,
            "df_method": self.df_method,
        }


@dataclass(frozen=True, slots=True)
class DiagnosticSettings:
    residuals: bool
    raw_scale: bool
    winsorized: bool
    winsor_robust_z: float
    all_fit: bool
    influence: bool
    leave_one_out: bool

    def normalized(self) -> dict[str, Any]:
        return {
            "residuals": self.residuals,
            "raw_scale": self.raw_scale,
            "winsorized": self.winsorized,
            "winsor_robust_z": self.winsor_robust_z,
            "all_fit": self.all_fit,
            "influence": self.influence,
            "leave_one_out": self.leave_one_out,
        }


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    executable: str
    library_paths: tuple[Path, ...]
    dependency_contract_kind: str
    dependency_contract: Path
    dependency_contract_sha256: str
    strict_renv: bool
    renv_lockfile: Path | None
    renv_lockfile_sha256: str | None
    timeout_seconds: int

    def normalized(self) -> dict[str, Any]:
        return {
            "executable": self.executable,
            "library_paths": [str(path) for path in self.library_paths],
            "dependency_contract_kind": self.dependency_contract_kind,
            "dependency_contract": (
                "builtin:r-package-contract.tsv"
                if self.dependency_contract_kind == "builtin"
                else str(self.dependency_contract)
            ),
            "dependency_contract_sha256": self.dependency_contract_sha256,
            "strict_renv": self.strict_renv,
            "renv_lockfile": None if self.renv_lockfile is None else str(self.renv_lockfile),
            "renv_lockfile_sha256": self.renv_lockfile_sha256,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class ValidationSettings:
    absolute_tolerance: float
    relative_tolerance: float

    def normalized(self) -> dict[str, Any]:
        return {
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
        }


@dataclass(frozen=True, slots=True)
class NetworkModelConfig:
    networks: tuple[NetworkDefinition, ...]
    families: tuple[AnalysisFamily, ...]
    model: ModelSettings
    diagnostics: DiagnosticSettings
    runtime: RuntimeSettings
    validation: ValidationSettings

    @property
    def networks_by_name(self) -> dict[str, NetworkDefinition]:
        return {item.name: item for item in self.networks}

    def normalized(self) -> dict[str, Any]:
        return {
            "networks": [item.normalized() for item in self.networks],
            "families": [item.normalized() for item in self.families],
            "model": self.model.normalized(),
            "diagnostics": self.diagnostics.normalized(),
            "runtime": self.runtime.normalized(),
            "validation": self.validation.normalized(),
        }


def _resource_path() -> Path:
    return Path(__file__).parent / "resources" / "r-package-contract.tsv"


def _runtime(project: CleanProject, raw: Any) -> RuntimeSettings:
    value = _mapping(raw, "network_models.runtime")
    _unknown(value, _RUNTIME_KEYS, "network_models.runtime")
    executable = _string(value.get("executable"), "network_models.runtime.executable")
    if Path(executable).is_absolute() or "/" in executable or "\\" in executable:
        executable_path = (project.project_file.parent / executable).resolve()
        if not executable_path.is_file():
            raise NetworkModelConfigError(
                f"network_models.runtime.executable is not a file: {executable_path}"
            )
        executable = str(executable_path)
    paths_raw = value.get("library_paths", [])
    if not isinstance(paths_raw, (list, tuple)):
        raise NetworkModelConfigError("network_models.runtime.library_paths must be a list")
    library_paths: list[Path] = []
    for index, item in enumerate(paths_raw):
        path = (project.project_file.parent / _string(item, f"network_models.runtime.library_paths[{index}]")).resolve()
        if not path.is_dir():
            raise NetworkModelConfigError(f"network_models.runtime.library_paths[{index}] is not a directory: {path}")
        library_paths.append(path)
    if len(set(library_paths)) != len(library_paths):
        raise NetworkModelConfigError("network_models.runtime.library_paths must not contain duplicates")
    contract_raw = _string(value.get("dependency_contract", "builtin"), "network_models.runtime.dependency_contract")
    if contract_raw == "builtin":
        contract_kind, contract = "builtin", _resource_path().resolve()
    else:
        contract_kind = "external"
        contract = (project.project_file.parent / contract_raw).resolve()
    if not contract.is_file():
        raise NetworkModelConfigError(f"network_models dependency contract is not a file: {contract}")
    strict_renv = _boolean(value.get("strict_renv", False), "network_models.runtime.strict_renv")
    lock_raw = value.get("renv_lockfile")
    lockfile = None
    if lock_raw is not None:
        lockfile = (project.project_file.parent / _string(lock_raw, "network_models.runtime.renv_lockfile")).resolve()
        if not lockfile.is_file():
            raise NetworkModelConfigError(f"network_models.runtime.renv_lockfile is not a file: {lockfile}")
    if strict_renv and lockfile is None:
        raise NetworkModelConfigError("strict_renv requires an explicit renv_lockfile")
    return RuntimeSettings(
        executable=executable,
        library_paths=tuple(library_paths),
        dependency_contract_kind=contract_kind,
        dependency_contract=contract,
        dependency_contract_sha256=sha256_file(contract),
        strict_renv=strict_renv,
        renv_lockfile=lockfile,
        renv_lockfile_sha256=None if lockfile is None else sha256_file(lockfile),
        timeout_seconds=_integer(value.get("timeout_seconds", 3600), "network_models.runtime.timeout_seconds"),
    )


def _family(
    raw: Any,
    index: int,
    project: CleanProject,
    networks: dict[str, NetworkDefinition],
) -> AnalysisFamily:
    prefix = f"network_models.families[{index}]"
    value = _mapping(raw, prefix)
    _unknown(value, _FAMILY_KEYS, prefix)
    name = _name(value.get("name"), f"{prefix}.name")
    condition = _choice(value.get("condition"), f"{prefix}.condition", set(project.study.conditions))
    bands = _strings(value.get("bands"), f"{prefix}.bands")
    network_names = _strings(value.get("networks"), f"{prefix}.networks")
    unknown_networks = set(network_names) - set(networks)
    if unknown_networks:
        raise NetworkModelConfigError(f"{prefix}.networks references unknown values: {', '.join(sorted(unknown_networks))}")
    sessions = _strings(value.get("sessions"), f"{prefix}.sessions", minimum=2)
    undeclared_sessions = set(sessions) - set(project.study.sessions)
    if undeclared_sessions:
        raise NetworkModelConfigError(f"{prefix}.sessions references undeclared values: {', '.join(sorted(undeclared_sessions))}")
    boundary = _choice(
        value.get("boundary_observation_count"),
        f"{prefix}.boundary_observation_count",
        {"scout_samples", "connectivity_windows"},
    )
    between_raw = value.get("between_arm_contrasts")
    if not isinstance(between_raw, (list, tuple)) or not between_raw:
        raise NetworkModelConfigError(f"{prefix}.between_arm_contrasts must be a nonempty list")
    between: list[BetweenArmContrast] = []
    for contrast_index, item in enumerate(between_raw):
        cp = f"{prefix}.between_arm_contrasts[{contrast_index}]"
        entry = _mapping(item, cp)
        _unknown(entry, _BETWEEN_KEYS, cp)
        numerator = _choice(entry.get("numerator_arm"), f"{cp}.numerator_arm", set(project.study.arms))
        denominator = _choice(entry.get("denominator_arm"), f"{cp}.denominator_arm", set(project.study.arms))
        if numerator == denominator:
            raise NetworkModelConfigError(f"{cp} numerator and denominator arms must differ")
        between.append(
            BetweenArmContrast(
                _name(entry.get("name"), f"{cp}.name"),
                _choice(entry.get("session"), f"{cp}.session", set(sessions)),
                numerator,
                denominator,
            )
        )
    within_raw = value.get("within_arm_contrasts")
    if not isinstance(within_raw, (list, tuple)) or not within_raw:
        raise NetworkModelConfigError(f"{prefix}.within_arm_contrasts must be a nonempty list")
    within: list[WithinArmContrast] = []
    for contrast_index, item in enumerate(within_raw):
        cp = f"{prefix}.within_arm_contrasts[{contrast_index}]"
        entry = _mapping(item, cp)
        _unknown(entry, _WITHIN_KEYS, cp)
        arms = _strings(entry.get("arms"), f"{cp}.arms")
        if set(arms) - set(project.study.arms):
            raise NetworkModelConfigError(f"{cp}.arms contains an undeclared arm")
        numerator = _choice(entry.get("numerator_session"), f"{cp}.numerator_session", set(sessions))
        denominator = _choice(entry.get("denominator_session"), f"{cp}.denominator_session", set(sessions))
        if numerator == denominator:
            raise NetworkModelConfigError(f"{cp} numerator and denominator sessions must differ")
        within.append(WithinArmContrast(_name(entry.get("name"), f"{cp}.name"), arms, numerator, denominator))
    between_names = [item.name for item in between]
    within_names = [item.name for item in within]
    if len(set(between_names)) != len(between_names) or len(set(within_names)) != len(within_names):
        raise NetworkModelConfigError(f"{prefix} contrast names must be unique within each target")
    corrections_raw = value.get("correction_families")
    if not isinstance(corrections_raw, (list, tuple)) or not corrections_raw:
        raise NetworkModelConfigError(f"{prefix}.correction_families must be a nonempty list")
    corrections: list[CorrectionFamily] = []
    for correction_index, item in enumerate(corrections_raw):
        cp = f"{prefix}.correction_families[{correction_index}]"
        entry = _mapping(item, cp)
        _unknown(entry, _CORRECTION_KEYS, cp)
        target = _choice(entry.get("target"), f"{cp}.target", {"anova", "between", "within"})
        if target == "anova":
            if "contrasts" in entry:
                raise NetworkModelConfigError(f"{cp}.contrasts is invalid for target anova")
            contrast_names: tuple[str, ...] = ()
            terms = _strings(entry.get("terms"), f"{cp}.terms")
            if set(terms) - {"arm", "session", "arm:session"}:
                raise NetworkModelConfigError(f"{cp}.terms contains an unsupported Type-III term")
        else:
            if "terms" in entry:
                raise NetworkModelConfigError(f"{cp}.terms is only valid for target anova")
            terms = ()
            contrast_names = _strings(entry.get("contrasts"), f"{cp}.contrasts")
            allowed_contrasts = set(between_names if target == "between" else within_names)
            if set(contrast_names) - allowed_contrasts:
                raise NetworkModelConfigError(f"{cp}.contrasts references a contrast outside target {target}")
        selected_bands = _strings(entry.get("bands"), f"{cp}.bands")
        selected_networks = _strings(entry.get("networks"), f"{cp}.networks")
        if set(selected_bands) - set(bands) or set(selected_networks) - set(network_names):
            raise NetworkModelConfigError(f"{cp} references a band or network outside its analysis family")
        methods = _strings(entry.get("methods"), f"{cp}.methods")
        canonical_methods: list[str] = []
        aliases = {"BH": "BH", "bh": "BH", "fdr_bh": "BH", "holm": "holm", "bonferroni": "bonferroni"}
        for method in methods:
            if method not in aliases:
                raise NetworkModelConfigError(f"{cp}.methods has unsupported correction {method!r}")
            canonical_methods.append(aliases[method])
        if len(set(canonical_methods)) != len(canonical_methods):
            raise NetworkModelConfigError(f"{cp}.methods contains duplicate correction aliases")
        corrections.append(
            CorrectionFamily(
                _name(entry.get("name"), f"{cp}.name"),
                target,
                contrast_names,
                terms,
                selected_bands,
                selected_networks,
                tuple(canonical_methods),
            )
        )
    correction_names = [item.name for item in corrections]
    if len(set(correction_names)) != len(correction_names):
        raise NetworkModelConfigError(f"{prefix}.correction_families names must be unique")
    expected_memberships = {
        (target, identifier, band, network)
        for target, identifiers in (
            ("anova", ("arm", "session", "arm:session")),
            ("between", between_names),
            ("within", within_names),
        )
        for identifier in identifiers
        for band in bands
        for network in network_names
    }
    actual_memberships: dict[tuple[str, str, str, str], list[str]] = {}
    for correction in corrections:
        identifiers = correction.terms if correction.target == "anova" else correction.contrasts
        for contrast in identifiers:
            for band in correction.bands:
                for network in correction.networks:
                    actual_memberships.setdefault(
                        (correction.target, contrast, band, network), []
                    ).append(correction.name)
    missing_memberships = expected_memberships - set(actual_memberships)
    overlapping = {key: names for key, names in actual_memberships.items() if len(names) != 1}
    if missing_memberships:
        first = sorted(missing_memberships)[0]
        raise NetworkModelConfigError(
            f"{prefix}.correction_families does not cover inferential result {first}"
        )
    if overlapping:
        first, names = sorted(overlapping.items())[0]
        raise NetworkModelConfigError(
            f"{prefix}.correction_families assigns inferential result {first} more than once: "
            f"{', '.join(names)}"
        )
    return AnalysisFamily(
        name,
        condition,
        bands,
        network_names,
        sessions,
        boundary,
        tuple(between),
        tuple(within),
        tuple(corrections),
    )


def parse_network_model_config(project: CleanProject) -> NetworkModelConfig:
    """Parse the optional project section only when this stage is requested."""
    raw = project.extras.get("network_models")
    if raw is None:
        raise NetworkModelConfigError("project has no network_models configuration")
    value = _mapping(raw, "network_models")
    if set(value) != _ROOT_KEYS:
        missing = _ROOT_KEYS - set(value)
        extra = set(value) - _ROOT_KEYS
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if extra:
            details.append(f"unknown {', '.join(sorted(extra))}")
        raise NetworkModelConfigError(f"network_models must contain exactly the typed sections ({'; '.join(details)})")
    networks_raw = _mapping(value["networks"], "network_models.networks")
    if not networks_raw:
        raise NetworkModelConfigError("network_models.networks must not be empty")
    networks: list[NetworkDefinition] = []
    for raw_name, raw_labels in networks_raw.items():
        name = _name(raw_name, "network_models.networks name")
        labels = _strings(raw_labels, f"network_models.networks.{name}", minimum=2)
        networks.append(NetworkDefinition(name, labels))
    network_map = {item.name: item for item in networks}
    if len(network_map) != len(networks):
        raise NetworkModelConfigError("network_models.networks names must be unique")
    families_raw = value["families"]
    if not isinstance(families_raw, (list, tuple)) or not families_raw:
        raise NetworkModelConfigError("network_models.families must be a nonempty list")
    families = tuple(_family(item, index, project, network_map) for index, item in enumerate(families_raw))
    if len({item.name for item in families}) != len(families):
        raise NetworkModelConfigError("network_models.families names must be unique")

    model_raw = _mapping(value["model"], "network_models.model")
    _unknown(model_raw, _MODEL_KEYS, "network_models.model")
    model = ModelSettings(
        reml=_boolean(model_raw.get("reml", True), "network_models.model.reml"),
        optimizer=_choice(model_raw.get("optimizer", "nloptwrap"), "network_models.model.optimizer", {"nloptwrap"}),
        maxfun=_integer(model_raw.get("maxfun", 100000), "network_models.model.maxfun"),
        arm_contrasts=_choice(model_raw.get("arm_contrasts", "sum"), "network_models.model.arm_contrasts", {"sum"}),
        session_contrasts=_choice(model_raw.get("session_contrasts", "poly"), "network_models.model.session_contrasts", {"poly"}),
        df_method=_choice(model_raw.get("df_method", "satterthwaite"), "network_models.model.df_method", {"satterthwaite"}),
    )
    if not model.reml or model.maxfun != 100000:
        raise NetworkModelConfigError("network_models.model requires reml: true and maxfun: 100000")

    diagnostics_raw = _mapping(value["diagnostics"], "network_models.diagnostics")
    _unknown(diagnostics_raw, _DIAGNOSTIC_KEYS, "network_models.diagnostics")
    diagnostics = DiagnosticSettings(
        residuals=_boolean(diagnostics_raw.get("residuals", True), "network_models.diagnostics.residuals"),
        raw_scale=_boolean(diagnostics_raw.get("raw_scale", True), "network_models.diagnostics.raw_scale"),
        winsorized=_boolean(diagnostics_raw.get("winsorized", False), "network_models.diagnostics.winsorized"),
        winsor_robust_z=_number(
            diagnostics_raw.get("winsor_robust_z", 3.5),
            "network_models.diagnostics.winsor_robust_z",
            minimum=1e-12,
        ),
        all_fit=_boolean(diagnostics_raw.get("all_fit", False), "network_models.diagnostics.all_fit"),
        influence=_boolean(diagnostics_raw.get("influence", False), "network_models.diagnostics.influence"),
        leave_one_out=_boolean(diagnostics_raw.get("leave_one_out", False), "network_models.diagnostics.leave_one_out"),
    )
    validation_raw = _mapping(value["validation"], "network_models.validation")
    _unknown(validation_raw, _VALIDATION_KEYS, "network_models.validation")
    validation = ValidationSettings(
        _number(validation_raw.get("absolute_tolerance", 1e-10), "network_models.validation.absolute_tolerance"),
        _number(validation_raw.get("relative_tolerance", 1e-8), "network_models.validation.relative_tolerance"),
    )
    return NetworkModelConfig(tuple(networks), families, model, diagnostics, _runtime(project, value["runtime"]), validation)
