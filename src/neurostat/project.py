"""Strict, lightweight project-file validation for clean-input studies."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .contracts import RecordKey

_RECORD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_TOP = {
    "schema_version",
    "study",
    "input",
    "records",
    "analysis_profiles",
    "output",
    "contrasts",
    "families",
    "runtime",
    "network_models",
    "graphvar",
    "brainnet",
}
_RECORD = {"record_id", "participant_id", "arm", "session", "condition", "data_kind", "format", "path", "analysis_profile", "mat_variable", "samples_per_epoch", "data_unit"}
_REQUIRED = {"record_id", "participant_id", "arm", "session", "condition", "data_kind", "format", "path", "analysis_profile"}
_FORMATS = {"set": {".set"}, "fif": {".fif", ".fif.gz"}, "fieldtrip_mat": {".mat"}}


class ProjectValidationError(ValueError):
    """Raised when a project does not meet the v2 clean-input contract."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in node.value:  # type: ignore[attr-defined]
        name = loader.construct_object(key, deep=deep)
        if name in result:
            raise ProjectValidationError(f"duplicate YAML mapping key: {name!r}")
        result[name] = loader.construct_object(value, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectValidationError(f"{name} must be a nonblank string")
    return value.strip()


def _list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProjectValidationError(f"{name} must be a nonempty list")
    result = tuple(_string(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ProjectValidationError(f"{name} must not contain duplicate values")
    return result


@dataclass(frozen=True, slots=True)
class Study:
    id: str
    arms: tuple[str, str]
    sessions: tuple[str, ...]
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    record_id: str
    key: RecordKey
    data_kind: str
    format: str
    path: Path
    analysis_profile: str
    mat_variable: str | None = None
    samples_per_epoch: int | None = None
    data_unit: str | None = None


@dataclass(frozen=True, slots=True)
class CleanProject:
    project_file: Path
    study: Study
    records: tuple[ProjectRecord, ...]
    analysis_profiles: Mapping[str, Mapping[str, Any]]
    output_root: Path
    extras: Mapping[str, Mapping[str, Any]]

    def normalized(self) -> dict[str, Any]:
        return {"schema_version": 1, "study": {"id": self.study.id, "arms": list(self.study.arms), "sessions": list(self.study.sessions), "conditions": list(self.study.conditions)}, "input": {"state": "cleaned"}, "records": [{"record_id": r.record_id, "participant_id": r.key.participant_id, "arm": r.key.arm, "session": r.key.session, "condition": r.key.condition, "data_kind": r.data_kind, "format": r.format, "path": str(r.path), "analysis_profile": r.analysis_profile, "mat_variable": r.mat_variable, "samples_per_epoch": r.samples_per_epoch, "data_unit": r.data_unit} for r in self.records], "analysis_profiles": _thaw(self.analysis_profiles), "output": {"root": str(self.output_root)}, **_thaw(self.extras)}


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping): return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple): return [_thaw(item) for item in value]
    return value


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file(): raise ProjectValidationError(f"records manifest is not a file: {path}")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    if path.suffix.lower() not in {".csv", ".tsv"}: raise ProjectValidationError("records manifest must be CSV or TSV")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames: raise ProjectValidationError("records manifest has no header")
        headers = reader.fieldnames
        if any(not name or not name.strip() for name in headers): raise ProjectValidationError("records manifest has blank header")
        if len(set(headers)) != len(headers): raise ProjectValidationError("records manifest has duplicate headers")
        unknown = set(headers) - _RECORD
        if unknown: raise ProjectValidationError(f"records manifest has unknown headers: {', '.join(sorted(unknown))}")
        missing = _REQUIRED - set(headers)
        if missing: raise ProjectValidationError(f"records manifest missing required headers: {', '.join(sorted(missing))}")
        rows = [dict(row) for row in reader]
        if not rows: raise ProjectValidationError("records manifest must contain at least one record")
        return rows


def _record(row: Mapping[str, Any], base: Path, index: int, study: Study, profiles: Mapping[str, Any]) -> ProjectRecord:
    unknown = set(row) - _RECORD
    if unknown: raise ProjectValidationError(f"record {index} has unknown fields: {', '.join(sorted(unknown))}")
    missing = [key for key in _REQUIRED if row.get(key) is None or row.get(key) == ""]
    if missing: raise ProjectValidationError(f"record {index} missing required fields: {', '.join(sorted(missing))}")
    rid = _string(row["record_id"], f"record {index}.record_id")
    if not _RECORD_ID.fullmatch(rid): raise ProjectValidationError(f"record {index}.record_id is not a safe artifact slug")
    key = RecordKey(*(_string(row[name], f"record {index}.{name}") for name in ("participant_id", "arm", "session", "condition")))
    if key.arm not in study.arms or key.session not in study.sessions or key.condition not in study.conditions: raise ProjectValidationError(f"record {rid} references an undeclared arm, session, or condition")
    kind, fmt = _string(row["data_kind"], "data_kind"), _string(row["format"], "format")
    if kind not in {"raw", "epochs"} or fmt not in _FORMATS: raise ProjectValidationError(f"record {rid} has unsupported format or data_kind")
    source = (base / _string(row["path"], "path")).resolve()
    if not source.is_file(): raise ProjectValidationError(f"record {rid} path is not a file: {source}")
    lower = source.name.lower()
    if not any(lower.endswith(ext) for ext in _FORMATS[fmt]): raise ProjectValidationError(f"record {rid} path extension does not match format {fmt}")
    profile = _string(row["analysis_profile"], "analysis_profile")
    if profile not in profiles: raise ProjectValidationError(f"record {rid} references undeclared analysis profile {profile!r}")
    spe = row.get("samples_per_epoch")
    if spe not in (None, ""):
        if isinstance(spe, bool) or not str(spe).isdigit() or int(spe) <= 0: raise ProjectValidationError(f"record {rid}.samples_per_epoch must be a positive integer")
        spe = int(spe)
    mat_raw = row.get("mat_variable")
    mat_variable = None if mat_raw is None or mat_raw == "" else _string(mat_raw, f"record {rid}.mat_variable")
    unit_raw = row.get("data_unit")
    unit = None if unit_raw is None or unit_raw == "" else unit_raw
    if unit is not None and (not isinstance(unit, str) or unit not in {"V", "mV", "uV", "native"}): raise ProjectValidationError(f"record {rid}.data_unit is invalid")
    if fmt in {"set", "fif"} and unit not in (None, "V"): raise ProjectValidationError(f"record {rid}: SET/FIF data_unit must be V or blank")
    if mat_variable is not None and fmt != "fieldtrip_mat": raise ProjectValidationError(f"record {rid}.mat_variable is only valid for FieldTrip")
    if spe is not None and (fmt != "fieldtrip_mat" or kind != "epochs"): raise ProjectValidationError(f"record {rid}.samples_per_epoch is only valid for FieldTrip epochs")
    return ProjectRecord(rid, key, kind, fmt, source, profile, mat_variable, spe, unit)


def load_project(project_file: str | Path) -> CleanProject:
    path = Path(project_file).resolve()
    if not path.is_file(): raise ProjectValidationError(f"project file is not a file: {path}")
    try:
        root = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except yaml.YAMLError as exc: raise ProjectValidationError(f"invalid YAML: {exc}") from exc
    if not isinstance(root, Mapping): raise ProjectValidationError("project must be a YAML mapping")
    unknown = set(root) - _TOP
    if unknown: raise ProjectValidationError(f"project has unknown keys: {', '.join(sorted(unknown))}")
    if type(root.get("schema_version")) is not int or root.get("schema_version") != 1: raise ProjectValidationError("schema_version must be integer 1")
    study_raw = root.get("study")
    if not isinstance(study_raw, Mapping) or set(study_raw) != {"id", "arms", "sessions", "conditions"}: raise ProjectValidationError("study must contain exactly id, arms, sessions, and conditions")
    arms = _list(study_raw["arms"], "study.arms")
    if len(arms) != 2: raise ProjectValidationError("study.arms must contain exactly two arms")
    sessions, conditions = _list(study_raw["sessions"], "study.sessions"), _list(study_raw["conditions"], "study.conditions")
    if len(sessions) < 2: raise ProjectValidationError("study.sessions must contain at least two sessions")
    study = Study(_string(study_raw["id"], "study.id"), (arms[0], arms[1]), sessions, conditions)
    if root.get("input") != {"state": "cleaned"}: raise ProjectValidationError("input must be exactly {state: cleaned}; v2 ingestion never cleans data")
    profile_raw = root.get("analysis_profiles")
    if not isinstance(profile_raw, Mapping) or not profile_raw: raise ProjectValidationError("analysis_profiles must be a nonempty mapping")
    if any(not isinstance(k, str) or not k.strip() or not isinstance(v, Mapping) for k, v in profile_raw.items()): raise ProjectValidationError("analysis_profiles needs nonblank mapping names and mapping values")
    profiles = _freeze(profile_raw)
    records_raw = root.get("records")
    base = path.parent
    if isinstance(records_raw, Mapping) and set(records_raw) == {"manifest"}:
        manifest = (base / _string(records_raw["manifest"], "records.manifest")).resolve(); records_raw = _read_manifest(manifest); base = manifest.parent
    if not isinstance(records_raw, list) or not records_raw: raise ProjectValidationError("records must be a nonempty inline list or manifest mapping")
    records = tuple(_record(item, base, i + 1, study, profiles) if isinstance(item, Mapping) else (_ for _ in ()).throw(ProjectValidationError(f"record {i + 1} must be a mapping")) for i, item in enumerate(records_raw))
    if len({r.record_id for r in records}) != len(records): raise ProjectValidationError("record_id values must be unique")
    if len({r.key for r in records}) != len(records): raise ProjectValidationError("RecordKey tuples must be unique")
    participant_arms: dict[str, str] = {}
    for record in records:
        previous = participant_arms.setdefault(record.key.participant_id, record.key.arm)
        if previous != record.key.arm: raise ProjectValidationError(f"participant {record.key.participant_id!r} maps to multiple arms")
    output = root.get("output")
    if not isinstance(output, Mapping) or set(output) != {"root"}: raise ProjectValidationError("output must contain exactly root")
    extras: dict[str, Mapping[str, Any]] = {}
    for key in (
        "contrasts",
        "families",
        "runtime",
        "network_models",
        "graphvar",
        "brainnet",
    ):
        if key in root:
            if not isinstance(root[key], Mapping): raise ProjectValidationError(f"{key} must be a mapping")
            extras[key] = _freeze(root[key])
    return CleanProject(path, study, records, profiles, (path.parent / _string(output["root"], "output.root")).resolve(), MappingProxyType(extras))
