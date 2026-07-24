from __future__ import annotations

from pathlib import Path

import pytest

from neurostat.project import ProjectValidationError, load_project


FIELDS = "record_id,participant_id,arm,session,condition,data_kind,format,path,analysis_profile"


def _source(directory: Path, name: str = "clean.fif") -> Path:
    path = directory / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"fixture"); return path


def _yaml(records: str, **replacement: str) -> str:
    values = {"schema": "1", "arms": "[active, comparison]", "sessions": "[baseline, followup]", "conditions": "[rest]", "profiles": "{default: {nested: {value: one}}}", "records": records, "input": "{state: cleaned}"} | replacement
    return "schema_version: {schema}\nstudy: {{id: demo, arms: {arms}, sessions: {sessions}, conditions: {conditions}}}\ninput: {input}\nrecords: {records}\nanalysis_profiles: {profiles}\noutput: {{root: output}}\n".format(**values)


def _row(path: str = "clean.fif", **changes: str) -> str:
    row = {"record_id": "r-1", "participant_id": "p-1", "arm": "active", "session": "baseline", "condition": "rest", "data_kind": "raw", "format": "fif", "path": path, "analysis_profile": "default"} | changes
    return "{" + ", ".join(f"{key}: {value}" for key, value in row.items()) + "}"


def _load(tmp_path: Path, text: str):
    path = tmp_path / "project.yaml"; path.write_text(text, encoding="utf-8"); return load_project(path)


def test_inline_and_manifest_relative_bases(tmp_path: Path) -> None:
    _source(tmp_path)
    inline = _load(tmp_path, _yaml("[" + _row() + "]"))
    assert inline.records[0].path == (tmp_path / "clean.fif").resolve()
    data = tmp_path / "manifest"; _source(data)
    for suffix, delimiter in (("csv", ","), ("tsv", "\t")):
        manifest = data / f"records.{suffix}"
        manifest.write_text(delimiter.join(FIELDS.split(",")) + "\n" + delimiter.join(["r-1", "p-1", "active", "baseline", "rest", "raw", "fif", "clean.fif", "default"]) + "\n", encoding="utf-8")
        project = _load(tmp_path, _yaml("{manifest: manifest/" + manifest.name + "}"))
        assert project.records[0].path == (data / "clean.fif").resolve()


@pytest.mark.parametrize(("needle", "records", "changes"), [
    ("integer 1", "[]", {"schema": "true"}),
    ("exactly two", "[]", {"arms": "[active]"}),
    ("at least two", "[]", {"sessions": "[baseline]"}),
    ("duplicate values", "[]", {"conditions": "[rest, rest]"}),
    ("undeclared", "[" + _row(arm="other") + "]", {}),
    ("record_id values", "[" + _row() + ", " + _row() + "]", {}),
    ("RecordKey tuples", "[" + _row() + ", " + _row(record_id="r-2") + "]", {}),
    ("multiple arms", "[" + _row() + ", " + _row(record_id="r-2", arm="comparison") + "]", {}),
    ("extension", "[" + _row(path="clean.mat") + "]", {}),
    ("samples_per_epoch", "[" + _row(samples_per_epoch="4") + "]", {}),
    ("must be V", "[" + _row(data_unit="uV") + "]", {}),
])
def test_study_and_record_contract_rejections(tmp_path: Path, needle: str, records: str, changes: dict[str, str]) -> None:
    _source(tmp_path); _source(tmp_path, "clean.mat")
    with pytest.raises(ProjectValidationError, match=needle): _load(tmp_path, _yaml(records, **changes))


def test_duplicate_yaml_unknown_and_missing_record_values(tmp_path: Path) -> None:
    _source(tmp_path)
    with pytest.raises(ProjectValidationError, match="duplicate YAML"):
        _load(tmp_path, "schema_version: 1\nschema_version: 1\n")
    with pytest.raises(ProjectValidationError, match="unknown"):
        _load(tmp_path, _yaml("[" + _row(extra="no") + "]"))
    with pytest.raises(ProjectValidationError, match="missing"):
        _load(tmp_path, _yaml("[{record_id: r-1}]"))


@pytest.mark.parametrize("header", [
    FIELDS + ",extra",
    FIELDS + ",record_id",
    "," + FIELDS,
    FIELDS.replace("path,", ""),
])
def test_manifest_header_contract(tmp_path: Path, header: str) -> None:
    _source(tmp_path); manifest = tmp_path / "records.csv"; manifest.write_text(header + "\n", encoding="utf-8")
    with pytest.raises(ProjectValidationError): _load(tmp_path, _yaml("{manifest: records.csv}"))


def test_profiles_and_later_config_are_recursively_immutable(tmp_path: Path) -> None:
    _source(tmp_path)
    project = _load(tmp_path, _yaml("[" + _row() + "]") + "runtime: {deep: {items: [one]}}\n")
    with pytest.raises(TypeError): project.analysis_profiles["default"]["nested"]["value"] = "two"  # type: ignore[index]
    with pytest.raises(TypeError): project.extras["runtime"]["deep"]["items"] += ("two",)  # type: ignore[index]
