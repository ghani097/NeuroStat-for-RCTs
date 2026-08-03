"""Relocatable, self-contained GraphVar 2.03a workspace construction."""

from __future__ import annotations

import csv
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from xml.etree import ElementTree

import numpy as np

from ..network_models.data import (
    VerifiedConnectivity,
    verify_connectivity_run,
)
from ..project import CleanProject, ProjectRecord, load_project
from ..stage_io import atomic_json, canonical_json, digest_json, load_json, sha256_file
from .config import GraphVarConfig, GraphVarFamily, parse_graphvar_config

SCHEMA = "neurostat.graphvar_workspaces.v1"
_MATLAB_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,62}\Z")
_LAYOUT = (
    "data/Signals",
    "data/CorrMatrix",
    "results/Saved",
    "results/CorrResults",
    "results/CorrMatrix",
    "results/SICEMatrix",
    "results/CorrectedAlpha",
    "results/RandomizedTimeSeries",
    "results/FragCheck",
    "results/GroupCommunity",
    "results/default/RandomizedShuffel",
    "results/default/GraphVars",
    "results/default/GroupCommunity",
)


class GraphVarError(RuntimeError):
    """Raised when GraphVar inputs, execution, or output integrity is unsafe."""


@dataclass(frozen=True, slots=True)
class WorkspacePlan:
    workspace_id: str
    family: str
    holm_family: str
    condition: str
    band: str
    pair: str
    first_session: str
    second_session: str
    seed: int

    def normalized(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "family": self.family,
            "holm_family": self.holm_family,
            "condition": self.condition,
            "band": self.band,
            "pair": self.pair,
            "first_session": self.first_session,
            "second_session": self.second_session,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class GraphVarWorkspaceRun:
    run_id: str
    run_dir: Path
    manifest: Path
    reused: bool
    workspaces: tuple[Path, ...]


def _safe_name(*values: str) -> str:
    return "__".join(values)


def workspace_plans(config: GraphVarConfig) -> tuple[WorkspacePlan, ...]:
    plans = []
    for family in config.families:
        for band in family.bands:
            for pair in family.session_pairs:
                workspace_id = _safe_name(family.name, band, pair.name)
                seed = (
                    family.seed_base
                    + int(digest_json({"workspace": workspace_id})[:8], 16)
                ) % 2147483646 + 1
                plans.append(
                    WorkspacePlan(
                        workspace_id,
                        family.name,
                        family.holm_family,
                        family.condition,
                        band,
                        pair.name,
                        pair.first_session,
                        pair.second_session,
                        seed,
                    )
                )
    return tuple(plans)


def _all_files(root: Path, excluded: set[str]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def _verify_hash_set(
    root: Path, expected: Any, excluded: set[str], description: str
) -> None:
    if not isinstance(expected, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in expected.items()
    ):
        raise GraphVarError(f"{description} manifest hash set is invalid")
    expected = cast(dict[str, str], expected)
    actual = _all_files(root, excluded)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise GraphVarError(
            f"{description} exact file set differs; missing={missing}, extra={extra}"
        )
    for relative, digest in expected.items():
        if actual[relative] != digest:
            raise GraphVarError(f"{description} SHA-256 mismatch: {relative}")


def _xlsx_rows(path: Path) -> list[list[Any]]:
    """Read simple cell values from the first XLSX worksheet using stdlib only."""
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    relations_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    document_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", namespace):
                shared.append(
                    "".join(
                        node.text or "" for node in item.findall(".//x:t", namespace)
                    )
                )
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheet = workbook.find("x:sheets/x:sheet", namespace)
        if sheet is None:
            raise GraphVarError("atlas workbook has no worksheet")
        relation_id = sheet.attrib[f"{{{document_ns}}}id"]
        relationships = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        target = None
        for relation in relationships.findall("r:Relationship", relations_ns):
            if relation.attrib.get("Id") == relation_id:
                target = relation.attrib.get("Target")
        if not target:
            raise GraphVarError(
                "atlas workbook first worksheet relationship is invalid"
            )
        sheet_path = "xl/" + target.lstrip("/").removeprefix("xl/")
        root = ElementTree.fromstring(archive.read(sheet_path))
        rows: list[list[Any]] = []
        previous_row = 0
        for row in root.findall(".//x:sheetData/x:row", namespace):
            try:
                row_number = int(row.attrib["r"])
            except (KeyError, ValueError):
                raise GraphVarError(
                    "atlas workbook has an invalid worksheet row index"
                ) from None
            if row_number <= previous_row:
                raise GraphVarError(
                    "atlas workbook worksheet row indices are duplicate or out of order"
                )
            while len(rows) < row_number - 1:
                rows.append([])
            values: dict[int, Any] = {}
            for cell in row.findall("x:c", namespace):
                reference = cell.attrib.get("r", "")
                letters = "".join(
                    character for character in reference if character.isalpha()
                )
                column = 0
                for character in letters:
                    column = column * 26 + ord(character.upper()) - 64
                kind = cell.attrib.get("t")
                value_node = cell.find("x:v", namespace)
                if kind == "inlineStr":
                    text_node = cell.find("x:is/x:t", namespace)
                    value: Any = "" if text_node is None else text_node.text or ""
                elif value_node is None:
                    value = ""
                elif kind == "s":
                    value = shared[int(value_node.text or "0")]
                else:
                    text = value_node.text or ""
                    try:
                        value = float(text)
                    except ValueError:
                        value = text
                values[column] = value
            width = max(values, default=0)
            rows.append([values.get(column, "") for column in range(1, width + 1)])
            previous_row = row_number
        return rows


def validate_atlas_workbook(config: GraphVarConfig) -> tuple[dict[str, Any], ...]:
    try:
        rows = _xlsx_rows(config.atlas.workbook)
    except GraphVarError:
        raise
    except (
        KeyError,
        ValueError,
        IndexError,
        OSError,
        zipfile.BadZipFile,
        ElementTree.ParseError,
    ) as exc:
        raise GraphVarError(f"atlas workbook structure is invalid: {exc}") from exc
    mapping = []
    selected = []
    for row_index, row in enumerate(rows, 1):
        flag = row[0] if row else ""
        if isinstance(flag, (int, float)) and float(flag) == 1.0:
            selected.append(row_index)
    expected = [node.atlas_index_1based for node in config.atlas.nodes]
    if selected != expected:
        raise GraphVarError(
            "atlas workbook selected-row flags/order differ from configured atlas indices"
        )
    for result_index, node in enumerate(config.atlas.nodes, 1):
        row = rows[node.atlas_index_1based - 1]
        display = str(row[1]).strip() if len(row) > 1 else ""
        if display != node.display_label:
            raise GraphVarError(
                f"atlas display label mismatch at row {node.atlas_index_1based}: "
                f"{display!r} != {node.display_label!r}"
            )
        workbook_coords = None
        if len(row) >= 6 and all(isinstance(item, (int, float)) for item in row[3:6]):
            workbook_coords = tuple(float(item) for item in row[3:6])
        if node.coordinates is not None and workbook_coords != node.coordinates:
            raise GraphVarError(
                f"atlas coordinates mismatch at row {node.atlas_index_1based}"
            )
        mapping.append(
            {
                "result_index_1based": result_index,
                "atlas_index_1based": node.atlas_index_1based,
                "connectivity_label": node.connectivity_label,
                "display_label": node.display_label,
                "coordinates": None
                if workbook_coords is None
                else list(workbook_coords),
            }
        )
    return tuple(mapping)


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
    delimiter: str = ",",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter=delimiter, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_mat(path: Path, content: dict[str, Any], *, compress: bool = False) -> None:
    from scipy.io import savemat

    savemat(path, content, do_compression=compress, oned_as="row")
    # scipy's MAT v5 header embeds wall-clock time. Freeze it for reproducible hashes.
    with path.open("r+b") as handle:
        header = b"MATLAB 5.0 MAT-file, NeuroStat deterministic GraphVar workspace"
        handle.write(header.ljust(116, b" "))


def _family_cohort(
    project: CleanProject, family: GraphVarFamily
) -> tuple[tuple[str, ...], dict[tuple[str, str], ProjectRecord], list[dict[str, Any]]]:
    sessions = tuple(
        dict.fromkeys(
            session
            for pair in family.session_pairs
            for session in (pair.first_session, pair.second_session)
        )
    )
    records = [
        record for record in project.records if record.key.condition == family.condition
    ]
    subjects = sorted({record.key.participant_id for record in project.records})
    global_arms: dict[str, set[str]] = {}
    for record in project.records:
        global_arms.setdefault(record.key.participant_id, set()).add(record.key.arm)
    lookup: dict[tuple[str, str], ProjectRecord] = {}
    cohort_rows = []
    included = []
    for subject in subjects:
        subject_records = [
            record for record in records if record.key.participant_id == subject
        ]
        by_session: dict[str, list[ProjectRecord]] = {
            session: [
                record for record in subject_records if record.key.session == session
            ]
            for session in sessions
        }
        duplicate = [
            session for session, values in by_session.items() if len(values) > 1
        ]
        if duplicate:
            raise GraphVarError(
                f"{family.name}: duplicate subject/session records for {subject}: {duplicate}"
            )
        missing = [session for session, values in by_session.items() if not values]
        arms = global_arms[subject]
        if len(arms) != 1:
            raise GraphVarError(
                f"{family.name}: participant {subject!r} has ambiguous arm"
            )
        is_included = not missing
        if is_included:
            included.append(subject)
            for session in sessions:
                lookup[(subject, session)] = by_session[session][0]
        cohort_rows.append(
            {
                "family": family.name,
                "condition": family.condition,
                "subject": subject,
                "arm": next(iter(arms)),
                "included": str(is_included).lower(),
                "available_sessions": ";".join(
                    session for session in sessions if by_session[session]
                ),
                "missing_sessions": ";".join(missing),
                "reason": "complete_family_cohort"
                if is_included
                else "missing_family_session",
            }
        )
    arm_counts = {
        arm: sum(
            1 for subject in included if lookup[(subject, sessions[0])].key.arm == arm
        )
        for arm in project.study.arms
    }
    if any(count == 0 for count in arm_counts.values()):
        raise GraphVarError(f"{family.name}: common cohort has an empty configured arm")
    return tuple(included), lookup, cohort_rows


def _validate_record_labels(
    upstream: VerifiedConnectivity, record: ProjectRecord, config: GraphVarConfig
) -> None:
    item = upstream.records[record.record_id]
    labels = item.connectivity_header.labels
    for node in config.atlas.nodes:
        index = node.atlas_index_1based - 1
        if index >= len(labels) or labels[index] != node.connectivity_label:
            raise GraphVarError(
                f"{record.record_id}: configured atlas index/label differs from connectivity: "
                f"{node.atlas_index_1based}/{node.connectivity_label!r}"
            )


def _validate_workspace(workspace: Path, entry: dict[str, Any]) -> None:
    manifest = load_json(
        workspace / "workspace-manifest.json", "GraphVar workspace manifest"
    )
    if manifest != entry or manifest.get("state") != "prepared":
        raise GraphVarError(f"workspace manifest identity differs: {workspace.name}")
    _verify_hash_set(
        workspace,
        manifest.get("artifacts"),
        {"workspace-manifest.json"},
        f"workspace {workspace.name}",
    )
    from scipy.io import loadmat

    settings = loadmat(workspace / "Workspace.mat", simplify_cells=True)
    expected_settings = {
        "brainSheet": "atlas.xlsx",
        "variableSheet": "Variables.csv",
        "fieldName": manifest["matrix_field"],
    }
    for name, expected in expected_settings.items():
        actual = str(settings.get(name, ""))
        if actual != expected:
            raise GraphVarError(f"workspace {name} differs: {workspace.name}")
    data_path = Path(str(settings.get("data_path", "")))
    if not data_path.is_absolute() or data_path.name.casefold() != "data":
        raise GraphVarError(f"workspace data_path is invalid: {workspace.name}")
    relative_files = [str(value) for value in np.atleast_1d(settings["files_rel"])]
    absolute_files = [str(value) for value in np.atleast_1d(settings["files"])]
    if (
        len(relative_files) != manifest["observations"]
        or len(absolute_files) != manifest["observations"]
    ):
        raise GraphVarError(f"workspace file cardinality differs: {workspace.name}")
    for relative, absolute in zip(relative_files, absolute_files, strict=True):
        relative_path = Path(relative)
        if relative_path.name != relative or relative in {"", ".", ".."}:
            raise GraphVarError(
                f"workspace files_rel entry is not a bare filename: {relative!r}"
            )
        resolved_relative = (workspace / "data" / "CorrMatrix" / relative).resolve()
        try:
            resolved_relative.relative_to(workspace.resolve())
        except ValueError:
            raise GraphVarError(
                f"workspace contains an unsafe relative matrix reference: {workspace.name}"
            ) from None
        if not resolved_relative.is_file():
            raise GraphVarError(f"workspace matrix reference is stale: {relative}")
        absolute_carrier = Path(absolute)
        expected_suffix = ("data", "corrmatrix", relative.casefold())
        absolute_parts = tuple(part.casefold() for part in absolute_carrier.parts)
        if (
            not absolute_carrier.is_absolute()
            or absolute_parts[-len(expected_suffix) :] != expected_suffix
        ):
            raise GraphVarError(
                f"workspace absolute carrier disagrees with files_rel: {relative}"
            )


def verify_graphvar_workspace_run(
    run_dir: str | Path, expected_lock: dict[str, Any] | None = None
) -> GraphVarWorkspaceRun:
    run_dir = Path(run_dir).resolve()
    lock = load_json(run_dir / "project.lock.json", "GraphVar workspace lock")
    manifest = load_json(run_dir / "manifest.json", "GraphVar workspace run manifest")
    if expected_lock is not None and canonical_json(lock) != canonical_json(
        expected_lock
    ):
        raise GraphVarError(
            "GraphVar workspace lock differs from the requested identity"
        )
    if (
        not isinstance(lock, dict)
        or lock.get("stage") != "graphvar_workspaces"
        or not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or manifest.get("run_id") != lock.get("run_id")
        or manifest.get("state") != "prepared"
        or not isinstance(manifest.get("workspaces"), list)
    ):
        raise GraphVarError("GraphVar workspace run identity is invalid or non-final")
    for entry in manifest["workspaces"]:
        if not isinstance(entry, dict) or not isinstance(
            entry.get("workspace_id"), str
        ):
            raise GraphVarError("GraphVar workspace manifest entry is invalid")
        _validate_workspace(run_dir / "workspaces" / entry["workspace_id"], entry)
    _verify_hash_set(
        run_dir, manifest.get("artifacts"), {"manifest.json"}, "workspace run"
    )
    paths = tuple(
        run_dir / "workspaces" / entry["workspace_id"]
        for entry in manifest["workspaces"]
    )
    return GraphVarWorkspaceRun(
        str(lock["run_id"]), run_dir, run_dir / "manifest.json", True, paths
    )


def build_graphvar_workspaces(
    project: CleanProject | str | Path,
    connectivity_run: str | Path,
) -> GraphVarWorkspaceRun:
    """Build every planned band-by-pair workspace from a verified Phase-3 run."""
    project = (
        load_project(project) if not isinstance(project, CleanProject) else project
    )
    config = parse_graphvar_config(project)
    if (
        sha256_file(config.atlas.workbook) != config.atlas.workbook_sha256
        or sha256_file(config.atlas.brain_viewer_config)
        != config.atlas.brain_viewer_config_sha256
    ):
        raise GraphVarError(
            "configured GraphVar atlas/viewer resource changed after parsing"
        )
    upstream = verify_connectivity_run(project, connectivity_run)
    try:
        mapping = validate_atlas_workbook(config)
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise GraphVarError(f"atlas workbook is invalid: {exc}") from exc
    identity = {
        "stage": "graphvar_workspaces",
        "project": project.normalized(),
        "config": config.normalized(),
        "upstream": {
            "run_id": upstream.run_id,
            "scientific_identity_sha256": upstream.stable_identity_sha256,
            "manifest": upstream.stable_identity,
        },
        "plans": [plan.normalized() for plan in workspace_plans(config)],
    }
    run_id = digest_json(identity)[:12]
    lock = identity | {"run_id": run_id}
    runs_root = project.output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    final = (runs_root / f"graphvar_workspaces-{run_id}").resolve()
    if final.exists():
        return verify_graphvar_workspace_run(final, lock)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".graphvar_workspaces-{run_id}.staging-", dir=runs_root
        )
    ).resolve()
    try:
        atomic_json(staging / "project.lock.json", lock)
        shutil.copy2(config.atlas.workbook, staging / "atlas.xlsx")
        shutil.copy2(
            config.atlas.brain_viewer_config, staging / "ConfigBrainViewer.mat"
        )
        if (
            sha256_file(staging / "atlas.xlsx") != config.atlas.workbook_sha256
            or sha256_file(staging / "ConfigBrainViewer.mat")
            != config.atlas.brain_viewer_config_sha256
        ):
            raise GraphVarError("copied GraphVar atlas/viewer resource hash differs")
        atomic_json(staging / "selected-node-map.json", list(mapping))
        family_lookup = {family.name: family for family in config.families}
        cohorts: dict[
            str, tuple[tuple[str, ...], dict[tuple[str, str], ProjectRecord]]
        ] = {}
        cohort_rows: list[dict[str, Any]] = []
        for family in config.families:
            subjects, lookup, rows = _family_cohort(project, family)
            cohorts[family.name] = (subjects, lookup)
            cohort_rows.extend(rows)
        _write_csv(
            staging / "cohort-inclusion.csv",
            cohort_rows,
            (
                "family",
                "condition",
                "subject",
                "arm",
                "included",
                "available_sessions",
                "missing_sessions",
                "reason",
            ),
        )
        entries = []
        for plan in workspace_plans(config):
            family = family_lookup[plan.family]
            subjects, lookup = cohorts[family.name]
            workspace_final = final / "workspaces" / plan.workspace_id
            workspace_staging = staging / "workspaces" / plan.workspace_id
            for relative in _LAYOUT:
                (workspace_staging / relative).mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging / "atlas.xlsx", workspace_staging / "atlas.xlsx")
            shutil.copy2(
                staging / "ConfigBrainViewer.mat",
                workspace_staging / "ConfigBrainViewer.mat",
            )
            atomic_json(workspace_staging / "selected-node-map.json", list(mapping))
            variables: list[dict[str, Any]] = []
            audits: list[dict[str, Any]] = []
            files_relative: list[str] = []
            files_absolute: list[str] = []
            field_name = family.matrix_field_template.format(band=plan.band)
            if not _MATLAB_IDENTIFIER.fullmatch(field_name):
                raise GraphVarError(
                    f"formatted matrix field is not a MATLAB identifier: {field_name!r}"
                )
            for subject in subjects:
                for pair_order, session in enumerate(
                    (plan.first_session, plan.second_session), 1
                ):
                    record = lookup[(subject, session)]
                    verified = upstream.records[record.record_id]
                    _validate_record_labels(upstream, record, config)
                    try:
                        band_index = verified.connectivity_header.bands.index(plan.band)
                    except ValueError:
                        raise GraphVarError(
                            f"{record.record_id}: requested band {plan.band!r} is absent"
                        ) from None
                    matrix = np.asarray(verified.matrix[band_index], dtype=np.float64)
                    if (
                        matrix.ndim != 2
                        or matrix.shape[0] != matrix.shape[1]
                        or not np.isfinite(matrix).all()
                        or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)
                        or not np.allclose(np.diag(matrix), 0, atol=1e-12, rtol=0)
                        or matrix.min() < -1e-12
                        or matrix.max() > 1 + 1e-12
                    ):
                        raise GraphVarError(
                            f"{record.record_id}: invalid full connectivity matrix"
                        )
                    file_name = (
                        f"{len(variables) + 1:04d}__{record.record_id}__{plan.band}.mat"
                    )
                    relative = f"data/CorrMatrix/{file_name}"
                    target = workspace_staging / relative
                    _write_mat(target, {field_name: matrix}, compress=True)
                    output_sha = sha256_file(target)
                    variables.append(
                        {
                            "ObservationID": f"{subject}__{session}",
                            "arm": record.key.arm,
                            "subject": subject,
                            "session": session,
                        }
                    )
                    files_relative.append(file_name)
                    files_absolute.append(
                        str(workspace_final / "data" / "CorrMatrix" / file_name)
                    )
                    audits.append(
                        {
                            "row_order": len(variables),
                            "pair_order": pair_order,
                            "record_id": record.record_id,
                            "record_sha256": verified.connectivity_sha256,
                            "payload_digest": verified.connectivity_header.payload_digest,
                            "band": plan.band,
                            "field_name": field_name,
                            "workspace_matrix": relative,
                            "workspace_sha256": output_sha,
                        }
                    )
            _write_csv(
                workspace_staging / "Variables.csv",
                variables,
                ("ObservationID", "arm", "subject", "session"),
                delimiter=";",
            )
            _write_csv(
                workspace_staging / "matrix-audit.csv",
                audits,
                (
                    "row_order",
                    "pair_order",
                    "record_id",
                    "record_sha256",
                    "payload_digest",
                    "band",
                    "field_name",
                    "workspace_matrix",
                    "workspace_sha256",
                ),
            )
            _write_mat(
                workspace_staging / "Workspace.mat",
                {
                    "data_path": str(workspace_final / "data"),
                    "brainSheet": "atlas.xlsx",
                    "variableSheet": "Variables.csv",
                    "fieldName": field_name,
                    "filename_start": np.asarray([[1.0]]),
                    "filename_end": np.asarray([[40.0]]),
                    "partVar": np.asarray([[1.0]]),
                    "corrVar": np.asarray([[2.0]]),
                    "files": np.asarray(files_absolute, dtype=object).reshape(1, -1),
                    "files_rel": np.asarray(files_relative, dtype=object).reshape(
                        1, -1
                    ),
                },
                compress=True,
            )
            _write_mat(
                workspace_staging / "ImportSettings.mat",
                {
                    "useVar": np.asarray([[0.0, 1.0, 0.0, 0.0]]),
                    "userVar": np.asarray([[1.0]]),
                },
            )
            _write_mat(
                workspace_staging / "results" / "InterimResultsID.mat",
                {"InterimResultsID": np.asarray([[0.0]])},
            )
            request = plan.normalized() | {
                "numerator_arm": family.numerator_arm,
                "denominator_arm": family.denominator_arm,
                "effect_label": family.effect_label,
                "component_forming_alpha": family.component_forming_alpha,
                "familywise_alpha": family.familywise_alpha,
                "permutations": family.permutations,
                "matrix_field": field_name,
                "observations": len(variables),
                "subjects": len(subjects),
                "nodes": len(config.atlas.nodes),
                "links": len(config.atlas.nodes) * (len(config.atlas.nodes) - 1) // 2,
            }
            atomic_json(workspace_staging / "request.json", request)
            entry = request | {
                "state": "prepared",
                "artifacts": _all_files(workspace_staging, {"workspace-manifest.json"}),
            }
            atomic_json(workspace_staging / "workspace-manifest.json", entry)
            entries.append(entry)
        manifest = {
            "schema": SCHEMA,
            "run_id": run_id,
            "state": "prepared",
            "workspaces": entries,
        }
        manifest["artifacts"] = _all_files(staging, {"manifest.json"})
        atomic_json(staging / "manifest.json", manifest)
        if final.exists():
            raise GraphVarError(
                f"GraphVar workspace run appeared concurrently: {final}"
            )
        os.replace(staging, final)
        verified = verify_graphvar_workspace_run(final, lock)
        return GraphVarWorkspaceRun(
            verified.run_id,
            verified.run_dir,
            verified.manifest,
            False,
            verified.workspaces,
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
