"""Resumable execution/import, extraction, validation, and immutable finalization."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..project import CleanProject, load_project
from ..stage_io import atomic_json, canonical_json, digest_json, load_json, sha256_file
from .config import GraphVarConfig, parse_graphvar_config
from .extract import apply_condition_holm, extract_workspace_result
from .runtime import (
    GraphVarRuntimeMetadata,
    MatlabExecution,
    preflight_graphvar,
    run_graphvar_workspace,
)
from .workspace import (
    GraphVarError,
    GraphVarWorkspaceRun,
    _all_files,
    _verify_hash_set,
    _write_mat,
    verify_graphvar_workspace_run,
)

SCHEMA = "neurostat.graphvar_analysis.v1"
COMPONENT_FIELDS = (
    "holm_family",
    "family",
    "condition",
    "workspace",
    "band",
    "session_pair",
    "component_id",
    "effect_label",
    "effect_orientation_multiplier",
    "component_node_count",
    "component_link_count",
    "member_rois",
    "component_forming_alpha",
    "permutations",
    "primary_node_exceedances",
    "primary_node_pfwe_graphvar_tail",
    "primary_node_pfwe_plus_one",
    "primary_node_pfwe_supported",
    "posthoc_nbs_edge_exceedances",
    "posthoc_nbs_edge_pfwe_graphvar_tail",
    "posthoc_nbs_edge_pfwe_plus_one",
    "posthoc_nbs_edge_pfwe_supported",
    "graphvar_observation_rows",
    "graphvar_numerator_df",
    "graphvar_denominator_df",
    "inference_scope",
    "primary_node_holm_all_candidates",
    "primary_studywide_supported",
    "posthoc_nbs_edge_holm_all_candidates",
    "posthoc_nbs_studywide_supported",
)
LINK_FIELDS = (
    "holm_family",
    "condition",
    "workspace",
    "band",
    "session_pair",
    "component_id",
    "component_node_count",
    "component_link_count",
    "roi1_result_index_1based",
    "roi2_result_index_1based",
    "roi1_atlas_index_1based",
    "roi2_atlas_index_1based",
    "roi1_connectivity_label",
    "roi2_connectivity_label",
    "roi1",
    "roi2",
    "effect_label",
    "paired_change_contrast_b",
    "f_statistic",
    "parametric_p_uncorrected",
    "permutation_p_uncorrected",
    "effect_direction",
    "inference_scope",
    "primary_node_pfwe_plus_one",
    "primary_node_holm_all_candidates",
    "primary_studywide_supported",
    "posthoc_nbs_edge_pfwe_plus_one",
    "posthoc_nbs_edge_holm_all_candidates",
    "posthoc_nbs_studywide_supported",
)


@dataclass(frozen=True, slots=True)
class GraphVarAnalysisRun:
    run_id: str
    run_dir: Path
    manifest: Path
    reused: bool
    artifacts: tuple[Path, ...]


WorkspaceRunner = Callable[
    [GraphVarConfig, Path, dict[str, Any], Path],
    MatlabExecution,
]
Preflight = Callable[[GraphVarConfig, Path], GraphVarRuntimeMetadata]


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | None = None
) -> None:
    if fields is None:
        ordered: list[str] = []
        for row in rows:
            for name in row:
                if name not in ordered:
                    ordered.append(name)
        fields = tuple(ordered)
    if not fields:
        raise GraphVarError("CSV schema must be explicit even for zero rows")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _analysis_input_files(workspace: Path) -> dict[str, str]:
    result_root = (workspace / "results" / "CorrResults").resolve()
    result = {}
    for path in sorted(workspace.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace).as_posix()
        if relative == "analysis-input-manifest.json":
            continue
        parts = path.relative_to(workspace).parts
        if (
            len(parts) >= 2
            and parts[0] == "results"
            and parts[1].startswith(".neurostat-CorrResults-attempt-")
        ):
            continue
        try:
            path.resolve().relative_to(result_root)
            continue
        except ValueError:
            pass
        result[relative] = sha256_file(path)
    return result


def _copy_prepared_workspace(source: Path, target: Path, final_target: Path) -> None:
    shutil.copytree(source, target)
    import numpy as np
    from scipy.io import loadmat

    settings = loadmat(target / "Workspace.mat", simplify_cells=True)
    relative_files = [str(item) for item in np.atleast_1d(settings["files_rel"])]
    absolute_files = [str(final_target / item) for item in relative_files]
    _write_mat(
        target / "Workspace.mat",
        {
            "data_path": str(final_target / "data"),
            "brainSheet": str(settings["brainSheet"]),
            "variableSheet": str(settings["variableSheet"]),
            "fieldName": str(settings["fieldName"]),
            "filename_start": np.asarray([[float(settings["filename_start"])]]),
            "filename_end": np.asarray([[float(settings["filename_end"])]]),
            "partVar": np.asarray([[float(settings["partVar"])]]),
            "corrVar": np.asarray([[float(settings["corrVar"])]]),
            "files": np.asarray(absolute_files, dtype=object).reshape(1, -1),
            "files_rel": np.asarray(relative_files, dtype=object).reshape(1, -1),
        },
        compress=True,
    )
    (target / "workspace-manifest.json").rename(
        target / "prepared-workspace-manifest.json"
    )
    input_hashes = _analysis_input_files(target)
    atomic_json(
        target / "analysis-input-manifest.json",
        {
            "schema": "neurostat.graphvar_analysis_inputs.v1",
            "artifacts": input_hashes,
        },
    )


def _verify_analysis_inputs(workspace: Path) -> None:
    manifest = load_json(
        workspace / "analysis-input-manifest.json", "GraphVar analysis input manifest"
    )
    actual = _analysis_input_files(workspace)
    expected = manifest.get("artifacts")
    # The original manifest included itself neither before nor after writing.
    if not isinstance(expected, dict) or actual != expected:
        raise GraphVarError(
            f"analysis workspace inputs were modified: {workspace.name}"
        )


def _runtime_carriers(preflight_dir: Path) -> tuple[Path, ...]:
    required = (
        "request.json",
        "preflight.json",
        "preflight.stdout.log",
        "preflight.stderr.log",
    )
    paths = tuple(preflight_dir / name for name in required)
    if any(not path.is_file() for path in paths):
        raise GraphVarError(
            "GraphVar preflight did not produce its complete carrier set"
        )
    return paths


def _preflight_bundle(
    config: GraphVarConfig,
    runs_root: Path,
    preflight_function: Preflight | None,
) -> tuple[GraphVarRuntimeMetadata, dict[str, bytes]]:
    """Run preflight in a unique directory that is always cleaned."""
    with tempfile.TemporaryDirectory(
        prefix=".graphvar-preflight-", dir=runs_root
    ) as name:
        directory = Path(name)
        runtime = (
            preflight_graphvar(config, directory)
            if preflight_function is None
            else preflight_function(config, directory)
        )
        carriers = {
            path.name: path.read_bytes() for path in _runtime_carriers(directory)
        }
        return runtime, carriers


def _cleanup_attempts(workspace: Path) -> None:
    results = (workspace / "results").resolve()
    if not results.is_dir():
        raise GraphVarError(f"workspace results directory is missing: {workspace.name}")
    for candidate in results.glob(".neurostat-CorrResults-attempt-*"):
        resolved = candidate.resolve()
        try:
            resolved.relative_to(results)
        except ValueError:
            raise GraphVarError(f"unsafe GraphVar attempt path: {candidate}") from None
        if not resolved.is_dir():
            raise GraphVarError(
                f"GraphVar attempt path is not a directory: {candidate}"
            )
        shutil.rmtree(resolved)


def verify_graphvar_analysis_run(
    run_dir: str | Path, expected_lock: dict[str, Any] | None = None
) -> GraphVarAnalysisRun:
    run_dir = Path(run_dir).resolve()
    lock = load_json(run_dir / "project.lock.json", "GraphVar analysis lock")
    manifest = load_json(
        run_dir / "graphvar" / "manifest.json", "GraphVar analysis manifest"
    )
    if expected_lock is not None and canonical_json(lock) != canonical_json(
        expected_lock
    ):
        raise GraphVarError(
            "existing GraphVar analysis lock differs from requested identity"
        )
    if (
        not isinstance(lock, dict)
        or lock.get("stage") != "graphvar"
        or not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or manifest.get("run_id") != lock.get("run_id")
        or manifest.get("scientific_identity_sha256")
        != digest_json({key: value for key, value in lock.items() if key != "run_id"})
    ):
        raise GraphVarError("GraphVar analysis identity is invalid")
    _verify_hash_set(
        run_dir,
        manifest.get("artifacts"),
        {"graphvar/manifest.json"},
        "GraphVar analysis",
    )
    status = load_json(run_dir / "graphvar" / "status.json", "GraphVar analysis status")
    validation = load_json(
        run_dir / "graphvar" / "validation.json", "GraphVar analysis validation"
    )
    if status.get("state") != "succeeded" or validation.get("status") != "passed":
        raise GraphVarError("GraphVar analysis is not successfully validated")
    artifacts = tuple(run_dir / relative for relative in sorted(manifest["artifacts"]))
    return GraphVarAnalysisRun(
        str(lock["run_id"]),
        run_dir,
        run_dir / "graphvar" / "manifest.json",
        True,
        artifacts,
    )


def _link_holm(components: list[dict[str, Any]], links: list[dict[str, Any]]) -> None:
    by_id = {row["component_id"]: row for row in components}
    if len(by_id) != len(components):
        raise GraphVarError("component identifiers are not unique")
    for link in links:
        component = by_id.get(link["component_id"])
        if component is None:
            raise GraphVarError("member link does not map to an extracted component")
        link["primary_node_pfwe_plus_one"] = component["primary_node_pfwe_plus_one"]
        link["primary_node_holm_all_candidates"] = component[
            "primary_node_holm_all_candidates"
        ]
        link["primary_studywide_supported"] = component["primary_studywide_supported"]
        link["posthoc_nbs_edge_pfwe_plus_one"] = component[
            "posthoc_nbs_edge_pfwe_plus_one"
        ]
        link["posthoc_nbs_edge_holm_all_candidates"] = component[
            "posthoc_nbs_edge_holm_all_candidates"
        ]
        link["posthoc_nbs_studywide_supported"] = component[
            "posthoc_nbs_studywide_supported"
        ]


def _workspace_completion_valid(
    workspace: Path, request: dict[str, Any], node_map: tuple[dict[str, Any], ...]
) -> bool:
    completion_path = workspace / "results" / "CorrResults" / "completion.json"
    if not completion_path.is_file():
        return False
    output = completion_path.parent
    output_manifest = load_json(
        output / "output-manifest.json", "GraphVar workspace output manifest"
    )
    _verify_hash_set(
        output,
        output_manifest.get("artifacts"),
        {"output-manifest.json"},
        f"GraphVar workspace output {workspace.name}",
    )
    completion = load_json(completion_path, "GraphVar workspace completion")
    result = workspace / "results" / "CorrResults" / "corr_area_1_Group.mat"
    if completion.get("state") != "succeeded" or completion.get("result_sha256") != (
        sha256_file(result) if result.is_file() else None
    ):
        raise GraphVarError(f"completed GraphVar result was modified: {workspace.name}")
    extract_workspace_result(result, request, node_map)
    return True


def run_graphvar_analysis(
    project: CleanProject | str | Path,
    workspace_run: GraphVarWorkspaceRun | str | Path,
    *,
    result_files: Mapping[str, str | Path] | None = None,
    workspace_runner: WorkspaceRunner | None = None,
    preflight_function: Preflight | None = None,
) -> GraphVarAnalysisRun:
    """Execute or safely import all results, then freeze all-candidate inference."""
    project = (
        load_project(project) if not isinstance(project, CleanProject) else project
    )
    config = parse_graphvar_config(project)
    prepared = (
        verify_graphvar_workspace_run(workspace_run.run_dir)
        if isinstance(workspace_run, GraphVarWorkspaceRun)
        else verify_graphvar_workspace_run(workspace_run)
    )
    prepared_manifest = load_json(prepared.manifest, "GraphVar workspace manifest")
    prepared_lock = load_json(
        prepared.run_dir / "project.lock.json", "GraphVar workspace lock"
    )
    if (
        canonical_json(prepared_lock.get("project"))
        != canonical_json(project.normalized())
        or canonical_json(prepared_lock.get("config"))
        != canonical_json(config.normalized())
        or canonical_json(prepared_lock.get("plans"))
        != canonical_json(
            [
                {
                    key: entry[key]
                    for key in (
                        "workspace_id",
                        "family",
                        "holm_family",
                        "condition",
                        "band",
                        "pair",
                        "first_session",
                        "second_session",
                        "seed",
                    )
                }
                for entry in prepared_manifest["workspaces"]
            ]
        )
    ):
        raise GraphVarError(
            "prepared GraphVar workspace run does not match the current project/config/plans"
        )
    workspace_entries = prepared_manifest["workspaces"]
    expected_ids = {entry["workspace_id"] for entry in workspace_entries}
    import_hashes = None
    if result_files is not None:
        if set(result_files) != expected_ids:
            raise GraphVarError(
                "manual result mapping must cover every planned workspace exactly"
            )
        import_hashes = {}
        for name, source in result_files.items():
            path = Path(source).resolve()
            if not path.is_file():
                raise GraphVarError(f"manual GraphVar result is not a file: {path}")
            import_hashes[name] = sha256_file(path)
    runs_root = project.output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    runtime, preflight_carriers = _preflight_bundle(
        config, runs_root, preflight_function
    )
    identity = {
        "stage": "graphvar",
        "project": project.normalized(),
        "config": config.normalized(),
        "workspace_run_id": prepared.run_id,
        "workspace_manifest_sha256": sha256_file(prepared.manifest),
        "runtime": runtime.normalized(),
        "mode": "import" if result_files is not None else "execute",
        "import_result_sha256": import_hashes,
        "manual_import_binding_scope": (
            None
            if result_files is None
            else "The copied result is hash-bound and structurally validated against the "
            "configured request; an external file alone cannot cryptographically "
            "prove which historical workspace produced it."
        ),
    }
    scientific_sha = digest_json(identity)
    run_id = scientific_sha[:12]
    lock = identity | {"run_id": run_id}
    final = (runs_root / f"graphvar-{run_id}").resolve()
    if final.exists():
        return verify_graphvar_analysis_run(final, lock)
    partial = (runs_root / f".graphvar-{run_id}.partial").resolve()
    graphvar_dir = partial / "graphvar"
    if partial.exists():
        existing_lock = load_json(
            partial / "project.lock.json", "partial GraphVar lock"
        )
        if canonical_json(existing_lock) != canonical_json(lock):
            raise GraphVarError(
                "partial GraphVar run lock differs from requested identity"
            )
    else:
        initializing = Path(
            tempfile.mkdtemp(prefix=f".graphvar-{run_id}.initializing-", dir=runs_root)
        ).resolve()
        try:
            initializing_graphvar = initializing / "graphvar"
            initializing_graphvar.mkdir(parents=True)
            atomic_json(initializing / "project.lock.json", lock)
            atomic_json(
                initializing_graphvar / "status.json",
                {"state": "running", "completed": 0, "total": len(workspace_entries)},
            )
            resources = initializing_graphvar / "matlab"
            resources.mkdir()
            shutil.copy2(
                Path(__file__).parent / "matlab" / "neurostat_graphvar_entry.m",
                resources / "neurostat_graphvar_entry.m",
            )
            runtime_dir = initializing_graphvar / "runtime"
            runtime_dir.mkdir()
            for name, content in preflight_carriers.items():
                (runtime_dir / name).write_bytes(content)
            shutil.copy2(
                prepared.run_dir / "selected-node-map.json",
                initializing_graphvar / "selected-node-map.json",
            )
            for entry in workspace_entries:
                source = prepared.run_dir / "workspaces" / entry["workspace_id"]
                target = initializing_graphvar / "workspaces" / entry["workspace_id"]
                final_target = final / "graphvar" / "workspaces" / entry["workspace_id"]
                _copy_prepared_workspace(source, target, final_target)
            os.replace(initializing, partial)
        except BaseException:
            if initializing.exists():
                shutil.rmtree(initializing)
            raise
        graphvar_dir = partial / "graphvar"
    node_map = tuple(
        load_json(graphvar_dir / "selected-node-map.json", "selected-node mapping")
    )
    try:
        completed = 0
        for entry in workspace_entries:
            name = entry["workspace_id"]
            workspace = graphvar_dir / "workspaces" / name
            _cleanup_attempts(workspace)
            _verify_analysis_inputs(workspace)
            output = workspace / "results" / "CorrResults"
            output.mkdir(parents=True, exist_ok=True)
            if _workspace_completion_valid(workspace, entry, node_map):
                completed += 1
                continue
            if any(output.iterdir()):
                raise GraphVarError(
                    f"unverified GraphVar output exists and will not be overwritten: {output}"
                )
            output.rmdir()
            attempt = Path(
                tempfile.mkdtemp(
                    prefix=".neurostat-CorrResults-attempt-",
                    dir=workspace / "results",
                )
            )
            result = attempt / "corr_area_1_Group.mat"
            if result_files is not None:
                assert import_hashes is not None
                source = Path(result_files[name]).resolve()
                temporary = attempt / ".corr_area_1_Group.mat.importing"
                shutil.copy2(source, temporary)
                if sha256_file(temporary) != import_hashes[name]:
                    raise GraphVarError(
                        f"manual GraphVar result changed during import: {name}"
                    )
                os.replace(temporary, result)
                (attempt / "import.stdout.log").write_text("", encoding="utf-8")
                (attempt / "import.stderr.log").write_text("", encoding="utf-8")
                atomic_json(
                    attempt / "import.json",
                    {
                        "mode": "verified_manual_import",
                        "source_sha256": import_hashes[name],
                        "result_sha256": sha256_file(result),
                        "binding_scope": (
                            "Hash-bound copy plus configured structural/effect/shape checks; "
                            "the external file does not cryptographically prove its source workspace."
                        ),
                    },
                )
            else:
                runner = workspace_runner or (
                    lambda cfg, ws, req, out: run_graphvar_workspace(cfg, ws, req, out)
                )
                execution = runner(config, workspace, entry, attempt)
                if execution.returncode != 0:
                    raise GraphVarError(f"GraphVar runner returned nonzero for {name}")
            extracted = extract_workspace_result(result, entry, node_map)
            np_node = extracted.null_node_max
            np_edge = extracted.null_edge_max
            import numpy as np

            np.save(attempt / "null-node-max.npy", np_node, allow_pickle=False)
            np.save(attempt / "null-edge-max.npy", np_edge, allow_pickle=False)
            atomic_json(
                attempt / "completion.json",
                {
                    "state": "succeeded",
                    "result_sha256": sha256_file(result),
                    "status": extracted.status,
                    "candidate_components": len(extracted.components),
                },
            )
            atomic_json(
                attempt / "output-manifest.json",
                {
                    "schema": "neurostat.graphvar_workspace_outputs.v1",
                    "artifacts": _all_files(attempt, {"output-manifest.json"}),
                },
            )
            _verify_hash_set(
                attempt,
                load_json(
                    attempt / "output-manifest.json",
                    "GraphVar attempt output manifest",
                )["artifacts"],
                {"output-manifest.json"},
                f"GraphVar attempt output {name}",
            )
            _verify_analysis_inputs(workspace)
            os.replace(attempt, output)
            completed += 1
            atomic_json(
                graphvar_dir / "status.json",
                {
                    "state": "running",
                    "completed": completed,
                    "total": len(workspace_entries),
                },
            )

        statuses: list[dict[str, Any]] = []
        components: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        for entry in workspace_entries:
            workspace = graphvar_dir / "workspaces" / entry["workspace_id"]
            result = workspace / "results" / "CorrResults" / "corr_area_1_Group.mat"
            extracted = extract_workspace_result(result, entry, node_map)
            statuses.append(extracted.status)
            components.extend(dict(row) for row in extracted.components)
            links.extend(dict(row) for row in extracted.links)
        alpha_by_holm: dict[str, float] = {}
        for family in config.families:
            previous = alpha_by_holm.setdefault(
                family.holm_family, family.familywise_alpha
            )
            if previous != family.familywise_alpha:
                raise GraphVarError("condition-level Holm family alpha differs")
        apply_condition_holm(components, alpha_by_holm)
        _link_holm(components, links)
        _write_csv(graphvar_dir / "workspace-status.csv", statuses)
        _write_csv(graphvar_dir / "components-all.csv", components, COMPONENT_FIELDS)
        _write_csv(graphvar_dir / "component-member-links-all.csv", links, LINK_FIELDS)
        validation = {
            "status": "passed",
            "planned_workspaces": len(workspace_entries),
            "complete_workspaces": len(statuses),
            "no_component_workspaces": sum(
                row["status"] == "complete_no_component" for row in statuses
            ),
            "candidate_components": len(components),
            "member_links": len(links),
            "primary_workspace_supported": sum(
                bool(row["primary_node_pfwe_supported"]) for row in components
            ),
            "primary_holm_supported": sum(
                bool(row["primary_studywide_supported"]) for row in components
            ),
            "posthoc_nbs_workspace_supported": sum(
                bool(row["posthoc_nbs_edge_pfwe_supported"]) for row in components
            ),
            "posthoc_nbs_holm_supported": sum(
                bool(row["posthoc_nbs_studywide_supported"]) for row in components
            ),
            "inference_scope": (
                "Primary study-wide support comes only from node-count pFWE and "
                "all-candidate condition-level Holm. Edge-extent NBS is a separate "
                "post-hoc result family. Individual links are descriptive."
            ),
        }
        if len(statuses) != len(workspace_entries):
            raise GraphVarError("not every planned workspace has an explicit status")
        atomic_json(graphvar_dir / "validation.json", validation)
        atomic_json(
            graphvar_dir / "status.json",
            {
                "state": "succeeded",
                "completed": len(workspace_entries),
                "total": len(workspace_entries),
            },
        )
        artifacts = _all_files(partial, {"graphvar/manifest.json"})
        atomic_json(
            graphvar_dir / "manifest.json",
            {
                "schema": SCHEMA,
                "run_id": run_id,
                "scientific_identity_sha256": scientific_sha,
                "workspace_run_id": prepared.run_id,
                "artifacts": artifacts,
            },
        )
        if final.exists():
            raise GraphVarError(f"GraphVar final run appeared concurrently: {final}")
        os.replace(partial, final)
        verified = verify_graphvar_analysis_run(final, lock)
        return GraphVarAnalysisRun(
            verified.run_id,
            verified.run_dir,
            verified.manifest,
            False,
            verified.artifacts,
        )
    except BaseException as exc:
        atomic_json(
            graphvar_dir / "status.json",
            {
                "state": "failed_nonfinal",
                "completed": sum(
                    (
                        graphvar_dir
                        / "workspaces"
                        / entry["workspace_id"]
                        / "results"
                        / "CorrResults"
                        / "completion.json"
                    ).is_file()
                    for entry in workspace_entries
                ),
                "total": len(workspace_entries),
                "error": str(exc),
            },
        )
        raise
