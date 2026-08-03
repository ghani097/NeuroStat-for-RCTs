"""Command-line entry point for the staged NeuroStat v2 package."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__

_RUN_STAGE_CHOICES = (
    "ingest",
    "scouts",
    "connectivity",
    "network_models",
    "graphvar",
    "brainnet",
    "all",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurostat",
        description=(
            "NeuroStat v2 staged clean-data, source-scout, connectivity, "
            "network-model, GraphVar, and BrainNet workflow."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"neurostat-rcts {__version__}",
        help="Print the installed NeuroStat version and exit.",
    )
    commands = parser.add_subparsers(
        dest="command",
        metavar="{validate,ingest,scouts,connectivity,network-models,graphvar,brainnet}",
    )
    run = commands.add_parser("run")
    run.add_argument("project", metavar="PROJECT")
    run.add_argument(
        "--stage",
        action="append",
        choices=_RUN_STAGE_CHOICES,
        metavar="STAGE",
        help="Workflow stage to run; repeat to include multiple stages.",
    )
    run.add_argument("--json", action="store_true", dest="as_json")
    for name in ("validate", "ingest"):
        command = commands.add_parser(name)
        command.add_argument("project", metavar="PROJECT")
        command.add_argument("--json", action="store_true", dest="as_json")
    scouts = commands.add_parser("scouts")
    scouts.add_argument("project", metavar="PROJECT")
    scouts.add_argument("--ingest-run", required=True, metavar="RUN_DIR")
    scouts.add_argument("--json", action="store_true", dest="as_json")
    connectivity = commands.add_parser("connectivity")
    connectivity.add_argument("project", metavar="PROJECT")
    connectivity.add_argument("--scouts-run", required=True, metavar="RUN_DIR")
    connectivity.add_argument("--json", action="store_true", dest="as_json")
    network_models = commands.add_parser("network-models")
    network_models.add_argument("project", metavar="PROJECT")
    network_models.add_argument("--connectivity-run", required=True, metavar="RUN_DIR")
    network_models.add_argument("--json", action="store_true", dest="as_json")
    graphvar = commands.add_parser(
        "graphvar",
        help="Build, execute, or safely import GraphVar 2.03a workspaces.",
    )
    graphvar_commands = graphvar.add_subparsers(dest="graphvar_command", required=True)
    graphvar_build = graphvar_commands.add_parser("build")
    graphvar_build.add_argument("project", metavar="PROJECT")
    graphvar_build.add_argument("--connectivity-run", required=True, metavar="RUN_DIR")
    graphvar_build.add_argument("--json", action="store_true", dest="as_json")
    for action in ("run", "extract"):
        command = graphvar_commands.add_parser(action)
        command.add_argument("project", metavar="PROJECT")
        command.add_argument("--workspace-run", required=True, metavar="RUN_DIR")
        if action == "extract":
            command.add_argument(
                "--result",
                action="append",
                required=True,
                metavar="WORKSPACE=RESULT_MAT",
                help="One exact workspace-to-result mapping; repeat for every workspace.",
            )
        command.add_argument("--json", action="store_true", dest="as_json")
    brainnet = commands.add_parser(
        "brainnet",
        help="Build deterministic BrainNet artifacts or render verified PNGs.",
    )
    brainnet_commands = brainnet.add_subparsers(
        dest="brainnet_command",
        required=True,
    )
    brainnet_build = brainnet_commands.add_parser("build")
    brainnet_build.add_argument("project", metavar="PROJECT")
    brainnet_build.add_argument("--graphvar-run", required=True, metavar="RUN_DIR")
    brainnet_build.add_argument("--json", action="store_true", dest="as_json")
    brainnet_render = brainnet_commands.add_parser("render")
    brainnet_render.add_argument("project", metavar="PROJECT")
    brainnet_render.add_argument("--artifact-run", required=True, metavar="RUN_DIR")
    brainnet_render.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _workflow_json(result: Any) -> dict[str, Any]:
    stages = []
    for outcome in result.outcomes:
        item = {
            "stage": outcome.stage.value,
            "state": outcome.state.value,
            "run_id": outcome.run_id,
            "run_dir": None if outcome.run_dir is None else str(outcome.run_dir),
        }
        if outcome.reused is not None:
            item["reused"] = outcome.reused
        stages.append(item)
    return {"stages": stages, "success": result.success}


def _print_workflow(result: Any) -> None:
    for outcome in result.outcomes:
        line = f"{outcome.stage.value}: {outcome.state.value}"
        if outcome.run_id is not None:
            line += f" {outcome.run_id}"
        if outcome.run_dir is not None:
            line += f" ({outcome.run_dir})"
        if outcome.reused is not None:
            line += f" reused={outcome.reused}"
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the foundation CLI and return a process status code."""
    args = _build_parser().parse_args(argv)
    if not args.command:
        return 0
    try:
        if args.command == "run":
            from .workflow import run_workflow

            workflow = run_workflow(args.project, args.stage or ("all",))
            if args.as_json:
                print(json.dumps(_workflow_json(workflow), sort_keys=True))
            else:
                _print_workflow(workflow)
            return 0

        from .project import load_project

        project = load_project(args.project)
        if args.command == "validate":
            result = {
                "valid": True,
                "project": str(project.project_file),
                "records": len(project.records),
                "output_root": str(project.output_root),
            }
        elif args.command == "ingest":
            from .ingest import ingest_project

            run = ingest_project(project)
            result = {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "records": len(run.records),
                "reused": run.reused,
                "written": run.written,
            }
        elif args.command == "scouts":
            from .scouts import run_scouts

            run = run_scouts(project, args.ingest_run)
            result = {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "records": len(run.records),
                "reused": run.reused,
                "written": run.written,
            }
        elif args.command == "connectivity":
            from .connectivity import run_connectivity

            run = run_connectivity(project, args.scouts_run)
            result = {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "records": len(run.records),
                "reused": run.reused,
                "written": run.written,
            }
        elif args.command == "network-models":
            from .network_models import run_network_models

            run = run_network_models(project, args.connectivity_run)
            result = {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "artifacts": len(run.artifacts),
                "reused": run.reused,
            }
        elif args.command == "brainnet":
            if args.brainnet_command == "build":
                from .brainnet import build_brainnet_artifacts

                run = build_brainnet_artifacts(project, args.graphvar_run)
                result = {
                    "run_id": run.run_id,
                    "run_dir": str(run.run_dir),
                    "workspaces": len(run.workspaces),
                    "reused": run.reused,
                    "state": "prepared",
                }
            else:
                from .brainnet import render_brainnet_artifacts

                run = render_brainnet_artifacts(project, args.artifact_run)
                result = {
                    "run_id": run.run_id,
                    "run_dir": str(run.run_dir),
                    "images": len(run.images),
                    "reused": run.reused,
                    "state": "rendered",
                }
        elif args.graphvar_command == "build":
            from .graphvar import build_graphvar_workspaces

            run = build_graphvar_workspaces(project, args.connectivity_run)
            result = {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "workspaces": len(run.workspaces),
                "reused": run.reused,
                "state": "prepared",
            }
        else:
            from .graphvar import run_graphvar_analysis

            imported = None
            if args.graphvar_command == "extract":
                imported = {}
                for item in args.result:
                    if "=" not in item:
                        raise ValueError("--result must use WORKSPACE=RESULT_MAT")
                    workspace, path = item.split("=", 1)
                    if not workspace or not path or workspace in imported:
                        raise ValueError(
                            "--result mappings must have unique nonblank workspaces",
                        )
                    imported[workspace] = path
            run = run_graphvar_analysis(
                project,
                args.workspace_run,
                result_files=imported,
            )
            result = {
                "run_id": run.run_id,
                "run_dir": str(run.run_dir),
                "artifacts": len(run.artifacts),
                "reused": run.reused,
                "mode": args.graphvar_command,
            }
        if args.as_json:
            print(json.dumps(result, sort_keys=True))
        else:
            verbs = {
                "validate": "valid",
                "ingest": "ingested",
                "scouts": "generated scouts for",
                "connectivity": "generated connectivity for",
            }
            if args.command == "validate":
                print("valid")
            elif args.command == "network-models":
                print(f"generated network models: {result['run_id']}")
            elif args.command == "graphvar":
                print(
                    f"GraphVar {args.graphvar_command}: {result['run_id']} "
                    f"({result['run_dir']})",
                )
            elif args.command == "brainnet":
                print(
                    f"BrainNet {args.brainnet_command}: {result['run_id']} "
                    f"({result['run_dir']})",
                )
            else:
                print(
                    f"{verbs[args.command]} {result['records']} record(s): "
                    f"{result['run_id']}",
                )
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]
