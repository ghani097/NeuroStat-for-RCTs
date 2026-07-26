"""Command-line entry point for the staged NeuroStat v2 package."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__


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
    commands = parser.add_subparsers(dest="command")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the foundation CLI and return a process status code."""
    args = _build_parser().parse_args(argv)
    if not args.command:
        return 0
    try:
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
                            "--result mappings must have unique nonblank workspaces"
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
                    f"({result['run_dir']})"
                )
            else:
                print(
                    f"{verbs[args.command]} {result['records']} record(s): {result['run_id']}"
                )
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
