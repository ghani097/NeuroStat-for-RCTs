"""Command-line entry point for the staged NeuroStat v2 package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import sys

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
            result = {"valid": True, "project": str(project.project_file), "records": len(project.records), "output_root": str(project.output_root)}
        elif args.command == "ingest":
            from .ingest import ingest_project
            run = ingest_project(project)
            result = {"run_id": run.run_id, "run_dir": str(run.run_dir), "records": len(run.records), "reused": run.reused, "written": run.written}
        elif args.command == "scouts":
            from .scouts import run_scouts
            run = run_scouts(project, args.ingest_run)
            result = {"run_id": run.run_id, "run_dir": str(run.run_dir), "records": len(run.records), "reused": run.reused, "written": run.written}
        else:
            from .connectivity import run_connectivity
            run = run_connectivity(project, args.scouts_run)
            result = {"run_id": run.run_id, "run_dir": str(run.run_dir), "records": len(run.records), "reused": run.reused, "written": run.written}
        if args.as_json: print(json.dumps(result, sort_keys=True))
        else:
            verbs = {"validate": "valid", "ingest": "ingested", "scouts": "generated scouts for", "connectivity": "generated connectivity for"}
            print("valid" if args.command == "validate" else f"{verbs[args.command]} {result['records']} record(s): {result['run_id']}")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
