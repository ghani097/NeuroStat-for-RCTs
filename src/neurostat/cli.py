"""Command-line entry point for the staged NeuroStat v2 package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurostat",
        description=(
            "NeuroStat v2 foundation. Future commands will run the staged "
            "ingest, scouts, connectivity, network_models, graphvar, and "
            "brainnet workflow."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"neurostat-rcts {__version__}",
        help="Print the installed NeuroStat version and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the foundation CLI and return a process status code."""
    _build_parser().parse_args(argv)
    return 0
