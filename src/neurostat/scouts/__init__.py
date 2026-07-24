"""Source-scout stage for fixed operators and MNE inverse solutions."""
from .artifacts import SCHEMA, ScoutError, ScoutHeader, ScoutStream, verify_scout_artifact
from .service import ScoutRun, run_scouts

__all__ = [
    "SCHEMA",
    "ScoutError",
    "ScoutHeader",
    "ScoutRun",
    "ScoutStream",
    "run_scouts",
    "verify_scout_artifact",
]
