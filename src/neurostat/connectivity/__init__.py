"""PLI connectivity stage for evoked and continuous scout recordings."""
from .artifacts import SCHEMA, ConnectivityHeader, verify_connectivity_artifact
from .methods import (
    ConnectivityError,
    ConnectivityValues,
    continuous_multitaper_pli,
    evoked_average_hilbert_pli,
    iter_scout_windows,
    scout_window_count,
    validate_pli_matrix,
)
from .service import ConnectivityRun, run_connectivity

__all__ = [
    "SCHEMA",
    "ConnectivityError",
    "ConnectivityHeader",
    "ConnectivityRun",
    "ConnectivityValues",
    "continuous_multitaper_pli",
    "evoked_average_hilbert_pli",
    "iter_scout_windows",
    "run_connectivity",
    "scout_window_count",
    "validate_pli_matrix",
    "verify_connectivity_artifact",
]
