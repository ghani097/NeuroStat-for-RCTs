"""Tests that v2 stays lightweight until compatibility is explicitly requested."""

import os
from pathlib import Path
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).parents[1] / "src"


def _subprocess_environment() -> dict[str, str]:
    existing = os.environ.get("PYTHONPATH")
    pythonpath = str(PACKAGE_ROOT)
    if existing:
        pythonpath = os.pathsep.join((pythonpath, existing))
    return os.environ | {"PYTHONPATH": pythonpath}


def _run_import(module: str) -> set[str]:
    code = "import sys; before = set(sys.modules); " + f"import {module}; " + "print('\\n'.join(sorted(set(sys.modules) - before)))"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
    )
    return set(filter(None, completed.stdout.splitlines()))


def _assert_scientific_stack_not_loaded(modules: set[str]) -> None:
    prohibited = ("mne", "matplotlib", "PyQt5", "pli_pipeline")
    assert not any(module == prefix or module.startswith(f"{prefix}.") for module in modules for prefix in prohibited)


def test_package_import_is_lightweight() -> None:
    _assert_scientific_stack_not_loaded(_run_import("neurostat"))


def test_legacy_adapter_import_is_lightweight() -> None:
    _assert_scientific_stack_not_loaded(_run_import("neurostat.legacy"))


def test_legacy_adapter_loads_v1_only_on_demand() -> None:
    from neurostat.legacy import load_legacy_pipeline

    assert load_legacy_pipeline().__name__ == "pli_pipeline"


def test_cli_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "neurostat", "--version"],
        check=True,
        capture_output=True,
        text=True,
        env=_subprocess_environment(),
    )
    assert completed.stdout == "neurostat-rcts 2.0.0a1\n"
