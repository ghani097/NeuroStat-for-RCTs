from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from neurostat.graphvar.config import (
    AtlasNode,
    AtlasSettings,
    GraphVarConfig,
    GraphVarRuntime,
)
from neurostat.graphvar.runtime import preflight_graphvar
from neurostat.graphvar.workspace import GraphVarError
from neurostat.stage_io import sha256_file


def _config(tmp_path: Path) -> GraphVarConfig:
    executable = tmp_path / "MATLAB Runtime With Spaces" / "matlab.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"matlab")
    root = tmp_path / "GraphVar Root With Spaces"
    (root / "src" / "calc").mkdir(parents=True)
    (root / "src" / "calc" / "graphvar_glm.m").write_text(
        "function graphvar_glm\nend\n", encoding="utf-8"
    )
    (root / "start_GraphVar.m").write_text("% Release= 2.03a\n", encoding="utf-8")
    atlas = tmp_path / "atlas.xlsx"
    atlas.write_bytes(b"atlas")
    viewer = tmp_path / "viewer.mat"
    viewer.write_bytes(b"viewer")
    return GraphVarConfig(
        AtlasSettings(atlas, sha256_file(atlas), viewer, sha256_file(viewer), ()),
        (),
        GraphVarRuntime(str(executable), root, "2025b", 30),
    )


def test_matlab_preflight_uses_argument_vector_no_shell_and_exact_hashes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    observed: dict[str, object] = {}

    def process(args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        request = json.loads(
            Path(kwargs["env"]["NEUROSTAT_GRAPHVAR_REQUEST"]).read_text(
                encoding="utf-8"
            )
        )
        root = Path(request["graphvar_root"])
        Path(request["metadata_path"]).write_text(
            json.dumps(
                {
                    "matlab_release": "2025b",
                    "graphvar_release": "2.03a",
                    "graphvar_glm": str(
                        (root / "src" / "calc" / "graphvar_glm.m").resolve()
                    ),
                    "graphvar_glm_sha256": sha256_file(
                        root / "src" / "calc" / "graphvar_glm.m"
                    ),
                    "start_graphvar_sha256": sha256_file(root / "start_GraphVar.m"),
                    "statistics_toolbox": True,
                    "required_functions": [
                        "jsondecode",
                        "readcell",
                        "rng",
                        "regress",
                        "fcdf",
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "preflight ok", "")

    metadata = preflight_graphvar(
        config, tmp_path / "output with spaces", process_runner=process
    )
    assert metadata.matlab_release == "2025b"
    assert observed["args"][0].endswith("matlab.exe")  # type: ignore[index]
    assert observed["kwargs"]["shell"] is False  # type: ignore[index]
    assert observed["kwargs"]["check"] is False  # type: ignore[index]
    assert isinstance(observed["args"], tuple)


def test_nonzero_preflight_is_failure_even_if_metadata_appears(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def fail(args, **kwargs):
        request = json.loads(
            Path(kwargs["env"]["NEUROSTAT_GRAPHVAR_REQUEST"]).read_text(
                encoding="utf-8"
            )
        )
        Path(request["metadata_path"]).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(args, 9, "false output", "actual failure")

    with pytest.raises(GraphVarError, match="exit code 9"):
        preflight_graphvar(config, tmp_path / "failed", process_runner=fail)


def test_timeout_and_preflight_release_mismatch_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def timeout(args, **_kwargs):
        raise subprocess.TimeoutExpired(args, 30, output="partial", stderr="hung")

    with pytest.raises(GraphVarError, match="exceeded"):
        preflight_graphvar(config, tmp_path / "timeout", process_runner=timeout)

    def mismatch(args, **kwargs):
        request = json.loads(
            Path(kwargs["env"]["NEUROSTAT_GRAPHVAR_REQUEST"]).read_text(
                encoding="utf-8"
            )
        )
        root = Path(request["graphvar_root"])
        Path(request["metadata_path"]).write_text(
            json.dumps(
                {
                    "matlab_release": "2024b",
                    "graphvar_release": "2.03a",
                    "graphvar_glm": str(root / "src" / "calc" / "graphvar_glm.m"),
                    "graphvar_glm_sha256": sha256_file(
                        root / "src" / "calc" / "graphvar_glm.m"
                    ),
                    "start_graphvar_sha256": sha256_file(root / "start_GraphVar.m"),
                    "statistics_toolbox": True,
                    "required_functions": [
                        "jsondecode",
                        "readcell",
                        "rng",
                        "regress",
                        "fcdf",
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "", "")

    with pytest.raises(GraphVarError, match="MATLAB release"):
        preflight_graphvar(config, tmp_path / "mismatch", process_runner=mismatch)


@pytest.mark.skipif(
    not all(
        os.environ.get(name)
        for name in (
            "NEUROSTAT_TEST_MATLAB",
            "NEUROSTAT_TEST_GRAPHVAR_ROOT",
            "NEUROSTAT_TEST_GRAPHVAR_SMOKE_WORKSPACE",
            "NEUROSTAT_TEST_GRAPHVAR_ATLAS_INDICES",
        )
    ),
    reason="set real MATLAB/GraphVar/workspace/atlas-index environment variables",
)
def test_real_graphvar_tiny_permutation_smoke() -> None:
    """Run the actual RFUN boundary on a disposable workspace copy."""
    import csv

    from neurostat.graphvar.runtime import run_graphvar_workspace

    source = Path(os.environ["NEUROSTAT_TEST_GRAPHVAR_SMOKE_WORKSPACE"])
    graphvar_root = Path(os.environ["NEUROSTAT_TEST_GRAPHVAR_ROOT"])
    indices = tuple(
        int(value)
        for value in os.environ["NEUROSTAT_TEST_GRAPHVAR_ATLAS_INDICES"].split(",")
    )
    with TemporaryDirectory(prefix="neurostat-graphvar-smoke-") as name:
        root = Path(name)
        workspace = root / "workspace"
        shutil.copytree(
            source, workspace, ignore=shutil.ignore_patterns("corr_area_1_Group.mat")
        )
        variables = workspace / "Variables.csv"
        with variables.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter=";"))
        rows[0] = ["ObservationID", "arm", "subject", "session"]
        with variables.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, delimiter=";", lineterminator="\n").writerows(rows)
        # Historical workspaces may store bare files_rel names. The packaged
        # builder stores data/CorrMatrix paths; root copies make either usable.
        for matrix in (workspace / "data" / "CorrMatrix").glob("*.mat"):
            shutil.copy2(matrix, workspace / matrix.name)
        nodes = tuple(
            AtlasNode(str(index), index, str(index), None) for index in indices
        )
        config = GraphVarConfig(
            AtlasSettings(
                workspace / "atlas.xlsx", "", workspace / "viewer.mat", "", nodes
            ),
            (),
            GraphVarRuntime(
                os.environ["NEUROSTAT_TEST_MATLAB"],
                graphvar_root,
                "2025b",
                600,
            ),
        )
        execution = run_graphvar_workspace(
            config,
            workspace,
            {
                "seed": 1729,
                "permutations": 2,
                "effect_label": os.environ.get(
                    "NEUROSTAT_TEST_GRAPHVAR_EFFECT_LABEL",
                    "arm_Chiro-arm_Control=0",
                ),
            },
            root / "output",
        )
        assert execution.returncode == 0
        assert execution.result.is_file() and execution.result.stat().st_size > 0
