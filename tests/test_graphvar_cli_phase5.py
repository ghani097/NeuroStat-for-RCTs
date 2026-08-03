from __future__ import annotations

from neurostat.cli import main


def test_graphvar_extract_rejects_malformed_and_duplicate_result_maps(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("neurostat.project.load_project", lambda _path: object())
    assert (
        main(
            [
                "graphvar",
                "extract",
                "project.yaml",
                "--workspace-run",
                "work",
                "--result",
                "malformed",
            ]
        )
        == 2
    )
    assert "WORKSPACE=RESULT_MAT" in capsys.readouterr().err
    assert (
        main(
            [
                "graphvar",
                "extract",
                "project.yaml",
                "--workspace-run",
                "work",
                "--result",
                "one=a.mat",
                "--result",
                "one=b.mat",
            ]
        )
        == 2
    )
    assert "unique nonblank" in capsys.readouterr().err
