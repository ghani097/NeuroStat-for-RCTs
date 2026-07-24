from __future__ import annotations

from pathlib import Path

import pytest
from scipy.io import savemat

from neurostat.analysis_profiles import AnalysisProfileError, FixedLinearSource, parse_analysis_profile
from neurostat.atlas import DK68_LABELS
from neurostat.project import load_project


BASE34 = (
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "frontalpole",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "insula",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "paracentral",
    "parahippocampal",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "temporalpole",
    "transversetemporal",
)


def _project(tmp_path: Path, profile: str) -> Path:
    (tmp_path / "clean.fif").write_bytes(b"placeholder")
    savemat(tmp_path / "operator.mat", {"SensorToScoutOperator": [[1.0, 0.0], [0.0, 1.0]]})
    path = tmp_path / "project.yaml"
    path.write_text(
        f"""schema_version: 1
study: {{id: demo, arms: [active, comparison], sessions: [baseline, followup], conditions: [sep]}}
input: {{state: cleaned}}
records: [{{record_id: r1, participant_id: p1, arm: active, session: baseline, condition: sep, data_kind: epochs, format: fif, path: clean.fif, analysis_profile: fixed}}]
analysis_profiles:
  fixed:
{profile}
output: {{root: output}}
""",
        encoding="utf-8",
    )
    return path


def _valid_profile() -> str:
    return """    source:
      backend: fixed_linear_operator
      operator: operator.mat
      sensor_labels: [B, A]
      scout_labels: [left, right]
      epoch_chunk_size: 32
      output_chunk_samples: 128
      export_matlab: false
    connectivity:
      backend: evoked_average_hilbert_pli
      filter_order: 4
      bands: {theta: [3.5, 7.5], alpha: [7.5, 12.5], beta: [12.5, 30], gamma: [30, 40]}
"""


def test_accepted_graphvar_dk68_order_is_exact_and_interleaved() -> None:
    expected = tuple(label for name in BASE34 for label in (f"{name}-lh", f"{name}-rh"))
    assert DK68_LABELS == expected
    assert DK68_LABELS[14:20] == (
        "inferiorparietal-lh",
        "inferiorparietal-rh",
        "inferiortemporal-lh",
        "inferiortemporal-rh",
        "insula-lh",
        "insula-rh",
    )


def test_fixed_profile_resolves_resources_and_freezes_exact_v7_contract(tmp_path: Path) -> None:
    parsed = parse_analysis_profile(load_project(_project(tmp_path, _valid_profile())), "fixed")
    assert isinstance(parsed.source, FixedLinearSource)
    assert parsed.source.operator == (tmp_path / "operator.mat").resolve()
    assert parsed.source.sensor_labels == ("B", "A")
    assert parsed.source.epoch_chunk_size == 32
    assert parsed.connectivity.filter_order == 4
    assert [(band.name, band.low_hz, band.high_hz) for band in parsed.connectivity.bands] == [
        ("theta", 3.5, 7.5),
        ("alpha", 7.5, 12.5),
        ("beta", 12.5, 30.0),
        ("gamma", 30.0, 40.0),
    ]


@pytest.mark.parametrize(
    ("old", "new", "needle"),
    [
        ("export_matlab: false", "export_matlab: 0", "boolean"),
        ("epoch_chunk_size: 32", "epoch_chunk_size: true", "integer"),
        ("filter_order: 4", "filter_order: true", "integer"),
        ("filter_order: 4", "filter_order: 3", "must be 4"),
        ("sensor_labels: [B, A]", "sensor_labels: [A, A]", "duplicates"),
        ("bands: {theta: [3.5, 7.5]", "bands: {theta: [7.5, 3.5]", "low < high"),
        ("operator: operator.mat", "operator: missing.mat", "not a file"),
        ("output_chunk_samples: 128", "output_chunk_samples: 0", "integer"),
    ],
)
def test_strict_profile_boundaries(tmp_path: Path, old: str, new: str, needle: str) -> None:
    profile = _valid_profile().replace(old, new)
    with pytest.raises(AnalysisProfileError, match=needle):
        parse_analysis_profile(load_project(_project(tmp_path, profile)), "fixed")


def test_unknown_profile_key_and_unsupported_pair_fail(tmp_path: Path) -> None:
    unknown = _valid_profile().replace("export_matlab: false", "export_matlab: false\n      surprise: hidden")
    with pytest.raises(AnalysisProfileError, match="unknown"):
        parse_analysis_profile(load_project(_project(tmp_path, unknown)), "fixed")
    pair = _valid_profile().replace("evoked_average_hilbert_pli", "continuous_multitaper_pli").replace(
        "      filter_order: 4\n", "      window_seconds: 4\n"
    )
    with pytest.raises(AnalysisProfileError, match="unsupported source/connectivity combination"):
        parse_analysis_profile(load_project(_project(tmp_path, pair)), "fixed")


def test_ingestion_only_project_loading_remains_dependency_light_and_analysis_fails_lazily(tmp_path: Path) -> None:
    # Phase-2 projects remain ingestible; a scientific stage demands the new
    # strict profile before importing MNE/SciPy backends.
    path = _project(tmp_path, "    legacy_option: value\n")
    project = load_project(path)
    with pytest.raises(AnalysisProfileError, match="exactly source and connectivity"):
        parse_analysis_profile(project, "fixed")
