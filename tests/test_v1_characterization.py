"""Characterization tests for the v1 compatibility baseline."""

from pathlib import Path

import numpy as np
import pytest

from pli_pipeline import ConnectivityConfig, PipelineConfig, StudyDesign, _clean_matrices, _infer_paths, find_set_files


def test_study_design_preserves_supplied_group_and_session_order() -> None:
    groups = ["treatment", "control"]
    sessions = ["follow_up", "baseline"]
    design = StudyDesign(groups=groups, sessions=sessions)
    assert design.groups == groups
    assert design.sessions == sessions


def test_find_set_files_is_sorted_nonrecursive_and_excludes_cleaned(tmp_path: Path) -> None:
    (tmp_path / "z.set").touch()
    (tmp_path / "a.set").touch()
    (tmp_path / "a_cleaned.set").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "inside.set").touch()
    assert [path.name for path in find_set_files(tmp_path)] == ["a.set", "z.set"]


def test_find_set_files_raises_when_no_eligible_files_exist(tmp_path: Path) -> None:
    (tmp_path / "only_cleaned.set").touch()
    with pytest.raises(FileNotFoundError, match="No .set files found"):
        find_set_files(tmp_path)


def test_infer_paths_uses_trailing_group_and_session_and_creates_output(tmp_path: Path) -> None:
    source = tmp_path / "input" / "arm_a" / "post" / "participant.set"
    source.parent.mkdir(parents=True)
    source.touch()
    output_root = tmp_path / "output"
    group, session, output = _infer_paths(source, output_root)
    assert (group, session) == ("arm_a", "post")
    assert output == output_root / "arm_a" / "post" / "participant"
    assert output.is_dir()


def test_pipeline_config_defaults_match_the_v1_baseline() -> None:
    config = PipelineConfig()
    assert config.output_root == Path("processed")
    assert config.preprocessing.l_freq == 1.0
    assert config.preprocessing.h_freq == 40.0
    assert config.preprocessing.use_asr is True
    assert config.preprocessing.use_iclabel is True
    assert config.preprocessing.use_ged is False
    assert config.source.trans_path == Path("fsaverage-trans.fif")
    assert config.connectivity.epoch_length == 4.0
    assert config.connectivity.method == "pli"
    assert config.connectivity.use_gpu is False


def test_default_frequency_bands_match_the_v1_baseline() -> None:
    assert ConnectivityConfig().frequency_bands == {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}


def test_clean_matrices_returns_empty_array_for_empty_input() -> None:
    assert _clean_matrices([]).shape == (0,)


def test_clean_matrices_discards_all_invalid_subjects() -> None:
    result = _clean_matrices([np.full((2, 2), np.nan), np.full((2, 2), np.inf)])
    assert result.shape == (0,)


def test_clean_matrices_replaces_nonfinite_values_with_subject_mean() -> None:
    result = _clean_matrices([np.array([[1.0, np.nan], [np.inf, 3.0]])])
    np.testing.assert_allclose(result, np.array([[[1.0, 2.0], [2.0, 3.0]]]))
