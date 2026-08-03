"""Qt-free bridge from study setup forms to the frozen v1 preprocessor."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from neurostat.legacy import load_legacy_pipeline

from .study_builder import StudyForm


def run_v1_preprocessing(
    form: StudyForm,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Run v1 preprocessing for the form's declared arms and sessions."""
    legacy = load_legacy_pipeline()
    processed_root = form.resolved_preprocessing_output_root
    design = legacy.StudyDesign(groups=list(form.arms), sessions=list(form.sessions))
    config = legacy.PipelineConfig()
    config.output_root = processed_root
    preprocessing = getattr(config, "preprocessing", None)
    for name, value in form.preprocessing_settings.items():
        target = (
            preprocessing
            if preprocessing is not None and hasattr(preprocessing, name)
            else config
        )
        if not hasattr(target, name):
            raise ValueError(f"Unknown v1 preprocessing setting: {name}")
        setattr(target, name, value)
    legacy.run_preprocessing_for_design(
        Path(form.base_folder),
        design,
        config=config,
        progress_cb=progress,
    )
    return processed_root


__all__ = ["run_v1_preprocessing"]
