"""Qt-free bridge from study setup forms to the frozen v1 preprocessor."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from neurostat.legacy import load_legacy_pipeline

from .discovery import discover_records
from .study_builder import StudyForm


def stage_raw_input(
    form: StudyForm,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Return the folder v1 should preprocess, staging non-SET input first.

    v1 reads only ``<base>/<arm>/<session>/*.set``. When the study folder
    already has that shape it is used as is, so existing SET studies run
    exactly as before. Anything else (BDF, EDF, BrainVision, Poly5, XDF, WiBCI
    CSV, or a subject-folder layout) is converted into the staging root.
    """
    from neurostat.io.staging import needs_staging, stage_as_set

    base = Path(form.base_folder)
    records = discover_records(base, form.arms, form.sessions, form.conditions)
    if not records or not needs_staging(records, base):
        return base
    root = form.resolved_staging_root
    if progress:
        progress(f"Staging {len(records)} raw recording(s) as EEGLAB SET under {root}")
    stage_as_set(records, root, reader_options=form.reader_options(), progress=progress)
    return root


def run_v1_preprocessing(
    form: StudyForm,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Run v1 preprocessing for the form's declared arms and sessions."""
    legacy = load_legacy_pipeline()
    input_root = stage_raw_input(form, progress)
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
        input_root,
        design,
        config=config,
        progress_cb=progress,
    )
    return processed_root


__all__ = ["run_v1_preprocessing", "stage_raw_input"]
