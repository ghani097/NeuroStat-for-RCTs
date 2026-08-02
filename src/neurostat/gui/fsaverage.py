"""Lazy MNE fsaverage provisioning for GUI study setup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MNEResourcePaths:
    """Resolved fsaverage resources required by the MNE profile backend."""

    subjects_dir: Path
    subject: str
    trans: Path
    source_space: Path
    bem: Path


def provision_fsaverage(
    subjects_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> MNEResourcePaths:
    """Ensure fsaverage exists and return profile-ready resource paths."""
    try:
        import mne
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("MNE is required to provision fsaverage resources.") from exc

    subject = "fsaverage"
    try:
        resolved_subjects_dir = _resolve_subjects_dir(subjects_dir, progress)
        bem_dir = resolved_subjects_dir / subject / "bem"
        label_dir = resolved_subjects_dir / subject / "label"
        source_space = bem_dir / "fsaverage-ico5-src.fif"
        bem = bem_dir / "fsaverage-5120-5120-5120-bem-sol.fif"
        trans = bem_dir / "fsaverage-trans.fif"
        if not source_space.is_file():
            if progress is not None:
                progress(f"Building missing source space: {source_space}")
            src = mne.setup_source_space(
                subject=subject,
                spacing="ico5",
                subjects_dir=resolved_subjects_dir,
                add_dist=False,
            )
            mne.write_source_spaces(source_space, src, overwrite=True)
        _require_file(trans, "head<->MRI transform")
        _require_file(source_space, "ico5 source space")
        _require_file(bem, "BEM solution")
        _require_file(label_dir / "lh.aparc.annot", "left aparc annotation")
        _require_file(label_dir / "rh.aparc.annot", "right aparc annotation")
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(str(exc) or exc.__class__.__name__) from None
    return MNEResourcePaths(
        subjects_dir=resolved_subjects_dir,
        subject=subject,
        trans=trans,
        source_space=source_space,
        bem=bem,
    )


def _resolve_subjects_dir(
    subjects_dir: str | Path | None,
    progress: Callable[[str], None] | None,
) -> Path:
    if subjects_dir is not None:
        candidate = Path(subjects_dir).expanduser()
        if candidate.exists():
            return candidate
    if progress is not None:
        progress("Fetching fsaverage subject")
    from mne.datasets import fetch_fsaverage

    return Path(fetch_fsaverage(verbose=True)).parent


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Missing fsaverage {label}: {path}")


__all__ = ["MNEResourcePaths", "provision_fsaverage"]
