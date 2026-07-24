"""Production MNE source backend with a replaceable orchestration seam."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from ..analysis_profiles import MNEFsaverageSource
from ..atlas import DK68_LABELS
from .artifacts import ScoutError, ScoutStream


class MNEProjector(Protocol):
    """Backend seam used by tests and alternate execution environments."""

    def project(self, canonical_path: Path, config: MNEFsaverageSource) -> ScoutStream:
        """Return a bounded, ordered scout stream."""


def _streamed_covariance(dataset: object, channels: int, samples: int, chunk_samples: int) -> np.ndarray:
    total = np.zeros(channels, dtype=np.float64)
    cross = np.zeros((channels, channels), dtype=np.float64)
    count = 0
    for start in range(0, samples, chunk_samples):
        block = np.asarray(dataset[0, :, start : start + chunk_samples], dtype=np.float64)  # type: ignore[index]
        total += np.sum(block, axis=1, dtype=np.float64)
        cross += block @ block.T
        count += block.shape[1]
    if count < 2:
        raise ScoutError("MNE empirical covariance requires at least two samples")
    covariance = (cross - np.outer(total, total) / count) / (count - 1)
    if not np.isfinite(covariance).all():
        raise ScoutError("MNE empirical covariance is non-finite")
    return covariance


class DefaultMNEProjector:
    """MNE forward/inverse implementation; it never fetches FreeSurfer data."""

    def project(self, canonical_path: Path, config: MNEFsaverageSource) -> ScoutStream:
        import h5py
        import mne
        from mne.minimum_norm import apply_inverse_raw, make_inverse_operator

        with h5py.File(canonical_path, "r") as file:
            if str(file.attrs["data_kind"]) != "raw" or file["data"].shape[0] != 1:
                raise ScoutError("mne_fsaverage requires one canonical continuous raw epoch")
            names = tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in file["channel_names"][:])
            if "channel_positions" not in file:
                raise ScoutError("mne_fsaverage requires finite canonical channel positions")
            positions = np.asarray(file["channel_positions"][:], dtype=np.float64)
            if positions.shape != (len(names), 3) or not np.isfinite(positions).all():
                raise ScoutError("mne_fsaverage canonical channel positions are invalid")
            sfreq = float(file.attrs["sfreq"])
            times = np.asarray(file["times"][:], dtype=np.float64)
            sample_count = int(file["data"].shape[-1])
            covariance_data = _streamed_covariance(file["data"], len(names), sample_count, config.chunk_samples)

        info = mne.create_info(list(names), sfreq, ch_types="eeg")
        montage = mne.channels.make_dig_montage(
            ch_pos={name: positions[index] for index, name in enumerate(names)},
            coord_frame=config.coordinate_frame,
        )
        info.set_montage(montage, on_missing="raise")
        if config.average_reference_projection:
            reference_holder = mne.io.RawArray(np.zeros((len(names), 1)), info, verbose="ERROR")
            reference_holder.set_eeg_reference("average", projection=True, verbose="ERROR")
            info = reference_holder.info.copy()
        else:
            raise ScoutError("MNE inverse requires the explicitly configured average-reference projection")

        source_space = mne.read_source_spaces(config.source_space, verbose="ERROR")
        bem = mne.read_bem_solution(config.bem, verbose="ERROR")
        forward = mne.make_forward_solution(
            info,
            trans=str(config.trans),
            src=source_space,
            bem=bem,
            meg=False,
            eeg=True,
            mindist=config.mindist_mm,
            n_jobs=1,
            verbose="ERROR",
        )
        covariance = mne.Covariance(
            covariance_data,
            names=list(names),
            bads=[],
            projs=list(info["projs"]),
            nfree=sample_count - 1,
        )
        inverse = make_inverse_operator(
            info,
            forward,
            covariance,
            loose=config.loose,
            depth=config.depth,
            verbose="ERROR",
        )
        available = mne.read_labels_from_annot(
            config.subject,
            parc=config.atlas,
            subjects_dir=config.subjects_dir,
            verbose="ERROR",
        )
        by_name = {
            label.name: label
            for label in available
            if "unknown" not in label.name.casefold() and "corpuscallosum" not in label.name.casefold()
        }
        missing = [name for name in DK68_LABELS if name not in by_name]
        extra = sorted(set(by_name) - set(DK68_LABELS))
        if missing or extra or len(by_name) != 68:
            raise ScoutError(f"aparc atlas does not match the stable DK68 contract (missing={missing}, extra={extra})")
        ordered_labels = [by_name[name] for name in DK68_LABELS]
        pick_orientation = None if config.pick_orientation == "none" else config.pick_orientation

        def chunks():
            with h5py.File(canonical_path, "r") as file:
                data = file["data"]
                for start in range(0, sample_count, config.chunk_samples):
                    block = np.asarray(data[0, :, start : start + config.chunk_samples], dtype=np.float64)
                    raw = mne.io.RawArray(block, info.copy(), verbose="ERROR")
                    stc = apply_inverse_raw(
                        raw,
                        inverse,
                        lambda2=config.lambda2,
                        method=config.inverse_method,
                        pick_ori=pick_orientation,
                        buffer_size=config.chunk_samples,
                        verbose="ERROR",
                    )
                    values = mne.extract_label_time_course(
                        stc,
                        ordered_labels,
                        src=inverse["src"],
                        mode=config.label_mode,
                        verbose="ERROR",
                    )
                    values = np.asarray(values, dtype=np.float64)
                    if values.ndim == 3 and values.shape[0] == 1:
                        values = values[0]
                    yield start, values

        provenance = {
            "backend": config.backend,
            "subject": config.subject,
            "subjects_dir": str(config.subjects_dir),
            "trans": str(config.trans),
            "source_space": str(config.source_space),
            "bem": str(config.bem),
            "resource_hashes": [list(value) for value in config.resource_hashes],
            "atlas": config.atlas,
            "label_order": list(DK68_LABELS),
            "inverse_method": config.inverse_method,
            "snr": config.snr,
            "lambda2": config.lambda2,
            "pick_orientation": config.pick_orientation,
            "label_mode": config.label_mode,
            "depth": config.depth,
            "loose": config.loose,
            "mindist_mm": config.mindist_mm,
            "noise_covariance": "streamed empirical covariance across clean continuous samples",
            "average_reference_projection": config.average_reference_projection,
            "coordinate_frame": config.coordinate_frame,
            "source_chunk_samples": config.chunk_samples,
            "hidden_cleaning": False,
            "mne_version": mne.__version__,
        }
        return ScoutStream(DK68_LABELS, times, sfreq, chunks(), provenance)
