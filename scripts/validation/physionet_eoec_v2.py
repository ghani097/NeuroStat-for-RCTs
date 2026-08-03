#!/usr/bin/env python
"""Confirm a published source-PLI eye-state effect with public EEG and V2.

The discovery cohort is EEGMMIDB S001-S020. The independent confirmation
cohort is S021-S109. The primary endpoint is within-participant change in
global DK68 alpha PLI from eyes open to eyes closed (EC > EO).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validation import lemon_discover_eeg_v2 as shared
from scripts.validation.physionet_validation import (
    PREPROC_CFG,
    _load_eegbci,
)

DATASET_DOI = "10.13026/C28G6P"
DATASET_VERSION = "1.0.0"
DATASET_URL = "https://physionet.org/content/eegmmidb/1.0.0/"
PUBLISHED_DOI = "10.1038/s41598-024-68532-2"
DISCOVERY_SUBJECTS = tuple(range(1, 21))
CONFIRMATION_SUBJECTS = tuple(range(21, 110))
SESSIONS = {"eyes_open": 1, "eyes_closed": 2}
BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 39.5),
}
POSTERIOR_BASE = (
    "cuneus",
    "lateraloccipital",
    "lingual",
    "pericalcarine",
    "precuneus",
    "superiorparietal",
    "inferiorparietal",
)
POSTERIOR_LABELS = tuple(
    f"{base}-{hemisphere}" for base in POSTERIOR_BASE for hemisphere in ("lh", "rh")
)
PERMUTATIONS = 100_000
RANDOM_SEED = 20260804


def cohort_subjects(cohort: str) -> tuple[int, ...]:
    return DISCOVERY_SUBJECTS if cohort == "discovery" else CONFIRMATION_SUBJECTS


def _sign_flip_p(differences: np.ndarray, seed: int) -> float:
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(PERMUTATIONS):
        signs = rng.choice((-1.0, 1.0), size=len(differences))
        count += float(np.mean(differences * signs)) >= observed - 1e-15
    return (count + 1) / (PERMUTATIONS + 1)


def _paired_d(differences: np.ndarray) -> float:
    sd = differences.std(ddof=1)
    return float(differences.mean() / sd) if sd else math.nan


def prepare_dataset(root: Path, cohort: str) -> Path:
    from pli_pipeline import preprocess_raw

    prepared = root / "prepared" / cohort
    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for subject in cohort_subjects(cohort):
        for session, run in SESSIONS.items():
            destination = (
                prepared
                / f"S{subject:03d}"
                / f"S{subject:03d}_{session}_cleaned_raw.fif"
            )
            qc_path = destination.with_suffix(".qc.json")
            if destination.is_file() and qc_path.is_file():
                qc = json.loads(qc_path.read_text(encoding="utf-8"))
                if qc.get("output_sha256") != shared._sha256(destination):
                    raise RuntimeError(f"prepared checksum mismatch: {destination}")
            else:
                raw = _load_eegbci(subject, run, root / "raw")
                source_filename = raw.filenames[0]
                if source_filename is None:
                    raise RuntimeError(f"S{subject:03d} {session} has no source file")
                source_path = Path(source_filename)
                preprocessed = cast(
                    tuple[Any, Any, Any, dict[str, Any]],
                    preprocess_raw(raw, PREPROC_CFG),
                )
                _, cleaned, _, cleaning = preprocessed
                if (
                    len(cleaned.copy().pick("eeg").ch_names) < 60
                    or cleaned.n_times / cleaned.info["sfreq"] < 59
                ):
                    raise RuntimeError(
                        f"S{subject:03d} {session} failed input eligibility"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                cleaned.save(destination, overwrite=True, verbose="ERROR")
                qc = {
                    "schema": "neurostat.physionet_eoec_cleaning.v1",
                    "subject": subject,
                    "session": session,
                    "source_path": str(source_path.resolve()),
                    "source_sha256": shared._sha256(source_path),
                    "output_sha256": shared._sha256(destination),
                    "channels": len(cleaned.ch_names),
                    "sfreq_hz": float(cleaned.info["sfreq"]),
                    "duration_seconds": cleaned.n_times / float(cleaned.info["sfreq"]),
                    "cleaning": cleaning,
                    "contract": {
                        "bandpass_hz": [PREPROC_CFG.l_freq, PREPROC_CFG.h_freq],
                        "ica_method": PREPROC_CFG.ica_method,
                        "ica_components": PREPROC_CFG.ica_n_components,
                        "ica_random_state": PREPROC_CFG.random_state,
                        "iclabel": PREPROC_CFG.use_iclabel,
                        "asr": PREPROC_CFG.use_asr,
                    },
                }
                shared._atomic_json(qc_path, qc)
            rows.append(
                {
                    "record_id": f"S{subject:03d}-{session}",
                    "participant_id": f"S{subject:03d}",
                    "arm": "validation",
                    "session": session,
                    "condition": "rest",
                    "data_kind": "raw",
                    "format": "fif",
                    "path": str(destination.resolve()),
                    "analysis_profile": "rest",
                    "data_unit": "V",
                }
            )
            qc_rows.append(
                {
                    "participant_id": f"S{subject:03d}",
                    "session": session,
                    "source_sha256": qc["source_sha256"],
                    "output_sha256": qc["output_sha256"],
                    "channels": qc["channels"],
                    "sfreq_hz": qc["sfreq_hz"],
                    "duration_seconds": qc["duration_seconds"],
                }
            )
            print(f"prepared S{subject:03d} {session}", flush=True)
    prepared.mkdir(parents=True, exist_ok=True)
    for name, values in (("records.csv", rows), ("input_qc.csv", qc_rows)):
        with (prepared / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    return prepared / "records.csv"


def build_project(root: Path, cohort: str) -> Path:
    import mne
    import yaml

    from neurostat.gui.fsaverage import provision_fsaverage
    from neurostat.project import load_project

    source = root / "prepared" / cohort / "records.csv"
    if not source.is_file():
        raise RuntimeError(f"missing prepared records: {source}")
    project_dir = root / "projects" / cohort
    project_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(source.open(encoding="utf-8", newline="")))
    for row in rows:
        row["path"] = shared._relative_to_project(project_dir, Path(row["path"]))
    with (project_dir / "records.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    resources = provision_fsaverage(progress=print)
    source_space = root / "resources" / "fsaverage-ico3-src.fif"
    if not source_space.is_file():
        source_space.parent.mkdir(parents=True, exist_ok=True)
        src = mne.setup_source_space(
            resources.subject,
            spacing="ico3",
            subjects_dir=resources.subjects_dir,
            add_dist=False,
            verbose="ERROR",
        )
        mne.write_source_spaces(source_space, src, overwrite=True, verbose="ERROR")
    project = {
        "schema_version": 1,
        "study": {
            "id": "physionet-eoec-v2-confirmation",
            "arms": ["validation", "unused_comparison"],
            "sessions": list(SESSIONS),
            "conditions": ["rest"],
        },
        "input": {"state": "cleaned"},
        "records": {"manifest": "records.csv"},
        "analysis_profiles": {
            "rest": {
                "source": {
                    "backend": "mne_fsaverage",
                    "subject": resources.subject,
                    "subjects_dir": shared._relative_to_project(
                        project_dir, resources.subjects_dir
                    ),
                    "trans": shared._relative_to_project(project_dir, resources.trans),
                    "source_space": shared._relative_to_project(
                        project_dir, source_space
                    ),
                    "bem": shared._relative_to_project(project_dir, resources.bem),
                    "atlas": "aparc",
                    "inverse_method": "MNE",
                    "snr": 3.0,
                    "pick_orientation": "normal",
                    "label_mode": "mean_flip",
                    "chunk_samples": 8192,
                    "output_chunk_samples": 4096,
                    "depth": 0.8,
                    "loose": 0.2,
                    "mindist_mm": 5.0,
                    "noise_covariance": "empirical",
                    "average_reference_projection": True,
                    "coordinate_frame": "head",
                    "export_matlab": False,
                },
                "connectivity": {
                    "backend": "continuous_multitaper_pli",
                    "bands": {name: list(edges) for name, edges in BANDS.items()},
                    "window_seconds": 2.0,
                    "tail_policy": "drop",
                    "mt_bandwidth": 2.0,
                    "mt_adaptive": False,
                    "mt_low_bias": True,
                    "n_jobs": 1,
                },
            }
        },
        "output": {"root": "output"},
    }
    project_path = project_dir / "project.yaml"
    project_path.write_text(yaml.safe_dump(project, sort_keys=False), encoding="utf-8")
    load_project(project_path)
    shared._atomic_json(
        project_dir / "validation_protocol.json",
        {
            "schema": "neurostat.physionet_eoec_protocol.v1",
            "frozen_before_confirmation_outcomes": True,
            "dataset": {
                "doi": DATASET_DOI,
                "version": DATASET_VERSION,
                "url": DATASET_URL,
            },
            "published_target": {
                "doi": PUBLISHED_DOI,
                "finding": "source-level DK68 alpha PLI network strength decreases on eye opening and increases on eye closing",
                "method_overlap": "source reconstruction, DK68 parcellation, PLI, alpha 8-13 Hz",
            },
            "cohort": cohort,
            "subjects": list(cohort_subjects(cohort)),
            "split": "S001-S020 discovery; S021-S109 independent confirmation",
            "primary_endpoint": "within-participant eyes-closed minus eyes-open global mean upper-triangle DK68 alpha PLI",
            "primary_hypothesis": "eyes closed > eyes open",
            "primary_test": {
                "method": "one-sided sign-flip permutation",
                "permutations": PERMUTATIONS,
                "seed": RANDOM_SEED,
                "alpha": 0.05,
                "pass": "mean difference > 0 and p < 0.05",
            },
            "supporting_endpoint": "posterior DK14 alpha PLI eyes closed > eyes open",
            "arm_contract_note": "V2 requires two declared RCT arms. unused_comparison has no records; no EEG is duplicated or relabeled.",
            "connectivity": {
                "method": "V2 continuous multitaper PLI",
                "bands_hz": BANDS,
                "window_seconds": 2.0,
            },
        },
    )
    return project_path


def run_v2(project_path: Path) -> Path:
    import time

    from neurostat.workflow import run_workflow

    started = time.time()
    result = run_workflow(
        project_path,
        ["connectivity"],
        progress=lambda event: print(
            f"[{event.stage.value}] {event.message}", flush=True
        ),
    )
    target = project_path.parent / "workflow_result.json"
    shared._atomic_json(
        target,
        {
            "schema": "neurostat.physionet_eoec_workflow_result.v1",
            "started_unix": started,
            "finished_unix": time.time(),
            "success": result.success,
            "stages": [
                {
                    "stage": item.stage.value,
                    "state": item.state.value,
                    "run_id": item.run_id,
                    "run_dir": str(item.run_dir) if item.run_dir else None,
                    "reused": item.reused,
                }
                for item in result.outcomes
            ],
        },
    )
    if not result.success:
        raise RuntimeError(f"V2 workflow failed; inspect {target}")
    return target


def summarize(project_path: Path) -> Path:
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from neurostat.connectivity import verify_connectivity_artifact
    from neurostat.project import load_project

    workflow = json.loads(
        (project_path.parent / "workflow_result.json").read_text(encoding="utf-8")
    )
    stage = next(item for item in workflow["stages"] if item["stage"] == "connectivity")
    manifest_path = Path(stage["run_dir"]) / "connectivity" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {item["record_id"]: Path(item["path"]) for item in manifest["records"]}
    project = load_project(project_path)
    rows: list[dict[str, Any]] = []
    matrices: dict[tuple[str, str, str], np.ndarray] = {}
    labels: tuple[str, ...] | None = None
    for record in project.records:
        path = artifacts[record.record_id]
        header = verify_connectivity_artifact(path)
        labels = header.labels
        posterior = [header.labels.index(name) for name in POSTERIOR_LABELS]
        with h5py.File(path, "r") as handle:
            values = np.asarray(handle["matrix"][:], dtype=float)
        global_triangle = np.triu_indices(len(header.labels), 1)
        posterior_triangle = np.triu_indices(len(posterior), 1)
        for index, band in enumerate(header.bands):
            matrix = values[index]
            posterior_matrix = matrix[np.ix_(posterior, posterior)]
            matrices[(record.key.participant_id, record.key.session, band)] = matrix
            rows.append(
                {
                    "participant_id": record.key.participant_id,
                    "session": record.key.session,
                    "band": band,
                    "global_mean_pli": float(matrix[global_triangle].mean()),
                    "posterior_mean_pli": float(
                        posterior_matrix[posterior_triangle].mean()
                    ),
                    "artifact_sha256": shared._sha256(path),
                }
            )
    if labels is None:
        raise RuntimeError("no connectivity artifacts")
    results_dir = project_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "subject_band_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries: list[dict[str, Any]] = []
    participants = sorted({row["participant_id"] for row in rows})
    for index, band in enumerate(BANDS):
        for scope in ("global", "posterior"):
            lookup = {
                (row["participant_id"], row["session"]): row[f"{scope}_mean_pli"]
                for row in rows
                if row["band"] == band
            }
            differences = np.asarray(
                [
                    lookup[(participant, "eyes_closed")]
                    - lookup[(participant, "eyes_open")]
                    for participant in participants
                ]
            )
            summaries.append(
                {
                    "band": band,
                    "scope": scope,
                    "n": len(differences),
                    "eyes_open_mean": float(
                        np.mean(
                            [
                                lookup[(participant, "eyes_open")]
                                for participant in participants
                            ]
                        )
                    ),
                    "eyes_closed_mean": float(
                        np.mean(
                            [
                                lookup[(participant, "eyes_closed")]
                                for participant in participants
                            ]
                        )
                    ),
                    "difference_ec_minus_eo": float(differences.mean()),
                    "cohens_dz": _paired_d(differences),
                    "one_sided_sign_flip_p": _sign_flip_p(
                        differences, RANDOM_SEED + index * 2 + (scope == "posterior")
                    ),
                    "direction_ec_gt_eo": int(np.sum(differences > 0)),
                }
            )
    with (results_dir / "band_paired_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    primary = next(
        item
        for item in summaries
        if item["band"] == "alpha" and item["scope"] == "global"
    )
    support = next(
        item
        for item in summaries
        if item["band"] == "alpha" and item["scope"] == "posterior"
    )
    ec_alpha = np.stack(
        [
            matrices[(participant, "eyes_closed", "alpha")]
            for participant in participants
        ]
    )
    eo_alpha = np.stack(
        [matrices[(participant, "eyes_open", "alpha")] for participant in participants]
    )
    difference_matrix = (ec_alpha - eo_alpha).mean(axis=0)
    payload = {
        "schema": "neurostat.physionet_eoec_results.v1",
        "dataset_doi": DATASET_DOI,
        "published_target_doi": PUBLISHED_DOI,
        "cohort": project_path.parent.name,
        "participants": len(participants),
        "records_verified": len(artifacts),
        "summaries": summaries,
        "primary": primary,
        "supporting_posterior": support,
        "primary_replication_passed": bool(
            primary["difference_ec_minus_eo"] > 0
            and primary["one_sided_sign_flip_p"] < 0.05
        ),
        "connectivity_manifest_sha256": shared._sha256(manifest_path),
        "source_data_included": False,
    }
    shared._atomic_json(results_dir / "validation_results.json", payload)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))
    colours = ["#147d92", "#c6533d"]
    for axis, scope, title in (
        (axes[0, 0], "global", "Primary: global DK68 alpha PLI"),
        (axes[0, 1], "posterior", "Supporting: posterior DK14 alpha PLI"),
    ):
        lookup = {
            (row["participant_id"], row["session"]): row[f"{scope}_mean_pli"]
            for row in rows
            if row["band"] == "alpha"
        }
        eo = np.asarray(
            [lookup[(participant, "eyes_open")] for participant in participants]
        )
        ec = np.asarray(
            [lookup[(participant, "eyes_closed")] for participant in participants]
        )
        for left, right in zip(eo, ec):
            axis.plot([0, 1], [left, right], color="#9aa1a6", alpha=0.3, linewidth=0.7)
        axis.scatter(np.zeros(len(eo)), eo, color=colours[0], s=17, alpha=0.65)
        axis.scatter(np.ones(len(ec)), ec, color=colours[1], s=17, alpha=0.65)
        summary = primary if scope == "global" else support
        axis.set_xticks([0, 1], ["Eyes open", "Eyes closed"])
        axis.set_ylabel("Mean source alpha PLI")
        axis.set_title(
            f"{title}\ndz = {summary['cohens_dz']:.2f}, p = {summary['one_sided_sign_flip_p']:.5f}"
        )
    global_summaries = [item for item in summaries if item["scope"] == "global"]
    axes[1, 0].bar(
        [item["band"].title() for item in global_summaries],
        [item["cohens_dz"] for item in global_summaries],
        color=["#9aa1a6", "#c6533d", "#9aa1a6", "#9aa1a6"],
    )
    axes[1, 0].axhline(0, color="#202020", linewidth=0.8)
    axes[1, 0].set_ylabel("Paired Cohen's dz (EC - EO)")
    axes[1, 0].set_title("Band specificity")
    vmax = max(
        0.01,
        float(
            np.percentile(
                np.abs(difference_matrix[np.triu_indices(len(labels), 1)]), 98
            )
        ),
    )
    image = axes[1, 1].imshow(
        difference_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest"
    )
    axes[1, 1].set_title("Mean alpha PLI change: eyes closed - eyes open")
    axes[1, 1].set_xlabel("DK68 parcel")
    axes[1, 1].set_ylabel("DK68 parcel")
    fig.colorbar(image, ax=axes[1, 1], label="PLI difference", shrink=0.82)
    status = "PASS" if payload["primary_replication_passed"] else "NOT REPLICATED"
    fig.suptitle(
        f"NeuroStat V2 public confirmation: source-PLI eye-state effect [{status}]",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(
        results_dir / "physionet_eoec_v2_dashboard.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    return results_dir / "validation_results.json"


def publish_results(project_path: Path) -> Path:
    if project_path.parent.name != "confirmation":
        raise RuntimeError("only the independent confirmation cohort can be published")
    source = project_path.parent / "results"
    target = PROJECT_ROOT / "scripts" / "validation" / "physionet_eoec_v2" / "results"
    target.mkdir(parents=True, exist_ok=True)
    names = (
        "subject_band_results.csv",
        "band_paired_summary.csv",
        "physionet_eoec_v2_dashboard.png",
        "validation_results.json",
    )
    for name in names:
        shutil.copy2(source / name, target / name)
    shutil.copy2(
        project_path.parent / "validation_protocol.json",
        target / "validation_protocol.json",
    )
    result = json.loads(
        (target / "validation_results.json").read_text(encoding="utf-8")
    )
    primary = result["primary"]
    status = "PASSED" if result["primary_replication_passed"] else "DID NOT PASS"
    (target / "README.md").write_text(
        f"# PhysioNet Eye-State V2 Confirmation\n\n![Validation dashboard](physionet_eoec_v2_dashboard.png)\n\n"
        f"**Independent primary criterion: {status}.** In {primary['n']} participants, "
        f"eyes-open mean global alpha PLI was {primary['eyes_open_mean']:.6f} and "
        f"eyes-closed mean was {primary['eyes_closed_mean']:.6f}. The paired change was "
        f"{primary['difference_ec_minus_eo']:.6f} (dz = {primary['cohens_dz']:.3f}; "
        f"one-sided 100,000-draw sign-flip p = {primary['one_sided_sign_flip_p']:.6f}; "
        f"{primary['direction_ec_gt_eo']}/{primary['n']} participants in the predicted direction).\n\n"
        f"This independently confirms the direction reported in DOI `{PUBLISHED_DOI}` using "
        f"public PhysioNet EEGMMIDB v{DATASET_VERSION} (DOI `{DATASET_DOI}`) and the actual "
        "NeuroStat V2 DK68 source-connectivity workflow. The published study's own recordings "
        "were not public, so this is an independent conceptual confirmation rather than an "
        "exact numerical reproduction. Raw EEG and intermediate artifacts are excluded from Git.\n",
        encoding="utf-8",
        newline="\n",
    )
    published = [*names, "validation_protocol.json", "README.md"]
    shared._atomic_json(
        target / "publication_manifest.json",
        {
            "schema": "neurostat.physionet_eoec_publication_bundle.v1",
            "dataset_doi": DATASET_DOI,
            "published_target_doi": PUBLISHED_DOI,
            "files": {name: shared._sha256(target / name) for name in published},
        },
    )
    return target


def _default_root() -> Path:
    configured = os.environ.get("NEUROSTAT_PHYSIONET_VALIDATION_DIR")
    return (
        Path(configured).expanduser()
        if configured
        else PROJECT_ROOT.parent / "NeuroStat-v2-public-validation" / "physionet_eoec"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "project", "run", "summarize", "publish", "all")
    )
    parser.add_argument(
        "--cohort", choices=("discovery", "confirmation"), default="discovery"
    )
    parser.add_argument("--data-root", type=Path, default=_default_root())
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    project_path = root / "projects" / args.cohort / "project.yaml"
    if args.action in {"prepare", "all"}:
        prepare_dataset(root, args.cohort)
    if args.action in {"project", "all"}:
        project_path = build_project(root, args.cohort)
    if args.action in {"run", "all"}:
        run_v2(project_path)
    if args.action in {"summarize", "all"}:
        summarize(project_path)
    if args.action == "publish" or (
        args.action == "all" and args.cohort == "confirmation"
    ):
        publish_results(project_path)
    print(f"completed {args.action} for {args.cohort}: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
