# Validation Analyses

This folder contains validation scripts used to support the NeuroStat manuscript.
The analyses are intentionally separated by the claim they test.

## Evidence Streams

1. Sensor-space simulation
   - Existing validation scripts test whether preprocessing preserves known
     sensor-level PLI patterns under artifact contamination.
   - These outputs should not be interpreted as source-localization validation.

2. Source-space simulation
   - `source_space_simulation.py` tests whether the BEM/MNE/Desikan-Killiany
     source-space workflow can recover a known cortical parcel-level contrast.
   - The script generates alpha activity in bilateral lateral-occipital parcels,
     forward-projects it to scalp EEG, reconstructs source time courses, and
     compares the known parcel pair against anterior/non-posterior control edges.

3. PhysioNet source-space benchmark
   - `physionet_source_space_iclabel.py` tests whether the source-space pipeline
     detects the expected posterior Desikan-Killiany alpha increase during
     eyes-closed versus eyes-open rest in healthy adults.
   - This is a short-recording benchmark using modified settings: 15 ICA
     components, ASR disabled, ICLabel active with brain threshold 0.50, ico3
     source spacing, and 2-second epochs with 50% overlap.

4. Public longitudinal v2 workflow validation
   - `openneuro_ds006801_v2.py` uses the CC0 OpenNeuro dataset
     [ds006801](https://openneuro.org/datasets/ds006801), version 1.0.0 and DOI
     `10.18112/openneuro.ds006801.v1.0.0`.
   - The cohort is fixed in the script before NeuroStat results are read: all
     six active-learning participants and all seven controls, at pre and post.
   - It downloads checksum-pinned continuous BrainVision recordings, prepares
     cleaned FIF inputs with explicit QC, generates a v2 project, runs the
     shared high-level engine through source connectivity, verifies every
     artifact, and exports subject-level and group-change summaries.
   - The primary claim is reproducible two-arm longitudinal workflow execution
     and artifact integrity. Group-by-time PLI estimates are exploratory because
     no directional connectivity hypothesis was preregistered for this dataset.

The posterior-channel GEDAI versus Traditional PhysioNet comparison is a
preprocessing sensitivity check, not a source-space validation endpoint.

## Requirements

Install the project dependencies from the repository root:

```powershell
pip install -r requirements.txt
```

ICLabel requires `mne-icalabel` and a PyTorch runtime. The validation scripts also
use MNE's fsaverage assets, which MNE downloads/caches locally when needed.

## Run Commands

From the repository root:

```powershell
python scripts\validation\source_space_simulation.py --n-repeats 20
```

Main outputs:

- `scripts/validation/source_space_simulation/results/source_space_simulation_results.csv`
- `scripts/validation/source_space_simulation/results/source_space_simulation_summary_by_condition.csv`
- `scripts/validation/source_space_simulation/results/source_space_simulation_summary.json`
- `scripts/validation/source_space_simulation/results/source_space_simulation_recovery.png`

Run the PhysioNet source-space benchmark:

```powershell
python scripts\validation\physionet_source_space_iclabel.py --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
```

Main outputs:

- `scripts/validation/physionet_data/source_space_iclabel/physionet_source_subject_results.csv`
- `scripts/validation/physionet_data/source_space_iclabel/physionet_source_results.json`
- `scripts/validation/physionet_data/source_space_iclabel/physionet_source_paired.png`

The scripts do not store raw PhysioNet data or MNE fsaverage files in this
repository. Those assets are downloaded or cached locally by MNE/PhysioNet tooling.

## OpenNeuro ds006801 v2 validation

Run a two-participant pilot first. By default, data and generated outputs are
kept in a sibling `NeuroStat-v2-public-validation/ds006801` directory outside
Git:

```powershell
python scripts\validation\openneuro_ds006801_v2.py all --cohort pilot
```

After the pilot passes, run the fixed full cohort (13 participants, 26
recordings, roughly 0.9 GB of source downloads):

```powershell
python scripts\validation\openneuro_ds006801_v2.py all --cohort full
```

Each action is resumable and can also be run separately: `download`, `prepare`,
`project`, `run`, `summarize`, or `publish`. `publish` accepts only the fixed
full cohort and refreshes the portable tracked bundle under
`scripts/validation/openneuro_ds006801/results`. Set
`NEUROSTAT_PUBLIC_VALIDATION_DIR` or pass
`--data-root` to choose another external data directory. The downloader resolves
file size and SHA-256 from pinned OpenNeuro Git snapshot
`c4e91b19132fed86ee24a42ec5105bf14f961011` and refuses mismatched content.

The preparation protocol is fixed at 200 Hz, 1-45 Hz, applied average reference,
seed-42 extended-infomax ICA, ICLabel artifact-class removal, and exact 4-second
window QC. A recording fails rather than being admitted when channel geometry,
classification, finite-data, or retained-duration requirements are not met.

Main full-cohort outputs:

- `projects/full/project.yaml`
- `projects/full/workflow_result.json`
- `projects/full/results/subject_band_results.csv`
- `projects/full/results/band_change_summary.csv`
- `projects/full/results/validation_results.json`
- `projects/full/results/global_pli_pre_post.png`
