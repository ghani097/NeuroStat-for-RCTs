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

The posterior-channel GEDAI versus Traditional PhysioNet comparison is a
preprocessing sensitivity check, not a source-space validation endpoint.

4. OpenNeuro longitudinal replication
   - `openneuro_ds005385_replication.py` downloads a fixed, age/sex-balanced
     12-participant subset of OpenNeuro `ds005385` to a user-supplied directory
     outside Git.
   - It processes pre-cognitive eyes-open and eyes-closed continuous resting EEG
     from baseline and approximately five-year follow-up through the NeuroStat
     Traditional sensor-space path.
   - The pre-registered endpoint and unchanged pass/fail thresholds are recorded
     in `ds005385_replication_plan.json`. Compact checksums, subject results, and
     runtime provenance are written to `provenance/ds005385_v1.0.3/`.

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

Run the OpenNeuro longitudinal replication (PowerShell example):

```powershell
python scripts\validation\openneuro_ds005385_replication.py `
  --data-root "E:\path\outside\the\repository\ds005385-v1.0.3"
```

Use `--download-only` or `--run-only` to resume one stage. The script rejects a
data root located inside the Git worktree and verifies the dataset DOI before
analysis. The default pilot downloads 48 EDF recordings (four per subject) plus
BIDS metadata; pass `--subjects` to expand the analysis deterministically.

The scripts do not store raw PhysioNet data or MNE fsaverage files in this
repository. Those assets are downloaded or cached locally by MNE/PhysioNet tooling.
