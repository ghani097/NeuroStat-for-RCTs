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
