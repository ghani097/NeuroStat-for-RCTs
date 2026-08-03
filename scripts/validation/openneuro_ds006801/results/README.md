# OpenNeuro ds006801 NeuroStat v2 results

This portable bundle records the fixed full-cohort public validation:

- OpenNeuro `ds006801`, version 1.0.0
- DOI `10.18112/openneuro.ds006801.v1.0.0`
- pinned dataset snapshot `c4e91b19132fed86ee24a42ec5105bf14f961011`
- all six active-learning participants and all seven controls
- pre and post mixed eyes-open/eyes-closed resting-state recordings
- 26 cleaned recordings projected to DK68 source space
- theta, alpha, and beta continuous multitaper PLI

The primary validation claim is reproducible V2 execution and artifact
integrity. The group-by-time statistics are exploratory and do not establish
intervention efficacy. No directional connectivity hypothesis was preregistered
for this dataset, and no band survives exact-permutation FDR correction.

Raw BrainVision EEG, cleaned FIF files, scout time series, and connectivity HDF5
artifacts are intentionally excluded from Git. Recreate them with:

```powershell
python scripts\validation\openneuro_ds006801_v2.py all --cohort full
```

`publication_manifest.json` freezes the SHA-256 of every tracked result artifact.
