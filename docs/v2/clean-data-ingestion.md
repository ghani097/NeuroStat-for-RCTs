# Clean-data project and ingestion

`neurostat validate PROJECT` validates schema version 1 without importing scientific readers. `neurostat ingest PROJECT` reads an externally cleaned input and writes a resumable run. The input section must be exactly `input: {state: cleaned}`: ingestion deliberately performs no filtering, resampling, rereferencing, rejection, ICA, interpolation, or other cleaning.

Projects declare a two-arm study, at least two sessions, conditions, profiles, output root, and records inline or in CSV/TSV. Supported explicit format/kind combinations are `set`, `fif`, and `fieldtrip_mat`, each with `raw` or `epochs`, and `bdf`, `edf`, `brainvision`, `poly5`, `xdf` and `wibci_csv` with `raw` (see [input formats](input-formats.md) for `channel_map`, `stream` and `sampling_rate`). Paths are relative to YAML for inline records and to the manifest for tabular records.

Canonical artifacts use HDF5 schema `neurostat.clean_recording.v1`, float64 `(epochs, channels, samples)` data, UTF-8 labels, times, optional events/positions, source provenance, units, dimensions, and the `externally_cleaned_bypass` marker. MNE SET/FIF inputs are volts; FieldTrip values may declare `V`, `mV`, `uV`, or `native`.

FieldTrip readers support classic MAT and MATLAB v7.3/HDF5 cell-trial structures, including a concatenated trial split with `samples_per_epoch` or top-level `pnts`. Runs live at `output/runs/ingest-<digest>/`; the digest freezes validated configuration and every source-bundle hash. Repeated unchanged inputs validate and reuse artifacts, while corruption is refused rather than overwritten. EEGLAB companion `.fdt` content participates in provenance.

Use `--json` on either command for stable machine-readable summaries.
