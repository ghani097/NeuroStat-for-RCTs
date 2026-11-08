# Input formats

NeuroStat reads continuous recordings from the common EEG amplifiers and file
formats. The format is detected from the file extension, so a study folder can
hold any of them.

| Format | Extension | Reader | Notes |
|---|---|---|---|
| EEGLAB | `.set` (+ `.fdt`) | MNE | raw or epochs |
| MNE FIFF | `.fif`, `.fif.gz` | MNE | raw or epochs |
| FieldTrip | `.mat` | NeuroStat | raw or epochs, cleaned input only |
| BioSemi | `.bdf` | MNE | `Status` low 16 bits become events |
| EDF / EDF+ | `.edf` | MNE | EDF+ annotations kept |
| BrainVision | `.vhdr` (+ `.vmrk`, `.eeg`) | MNE | all three files are hashed for provenance |
| TMSi Poly5 | `.Poly5` | NeuroStat | version 2.03 float32, `Digi` becomes events |
| LSL XDF | `.xdf`, `.xdfz` | pyxdf | needs `pip install neurostat-rcts[formats]` |
| WiBCI headset | `.csv` | NeuroStat | 250 Hz by default, packet gaps filled |

Every reader returns volts. Non-EEG channels (triggers, BIP, AUX, accelerometer)
are dropped after their events are copied into annotations. Channels that match
the standard 10-05 cap (or the BioSemi 128/256 caps) get positions, which the
MNE fsaverage source route needs.

## Channel labels

Amplifiers often store labels such as `ExG1` or `A1` rather than 10-20 names.
The channel map turns them into 10-20 names before positions are set.

| Map | Use |
|---|---|
| `auto` (default) | picks one of the maps below from the labels in the file |
| `tmsi64` | TMSi REFA 64-channel cap, `ExG1..ExG64`, M1 and M2 dropped (UK and ANN studies) |
| `biosemi32`, `biosemi64` | BioSemi `A1..B32` in cap order |
| `none` | keep the labels as they are |
| a `.tsv` or `.csv` file | two columns, file label then new label (blank drops the channel) |

`auto` also strips prefixes and suffixes such as `EEG Fp1-REF` when that gives
unique names.

## WiBCI CSV

WiBCI exports store absolute values in the first row and differences after it.
The reader follows `loadWiBCIData_UG.m`: later rows add one packet, add 4000 us to
each timestamp and scale EEG differences by 0.02235 uV, then the table is summed.
Missing packets are inserted and filled by pchip interpolation, and the share of
filled samples is reported. `Event A` and `Event B` values of 100 or more are
software events and values from 1 to 99 are hardware events.

The MATLAB converter keeps 10 of the 15 channels. NeuroStat keeps all 15 and
leaves channel rejection to preprocessing.

## Where the formats are used

- **Raw data (GUI, data not yet cleaned).** v1 preprocessing reads only
  `<group>/<session>/*.set`. Any other format or layout is first converted into
  `<output>/staged_raw/<group>/<session>/<name>.set`, with a
  `<name>.provenance.json` holding the source path, its SHA-256, the format and
  the reader options. A rerun skips files whose source and options have not
  changed. Studies that already have the flat `.set` layout run exactly as before.
- **Cleaned data (project files).** Records may use `format: bdf`, `edf`,
  `brainvision`, `poly5`, `xdf` or `wibci_csv` with `data_kind: raw`. Optional
  record fields are `channel_map`, `stream` (XDF) and `sampling_rate` (WiBCI).
  `data_unit` may be set for XDF (`V`, `mV`, `uV`) and WiBCI (`uV`).
- **One-off conversion.**

  ```
  neurostat convert recording.Poly5 --to set --out converted/
  neurostat convert *.csv --sampling-rate 500 --to fif
  ```

The GUI **Check files** button reads the first file of each group and lists the
channels, rate, duration, events and any labels without a standard position,
so a wrong channel map shows up before a long run.

## Not supported

- delta-compressed or 16-bit Poly5 files (the reader stops with a clear error)
- epoched data in the new formats (epochs are made downstream)
- other CSV layouts (only WiBCI exports are recognised, so result CSVs in a study
  folder are ignored)
