# Source scouts and PLI connectivity

NeuroStat v2 has two explicit, scientifically distinct routes from an ingested clean recording to source scouts and connectivity. A profile used by either analysis stage must contain exactly `source` and `connectivity`. Unknown fields, ambiguous labels, unsupported backend pairs, non-finite parameters, Boolean values supplied as numbers, and missing resources are rejected before analysis.

The supported pairs are:

| Clean recording | Source backend | Connectivity backend |
| --- | --- | --- |
| Retained evoked epochs | `fixed_linear_operator` | `evoked_average_hilbert_pli` |
| One continuous MNE-ingested SET/FIF recording | `mne_fsaverage` | `continuous_multitaper_pli` |

Phase-2 ingestion-only projects may retain opaque profiles. The strict scientific profile is required when `scouts` starts. Profile validation is dependency-light and does not import MNE, SciPy, h5py, matplotlib, the GUI, or the v1 pipeline.

## Commands and run boundaries

Run each stage with the upstream run directory printed by the previous command:

```text
neurostat ingest project.yaml --json
neurostat scouts project.yaml --ingest-run outputs/runs/ingest-<digest> --json
neurostat connectivity project.yaml --scouts-run outputs/runs/scouts-<digest> --json
```

The service equivalents are `ingest_project`, `run_scouts`, and `run_connectivity`. They accept the same validated project model and publish `ProgressEvent` values without depending on GUI code.

Every stage verifies that the upstream lock belongs to the same normalized project, source/resource hashes still match, record sets are exact, artifact paths remain inside the supplied run, and artifact hashes and payload digests are intact. Run identities use scientific configuration and stable upstream content; operational reuse counters do not change a downstream identity. Writes are atomic. An existing mismatched or corrupt artifact is refused instead of overwritten. Reuse counts are reported in the return value and status file, not frozen into scientific manifests.

## Fixed operator and V7-compatible evoked PLI

The fixed backend requires a MAT operator and ordered `sensor_labels` and `scout_labels`. The default matrix variable is `SensorToScoutOperator`. Classic MAT and MATLAB v7.3/HDF5 numeric matrices are supported. Rows must be scouts and columns must be sensors. Only the documented physical axis reversal of a MATLAB v7.3 numeric dataset is undone; a transposed classic matrix is not guessed. If optional `SensorLabels` or `ScoutLabels` metadata are present, they must agree exactly with YAML.

This route accepts retained epochs only; a canonical raw recording is rejected. Canonical channels must match the sensor contract exactly. NeuroStat reorders them by label, reads retained epochs in bounded blocks (default 32), accumulates their sum in float64, divides by the retained epoch count, then calculates:

```text
scouts(scouts × samples) =
    operator(scouts × sensors) @ sensor_average(sensors × samples)
```

The V7-compatible connectivity method processes every scout band as follows:

1. `scipy.signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")`
2. `scipy.signal.sosfiltfilt(..., axis=-1)` (zero-phase forward/backward; prototype order 4, realised band-pass order 8)
3. `phase = angle(scipy.signal.hilbert(filtered, axis=-1))`
4. `PLI(i,j) = abs(mean(sign(sin(phase_i - phase_j))))` across all retained samples

The implementation reads and filters one scout row at a time, retains one `nodes × samples` phase array, and calculates each upper-triangle pair with integer sums over bounded sample blocks. It therefore avoids both a second eager HDF5 matrix read and the `nodes × nodes × samples` phase-difference tensor while preserving the exact all-sample formula. The diagonal is exactly zero. There is no 40 ms or other temporal subdivision. The generic manuscript-compatible band declaration is:

```yaml
connectivity:
  backend: evoked_average_hilbert_pli
  filter_order: 4
  bands:
    theta: [3.5, 7.5]
    alpha: [7.5, 12.5]
    beta: [12.5, 30]
    gamma: [30, 40]
```

Filter coefficients, band edges, SciPy/NumPy versions, realised order, sample policy, and formula are stored in the output. See `examples/v2/source-connectivity` for a generic runnable fixed-operator project. No study-specific operator, path, participant identifier, or accepted result is distributed.

## MNE source model and continuous multitaper PLI

`mne_fsaverage` accepts only clean SET/FIF recordings ingested by MNE, whose channel locations are canonical metres. FieldTrip recordings are rejected at this boundary because Phase 2 does not yet carry a typed coordinate-unit contract; millimetres and metres cannot be safely inferred from numeric positions. The backend also requires local resources and never downloads them:

- a FreeSurfer `subjects_dir` with both `<subject>/label/lh.aparc.annot` and `rh.aparc.annot`;
- an explicit head↔MRI transform;
- an explicit source-space FIF;
- an explicit BEM solution FIF;
- finite canonical EEG channel positions in the head coordinate frame.

The relevant profile fields are explicit; paths are resolved relative to the project YAML:

```yaml
source:
  backend: mne_fsaverage
  subject: fsaverage
  subjects_dir: resources/subjects
  trans: resources/fsaverage-trans.fif
  source_space: resources/fsaverage-ico3-src.fif
  bem: resources/fsaverage-bem-sol.fif
  atlas: aparc
  inverse_method: MNE
  snr: 3
  pick_orientation: normal
  label_mode: mean_flip
  chunk_samples: 16384
  output_chunk_samples: 4096
  depth: 0.8
  loose: 0.2
  mindist_mm: 5
  noise_covariance: empirical
  average_reference_projection: true
  coordinate_frame: head
  export_matlab: false
```

The profile freezes hashes for the transform, source space, BEM and annotation files. It also records the subject, atlas, inverse method, SNR and derived `lambda2`, orientation, label extraction mode, depth, looseness, minimum source/sensor distance, empirical covariance policy, average-reference projection assumption, coordinate frame, and source/output chunk sizes.

The streaming-safe label modes are `mean_flip` (default) and `mean`. `mean_flip` uses fixed anatomical sign flips and does not estimate time-dependent weights per chunk. `pca_flip` is deliberately rejected: MNE estimates its spatial weights with an SVD over the supplied time samples, so applying it independently to chunks changes the weights, scaling and boundaries. Supporting PCA-flip safely requires a future global two-pass or fixed-weight implementation.

The backend calculates empirical sensor covariance with bounded reads, constructs the forward and inverse operators once, then applies the inverse to bounded raw chunks. It extracts and writes each chunk incrementally. The accepted DK68 order is explicit and hemisphere-interleaved (`bankssts-lh`, `bankssts-rh`, …, `transversetemporal-lh`, `transversetemporal-rh`). Unknown and corpus-callosum labels are excluded; any missing or extra cortical label fails the run.

Continuous connectivity reads the scout HDF5 as a generator of non-overlapping windows. The default is 4 seconds. `window_seconds × sfreq` must be an integer number of samples. The default and only current tail policy is `drop`; its dropped sample count is recorded, and zero complete windows is an error. Windows are passed directly to:

```text
mne_connectivity.spectral_connectivity_epochs(
    method="pli", mode="multitaper", faverage=True, ...)
```

Band tuples, sampling rate, taper settings and job count are explicit. Dense one-triangle output is normalized to a symmetric matrix with an exactly zero diagonal.

## Artifacts

Scout files use schema `neurostat.scouts.v1`:

- float64 `data` shaped `scouts × samples`, chunked on time;
- ordered UTF-8 `labels`, float64 `times`, and `sfreq`;
- record/study keys, source backend, canonical hash and payload digest;
- profile digest and JSON source provenance;
- a payload digest covering all values and provenance.

An opt-in `export_matlab: true` writes a generic MAT export with `Value`, `Time`, `Labels`, `SamplingRateHz`, `RecordID`, and `Schema`. It is intentionally off by default because long scout matrices can be large.

Connectivity files use schema `neurostat.connectivity.v1`:

- float64 `matrix` shaped `bands × nodes × nodes`;
- ordered labels, band names and band edges;
- SOS coefficients for the evoked Hilbert route;
- upstream scout hashes, profile digest and exact method provenance;
- a payload digest and matrix checks for finiteness, symmetry, zero diagonal and `[0,1]`.

## Scientific non-goals

These stages do not clean, rereference, reject epochs, interpolate sensors, filter broadband data, resample, infer coordinate units, or infer channel/operator correspondence by column position. They do not fetch fsaverage, choose a transform, generate a BEM implicitly, substitute missing atlas nodes, estimate PCA-flip weights independently per chunk, pad an incomplete connectivity window, or silently change backend combinations.
