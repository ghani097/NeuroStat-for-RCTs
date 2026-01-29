# Methods

## EEG Data Acquisition and Preprocessing

### Data Format and Initial Processing

Continuous EEG data stored in EEGLAB format (.set) were imported into Python using the MNE-Python library. Data were organized hierarchically by experimental group and recording session to facilitate batch processing. Channel locations were standardized to the international 10-20 system using the standard montage template.

### Bad Channel Detection and Interpolation

Bad channels were identified using a variance-based detection algorithm. Channel-wise variance was computed across the entire recording, and channels exhibiting variance exceeding 4.0 standard deviations from the mean were flagged as artifactual. Identified bad channels were subsequently interpolated using spherical spline interpolation to maintain electrode density for subsequent processing steps.

### Artifact Removal

Two artifact removal pipelines are available, selectable via the GUI:

#### Option A: Traditional Pipeline (ASR + ICLabel)

Artifact rejection was implemented using Artifact Subspace Reconstruction (ASR), a method that identifies high-variance subspaces in the data corresponding to transient high-amplitude artifacts. The algorithm learns the statistical properties of clean EEG segments and reconstructs contaminated segments by rejecting signal components that deviate beyond a cutoff parameter of 20 standard deviations from the learned clean signal statistics. This approach preserves underlying brain activity while removing non-stationary high-amplitude artifacts including movement, electrode pops, and muscle bursts.

#### Option B: GEDAI Pipeline (Official Implementation)

The GEDAI (Generalized Eigenvalue De-Artifacting Instrument) pipeline uses the official Python implementation from the neurotuning lab (https://github.com/neurotuning/gedai, version 0.1.0). This method employs leadfield-based spatial filtering to separate brain signals from artifacts.

**Algorithm Overview:**

1. **Leadfield-Based Reference Covariance:** GEDAI uses an EEG forward model (leadfield matrix) to define the expected spatial patterns of brain-generated electrical activity. This physics-based approach provides a principled reference for distinguishing neural from artifactual sources.

2. **Generalized Eigenvalue Decomposition (GEVD):** The data covariance matrix is decomposed relative to the leadfield-based reference covariance, yielding components ranked by their "brain-likeness."

3. **SENSAI Threshold Selection:** The Signal & Noise Subspace Alignment Index (SENSAI) quantifies the alignment between each component and the theoretical brain signal subspace. A grid search optimization determines the optimal threshold for artifact rejection.

4. **Wavelet Multi-Band Processing:** GEDAI processes each frequency band separately using maximal overlap discrete wavelet transform (MODWT), enabling frequency-specific artifact removal.

**Parameters:**
- Epoch duration: 2.0 seconds with 50% overlap
- Reference covariance: Leadfield-based
- Threshold optimization: Grid search (SENSAI method)
- Noise multiplier: User-configurable (2=aggressive, 3=balanced, 4=conservative)

**Citation:** Ros, T., Férat, V., Huang, Y., Colangelo, C., Kia, S.M., Wolfers, T., Vulliemoz, S., & Michela, A. (2025). Return of the GEDAI: Unsupervised EEG Denoising based on Leadfield Filtering. *bioRxiv*. DOI: 10.1101/2025.10.04.680449

### Temporal Filtering

The continuous EEG signal was bandpass filtered between 1.0 and 40.0 Hz using a finite impulse response (FIR) filter designed with the windowed-sinc (firwin) method. This frequency range was selected to capture the canonical EEG oscillatory activity while attenuating slow drifts and high-frequency noise. Power line noise at 50 Hz and its harmonic at 100 Hz was removed using notch filtering. Filter parameters were validated to ensure that notch frequencies did not exceed the Nyquist frequency of the acquired data.

### Re-referencing

Following filtering, data were re-referenced to the common average reference. This reference scheme was selected to provide a neutral reference that does not bias topographical distributions and is well-suited for source localization procedures.

### Independent Component Analysis

Independent Component Analysis (ICA) was performed using the extended Infomax algorithm (FastICA implementation) to decompose the EEG signal into statistically independent components. The number of components was set to the minimum of (n_channels − 1) and 25 to balance between data dimensionality reduction and preservation of neural sources. A fixed random seed (42) was used to ensure reproducibility across analyses.

### Automated Component Classification

Independent components were automatically classified using the ICLabel algorithm, a machine learning classifier trained on a large database of manually labeled ICA components. ICLabel categorizes each component according to its probability of representing brain activity, muscle artifact, eye blink, cardiac activity, line noise, channel noise, or other sources. Components with a brain probability below 0.85 were rejected, effectively removing physiological and environmental artifacts while preserving neurally-generated signals. This threshold was selected to provide conservative artifact rejection while minimizing the removal of neural sources.

### Preprocessing Quality Assessment

Preprocessing quality was quantified through multiple metrics including: signal-to-noise ratio improvement (in decibels), variance reduction percentage, and band-specific power changes across delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), and beta (13–30 Hz) frequency ranges. Diagnostic visualizations including before/after signal traces, topographic maps of root-mean-square amplitude, global field power time series, and power spectral density plots were generated for quality verification.

---

## Source Localization

### Forward Model Construction

Source localization was performed using a distributed source model approach. The cortical source space was defined on the FreeSurfer *fsaverage* template brain using an icosahedron-5 (ico5) tessellation, yielding approximately 10,242 vertices per hemisphere. This resolution provides adequate spatial sampling of the cortical surface while maintaining computational tractability.

The forward solution was computed using a three-layer boundary element model (BEM) consisting of the brain, inner skull, and outer skull surfaces. The BEM conductivity model accounts for the differential conductivity of neural tissue, cerebrospinal fluid, skull, and scalp in the propagation of electrical potentials from cortical sources to scalp electrodes.

### Inverse Solution

The inverse problem was solved using the minimum-norm estimate (MNE) approach with depth weighting (exponent = 0.8) to compensate for the inherent bias toward superficial sources. The regularization parameter λ² was set to 1/9, corresponding to an assumed signal-to-noise ratio of 3. Noise covariance was estimated from the preprocessed continuous data using a shrinkage estimator to improve conditioning of the covariance matrix.

### Region-of-Interest Extraction

Cortical source activity was parcellated according to the Desikan-Killiany atlas, yielding 68 cortical regions of interest (34 per hemisphere). Regional time courses were extracted using principal component analysis with sign-flipping (PCA-flip), which summarizes the activity of all vertices within each parcel by computing the first principal component and ensuring consistent sign across subjects.

---

## Functional Connectivity Analysis

### Phase Lag Index Computation

Functional connectivity between cortical regions was estimated using the Phase Lag Index (PLI), a measure that quantifies the asymmetry of the distribution of instantaneous phase differences between two signals. PLI is insensitive to volume conduction artifacts, as zero-lag interactions (which may arise from common sources or field spread) do not contribute to the index. PLI values range from 0 (no consistent phase relationship) to 1 (perfect phase locking with consistent lead/lag relationship).

Regional time courses were segmented into non-overlapping epochs of 4.0 seconds duration. Connectivity was computed using a multitaper spectral estimation approach with adaptive taper weighting, which provides optimal control of spectral leakage and frequency resolution. Connectivity was computed separately for five canonical frequency bands:

| Band  | Frequency Range |
|-------|-----------------|
| Delta | 0.5–4 Hz        |
| Theta | 4–8 Hz          |
| Alpha | 8–13 Hz         |
| Beta  | 13–30 Hz        |
| Gamma | 30–45 Hz        |

The resulting 68×68 connectivity matrices were symmetrized by taking the maximum value between each pair of connections (i.e., max(PLI_ij, PLI_ji)) and diagonal elements were set to zero. This procedure yields undirected weighted connectivity matrices representing inter-regional phase-based coupling strength.

### Network-Level Analysis

To reduce dimensionality and facilitate interpretation, connectivity values were aggregated within three canonical resting-state networks derived from the Desikan-Killiany parcellation:

| Network                      | Abbreviation | Regions |
|------------------------------|--------------|---------|
| Salience Network             | SN           | 4       |
| Default Mode Network         | DMN          | 16      |
| Central Executive Network    | CEN          | 8       |

Mean within-network PLI was computed as the average of upper-triangular matrix elements for each network, excluding the diagonal.

---

## Statistical Analysis

### Edge-Wise Comparisons

Between-group differences in connectivity were assessed using Welch's t-tests, which account for potentially unequal variances between groups. Tests were performed independently for each connection (edge), session, and frequency band. Within-group changes across sessions were evaluated using paired t-tests. Statistical significance was assessed at α = 0.05 (uncorrected).

### Cluster-Based Permutation Testing

To address the multiple comparisons problem inherent in edge-wise testing while preserving sensitivity to spatially distributed effects, cluster-based permutation testing was employed. Edge adjacency was defined based on graph topology: two edges were considered neighbors if they shared a common node. This approach respects the network structure of the connectivity data.

The cluster-forming threshold was set to the default t-statistic corresponding to α = 0.05, and 1,000 permutations were performed to estimate the null distribution of cluster mass statistics. Both between-group comparisons (at each session) and within-group comparisons (across sessions) were evaluated using two-tailed tests. Clusters were deemed significant at a family-wise error rate of *p* < 0.05.

---

## Software and Implementation

All analyses were implemented in Python 3.12 using:

- **MNE-Python** — EEG preprocessing and source localization
- **MNE-Connectivity** — PLI computation
- **SciPy** — Statistical testing
- **mne-icalabel** — ICA component classification
- **asrpy / meegkit** — Artifact Subspace Reconstruction
- **Matplotlib** — Visualizations

The complete analysis pipeline is available as open-source software to facilitate reproducibility.

---

*Note: Add specific details about your data acquisition parameters (sampling rate, electrode count, amplifier specifications, recording duration, and participant demographics) as appropriate for your study.*
