# NeuroStat Pipeline Validation

## Purpose

This validation suite tests whether the NeuroStat EEG processing pipeline
accurately recovers functional connectivity (Phase Lag Index) from EEG data
containing known ground-truth connectivity patterns embedded at different
signal-to-noise ratios.

Specifically, the validation:

1. **Generates** simulated 64-channel EEG data (EEGLAB `.set` format) with
   three planted phase-coupled oscillator pairs at known PLI values.
2. **Adds** realistic artifacts (eye blinks, muscle noise, 50 Hz line noise)
   at four severity levels (clean, light, moderate, heavy).
3. **Processes** each dataset through the full NeuroStat pipeline using both
   the **GEDAI** and **Traditional (ASR + ICLabel)** artifact-removal methods.
4. **Compares** recovered sensor-level and source-level PLI against the known
   ground truth and produces publication-quality figures and tables.

## Ground Truth Connectivity Patterns

| Pattern | Channels | Frequency | Expected PLI |
|---------|----------|-----------|-------------|
| Strong alpha (frontal ↔ parietal) | Fp1, AF7, AF3, F1 ↔ P1, P3, P5, P7 | 10 Hz | 0.80 |
| Moderate beta (central ↔ occipital) | C1, C3, C2, C4 ↔ O1, Oz, O2 | 20 Hz | 0.50 |
| Weak theta (left ↔ right temporal) | C5, T7 ↔ C6, T8 | 6 Hz | 0.30 |

## Artifact Severity Levels

| Level | Target SNR | Blinks/min | Muscle/min | Line Noise (µV) |
|-------|-----------|------------|------------|------------------|
| Clean | 10 dB | 0 | 0 | 0 |
| Light | 5 dB | 10 | 5 | 2 |
| Moderate | 5 dB | 30 | 15 | 5 |
| Heavy | 2 dB | 50 | 30 | 10 |

## How to Run

### Full validation (one command)

```bash
# From project root
python scripts/validation/run_complete_validation.py

# Or on Unix/macOS
bash scripts/validation/run_complete_validation.sh
```

### Step by step

```bash
# Step 1: Generate simulated data (~30 s)
python scripts/validation/generate_simulated_eeg.py

# Step 2: Run pipeline on all datasets (~15–25 min)
python scripts/validation/run_neurostat_batch.py

# Step 3: Analyse results and generate figures (~10 s)
python scripts/validation/analyze_validation_results.py
```

### Re-running individual steps

```bash
# Skip data generation, reuse existing simulated data
python scripts/validation/run_complete_validation.py --skip-generate

# Skip pipeline processing, only re-generate figures
python scripts/validation/run_complete_validation.py --skip-generate --skip-pipeline
```

## Output Structure

```
scripts/validation/
├── simulated_data/
│   ├── clean_high_snr.set
│   ├── light_medium_snr.set
│   ├── moderate_medium_snr.set
│   ├── heavy_low_snr.set
│   └── ground_truth.json
│
└── validation_data/
    ├── neurostat_outputs/
    │   ├── clean_high_snr/
    │   │   ├── GEDAI/
    │   │   │   ├── alpha_PLI.csv          (68×68 source-level)
    │   │   │   ├── beta_PLI.csv
    │   │   │   ├── theta_PLI.csv
    │   │   │   ├── delta_PLI.csv
    │   │   │   ├── gamma_PLI.csv
    │   │   │   ├── sensor_alpha_PLI.csv   (64×64 sensor-level)
    │   │   │   ├── sensor_beta_PLI.csv
    │   │   │   ├── sensor_theta_PLI.csv
    │   │   │   ├── preprocessing_metrics.json
    │   │   │   └── preprocessing_info.json
    │   │   └── Traditional/
    │   │       └── ...
    │   ├── light_medium_snr/ ...
    │   ├── moderate_medium_snr/ ...
    │   └── heavy_low_snr/ ...
    │
    ├── raw_sensor_pli/
    │   └── (baseline PLI from unprocessed data)
    │
    ├── results/
    │   ├── validation_results.csv
    │   ├── validation_scatter.png
    │   ├── method_comparison.png
    │   └── pattern_recovery.png
    │
    └── batch_manifest.json
```

## Interpreting Results

### validation_results.csv

Each row is one condition × pattern combination (4 conditions × 3 patterns = 12 rows).

| Column | Meaning |
|--------|---------|
| `Expected_PLI` | Ground truth PLI planted in simulation |
| `GEDAI_PLI` | PLI recovered after GEDAI preprocessing |
| `GEDAI_Error` | Absolute error `|recovered − expected|` |
| `Traditional_PLI` | PLI recovered after ASR + ICLabel preprocessing |
| `Traditional_Error` | Absolute error for Traditional method |
| `GEDAI_Better` | True if GEDAI error < Traditional error |

### Figure 1 — `validation_scatter.png`

Four-panel scatter plot (one per artifact level). Each point is one planted
pattern; the x-axis is expected PLI and y-axis is recovered PLI. Points on
the identity line indicate perfect recovery. Pearson *r* and RMSE are shown
per method.

**What to look for:**
- Points close to the identity line = accurate recovery.
- High *r* (> 0.85) = strong correlation between expected and recovered.
- Low RMSE (< 0.10) = small absolute errors.

### Figure 2 — `method_comparison.png`

Bar plot of mean absolute error across artifact severity levels. Two bars per
condition compare GEDAI and Traditional methods.

**What to look for:**
- Both methods should show low error for clean data.
- As artifact severity increases, error should increase gracefully.
- GEDAI should maintain lower error than Traditional in more conditions.

### Figure 3 — `pattern_recovery.png`

Grouped bar plot (Ground Truth / GEDAI / Traditional) for the moderate-
artifact condition, showing all three planted patterns.

**What to look for:**
- Strong alpha pattern should be recovered most accurately.
- Weak theta pattern is the hardest to recover.
- Error annotations on bars show absolute deviation from ground truth.

## Expected Values for Publication

For a well-functioning pipeline, you should expect:

| Metric | Expected Range |
|--------|---------------|
| Mean absolute error (clean) | < 0.05 PLI |
| Mean absolute error (moderate) | < 0.10 PLI |
| Pearson *r* (clean) | > 0.95 |
| Pearson *r* (moderate) | > 0.85 |
| RMSE (clean) | < 0.05 |
| RMSE (moderate) | < 0.10 |
| GEDAI outperforms Traditional | ≥ 7/12 cases |

## Pipeline Parameters Used

| Parameter | Value |
|-----------|-------|
| Bandpass filter | 1–40 Hz |
| Notch filter | 50, 100 Hz |
| ICA method | FastICA |
| ICA components | 25 |
| ICLabel brain threshold | 0.50 (see note below) |
| GEDAI threshold | 3.0 (balanced) |
| Source model | fsaverage, ico5, BEM |
| Inverse method | MNE (depth=0.8) |
| Epoch length | 4.0 s |
| Epoch overlap | 50% |
| PLI method | Multitaper PLI |
| Frequency bands | delta (0.5–4), theta (4–8), alpha (8–13), beta (13–30), gamma (30–45 Hz) |

**Note on ICLabel threshold:** The default NeuroStat threshold (0.85) is
optimised for real EEG recordings. Simulated data has simpler spectral
structure that ICLabel (trained on real EEG) can misclassify, leading to
excessive component rejection. The validation uses 0.50 ("more likely brain
than not"), which is appropriate for synthetic signals and can be adjusted
in `run_neurostat_batch.py`.

## Reproducibility

All simulated data use `random_seed=42`. Regenerating the data and rerunning
the pipeline should produce identical results on the same platform.

## Dependencies

All packages listed in `requirements.txt` plus:
- `scipy` (signal generation, statistics)
- `matplotlib` (figure generation)
- `mne` with fsaverage dataset (auto-downloaded on first source localisation run)
