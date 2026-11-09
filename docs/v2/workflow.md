# High-level workflow, CLI, and GUI

Phases 2 through 6 each added one stage with its own command. Running a full study
meant invoking those commands in order and hand-threading each run directory into
the next command. Phase 7 removes that bookkeeping. One shared workflow engine
resolves stage dependencies, runs the stages in order, and passes each run object
to the next stage in memory.

The CLI and the GUI both call that engine directly. The GUI does not shell out to
the CLI, and neither one owns a second copy of the orchestration logic.

## What v2 ingestion does and does not do

**v2 ingestion does not clean raw data. It converts already-cleaned inputs into the
canonical contract used by the later stages.**

Artifact rejection, filtering, re-referencing, ICA, and every other cleaning
decision stay with whatever tool produced your data. The project schema enforces
this: `input` must be exactly `{state: cleaned}`, and any other value is rejected
when the project loads. Accepted input formats are EEGLAB `.set`, FIFF `.fif`,
FieldTrip `.mat`, BioSemi `.bdf`, EDF, BrainVision `.vhdr`, TMSi `.Poly5`, LSL `.xdf`
and WiBCI `.csv` (see [input formats](input-formats.md)).

## Installation

```
pip install -e .
pip install -e ".[gui]"     # adds PyQt6 for the desktop GUI
pip install -e ".[dev]"     # adds pytest, build, setuptools
```

The GUI extra installs PyQt6. The v1 launchers keep working regardless.

## Stages and dependencies

```
ingest -> scouts -> connectivity -> network_models
                                 -> graphvar -> brainnet
```

Ask for a stage and the engine runs everything that stage needs, in canonical
order, and nothing it does not need:

| Requested | Runs |
|---|---|
| `ingest` | ingest |
| `scouts` | ingest, scouts |
| `connectivity` | ingest, scouts, connectivity |
| `network_models` | ingest, scouts, connectivity, network_models |
| `graphvar` | ingest, scouts, connectivity, graphvar |
| `brainnet` | ingest, scouts, connectivity, graphvar, brainnet |
| `all` | all six, in canonical order |

Requesting `brainnet` deliberately does not run `network_models`. The two are
sibling branches off connectivity, not a chain, so a rendering run does not pay for
mixed models it will never use. Ask for `all` when you want both.

Two stages are internally two-step. The `graphvar` stage builds the workspaces and
then runs the analysis. The `brainnet` stage builds the artifacts and then renders
them. Each appears as a single stage in the plan.

## Command line

```
neurostat run PROJECT --stage all
neurostat run PROJECT --stage brainnet
neurostat run PROJECT --stage network_models --stage brainnet
neurostat run PROJECT --stage all --json
```

`--stage` repeats. Omitting it entirely means `all`. Passing `--json` prints one
deterministic object with sorted keys, giving each stage its state, run id, run
directory, and whether the run was reused.

Runs are content-addressed, so re-running a workflow after a completed stage reuses
that stage instead of recomputing it. The `reused` field tells you which stages were
served from existing output.

Failures name the stage that failed and print nothing else to stderr but
`error: ...`, with exit status 2. Stages that had already succeeded keep their
results, and stages after the failure are reported as pending rather than dropped.

### The low-level commands are still there

Every per-stage command from Phases 2 through 6 is unchanged. Use them when you need
manual control or recovery:

```
neurostat validate PROJECT
neurostat ingest PROJECT
neurostat scouts PROJECT --ingest-run RUN_DIR
neurostat connectivity PROJECT --scouts-run RUN_DIR
neurostat network-models PROJECT --connectivity-run RUN_DIR
neurostat graphvar build PROJECT --connectivity-run RUN_DIR
neurostat graphvar run PROJECT --workspace-run RUN_DIR
neurostat brainnet build PROJECT --graphvar-run RUN_DIR
neurostat brainnet render PROJECT --artifact-run RUN_DIR
```

The import path matters most when GraphVar has already been run by hand in MATLAB.
`graphvar extract` takes an explicit workspace-to-result mapping and imports those
results without re-running the analysis:

```
neurostat graphvar extract PROJECT --workspace-run RUN_DIR \
    --result Theta_Baseline_Followup=/path/to/corr_area_1_Group.mat \
    --result Alpha_Baseline_Followup=/path/to/corr_area_1_Group.mat
```

Every workspace needs its own mapping, and workspace names must be unique.

## GUI

```
neurostat-gui
```

You do not write a project file. You describe the study the way v1 asked for it,
and the GUI generates the project for you.

### Study Setup

Point **Base Folder** at your data, then type **Groups** and **Sessions** as
comma-separated lists, exactly as in v1. Add **Conditions** (v1 had no condition
concept; a single `rest` is the normal case) and an **Output Folder**.

Three on-disk layouts are recognised automatically:

| Layout | Example |
|---|---|
| `Group/Session/file` | `study/GroupA/pre/subject-01.set` |
| `Group/Subject/Session/file` | `study/GroupA/Subject 1/pre/data_ft.mat` |
| `Group/Session/Subject/file` | `study/GroupA/pre/subject-01/preprocessed_raw.fif` |

The last is what v1 preprocessing itself writes, so cleaned output is discovered
without rearranging anything. Every format in [input formats](input-formats.md)
is recognised, and `*_cleaned` files are skipped the way v1 skipped them. Raw
files that v1 cannot read are converted to `.set` under `<output>/staged_raw`
before preprocessing.

The middle two layouts are the same shape on disk, so the session names you type
are what tells them apart. If a folder does not match any layout the GUI says so
rather than guessing.

Participants are namespaced by group, because subject folders are usually
numbered per group and `GroupA/Subject 1` is not the same person as
`GroupB/Subject 1`.

### Already cleaned, or not

**My data is already cleaned** is ticked by default, matching v2's contract.

Untick it and v1's preprocessing controls appear with their original defaults:
bandpass 1-40 Hz, notch 50/100 Hz, Traditional (ASR + ICLabel) or GEDAI, bad
channel interpolation, ICA method and component count. The GUI then runs the
unmodified v1 cleaning pipeline first, writes `preprocessed_raw.fif` per subject,
and feeds that straight into the v2 stages. Raw data in, statistics out, in one
run.

### Source localization

The default pairing is MNE with fsaverage, and **Auto (fetch fsaverage)** is on,
so the subject directory, transform, source space and BEM are provisioned for you
exactly as v1 did. Untick it to supply your own paths, or switch to the fixed
linear operator pairing if you have a precomputed sensor-to-scout matrix.

Only two source/connectivity pairings are valid, so they are offered as a single
choice rather than two menus that could be combined illegally.

### GraphVar and BrainNet

Point the GUI at MATLAB, the GraphVar 2.03a folder and its viewer config, then
pick the regions GraphVar analyses:

| Choice | Regions |
|---|---|
| All 68 DK regions | every Desikan-Killiany region |
| Triple network | SN 4, CEN 8 and DMN 16 (the v1 and published definition) |
| Custom set | any regions ticked from the 68, under a name you give (for example a 12-region DAN) |

The atlas workbook is written for you from fsaverage MNI centroids, so there is
nothing to prepare in Excel. GraphVar compares the first session with each later
session, first arm against second, for every band and condition. The default is
5000 permutations.

BrainNet draws only the bands whose GraphVar component passes the node-level
permutation test, using the top links of that component. That rule is fixed
before any result is seen. If no band passes, BrainNet is marked skipped with
the reason, not failed. The BrainNet settings go in a sibling
`<study>.brainnet.project.yaml`, which keeps every upstream run reusable.

MATLAB cannot save GraphVar results to a Windows path of 260 characters or
more. The GUI checks this before a run and asks for a shorter output folder when
it is needed.

### Pipeline

Press **Scan** to see every discovered record in a table before anything runs.
Tick the stages you want; the resolved plan shows the full dependency closure, so
asking for BrainNet visibly pulls in ingest, scouts, connectivity and GraphVar.
Progress arrives per stage, and finished stages fill a table with their state,
run id and reuse flag.

The generated `project.yaml` is written into your output folder and its path is
logged. Nothing about it is hidden: you can read it, edit it, commit it, or feed
it straight to `neurostat run`. That is the reproducibility v1 never had, without
asking you to author anything.

**Open existing project.yaml…** runs a project you already have, including one the
GUI generated earlier.

All engine work happens on a worker thread, so the window stays responsive. Failures
report the failing stage together with the stages that succeeded before it.

Unticking every stage is a validation error rather than a silent request to run
everything. Pass `all` when you mean all.

### Embedding the controller

`neurostat.gui.controller` imports no Qt, so it can be driven from a script or a
test:

```python
from neurostat.gui.controller import WorkflowController

controller = WorkflowController()
plan = controller.resolve_plan(["brainnet"])
result = controller.run("study.yaml", ["brainnet"], progress=print)
```

## External tools

The pure-Python stages need nothing beyond the installed dependencies. The rest
call out to external software, which is checked by a preflight step before any work
starts:

| Stage | Requires |
|---|---|
| `ingest`, `scouts`, `connectivity` | nothing external |
| `network_models` | R, with the packages the model configuration names |
| `graphvar` | MATLAB and GraphVar 2.03a |
| `brainnet` | MATLAB and BrainNet Viewer |

Preflight records the resolved paths, versions, and file hashes of whatever it
found, so a run carries evidence of the toolchain that produced it.

## A minimal project

```yaml
schema_version: 1
study:
  id: example_study
  arms: [intervention, comparison]
  sessions: [baseline, followup]
  conditions: [rest]
input: {state: cleaned}
analysis_profiles:
  default: {}
records:
  - record_id: participant-001-baseline-rest
    participant_id: participant-001
    arm: intervention
    session: baseline
    condition: rest
    data_kind: raw
    format: fif
    path: cleaned/participant-001-baseline-rest.fif
    analysis_profile: default
    data_unit: V
output:
  root: outputs
```

Records can also live in a CSV or TSV manifest:

```yaml
records:
  manifest: records.csv
```

For a public, reproducible example of this contract on randomized pre/post
resting-state EEG, see
[`scripts/validation/openneuro_ds006801_v2.py`](../../scripts/validation/openneuro_ds006801_v2.py).
It prepares checksum-pinned OpenNeuro data as cleaned FIF, generates the project,
and calls this same high-level workflow through `connectivity`.

Add a `network_models:`, `graphvar:`, or `brainnet:` block to configure those
stages. Requesting a stage whose block is missing fails immediately with a message
naming the block, before any work starts, rather than partway through a long run.

## Migrating from v1

The v1 entry points `run_study.py`, `run_study_app.py`, `pli_pipeline.py`, and
`app_gui.py` are unchanged and still supported.

To move an existing study to v2, point a project file at data you have already
cleaned, declare the arms, sessions, and conditions, and run `neurostat validate`
before anything else. Validation is strict on purpose: it checks that every record
path exists, that formats match their extensions, that record ids and participant
keys are unique, and that no participant appears in more than one arm. Fixing those
complaints up front is cheaper than discovering them during a MATLAB render.
