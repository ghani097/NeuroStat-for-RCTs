# NeuroStat v2 architecture foundation

NeuroStat v2 is being introduced alongside the existing v1 application. The v1 launchers, `run_study.py` and `run_study_app.py`, remain supported throughout the migration and continue to invoke the unmodified root-level pipeline and GUI. The v2 package provides a compatibility seam rather than moving that implementation prematurely: `neurostat.legacy.load_legacy_pipeline()` imports `pli_pipeline` only when a caller explicitly needs v1 behavior. Ordinary imports of `neurostat` and `neurostat.legacy` do not load the scientific or GUI stack.

## Public pipeline

The public v2 pipeline has six named stages: `ingest`, `scouts`, `connectivity`, `network_models`, `graphvar`, and `brainnet`. Phase 1 establishes only their names and the contracts used to communicate outcomes; it does not yet ingest projects, process signals, run GraphVar or BrainNet, or supply a replacement GUI.

Future CLI and GUI interfaces will call shared application services for these stages instead of each owning a separate processing implementation. This keeps user-interface concerns at the edge, permits consistent progress reporting, and makes results reusable across scripted and interactive workflows.

## Foundation contracts

`neurostat.contracts` defines lightweight typed records that do not depend on MNE, matplotlib, PyQt, or the legacy module. `RecordKey` identifies a participant, arm, session, and condition. `ArtifactRef` uses `pathlib.Path` to locate a typed output. `StageResult` contains immutable artifact tuples and a lifecycle state. `ProgressEvent` and `ProgressCallback` give services a UI-neutral way to report work. `ProjectPaths` holds the project file and output root as `Path` objects.

The contracts use frozen, slotted dataclasses where records are appropriate. They validate nonblank identifiers and artifact kinds, and progress counts cannot be negative or exceed their total. This small boundary is the stable base for later migration phases.

Phase 2 adds strict externally-cleaned project files and resumable canonical ingestion; see [clean-data ingestion](clean-data-ingestion.md). Phase 3 adds variable-length source scouts and the two explicit PLI routes described in [source scouts and connectivity](source-connectivity.md). Phase 4 adds exact-label network summaries, common-cohort repeated-measures mixed models, explicit multiple-testing families, R dependency preflight, and immutable validated outputs described in [network mixed models](network-models.md).
Phase 5 adds self-contained GraphVar workspaces and all-candidate component
inference described in [GraphVar component inference](graphvar.md). Phase 6
adds atlas-derived BrainNet artifacts and a separately immutable, resumable
render boundary described in [BrainNet rendering](brainnet.md). Phase 7 adds the
shared workflow engine that resolves stage dependencies and threads run objects
between stages, plus the high-level `neurostat run` command and the PyQt6 GUI,
described in [high-level workflow, CLI, and GUI](workflow.md).

The intent stated above, that CLI and GUI interfaces call shared application
services rather than each owning a processing implementation, is realised in
`neurostat.workflow`. Both front ends call `run_workflow` directly. The GUI
controller in `neurostat.gui.controller` imports no Qt, so orchestration stays
testable without a display, and the Qt view layer is confined to
`neurostat.gui.app`.

The GUI does not ask for a project file. `neurostat.gui.discovery` recognises the
common on-disk study layouts and returns one record per input file;
`neurostat.gui.study_builder` turns the collected study details into a project
document and validates it through both `load_project` and
`parse_analysis_profile`, since a profile can satisfy the project schema and
still be rejected by the analysis layer. `neurostat.gui.fsaverage` provisions the
MNE resources automatically, and `neurostat.gui.legacy_bridge` runs the
unmodified v1 preprocessor through the `neurostat.legacy` seam when the inputs
are not yet cleaned. All four are Qt-free, so the whole path from typed study
details to a validated project is testable without a display.
