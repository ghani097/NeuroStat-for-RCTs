# NeuroStat v2 architecture foundation

NeuroStat v2 is being introduced alongside the existing v1 application. The v1 launchers, `run_study.py` and `run_study_app.py`, remain supported throughout the migration and continue to invoke the unmodified root-level pipeline and GUI. The v2 package provides a compatibility seam rather than moving that implementation prematurely: `neurostat.legacy.load_legacy_pipeline()` imports `pli_pipeline` only when a caller explicitly needs v1 behavior. Ordinary imports of `neurostat` and `neurostat.legacy` do not load the scientific or GUI stack.

## Public pipeline

The public v2 pipeline has six named stages: `ingest`, `scouts`, `connectivity`, `network_models`, `graphvar`, and `brainnet`. Phase 1 establishes only their names and the contracts used to communicate outcomes; it does not yet ingest projects, process signals, run GraphVar or BrainNet, or supply a replacement GUI.

Future CLI and GUI interfaces will call shared application services for these stages instead of each owning a separate processing implementation. This keeps user-interface concerns at the edge, permits consistent progress reporting, and makes results reusable across scripted and interactive workflows.

## Foundation contracts

`neurostat.contracts` defines lightweight typed records that do not depend on MNE, matplotlib, PyQt, or the legacy module. `RecordKey` identifies a participant, arm, session, and condition. `ArtifactRef` uses `pathlib.Path` to locate a typed output. `StageResult` contains immutable artifact tuples and a lifecycle state. `ProgressEvent` and `ProgressCallback` give services a UI-neutral way to report work. `ProjectPaths` holds the project file and output root as `Path` objects.

The contracts use frozen, slotted dataclasses where records are appropriate. They validate nonblank identifiers and artifact kinds, and progress counts cannot be negative or exceed their total. This small boundary is the stable base for later migration phases.

Phase 2 adds strict externally-cleaned project files and resumable canonical ingestion; see [clean-data ingestion](clean-data-ingestion.md). Phase 3 adds variable-length source scouts and the two explicit PLI routes described in [source scouts and connectivity](source-connectivity.md).
