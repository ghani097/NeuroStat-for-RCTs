# BrainNet artifacts and reproducible rendering

The `brainnet` stage consumes only a completed, integrity-verified GraphVar
analysis run. It has two deliberately separate outputs:

1. an immutable **prepared** run containing atlas-derived `.node` files,
   deterministic descriptive `.edge` files, exact top-link manifests, frozen
   viewer resources, validation, and provenance; and
2. an immutable **rendered** run containing one verified PNG and its MATLAB
   logs, metadata, source hashes, and image QA per requested workspace.

```text
neurostat brainnet build project.yaml \
  --graphvar-run outputs/runs/graphvar-<id>
neurostat brainnet render project.yaml \
  --artifact-run outputs/runs/brainnet-artifacts-<id>
```

Prepared artifacts remain useful when MATLAB or BrainNet Viewer is not
available. A failed render remains a nonfinal resumable partial and cannot
change or invalidate the prepared scientific artifacts.

## Selection and inference boundary

Every configured workspace must identify exactly one display component.
Candidates are restricted to components supported by the primary
within-workspace **node-count pFWE** result. They are sorted by node count
descending, primary pFWE ascending, and component identifier ascending. The
first component is selected.

Post-hoc NBS support cannot qualify a component for a figure. Condition-level
Holm support is retained in the manifest but is not required for descriptive
display. Individual member links remain uncorrected and descriptive.

Within the selected component, the top `K` links are ordered by permutation P
ascending, absolute GraphVar B descending, and the two display labels
alphabetically. The signed edge weight is `sign(B) * sqrt(F)`. Each exported
matrix is symmetric, has a zero diagonal, and contains exactly `K` unique
nonzero upper-triangle edges. Edge values use four decimal places; the
manifest retains source metrics and weights at ten significant digits.

## Atlas and node styling

Coordinates and node order come exclusively from GraphVar's frozen
`selected-node-map.json`. Approximate centroids are never substituted.
Configuration supplies one exact style entry per frozen
connectivity/display-label pair. BrainNet column four is named `color` because
it may represent a discrete module index or another viewer colour scalar;
column five is a positive node size. Missing, extra, or mismatched styles fail
before any artifact is finalized.

## Runtime boundary and image QA

MATLAB is launched as an argument vector with `shell=False`. Preflight requires
the configured MATLAB release, exact `BrainNet_MapCfg.m` resolution inside the
configured BrainNet root, and the packaged `whitebg.m` shim first on the path.
Hashes and literal version evidence are recorded separately for
`BrainNet_MapCfg.m`, `BrainNet.m`, and the update log: a single directory-wide
version is not inferred when bundled files disagree.

Each image is rendered in a disposable attempt. The packaged MATLAB entry
calls:

```matlab
BrainNet_MapCfg(surface, node, edge, viewer_config)
```

It then forces text and axes foreground colours to black, prints the exact
configured dimensions and DPI, and closes figures. The Python boundary rejects
timeouts, nonzero exits, missing output, false-success metadata, installation
changes after preflight, wrong dimensions/DPI, corrupt PNGs, and
blank/near-constant pixels. Resume reuses only attempts whose complete hash
manifest, source node/edge hashes, metadata, and pixel QA still agree.

All filesystem resources are explicitly configured. Relative paths resolve
against the project file; absolute paths are accepted only when explicitly
written by the operator. The package contains no installation- or
study-specific path.
