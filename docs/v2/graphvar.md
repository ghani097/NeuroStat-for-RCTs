# GraphVar 2.03a component inference

The `graphvar` stage consumes one completed, integrity-verified connectivity
run. It builds self-contained GraphVar 2.03a workspaces, optionally executes
them through MATLAB, and extracts all candidate connected components from the
v7.3 result files.

```text
neurostat graphvar build project.yaml --connectivity-run outputs/runs/connectivity-<id>
neurostat graphvar run project.yaml --workspace-run outputs/runs/graphvar_workspaces-<id>
```

For manual GraphVar execution, pass one exact mapping for every planned
workspace. Imported files are copied, hash-bound, and subjected to the same
shape, effect, permutation, and extraction checks. A standalone historical
result cannot cryptographically prove which external workspace generated it;
that limitation is recorded in the immutable lock.

```text
neurostat graphvar extract project.yaml \
  --workspace-run outputs/runs/graphvar_workspaces-<id> \
  --result family__theta__pre-post=/path/to/corr_area_1_Group.mat
```

## Strict project contract

The `graphvar` project section declares an ordered atlas node mapping. Every
node has an exact connectivity label, a 1-based atlas/matrix index, a GraphVar
display label, and optional coordinates. The selected rows, display labels,
and coordinates in the atlas workbook must agree. Each analysis family names
one condition, requested bands, ordered session pairs, the two-arm numerator
and denominator, an exact GraphVar effect label, a condition-level Holm family,
both alpha levels, permutation count, deterministic seed policy, and MATLAB
matrix field template. Unknown keys and ambiguous or duplicate analyses fail
before workspace construction.

Families sharing a Holm identifier must share the condition and inference
semantics. Holm is applied across every candidate component in all planned
workspaces in that family. Workspaces with no component remain explicit in the
status table and contribute no fabricated p-value.

## Workspaces and execution

One workspace is built for every band by ordered session pair. A participant
is included only when every session used anywhere in that family is present.
Rows are deterministic by subject and then first/second session. Each verified
full connectivity matrix is copied into a MAT file; no link to the source
dataset is retained. `Variables.csv`, `Workspace.mat`, `ImportSettings.mat`,
the frozen atlas/viewer resources, selected-node mapping, matrix audit, and all
nested manifests are hashed.

MATLAB is invoked with an argument vector and `shell=False`. Preflight verifies
the configured MATLAB release, Statistics Toolbox functions, literal GraphVar
release 2.03a, exact resolved `graphvar_glm.m`, and source/resource hashes.
The packaged runner uses GraphVar's Within-ID repeated-measures GLM, configured
second-minus-first session order, arm contrast, RFUN permutations, and a
workspace-derived deterministic seed. The GraphVar installation is read-only.

Partial attempts are never final. Each workspace result is created in a
disposable attempt directory, fully extracted, and hash-manifested before
atomic promotion. Resume skips only hash-verified completions. The complete
analysis is atomically finalized only after every requested workspace and the
independent validation pass.

## Inference

The observed component mask is `NP <= component_forming_alpha`. Null link
p-values are conservative upper-tail empirical ranks from `NF`, with the last
rank assigned to ties. The primary statistic is participating node count; the
null statistic is the largest node component in each permutation. Both the
GraphVar tail and conservative plus-one pFWE are exported.

Edge extent is calculated from the same masks as a separate, explicitly
post-hoc NBS family. It cannot replace or strengthen primary node-count
inference. Separate condition-level Holm corrections are calculated across all
primary candidate pFWE values and all post-hoc NBS candidate pFWE values.
Every component and every member link is exported. Link B, F, parametric P,
and permutation NP values are descriptive and not independently corrected.
