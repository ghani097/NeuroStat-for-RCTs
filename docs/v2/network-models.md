# Network summaries and repeated-measures mixed models

The `network-models` stage consumes one completed, validated connectivity run. It never discovers study cells from filenames or row order. Networks, condition families, bands, ordered sessions, contrast directions, correction families, R runtime, and optional diagnostics are all explicit in the project YAML.

```text
neurostat network-models project.yaml \
  --connectivity-run outputs/runs/connectivity-<digest> --json
```

The Python service is `neurostat.network_models.run_network_models`. CLI and future GUI callers use this same service.

## Typed project contract

Add one `network_models` section to the same project used for ingestion, scouts, and connectivity. A generic complete fragment is in [`examples/v2/network-models`](../../examples/v2/network-models). The section contains exactly:

- `networks`: arbitrary safe names mapped to at least two unique, exact atlas labels;
- `families`: named analysis families, each selecting exactly one declared condition, requested bands/networks, ordered sessions, an explicit boundary-observation policy, directed between-arm and within-arm contrasts, and complete correction-family membership for Type III and planned contrasts;
- `model`: the frozen primary model contract;
- `diagnostics`: residual, raw-scale, robust winsorization, allFit, influence, and leave-one-out controls;
- `runtime`: the Rscript executable, optional library search paths, dependency contract, optional strict renv lock check, and timeout;
- `validation`: explicit numeric tolerances for independent Python reconstruction.

Unknown keys, duplicate names/labels, undeclared conditions/sessions/arms, same-side contrasts, unsafe resources, missing correction membership, and overlapping correction families are rejected. Every primary inferential result belongs to exactly one named correction family. A correction family explicitly targets `anova`, `between`, or `within`; lists Type III `terms` or planned `contrasts`; and lists bands, networks, and methods. Separate `anova` families for `arm`, `session`, and `arm:session` reproduce by-term adjustment across band/network jobs. No per-file adjustment is hidden.

## Exact network values and transform

For one connectivity matrix and a configured network with `n` nodes, the raw summary is:

```python
submatrix = matrix[np.ix_(indexes, indexes)]
raw_pli = submatrix[np.triu_indices(n, 1)].mean()
```

The diagonal and duplicate lower triangle are therefore excluded. Atlas labels must agree across every record in an analysis family, and every configured label must be present exactly.

Fisher-z boundary handling is deliberately narrow. With configured, verified observation count `N`, epsilon is `1 / (2N)`. A raw value exactly equal to zero becomes epsilon; a value exactly equal to one becomes `1 - epsilon`; every interior value is unchanged. The adjusted value is then passed to `atanh`. The long table records raw PLI, adjusted PLI, z PLI, observation count, observation-count provenance, and boundary policy.

`boundary_observation_count: scout_samples` is valid only for the all-sample evoked Hilbert route and is resolved from the integrity-checked upstream scout artifact. `connectivity_windows` is valid only for continuous multitaper PLI; its complete-window count is checked against scout samples, window length, and dropped tail. NeuroStat never guesses 512, treats a window length as an observation count, or silently substitutes another denominator.

## One cohort per condition family

Each family selects one condition so records from multiple conditions cannot collide. A participant enters a family only when exactly one record exists for every requested session. This common complete cohort is used for every requested band and network in that family. Participants missing sessions are retained in an exclusion table with available/missing sessions and reason. The stage fails if a configured arm has no included participant.

`network_summaries.csv`, `cohort_inclusion.csv`, and `cohort_counts.csv` are deterministic and hashed. Record IDs and connectivity hashes make every modeled row traceable.

## Primary R model and planned estimands

The packaged noninteractive runner fits, for every configured band/network:

```text
z_pli ~ arm * session + (1 | subject_id)
```

The frozen settings are REML, `nloptwrap`, `maxfun = 100000`, sum/poly contrasts, lmerTest Type III tests, Satterthwaite degrees of freedom, and emmeans with `lmer.df = "satterthwaite"`. Output includes:

- complete fixed effects and the `arm`, `session`, and `arm:session` Type III terms;
- z-scale EMMs and `tanh`-back-transformed estimates and limits;
- every directed configured between-arm/session and within-arm/session contrast;
- residual sigma and `estimate / residual_sigma` (`d_LMM`) for planned contrasts;
- convergence code/messages, singularity, random/residual SD, ICC, and maximum gradient;
- residual summaries plus per-observation fitted, residual, and standardized-residual values.

Raw-scale sensitivity repeats fixed effects, Type III tests, and planned between-arm estimands. Robust winsorization, when enabled, operates on raw PLI within each arm/session cell: median, `MAD × 1.4826`, then `IQR / 1.349`, then sample SD as fallbacks; values outside the configured robust-z threshold are bounded before the exact boundary transform and `atanh`. It also exports Type III and planned between-arm results.

Requested allFit output records success, singularity, log likelihood, maximum fixed-effect change, and messages for every optimizer. Requested influence output records Cook distance, the `4/N` cutoff, flags, and failures. Requested leave-one-participant-out analysis refits every job and exports every planned between-arm contrast, full/reduced estimates and t statistics, direction reversals, singularity, convergence, and summaries. Optional features are preflighted and must produce their complete auditable row sets; empty outputs cannot masquerade as success.

## R runtime and dependency contracts

R is invoked with an argument vector, never a shell string, so executable and library paths containing spaces are safe. Preflight:

1. resolves the configured executable;
2. verifies required package availability and declared minimum or exact versions;
3. captures R, package, and library-path metadata;
4. checks optional-package requirements;
5. optionally checks an explicit renv lock against installed packages.

The built-in tab-separated dependency contract supports an already provisioned runtime and is packaged in the wheel. An external contract can pin exact versions with `==`. `strict_renv: true` requires both an explicit lockfile and the `renv` package; it fails with an actionable message when either is unavailable. Process exit failures are never accepted merely because files appeared, and stdout and stderr are both preserved.

## Multiple testing, validation, and immutable runs

BH, Holm, and Bonferroni are calculated across each complete named result family. This includes explicit Type III term families as well as between/within estimands. `multiple_testing.csv` records target, family, method, family size, term or contrast, raw p value, adjusted p value, and result identity.

Before finalization, Python independently:

- reopens verified HDF5 matrices and reconstructs every network mean and boundary transform;
- checks exact summary, cohort, model-job, cell, fixed-effect, Type III, residual, and optional-output cardinality;
- reconstructs cell means and every configured contrast direction;
- verifies raw/back-transformed EMMs and standardized estimates;
- cross-links every correction membership and raw p value to its exact R contrast, then recalculates the adjustment.

A run is first written under a private staging directory and atomically renamed only after validation. Its identity covers normalized project/config, stable connectivity/scout scientific identities, runtime/package metadata, dependency contracts, and packaged R source. The manifest hashes the lock, inputs, summaries, cohort files, R source, dependency metadata, stdout/stderr, every statistical table, status, and validation report. Reuse verifies the exact file set and every hash; modified artifacts fail rather than being overwritten.

This stage does not run GraphVar or BrainNet. Those are separate downstream stages so network mixed-model inference, permutation graph inference, and rendering cannot be conflated.
