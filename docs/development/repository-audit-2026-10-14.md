> Audit baseline and historical findings. [Current execution plan](../../REPOSITORY_PLAN.md)
> supersedes prospective status statements below.

# Repository audit — 14 October 2026

Audit date: 14 October 2026. Baseline: commit `711aba8`, plus the existing
uncommitted wave 8 changes. Package metadata version: `0.6.1`.

At audit start, the library had **108 committed schemes and 137
schemes in the working tree**, across 11 families, with explicit methods
up to order 14. Waves 1–7 of the expansion are committed. Wave 8 has 29 new DIRK
methods implemented locally; at audit start they still needed a clean validation
gate and release documentation. Phase 1 has now completed those local gates. Waves 9–16 remain future work. The most useful immediate work is
to finish that gate, repair gaps in CI and executable documentation, and make the
catalog easier to use before adding more drivers.

This is the current execution plan. The existing specialist plans retain their
method lists, derivations, and experiment records; their status needs reconciling
with this audit. Checkboxes below represent future work unless marked complete.

## Phase 1 execution status — 14 October 2026

The numerical, coverage and documentation changes are validated in the working
tree: **6,459 passed, 605 skipped, zero failures** in the final full CPU gate;
the Docusaurus production build passes. All six CI shards pass, and their 7,064
case IDs partition the full collection exactly once. The existing wave 8 coefficient arrays,
enums and exports were preserved. Wave 8 and Phase 1 are landed together in
a bundled commit; all 137 schemes remain **unreleased** beyond package 0.6.1.
See [the validation record](../../docs/development/phase1-validation.md) for numerical
failures, solver-floor evidence, environment, timings and reproducible commands.

## What the repository does

`warpSPHIntegrators` is a Python and PyTorch library for stepping ODE systems whose
state may contain several physical fields and auxiliary buffers. A user supplies
a state, an update object, a system implementing the update protocol, and a
right-hand side. A scheme returns an `IntegrationResult` containing the new state,
stages, optional error estimate, history, and solver diagnostics.

The useful architectural separation is between field metadata, update operations,
and numerical drivers. The drivers operate through tags and system methods rather
than owning a particular SPH solver. This supports both ordinary ODE examples and
SPH-shaped states with constant, copied, ephemeral, and integrated fields.

| Area | Implementation | Current capability |
|---|---|---|
| State and updates | [fields.py](../../src/warpSPHIntegrators/state/fields.py), [specs.py](../../src/warpSPHIntegrators/state/specs.py), [protocol.py](../../src/warpSPHIntegrators/state/protocol.py), [util.py](../../src/warpSPHIntegrators/state/lifecycle.py) | Tagged dataclass fields, clone dispatch, typed and legacy update protocols, lifecycle hooks, integrated-field flattening |
| Catalog and lookup | [enums.py](../../src/warpSPHIntegrators/schemes/enums.py), [integration.py](../../src/warpSPHIntegrators/schemes/registry.py), [reuse.py](../../src/warpSPHIntegrators/analysis/reuse.py) | Family enums, scheme lookup, order and stability metadata, first-stage reuse analysis and warnings |
| Explicit and geometric stepping | `butcher.py`, `euler.py`, `tvd.py`, `verlet.py`, `ruth.py`, `prrk.py` | Classical, SSP, embedded and high-order RK; second-order dynamics; three unregistered parameterized pRRK drivers |
| Implicit stepping | `dirk.py`, `fullyimplicit.py`, `newmark.py`, `solvers.py`, `jfnk.py` | DIRK, coupled RK and Newmark; matrix-free Newton and GMRES; FD/JVP matvecs, preconditioning, diagnostics, implicit differentiation |
| Multistep and splits | `history.py`, `multistep.py`, `bdf.py`, `ssprkmsvs.py`, `ark.py`, `imex.py`, `imexmultistep.py`, `rhs.py` | Adams, BDF, SSP multistep, additive IMEX and semilinear RHS capabilities; fixed-step history and restart guards |
| Specialized stepping | `rkc.py`, `rosenbrock.py`, `exponential.py` | Stabilized explicit RKC/RKL, ROS3P, ETD2RK and EXPRB32 |
| Adaptation and analysis | `adaptive.py`, `stability.py`, `tvd_analysis.py`, `testing.py` | Error norm and timestep helpers, DP5 dense output, stability/TVD tools, reusable benchmark problems |
| User and development material | `docs/`, `tests/`, `scripts/`, root notebooks and Markdown | Docusaurus family wiki, property tests, benchmark scripts, numerical experiment records |

The current runtime family counts are:

| Family | Committed baseline | Working tree |
|---|---:|---:|
| Explicit RK | 63 | 63 |
| Symplectic | 6 | 6 |
| Relaxed Chebyshev | 3 | 3 |
| Multistep explicit | 9 | 9 |
| DIRK | 7 | 36 |
| Coupled RK | 2 | 2 |
| Multistep implicit | 8 | 8 |
| IMEX | 6 | 6 |
| Linearly implicit | 1 | 1 |
| Exponential | 2 | 2 |
| Newmark | 1 | 1 |
| **Total** | **108** | **137** |

The numerical implementation is currently PyTorch based. Clone support for Warp
arrays and downstream Warp-backed RHS operations do not constitute a complete
native Warp stepping backend. Buffer reuse, update kernels, graph capture, and the
Warp gradient ownership decision remain open. Adaptive control currently consists
of caller-driven helpers; there is no general trajectory solver with events and
output sampling. Multistep coefficients are fixed-step and history restarts when
the timestep changes.

## Expansion progress

At audit start, [SCHEME_EXPANSION_PLAN.md](plan-reconciliation-2026-10-23.md#expansion-inventory-dispositions) said
waves 8–16 were unstarted and the registry contained 108 schemes, despite its
wave 8 section recording the local implementation. [NOTES.md](../../docs/research/integrators/notes-index.md), section
3.28, labeled wave 8 done before its final gate. Phase 1 reconciles those wave 8
states: implementation, validation and documentation are complete locally;
commit completion and package release remain separate.

| Wave | Actual position | Remaining work or qualification |
|---|---|---|
| 1 — plain SSP | Committed; 7 registered | Reconcile planned SSP constants with measured values and recorded source discrepancies; retain the measured evidence |
| 2 — adaptive SSP | Committed; 3 registered | SSPRK43/432 have estimates; SSPRK932 deliberately ships fixed-step because the transcribed source estimator failed the rate check |
| 3 — SSP multistep | Committed; 2 registered and a new driver | Fixed-step history and startup behavior implemented |
| 4 — pRRK | Committed; 3 standalone drivers | Not counted in the registry; tableaus depend on both parameter and timestep. Update the original factory proposal to the implemented contract |
| 5 — low-storage SSP screening | Committed; 3 registered | Screening found distinct maps; these use the ordinary driver and do not receive the source implementation's storage savings |
| 6 — low-order RK | Committed; 18 registered | Anas5 and FRK65 remain deferred parameterized methods; SplitEuler was treated as an existing equivalent. PSRK members are fixed-step, despite the initial embedded-pair assumption |
| 7 — high-order RK | Committed; 11 registered | DP8 and Feagin methods ship without estimates. PFRK87 is listed in the plan but has no recorded disposition in code, notes, or changelog. The additional ExplicitTableaus enumeration also needs a recorded outcome |
| 8 — DIRK | Implemented, validated, documented and committed; 29 registered, 17 new estimates | Package release remains. SDIRK22 is intentionally omitted because the inspected upstream constructor was a stub |
| 9 — coupled FIRK | Unstarted; existing block driver only has the two-stage pair | Prove the three-stage solve and preconditioner at scale before adding higher-stage tableaus. Resolve the heading's six-method count against seven listed stage configurations and distinguish upstream order names from stage counts |
| 10 — multistep extensions | Unstarted | Variable-step history/coefficient design, NDF/Nordsieck methods, ABM54 estimate, CNLF2 and AMF source enumeration |
| 11 — stabilized and Krylov extensions | Unstarted | Enumerate actual candidates; determine which need new recurrences or drivers. Generalized-alpha overlaps the implicit roadmap |
| 12 — RKN and second-order ODEs | Unstarted | Design a general position/velocity driver and force-dependence contract; validate the existing Nystrom5th behavior before replacement; size implicit and partitioned variants separately |
| 13 — medium drivers | Unstarted | Extrapolation, RKIP, IRKC, FunctionMap, Composite. Coordinate Composite with the separate splitting plan |
| 14 — linear and multirate interfaces | Unstarted | Design explicit matrix/operator and split-time contracts, then validate cost on representative problems |
| 15 — Taylor series | Unstarted | Prototype derivative chains and measure their cost and compatibility before considering adaptive order |
| 16 — LowStorageRK screening | Unstarted | Screen map equivalence and record dispositions; a real storage benefit also needs a low-storage driver |

Eight of sixteen waves are committed. This is not a
percentage of implementation effort: the later interface and driver waves are
substantially larger than tableau transcription. The plan's estimated 120–150
scheme landing target already overlaps the current 137; revise the target around
useful capabilities and measured cost rather than treating catalog parity as the
completion criterion.

### Wave 8 completion checklist

- [x] Resolve failures from the current full-suite run and preserve regression
  evidence. The previous `.tmp/wave8_dirk/test_dirk_run1.txt` records 18 failures
  before the subsequent fixes; the final green result is recorded in
  `docs/development/phase1-validation.md`.
- [x] Correct the stage counts and Cash4 stage-time explanation in NOTES section
  3.28. Cash4's first node is approximately `0.4359`; the `-0.7` node is later.
  Record extra upstream stages, weight adjustments, floor grids, superconvergence,
  and deterministic reuse counting as described in the wave 8 checklist.
- [x] Add the wave 8 changelog entry. Extend `docs/docs/families/dirk.md` from 7
  to 36 methods; reconcile overview, API, selection, README, and site landing data.
- [x] Add adaptive-loop coverage for the new embedded pairs and gradient coverage
  for the new implicit schemes; both inventories are now metadata-driven.
- [x] Check semilinear/stiff convergence and stability classifications against the
  wave gate. In particular, Kvaerno4 and SSPSDIRK2 are recorded A-stable but not
  L-stable, and SFSDIRK4–8 are recorded as having bounded stability regions.
- [x] Measure and record the new DIRK TVD/SSP verdicts. The maintained
  `tests/test_tvd.py::EXPECTED_VERDICTS` table now covers all 137 registrations,
  with a cheap completeness gate and per-scheme measurements.
- [x] Re-measure CI shard balance after adding 29 schemes. Preserve coverage before
  moving slow cases between shards.
- [x] Pass the touched suites and the full gate after the final changes, update
  status, and land the bundled wave. Keep the existing coefficient edits intact
  while making audit and organization changes.

### Audit-time wave 8 validation blockers — resolved locally

The audit full-suite run had **26 failures**, listed below as the initial failure
ledger. Phase 1 resolved these and four further failures exposed by the expanded
FSAL/geometry checks; the final full Python gate is green. See the validation
record for the strict-solve evidence and retained default-solve tests:

| Failure group | Cases | Required action |
|---|---:|---|
| `test_step_reuse.py::test_non_fsal_schemes_are_not_claimed_as_fsal` | 20 | The old FSAL expectation list omits the new losslessly reusing DIRKs. Extend the expected classifications and verify their first/last-stage relationships rather than clearing their metadata |
| `test_tvd.py::test_classifier_verdicts_match_the_recorded_landscape` | 1 | Measure and record all 29 new verdicts; the expected inventory is still the old 108-scheme registry |
| `test_convergence.py`, Kepler order and monotonicity | 4 | Investigate ESDIRK4(3)7L[2]SA (order 3.39 against 4), ESDIRK6(5)9L[2]SA (order 5.00 against 6 plus nonmonotone errors), and SFSDIRK6 (3.835 against the 3.85 lower bound). Separate solver error and precision floors from tableau/order defects before changing grids or tolerances |
| `test_hamiltonian.py::test_dissipation_flag_predicts_energy_behaviour[SSPSDIRK2-kepler]` | 1 | Reconcile SSPSDIRK2's geometric classification: its tableau satisfies the symplectic RK condition exactly and represents two implicit-midpoint half-steps; update metadata and the property expectations with direct tests |

For SSPSDIRK2, the checked tableau satisfies
`b_i*a_ij + b_j*a_ji - b_i*b_j = 0` with maximum defect `0.0`. Its scalar
amplification agrees with `((1 + z/4)/(1 - z/4))**2`, approaching **+1**, whereas
the audit-time NOTES section 3.28 said `R(inf) = -1`. Phase 1 corrects this sign
and the classification record together, backed by direct tests. The energy test's bounded-error finding has algebraic support;
it is not a reason to weaken the test.

The new KenCarp/ARS-style entries are currently their **pure implicit tableaus**
on the DIRK driver. Their names do not add the upstream explicit/implicit pair
capability to this library. ARS222 and ARS232 have identical stored implicit
tableaus here, even though their upstream additive methods differ. Record such
equivalences in the catalog and distinguish registered names from distinct maps;
preserve public identifiers when deciding how aliases should be presented.

## Other plan documents

| Document | Current position | Reconciliation needed |
|---|---|---|
| [IMPLICIT_ROADMAP.md](plan-reconciliation-2026-10-23.md#implicit-roadmap-dispositions) | Its original phases 0–14 are recorded complete; remaining items are gated/deferred extensions | SSPRK54 is still called unimplemented, although wave 1 registered it. Connect generalized-alpha to wave 11, RKN to wave 12, FIRK scaling to wave 9, and composition to the splitting plan |
| [SPLITTING_PLAN.md](operator-splitting-design.md) | Detailed design and realization phases A–F; foundational implementation is absent | `rhs.CAPABILITIES` still contains only explicit, implicit, linear, nonlinear. No `ConservativeDissipativeRHS`, `classify_separable`, or general `compose` driver is present. The pressure-entropy downstream prototype is a prerequisite in the plan; the proposed gate removals are not accomplished by writing the design |
| [JFNK_PLAN.md](../research/downstream/wcsph-jfnk-design.md) | Records A/B and E1 through E1.10 complete across three repositories; C/D, E2 and E3 remain open | The library solver exists, but full WCSPH wiring, pressure-force JVP work, mDBC and free-surface support belong partly to `warpSPH`/`warpSPHCore`. Verify downstream status there before scheduling. Historical Picard-default and sole-solver statements need dating or archiving |
| [JFNK_DRIVER_OVERHEAD_NOTES.md](../../docs/research/integrators/jfnk-driver-overhead-2026-08-24.md) | Historical performance investigation with same-day fixes and findings | Extract any still-applicable performance tasks using fresh profiles; old line numbers and costs are not a current benchmark baseline |
| [NOTES.md](../../docs/research/integrators/notes-index.md) | Extensive implementation and measurement history, plus open-work sections | Separate current architecture and open work from dated experiments. The architecture section still describes 23 explicit schemes; the reproduction section quotes an August test run. Retain numerical caveats and derivations |
| [hamiltonian.md](../../docs/research/integrators/hamiltonian-proposal.md) | Proposal assessed and corrected by the splitting plan | Keep it as background research with a prominent pointer to the corrected design |
| `scratch_dfsph_*.md`, `docs/research/downstream/neighborlist-rebuild-2026-08-26.md` | Downstream research records at the package root | Preserve under a downstream research directory and label repository, date, and applicability |

Open implicit items still include BDF6 only when its narrower stability sector is
useful, robust stiff startup for BDF3–5, higher-stage coupled solves, partitioned
Lobatto methods, implicit warm starts, and Warp gradients. Each needs a concrete
consumer or benchmark. The BDF startup ceiling is especially relevant if the
library advertises stiff multistep use: its explicit DP5 starter can constrain the
initial step even when the steady-state BDF formula is stable.

## Usability and capability gaps

### Immediate correctness and onboarding gaps (audit baseline)

1. **CI silently omits two files.** `tests/conftest.py::CATEGORIES` does not include
   `test_drift_fields.py` or `test_prrk.py`. Collection contains 26 and 16 cases
   respectively, so the six category-based CI shards omit 42 cases. Marker
   assignment currently does not reject uncategorized files.
2. **The site overview example fails.** `docs/docs/index.md` constructs an
   undecorated `BaseState` subclass with keyword fields, gives an RHS with the
   wrong contract for the shown system, and accesses nonexistent
   `result.rhs_evaluations`. Executing the block first fails at construction.
3. **The quick start lacks imports.** Running the core Python blocks in
   `docs/docs/quickstart.md` reaches `NameError: reference_state`. Required helpers
   are imported in the separate README validation script, which reconstructs
   examples rather than executing the published blocks verbatim. Its passing
   result therefore does not validate the copy-and-paste path.
4. **Counts and descriptions disagree.** The site landing page/config and README
   still mention 64 schemes; overview/API say 108; runtime says 137. README says
   RK up to fifth order and twelve estimate emitters. Package description still
   says explicit ODE integrators. Quick start calls `IntegrationResult` a named
   tuple, but it is a frozen dataclass. README also links to nonexistent
   `docs/index.md`.
5. **Adaptive metadata is incomplete.** `test_embedded.py` lists 45 estimate
   emitters, while `test_adaptive.py` covers 14; the 31 wave 6–8 additions lack
   this adaptive-loop coverage. The controller documentation mentions only
   TR-BDF2's exception. Gauss-Legendre 2 also needs estimate exponent 3 despite
   propagated order 4, and tests already special-case it.
6. **Gradient claims exceed the growing test list.** `test_gradients.py` has a
   fixed list of 24 implicit schemes that excludes all 29 new DIRKs. Shared
   solver machinery is useful, but it does not replace checks of the actual
   registered drivers, stage handling, and parameter gradients.
7. **Fluid masking is declared but unused.** `integrated(fluid_only=True)` stores
   metadata, but no code reads that flag. The API does not enforce exclusion of
   boundary or inactive particles. Clarify the current responsibility of the
   user's system, then design masking consistently across updates, flattened
   unknowns, norms, and adjoint solves before claiming boundary-aware integration.

### Improvements to the public workflow

- Add a minimal tensor ODE adapter so a user can integrate `f(t, y)` without
  implementing an SPH-shaped system. Keep the full state protocol for advanced
  users and demonstrate both entry points with runnable examples.
- Provide catalog filtering by family, order, estimate support, dense output,
  RHS requirements, reuse policy, and force assumptions. `get_scheme` already
  has useful suggestions for unknown names; discovery needs comparable support.
- Consolidate metadata for stage count, propagated/embedded order, estimate
  exponent, required RHS capabilities, startup, and validity conditions. Generate
  the public catalog and counts from it. Avoid duplicating scheme names in every
  property-test list. Keep mathematical derivations and measured qualifications
  authored rather than generated.
- Clarify `getPreferredScheme`: it currently returns raw functions from an
  order-only selection for orders 1–5. A documented selector should account for
  stiffness, structure, and estimates, and return the same scheme object used by
  normal lookup. Do not silently change its legacy behavior.
- Add an optional trajectory runner that owns accepted-state history, reuse,
  final-time clipping, output sampling, callbacks, and adaptive rejection. Keep
  the single-step API as the lower layer. Define fixed-step support first.
- Give implicit failures a public policy. DIRK and coupled RK append diagnostics
  but do not check `SolveResult.converged` before returning a step. Distinguish
  tolerance success, stagnation, exhaustion, and nonfinite failure; let callers
  choose raise, return status, or rejection. Rejected attempts must not commit
  state, histories, or lifecycle side effects.
- Add tolerances per integrated field where mixed physical scales require them,
  finite-input checks in adaptive helpers, a minimum step/attempt limit, and
  guidance about float32 convergence floors. Extend dense output only with
  verified coefficients; add event handling after an output/interpolation contract
  exists.
- Document checkpoint/restart and history detach policies, particularly for
  differentiated multistep trajectories. Decide the serializable state boundary
  before providing convenience persistence.

### Improvements to numerical reach and performance

- Prioritize stiff multistep startup, stage-aware preconditioning, and measured
  warm starts over another large tranche of tableaus. Higher-stage coupled RK
  needs representative system sizes, solver cost, and memory measurements.
- Implement conservative/dissipative composition only with explicit structural
  contracts and lifecycle semantics. The existing velocity-dependent-force order
  loss is a useful regression control. Splitting is a route to better methods
  when the conservative half meets the assumptions.
- Treat native Warp buffers and genuine low-storage RK as separate performance
  projects. Adding a low-storage method to `RungeKuttaB` does not itself reduce
  allocations. Profile CPU/CUDA update cost, clone traffic, and Krylov synchronization
  before choosing kernels or pools.
- Add representative vector, dtype, and device gates. The existing numerical
  suite is largely small float64 work, and hosted CI installs CPU Torch. Preserve
  those precise checks while adding a small float32/CUDA smoke gate and a separate
  reproducible large-state benchmark.
- Strengthen high-order evidence where scalar convergence hits roundoff or
  superconvergence. Retain justified grid exceptions, but complement them with
  algebraic order conditions and nonlinear vector reference problems rather than
  continually widening empirical bands.
- Separate startup and steady-state multistep classification. The current TVD sweep
  does not thread history, so several verdicts describe repeated DP5 startup rather
  than the advertised multistep formula. Add history-aware measurements and label
  both regimes. A pass at the maximum sampled CFL is a measured lower bound;
  distinguish it from a proof of unconditional stability in user-facing tables.
- Define how active-particle masks and changing field shapes interact with history,
  reuse, and implicit unknowns. Shape changes and inflow/outflow need an explicit
  restart policy; the current downstream's fixed-layout assumption is not a general
  dynamic-particle capability.
- Establish minimum supported numerical dependencies and a reproducible development
  environment. The base Python here lacks Torch; the existing `warp` environment
  supplies the test stack. Notebook plotting/Jupyter dependencies have no declared
  extra, and the wave-equation notebook additionally needs sibling SPH packages.

## Repository organization

Keep one small root entry point for users and one current execution plan. Use the
existing Docusaurus source directory for user documentation and explicit adjacent
directories for developer plans and research. Preserve the current import package
and module paths during the first cleanup.

```text
README.md
CHANGELOG.md
license.md
REPOSITORY_PLAN.md
CONTRIBUTING.md
pyproject.toml
src/warpSPHIntegrators/
tests/
examples/
  notebooks/
  integrations/                # examples requiring downstream SPH packages
benchmarks/                    # measured performance and stability drivers
scripts/                       # maintenance, validation, release utilities
docs/
  docs/                        # Docusaurus user documentation
  plans/                       # specialized active plans
  development/                 # architecture, solver and backend contracts
  research/
    integrators/               # dated numerical and performance records
    downstream/                # SPH and neighbor-list experiments
  assets/                      # retained plots used in repository documents
  src/
  static/
```

| Current material | Destination or treatment |
|---|---|
| `README.md` | Shorten to purpose, install, one runnable example, selection summary, and links; move full API, galleries and contributor detail into the wiki/development docs |
| `docs/plans/scheme-expansion.md` | `docs/plans/scheme-expansion.md`; preserve source inventory, update status and dispositions |
| `docs/plans/implicit.md` | `docs/plans/implicit.md`; retain gated items and link the completed phase index to research records |
| `docs/plans/splitting.md` | `docs/plans/splitting.md`; keep active design and implementation gates together |
| `docs/plans/wcsph-jfnk.md` | `docs/plans/wcsph-jfnk.md`; split active downstream tasks from dated experiments and identify repository ownership |
| `NOTES.md` | Extract current architecture into `docs/development/architecture.md`; split dated sections into `docs/research/integrators/` with an index preserving section identifiers and backlinks |
| `docs/research/integrators/jfnk-driver-overhead-2026-08-24.md` | `docs/research/integrators/jfnk-driver-overhead-2026-08-24.md` |
| `docs/research/integrators/hamiltonian-proposal.md` | `docs/research/integrators/hamiltonian-proposal.md` with a pointer to the corrected splitting plan |
| Three root `scratch_*.md` files | `docs/research/downstream/`, with dates, target repository and follow-up status |
| `examples/notebooks/integrators.ipynb`, `examples/notebooks/viscous_burgers_demo.ipynb` | `examples/notebooks/`; declare plotting/notebook dependencies and repair relative output paths |
| `examples/integrations/jfnk_wave_equation.ipynb` | `examples/integrations/`; state required SPH repositories, installation and device assumptions |
| `images/` | Move referenced durable plots to `docs/assets/` only after updating paths; define how the site imports or serves them |
| Benchmark/gallery scripts | `benchmarks/`; retain maintenance and publishing commands under `scripts/` |
| Untracked `.tmp/` | Promote extraction/oracle tools and irreproducible findings that are needed to reproduce coefficients; then ignore scratch output. Preserve local files rather than deleting them during the cleanup |
| Ignored `literature/`, `dist/`, caches | Document local purpose; keep generated artifacts out of the source tree and preserve citation provenance in tracked records |

Moving files needs a coordinated link migration. Check README image paths, notebook
paths, benchmark output locations, changelog links, research section anchors,
Docusaurus navigation and footer URLs, packaging's readme/license references, and
references into sibling repositories. `docs/plans/` and `docs/research/` are not
automatically part of the site's `docs/docs/` content tree: deliberately add a
developer index or keep them as repository links. Use temporary compatibility
links for known external references and remove them after migration. Keep the
current license filename unless changing all packaging references in the same
change.

## Audit validation

- Runtime inspection: 137 schemes, 11 families, maximum registered order 14.
- Collection: 6,537 test cases in the current working tree.
- Existing README example validator: 13/13 checks pass, with the published-block
  limitations described above.
- Docusaurus production build: passes with the installed dependencies. This checks
  site compilation and configured links, not Python example behavior.
- Published examples executed as written: overview fails at `BaseState` construction;
  quick start fails at missing `reference_state` after running the core blocks.
- CI category audit: 42 cases from two files are outside the current shards.
- Full-suite validation: **5,907 passed, 604 skipped, 26 failed, 103 warnings** in
  **870.42 seconds (14 minutes 30 seconds)** with four xdist workers and one BLAS
  thread per worker. The failures are grouped above; wave 8 is not validated.
- Slowest individual test: the all-registry TVD classifier took **227.95 seconds**,
  compared with approximately 8–13 seconds for the other reported slow tests.
  Distributing that sweep per scheme is a concrete opportunity to shorten the gate.

Local validation uses Python `3.13.14`, Torch `2.13.0+cu130`, pytest `9.1.1`, and
the existing `warp` environment. CUDA is available locally; this does not establish
native Warp backend coverage or the hosted CI result. The previously recorded wave
7 run was 4,555 passed / 470 skipped; it applies to the older 108-scheme tree.

Reproduction commands:

```bash
conda run -n warp python -m pytest --collect-only -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n warp python -m pytest -q -n 4 --durations=15
conda run -n warp python scripts/validate_readme_examples.py
cd docs
npm run build
```

For a new environment, install the package's test extra and the documented
numerical dependencies first. After Phase 2 adds direct example validation, include
that command in this gate. After Phase 3 moves files, update these paths together
with the corresponding contributor documentation.
