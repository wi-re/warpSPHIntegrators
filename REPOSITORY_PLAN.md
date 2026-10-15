# Repository execution plan

Updated 14 October 2026. This is the authoritative current task list.
Current publishing instruction: keep subsequent commits local. The user will
push manually after local work is finished; avoid triggering long cloud CI runs.
Specialist designs retain source inventories and implementation gates under
[docs/plans](docs/plans/README.md); historical task statements in numerical records
are superseded by this plan. The [original audit](docs/development/repository-audit-2026-10-14.md)
retains the baseline findings and migration rationale.

## Completed phases

| Phase | Result | Commit / evidence |
|---|---|---|
| 1 — reliable gates and wave 8 | 137 registered schemes; numerical, CI partition, adaptive and gradient gates restored | `413d4c4`; [validation](docs/development/phase1-validation.md) |
| 2 — executable onboarding and catalog | Derived metadata, generated catalog, runnable source examples and PR documentation gate | `5eb45e0`; [validation](docs/development/phase2-validation.md) |
| 3 — consolidation and organization | Current architecture, method dispositions, indexed research, contributor workflow, relocated examples/assets/benchmarks and preserved oracles | [migration and validation](docs/development/phase3-validation.md) |
| 4 — ordinary ODE usability and packaging | Tensor adapter, catalog filters, structural metadata, compatible selection, option guards and installed-wheel gate | [validation](docs/development/phase4-validation.md) |
| 5 — reliable run orchestration | Transactional fixed/adaptive runs, step status, sampling, events and restart checkpoints | [validation](docs/development/phase5-validation.md) |
| 6 — stiff startup and coupled scaling | Order-preserving implicit BDF bootstrap, warm/preconditioner measurements, s=3 prototypes; diffusion scale gate failed, wave 9 deferred | [validation](docs/development/phase6-validation.md) |
| 7 — physical splitting on known toy consumers | Physical capabilities, Lie–Trotter/Strang, implicit/dense exact D, bounded Yoshida4, lifecycle/gradient and geometric diagnostics | `78d6a25`; [validation](docs/development/phase7-foundation-validation.md) |

The 142-scheme catalog remains unreleased beyond package 0.6.1. Expansion
waves 1–8 and the first s=3 wave-9 batch are complete with recorded exceptions;
the remaining batches are subject to the gates below. Commit creation and package publication are
separate actions. **Phase 8 is underway: its first reviewed batch registers
Gauss-Legendre 3 / Radau IIA s=3 after their source, companion, scale and numerical
gates. A parameterized generalized-alpha driver now qualifies selective damping
on the known structural consumer, with accepted acceleration, restart, lifecycle,
quantity coupling and gradient checks. Subsequent reviewed backends supply
collocation stages 2–9 with Schur spatial preconditioning, variable Adams,
MEBDF2/CNLF2, Legendre/Gegenbauer stabilization, generalized Nyström, fixed and
adaptive-order NDF, explicit extrapolation, owned algorithm switching and direct
state maps and 54 screened full-stage RK expansions. Their source/consumer/restart
evidence is linked below; remaining driver families retain their own gates.
PESPH integration is later application work.**

## Phase 3 completion record

- [x] Reconcile plan statuses; close SSPRK54; record PFRK87, Anas5/FRK65,
  additional ExplicitTableaus candidates and upstream estimator exceptions.
- [x] Record implemented/equivalent/deferred/excluded/awaiting-verification method
  dispositions, source pins, constructors, transformations, tests and consumers.
- [x] Organize specialist plans and dated records, with a measurement index and
  compatibility anchors for old numbered references.
- [x] Consolidate documentation; document environments, category gates, multicore
  execution, registration, provenance and release validation in CONTRIBUTING.
  The initially shortened README was subsequently expanded at user review to
  restore self-contained explanations, method advice, the complete scheme list,
  understood examples, stability galleries and benchmark results.
- [x] Relocate notebooks, benchmarks and assets; repair paths; promote durable
  coefficient/oracle tools and evidence before ignoring preserved scratch output.

## Remaining execution phases

### Phase 4 Improve discovery and ordinary ODE usability — complete

Priority: high. Size: medium.

- [x] Add the tensor adapter and catalog filter API with complete minimal examples.
- [x] Document a common option surface and scheme-specific options. Validate critical
  solver options while preserving intentional passthrough to the RHS.
- [x] Remove unused `fluid_only` after checking warpSPHIntegrators, warpSPH and
  warpSPHCore for callers/metadata consumers. Document system/RHS ownership of
  particle masking; do not introduce a mask flag without a consumer.
  [Consumer audit and validation](docs/development/fluid-only-removal.md).
- [x] Define a compatible future for `getPreferredScheme` and metadata for
  separability, drift fields, history, estimates and dense output.
- [x] Declare development/example/benchmark dependencies and test an installed
  wheel separately from the editable checkout.

Validated: 6,907 tests passed, 609 skipped; installed-wheel and documentation
gates pass. See [the Phase 4 record](docs/development/phase4-validation.md).

Completion: a new user can choose an appropriate method and integrate a tensor ODE
without implementing lifecycle hooks; advanced SPH state behavior remains covered
by the existing protocol tests.

### Phase 5 Add reliable run orchestration and failure handling — complete

Priority: high. Depends on metadata from Phases 2 and 4. Size: medium to large.

- [x] Define public step success and failure semantics, including the distinction
  between Newton stagnation at a precision floor and true tolerance convergence.
- [x] Add a fixed-step runner with history/reuse threading and output callbacks.
- [x] Add adaptive acceptance/rejection, field tolerances, nonfinite checks, limits,
  and rollback of state and lifecycle effects; use the recorded estimate exponent.
- [x] Specify sampling/dense-output behavior, then add supported events and restart
  checkpoints as separate increments.

Completion: deterministic fixed-step runs and controlled adaptive runs have correct
final time, preserved caller state, observable failures, and no history or side
effect committed by rejected attempts. Existing single-step users remain supported.

Validated: 6,938 tests passed, 609 skipped; installed-wheel and documentation
gates pass. See [the Phase 5 record](docs/development/phase5-validation.md).

### Phase 6 Address stiff startup and evaluate coupled scaling — complete

Priority: high for stiff consumers; prerequisite for wave 9. Size: medium to large.

- [x] Design an order-preserving implicit bootstrap for BDF3–5 and compare it with
  the explicit starter on a stiff transient. A low-order warm-up alone must not
  cap the trajectory's advertised order.
- [x] Measure implicit first-stage warm guesses and block/stage-aware preconditioners
  against cold starts on representative vector systems.
- [x] Prototype three-stage Gauss/Radau solves with configurable, recorded DOF
  sweeps, iteration counts, work units, peak memory, failure rates and wall time.
- [x] Enforce the wave 9 registration gate. Stage/order names, prototype orders 6/5,
  stability/symmetry and gradients are verified; companions remain absent. The
  diffusion scale bar fails, so no higher-stage registrations are added.

Completion: stiff startup retains measured orders 3–5 at rate 100. The
114-configuration benchmark records time/work/memory and failures; reaction
passes, while diffusion exceeds the work bar at N=256 and fails solves at N=1024.
Wave 9 remains gated. Validated: 6,963 passed, 609 skipped; wheel and docs
gates pass. See [the Phase 6 record](docs/development/phase6-validation.md).

### Phase 7 Implement useful splitting and second-order methods — complete on toy consumers

Scope revised by the user on 14 October: known toy behavior is the validation
target while PESPH is being built. Production SPH physics claims stay conditional.

- [x] Audit the downstream prototype/baseline: no executable PESPH yet. Use the
  known damped oscillator and a controlled unsplit Burgers reference for this phase;
  retain entropy/separability/compatible-energy assumptions for later SPH adoption.
- [x] Implement conservative/dissipative capabilities and structural checks, then
  explicit Lie-Trotter and Strang orchestration with a single outer finalize.
- [x] Verify order on damped/split test problems. Backward Euler as the dissipative
  substep is the first-order control, not the advertised second-order variant.
- [x] Add implicit/dense exact dissipative flows and measure the viscous CFL benefit.
  Demonstrate fixed/static exact-flow merging in a benchmark; preserve existing
  synchronized callbacks/output/restart semantics in the ordinary runner.
- [x] Provide a shared one-step composition contract for existing geometric schemes
  and future RKN/Composite additions. Restricted Yoshida4 reaches order four with
  a checked anti-diffusion bound; compatible-energy substeps remain application work.
  General RKN, Suzuki and arbitrary negative-step recipes are separate expansion
  work, not implicit additions to this phase.

Completion: the split meets its contracts, preserves lifecycle and gradients, and
delivers measured order/stability/cost improvements on executable toy consumers.
The constant-damping determinant and integrated mechanical-energy budget are also
measured. See [the Phase 7 record](docs/development/phase7-foundation-validation.md).
Validated: 6,993 passed, 609 skipped; installed-wheel, runnable examples, catalog,
repository links and documentation-site gates pass.

### Phase 8 Resume catalog expansion in reviewed batches

Priority: after reliable gates and a concrete benefit. Size: varies by driver.

Execution direction updated 14 October 2026: implement the backend backlog now,
using understood standalone problems and backend scale tests. Frontend/PESPH
integration is later application work and does not gate backend implementation.
Continue between reviewed batches; retain source, numerical, ownership and cost
gates for each driver. The remaining execution order is higher-stage collocation,
variable-step multistep, stabilized/linearized/exponential drivers, general RKN,
extrapolation/maps/switching, then operator/multirate/Taylor interfaces and
low-storage screening. A source equivalence or unsupported problem-class finding
must be recorded rather than counted as an implemented method.

- [x] Clear the recorded Phase 6 spatial scale bar with shifted spatial ILU.
  [Follow-up evidence](docs/development/spatial-followup-validation.md).
- [x] Complete wave 9 companion/source/catalog gates for the first s=3 batch;
  [validation](docs/development/phase8-wave9-validation.md). General downstream
  SPH spatial/field assembly remains application-specific.
- [x] Qualify higher-stage Gauss/Radau with their own scale/estimate/source evidence.
  `Collocation(family, stages)` supplies stages 2–9, fixed/adaptive stepping and
  a real Schur spatial-preconditioner interface; [backend evidence](docs/development/higher-collocation-validation.md).
  Source order-switching and filtered controllers remain distinct driver work.
- [x] Re-enumerate waves 10–11 against pinned sources and split simple tableaus
  from history, recurrence, or driver changes. Variable coefficients are primarily
  a timestep/history issue; do not assume fixed-step methods exclude non-autonomous
  RHS functions.
- [x] Implement variable-step AB/ABM orders 3–5, ramped startup estimates,
  adaptive acceptance/rollback and portable tensor history. Register classical
  fixed-step AB5/AM5 PECE and fix before-consumption timestep invalidation in
  constant Adams. [Backend evidence](docs/development/variable-adams-validation.md).
- [x] Add source-checked MEBDF2 and CNLF2 backends, with stiff-decay and split
  oscillation qualifications, safe history/reset and portable restart.
  [Backend evidence](docs/development/implicit-extensions-validation.md).
- [x] Implement stage-bound RKL1/2 and RKG1/2 factories with internally consistent
  time nodes, spectral-bound sizing, adaptive/restart and full-state ownership.
  [Backend evidence](docs/development/stabilized-backend-validation.md).
- [x] Implement fourteen generalized Nyström configurations with explicit VI/VD
  and thermal-coupling contracts, component quadratures and Phase 7 composition.
  Screen the pinned RKN4 velocity-order limitation. Improved IRKN history remains
  separate. [Backend evidence](docs/development/nystrom-backend-validation.md).
- [x] Implement fixed-target-order NDF 1–5 difference state, source resampling,
  order-preserving implicit startup, adaptive quasi-constant blocks and portable
  tensor continuation. Arbitrary rapid grid changes restart safely.
  [Backend evidence](docs/development/ndf-backend-validation.md).
- [x] Add adaptive NDF target-order selection with source-checked neighboring
  proposals, owned mature-history transitions, field-tolerance normalization and
  portable controller state. Exact upstream QNDF retry/discontinuity policy remains
  distinct. [Backend evidence](docs/development/adaptive-ndf-validation.md).
- [ ] Resolve the remaining wave-10 history/controller candidates: adaptive
  Adams, Nordsieck/AN5/JVODE variants and source-specific QNDF policies. Record
  supported behavior, equivalences and justified dispositions individually;
  existing fixed-target factories do not complete these distinct controllers.
  AdaptiveAdams now supplies source-checked neighboring defects and accepted
  target orders 3–5, with portable history, switching and consumer gates;
  [evidence](docs/development/adaptive-adams-validation.md). Full source VCABM
  startup/orders 1–12/FSAL and Nordsieck/AN5/JVODE remain separate work.
- [ ] Resolve the remaining wave-11 stabilized, Rosenbrock and exponential
  candidates, plus approximate-matrix-factorization operator contracts. These
  need recurrence/operator/solver and consumer gates, rather than only tableaus.
  Rosenbrock now supplies the corrected-time Rosenbrock23/32 pair and source
  W6S4OS map through full-JVP linear stages, complete Jacobian derivatives,
  isolated lifecycle probes and measured nonlinear/stiff consumers;
  [validation](docs/development/linearized-validation.md). Individual stability
  limits are retained. OrthogonalRK now supplies ROCK2/4, SERK2 and RKMC2 with
  translated source kernels, guarded stage bounds, high-count instability
  dispositions, nonlinear/lifecycle/gradient/adaptive gates and measured heat
  consumers; [validation](docs/development/orthogonal-validation.md).
  The fixed RKC1/RKC2/RKL2 paths now use consistent clocks, exact passive fields
  and complete finalization weights, restoring RKC2/RKL2 time-dependent order
  without extra RHS calls; [audit](docs/development/rkc-clock-validation.md).
  Exponential and AMF work remains separate.
- [x] Resolve improved IRKN history and parallel DIRK candidates from wave 12
  with their startup, second-order state, solve and source qualifications.
  [The improved-IRKN source audit](docs/development/irkn-source-audit.md) now
  records constant-cache failures and order-two nonautonomous source startup.
  ImprovedNystrom now implements corrected startup with source warm trajectories,
  nonlinear/time-dependent order, lifecycle, gradients, guarded history, restart
  and measured force/accuracy gates; [validation](docs/development/improved-nystrom-validation.md).
  ParallelDIRK now implements the ordinary first-order PDIRK44 map through the
  serial DIRK solver, with source branch equivalence, exact order/stability,
  lifecycle/gradient/restart and consumer gates; [validation](docs/development/parallel-dirk-validation.md).
  Source inspection corrected the old plan's mistaken partitioned-mechanics
  classification. Internal stage threading remains an execution-backend concern.
- [x] Prioritize a benchmarked niche: s=3 collocation improves known-consumer
  accuracy/work while passing the spatial bar. Generalized-alpha is source-confirmed
  and now has a [known two-mass consumer and linear prototype](docs/development/structural-demo-validation.md).
  It demonstrates selective damping and identifies pinned-source parameter/memory
  issues. The parameterized package driver now carries accepted acceleration
  through failures, shortened steps and in-memory restart, with nonlinear/time-dependent
  force, quantity, lifecycle and gradient gates. [Driver validation](docs/development/generalized-alpha-validation.md).
  Further Rosenbrock additions need stage-driver
  compatibility and a measured advantage.
- [ ] Advance wave 12/13 work through Phase 7 where contracts overlap. Design
  extrapolation and map semantics independently where they do not.
  Explicit Euler/midpoint extrapolation now has fixed-column source, order,
  quantity, gradient, adaptive and composition gates; [backend evidence](docs/development/extrapolation-validation.md).
  Algorithm switching now retains active-child history, cold-starts on transitions,
  delegates adaptive control and supports portable nested restart with a declared-
  bound varying-stiffness consumer; [backend evidence](docs/development/algorithm-switch-validation.md).
  Direct FunctionMap now has an explicit StateMap/tensor_map contract, endpoint
  clock, owned lifecycle and analytic recurrence/heat consumers;
  [backend evidence](docs/development/function-map-validation.md).
  Implicit extrapolation and source-specific cache transfer retain their own contracts.
  Remaining implicit extrapolation, RKIP and IRKC candidates need their own
  frozen-operator/solver and known-consumer gates.
- [ ] Keep waves 14–15 as prototypes until matrix/operator, multirate, and derivative
  contracts and cost models are convincing.
- [x] Screen all 60 wave-16 source methods and retain individual dispositions.
  ExpandedRK supplies 54 qualified main maps / 19 estimates outside registration;
  six source methods and one companion fail the binary64 gates. Canonical-map
  comparisons find no fixed-tableau duplicates; [source/consumer evidence](docs/development/expanded-rk-validation.md).
- [ ] Consider a genuine low-storage execution path when allocation measurements
  justify it; the qualified expanded maps retain full-stage storage.

Completion per batch: pinned coefficients, numerical property gates, gradient and
adaptive coverage where applicable, complete user docs, disposition records, and a
measured reason to retain the method. Reassess the plan at each driver boundary.

First batch validated: 7,096 passed, 621 skipped; installed-wheel, examples,
catalog, repository links and site gates pass. Source/companion and actual
registered-driver scale evidence are in [the batch record](docs/development/phase8-wave9-validation.md).

### Package structure and public API refactor after Phase 8

Priority: after Phase 8 backend completion and before the user-facing documentation
pass. Requested at user review on 14 October 2026. Treat warpSPH and its sibling
projects as the supported consumer set for this rewrite; do not retain the broad
root API or flat module paths solely for hypothetical external callers.
[Refactor design and initial consumer audit](docs/plans/package-api-refactor.md).

- [ ] Refresh the supported-consumer import and usage audit after Phase 8. Include
  wildcard-derived uses, deep imports, serialized scheme identifiers, custom
  lifecycle behavior, tableau access and device-clock/graph hooks.
- [ ] Split state/contracts, scheme definitions and coefficients, numerical
  drivers, solvers, runtime orchestration, wrappers and adapters into meaningful
  subpackages. Establish dependency direction and remove registry/driver cycles
  and concrete-family dispatch lists where a common stepping contract belongs.
- [ ] Define a small explicit root API for ordinary use and focused public
  subpackage APIs for SPH systems and solver extensions. Select fixed schemes
  through one canonical lookup and enums; keep raw named scheme functions and
  coefficient-generation internals out of the root facade. Retain explicit
  parameterized factories where configuration is genuinely necessary.
- [ ] Migrate warpSPH, warpSPHCore, warpSPHCases and warpSPHPlotting together with
  repository examples, tests, benchmarks, source tools and packaging. Replace
  wildcard imports with explicit imports. Preserve serialized identifiers or
  supply an explicit migration where storage formats change.
- [ ] Preserve numerical maps, method metadata, lifecycle/quantity behavior,
  solver diagnostics, gradients, accepted-history/rejection semantics, restart
  and the consumer graph/clock contract throughout the structural change.
- [ ] Validate the full numerical suite, installed-wheel exports/imports and
  source/catalog tooling, then exercise representative supported-consumer flows.
  Separate missing downstream dependencies from actual regressions and record
  any consumer check that cannot be run.

Completion: code ownership and dependencies are clear, the root API is small
and explicit, supported consumers use the new interfaces, and measured numerical
and state behavior is preserved. A directory move with the old export surface
re-created through wildcard imports does not complete this pass. Update API
references and runnable examples during migration; the subsequent documentation
pass explains the resulting interfaces more broadly.

### User-facing documentation pass after the refactor

Priority: immediately after the post-Phase 8 package/API refactor, before Phase 9
performance work. Requested at user review on 14 October 2026. The revised README establishes
the tone: explain the problem and behavior directly, use friendly method names
alongside exact enum members, and show an understood runnable example before
specialist detail. Backend interfaces should be settled before this full pass.

- [ ] Review the documentation landing page, quick start, method-selection guide,
  family guides, solver explanations, run/options/API pages, examples and notebook
  introductions as one user journey. Explain terms and prerequisites where used.
- [ ] Use friendly author/construction names consistently, retaining exact enum
  members, signatures and variant distinctions so examples remain discoverable
  and usable. Preserve source and numerical qualifications.
- [ ] Explain what each family does, when to choose it, what its important
  parameters mean and what its behavior looks like on a known problem. Keep
  essential assumptions, limitations and practical advice on the relevant page.
- [ ] Replace internal phase/wave shorthand and stale implementation status in
  user-facing prose with current capabilities and plain explanations. Keep
  development history and detailed provenance in their existing reference records.
- [ ] Reconcile examples, benchmark interpretation, method properties and links
  with the completed Phase 8 contracts; avoid making readers follow several
  documents to understand a basic choice or run a first example.
- [ ] Execute published examples and validate generated catalogs, links/anchors
  and the documentation-site build. Review the rendered pages for readability.

Completion: a new user can understand the available approaches, select a method,
run a complete example and interpret its result from the user documentation.
This is a documentation and usability pass; new numerical methods or backend
performance projects remain separate work.

### Phase 9 Establish backend and performance projects

Priority: guided by real SPH scale. The planned sequence is Phase 8, the
package/API refactor, the user-facing documentation pass, then this performance
work. Size: large.

- [ ] Profile full steps at realistic field/vector sizes on CPU and CUDA, separating
  RHS work, clone/update traffic, Krylov work, synchronization and graph memory.
- [ ] Choose gradient ownership for a native Warp path and specify buffer/aliasing
  and lifecycle contracts before implementing kernels or pooling.
- [ ] Prototype a buffer-backed RK path and, separately, fixed-iteration graph
  capture; preserve the allocating torch path's gradients and state semantics.
- [ ] Add representative device/dtype checks and benchmark artifacts with versions,
  hardware, problem size, tolerances and commands.

Completion: a native backend or storage optimization has measured speed/memory
benefit and verified numerical/state/gradient parity. Keep downstream pressure JVP,
mDBC and surface work in their owning repositories with explicit integration gates.

## Test performance follow-up

The suite already runs in multicore worker processes. Before changing its
structure, measure 1/2/4/8-worker scaling, profile expensive trajectory cases and
rebalance CI by durations while preserving complete merge coverage. Keep this as
measured follow-up work; Phase 3 documents the existing runner and does not change
numerical tolerances or drop cases.
