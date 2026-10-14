# Repository execution plan

Updated 14 October 2026. This is the authoritative current task list.
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

The 137-scheme catalog is committed locally and remains unreleased beyond package
0.6.1. Expansion waves 1–8 are complete with recorded exceptions; waves 9–16
remain subject to the gates below. Commit creation and package publication are
separate actions. **Phase 7's package/toy scope is complete. Phase 8 is next.
The spatial follow-up clears the recorded wave 9 scale cases; companion/source/
catalog gates remain before registration. PESPH integration is later application work.**

## Phase 3 completion record

- [x] Reconcile plan statuses; close SSPRK54; record PFRK87, Anas5/FRK65,
  additional ExplicitTableaus candidates and upstream estimator exceptions.
- [x] Record implemented/equivalent/deferred/excluded/awaiting-verification method
  dispositions, source pins, constructors, transformations, tests and consumers.
- [x] Organize specialist plans and dated records, with a measurement index and
  compatibility anchors for old numbered references.
- [x] Shorten README; document environments, category gates, multicore execution,
  registration, provenance and release validation in CONTRIBUTING.
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

- [x] Clear the recorded Phase 6 spatial scale bar with shifted spatial ILU.
  [Follow-up evidence](docs/development/spatial-followup-validation.md).
- [ ] Complete wave 9 companion/source/catalog gates before registration; general
  downstream SPH spatial/field assembly remains application-specific.
- [ ] Re-enumerate waves 10–11 against pinned sources and split simple tableaus
  from history, recurrence, or driver changes. Variable coefficients are primarily
  a timestep/history issue; do not assume fixed-step methods exclude non-autonomous
  RHS functions.
- [ ] Prioritize additions with a benchmarked niche, such as generalized-alpha for
  structural dynamics or an improved stiff Rosenbrock method.
- [ ] Advance wave 12/13 work through Phase 7 where contracts overlap. Design
  extrapolation and map semantics independently where they do not.
- [ ] Keep waves 14–15 as prototypes until matrix/operator, multirate, and derivative
  contracts and cost models are convincing.
- [ ] Screen wave 16 and maintain equivalence records; consider a genuine low-storage
  execution path only when allocation measurements justify it.

Completion per batch: pinned coefficients, numerical property gates, gradient and
adaptive coverage where applicable, complete user docs, disposition records, and a
measured reason to retain the method. Reassess the plan at each driver boundary.

### Phase 9 Establish backend and performance projects

Priority: guided by real SPH scale. Can follow Phase 3 without waiting for all
catalog additions. Size: large.

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
