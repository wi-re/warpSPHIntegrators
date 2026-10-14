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

The 137-scheme catalog is committed locally and remains unreleased beyond package
0.6.1. Expansion waves 1–8 are complete with recorded exceptions; waves 9–16
remain subject to the gates below. Commit creation and package publication are
separate actions. **Phase 5 is next.**

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

### Phase 5 Add reliable run orchestration and failure handling

Priority: high. Depends on metadata from Phases 2 and 4. Size: medium to large.

- [ ] Define public step success and failure semantics, including the distinction
  between Newton stagnation at a precision floor and true tolerance convergence.
- [ ] Add a fixed-step runner with history/reuse threading and output callbacks.
- [ ] Add adaptive acceptance/rejection, field tolerances, nonfinite checks, limits,
  and rollback of state and lifecycle effects; use the recorded estimate exponent.
- [ ] Specify sampling/dense-output behavior, then add supported events and restart
  checkpoints as separate increments.

Completion: deterministic fixed-step runs and controlled adaptive runs have correct
final time, preserved caller state, observable failures, and no history or side
effect committed by rejected attempts. Existing single-step users remain supported.

### Phase 6 Address stiff startup and prove coupled scaling

Priority: high for stiff consumers; prerequisite for wave 9. Size: medium to large.

- [ ] Design an order-preserving implicit bootstrap for BDF3–5 and compare it with
  the explicit starter on a stiff transient. A low-order warm-up alone must not
  cap the trajectory's advertised order.
- [ ] Measure implicit first-stage warm guesses and block/stage-aware preconditioners
  against cold starts on representative vector systems.
- [ ] Prototype three-stage Gauss/Radau solves with configurable, recorded DOF
  sweeps, iteration counts, work units, peak memory, failure rates and wall time.
- [ ] Land wave 9 only after the scale gate; resolve exact stage/order nomenclature,
  companion estimates, symmetry/stability and gradient gates before registration.

Completion: stiff startup retains measured order at the target stiffness; the
three-stage block solve clears a documented cost/memory bar and has reproducible
benchmark evidence. If it fails the bar, keep wave 9 gated and record the result.

### Phase 7 Implement useful splitting and second-order methods

Priority: conditional on downstream physics. Size: large.

- [ ] Verify the downstream entropy-carrying prototype and baseline described in
  the splitting plan; retain its separability and energy assumptions.
- [ ] Implement conservative/dissipative capabilities and structural checks, then
  explicit Lie-Trotter and Strang orchestration with a single outer finalize.
- [ ] Verify order on damped/split test problems. Backward Euler as the dissipative
  substep is the first-order control, not the advertised second-order variant.
- [ ] Add implicit/exponential dissipative flows and measure the viscous CFL benefit.
  Attempt merging only under its fixed-step and exact-flow qualifications.
- [ ] Coordinate general RKN (wave 12), Composite (wave 13), and conditional
  high-order composition with this driver. Keep compatible-energy substeps and
  anti-diffusive negative-step limits as explicit downstream gates.

Completion: the split meets its contracts, preserves lifecycle and gradients, and
delivers measured order/stability/cost improvements on an actual consumer.

### Phase 8 Resume catalog expansion in reviewed batches

Priority: after reliable gates and a concrete benefit. Size: varies by driver.

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
