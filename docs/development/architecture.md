# Current architecture

Status: 15 October 2026, after the Phase 8 candidate review and 0.7 package/API
migration. The [generated catalog](../docs/catalog.md) owns fixed method counts
and capability metadata. Parameterized configurations extend that catalog;
experimental tensor prototypes have separate, bounded contracts. The
[execution plan](../../REPOSITORY_PLAN.md) owns work status and the
[refactor validation](package-refactor-validation.md) records preserved numerical,
storage and consumer behavior.

## Package ownership

| Package | Responsibility | Dependency boundary |
|---|---|---|
| `state` | Field behaviors, typed updates, lifecycle, clocks, ownership, history and shared result/stepping contracts | Independent of method registration and run orchestration |
| `schemes` | Enums, metadata, discovery, registry and family configuration | Binds definitions to numerical drivers |
| `schemes.coefficients` | Pure coefficient construction and pinned data | Does not select or execute methods |
| `schemes.definitions` | Immutable generic explicit, DIRK, coupled and additive definitions | Owns read-only coefficients and delegates execution |
| `drivers` | Numerical stepping by RK, history, mechanics, stabilized, exponential, extrapolation and map families | Consumes low-level contracts, definitions and solvers without importing the registry/runtime |
| `solvers` | Nonlinear iteration, Krylov actions, preconditioners, matrix functions and implicit adjoints | Independent of concrete method registration |
| `runtime` | Fixed/adaptive run control, accepted/rejected history, outputs/events and portable checkpoints | Uses the structural stepping/control contracts |
| `wrappers` | Physical composition and algorithm switching | Delegates child steps and owns wrapper history/acceptance |
| `adapters` | Tensor ODE, split/semilinear RHS and complete-state-map callbacks | Connects user data/callables to the state contracts |
| `analysis` | Stability, reuse, TVD and geometric inspection | Optional analysis rather than driver dispatch |
| `experiments` | Source-audited tensor prototypes | Outside registered scheme lookup and full-state solve |

The root exports exactly `get_scheme`, `solve`, `TensorState` and `tensor_rhs`.
Use enums and the canonical lookup for fixed methods, and explicit factories in
`schemes` for configurations such as collocation stages or generalized-alpha
damping. Raw per-method functions, preferred-method aliases and old flat modules
are removed. The small `enums` bridge preserves stored enum pickles; it does not
restore the old import facade. See the [migration guide](../docs/migration.md).

## Package design rationale

The 0.7 change separates ownership and dependency direction as well as imports.
Moving the old flat surface under another generic package would preserve its
coupling. Pure coefficient builders therefore do not execute methods or select
the registry; definitions bind immutable coefficients to shared engines. History
and result contracts live below runtime/driver consumers, while each numerical
driver owns its method's candidate memory. The runtime alone decides acceptance.

Configuration and physical composition remain distinct: selecting a child
algorithm is not splitting a vector field. Direct FunctionMap stepping belongs
to a driver; adapting user callbacks belongs to adapters. Structural stepping
and accepted-control protocols avoid reproducing each child's algorithm or
maintaining concrete factory whitelists in wrappers and run control.

The supported migration set is warpSPH and its sibling projects. The small root
serves ordinary users; intentional coefficient, solver and state interfaces serve
advanced consumers without restoring every old raw-function export. Stored enum
and checkpoint compatibility is separate from import compatibility. The
[initial consumer audit](package-refactor-validation.md#initial-consumer-audit)
retains the original scope and why tableau, finalizer, serialization and clock
checks were required. Numerical changes need their own evidence rather than
being hidden in a structural move.

## State, updates and lifecycle

The package advances application systems through tagged fields and update
operations. `state.fields` owns dataclass field behaviors (`integrated`,
`constant`, `copied`, `ephemeral`, `custom`), cloning and integrated-field
flattening. `state.specs` describes blends, component/position updates and stage
results. `state.protocol` and `state.lifecycle` dispatch typed updates or legacy
system methods and invoke initialize/preprocess/postprocess/finalize hooks.
Particle storage, neighbor lists and particle masking belong to the application.

Drivers build isolated stage states, set stage times and combine derivatives or
method-specific increments. All integrated quantities participate in the declared
state map. Finalizers must respect scalar, component or operator quadratures
rather than assume every family has ordinary RK weights. Analysis metadata gates
first-stage reuse; derivative-stage counts can differ from a paper's mathematical
stage count. Nonlinear/Jacobian probes follow the owned-state contract.

## Definitions, execution and run control

Frozen RK definition records own read-only coefficient arrays and retain callable
identity equality/hash semantics. The registry binds those records and assigns
metadata. Parameterized configurations validate their inputs and delegate their
numerical maps to family drivers. Pure builders and execution helpers have
separate modules; execution is not attached to raw function exports.

`state.Stepper` and optional accepted-control protocols let custom methods run,
switch and compose without joining a concrete factory whitelist. Optional
capabilities still have explicit gates: physical subflows reject unsupported
history, negative dissipative steps remain guarded, and family-specific RHS and
force assumptions remain part of selection.

The runtime owns accepted/trial transitions, rollback, error control, output and
events. Drivers own method-specific numerical history candidates. A failed or
rejected trial does not silently become accepted memory. Portable checkpoints
store tensor/primitive state and scheme identifiers; callers resupply functions
and external solver/application resources. Single-step callers can keep their
own loops. External side effects require application transaction hooks.

The solver packages provide fixed-point and matrix-free Newton/GMRES iteration,
FD/JVP products, preconditioning, diagnostics and implicit differentiation.
Nonlinear convergence, physical RHS evaluations and linear matvec counts are
separate measurements. Full-JVP and exponential operator derivatives have their
own documented size/action limits; differentiability is a family contract.

## Numerical and backend limits

Numerical stepping is PyTorch based. Warp array cloning and downstream Warp RHS
operations do not establish a native Warp integrator backend. Storage pooling,
CUDA graph capture and native gradient ownership remain deferred backend and
performance work. The
consumer migration checks real CPU configuration, state, restart and deferred
clock paths; CUDA hardware was unavailable.

The unused `fluid_only` option remains removed following the consumer audit.
Velocity-dependent forces still violate several geometric methods' assumptions;
order-loss controls remain tests. BDF/NDF/Adams startup and adaptive order
contracts differ by configuration. Deferred upstream controller ports and source
order failures retain explicit dispositions in the candidate-review records.

## Repository layout

`docs/docs` contains user documentation; `docs/plans` contains the durable method
disposition ledger and its index. `docs/development` contains architecture,
retained designs, reconciliation records and validation.
Dated numerical evidence lives in `docs/research/integrators`, and downstream SPH
experiments in `docs/research/downstream`. Examples/notebooks live in `examples`,
measurement drivers in `benchmarks`, coefficient/maintenance tools in `scripts`
and durable plots in `docs/assets`. Public package boundaries and import examples
are documented in the [API reference](../docs/api.md).
