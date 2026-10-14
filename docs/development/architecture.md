# Current architecture

Status: 14 October 2026, through Phase 4. The runtime catalog contains 137
registered schemes in 11 families, plus three standalone parameterized pRRK
drivers. [The generated catalog](../docs/catalog.md) owns current method counts
and capability metadata. [The execution plan](../../REPOSITORY_PLAN.md) owns work
statuses; [numerical records](../research/integrators/notes-index.md) explain the
measurements and coefficient decisions.

## State, updates and lifecycle

The package advances user-owned systems through field tags and update operations.
`fields.py` defines dataclass behaviors (`integrated`, `constant`, `copied`,
`ephemeral`, `custom`), cloning and integrated-field flattening. `specs.py` defines
typed blends, component/position updates, stage results and the frozen
`IntegrationResult` dataclass. `protocol.py` and `util.py` dispatch typed updates
or legacy system methods and invoke initialize/preprocess/postprocess/finalize
hooks. The integrator does not own particle storage or neighbor-list policy.

Drivers build stage states through these operations, set stage times and combine
derivatives. `reuse.py` analyzes first-stage reuse; callers pass `priorStep` only
when its metadata supports the intended order. Derivative-stage counts can differ
from the number of mathematical stages in a paper.

## Numerical drivers and discovery

`integration.py` registers the public `IntegrationScheme` tuples and lookup;
`enums.py` supplies identifiers and family enums. `catalog.py` provides derived
immutable metadata without changing the tuple's twelve-field contract.
`filter_schemes` composes discovery criteria. `TensorState` / `tensor_rhs` adapt
ordinary floating-point `(t, y)` callbacks, including splits, to the existing
protocol without changing numerical drivers. `getPreferredIntegrator` returns
conventional ordinary-ODE records; `getPreferredScheme` keeps its legacy raw
callable choices. Structural metadata records second-order/separability/drift
assumptions and retained-history length.
Explicit/geometric, DIRK, coupled RK, multistep, IMEX, stabilized, Rosenbrock,
exponential and Newmark methods use their respective drivers. `rhs.py` declares
combined, split and semilinear capabilities. A KenCarp/ARS name registered in the
DIRK family represents its implicit tableau, not an upstream additive pair.

`solvers.py` and `jfnk.py` provide fixed-point and matrix-free Newton/GMRES solves,
FD/JVP products, preconditioning, diagnostics and implicit differentiation.
`history.py` stores fixed-step history and restarts it when the timestep changes.
`adaptive.py` supplies error norms, step proposals and DP5 dense output. `run.py`
owns optional transactional fixed/adaptive orchestration, outputs and events;
`checkpoint.py` persists tensor-only restarts. Single-step callers can keep their
own loop. External side effects require application transaction hooks.

## Current boundaries

Numerical stepping is PyTorch based. Warp array clone dispatch and Warp-backed
downstream RHS operations do not establish a native Warp integrator backend.
Buffer pools, graph capture and gradient ownership remain future design work.
The unused `fluid_only` option was removed after checking this package and the
local warpSPH / warpSPHCore checkouts. Particle masking remains the user's
system/RHS responsibility; integrated fields do not imply particle selection.
Velocity-dependent forces violate the assumptions of several geometric methods;
the measured order loss is retained as a regression control. Explicit DP5 startup
can limit stiff BDF3–5 trajectories even where the steady-state formula is stable.

## Repository layout

`docs/docs` contains the user-facing site; `docs/plans` contains specialist designs
and method dispositions; `docs/development` contains contributor and validation
records. Dated numerical evidence lives in `docs/research/integrators`, and
cross-repository SPH experiments in `docs/research/downstream`. Standalone examples
and notebooks live in `examples`; performance/gallery drivers in `benchmarks`;
maintenance and coefficient utilities in `scripts`; durable plots in `docs/assets`.
Import package paths under `src/warpSPHIntegrators` remain stable.
