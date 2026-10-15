# Public API

The root has four ordinary entry points. Choose a method through `get_scheme`
and its enum, adapt the equations with `tensor_rhs`, and let `solve` own the
trajectory. A `TensorState` stores the current tensor and time.

[example:run api-basic]: #
```python
import torch
from warpSPHIntegrators import get_scheme, solve, TensorState, tensor_rhs
from warpSPHIntegrators.schemes import ExplicitRK

method = get_scheme(ExplicitRK.tsit5)  # Tsitouras 5(4)
run = solve(method, TensorState(torch.ones(2, dtype=torch.float64)), 1.,
            tensor_rhs(lambda t, y: -y), dt=.1, adaptive=True,
            rtol=1e-7, atol=1e-9)
assert run.successful
assert method.identifier.name == 'tsit5'
```

| Entry point | Meaning |
|---|---|
| `get_scheme(identifier)` | Return a fixed scheme record selected by enum or exact lookup name; a bound family configuration passes through unchanged |
| `solve(method, state, t_end, rhs, *, dt=..., ...)` | Integrate a fixed/adaptive trajectory with acceptance, history, sampling and restart |
| `TensorState(y, t=0.)` | A nonempty floating scalar, vector or batched tensor with its clock |
| `tensor_rhs(function, **parameters)` | Adapt f(t,y) and bind named equation parameters; step-control options stay outside the callback |

A direct step is `method(state, dt, rhs)`. It returns an `IntegrationResult` with
`state`, physical `stages`, optional `error`, `history`, solver diagnostics and
`step_status`. `solve` returns a `RunResult` with accepted/rejected counts,
samples, events and a restart checkpoint. The [run guide](runs) explains when
state, history and side effects are committed.

## Focused imports

| Package | Use it for |
|---|---|
| `warpSPHIntegrators.schemes` | Family enums, metadata, catalog filtering and validated method configurations |
| `warpSPHIntegrators.state` | Custom state/system protocols, field tags, typed component updates, stage/result and history contracts |
| `warpSPHIntegrators.solvers` | Fixed-point/Newton–Krylov solvers, GMRES, preconditioners and solve diagnostics |
| `warpSPHIntegrators.runtime` | Events, accepted-step records, portable checkpoint I/O and run-control helpers |
| `warpSPHIntegrators.adapters` | Structured additive/semilinear/physical RHS and complete state-map callbacks |
| `warpSPHIntegrators.wrappers` | Algorithm switching and physical flow composition |
| `warpSPHIntegrators.schemes.coefficients` | Advanced coefficient construction and inspection |
| `warpSPHIntegrators.analysis` | Reuse and other numerical inspection tools |

Raw scheme functions are absent from the root. `get_scheme` retains exact enum
and stored lookup identities; friendly display names are not extra string aliases.
For example, use `ExplicitRK.tsit5` for Tsitouras 5(4). See the
[migration guide](migration) for the deliberate import changes.

## Discover and inspect a method

[example:run api-inspect]: #
```python
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.schemes import ExplicitRK, filter_schemes

choices = filter_schemes(family='ExplicitRK', min_order=4,
                        has_error_estimate=True, state_kind='tensor')
assert get_scheme(ExplicitRK.tsit5) in choices
rk4 = get_scheme(ExplicitRK.rungeKutta4)
tableau = rk4.definition.tableau
assert len(tableau.c) == 4
assert not tableau.a.flags.writeable
assert rk4.metadata.state_kind == 'tensor'
```

Fixed records include order, implicit/FSAL/reuse properties and a stable
identifier. Metadata describes stage count, embedded order, error exponent,
required RHS capabilities, history and dense output. The error exponent q is the
power in an O(h^q) difference; it can change during multistep startup, so the
per-result `estimate_order` takes precedence. An embedded estimate controls local
error, not a universal bound on the final trajectory error.

The fixed catalog is immutable. Generic explicit, DIRK, coupled and additive RK
records hold explicit coefficient definitions; their bound arrays are read-only.
Advanced custom engines are available as `explicit_rk_step`, `dirk_step`,
`coupled_rk_step` and `additive_rk_step` in `warpSPHIntegrators.drivers`. They
consume a supplied tableau and do not select a catalog method themselves.

## Configure a family

Import these from `warpSPHIntegrators.schemes`. Each configuration is callable
and works with `solve` within its state/RHS contract.

| Configuration | What it offers | Main boundary |
|---|---|---|
| `Collocation('gauss', 4)` | Gauss–Legendre order eight; Radau IIA is also available, 2–9 stages | Coupled implicit solve; attainable accuracy has a binary64/solver floor |
| `VariableAdams(4, 'abm')`, `AdaptiveAdams()` | Variable-step Adams and accepted target orders 3–5 | Nonstiff history; discontinuities require a reset |
| `NDF(3)`, `AdaptiveNDF()` | Implicit difference history, target orders 1–5 | Quasi-constant step policy; distinct from full upstream QNDF control |
| `StabilizedRK('rkl2', lambda_max=...)` | RKL/RKG methods with a supplied spectral bound | Negative-real diffusion-like spectra |
| `OrthogonalRK('ROCK2', stages=10)` | ROCK/SERK/monotone-Chebyshev recurrences | Supported counts and conservative sizing; full stages are retained |
| `Nystrom('FineRKN5')` | Second-order mechanics, including velocity-dependent choices | Force structure and quantity lifting depend on the configuration |
| `ImprovedNystrom(4)` | Fewer warm-step forces in fixed-step mechanics | Velocity-independent acceleration and accepted history |
| `GeneralizedAlpha(.5)` | Selective damping for constant-mass mechanics | Accepted acceleration memory; no arbitrary constrained dynamics |
| `ParallelDIRK()` | L-stable fourth-order parallel-DIRK map | Serial execution here; four nonlinear solves, no estimate |
| `Rosenbrock('Rosenbrock23')` | Full-JVP linear stages and complete Jacobian derivatives | Individual methods have different stability limits |
| `ExponentialRK('ETDRK4')`, `ExponentialRK('Exprb43')` | Higher-order semilinear/full-Jacobian exponential maps | Action budgets and explicit dense operator-AD bounds |
| `Extrapolation('midpoint', 4)` | Explicit Euler/midpoint extrapolation | Fixed target order, retained full-state stages |
| `ExpandedRK('CarpenterKennedy2N54')` | A qualified source-expanded RK map | Full-stage execution; no native low-storage claim |
| `FunctionMap(tensor_map(...))` | A complete discrete transition | The callback returns a state, not an ODE rate |

[Family guides](catalog) explain the parameters and assumptions. Specialized
finalization types, including `NystromWeights`, `NDFQuadrature`,
`RosenbrockQuadrature` and `ExponentialQuadrature`, are available beside these
configurations. Their weights/increments differ; a custom finalizer must support
the selected family's contract.

## Custom systems, RHS and solvers

`state` provides `BaseState`, `BaseIntegrationSystem`, `integrated`, `constant`,
`copied`, `reference_state`, `tagged`, `ComponentUpdateSpec`, `PositionUpdateSpec`
and tag-driven update helpers. The [quick start](quickstart) and
[complete oscillator example](https://github.com/wi-re/warpSPHIntegrators/blob/main/examples/harmonic_oscillator.py)
show the field and lifecycle contract. Integrated quantities must take part in
all stages, errors and history; copied fields are refreshed from the relevant
physical endpoint/stage.

Use `IMEXRHS` for an additive explicit/implicit split, `SemilinearRHS` for a
constant linear operator plus a remainder, and `ConservativeDissipativeRHS` for
physical subflows. Plain callables remain valid for methods that do not require
a declared capability. `check_contracts` helps check that declared parts agree
with the combined equations.

Implicit steps accept `solver=` and `solver_opts=`. `JFNKSolver` solves nonlinear
stages, with FD or exact JVP actions; `FixedPointSolver` is a deliberate
fixed-iteration alternative. A small update from a fixed iteration budget does
not certify a converged nonlinear stage. Read [solver diagnostics and
preconditioning](solver) before interpreting implicit benchmark costs.

For extensions, `state.Stepper` describes the structural call/metadata contract.
The runtime and wrappers use it rather than concrete class whitelists. Optional
accepted-step controls can propose the next step/order. This permits a new bound
backend without adding its class to the runner; it does not waive its numerical,
state-ownership or failure-handling requirements.
