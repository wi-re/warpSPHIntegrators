---
sidebar_label: Solver stack
---

# Nonlinear solver, preconditioning, adaptive step control

The machinery underneath every implicit family: how a stage equation is closed,
how the linear solves stay matrix-free and differentiable, and how the embedded
error estimates turn into step-size proposals.

## The `NonlinearSolver` protocol

The nonlinear stages in DIRK, coupled RK, Newmark and implicit multistep methods is written as a fixed-point problem

$$
Y = G(Y), \qquad G(Y) = \text{base} + \text{(diagonal term evaluated at } Y\text{)},
$$

and closed by a `NonlinearSolver.solve(step_fn, y0, norm, **opts) -> SolveResult`
from `warpSPHIntegrators.solvers`. Three implementations:

| Solver | Behaviour | Use when |
|---|---|---|
| `FixedPointSolver(iterations=2)` | Exactly `iterations` Picard iterates, no convergence check. | Non-stiff problems where a fixed depth is known to be accurate; fixed-depth solves can suit graph capture when the state and RHS backend also support it. |
| `RelaxedFixedPointSolver(relaxation=0.5, iterations=2)` | Picard with under-relaxation and optional early exit. | Mildly stiff problems, cheaper than Newton. |
| `JFNKSolver(...)` | Inexact Newton (outer loop) + GMRES (inner linear solve), matrix-free. | Default for DIRK, coupled RK, Newmark and implicit multistep/IMEX stages. ROS3P instead uses a frozen-operator linear solve. |

`SolveResult` carries the converged iterate and a `SolveDiagnostics` entry
(residual, `gmres_iterations`, `rhs_evaluations`, `termination`
`'tolerance'`/`'stagnation'`, `line_search_backtracks`). Count outer map/RHS
calls, Jacobian actions and physical stage work separately: exact JVP, finite-
difference and frozen-operator drivers do not share one universal RHS-cost
formula. The benchmark's counted callback calls and wall time give the practical
cost alongside solver diagnostics.

## JFNK (`JFNKSolver`)

Outer loop: evaluate $G(Y)$ once, check convergence, otherwise take one Newton
correction. Each correction is a GMRES solve against the Jacobian of
$Y - G(Y)$, applied **matrix-free** by one of two matvecs:

- `matvec='fd'` (the solver's own default) — finite differences; works for any
  `step`. `fd_eps` defaults to the `eps^(1/3)`-scale optimal.
- `matvec='jvp'` — the **exact** Jacobian action by forward-mode autograd
  (`jvp_matvec`). One forward-mode pass, no `fd_eps`; recommended for RHS built
  from warpSPHCore's wrapped operators. `JVP == FD` is pinned to
  $5.3\times10^{-9}$ relative in `tests/test_fullyimplicit.py`.

Convergence has two independent exits:

- `norm(Y, G(Y)) < newton_tol` (defaults to `tol`, the GMRES tolerance, when not
  given). Callers with a differently-scaled `norm` — e.g. the DIRK driver's
  Hairer-Wanner weighted-RMS, whose own convention is "< 1 means converged" —
  can pass `newton_tol` explicitly. DIRK currently leaves the solver's
  threshold unchanged unless `solver_opts` overrides it.
- **stagnation floor**: if the norm fails to improve by a factor of
  `newton_stagnation_ratio` (0.9) against both its best-so-far and
  the immediately preceding value for `newton_stagnation_patience` (2)
  consecutive iterations, the plateau *is* the solve's floor (it is
  resolution-dependent: ~1.5e-4 at 16K particles, ~2.3e-3 at 1M) and the
  low-level solve reports `termination='stagnation'`. This is a precision-floor
  exit, not proof that the requested tolerance was met. `solve` rejects it by
  default unless `allow_stagnation=True` is deliberately selected.

Options: `max_iterations` (Newton-correction budget, 20; the `step` call count
is budget + 1), `gmres_tol`/`gmres_maxiter`/`gmres_restart` (forwarded to
`gmres`; restart 30 is the default), optional `line_search` (backtracking on
the trial residual; rejections counted in `line_search_backtracks`).

For an exactly linear stage operator Newton converges in one correction.
For a nonlinear one, 3-6 outer iterations at GMRES tolerance $10^{-8}$ are the
typical float64 cost (see the wave-equation notebook for the measured
breakdown).

## GMRES and preconditioning

`jfnk.gmres(matvec, b, tol, maxiter, restart, ...)` builds an Arnoldi basis
and a small Hessenberg system, with Givens rotations for the least-squares
solve. It applies the system operator through matvecs and reports the
`gmres_iterations` count. It supports **left and right
preconditioning** :

[example:skip Provide my_preconditioner and the step dt]: #
```python
from warpSPHIntegrators.solvers import JFNKSolver

solver = JFNKSolver(
    preconditioner=my_preconditioner,   # callable: (v, state, context) -> vector
    preconditioning='right',            # 'right' solves A M z = b, returns M z
                                        # 'left'  solves M A x = M b, returns x
    preconditioner_context={'dt': dt},  # merged into context, alongside 'state'
)
```

Ready-made examples: `identity_preconditioner` (bitwise the unpreconditioned
solve) and `diagonal_preconditioner` (mass/diagonal scaling). The hook is a
3-argument callable so a preconditioner can see the current Newton iterate and
problem context. Measured on the advection benchmark, diagonal preconditioning
helps only weakly (condition number 5.06 on the 256-DOF block system) — the
block system's spectrum is a two-dimensional annulus, not a diagonal-dominant
cluster (see the numerical record).

## Stage-aware preconditioning and warm guesses

Coupled RK supplies `dt`, `tableau_a` and `stage_times` in preconditioner context.
`stage_block_preconditioner(jacobian_diagonal)` accepts a flattened length-N
Jacobian diagonal, an s-by-N tensor or `(BlockState, context) -> tensor`. It
factors an s-by-s matrix per DOF, retaining stage coupling while omitting spatial
and field off-diagonals. LU storage is O(s²N), cached within one Newton iterate.
This is a forward preconditioner; the adjoint retains its existing matrix-free
transpose solve. For example:

[example:run stage-preconditioner]: #
```python
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, get_scheme
from warpSPHIntegrators.solvers import stage_block_preconditioner

rates = torch.logspace(0, 3, 64, dtype=torch.float64)
state = TensorState(torch.ones_like(rates))
rhs = tensor_rhs(lambda t, y: -rates*y)
result = get_scheme('Radau IIA s=2')(state, .01, rhs, solver_opts={
    'matvec': 'jvp', 'tol': 1e-10, 'newton_tol': 1e-7,
    'preconditioner': stage_block_preconditioner(-rates),
})
assert result.step_status.tolerance_converged
```

DIRK supplies `dt`, `stage_index`, `stage_time` and
`stage_coefficient=dt*a_ii`. Driver-owned context keys override application keys.
An implicit-first DIRK can accept `warmStart=previous.stages[0]`; coupled RK
accepts the full previous stage list. These are guesses, never derivative reuse.
They remain opt-in: the measured transient cases show no consistent gain, and a
warm guess can cost more than a cold solve. The local stage-block approximation
passes reaction cases but fails the original diffusion scale gate. Shifted spatial
solves subsequently qualified fixed s=3 registrations, and real Schur spatial
preconditioning supports the parameterized stages 2–9 backend. See the
[scale evidence](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/phase6-validation.md)
and [higher-stage qualification](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/higher-collocation-validation.md).

## Spatial solves for coupled stages

`stage_shifted_preconditioner(build_shifted_solver)` diagonalizes the small RK
stage matrix and applies shifted spatial inverses. The factory receives the
current BlockState and context, then returns `solve(shift, complex_vector)` for
`(I - shift*J_frozen)`. The application supplies sparse factors, a multigrid
hierarchy or another spatial solve. A common Jacobian decouples stages exactly
for linear systems; nonlinear stage Jacobians remain distinct in the actual
Newton residual. Ill-conditioned eigenbases are refused. This helper uses Torch
only and is a forward preconditioner; adjoint solves retain their existing path.

[example:run shifted-spatial]: #
```python
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, get_scheme
from warpSPHIntegrators.solvers import stage_shifted_preconditioner

rates = torch.logspace(0, 3, 32, dtype=torch.float64)
preconditioner = stage_shifted_preconditioner(
    lambda state, context: lambda shift, vector: vector / (1 + shift*rates))
result = get_scheme('Radau IIA s=2')(
    TensorState(torch.ones_like(rates)), .01, tensor_rhs(lambda t, y: -rates*y),
    solver_opts={'matvec': 'jvp', 'tol': 1e-10, 'newton_tol': 1e-7,
                 'preconditioner': preconditioner})
assert result.step_status.tolerance_converged
```

The [spatial follow-up](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/spatial-followup-validation.md)
compares direct sparse LU, ILU and shifted spatial solves on the unchanged
reaction/diffusion gate. Its passing 1D result does not establish a generic
unstructured SPH preconditioner or remove the method-specific companion/source gates.

## Adaptive step control

The [run API](runs) owns adaptive acceptance and rollback. For applications that
keep their own stepping loop, `warpSPHIntegrators.runtime` exposes these helpers:

- `estimate_error_norm(result, rtol=1e-3, atol=1e-6)` measures the
  weighted RMS of `result.error`, scaled against `result.state`. A value of
  1 is at the requested tolerance; a result without an estimate raises.
- `propose_dt(error_norm, dt, order, target=1.0, safety=0.9,
  growth_min=0.2, growth_max=5.0)` scales the step by
  $\mathrm{safety}\,(\mathrm{target}/\mathrm{error\_norm})^{1/q}$,
  clamped to the supplied growth bounds. Zero error grows to `growth_max`.
  The positional `order` argument is the **estimate exponent q**:
  pass `scheme.estimate_order`. TR-BDF2 and Gauss-Legendre 2 both use q=3,
  despite propagated orders 2 and 4.
- `dormand_prince_dense_output(state, stages, dt, theta)` evaluates DP5's
  quartic at the fraction `theta` of a step, with exact endpoint reproduction.

[The generated catalog](catalog) lists every estimate emitter and exponent.
The fixed-catalog multistep schemes emit no estimates, and changing dt restarts
their history. VariableAdams, NDF and AdaptiveNDF instead provide their own
comparison exponents and variable-history/control contracts.
Use [the run API](runs) for acceptance/rejection, final-time clipping and state/history
commit, or retain a caller-owned loop with these helpers.

[example:run adaptive]: #
```python
from warpSPHIntegrators.runtime import estimate_error_norm, propose_dt
from warpSPHIntegrators import get_scheme
import warpSPHIntegrators.testing as testing

problem = testing.oscillator_problem()
scheme = get_scheme('Gauss-Legendre 2')
initial = problem.initial()
dt = 0.1
result = scheme(initial, dt=dt, f=problem.rhs)
norm = estimate_error_norm(result, rtol=1e-6, atol=1e-9)
next_dt = propose_dt(norm, dt, scheme.estimate_order)
if norm <= 1.0:
    system = result.state  # accept the candidate
else:
    system = initial      # retry from the unchanged accepted state
    next_dt = min(next_dt, dt)
```

## Cost model

The nonlinear-solver overhead is benchmarked, not assumed:
`benchmarks/jfnk_preconditioner_benchmark.py` and the `examples/integrations/jfnk_wave_equation.ipynb` notebook
measure GMRES iterations vs. the frozen-operator condition number for the wave
equation, and `benchmarks/blockrk_benchmark.py` the block-solve cost for the
coupled pair. In those measured problems, for a problem whose
stage-operator spectrum is clustered, JFNK at the default tolerances costs
3-6 `step` evaluations per stage; for a spread (annular) spectrum, the inner
Krylov may need no-restart GMRES on the full system (`gmres_restart` ≥
system size) rather than a cheap restart (see the numerical record).
