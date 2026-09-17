# Nonlinear solver, preconditioning, adaptive step control

The machinery underneath every implicit family: how a stage equation is closed,
how the linear solves stay matrix-free and differentiable, and how the embedded
error estimates turn into step-size proposals.

## The `NonlinearSolver` protocol

Every stage equation in this library is written as a fixed-point problem

$$
Y = G(Y), \qquad G(Y) = \text{base} + \text{(diagonal term evaluated at } Y\text{)},
$$

and closed by a `NonlinearSolver.solve(step_fn, y0, norm, **opts) -> SolveResult`
(`solvers.py` / `jfnk.py`). Three implementations:

| Solver | Behaviour | Use when |
|---|---|---|
| `FixedPointSolver(iterations=2)` | Exactly `iterations` Picard iterates, no convergence check. | Non-stiff problems where a fixed depth is known to be accurate; the only choice that stays CUDA-graph-capturable. |
| `RelaxedFixedPointSolver(relaxation=0.5, iterations=2)` | Picard with under-relaxation and optional early exit. | Mildly stiff problems, cheaper than Newton. |
| `JFNKSolver(...)` | Inexact Newton (outer loop) + GMRES (inner linear solve), matrix-free. | The default for every registered implicit scheme; required once the Picard map is not contractive. |

`SolveResult` carries the converged iterate and a `SolveDiagnostics` entry
(residual, `gmres_iterations`, `rhs_evaluations`, `termination`
`'tolerance'`/`'stagnation'`, `line_search_backtracks`). The Phase 1
**work-unit invariant** holds for every step:

$$
\text{total step evaluations} \;=\; \texttt{rhs\_evaluations} + \texttt{gmres\_iterations}.
$$

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
  given). Callers with a differently-scaled `norm` — e.g. `dirk.py`'s
  Hairer-Wanner weighted-RMS, whose own convention is "< 1 means converged" —
  must pass a matching `newton_tol` (`dirk.py` passes `1e-3`).
- **stagnation floor**: if the norm fails to improve by a factor of
  `newton_stagnation_ratio` (0.9) against both its best-so-far and
  the immediately preceding value for `newton_stagnation_patience` (2)
  consecutive iterations, the plateau *is* the solve's floor (it is
  resolution-dependent: ~1.5e-4 at 16K particles, ~2.3e-3 at 1M) and the
  iterate is reported converged with `termination='stagnation'`.

Options: `max_iterations` (Newton-correction budget, 20; the `step` call count
is budget + 1), `gmres_tol`/`gmres_maxiter`/`gmres_restart` (forwarded to
`gmres`; restart 30 is the default), optional `line_search` (backtracking on
the trial residual; rejections counted in `line_search_backtracks`).

For an exactly linear stage operator Newton converges in one correction.
For a nonlinear one, 3-6 outer iterations at GMRES tolerance $10^{-8}$ are the
typical float64 cost (see the wave-equation notebook for the measured
breakdown).

## GMRES and preconditioning

`jfnk.gmres(matvec, b, tol, maxiter, restart, ...)` is a Givens-free
GMRES: the Arnoldi basis is built with Householder-free Hessenbergization, so
the whole solve is a chain of matvec applications (cheap to record, and the
`gmres_iterations` count is exact). It supports **left and right
preconditioning** (Phase 2, NOTES.md §3.4):

```python
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
cluster (NOTES.md §3.18).

## Adaptive step control (helpers only)

Phase 11 deliberately landed **helpers, not a driver**: the library proposes,
the caller disposes. `adaptive.py` exposes:

- `estimate_error_norm(error, rtol, atol, reference)` — the embedded pair's
  state-shaped difference, scaled by the SUNDIALS-style control
  $\mathrm{rtol}\,|\mathrm{ref}| + \mathrm{atol}$ per component (RMS).
- `propose_dt(dt, error_norm, order, safety=0.9, max_factor=5.0,
  min_factor=0.2, dt_min, dt_max)` — the standard
  $\mathrm{dt}_{\text{new}} = \mathrm{dt}\,\mathrm{safety}\,
  \mathrm{err}^{-1/p}$ with clamps; rejects a zero/negative `dt` and an
  error of zero (keeps `dt`).
- `dormand_prince_dense_output(stages, dt)` — the 4th-degree interpolating
  polynomial through the DP5(4) stages, for output at times between steps.

The embedded pairs that feed these are listed on the
[explicit-rk](families/explicit-rk) and [dirk](families/dirk) pages
(twelve registered schemes in total, including the two coupled-block
companion pairs). Multistep schemes **refuse** adaptive control: varying
`dt` would desynchronize their history (NOTES.md §3.17). A full
adaptive driver (output at fixed times, event location) is on the
roadmap as an open item.

## Cost model

The nonlinear-solver overhead is benchmarked, not assumed:
`scripts/jfnk_cost_model.py` (and the `jfnk_wave_equation.ipynb` notebook)
measure GMRES iterations vs. the frozen-operator condition number for the wave
equation, and `scripts/blockrk_benchmark.py` the block-solve cost for the
coupled pair. The practical rule of thumb (NOTES.md §3.4): for a problem whose
stage-operator spectrum is clustered, JFNK at the default tolerances costs
3-6 `step` evaluations per stage; for a spread (annular) spectrum, the inner
Krylov may need no-restart GMRES on the full system (`gmres_restart` ≥
system size) rather than a cheap restart (NOTES.md §3.18).
