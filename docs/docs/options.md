# Step and solver options

Registered schemes use `scheme(state, dt, f, *args, **kwargs)` and return an
`IntegrationResult`. Use the run API or a caller-owned stepping loop. Options are read by
the relevant driver; arbitrary kwargs continue to reach the full state/RHS
protocol. A custom RHS should accept the kwargs its consumer forwards.

| Option | Consumer and contract |
|---|---|
| `history` | A `StepHistory`, threaded from `result.history`; required for a multistep method's steady-state formula |
| `priorStep` | Previous `StageResult`; consult `supports_reuse` / `reuse_order`. Unsupported drivers reject or warn; order-losing reuse is documented |
| `verbose` | Driver/lifecycle logging; does not change numerical behavior |
| `solver` | Implicit drivers accept a `NonlinearSolver`; default is JFNK |
| `solver_opts` | Mapping passed to the selected solver; validates known numerical controls at solve entry |

## Solver controls

`SolverOptions` provides type hints. Built-in solvers validate finite positive
tolerances (`tol`, `gmres_tol`, `newton_tol`, `fd_eps`), positive integer iteration
counts (`iterations`, `max_iterations`, `gmres_maxiter`, `gmres_restart`,
`newton_stagnation_patience`), and ratios in `(0,1]` (`relaxation`,
`newton_stagnation_ratio`, `line_search_min_step`). `line_search` must be bool;
`matvec` is `fd` or `jvp`, and `preconditioning` is `left` or `right`.
Preconditioners must be callable, and their optional context must be a mapping.

Known override values are checked before evaluating the stage map, as well as
constructor settings. Optional controls can use None to select documented
defaults. Unknown extension keys retain their historical passthrough behavior;
use `SolverOptions` type checking to catch misspellings. Arbitrary top-level RHS
kwargs are not globally rejected. Custom solvers retain their own option contract.

JFNK validates its Newton/Krylov options; fixed-point solvers use `iterations`
and optional `tol`, with `relaxation` for damped Picard. Valid unrelated options remain
ignored by a solver that does not consume them. See [solver diagnostics](solver)
for termination semantics and precision-floor qualifications.

## Family options

| Family | Options / assumptions |
|---|---|
| RKC1/RKC2/RKL2 | `s` is a positive integer, clamped to the method's documented minimum. Or supply finite nonnegative `lambda_max` to select stages. Intended for parabolic/autonomous dynamics; stage-time approximation remains documented |
| Newmark | `beta`, `gamma` within the driver's admissible range; position/velocity state required |
| ROS3P | `w='jvp'/'fd'/'linear'`, time derivative options and GMRES controls; `linear` mode carries its documented order qualification |
| ETD2RK | Mandatory linear RHS action, optional nonlinear remainder; `phi_dim`, `phi_tol` |
| EXPRB32 | `w='jvp'/'fd'`, `f_t='fd'/'none'`, Krylov and derivative controls |
| DIRK / coupled RK | `warmStart` is a first-stage StageResult for implicit-first DIRK, or a previous-stage list for coupled RK; an initial guess only |
| BDF | `startup='implicit'/'explicit'`; defaults to implicit on BDF3–5 and explicit on BDF2. Implicit mode uses order-matched L-stable starters |
| Multistep | Fixed `dt`, sufficient history and startup; timestep changes restart history |
| Geometric | Position/velocity fields and force/separability assumptions; optional declared drift fields |

See each family page for full signatures and numerical qualifications. Tensor
callbacks bind their own parameters through `tensor_rhs`; driver options do not
enter their `(t, y)` signature. Particle selection/masking is system/RHS behavior.
