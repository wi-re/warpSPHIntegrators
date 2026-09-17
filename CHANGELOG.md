# Changelog

All notable changes to `warpSPHIntegrators` are documented here. History before
0.6.0 lives in the git log; the per-phase measurement record lives in
[NOTES.md](NOTES.md).

## [0.6.0] — 2026-09-17

The implicit-integration release: 38 new schemes (26 → 64), a Jacobian-free
Newton–Krylov solver, the structured RHS interface, and adaptive-step helpers.
Supported interpreters: Python 3.11–3.13 (`requires-python` is now enforced).

### Added — schemes (26 → 64)

| Family | Schemes |
|---|---|
| BDF | BDF1–BDF5, TR-BDF2 |
| Adams–Moulton (JFNK-corrected) | AM2, AM3, AM4 |
| DIRK | Backward Euler (implicit), Implicit Midpoint, SDIRK2, ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA |
| Coupled fully implicit RK | Gauss-Legendre 2, Radau IIA s=2 (with `BlockState` for multi-component states) |
| Explicit multistep | Adams-Bashforth 2–5, ABM 2/3/4 (PECE), Trapezoidal (Crank-Nicolson) |
| IMEX RK | IMEX Euler, ARK3(2)4L[2]SA, ARK4(3)6L[2]SA |
| IMEX linear multistep | SBDF2, SBDF3, CNAB2 |
| Rosenbrock | ROS3P (Rosenbrock-W) |
| Exponential integrators | ETD2RK, EXPRB32 |
| Relaxed Chebyshev | RKC1, RKC2, RKL2 (super-timestepping) |
| SSP | SSPRK(10,4) |
| Newmark | Newmark (beta/gamma, with average- and linear-acceleration presets) |

### Added — solver

- `JFNKSolver`: Jacobian-free Newton–Krylov nonlinear solver (GMRES inner
  iterations, finite-difference or JVP matvecs, optional backtracking line
  search, resolution-independent stagnation detection).
- Preconditioned GMRES: left/right-preconditioned solves, the
  `preconditioner(v, state, context)` hook, and `identity_preconditioner` /
  `diagonal_preconditioner` helpers.
- `SolveDiagnostics` / `SolveResult` on `IntegrationResult`: residual, GMRES
  iteration counts, RHS evaluations, termination reason, backtracks.

### Added — interface and API

- Structured RHS interface: `RHS` / `IMEXRHS` / `SemilinearRHS` with
  capability-checked resolution — a scheme that needs a split the RHS does not
  declare fails before the solve, naming the missing capability.
- Adaptive-step helpers: `estimate_error_norm`, `propose_dt`, and
  `dormand_prince_dense_output` (multistep schemes refuse adaptation
  explicitly).
- First-stage reuse extended to the stiffly accurate DIRK tableaus (`warmStart=`;
  the explicit `priorStep=` mechanism predates 0.5.0), with the retained order
  registered per scheme (`step_reuse_order`, `is_fsal`).
- Gradients through every implicit scheme (implicit-function-theorem
  re-attachment of the Newton solves).
- Enum QoL pass: per-family enums (`ExplicitRK`, `DIRK`, `BDF`, ...) with
  `FAMILY_ENUMS` / `SCHEME_FAMILY`, alongside the existing
  `IntegrationSchemeType`.
- TVD/SSP classifier over the whole registry (`tvd_analysis.py`,
  `scripts/tvd_classifier.py`), on top of the existing `tvd.py` analysis
  primitives.

### Fixed

- JFNK: Newton no longer stops early — `newton_tol` is decoupled from the
  GMRES tolerance.
- JFNK: removed two redundant GPU→CPU synchronizations in the matvecs.
- Benchmark cost accounting: explicit stage counts and no double-counted
  ROS3P total.

### Changed / infrastructure

- `requires-python = ">=3.11"` with matching 3.11/3.12/3.13 classifiers (the
  CI matrix is the supported floor).
- Test suite categorised (17 `pytest -m` categories) and CI sharded into six
  parallel jobs; push runs Python 3.13 only, pull requests keep the full
  matrix (push wall time ~31 min → ~8 min).
- Scheme wiki moved from Sphinx to a Docusaurus site with KaTeX math at
  `https://fluids.dev/warpSPHIntegrators/`; demo notebooks added
  (`viscous_burgers_demo.ipynb`, `jfnk_wave_equation.ipynb`).
