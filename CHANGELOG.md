# Changelog

All notable changes to `warpSPHIntegrators` are documented here. History before
0.6.0 lives in the git log; the per-phase measurement record lives in
[NOTES.md](NOTES.md).

## [Unreleased]

### Added — schemes (64 → 71): the wave-1 SSPRK tableaus

- Seven plain SSP tableaus from OrdinaryDiffEq v7.8.1 on the generic Butcher
  driver (wave 1 of `SCHEME_EXPANSION_PLAN.md`): SSPRK53, SSPRK63, SSPRK73,
  SSPRK83 (Ruuth's globally optimized order-3 methods; measured SSP
  coefficients 2.6506 / 3.5184 / 4.2879 / 5.1071), SSPRK54 (order 4;
  1.7693), KYKSSPRK42 (4-stage order 2; 2.2838) and KYK2014DGSSPRK_3S2
  (3-stage order 2; 1.8939) — the last two Kubatko–Yeager–Ketcheson
  DG-optimized methods.
- Coefficients transcribed from ODE's Shu–Osher alphas/betas and stage
  recurrences and converted to Butcher form: row sums equal ODE's stored c to
  1e-16, measured order at or above the registered order for all seven, none
  FSAL (NOTES.md §3.21).
- `tests/test_tvd.py` pins each SSP boundary (scanned to CFL 8) and real-axis
  stability interval; the verdict table gains seven entries (SSPRK83 measures
  unconditionally TVD to the CFL-5 cap).

### Added — schemes (71 → 74): the wave-2 SSPRK 3(2) pairs and SSPRK932

- Three more OrdinaryDiffEq v7.8.1 SSP tableaus on the generic Butcher
  driver (wave 2 of `SCHEME_EXPANSION_PLAN.md`): SSPRK43 and SSPRK432
  (4-stage order 3 with an embedded order-2 estimate — the registry's first
  SSP 3(2) pairs; measured SSP coefficient 2.0, real axis [-5.149, 0] for
  both) and SSPRK932 (9-stage order 3; exact SSP coefficient 6.0, real axis
  [-13.290, 0]).
- All three reproduce ODE's verbatim stage recurrences to ~2.5e-16 under
  exact-rational verification, and ODE's published SSP coefficients match
  the library's all-stage measurements exactly (NOTES.md §3.22).
- SSPRK932 is registered fixed-step only: ODE's adaptive estimate
  `(uprev + 6*u6 + 6*dt*k)/7 - u_final` is `(5/7)*dt*u'(t) + O(dt^2)`
  (measured scaling 1.058), not a valid 3(2) embedded solution; SSPRK43 /
  SSPRK432 provide the SSP 3(2) adaptive capability.
- `tests/test_tvd.py` pins the SSP boundaries and real-axis intervals
  (now 48 tests); `tests/test_embedded.py` and `tests/test_adaptive.py`
  cover the new pairs' estimate scaling and the adaptive loop.

### Added — schemes (74 → 76): the wave-3 SSPRKMSVS multistep methods

- Shu's strong-stability-preserving *multistep* methods from OrdinaryDiffEq
  v7.8.1 (wave 3 of `SCHEME_EXPANSION_PLAN.md`), in a new driver
  `ssprkmsvs.py`: SSPRKMSVS32 (3-step, order 2, SSP C = 1/2) and
  SSPRKMSVS43 (4-step, order 3, SSP C = 1/3). The first non-Adams multistep
  schemes in the registry: the steady-state formula is a convex combination
  of forward-Euler substeps across past states, so the `StepHistory` carries
  a state snapshot plus the step-start derivative (BDF's mechanism).
- One right-hand-side evaluation per step once past the cold start (the past
  derivative is reused from the history). Cold start is the uniform
  Dormand-Prince 5(4) starter, not ODE's 2nd-order Heun self-start: a fixed
  `k-1`-step bootstrap of a `q`-order starter accumulates only `O(dt^{q+1})`
  starting error, so Heun is the minimum for the 3rd-order method and the
  full order is reached with either (NOTES.md §3.23).
- Fixed-step only (no embedded estimate; `result.error is None`), and
  `steps == order` (2 / 3), not `order - 1` as for the Adams family — the
  multistep tests now size their `StepHistory` from `scheme.steps`.
- `tests/test_multistep.py` gains the two schemes' order, one-evaluation
  steady state, fixed-step contract, and dt-restart coverage (now 87 tests);
  the TVD verdict table records `(None, 1.5, False)` for both (the shared
  Dormand-Prince cold-start limit, not the steady-state SSP bound).

### Added — parametric-relaxation pRRK (`prrk.py`, not registered)

- pRRK22 / pRRK33 / pRRK54 (Liu, Burchard, Kuzmin & Shang 2023) —
  parametric-relaxation variants of Heun / SSP RK3 / SSPRK54 with a free
  `κ` parameter that trades the SSP (TVD) constant for a much larger
  negative-real-axis stability interval (wave 4 of
  `SCHEME_EXPANSION_PLAN.md`).
- **Not registered** in `IntegrationSchemes`: the relaxation parameter is
  `z = κ·dt`, so the tableau depends on both `κ` and `dt` (not a fixed
  tableau), and at `κ = 0` the methods duplicate already-registered schemes.
  They are exposed as standalone drivers
  (`from warpSPHIntegrators import pRRK33`); the registered scheme count
  stays 76.
- The one-step map is exactly `u^{n+1} = u^n + dt·Σ bⱼ kⱼ` (the Shu–Osher
  relaxation unfolds to a standard Butcher tableau because every modified
  α-row sums to 1 for all `z`), but `(a, b, c)` is recomputed from the
  caller's `(κ, dt)` on every step and fed to the shared `RungeKuttaB` — a
  `butcherScheme` factory (tableau built once) cannot express it (NOTES.md
  §3.24).
- `κ = 0` reduces exactly to the base method (machine precision); the formal
  order (2 / 3 / 4) is preserved for every `κ`; the real-axis interval is
  bounded at `κ = 0` ([-2, 0] / [-2.51, 0] / [-5.33, 0]) and unbounded
  (A-stable, `R → 1` from below, not L-stable) for `κ ≥ 0.5`.
  `tests/test_prrk.py` (16 tests) pins all of it.

### Added — schemes (76 → 79): the wave-5 Higueras–Roldán low-storage SSPRK53 variants

- SSPRK53_2N1 / SSPRK53_2N2 / SSPRK53_H (Higueras & Roldán, *New third
  order low-storage SSP explicit Runge–Kutta methods*, arXiv:1809.04807) —
  5-stage order-3 low-storage SSP methods from OrdinaryDiffEq v7.8.1 (wave 5
  of `SCHEME_EXPANSION_PLAN.md`), on the generic Butcher driver.
- **Screened before registration** (the plan's gate): all three are
  algebraically *distinct* one-step maps from the already-registered
  SSPRK53 — different stages and final b-row, one-step map differing by
  $O(\mathrm{dt}^4) \approx 1.5\times 10^{-4}$–$4.6\times 10^{-3}$ per unit
  dt⁴ (not ~1e-16) — so they register as their own methods rather than being
  skipped as storage rearrangements (ODE's 1–2 temporary storage saving does
  not apply to this driver, which builds each stage state fresh).
- Coefficients transcribed from ODE's Shu–Osher alphas/betas and converted
  to Butcher form: row sums equal ODE's stored c to 1e-15, all order-3
  conditions hold to ≤5e-15, and the same derivation reproduces the
  registered SSPRK53 tableau to 3e-17 (validating the conversion)
  (NOTES.md §3.25).
- Measured: SSP coefficient 2.1808 / 2.1487 / 2.6506 (ODE publishes
  2.18 / 2.148 / 2.65), real-axis interval [-5.528, 0] / [-7.263, 0] /
  [-6.213, 0]. SSPRK53_2N2 has the largest real-axis interval of the four
  despite the smallest SSP coefficient; SSPRK53_H matches SSPRK53's
  interval to 1e-4 (it shares the first two nodes but has distinct later
  stages); SSPRK53_2N1's stage 3 sits at c₃ = 1.0053 > 1 (evaluated just
  past the step end, as ODE does). None FSAL.
- `tests/test_tvd.py` pins the SSP boundaries and real-axis intervals
  (expanded test, mu_max = 8) and the verdict table gains three entries
  (all `(ssp, 2.0, False)` — the same per-step TVD CFL as SSPRK53).

### Added — drift fields (`drift.py`)

- `integrated(update_key, drift_key=...)` declares a **drift field**: a
  configuration quantity whose rate has a velocity-linear part (the
  weakly-compressible continuity density `drho/dt = -rho div v`), reported on
  the update object under `drift_key`. A system opts in with
  `drift_rates(velocities, aux=None, **kwargs)` (that part at its own
  configuration for a given velocity field) and, optionally,
  `drift_fields_enabled(**kwargs)`.
- The Verlet-family schemes (Symplectic Euler, Velocity Verlet, Leap Frog,
  Semi-Implicit Euler, PEFRL, VEFRL) then advance that part with the **drifts**
  (the velocity the positions move with) and only the rest with the kicks. Cost:
  one velocity-linear evaluation per step (Symplectic Euler, Velocity Verlet,
  Semi-Implicit Euler), two (Leap Frog), six (PEFRL), four (VEFRL). Every other
  scheme ignores the declaration; systems that do not opt in are unchanged.
- Why: with the quantity updated alongside the kicks, every Verlet scheme
  amplifies the linear acoustic pair at every step size — Symplectic Euler
  (explicit midpoint on the pair) by `1 + (omega dt)^4/4` per step, Velocity
  Verlet by `~1 + h^2/2`, Leap Frog / Semi-Implicit Euler / PEFRL / VEFRL by
  `~1 + h^2` — and Velocity Verlet, Leap Frog, PEFRL and VEFRL drop to first
  order for such a field. With the drift path the pair is a symplectic map
  (bounded energy) and the field is second order (first for Semi-Implicit
  Euler; PEFRL/VEFRL reach 2, not 4, for it).
- `testing.DriftSystem`, `testing.acoustic_problem`,
  `testing.drift_oscillator_problem` (`testing.DRIFT_PROBLEMS`) and
  `tests/test_drift_fields.py` pin all of the above, including the legacy
  growth rates.
- Known limit: a drift field's *non*-velocity rate still rides the kicks, and
  PEFRL/VEFRL are first order for that part (as they were for any quantity).

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
