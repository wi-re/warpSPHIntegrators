# Changelog

All notable changes to `warpSPHIntegrators` are documented here. History before
0.6.0 lives in the git log; the per-phase measurement record lives in
[NOTES.md](docs/research/integrators/notes-index.md).

## [Unreleased]

### Changed — stiff startup and coupled scaling (Phase 6)

- BDF3–5 default to order-matched L-stable implicit starters; the explicit DP5
  comparison remains available. BDF2 can opt into TR-BDF2. Starter solve failures
  propagate, undersized BDF histories fail early, and timestep changes clear
  history before use. SBDF's pure-implicit limit delegates to the same BDF path.
- Added opt-in implicit-first DIRK warm guesses, driver-owned preconditioner
  context and a coupled stage-block preconditioner. Measurements show warm guesses
  are not consistently cheaper; automatic warm starting remains disabled.
- Added unregistered three-stage Gauss/Radau collocation prototypes and a recorded
  114-configuration CPU scale gate. Reaction passes; mesh-refined diffusion fails
  the declared work/tolerance bar, so wave 9 remains gated and the registry is 137.
  Corrected stage/order naming and Gauss symmetry explanations.

### Added — reliable runs and failure handling (Phase 5)

- Fixed/adaptive `solve` threads history and safe FSAL reuse, clips final time,
  isolates rejected attempts and provides field tolerances, bounded retries,
  accepted-step callbacks and explicit external transaction hooks.
- Step status retains custom-solver failures and distinguishes Newton precision
  floors from tolerance convergence; the runner requires opt-in for stagnation.
- DP5 sampling/events preserve the numerical grid, with terminal steps recomputed
  at roots. In-memory and weights-only tensor checkpoints retain run configuration
  and history; documented lifecycle, gradient and persistence qualifications.

### Added — tensor ODE usability and discovery (Phase 4)

- `TensorState` and `tensor_rhs` adapt ordinary `(t,y)` tensor ODEs, including
  additive/semilinear parts, to the existing drivers with shape/dtype/device
  checks. Added a complete standalone example and source-validated tensor guide.
- `filter_schemes` returns registered records matching combined catalog criteria;
  structural metadata covers state kind, separability, drift and retained history.
  `getPreferredIntegrator` returns ordinary-ODE records while preserving the
  existing raw-callable `getPreferredScheme` API.
- Validated critical solver/family numerical controls while retaining custom
  option/RHS passthrough. Added an isolated installed-wheel gate and CI job,
  declared development tooling, and documented common/per-family options.

### Removed — unused integrated-field option

- Removed `integrated(..., fluid_only=...)` and its automatic field metadata:
  no drivers or callers consumed it in the checked warpSPHIntegrators, warpSPH
  and warpSPHCore checkouts. Particle masking remains system/RHS behavior.
  `integrated` now accepts only `update_key` positionally; use `default=` for
  defaults. Old positional flag calls fail instead of becoming field defaults.

### Changed — repository organization (Phase 3)

- Consolidated current work in `REPOSITORY_PLAN.md`, added contributor/multicore
  validation instructions and method dispositions, and organized plans/research
  with numbered measurement indexes and compatibility links.
- Moved notebooks into `examples`, benchmark/gallery drivers into `benchmarks`,
  and durable plots into `docs/assets`; preserved coefficient extraction/oracle
  tools, source pins and run evidence before ignoring local scratch output.
- Corrected the historical SplitEuler equivalence: the pinned upstream method
  is Forward Euler on a combined RHS, not Symplectic Euler.
- Added offline repository link/anchor, notebook syntax, disposition inventory
  and promoted-source integrity checks to the documentation workflow.

### Added — executable onboarding and catalog metadata (Phase 2)

- Public `SchemeMetadata` and derived `IntegrationScheme` properties for stored
  stages, companion order, estimate exponent, required/optional RHS accessors,
  stage-count options and dense-output support. The existing 12-field scheme
  tuple layout is preserved. TR-BDF2 and Gauss-Legendre 2 both expose
  `estimate_order=3`, despite propagated orders 2 and 4.
- Generate the complete catalog, family counts and site summary from the runtime
  registry with `scripts/generate_catalog.py`; `--check` rejects stale artifacts.
- Complete CPU oscillator example, repaired README/quick-start imports, and a
  verbatim standalone script in the overview. The source validator executes
  published blocks directly, labels caller-defined sketches and rejects missing
  annotations or drift between included code and its source file.
- Documentation builds, examples and catalog freshness now gate pull requests;
  broken links and anchors fail the site build. Main-branch Pages deployment
  retains its publication path and deployment-only write permissions.
- Reconcile package description, result dataclass documentation, controller
  signatures/estimate exponents and the legacy preferred-scheme selector contract.

### Added — schemes (108 → 137): wave 8 DIRK tableaus

- 29 OrdinaryDiffEq v7.8.1 implicit tableaus: Kvaerno3/4/5,
  KenCarp3/4/47/5/58, six ESDIRK L[2]SA variants through order 6,
  SSPSDIRK2, ARS222/232/343/443, BHR553, Cash4, CFNLIRK3,
  Hairer4/42 and SFSDIRK4–8. DIRK grows from 7 to 36 methods;
  17 additions carry embedded estimates and 20 support lossless reuse.
  KenCarp/ARS entries provide pure implicit tableaus; ARS222/232
  have identical stored implicit maps. SDIRK22 is intentionally omitted
  because the inspected upstream constructor is a trapezoidal stub.
- Row-sum stage nodes correct inconsistent upstream stored nodes. Rounded
  main weights of Cash4 and SFSDIRK5/7/8 have only their last entry adjusted
  to sum to one. Stored stage counts, coefficient provenance, source
  differences and numerical qualifications are recorded in NOTES §3.28.
- Stability gates distinguish 22 L-stable, two A-stable/non-L-stable
  (Kvaerno4, SSPSDIRK2), and five bounded-region (SFSDIRK4–8) additions.
  SSPSDIRK2 is symplectic: two midpoint half-steps, with stiff limit +1;
  it registers `dissipation=False` and has direct geometry tests.

### Fixed — Phase 1 validation gates

- CI now includes drift-field and pRRK tests. Collection rejects missing
  categories and overlapping shard coverage; CI and pytest read the same
  shard inventory. TVD completeness is checked separately from the expensive
  measurements, which are parameterized per scheme for xdist.
- All 45 estimate emitters receive adaptive-loop coverage selected from
  driver metadata. Implicit gradients use the registry and startup metadata;
  all 36 DIRKs check captured RHS-parameter gradients against finite differences.
  ROS3P retains its dedicated frozen-operator gradient checks.
- Extend FSAL expectations and verify endpoint/first-stage relationships.
  Kepler convergence checks distinguish solver noise from tableau defects
  using default-solve coarse grids and strict-solve standard-grid regressions,
  with the existing order tolerance unchanged.
- Refresh the DIRK catalog, overview counts, selection notes, README and
  site landing data for the 137-scheme working tree. These changes are
  unreleased; package metadata remains at 0.6.1.

### Added — schemes (64 → 71): the wave-1 SSPRK tableaus

- Seven plain SSP tableaus from OrdinaryDiffEq v7.8.1 on the generic Butcher
  driver (wave 1 of `docs/plans/scheme-expansion.md`): SSPRK53, SSPRK63, SSPRK73,
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
  driver (wave 2 of `docs/plans/scheme-expansion.md`): SSPRK43 and SSPRK432
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
  v7.8.1 (wave 3 of `docs/plans/scheme-expansion.md`), in a new driver
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
  `docs/plans/scheme-expansion.md`).
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
  of `docs/plans/scheme-expansion.md`), on the generic Butcher driver.
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

### Added — schemes (79 → 97): the wave-6 Tsit5 + LowOrderRK workhorses

- 18 OrdinaryDiffEq v7.8.1 tableaus on the generic Butcher driver (wave 6 of
  `docs/plans/scheme-expansion.md`): the Tsit5 5(4) pair (ODE's recommended default
  for unknown stiffness) and the LowOrderRK workhorses — OwrenZen3 3(2) /
  OwrenZen4 4(3) / OwrenZen5 5(4) (Owren–Zennaro embedded pairs), BS5 5(4)
  (Bogacki–Shampine), Stepanov5 5(4) (7-stage FSAL pair), Alshina2 2(1) /
  Alshina3 / Alshina6 (Alshina–Zaks–Kalitkin), MSRK5 / MSRK6 (minimum-storage
  5th/6th-order, Stepanov arXiv:2202.08443), SIR54 (Kovalnogov–Simos–Tsitouras
  SIR-model pair), Ralston4, RKO65 (Tsitouras's free-first-stage 5th-order
  starter), RKM (Mead–Renaut pseudospectral), and the pseudo-symplectic
  PSRK4p7q6 / PSRK3p6q5 / PSRK3p5q4.
- All one-step explicit and dissipative; the seven embedded pairs are Tsit5 /
  OwrenZen3 / OwrenZen4 / OwrenZen5 / BS5 / Stepanov5 / Alshina2, the other
  ten register fixed-step only. RKM's ODE "estimate", Alshina3's `b2tilde`
  (sums to 5/9) and SIR54's `btilde` (sums to 7.854) are derivative
  estimates, not embedded solutions (NOTES.md §3.26).
- RKO65 is the registry's first **free-first-stage** method: with
  `a[0] = b[0] = 0` no order condition of any order involves `c[0]` (ODE's
  `2/3` is decorative), and with `(b^T a)[0] = 0` the stale reused k0 reaches
  the update only through two-stage paths — first-stage reuse therefore
  measures the full order 5, not 4. `reuse.tableau_reuse_analysis` gained
  the matching gain-3 branch (RKO65 is its first consumer; FSAL methods and
  Midpoint/EPEC unchanged).
- Every tableau was cross-checked against ODE with a one-step run:
  max |diff| = 0.0 for all 18 (the ODE sources give exact fractions and the
  perform_step recurrences reproduce the Butcher rows exactly). Measured: 9
  FSAL tableaus (the six embedded pairs + MSRK5/MSRK6/SIR54, whose last row
  IS the weight vector); SSP coefficients 0 / 0.7813 / 0.6444 / 0 / 0.5237 /
  0 / 1 / 1 / 0 / 0.3226 / 0.1009 / 0 / 0.7209 / 0 / 0.7531 / 0 / 0 / 0;
  real-axis intervals out to −11.41 (PSRK4p7q6).
- `tests/test_tvd.py` pins the SSP boundaries, real-axis intervals and the
  18 verdict entries; `tests/test_embedded.py` covers the seven pairs
  (estimate rate, consistency, FSAL); `tests/test_step_reuse.py` extends the
  FSAL set by nine.
- Deferred (documented, not registered): Anas5(w) and FRK65(omega) —
  parameter-dependent tableaus (the pRRK situation); SplitEuler is the
  registered `symplecticEuler` duplicate.

### Added — schemes (97 → 108): the wave-7 high-order tableaus (Verner / HighOrderRK / Feagin / DP8)

- Eleven high-order OrdinaryDiffEq v7.8.1 tableaus on the generic Butcher
  driver (wave 7 of `docs/plans/scheme-expansion.md`): Verner's optimal pairs
  Vern6 6(5) / Vern7 7(6) / Vern8 8(7) / Vern9 9(8) (*Numer. Algorithms* 53
  (2010) 383-396), TanYam7 7(6) (Tanaka–Muramatsu–Yamashita, *IPSJ* 33.12
  (1992) 1512-1526), RKV76IIa 7(6) (Verner's 2024 "most efficient" 10-stage
  pair), TsitPap8 8(7) (Tsitouras–Papakostas, *SISC* 20.6 (1999), cheap error
  estimation), DP8 (Hairer's 8/5/3 Dormand–Prince adaptation), and Feagin's
  symmetric methods Feagin10 / Feagin12 / Feagin14 (*The Computer Journal*
  26.5 (1983) 351-357) — the registry's highest-order one-step explicit
  methods (order 14).
- Seven embedded pairs (Vern6–Vern9, RKV76IIa, TanYam7, TsitPap8) expose
  `IntegrationResult.error`; DP8 (dual 5th/3rd-order companion — not a valid
  (8,7) pair) and the three Feagin tableaus register **plain**, fixed-step.
  Vern6 is the wave's only FSAL tableau (lossless reuse).
- Cross-checked against the actual ODE v7.8.1 solver on a real (nonlinear,
  non-autonomous) problem: max |Δy| ≤ 8.9e-16 at T = 2 for every member of
  the wave (NOTES.md §3.27).
- **Reuse: the first multi-stage refusal class.** Vern7, RKV76IIa, TanYam7
  and Feagin14 refuse `priorStep` reuse outright: the reuse-modified
  recurrence is a three-dimensional two-step map (state
  `(y^{n-1}, y^n, k_s^{n-1})`) whose stability interval on the imaginary
  axis (|Im(z)| ≲ 0.020 / 0.008 / 0.008 / 0.108) does not cover the standard
  test step, against one-step intervals to 0.352 / 4.435 / 4.435 / 1.157 —
  reuse diverges there instead of converging. `reuse.py` records the
  refusals in `REUSE_IMAGINARY_UNSTABLE` (the single-stage refusal's
  precedent); the remaining wave-7 schemes keep reuse at the reported lower
  order (5 for Vern8 / TsitPap8 / DP8, 6 for Vern9 / Feagin10 / Feagin12 —
  pre-asymptotic on every double-precision grid, banded in
  `tests/test_step_reuse.py`'s `HIGH_ORDER_REUSE`).
- `tests/test_tvd.py` pins the SSP boundaries (0.0 – 0.2371; DP8 the
  wave's largest) and real-axis intervals (Vern8's [-13.425, 0] the
  wave's widest; Feagin14's [-1.874, 0] the narrowest despite its order —
  verified genuine, |R(−2.0)| = 3.51) plus the 11 verdict entries;
  `tests/test_embedded.py` covers the seven pairs on per-pair grids
  (`EMBEDDED_GRID`). The main-order tests move to per-problem coarse grids
  (`conftest.PROBLEM_COARSE_GRIDS`) where the standard grid sits at the
  roundoff floor; Feagin12/14's main order is unmeasurable on `forced` at
  every dt that divides T = 2 (pinned as a skip, `FORCED_FLOOR`).
- Registry grows 97 → 108; the ExplicitRK family grows 52 → 63.

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
  `benchmarks/tvd_classifier.py`), on top of the existing `tvd.py` analysis
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
  (`examples/notebooks/viscous_burgers_demo.ipynb`, `examples/integrations/jfnk_wave_equation.ipynb`).
