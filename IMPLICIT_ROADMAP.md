# Implicit and Stiff Integration Roadmap

Status markers: `[x]` complete, `[>]` in progress, `[ ]` planned, `[?]` needs a downstream use case or design decision.

This roadmap covers the remaining implicit, stiff, IMEX, stability, RHS-interface, and nonlinear-validation work.

Phases 6 (coupled fully implicit RK) and 7 (Rosenbrock-W / exponential) were de-gated 2026-09-10: they were previously "needs a downstream", but Phase 6 fills capability gaps a general ODE library has on its own (only symplectic method above order 2; only no-compromise stiff method), and Phase 7's missing consumer — a linear/nonlinear RHS split — is now scoped as Phase 14, the structured RHS interface.

Adaptive timestep control *was* explicitly out of scope, on the grounds that it belongs to a problem-specific driving loop and only applies where a meaningful local error estimator is available. The second half of that reasoning has since expired: ten registered schemes now carry embedded estimators (the original "eight" predates ROS3P and EXPRB32) and `IntegrationResult.error` went from having no consumer at all to being consumed by Phase 11's `estimate_error_norm` / `propose_dt` helpers (landed 2026-09-14). The `StepHistory`-invalidation problem that was its actual blocker was resolved by decision: multistep is refused explicitly (NOTES §3.17).

## Current Baseline

- [x] Nonlinear-solver protocol with fixed Picard, relaxed Picard, and matrix-free JFNK.
- [x] JFNK is the default nonlinear closure for DIRK, Newmark, BDF, implicit Adams-Moulton, IMEX Euler, ARK, the coupled block RK pair (Gauss-Legendre 2, Radau IIA s=2), and IMEX multistep (SBDF2/SBDF3/CNAB2).
- [x] DIRK: Backward Euler, implicit midpoint, trapezoidal / Crank-Nicolson, SDIRK2, TR-BDF2, ESDIRK3(2)4L[2]SA, and ESDIRK4(3)6L[2]SA.
- [x] Newmark average-acceleration and linear-acceleration variants.
- [x] BDF1-BDF5 with state-bearing `StepHistory` and a safe Dormand-Prince cold start.
- [x] Fully implicit (iterated) Adams-Moulton AM2-AM4 with derivative-bearing `StepHistory` and an optional Adams-Bashforth predictor.
- [x] IMEX Euler with explicit `IMEXRHS(explicit=..., implicit=...)` callbacks; an ordinary RHS remains fully implicit.
- [x] Dahlquist stability-region plots and numerical boundary checks for tableau methods and BDF1-BDF5.
- [x] Damped-oscillator amplification matrices (closed form, cross-checked against the registered schemes' one-step maps) and damping-ratio stability gallery for the Verlet family and Newmark.
- [x] Stiff/nonlinear benchmark problem registry in `testing.py` (Prothero-Robinson in both signs, stiff damped oscillator, van der Pol, Robertson kinetics, semi-discrete diffusion) plus the error/cost benchmark figure `images/stiff_benchmark_suite.png`.
- [x] Nonlinear Kepler convergence coverage and nonlinear stiff relaxation coverage.
- [x] Gradient-through-step coverage for every registered scheme, implicit ones differentiated by the implicit function theorem rather than by unrolling the Newton/GMRES iteration (`tests/test_gradients.py`, Phase 9).
- [x] First-stage reuse for the stiffly accurate DIRK tableaus with an explicit first stage — Trapezoidal, TR-BDF2, ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA — lossless, saving one (cheap, explicit) RHS evaluation per step (`tests/test_dirk.py`, Phase 10).
- [x] `viscous_burgers_demo.ipynb` — the Phase 14 viscous-Burgers semilinear benchmark (n=64, ν=0.01, T=0.4) run through every registered family (explicit / JFNK-closed implicit / IMEX) with space–time shock-formation plots (x horizontal, t vertical), measured convergence orders against an RK4 (dt=2e-3) semi-discrete reference, and a Phase 1-diagnostics cost table (RHS evaluations + GMRES iterations per run).
- [x] Adaptive step control as helpers — `estimate_error_norm` / `propose_dt` (predictive controller, safety 0.9, clamp [0.2, 5.0]) with the caller-driven accept/reject loop, and DP5 dense output (Shampine 1986's quartic continuous extension, exact at the step endpoints, `O(dt^5)` interior); multistep refused explicitly (`StepHistory` restarts on any `dt` change) (`tests/test_adaptive.py`, `images/adaptive_benchmark.png`, NOTES §3.17, Phase 11).

## Phase 0: Keep the Public Story Accurate

- [x] Update `README.md` Known Limitations to say BDF1/2 and IMEX Euler are available.
- [x] Update `NOTES.md` status and scheme tables to replace old Picard-default claims with JFNK-default behavior.
- [x] Document the JFNK cost model: residual evaluations, GMRES matvecs, finite-difference versus exact JVP, and the need for preconditioning at scale.
- [x] Document which methods are A-stable, L-stable, symplectic, or only conditionally stable; distinguish the exact scheme from a truncated Picard override.
- [x] Add a concise supported-scheme table with conservative dynamics, stiff dissipation, split stiff terms, and long smooth stiff integrations.

Validation gate:

- [x] README examples execute in the `warp` environment.
- [x] Scheme names, registry metadata, and documentation tables agree.

## Phase 1: Solver Observability and Robustness

- [x] Extend `SolveResult` with optional diagnostics: nonlinear residual, GMRES iterations, total RHS evaluations, and termination reason.
- [x] Define termination reasons: converged tolerance, stagnation floor, maximum Newton iterations, maximum GMRES iterations, invalid residual.
- [x] Add finite-value checks for nonlinear residuals before accepting a Newton update.
- [x] Add optional line search / damping for JFNK Newton corrections when a full correction increases the nonlinear residual.
- [x] Add a reusable solver-options dataclass or typed dictionary, while preserving current keyword compatibility.
- [x] Record diagnostics in `IntegrationResult` without changing existing `stages` semantics.
- [x] Add tests for Newton convergence, stagnation, GMRES exhaustion, NaN/Inf detection, and line-search recovery.

Validation gate:

- [x] Existing JFNK stiff oscillator remains correct.
- [x] Failure paths report structured diagnostics instead of silently returning a poor iterate.
- [x] Full test suite remains green.

## Phase 2: Preconditioned JFNK

- [x] Add an optional `preconditioner(v, state, context) -> vector` hook to JFNK. (JFNKSolver binds the 3-arg callable to each Newton iterate: `state` is the current iterate, `context` is `{'state': Y, **preconditioner_context}`; settable at construction and via solve `**opts`.)
- [x] Upgrade GMRES to apply right or left preconditioning consistently and expose the selected mode. (`gmres(..., preconditioner=, preconditioning='right'|'left')`; right solves `A M z = b` returning `x = M z` with the true residual, left solves `M A x = M b`; unknown modes rejected.)
- [x] Provide an identity preconditioner baseline and preserve current behavior when none is supplied. (`identity_preconditioner`; a supplied identity is *bitwise* identical to no preconditioner in both modes, verified in `tests/test_preconditioner.py`.)
- [x] Add diagonal / block-diagonal examples suitable for diffusion and relaxation terms. (`diagonal_preconditioner`: fixed tensor or state-dependent callable — the `1/(1 + dt*damping)` relaxation shape.)
- [x] Add a sparse or operator-based SPH preconditioner integration point; do not form dense Jacobians. (The hook + `preconditioner_context` is the integration point; downstream `warpSPH/tests/test_implicitWaveEquation.py` exercises it with the block-lower-triangular factor of the backward-Euler wave stage Jacobian, applied through `warpOperationJVP` — one Laplacian apply, no dense matrix.)
- [x] Measure residual evaluations and Krylov iterations as problem size grows. (`scripts/jfnk_preconditioner_benchmark.py`, n = 64..1024; Krylov cost is flat at 9-13 iterations preconditioned vs 104-173 unpreconditioned, 9-12x in total stage-map evaluations.)
- [x] Add benchmark plots comparing unpreconditioned and preconditioned JFNK. (`images/jfnk_preconditioner_benchmark.png`.)

Validation gate:

- [x] Preconditioned and unpreconditioned solutions agree on linear reference problems. (Library-level variable-coefficient stiff relaxation, and the wave equation against the hand-eliminated CG reference.)
- [x] A representative stiff diffusion/relaxation problem uses materially fewer GMRES iterations with a supplied preconditioner. (n=200: 76 -> 32 right / 6 left; wave stage at nx=32: 41 -> 17 fd, 33 -> 7 jvp.)
- [x] The no-preconditioner path remains bitwise or tolerance-equivalent to the current solver. (Identity preconditioner bitwise-identical in both modes; `tests/test_jfnk.py` passes unchanged.)

## Phase 3: Higher-Quality Sequential DIRK

### TR-BDF2

- [x] Verify the TR-BDF2 SDIRK coefficients and independently check the second-order condition.
- [x] Add the three-stage tableau to `getDIRKTableau`.
- [x] Add an embedded estimator, returning `IntegrationResult.error`.
- [x] Register order 2, L-stable, stiffly accurate metadata.
- [x] Test convergence through the generic oscillator, forced, damped, and Kepler suites plus nonlinear stiff relaxation.
- [x] Add a method-specific L-stable damping test on the negative-real-axis scalar problem.

### ESDIRK3(2)4L[2]SA and ESDIRK4(3)6L[2]SA

- [x] Verify coefficients and named variants from Kennedy-Carpenter or the original source. (Sourced from SUNDIALS ARKODE v7.9.0, `src/arkode/arkode_butcher_dirk.def`, entries `ARKODE_ESDIRK324L2SA` / `ARKODE_ESDIRK436L2SA`; verified by symbolic Taylor-model order, numeric local-error rates on five closed-form ODEs, and exact stability-function sweeps — NOTES.md §3.6.)
- [x] Extend the tableau model only if needed for explicit first stages and embedded weights. (Not needed: the DIRK driver already handles `a_ii == 0` explicit stages and tuple `b` embedded pairs.)
- [x] Register both schemes with accurate stage count, stability, stiff-accuracy, and error-estimator metadata.
- [x] Add convergence and embedded-error tests on autonomous, non-autonomous, nonlinear, and stiff problems.
- [x] Add stability-region overlays for all new DIRK methods. (Numeric known-point containment and |R(-100)| L-damping assertions in `tests/test_stiff.py`, same convention as TR-BDF2; no new PNGs.)

Validation gate:

- [x] Every new tableau passes hand-checked consistency/order conditions and empirical convergence tests.
- [x] JFNK preserves copied-field lifecycle through every implicit stage.
- [x] Stability plots match the advertised A/L-stability class numerically. (A-stable on fine left-half-plane and imaginary-axis sweeps, max |R| = 1 only at z = 0; L-decay |R(-100)| = 2.65e-2 / 7.57e-2.)

## Phase 4: Higher-Order BDF and True Adams-Moulton

### BDF3-BDF5

- [x] Generalize `bdf.py` to coefficient tables for orders 1-5. BDF4/5 coefficients derived from the exact BDF order conditions and verified empirically on the non-autonomous `forced` problem (the standard formula evaluates `f` only at the new grid time, so it keeps its order there — measured 4th/5th order).
- [x] Extend `StepHistory` requirements and preserve state snapshots for all required previous states. (The history machinery already stored one state snapshot per entry; BDF4/5 need `maxlen` 3/4, driven by the scheme's `steps` metadata — no `history.py` change.)
- [x] Keep the safe high-order starter when history is insufficient or invalidated by timestep / particle identity changes. (Unchanged Dormand-Prince cold start; the `StepHistory.pushed` dt/uid invalidation guards are shared with BDF1-3 and now exercised at scheme level by `test_bdf4_history_reset_on_dt_change_stays_correct`.)
- [x] Register BDF3-BDF5 with correct A(alpha) metadata, not A-stable metadata. (BDF4 `stability='A(alpha)'` 73.35 deg, BDF5 51.84 deg — cone half-angles measured from the stability-region boundary curve, the same machinery that reproduces the literature 86.03 deg for BDF3.)
- [x] Add root-condition and numerical stability-boundary tests. (`tests/test_bdf.py`: Dahlquist root condition for orders 4/5, negative-real-axis retention to z = -1000, non-A-stability point, and a 55-deg cone point that separates BDF4 (inside) from BDF5 (outside).)
- [x] Test order on oscillator, forced, damped, Kepler, and nonlinear stiff relaxation. (Order runs use the exact JVP matvec + tight tolerance so the linear-solver noise floor cannot mask the finest-grid truncation error; stiff relaxation in `tests/test_stiff.py` at a rate the explicit startup can absorb — see the test docstring for why rate=100 is out of reach for BDF3-5.)

### True implicit Adams-Moulton 2-4

- [x] Add a JFNK residual using known history derivatives plus the unknown endpoint derivative. (`multistep.AdamsMoulton`: fixed point `y = known_part + gamma[0]*dt*f(t^{n+1}, y)`; history stores the `update` of each past step, `needed = order - 1`.)
- [x] Keep existing ABM PECE methods separate: predictor-corrector and fully implicit Adams-Moulton are different contracts. (ABM2-4 unchanged; the implicit family is registered under distinct names/identifiers `Adams-Moulton N (implicit)` / `amN`.)
- [x] Add an optional predictor using the matching Adams-Bashforth formula. (`predictor=True` default; `predictor=False` starts from the known part — `test_predictor_only_changes_the_initial_guess` pins both on the same fixed point.)
- [x] Register and test AM2-AM4 with explicit history requirements and startup behavior. (AM2 `stability='A'` — it is the trapezoidal rule; AM3/4 have only a bounded region, `stability=None`; Dormand-Prince cold start until history is full, bit-for-bit pinned in `tests/test_am.py`.)

Validation gate:

- [x] BDF3-BDF5 achieve their claimed order with threaded history. (Measured 3.98-4.97 across the four reference problems, `tests/test_bdf.py`.)
- [x] History resets safely on `dt` or particle-set changes. (Scheme-level restart tests for BDF4 and AM3; the shared `StepHistory` dt/uid guards were already covered by the Phase 0 state tests.)
- [x] AM methods converge their nonlinear corrector and outperform or match PECE on a stiff nonlinear case. (Prothero-Robinson rate=10, dt=0.1: corrector residuals reach ~1e-7; AM2/3/4 errors 6.0e-5 / 7.6e-5 / 2.3e-5 vs PECE 8.3e-4 / 4.0e-4 / 3.1e-3 — `tests/test_am.py::test_am_corrector_beats_pece_on_stiff_nonlinear_case`.)

### Deferred (gated)

- [?] **BDF6.** The A(α) cone shrinks below BDF5's 51.84°, so order 6 buys accuracy
  only while losing the stiff sector. Trigger: a downstream stiff problem that needs
  order ≥ 6 *and* whose stiff modes stay inside the narrower cone.
- [?] **BDF3-5 cold-start ceiling.** The Dormand-Prince bootstrap is explicit with a
  negative-real-axis boundary at |z| ≈ 3.3, so BDF3-5 cannot *start* at
  `dt·rate ≳ 3.3` on the stiff relaxation (the rate=100 case is out of reach for
  them — see the `tests/test_stiff.py` docstring). Options: document the per-order
  usable-rate ceiling, or add a JFNK-based startup (e.g. a BE or AM2 warm-up).
  Trigger: a downstream stiff problem where the ceiling binds.

## Phase 5: Higher-Order IMEX / Additive RK

- [x] Define an additive-tableau representation with explicit and diagonally-implicit coefficients, stage times, propagated weights, and embedded weights. (`ark.AdditiveTableau`)
- [x] Generalize IMEX Euler into an additive RK driver using `IMEXRHS`. (`ark.ARK`; reuses the existing `IMEXRHS` split — no protocol change.)
- [x] Preserve the compatibility rule: a normal RHS callable means fully implicit; only `IMEXRHS` activates a split.
- [x] Decide and document copied-field semantics when explicit and implicit callbacks each preprocess a stage. (Implicit callback owns the stage buffer; explicit runs on a throwaway clone; final copied fields come from the implicit buffer — `ark.py` docstring + `tests/test_ark.py`.)
- [x] Implement and verify ARK3(2)4L[2]SA. (SUNDIALS ARKODE v7.9.0 ERK+DIRK pair; order 3, L[2].)
- [x] Implement and verify ARK4(3)6L[2]SA. (Same source; order 4, L[2]. The implicit half is a different ESDIRK design than the standalone ESDIRK4(3)6.)
- [x] Add split test problems: explicit transport plus implicit linear decay/diffusion; nonlinear explicit forcing plus implicit relaxation. (`tests/test_ark.py`)
- [x] Add two-parameter IMEX stability plots over `(z_explicit, z_implicit)` slices. (`stability.imex_stability_function` + `scripts/stability_gallery.py` → `images/imex_stability_slices.png`; numeric assertions in `tests/test_ark.py`.)
- [x] Measure RHS cost: explicit evaluations, implicit residual evaluations, and GMRES iterations. (Per-stage `solver_diagnostics`; explicit-eval-per-stage and residual-count tests in `tests/test_ark.py`.)

Validation gate:

- [x] Pure-explicit and pure-implicit limits recover the corresponding component methods where the tableau guarantees it. (Pure-implicit limit reaches order 3/4 on oscillator/forced/damped; pure-explicit limit's stability function matches the ERK half exactly.)
- [x] Split linear test equations meet claimed order and remain stable in their published IMEX region. (Both split problems measure 3/4; L-damping and mixed-region stability asserted.)
- [x] Existing ordinary RHS behavior remains fully implicit and cannot silently discard a split term. (Ordinary RHS → implicit half alone; the split only activates via `IMEXRHS`.)

## Phase 6: Coupled Fully Implicit RK

**Status: done — landed 2026-09-15.** `fullyimplicit.py` (the coupled block
driver) + `fields.BlockState` + both tableaus registered: **Gauss-Legendre 2**
(order 4, A-stable, symplectic for separable Hamiltonians — pinned by
measurement, the tableau is *not* A-symmetric — not L-stable) and **Radau
IIA s=2** (order 3, L-stable, stiffly accurate), each with a null-stage
order-2 companion, the matrix-free JVP block matvec, the `warmStart=`
initial-guess flag, and the 256-DOF scale gate. NOTES §3.18 has the
coefficients, the measurement tables, and the GMRES-spectrum finding that
shaped the gate. The original gate was "a downstream that needs order-4
symplectic accuracy or L-stable Radau damping." It was lifted 2026-09-10 on
general-ODE-library grounds: Gauss-Legendre s=2 is the *only* route in the
library to a symplectic method above order 2 (the registered Verlet /
Forest-Ruth set is order 2, and NOTES §3.6 measures even those collapsing to
order 1 under a velocity-dependent force), and Radau IIA is the standard
no-compromise stiff method — L-stable, stiffly accurate, no A(α)-cone
restriction, which is exactly what BDF3-5 give up. Both fill capability gaps
that stand on their own.

**Technical dependency — a product-type state, not a raw vector.** The coupled
`s·N`-unknown solve needs `s` stage states presented to the solver at once. Add a
`BlockState` holding `s` sub-states and teach the four state↔vector bridge functions
in `fields.py` (`flatten_integrated`, `unflatten_integrated`, `integrated_field_names`,
`replace_integrated_fields` / `_maybe_reference_state`) to handle it. Each sub-state
stays a full tagged state, so `copied` fields, field tags and `apply_*_update` still
work per stage; the stacking is only at the `flatten_integrated` boundary the DIRK
solver already crosses. With that in place the coupled block `step` runs through
`JFNKSolver.solve` unchanged and inherits Phase 1 diagnostics, the Phase 2
preconditioner hook, and the Phase 9 implicit-function-theorem gradient path with no
new derivation.

- [x] Add `BlockState` and extend the four `fields.py` bridge functions over `s · N`
  unknowns; each sub-state keeps its full field structure (substate-major `'i:name'`
  layout).
- [x] Build the coupled stage residual `G(Y_1..Y_s)_i = Y_i − y^n − dt·Σ_j a_ij f(t_j, Y_j)`
  as a `step` fixed point, solved by `JFNKSolver` (one solve per step, `step_fn` =
  `s` RHS evaluations; no dense block Jacobian ever formed).
- [x] Implement Gauss-Legendre s=2 (order 4) first.
- [x] Validate direct symplectic-form behavior on oscillator and Kepler after tightening the default JFNK residual and finite-difference settings.
- [x] Implement Radau IIA s=2 (order 3, L-stable) as the first high-quality fully implicit stiff method.
- [x] Embedded error estimator — **correction to the plan**: the "standard
  Hairer-Wanner estimator" is only the 2-stage pair, and that pair is *degenerate*
  (it is `b` itself) for both shipped tableaus, so both ship the null-stage
  min-norm order-2 companion over `(k0, k1, k2)` (`companion_b[0] != 0` costs one
  extra RHS eval per step, the same cost the companion pairs already pay).
  Measured rate exactly 3.0 for both.
- [x] Exact-JVP block matvec (forward-AD seeding across the `s` sub-states via
  `replace_integrated_fields`) — landed as the *default* `matvec='jvp'` path,
  cross-checked against the FD path at 5.3e-9 relative (the FD matvec is kept as
  the `matvec='fd'` fallback).
- [x] Gradient coverage: the block stage solve reattaches through
  `_implicit_diff_reattach` once `flatten_integrated` / `unflatten_integrated` accept
  a `BlockState`; both schemes added to the `tests/test_gradients.py` cases.
- [?] Consider Gauss-Legendre s=3 and Radau IIA s=3 only after the block solver and preconditioner are proven at scale.
- [?] Consider Lobatto IIIA-IIIB only for a concrete partitioned/separable Hamiltonian downstream (it needs the component partition, not this additive block solve).

Validation gate:

- [x] Block residual and Jacobian-vector products agree with finite differences on small systems. (JVP == FD at 5.3e-9 relative on the 256-DOF advection block, JVP linear to 5.7e-14, and the JVP matches the hand-computed dense block Jacobian: `tests/test_fullyimplicit.py`.)
- [x] Gauss-Legendre order 4 and Radau order 3 are demonstrated on nonlinear reference problems. (Measured 3.98 / 3.00 on the oscillator order ladder, `scripts/blockrk_benchmark.py`; kepler + forced oscillator in `test_driver_order_on_nonlinear_problems`.)
- [x] Gauss-Legendre's symplectic-form / energy behaviour holds on oscillator and Kepler where the order-2 Verlet set drifts. (`tests/test_hamiltonian.py`: area defect 3.05e-13 under the strict solver, kepler-form 4.87e-09 at default settings, against Velocity Verlet's flat 1.00e-2 drift; the GL2-kepler default-solve energy test is documented-skipped — the drift there is JFNK stagnation noise, 6.9e-9 → 5.5e-8 in T, growth 8.0 above the 1e-8 floor, and collapses to 4.3e-12 under the strict solver.)
- [x] Radau IIA damps the negative-real-axis scalar problem where BDF3-5's A(α) cone does not. (`test_stiff.py` STATE_SPACE_IMPLICIT at z = -10, `|R(-100)| ≈ 0.019` pinned; benchmark panel 2 shows `|R(-10)| = 0.0959` vs BDF3's 0.349 and BDF5's 0.655.)
- [x] No dense Jacobian allocation occurs for large state sizes. (Matrix-free gate: the scale test runs the full 2-stage block at 256 DOF = 1536 unknowns with the JVP matvec only; the dense 1536x1536 Jacobian is built only in the one diagnostic that prints its spectrum.)
- [x] Gradients flow through both schemes and match a closed form on the linear oscillator (the Phase 9 bar). (Both schemes in `tests/test_gradients.py::IMPLICIT_SCHEMES`, JVP matvec, IFT re-attachment.)

## Phase 7: Alternative Stiff Families

**Status: done — all three methods landed: Rosenbrock-W (ROS3P) 2026-09-11,
ETD2RK (exponential, order 2) 2026-09-12, and EXPRB32 (exponential Rosenbrock,
order 3) 2026-09-13, which closes the family's cost gate: at fixed error
(rel L2 ≤ 1e-3, sine IC) EXPRB32 is the cheapest order-3 method on the work-unit
metric (640 < ROS3P 706 < ESDIRK3 715 < ARK3 793, `scripts/exponential_benchmark.py`
gate table, NOTES S3.16).** Phase 14's structured `RHS` interface (landed
2026-09-10) unblocked this phase. The original gate was "a downstream RHS that
splits into a linear stiff operator plus a mild nonlinear remainder." Phase 14's
structured `RHS` interface *is* that split, expressed as a first-class problem
description (`linear` / `nonlinear` accessors), and the viscous Burgers test
problem gives both families the standard semilinear benchmark they are
conventionally demonstrated on. Rosenbrock-W (ROS3P) is landed, and the
exponential sub-phase is complete: the matrix-free `phi_k` build (the largest new
piece), ETD2RK (order 2), and EXPRB32 (order 3, the family's cost-gate carrier —
full frozen Jacobian, no semilinear split needed) are all landed.

The efficiency argument is unchanged and still governs the validation gate:
Rosenbrock-W trades the nonlinear solve for one *linear* solve per stage, so it only
*wins* where that linear solve is cheap (operator-based, preconditioned via the Phase
2 `preconditioner(v, state, context)` hook) and the nonlinearity is weak. The bar any
candidate must clear: Phase 8's measured cost baseline (BE 3.40, BDF2 2.96, TR-BDF2
4.50, ESDIRK6 10.72 RHS evaluations/step; GMRES iterations/step equal to the
stage-solve count, at GMRES tolerance 1e-8).

`viscous_burgers_demo.ipynb` already carries that comparison on the benchmark
itself: §9 tabulates per-run RHS evaluations and GMRES iterations for every
registered stiff family at a fixed `dt` past the explicit wall (n=64, ν=0.01,
T=0.4), and §8 measures each family's convergence order against the RK4
(dt=2e-3) semi-discrete reference — the scaffold the gate's "to a fixed error"
comparison plugs into.

### Rosenbrock / W methods

- [x] A dedicated `rosenbrock.py` driver with its own coefficient structure
  (`alpha_ij`, `gamma_ij`, `gamma`, `b_i`) — not a DIRK tableau. Each stage is one
  `gmres` solve against `I/(dt·gamma) − W`, with no outer Newton loop.
  (Landed 2026-09-11: `integrateROS3P`; the three stage states collapse to two
  RHS points and stage 3 reuses stage 2's `StageResult`.)
- [x] `W` is one linearization held fixed across a step's stages, selectable as the
  exact Jacobian action (`jfnk.jvp_matvec` at `y^n`), a finite-difference action, or
  the Phase 14 `linear` accessor when the RHS supplies one (the "W": an inexact `J`
  is admissible without order loss).
  (Landed as `w='jvp'`/`'fd'`/`'linear'`. Correction to "without order loss": the
  `linear` accessor drops the nonlinear Jacobian `J_N`, so it is order 3 only for a
  linear RHS and **first order** on a genuinely semilinear one — documented in
  NOTES S3.14.)
- [x] Start with a verified 2nd- or 3rd-order Rosenbrock-W method with an embedded
  estimator (ROS3P / RODAS3 / ROS34PW2), coefficients from a citable source, order
  checked with the existing Taylor-model machinery.
  (Landed: ROS3P, order 3, from Lang & Verwer, *BIT* 41(4) 2001, with the
  embedded (3, 2) estimator `tau (K1-K2)/3`; measured order 3 on viscous Burgers
  for `w='jvp'` and `w='fd'`.)
- [x] Non-autonomous `∂f/∂t` term: finite-difference it or require autonomous;
  document which.
  (Landed: `f_t='fd'` default — one-sided FD at a `dt`-scaled step, reusing the
  stage-1 evaluation; `f_t='none'` for autonomous. NOTES S3.14.)
- [x] Gradient parity: per-stage adjoint via the `gmres` transpose solve, reusing the
  Phase 9 pattern (`_implicit_diff_reattach`'s `jacobian_transpose`).
  (Landed: frozen-`W` re-attachment, three transposed `gmres` solves plus the `W^T`
  VJP; exact on a linear RHS, a few percent on nonlinear — the `dW/dy` Hessian is
  dropped. NOTES S3.14.)

### Exponential integrators

- [x] Matrix-free `phi_k(hL)v` via a Krylov approximation (Arnoldi on `L`, then
  `phi_k` of the small dense Hessenberg matrix) — shares the Arnoldi core with
  `gmres`, consumes the Phase 14 `linear` accessor, forms no dense matrix.
  (Landed 2026-09-12 in `exponential.py`: `krylov_phi(matvec, v, k, m=, tol=)` —
  one Arnoldi build per distinct right-hand-side vector, `phi_k(H) e1` by a short
  Taylor-vector series (no matrix exponential / eigendecomposition); the
  `L`-matvec count is reported as the step's Krylov-iteration count.)
- [x] Start with ETD2RK / exponential Rosenbrock (exprb32) on semi-discrete viscous
  Burgers (Phase 14's test problem).
  (Landed 2026-09-12: **ETD2RK** — the base unsplit 2-stage, order-2, L-stable
  exponential integrator (Sarumi, arXiv:2601.06849; `phi` per Caliari & Ostermann
  2009). It integrates `L` exactly through `exp(hL)` / `phi_1` / `phi_2` (matrix-
  free, Krylov) and quadratures `N` (two evaluations per step). Measured **order 2**
  on viscous Burgers; **exact** on a purely linear problem (`N = 0`, error at the
  Krylov tolerance, not `O(h^2)`); L-stable (a stiff linear mode is damped by
  `exp(-rate·h)`, killed, not merely damped as in the Rosenbrock case).)
  (Landed 2026-09-13: **EXPRB32** — the order-3 exponential Rosenbrock method
  (Hochbrueck, Ostermann & Schweitzer, *SIAM J. Numer. Anal.* 47(1) 2009 786–803,
  `literature/080717717.pdf`), frozen-`Jn` reformulation with the embedded order-2
  estimate `û = U2`. Freezes the **full** Jacobian (forward-mode AD, `w='jvp'`/`'fd'`;
  deliberately no `w='linear'` — order 1 on a nonlinear problem) and applies every
  `phi_k(h·Jn)` matrix-free through the same `krylov_phi` machinery, so it runs on
  plain callables (no semilinear split needed). Measured **order 3** on viscous
  Burgers (`[2.96, 2.99, 2.99, 2.99]`); exact on linear RHS; L-stable; gradient
  O(dt²) on semilinear (frozen-`Jn`, one power better than the Rosenbrock frozen-`W`
  O(dt)). Clears the family's cost gate below.)
- [x] Gradient path through the `phi_k` Krylov approximation.
  (Landed 2026-09-12: because `L` is *constant* (independent of `y`), the three
  matrix-function actions are constant linear maps, each re-attached with its
  transposed operator as the adjoint (transpose `phi_k(hL)` actions via the same
  Krylov machinery on `L^T`) — no unrolled Krylov and, unlike the Rosenbrock
  frozen-`W` (which drops a `dW/dy` Hessian), **no structural approximation**, so
  the gradient is exact up to the Krylov tolerance. Scope note: the transpose
  reuses the forward `linear` accessor because the benchmark `L = nu·u_xx` is
  **self-adjoint**; a non-self-adjoint `L` would need `L^T` (a future extension).
  NOTES S3.15.)

Validation gate:

- [x] Rosenbrock-W reproduces the semi-discrete viscous Burgers reference to its
  advertised order at fixed `nu`. (Landed 2026-09-11: order 3 measured on viscous
  Burgers for `w='jvp'` and `w='fd'`; `w='linear'` is order 1 on this semilinear
  problem by design — it drops `J_N`.)
- [x] Rosenbrock-W beats the same-order JFNK DIRK / IMEX cost on viscous Burgers
  (total work-units to a fixed error) before its method set expands. (Landed
  2026-09-11: ROS3P (`w='jvp'`) is the cheapest order-3 total work-unit cost on the
  notebook's work-unit metric (706 < ARK3 793 < ESDIRK3 988) and beats ESDIRK3 on
  GMRES iterations (586 < 688). Caveat: the additive ARK3 split's GMRES iterations
  are cheap linear-operator applications, so it is cheaper per iteration in FLOPs.
  NOTES S3.14.)
- [x] Exponential integrators reproduce the reference to their advertised order.
  (Landed 2026-09-12: ETD2RK measured **order 2** on viscous Burgers — the order
  part of the gate. The *cost* half is assessed separately below, at exprb32,
  before the exponential method set expands.)
- [x] ... and beat the same-order cost, before the exponential method set expands.
  (Landed 2026-09-13 with **EXPRB32**: at fixed error — the largest `dt` reaching
  rel L2 ≤ 1e-3 on the sine-IC benchmark — EXPRB32 (`w='jvp'`, registered defaults)
  is the cheapest order-3 method on the work-unit metric: **640** (10 steps at
  `dt = 0.04`: 40 full-`f` evals + 600 Krylov iters) < ROS3P (jvp) **706** < ESDIRK3
  **715** < ARK3 **793**, and it lands the smallest error of the four (1.2e-4).
  The metric is fair in cost class: EXPRB32's Krylov matvecs are **full Jacobian
  JVPs** (forward-mode AD sweeps through the full RHS) — the same class as a full
  `f` evaluation and as ROS3P's frozen-Jacobian GMRES sweeps — in contrast to
  ETD2RK's cheap `nu·u_xx` `L`-matvecs, which is why the order-2 comparison was
  deferred (ETD2RK's 1782 work-units at `dt = 0.02` overstate its FLOP cost;
  NOTES S3.15). Caveats: the 9% margin over ROS3P is a same-class work-unit win,
  not a FLOP blowout, and at the *same* `dt = 0.02` EXPRB32 is the most expensive
  stiff method (1280) — the gate is the fixed-error one, and EXPRB32's ~13x smaller
  order-3 error constant is what turns that around. NOTES S3.16.)

## Phase 8: Stability and Nonlinear Benchmark Suite

- [x] Dahlquist stability regions for implemented tableau and BDF methods.
- [x] Add automatic numerical assertions for known points in every plotted one-step tableau and BDF1/2 region, not only image generation.
- [x] Add second-order undamped-oscillator amplification-matrix plots for Newmark and the Verlet family over `dt * omega`.
- [x] Extend the oscillator stability plots over damping ratio. (Landed 2026-09-09: closed-form damped amplification matrices in `stability.py`, verified against the registered schemes' one-step maps; `log10(rho)` heatmaps over `(h*omega, zeta)` in `scripts/oscillator_stability_gallery.py`.)
- [x] Add direct finite-difference phase-area checks across every registered scheme and nonlinear Kepler symplectic-form checks for the genuinely symplectic subset. (The Phase 6 symplectic-form validation — Gauss-Legendre 2's strict-solve area defect 3.05e-13 and kepler-form 4.87e-09, and Radau IIA s=2's 2.44e-2 non-symplectic defect — is part of this record; NOTES §3.18.)
- [x] Add two-parameter IMEX stability slices for IMEX Euler and each later ARK method. (Done in Phase 5, `imex_stability_slices`.)
- [x] Add Prothero-Robinson with configurable smooth forcing and stiffness parameter. (Landed 2026-09-09: `stiff_relaxation_problem(rate, forcing, sign)` — stable and unstable-PR variants.)
- [x] Add stiff van der Pol with a moderate parameter for deterministic CI and a large parameter for an opt-in benchmark. (Landed 2026-09-09: mu = 2 in CI, mu = 10 in the benchmark figure; explicit wall at mu = 10.)
- [x] Add Robertson or Oregonator kinetics as a positive, nonlinear stiff chemistry benchmark. (Landed 2026-09-09: Robertson kinetics; invariants (mass, positivity, monotone y3, QSS y2) stand in for the exact solution; 10 x dt = 0.001 bootstrap needed to cross the initial QSS layer.)
- [x] Add a semi-discrete diffusion or reaction-diffusion benchmark with a known spectral stiffness scale. (Landed 2026-09-09: `diffusion_problem(n)` on Laplacian eigenvectors 1 and 5 with the exact decay.)
- [x] Add a stiff damped oscillator separating high frequency from true dissipative stiffness. (Landed 2026-09-09: `stiff_damped_oscillator_problem(omega, c)` with the closed form in all three damping regimes.)
- [x] Produce figures comparing solution error, energy/dissipation, nonlinear iterations, Krylov iterations, and RHS cost. (Landed 2026-09-09, energy panel added 2026-09-10: `scripts/stiff_benchmark_suite.py` -> `images/stiff_benchmark_suite.png` — error vs dt on four stiff benchmarks, per-step RHS/GMRES cost panels (GMRES iterations per step equal the stage-solve count, i.e. the per-step nonlinear-iteration cost), and a van der Pol energy-vs-time panel for the settling vs diverging orbits.)

Validation gate:

- [x] CI tests remain small and deterministic.
- [x] Longer benchmark scripts produce versioned PNGs under `images/` and clearly state parameter values and solver settings.
- [x] Claimed stability advantages are demonstrated on a problem where the explicit step restriction is actually active.

## Phase 9: Gradient Coverage — done 2026-09-10

The library's headline claim is "fully differentiable", and until this phase nothing
in the suite ever called `.backward()` through a step. The claim was false for all 19
implicit schemes; see NOTES §2.2 for the finding and the derivation.

- [x] Run the JFNK Newton/GMRES iteration under `torch.no_grad()`. (`gmres` writes its
  Hessenberg factor, Givens rotations and back-substitution vector in place, which the
  autograd tape rejects outright; recording a linear solver is also simply the wrong
  thing to do.)
- [x] Re-attach gradients by the implicit function theorem rather than by unrolling.
  (`_implicit_diff_reattach`: a backward hook maps the incoming cotangent `g` to
  `λ = (I − Jᵀ)⁻¹ g` via one more matrix-free GMRES driven by reverse-mode VJPs.)
- [x] Keep the forward trajectory bit-for-bit identical when no gradient is requested,
  and when one is. (The re-attached tensor carries `out`'s gradient but `y_star`'s
  value; `test_values_are_identical_with_and_without_grad_tracking` pins all 19.)
- [x] Leave `SolveDiagnostics` untouched on the differentiable path — the extra `step`
  applications belong to the gradient machinery, not to the nonlinear solve, so the
  documented cost model (NOTES §3.4) still reads correctly.
- [x] Add `tests/test_gradients.py`: gradients flow through all 19 implicit schemes and
  7 explicit ones, match closed-form amplification matrices, and are independent of
  `newton_tol`.

Validation gate:

- [x] Every implicit scheme produces a finite gradient. (Previously every one raised
  `RuntimeError`; BDF2+/AM only after their explicit cold start handed over.)
- [x] Gradients match an independent closed form, not just each other. (Backward Euler,
  BDF1, IMEX Euler, trapezoidal and implicit midpoint against their exact linear
  amplification matrices on the oscillator: agreement to **1.1e-16**.)
- [x] The gradient is independent of the nonlinear tolerance. (`newton_tol` swept 1.0 →
  1e-3 → 1e-6, gradient unchanged to 1e-11 — the property that distinguishes implicit
  differentiation from unrolling.)
- [x] Full suite green: 2285 passed / 199 skipped, up from 2228 / 199.

### Deferred (gated)

- [?] **Warp gradient path.** Still the open `torch.autograd` vs `wp.Tape` decision of
  NOTES §2.2. Phase 9 settles the torch half only. Note that the adjoint solve needs
  *reverse*-mode VJPs, which warp has — so the implicit-differentiation design carries
  over to a warp backend more cleanly than an unrolled one would have.
- [?] **Gradients through the explicit cold start.** BDF/AM gradients currently flow
  through the Dormand-Prince bootstrap by unrolling it, which is correct but not free.
  Only worth revisiting if a downstream differentiates long multistep runs where the
  startup cost is measurable.

## Phase 10: First-Stage Reuse for Stiffly Accurate DIRK

**Trigger: none needed — this is a measured, order-free saving already sitting in the
driver.** TR-BDF2, ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA and Trapezoidal all satisfy
both halves of the FSAL condition: an explicit first stage (`a[0,0] == 0`) *and* stiff
accuracy (`b == a[-1]`, `c[-1] == 1`). The DIRK driver already re-evaluates the RHS on
the converged final stage, so `stages[-1].update` **is** `f(t^{n+1}, y^{n+1})` — verified
directly: it matches an independent evaluation at the returned state to 0.0 on `forced`
and ~1e-11 (the Newton tolerance) on `oscillator`. That value is exactly what the next
step's explicit first stage needs, so reuse costs no order at all.

Before this phase, `dirk.py` rejected `priorStep` for every scheme. The stated reason —
"a converged stage's `k` was evaluated one Picard iteration before the returned state" —
is contradicted by the driver's own inline comment ("not an approximation one iteration
behind"), and NOTES §3.6's argument that an implicit stage's value depends on `dt`
through the solve is correct for DIRK in general but does not apply to an *explicit*
first stage.

- [x] Teach `reuse.py` the stiffly-accurate-plus-explicit-first-stage case, without
  handing it the explicit-tableau substitution analysis that `_tableau_of` deliberately
  withholds from implicit schemes. (A separate `reuse.dirk_reuse_analysis`, gated on the
  registered `stiffly_accurate` flag with the tableau check as a drift guard;
  `step_reuse_analysis` dispatches to it via `.dirkTableau`.)
- [x] Give `stiffly_accurate` its first consumer — it is currently recorded metadata
  that nothing reads. (`dirk_reuse_analysis` reads it as the primary gate;
  `tests/test_dirk.py::test_dirk_stiffly_accurate_metadata_matches_the_tableau` pins the
  flag to the tableau for all seven schemes so the consumer is non-vacuous.)
- [x] Accept `priorStep` in the DIRK driver for exactly the qualifying tableaus, and
  keep rejecting it for SDIRK2 (stiffly accurate but `a[0,0] != 0`, so its first stage
  is implicit and cannot consume a `k0` directly). (`_dirk_reuses_first_stage` decides
  from the tableau alone; Trapezoidal / TR-BDF2 / both ESDIRKs accept, SDIRK2 and the
  single-stage backward Euler / midpoint still reject with the standard warning.)
- [x] Reconcile the `DIRK` docstring with the driver's actual behaviour.
- [?] **Warm-starting implicit first stages.** For SDIRK2 and any other
  non-explicit-first-stage tableau, a reused `k` is still a better Newton *initial
  guess* than the current cold start. That is a softer, iteration-count win rather than
  an evaluation-count one; measure before building. (Still open — deferred, no
  downstream trigger.)

Validation gate:

- [x] Reuse preserves the measured convergence order for all four qualifying schemes
  (`step_reuse_analysis` reports `reuse_order == order`, not a drop).
  (`tests/test_dirk.py::test_dirk_reuse_preserves_the_measured_order`; reuse/no-reuse
  answers agree to ~1e-11 on `oscillator`, exactly on `forced` — JFNK last-bit
  sensitivity, not an order cost.)
- [x] `supports_step_reuse` and `is_fsal` report these schemes correctly. (Both `True`
  for the four qualifying schemes; the four DIRK entries join the explicit FSAL pair in
  `tests/test_step_reuse.py`.)
- [x] Measured saving matches the prediction from the Phase 8 cost panels: one RHS
  evaluation per step, i.e. TR-BDF2 4.50 → ~3.50/step (~22%), ESDIRK6 10.72 → ~9.7
  (~9%). **The saved evaluation is the cheap explicit one**, so the wall-clock win is
  smaller than the evaluation-count win; report both rather than the flattering one.
  (Confirmed as one *explicit* `f`-call per reusable step, within JFNK iteration noise —
  `tests/test_dirk.py::test_dirk_reuse_saves_an_explicit_evaluation_per_step`. **Metric
  caveat:** the Phase 8 "RHS evaluations/step" panels count
  `SolveDiagnostics.rhs_evaluations`, which the DIRK driver records as `None` for the
  explicit first stage, so *that panel number does not move under reuse* (the 4.50 and
  10.72 exclude the saved evaluation). The saving shows up in raw `f`-call counts, not
  in those panels — see NOTES.md §3.6's Phase 10 note.)

## Phase 11: Adaptive Step Control and Dense Output

**Why this was no longer "out of scope".** This roadmap's header deferred adaptive `dt` to
"a problem-specific driving loop", which was the right call when almost nothing had an
error estimator. Eight schemes then did — Bogacki-Shampine 3(2), Dormand-Prince 5(4),
Cash-Karp 5(4), TR-BDF2, both ESDIRKs and both ARKs (ten today, with ROS3P and EXPRB32) —
and `IntegrationResult.error` was produced by all of them and **read by nothing**. The
controller was the missing consumer, not a missing estimator.

The genuinely hard part was not the controller. It was that `StepHistory` invalidates on
any `dt` change, so adaptive stepping and the entire multistep family (BDF1-5, AM2-4,
AB2-5, ABM2-4) were mutually exclusive. That interaction was the real blocker and was
decided before anything landed: **refuse the combination explicitly** (below), which
removed the blocker without the variable-step coefficient work.

- [x] Add a PI (or predictive) step-size controller with the standard safety factor,
  min/max growth clamps, and a rejection path that re-runs the step. — Predictive:
  `propose_dt(error_norm, dt, order, target=1.0, safety=0.9, growth_min=0.2,
  growth_max=5.0)` = `dt * clamp(safety * (target/error_norm)**(1/order),
  growth_min, growth_max)`, zero error → `dt * growth_max`. The re-run path is
  loop-side, and the demo loops apply the SciPy convention "never grow on a
  rejection".
- [x] Decide the contract: does the driver own the loop, or does the library expose a
  `propose_dt(error, dt, order)` helper the caller drives? — **Helper only**: the
  library exposes `estimate_error_norm` + `propose_dt` and the caller drives the
  accept/reject loop, per the "no solver contract without a downstream" caution
  (NOTES §3.8). The demo loop lives in `scripts/adaptive_benchmark.py` and
  `tests/test_adaptive.py`, not in the library.
- [x] Decide what adaptive `dt` means for multistep. — **Refuse explicitly**
  (the honest-and-cheap option): `estimate_error_norm` raises for the multistep
  family, which emits no estimate and would restart its `StepHistory` on every
  accepted step anyway (`tests/test_groundwork.py::test_step_history_restarts_on_dt_change`).
  Variable-step BDF/Adams coefficients (the CVODE/LSODA route) remain the future
  work if a downstream asks for them.
- [x] Add dense output / interpolants, starting with Dormand-Prince's published
  formula. — `dormand_prince_dense_output(state, stages, dt, theta)`: Shampine's
  1986 quartic continuous extension (optimum `c_6`, as implemented in SciPy's
  `RK45`), transcribed verbatim from the local SciPy 1.18.0 source. Exact at both
  step endpoints (`θ = 1` bit-for-bit the propagated state) and `O(dt^5)` in the
  interior (the continuous order conditions hold through order 4) — measured
  interior rates 4.83 → 5.00. DP5-only: the other nine emitters have no published
  continuous extension and are refused by a stage-count check.
- [x] **Error-norm convention.** — Reused, not a second scale:
  `estimate_error_norm(result, rtol, atol)` = `state_norm(result.error,
  reference=result.state)`, the weighted-RMS ("< 1.0 means at the tolerance") the
  DIRK driver already uses.

Validation gate:

- [x] A stiff problem from the Phase 8 registry completes in materially fewer steps
  under control than at the fixed `dt` needed for the same final error. — van der
  Pol μ = 10, T = 5: 75 accepted steps at rtol = 1e-5 vs 312 fixed (dt = 0.016),
  107 vs 625 (dt = 0.008) at rtol = 1e-6; the test pins `accepted < 0.75 ×` the
  coarsest matching fixed step count.
- [x] Step rejection actually triggers on van der Pol at mu = 10, where the Phase 8
  work already measured a hard explicit wall. — 17–19 rejections per adaptive run,
  out of 72–126 attempts, with the run starting at dt0 = 0.1, deliberately outside
  the wall.
- [x] Dense output reproduces the propagated solution at the step endpoints exactly and
  meets its advertised interpolation order in between. — endpoint exactness is
  pinned bit-for-bit; interior order 4 (local `O(dt^5)`) is measured on the linear
  oscillator.

(Landed 2026-09-14: `src/warpSPHIntegrators/adaptive.py` — `estimate_error_norm`,
`propose_dt`, `dormand_prince_dense_output` + the transcribed `_DP5_DENSE_P` /
`_DP5_B_MAIN` coefficient blocks; the ten emitters and the TR-BDF2 `q = order + 1`
exception; the gate table; `scripts/adaptive_benchmark.py` →
`images/adaptive_benchmark.png`; `tests/test_adaptive.py` (24 tests); NOTES §3.17.)

## Phase 12: Families Not Yet Represented

Ordered by relevance to this library's SPH downstream, not by classical prominence.
Each is gated on a concrete downstream need plus a cost comparison against the Phase 8
baseline — the bar Phase 7 also has to clear. (Phases 6 and 7 themselves were de-gated
2026-09-10 on general-ODE-library grounds; these families were not — they either
duplicate an existing capability or need a downstream structure that has not appeared.)

### Stabilized explicit (RKC / RKL / ROCK) — the strongest omission

- [x] Evaluate RKC or RKL2 ("super-time-stepping") for parabolic SPH terms. Explicit
  Chebyshev recursions whose real-axis stability grows as O(s²) in the stage count,
  matrix-free, with **no nonlinear solver at all**. They attack exactly the regime the
  Phase 2 preconditioning work and the `diffusion_problem(n)` benchmark exist for, and
  for SPH viscosity they are frequently the right answer over implicit. (Landed
  2026-09-11: `rkc.py` implements RKC1 / RKC2 / RKL2 from the published recurrences
  (Meyer, Balsara & Aslam 2014; Ruuth 2001), orders 1 / 2 / 2, real-axis ranges
  K(s) = 2s² / 2(s²−1)/3 / (s²+s−2)/2, exactly `s` RHS evaluations per step; NOTES §3.13.)
- [x] No tableau needed — a coefficient recursion plus a stage-count rule, so the
  marginal cost is close to the "tableau only" tier of NOTES §3.6. (Landed 2026-09-11:
  `stage_count(dt·|λ_max|, family)` returns the smallest admissible `s`, clamped to the
  family's minimum; the drivers take `s=` or `lambda_max=` and refuse first-stage reuse.)
- [x] Benchmark against BE / BDF2 / TR-BDF2 on `diffusion_problem` at several `n`,
  reporting RHS evaluations to a fixed error. (Landed 2026-09-11:
  `scripts/rkc_benchmark.py` → `images/rkc_benchmark.png`. To max error 1e-2 at
  n = 16 / 32 / 64 the family needs 32 / 64 / 128 (RKC1), 40 / 80 / 144 (RKC2),
  48 / 88 / 168 (RKL2) total RHS evaluations, against TR-BDF2 89 / 147 / 271, BE
  268 / 399 / 710, BDF2 224 / 896 / unstable — the implicit cost is dominated by the
  default finite-difference JFNK solve, which also caps BDF2's practical stability.)

### IMEX linear multistep (SBDF2/3, CNAB2)

- [x] Add semi-implicit BDF and Crank-Nicolson/Adams-Bashforth. All three ingredients
  already existed — BDF coefficients, AB coefficients, and `IMEXRHS` — so SBDF2 is BDF2
  with an endpoint extrapolation of the explicit half. These dominate PDE and fluid
  codes, and Phase 5 jumped straight from IMEX Euler to two ARK pairs with nothing
  multistep in between. (Landed 2026-09-15: `imexmultistep.py` — SBDF2 (BDF2 backbone,
  order 2), SBDF3 (BDF3 backbone, order 3), CNAB2 (trapezoidal backbone, order 2); the
  smooth part is the p-point Lagrange endpoint extrapolation (weights [2, −1] /
  [3, −3, 1]) for SBDF, the AB2 increment for CNAB2, and the BDF derivative weight β
  scales the whole endpoint derivative — dropping it from the extrapolation makes the
  combined method first order (measured as an O(1) error plateau, pinned as a
  regression guard). The Phase 5 compatibility rule holds: an ordinary RHS runs the
  pure-implicit limit — bit-exact against the registered BDF2/BDF3, and to solver
  tolerance against the registered trapezoidal scheme for CNAB2; zeroing the implicit
  part instead gives the effective explicit method (bit-exact AB2 for CNAB2, the
  zero-stable Lagrange-extrapolation companion for SBDF2/3, not Adams-Bashforth).
  Stability pinned: zero-stability, exact negative-real-axis boundaries 4/3 (SBDF2),
  20/21 (SBDF3), 1 (CNAB2), and the imaginary axis covered only in a weak whisker —
  the practical `dt·max|λ_E| ≲ 0.5` restriction, confirmed end-to-end on the n = 64
  Burgers convective spectrum (bounded at `|z| ≤ 0.512`, NaN at `|z| ≤ 0.64`), which
  is why the demo runs these at `dt = 0.008`/`0.004` against the ARK pair's `0.02`.
  Measured orders 2.03 / 3.02 / 2.01 on the canonical split (viscous_burgers_demo.ipynb
  §7/§10/§11); cold start and dt-change follow the BDF family (DP5 starter,
  state-snapshot history, re-bootstrap); `tests/test_imexmultistep.py` (34 tests),
  NOTES §3.19.)

### Generalized-alpha / HHT-alpha

- [ ] The natural sibling of the registered Newmark: adds controllable high-frequency
  numerical dissipation, and is close to a parameter generalization of the existing
  167-line `newmark.py`. Relevant to structural/solid SPH.

### Higher-order SSP

- [x] SSPRK(10,4). (Done 2026-09-15: Shu's 10-stage tableau (Shu, *J. Comput. Phys.*
  169 (2001) 208–228, Sec. 4.2) in `butcher.py` — order 4 measured on all three
  problems (4.005 / 4.000 / 3.995), exact SSP coefficient 6.0 (stage 2's constant
  coefficient 1 − mu/6 is the binding constraint; the registered scan caps at 4.0),
  negative real-axis stability interval [−13.916, 0], TVD verdict (4.0, 5.0, True)
  — "unconditionally TVD" up to the CFL cap — and reuse order 3 (the a[−1] weights
  satisfy the stability-polynomial condition w·(a·c) = 1/6 exactly, so reuse is
  lossless on the linear autonomous oscillator and order 3 elsewhere). `tests/
  test_tvd.py` pins the exact boundary and the verdict; NOTES §3.20.)
- [ ] SSPRK(5,4). Butcher's 5-stage order-4 SSP method (Butcher 1987 / Shu 2002) is
  the standard order-4 SSP workhorse. Still unimplemented — and not in the SUNDIALS
  ARKODE v7.9.0 bundle (only the LS-RK I/O layer references SSPRK tableaus), so the
  coefficients would come from the paper. The shipped SSP/TVD set stopped at order
  3 — precisely where the SSP barrier for explicit RK bites (order 4 needs 5+
  stages) — until SSPRK(10,4) landed 2026-09-15.

### High-order symplectic composition

- [?] Yoshida 4/6/8, Suzuki, Blanes-Moan, or a general `compose()` helper over the
  existing Verlet base. **Gated on a separable Hamiltonian downstream**, and that gate
  is real: NOTES already measures the whole Verlet/Forest-Ruth family dropping to first
  order under a velocity-dependent force, which is every actual SPH momentum equation.
  Worth noting against Phase 6's trigger, which cites "order-4 symplectic accuracy" —
  for a *separable* Hamiltonian, composition reaches order 6-8 explicitly and far more
  cheaply than Gauss-Legendre s=3, so Phase 6's trigger should say "non-separable"
  explicitly rather than leaving the cheaper route looking unconsidered.

### One-step Runge-Kutta-Nystrom

- [?] NOTES §3.6 lists *multistep* Nystrom (Stormer-Cowell) and Gauss-Jackson as
  not-done, but not one-step RKN. Same separability caveat as composition methods.

## Phase 13: Nonlinear Stability for the Explicit Side

Phase 8 built a thorough stability story that is entirely **linear** (Dahlquist regions,
A(alpha) cones, amplification matrices) plus nonlinear *stiff* benchmarks. The
explicit/hyperbolic half has no equivalent, and three registered schemes advertise a
property nothing measures.

- [x] Measure the SSP coefficient of SSP RK3, TVD RK2 and TVD RK3 directly, and
  document the forward-Euler CFL each one preserves. Right now the defining property of
  these three schemes is asserted by their names alone. (Measured: all three have the
  published `r = 1`, i.e. a convex-combination step up to CFL 1 = the forward-Euler
  CFL, from the Fourier stage maps of the upwind-advection model
  (`tvd_analysis.convex_combination_cfl`); `tests/test_tvd.py` pins it.)
- [x] Add a hyperbolic benchmark problem. All nine current problems are dissipative or
  Hamiltonian ODEs; `diffusion_problem` is the only PDE semi-discretization and there is
  nothing hyperbolic. A scalar advection or inviscid Burgers semi-discretization would
  give the SSP schemes something to be right about. (Two landed: `advection_problem`
  — 1D periodic first-order upwind advection, the classifier's model problem because
  every mode is an exact eigenmode — with `mode` (exact solution) and `step` (TV = 4)
  initialisations, and `burgers_problem`, semi-discrete inviscid Burgers with
  Lax-Friedrichs fluxes; the `PROBLEMS` registry now has eleven.)
- [x] Add total-variation / positivity assertions on that problem, in the same
  "numerical assertion, not just a figure" style Phase 8 used for the implicit side.
  (TVD RK2/3 per-step TV non-increasing at CFL 1 on step advection and Burgers, and
  positivity of a non-negative IC under TVD RK3 at CFL 1 — all in `tests/test_tvd.py`.)
- [x] Generalize the TVD check into a **classifier usable on any registered scheme**,
  not just the three TVD/SSP-named ones: confirm TVD RK2 and TVD RK3 are actually TVD
  at their documented CFL, then scan every other registered scheme for TVD-ness despite
  lacking a TVD name — TVD is strictly broader than SSP, so non-SSP candidates can pass.
  Report the verdict per scheme with the measured TVD CFL (or a concrete counterexample
  for the non-TVD ones). Requested 2026-09-10: TVD-ness is a property worth knowing for
  all schemes, so build it as a general classifier rather than per-scheme assertions.
  (`tvd_analysis.classify_tvd` / `classify_all` + `scripts/tvd_classifier.py`, two
  measurements per scheme: the rigorous SSP coefficient from the stage maps where a
  tableau is exposed, and the measured per-step TVD CFL from a trajectory sweep that
  also covers the tableau-less BDF/Adams family. The verdict table is the maintained
  fact of NOTES §3.11 and is pinned for the whole registry by
  `tests/test_tvd.py::test_classifier_verdicts_match_the_recorded_landscape`.
  Findings: TVD ⊋ SSP in the registry — RK4 (`r = 2/3`) is per-step TVD to CFL 1.25,
  and Nystrom 5th order / Dormand-Prince (`r = 0`, stage maps negative at every
  resolvable CFL, checked by exact rational arithmetic) are still per-step TVD to
  CFL 1.5; the L-stable implicits are TVD to the tested CFL 5; the multistep family
  to a measured CFL 1.5.)

Validation gate:

- [x] Each SSP scheme's measured SSP coefficient matches its published value.
  (SSP RK3 / TVD RK2 / TVD RK3 all measure `r = 1`; the landmark values of the rest
  of the registry — RK3 `1/2`, RK4 `2/3`, RK4alt `1/3`, Cash-Karp `5/12`, BE
  unconditional, IMID/Trapezoidal `2`, SDIRK2/TR-BDF2 `1+√2`, Nystrom/DP5/ESDIRK
  `0` — are pinned in `tests/test_tvd.py`.)
- [x] A non-SSP scheme of the same order visibly violates the TVD bound on the same
  problem where the SSP schemes hold it — otherwise the test is not measuring the
  property it claims to. (Per-step TV cannot separate them — all third-order RK
  methods share the same final map, convex to CFL 1, so classical RK3 and TVD RK3
  both show zero per-step TV increase at CFL 1. The violation is at the *stage*
  level: RK3's stage-3 row at CFL 1 is `[1, -1, 1]` (circulant entry `μ(1-2μ) = -1`)
  while TVD RK3's stage rows are `[0, 1]` and `[3/4, 0, 1/4]`;
  `tests/test_tvd.py::test_classical_rk3_stage_leaves_the_convex_hull` runs the
  registered drivers from a delta IC so each stage state is exactly its row.)
- [x] The TVD classifier's verdict is recorded for every registered scheme (the
  TVD-named ones confirmed, any newly-discovered TVD scheme's CFL documented), so the
  classification stays a maintained fact rather than a one-off script run. (The full
  table is in NOTES §3.11, printed by `scripts/tvd_classifier.py`, and pinned per
  scheme — including the "not applicable" Verlet/Newmark verdicts — in
  `tests/test_tvd.py::test_classifier_verdicts_match_the_recorded_landscape`.)

## Phase 14: Structured RHS Interface

**Status: done 2026-09-10.** (Was: not gated — groundwork that also tidies the
current API.) Today a scheme
receives its dynamics one of two ways: a bare callable (fully implicit, or just
evaluated) or `IMEXRHS(explicit=, implicit=)` (additive split, read by IMEX Euler and
ARK via `isinstance`). Phase 7's Rosenbrock-W and exponential integrators need a third
kind of structure — a linear-operator action `L·v` and its nonlinear remainder
`N = f − L·y` — with nowhere to put it. Rather than add a second NamedTuple beside
`IMEXRHS`, collapse the input surface to exactly two shapes: **a plain function, or a
typed `RHS`** whose declared capabilities cover every split any registered or planned
scheme asks for. `IMEXRHS` folds in as one shape of it; nothing downstream constructs
an `IMEXRHS`, so the reshape has no external contract to preserve.

- [x] Add a concrete `RHS` base class (not a bare structural `Protocol` — a plain
  function must never `isinstance`-match it) with `__call__(state) -> Update` (the
  combined `f = f_E + f_I = L·y + N`, always defined, the ground truth) and a
  `provides: frozenset[str]` over `{"explicit", "implicit", "linear", "nonlinear"}`.
- [x] Optional accessors `explicit(state)`, `implicit(state)`, `linear(v, state)`,
  `nonlinear(state)`, each with the same `(…) -> Update` contract as the RHS callable,
  so a driver routes each through `updateStep` exactly as `ark.py` already does with
  the two `IMEXRHS` halves.
- [x] A bare callable stays first-class: `f_combined = f` works whether `f` is a
  function or an `RHS`, and a scheme that only needs the combined RHS never branches.
- [x] `IMEXRHS(explicit=, implicit=)` becomes a thin constructor returning an `RHS`
  with `provides={"explicit","implicit"}` and a summing `__call__`; `ark.py`,
  `imex.py` and their tests move from `isinstance(f, IMEXRHS)` to the capability check
  with no behaviour change.
- [x] `SemilinearRHS(linear=, nonlinear=)` — and the `linear=` + combined-`f` form
  that synthesizes `nonlinear = f − L·y` — as the sugar for the semilinear split;
  `provides={"linear","nonlinear"}`.
- [x] `RHS(**parts)` general constructor accepting any subset, so one object can
  declare all four (the overlap case: the same stiff operator behind both `implicit`
  and `linear`).
- [x] Driver resolution rules: synthesize what can be (`implicit` ← combined,
  `explicit` ← 0, `nonlinear` ← `f − L·y`); require what cannot (`linear`); and fail
  before the solve with a message naming the missing capability when a scheme needs a
  split the RHS does not carry.
- [x] `J·v` stays derived from the combined `f` (`jfnk.jvp_matvec` / finite
  difference); it is not an `RHS` slot. A frozen local linearization `J(y^n)` handed
  to Rosenbrock is an explicit opt-in, never a silent stand-in for a supplied
  `linear`.
- [x] Document and debug-mode check the three contracts: additivity is disjoint
  (`explicit + implicit == f`), `linear` is linear and homogeneous (`L·0 = 0`,
  spot-checked `L(a+b) ≈ L(a)+L(b)` on random vectors — the identity-preconditioner
  regression style), and any two declared splits agree (`explicit+implicit ==
  linear·y+nonlinear == f`). (`check_contracts` in `rhs.py`, on demand, not on the
  production step path.)

### Test problem: viscous Burgers

- [x] Add semi-discrete **viscous Burgers** `u_t + (u²/2)_x = ν u_xx` to `testing.py`
  as the canonical semilinear benchmark: `ν u_xx` is a constant-coefficient linear
  operator `L` with a cheap matrix-free action, `(u²/2)_x` is the nonlinear remainder
  `N`. Ship it as a `SemilinearRHS` so the split is exercised end to end, and keep a
  manufactured or fine-grid reference so it doubles as the Phase 7 accuracy/cost
  problem — exponential integrators are conventionally demonstrated on exactly this
  equation. Distinct from the existing inviscid `burgers_problem` (Lax-Friedrichs,
  TVD/SSP classifier): different `ν` regime, different purpose. (Landed as
  `viscous_burgers_problem` / `PROBLEMS['viscousBurgers']`; the fine-grid reference
  stays a Phase 7 item — it only exists once there is a method to check it against.)

Validation gate:

- [x] Every registered scheme produces bit-identical trajectories on a bare callable
  before and after the refactor. (Golden master `tests/data/bitident_golden.json`,
  captured pre-refactor, dt = 0.1 × 3 steps on the one-DOF oscillator — strict
  float equality over all 52 registered schemes, pinned by
  `tests/test_rhs.py::test_bitidentical_bare_callable_trajectories`.)
- [x] ARK / IMEX Euler behave identically given an `IMEXRHS` (now a constructor) or a
  hand-built `RHS` with the same halves.
- [x] A `linear`-consuming scheme raises a specific capability error on a plain `f`
  and on an `IMEXRHS`, not a mid-solve shape mismatch.
- [x] `nonlinear` synthesized from `f − L·y` matches an independently supplied `N` on
  viscous Burgers to round-off.

### Out of scope — no accessor without a consumer

- Component partition (`f_q`, `f_p`) for partitioned RK — the
  `applyPositionUpdate` / `applyVelocityUpdate` path already carries that structure;
  revisit only with Lobatto IIIA-IIIB.
- Mass matrix `M y' = f` — a future accessor if a DAE downstream appears.
- More-than-two-way additive splits (multirate / MRI).

## Housekeeping

- [x] `midPoint` is imported in `integration.py` and never registered — dead import. (Done 2026-09-15: removed; the two `Midpoint` registrations and `IntegrationSchemeType.midPoint` were the only live paths, and no other `midPoint` references remain — grep-verified.)
- [x] `nonLagrangian` agrees with `dissipation` for every registered scheme, carries no
  information, and is documented as a removal candidate. Remove it, or give it a
  meaning. (Done 2026-09-15: removed — the branch it gated is dead on every bundled
  path and the field metadata already carries the meaning the flag documented;
  NOTES §2.14.)
- [x] Phase 6's `[x]` symplectic-form validation item predates the rest of the phase;
  now that Phase 6 is active (de-gated 2026-09-10) it can stay in place, but the
  finding itself also belongs in the Phase 8 record. (Done 2026-09-15: the Phase 6
  symplectic-form findings are appended to the Phase 8 validation item above.)
- [x] The Current Baseline's JFNK-default list omits ARK, which does default to
  `JFNKSolver`. (Done 2026-09-15: the line now names ARK, the coupled block RK pair,
  and IMEX multistep.)

## Recommended Execution Order

1. [x] Phase 0 documentation reconciliation.
2. [x] Phase 1 solver diagnostics and failure handling.
3. [x] Phase 3 TR-BDF2, then ESDIRK3. (Landed 2026-09-09: TR-BDF2 with SUNDIALS ARKODE's published (2, 3) embedded pair; ESDIRK3(2)4L[2]SA and ESDIRK4(3)6L[2]SA from the same source.)
4. [x] Phase 5 ARK3 IMEX after the existing `IMEXRHS` API is stress-tested. (Landed 2026-09-09: `ark.py` additive driver with ARK3(2)4L[2]SA and ARK4(3)6L[2]SA from SUNDIALS ARKODE v7.9.0; two-parameter IMEX stability function + slice figure; split and pure-limit convergence verified in `tests/test_ark.py`.)
5. [x] Phase 2 preconditioning using the first real SPH diffusion/acoustic downstream. (Landed 2026-09-09: left/right preconditioned GMRES + 3-arg `preconditioner(v, state, context)` hook on JFNKSolver, identity/diagonal helpers, size-sweep benchmark figure; validated on the wave equation via the block-lower-triangular Laplacian preconditioner — operator-based, no dense Jacobian.)
6. [x] Phase 4 BDF3-BDF5 and true Adams-Moulton. (Landed 2026-09-09: BDF4/BDF5 from the exact order conditions with measured A(α) cones 73.35°/51.84°, zero-stability + stability-boundary tests; JFNK-corrected Adams-Moulton AM2-AM4 with derivative history and an optional AB predictor; the iterated corrector beats same-order PECE by ~10x on the stiff nonlinear relaxation.)
7. [x] Phase 8 broadened nonlinear/stiff benchmark and stability suite throughout. (Landed 2026-09-09: five new problem factories (stiff PR both signs, stiff damped oscillator, van der Pol, Robertson, diffusion), damped-oscillator amplification matrices + damping-ratio stability gallery, `tests/test_benchmarks.py` (52 tests), and `images/stiff_benchmark_suite.png` with per-step JFNK cost panels.)
8. [x] Phase 9 gradient coverage. (Landed 2026-09-10: the "fully differentiable" claim was false for all 19 implicit schemes — `gmres`'s in-place workspace made the autograd tape reject every one. `JFNKSolver.solve` now solves under `no_grad` and re-attaches by the implicit function theorem; gradients verified against closed-form amplification matrices to 1.1e-16 and shown independent of `newton_tol`; `tests/test_gradients.py`, suite 2228 → 2285.)
9. [x] Phase 10 stiffly-accurate DIRK first-stage reuse. (Landed 2026-09-10: `reuse.dirk_reuse_analysis` + the driver's `_dirk_reuses_first_stage` accept `priorStep` losslessly for Trapezoidal / TR-BDF2 / both ESDIRKs — the stiffly-accurate tableaus with an explicit first stage — while SDIRK2 and the single-stage backward Euler / midpoint still reject; the `stiffly_accurate` metadata gained its first consumer; order preserved and one explicit RHS eval/step saved within JFNK noise, `tests/test_dirk.py`; suite 2285 → 2320.)
10. [x] Phase 13 explicit-side nonlinear stability — small, and it closes the one place where a registered scheme advertises a property nothing measures. Includes the requested general TVD classifier (verify the TVD-named schemes, scan all others; TVD is broader than SSP, 2026-09-10). (Landed 2026-09-10: `advection_problem` (upwind advection, the classifier's model problem) + `burgers_problem` in `testing.py`; `tvd_analysis.py` — rigorous SSP coefficient from the Fourier stage maps plus measured per-step TVD CFL from a trajectory sweep, and `classify_tvd`/`classify_all` over the whole registry; `scripts/tvd_classifier.py` prints the maintained verdict table (NOTES §3.11): TVD RK2/3 + SSP RK3 at the published r = 1, RK4 r = 2/3 yet TVD to CFL 1.25, Nystrom/DP5/ESDIRK/ARK r = 0 (stage maps negative at every resolvable CFL — Nystrom/DP5 by exact rational arithmetic) yet per-step TVD to 1.5, L-stable implicits to CFL 5, multistep family to a measured CFL 1.5; the stage-level gate separates classical RK3 (stage-3 row [1, -1, 1] at CFL 1) from TVD RK3 where per-step TV cannot; `tests/test_tvd.py` pins the whole table; suite 2320 → 2346.)
11. [x] Phase 14 structured RHS interface — collapses the RHS input surface to "a plain function or a typed `RHS`", folds in the `IMEXRHS` split, and adds the `linear`/`nonlinear` accessors plus the viscous Burgers semilinear test problem. (Landed 2026-09-10: `rhs.py` — concrete `RHS` class + `provides` capabilities, `IMEXRHS`/`SemilinearRHS` constructors, `resolve` with pre-solve capability errors, `check_contracts` for the three split contracts; `ark.py`/`imex.py` moved off `isinstance`; the old `IMEXRHS` NamedTuple is gone from `specs.py`; `viscous_burgers_problem` in `testing.py` as the Phase 7 benchmark; the pre-refactor bit-identical golden over all 52 registered schemes pinned in `tests/test_rhs.py` (19 tests); NOTES §3.12; suite 2346 → 2365.)
12. [x] Phase 11 adaptive step control, once the multistep-vs-variable-`dt` contract is decided. (Landed 2026-09-14: the contract was decided first — **multistep refused explicitly** (`estimate_error_norm` raises; `StepHistory` restarts on any `dt` change, so a variable-`dt` run would throw the history away every step) — then `src/warpSPHIntegrators/adaptive.py` as **helpers, not a solver**: `estimate_error_norm` (the embedded-pair difference as a dimensionless number, in the DIRK driver's weighted-RMS convention) + `propose_dt` (predictive controller: `dt * clamp(safety * (target/error)**(1/order), 0.2, 5.0)`, safety 0.9, zero error → max growth), with the accept/reject loop left to the caller (NOTES §3.8's caution; the demo loop lives in the benchmark and the tests), plus `dormand_prince_dense_output` — Shampine's 1986 quartic continuous extension of DP5 (the optimum-`c_6` formula as implemented in SciPy's `RK45`, transcribed verbatim from the local source), exact at both step endpoints and `O(dt^5)` interior (measured rates 4.83 → 5.00), DP5-only. The ten estimate emitters and TR-BDF2's `q = order + 1 = 3` exception (its published SUNDIALS pair is the propagated branch's true `O(h^3)` local error). Gate on van der Pol μ = 10, T = 5, dt0 = 0.1: 17–19 rejections per run at the Phase 8 explicit wall, and 75 accepted steps vs 312 fixed (dt = 0.016) at rtol = 1e-5, 107 vs 625 (dt = 0.008) at rtol = 1e-6; `scripts/adaptive_benchmark.py` → `images/adaptive_benchmark.png`; `tests/test_adaptive.py` (24 tests), NOTES §3.17; suite 2504 → 2528 passed / 344 skipped.)
13. [x] Phase 12 missing families, RKC/RKL first: it is the one candidate that directly challenges the Phase 8 cost baseline on a benchmark that already exists. (RKC/RKL landed 2026-09-11: RKC1 / RKC2 / RKL2 in `rkc.py` — the published recurrences, per-step stage count from `stage_count(dt·|λ_max|)`, orders 1 / 2 / 2 verified on the semi-discrete diffusion, stability pinned at the K(s) boundary; `scripts/rkc_benchmark.py` reaches max error 1e-2 at n = 16 / 32 / 64 with fewer total RHS evaluations than BE / BDF2 / TR-BDF2 under the default JFNK, and exposes BDF2's solver-limited practical stability there; `tests/test_rkc.py` (35 tests), NOTES §3.13; suite 2365 → 2400. The IMEX linear multistep subfamily landed 2026-09-15: SBDF2 / SBDF3 / CNAB2 in `imexmultistep.py` — BDF2 / BDF3 / trapezoidal implicit backbones with the smooth part extrapolated explicitly to the endpoint (Lagrange weights [2, −1] / [3, −3, 1], the AB2 increment for CNAB2, the BDF derivative weight β scaling the whole endpoint derivative), orders 2 / 3 / 2 measured on the canonical viscous-Burgers split (2.03 / 3.02 / 2.01), pure-implicit limits bit-exact against the registered BDF2 / BDF3 (trapezoidal to solver tolerance for CNAB2), effective-explicit-limit stability pinned at the exact negative-real-axis boundaries 4/3, 20/21, 1 with the imaginary axis only in a weak whisker (the practical `dt·max|λ_E| ≲ 0.5` cap), DP5 cold start and state-snapshot history as in the BDF family; `tests/test_imexmultistep.py` (34 tests), NOTES §3.19, viscous_burgers_demo.ipynb §7/§10/§11. The order-4 SSP method landed 2026-09-15: SSPRK(10,4) in `butcher.py` (exact SSP coefficient 6.0, real-axis interval [−13.916, 0], order 4 measured, NOTES §3.20); Phase 12's remaining families — generalized-alpha / HHT-alpha, SSPRK(5,4), high-order symplectic composition, one-step RKN — stay gated as listed in the phase section.)
14. [x] Phase 6 coupled implicit RK — done 2026-09-15 (de-gated 2026-09-10: Gauss-Legendre is the library's only symplectic method above order 2, Radau IIA its only no-compromise stiff method). `fullyimplicit.py` — the coupled block driver: one `JFNKSolver` call per step over a `BlockState` of `s` sub-states (`fields.py` bridge functions recurse substate-major), `step_fn` = `s` RHS evaluations, diagnostics scaled ×s, `priorStep` refused (no explicit first stage), `warmStart=` seeds the block solve from the previous step's converged stages. Tableaus hand-derived and verified (neither in SUNDIALS ARKODE v7.9.0): **Gauss-Legendre 2** (order 4, A-stable, symplectic for separable Hamiltonians — pinned by measurement at 3.05e-13 under the strict solver; its tableau is *not* A-symmetric, correcting the common claim; not L-stable, R(-100) = 2353/2653) and **Radau IIA s=2** (order 3, L-stable, stiffly accurate, |R(-100)| ≈ 0.019), each with a null-stage min-norm order-2 companion (the plan's "standard Hairer-Wanner estimator" is degenerate — it is `b` itself — for both 2-stage tableaus; measured rate 3.0 for both). The exact-JVP block matvec is the default (FD cross-check to 5.3e-9 relative); the 256-DOF = 1536-unknown scale gate is matrix-free — the gate's GMRES spectrum (128 complex pairs on a 2-D annulus, cond ≈ 89) is the reason the gate disables restarts (SciPy cross-checked). `tests/test_fullyimplicit.py` (32 tests), both schemes in the `test_hamiltonian` / `test_stiff` / `test_gradients` / `test_tvd` pins, `scripts/blockrk_benchmark.py` → `images/blockrk_benchmark.png`, NOTES §3.18; `tests/test_fullyimplicit.py` is the new file, the six other test files gained their entries.)
15. [x] Phase 7 Rosenbrock-W (outright) then exponential integrators. (Rosenbrock-W **landed 2026-09-11**: ROS3P in `rosenbrock.py` — a 3-stage order-3 A-stable (not L) Rosenbrock-W method (Lang & Verwer, *BIT* 41(4) 2001), `w='jvp'`/`'fd'`/`'linear'` frozen-Jacobian selector, `f_t='fd'` one-sided time derivative, frozen-`W` adjoint (three transposed `gmres` solves + the `W^T` VJP). Gate cleared on viscous Burgers: order 3 measured for `jvp`/`fd`, and ROS3P (`w='jvp'`) is the cheapest order-3 total work-unit cost at `dt = 0.02` (706 < ARK3 793 < ESDIRK3 988, `viscous_burgers_demo.ipynb` §11) — with the caveat that the additive ARK3 split is cheaper per iteration in FLOPs (linear-operator matvec). `tests/test_rosenbrock.py` (18 tests), NOTES §3.14. **Exponential integrators landed 2026-09-12/13**: the matrix-free `krylov_phi` build + ETD2RK (order 2, exact on linear, L-stable; NOTES §3.15) and EXPRB32 (order 3, L-stable, embedded (3, 2) estimate, full frozen Jacobian via forward-mode AD, no semilinear split needed; NOTES §3.16), which closes the family's cost gate: at fixed error (rel L2 ≤ 1e-3, sine IC, largest qualifying `dt`) EXPRB32 is the cheapest order-3 method on the work-unit metric — 640 < ROS3P 706 < ESDIRK3 715 < ARK3 793, with the smallest error of the four (1.2e-4) and a fair cost class (full-Jacobian JVP Krylov matvecs, unlike ETD2RK's cheap `L`-matvecs). `tests/test_exponential.py` (20 tests), suite → 2504 passed / 344 skipped (2026-09-13).)

## Completion Definition

- [x] Each registered scheme has verified coefficients, correct metadata, a documented RHS/history contract, and targeted regression tests. (Coefficients: SUNDIALS ARKODE v7.9.0 for TR-BDF2/ESDIRK/ARK pairs, exact order conditions for BDF4-5 and AM2-4, Taylor-model + local-error verification (Phases 3-5); metadata/tables reconciled in Phase 0; per-family regression tests in `tests/`.)
- [x] Every stiff-method claim is backed by a nonlinear stiff benchmark and a stability diagnostic appropriate to that method family. (Phase 8: van der Pol, Robertson, diffusion, Prothero-Robinson in both signs, stiff damped oscillator; per-family diagnostics: Dahlquist regions, A(α) cones, L-damping assertions, damped-oscillator amplification matrices, two-parameter IMEX slices.)
- [x] Solver diagnostics make failed or stalled nonlinear solves observable to callers. (Phase 1: `SolveDiagnostics` in `IntegrationResult` — residual, GMRES iterations, RHS evaluations, line-search backtracks, termination reason — with finite-value checks and structured failure paths.)
- [x] New methods do not regress copied fields, stage times, history invalidation, gradients, or existing public APIs. (Copied fields: Phase 3 gate; stage times: `forced`-problem coverage; history invalidation: Phase 4 gate + `StepHistory` dt/uid guards; public API: full suite green after every phase. Gradient-through-step **was** the exception and is now covered — see Phase 9 below: the "differentiable by construction" claim was false for every implicit scheme, and `tests/test_gradients.py` (57 tests) now pins it. The warp gradient path remains open, NOTES §2.2.)
- [x] The full test suite passes in the `warp` environment. (2365 passed / 187 skipped as of 2026-09-10.)
