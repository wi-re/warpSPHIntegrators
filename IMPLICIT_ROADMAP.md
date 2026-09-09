# Implicit and Stiff Integration Roadmap

Status markers: `[x]` complete, `[>]` in progress, `[ ]` planned, `[?]` needs a downstream use case or design decision.

This roadmap covers the remaining implicit, stiff, IMEX, stability, and nonlinear-validation work. Adaptive timestep control is explicitly out of scope: it belongs to a problem-specific driving loop and only applies where a meaningful local error estimator is available.

## Current Baseline

- [x] Nonlinear-solver protocol with fixed Picard, relaxed Picard, and matrix-free JFNK.
- [x] JFNK is the default nonlinear closure for DIRK, Newmark, BDF, implicit Adams-Moulton, and IMEX Euler.
- [x] DIRK: Backward Euler, implicit midpoint, trapezoidal / Crank-Nicolson, SDIRK2, TR-BDF2, ESDIRK3(2)4L[2]SA, and ESDIRK4(3)6L[2]SA.
- [x] Newmark average-acceleration and linear-acceleration variants.
- [x] BDF1-BDF5 with state-bearing `StepHistory` and a safe Dormand-Prince cold start.
- [x] Fully implicit (iterated) Adams-Moulton AM2-AM4 with derivative-bearing `StepHistory` and an optional Adams-Bashforth predictor.
- [x] IMEX Euler with explicit `IMEXRHS(explicit=..., implicit=...)` callbacks; an ordinary RHS remains fully implicit.
- [x] Dahlquist stability-region plots and numerical boundary checks for tableau methods and BDF1-BDF5.
- [x] Nonlinear Kepler convergence coverage and nonlinear stiff relaxation coverage.

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

- [ ] Create a block-state representation for all implicit stages, with flatten/unflatten over `s * N` unknowns.
- [ ] Extend JFNK residual construction to coupled stage systems without forming a dense block Jacobian.
- [ ] Implement Gauss-Legendre s=2 (order 4) first.
- [x] Validate direct symplectic-form behavior on oscillator and Kepler after tightening the default JFNK residual and finite-difference settings.
- [ ] Implement Radau IIA s=2 (order 3, L-stable) as the first high-quality fully implicit stiff method.
- [ ] Consider Gauss-Legendre s=3 and Radau IIA s=3 only after the block solver and preconditioner are proven at scale.
- [ ] Consider Lobatto IIIA-IIIB only for a concrete partitioned/separable Hamiltonian downstream.

Validation gate:

- [ ] Block residual and Jacobian-vector products agree with finite differences on small systems.
- [ ] Gauss-Legendre order 4 and Radau order 3 are demonstrated on nonlinear reference problems.
- [ ] No dense Jacobian allocation occurs for large state sizes.

## Phase 7: Alternative Stiff Families

### Rosenbrock / W methods

- [ ] Evaluate a linearly implicit Rosenbrock-W method as a lower-nonlinear-iteration alternative for diffusion-like SPH terms.
- [ ] Reuse JVP and preconditioner hooks, but introduce a dedicated linear-solve method interface rather than pretending it is a DIRK tableau.
- [ ] Start with a verified second- or third-order Rosenbrock-W method with an embedded estimator.

### Exponential integrators

- [ ] Only pursue when a downstream exposes a natural linear stiff operator plus nonlinear remainder.
- [ ] Define matrix-function-vector product hooks (`phi_k(hL)v`) without dense matrix construction.
- [ ] Start with ETD2 / exponential Rosenbrock on a diffusion-dominated semi-discretization.

Validation gate:

- [ ] Each family must beat JFNK DIRK or IMEX on a representative downstream cost/accuracy measurement before expanding its method set.

## Phase 8: Stability and Nonlinear Benchmark Suite

- [x] Dahlquist stability regions for implemented tableau and BDF methods.
- [x] Add automatic numerical assertions for known points in every plotted one-step tableau and BDF1/2 region, not only image generation.
- [x] Add second-order undamped-oscillator amplification-matrix plots for Newmark and the Verlet family over `dt * omega`.
- [x] Extend the oscillator stability plots over damping ratio. (Landed 2026-09-09: closed-form damped amplification matrices in `stability.py`, verified against the registered schemes' one-step maps; `log10(rho)` heatmaps over `(h*omega, zeta)` in `scripts/oscillator_stability_gallery.py`.)
- [x] Add direct finite-difference phase-area checks across every registered scheme and nonlinear Kepler symplectic-form checks for the genuinely symplectic subset.
- [x] Add two-parameter IMEX stability slices for IMEX Euler and each later ARK method. (Done in Phase 5, `imex_stability_slices`.)
- [x] Add Prothero-Robinson with configurable smooth forcing and stiffness parameter. (Landed 2026-09-09: `stiff_relaxation_problem(rate, forcing, sign)` — stable and unstable-PR variants.)
- [x] Add stiff van der Pol with a moderate parameter for deterministic CI and a large parameter for an opt-in benchmark. (Landed 2026-09-09: mu = 2 in CI, mu = 10 in the benchmark figure; explicit wall at mu = 10.)
- [x] Add Robertson or Oregonator kinetics as a positive, nonlinear stiff chemistry benchmark. (Landed 2026-09-09: Robertson kinetics; invariants (mass, positivity, monotone y3, QSS y2) stand in for the exact solution; 10 x dt = 0.001 bootstrap needed to cross the initial QSS layer.)
- [x] Add a semi-discrete diffusion or reaction-diffusion benchmark with a known spectral stiffness scale. (Landed 2026-09-09: `diffusion_problem(n)` on Laplacian eigenvectors 1 and 5 with the exact decay.)
- [x] Add a stiff damped oscillator separating high frequency from true dissipative stiffness. (Landed 2026-09-09: `stiff_damped_oscillator_problem(omega, c)` with the closed form in all three damping regimes.)
- [x] Produce figures comparing solution error, energy/dissipation, nonlinear iterations, Krylov iterations, and RHS cost. (Landed 2026-09-09: `scripts/stiff_benchmark_suite.py` -> `images/stiff_benchmark_suite.png`, error vs dt on four stiff benchmarks plus per-step RHS/GMRES cost panels.)

Validation gate:

- [x] CI tests remain small and deterministic.
- [x] Longer benchmark scripts produce versioned PNGs under `images/` and clearly state parameter values and solver settings.
- [x] Claimed stability advantages are demonstrated on a problem where the explicit step restriction is actually active.

## Recommended Execution Order

1. [x] Phase 0 documentation reconciliation.
2. [x] Phase 1 solver diagnostics and failure handling.
3. [x] Phase 3 TR-BDF2, then ESDIRK3. (Landed 2026-09-09: TR-BDF2 with SUNDIALS ARKODE's published (2, 3) embedded pair; ESDIRK3(2)4L[2]SA and ESDIRK4(3)6L[2]SA from the same source.)
4. [x] Phase 5 ARK3 IMEX after the existing `IMEXRHS` API is stress-tested. (Landed 2026-09-09: `ark.py` additive driver with ARK3(2)4L[2]SA and ARK4(3)6L[2]SA from SUNDIALS ARKODE v7.9.0; two-parameter IMEX stability function + slice figure; split and pure-limit convergence verified in `tests/test_ark.py`.)
5. [x] Phase 2 preconditioning using the first real SPH diffusion/acoustic downstream. (Landed 2026-09-09: left/right preconditioned GMRES + 3-arg `preconditioner(v, state, context)` hook on JFNKSolver, identity/diagonal helpers, size-sweep benchmark figure; validated on the wave equation via the block-lower-triangular Laplacian preconditioner — operator-based, no dense Jacobian.)
6. [x] Phase 4 BDF3-BDF5 and true Adams-Moulton. (Landed 2026-09-09: BDF4/BDF5 from the exact order conditions with measured A(α) cones 73.35°/51.84°, zero-stability + stability-boundary tests; JFNK-corrected Adams-Moulton AM2-AM4 with derivative history and an optional AB predictor; the iterated corrector beats same-order PECE by ~10x on the stiff nonlinear relaxation.)
7. [x] Phase 8 broadened nonlinear/stiff benchmark and stability suite throughout. (Landed 2026-09-09: five new problem factories (stiff PR both signs, stiff damped oscillator, van der Pol, Robertson, diffusion), damped-oscillator amplification matrices + damping-ratio stability gallery, `tests/test_benchmarks.py` (52 tests), and `images/stiff_benchmark_suite.png` with per-step JFNK cost panels.)
8. [ ] Phase 6 coupled implicit RK only when a high-order symplectic or Radau use case justifies the block solver.
9. [ ] Phase 7 Rosenbrock/exponential methods only when their downstream structure makes them competitive.

## Completion Definition

- [ ] Each registered scheme has verified coefficients, correct metadata, a documented RHS/history contract, and targeted regression tests.
- [ ] Every stiff-method claim is backed by a nonlinear stiff benchmark and a stability diagnostic appropriate to that method family.
- [ ] Solver diagnostics make failed or stalled nonlinear solves observable to callers.
- [ ] New methods do not regress copied fields, stage times, history invalidation, gradients, or existing public APIs.
- [ ] The full test suite passes in the `warp` environment.
