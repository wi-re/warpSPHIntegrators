# Implicit and Stiff Integration Roadmap

Status markers: `[x]` complete, `[>]` in progress, `[ ]` planned, `[?]` needs a downstream use case or design decision.

This roadmap covers the remaining implicit, stiff, IMEX, stability, and nonlinear-validation work.

Adaptive timestep control *was* explicitly out of scope, on the grounds that it belongs to a problem-specific driving loop and only applies where a meaningful local error estimator is available. The second half of that reasoning has since expired: eight registered schemes now carry embedded estimators and `IntegrationResult.error` has no consumer at all. It is scoped as Phase 11 below, together with the `StepHistory`-invalidation problem that is its actual blocker.

## Current Baseline

- [x] Nonlinear-solver protocol with fixed Picard, relaxed Picard, and matrix-free JFNK.
- [x] JFNK is the default nonlinear closure for DIRK, Newmark, BDF, implicit Adams-Moulton, and IMEX Euler.
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

**Trigger (what would justify starting):** a downstream simulation that needs order-4
symplectic accuracy — e.g. long-run Hamiltonian training where the order-2
shadow-Hamiltonian drift of the registered Verlet/Forest-Ruth set is measurable and
costly — or L-stable Radau damping for stiff oscillatory modes that fall outside
BDF5's 51.84° A(α) cone. Until one of those appears, the block solver is machinery
with no user (the same caution as NOTES §3.8's "no solver contract without a
downstream"). The Phase 8 cost panels quantify what the block solver must beat:
ESDIRK6 already costs ~3× BE in RHS evaluations per step with per-stage solves.

- [ ] Create a block-state representation for all implicit stages, with flatten/unflatten over `s * N` unknowns.
- [ ] Extend JFNK residual construction to coupled stage systems without forming a dense block Jacobian.
- [ ] Implement Gauss-Legendre s=2 (order 4) first.
- [x] Validate direct symplectic-form behavior on oscillator and Kepler after tightening the default JFNK residual and finite-difference settings.
- [ ] Implement Radau IIA s=2 (order 3, L-stable) as the first high-quality fully implicit stiff method.
- [?] Consider Gauss-Legendre s=3 and Radau IIA s=3 only after the block solver and preconditioner are proven at scale.
- [?] Consider Lobatto IIIA-IIIB only for a concrete partitioned/separable Hamiltonian downstream.

Validation gate:

- [ ] Block residual and Jacobian-vector products agree with finite differences on small systems.
- [ ] Gauss-Legendre order 4 and Radau order 3 are demonstrated on nonlinear reference problems.
- [ ] No dense Jacobian allocation occurs for large state sizes.

## Phase 7: Alternative Stiff Families

**Trigger (what would justify starting):** a downstream RHS that splits cleanly into
a linear stiff operator with a cheap matrix-free action (a mass-lumped diffusion or
acoustic stiffness term, or the existing wave-equation Laplacian of the Phase 2
preconditioning work) plus a comparatively mild nonlinear remainder. Rosenbrock-W
trades the nonlinear solve for one *linear* solve per stage, so it only wins when
that linear solve is cheap (operator-based, preconditioned — the Phase 2
`preconditioner(v, state, context)` hook is the integration point) and the
nonlinearity is weak enough that a low-order method suffices. Exponential
integrators need the same split plus a `phi_k(hL)v` hook. The bar any candidate
must clear: Phase 8's measured cost baseline (BE 3.40, BDF2 2.96, TR-BDF2 4.50,
ESDIRK6 10.72 RHS evaluations/step; GMRES iterations/step equal to the stage-solve
count, at GMRES tolerance 1e-8).

### Rosenbrock / W methods

- [ ] Evaluate a linearly implicit Rosenbrock-W method as a lower-nonlinear-iteration alternative for diffusion-like SPH terms.
- [ ] Reuse JVP and preconditioner hooks, but introduce a dedicated linear-solve method interface rather than pretending it is a DIRK tableau.
- [ ] Start with a verified second- or third-order Rosenbrock-W method with an embedded estimator.

### Exponential integrators

- [?] Only pursue when a downstream exposes a natural linear stiff operator plus nonlinear remainder.
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

**Why this is no longer "out of scope".** This roadmap's header defers adaptive `dt` to
"a problem-specific driving loop", which was the right call when almost nothing had an
error estimator. Eight schemes now do — Bogacki-Shampine 3(2), Dormand-Prince 5(4),
Cash-Karp 5(4), TR-BDF2, both ESDIRKs and both ARKs — and `IntegrationResult.error` is
produced by all of them and **read by nothing**. The controller is the missing consumer,
not a missing estimator.

The genuinely hard part is not the controller. It is that `StepHistory` invalidates on
any `dt` change, so adaptive stepping and the entire multistep family (BDF1-5, AM2-4,
AB2-5, ABM2-4) are mutually exclusive today. That interaction is the real blocker and
should be decided before any controller lands.

- [ ] Add a PI (or predictive) step-size controller with the standard safety factor,
  min/max growth clamps, and a rejection path that re-runs the step.
- [ ] Decide the contract: does the driver own the loop, or does the library expose a
  `propose_dt(error, dt, order)` helper the caller drives? The latter matches this
  library's existing "no solver contract without a downstream" caution (NOTES §3.8).
- [ ] Decide what adaptive `dt` means for multistep. Options: refuse the combination
  explicitly (honest, cheap), or implement variable-step BDF/Adams coefficients
  (correct, substantially more work, and the route CVODE/LSODA take).
- [ ] Add dense output / interpolants, starting with Dormand-Prince's published
  formula. Needed for output at fixed times under a varying `dt`, and for event
  location. Nothing in the library interpolates within a step today.
- [?] **Error-norm convention.** `fields.state_norm`'s weighted-RMS ("< 1.0 means
  converged") is already the DIRK driver's convention; a controller should reuse it
  rather than introduce a second scale.

Validation gate:

- [ ] A stiff problem from the Phase 8 registry completes in materially fewer steps
  under control than at the fixed `dt` needed for the same final error.
- [ ] Step rejection actually triggers on van der Pol at mu = 10, where the Phase 8
  work already measured a hard explicit wall.
- [ ] Dense output reproduces the propagated solution at the step endpoints exactly and
  meets its advertised interpolation order in between.

## Phase 12: Families Not Yet Represented

Ordered by relevance to this library's SPH downstream, not by classical prominence.
Each is gated the same way Phases 6 and 7 are: a concrete downstream need, and a cost
comparison against the Phase 8 baseline.

### Stabilized explicit (RKC / RKL / ROCK) — the strongest omission

- [ ] Evaluate RKC or RKL2 ("super-time-stepping") for parabolic SPH terms. Explicit
  Chebyshev recursions whose real-axis stability grows as O(s²) in the stage count,
  matrix-free, with **no nonlinear solver at all**. They attack exactly the regime the
  Phase 2 preconditioning work and the `diffusion_problem(n)` benchmark exist for, and
  for SPH viscosity they are frequently the right answer over implicit.
- [ ] No tableau needed — a coefficient recursion plus a stage-count rule, so the
  marginal cost is close to the "tableau only" tier of NOTES §3.6.
- [ ] Benchmark against BE / BDF2 / TR-BDF2 on `diffusion_problem` at several `n`,
  reporting RHS evaluations to a fixed error.

### IMEX linear multistep (SBDF2/3, CNAB2)

- [ ] Add semi-implicit BDF and Crank-Nicolson/Adams-Bashforth. All three ingredients
  already exist — BDF coefficients, AB coefficients, and `IMEXRHS` — so SBDF2 is BDF2
  with an AB2 extrapolation of the explicit half. These dominate PDE and fluid codes,
  and Phase 5 currently jumps straight from IMEX Euler to two ARK pairs with nothing
  multistep in between.

### Generalized-alpha / HHT-alpha

- [ ] The natural sibling of the registered Newmark: adds controllable high-frequency
  numerical dissipation, and is close to a parameter generalization of the existing
  167-line `newmark.py`. Relevant to structural/solid SPH.

### Higher-order SSP

- [ ] SSPRK(5,4) and/or SSPRK(10,4). The shipped SSP/TVD set stops at order 3, which is
  precisely where the SSP barrier for explicit RK bites (order 4 needs 5+ stages).

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

- [ ] Measure the SSP coefficient of SSP RK3, TVD RK2 and TVD RK3 directly, and
  document the forward-Euler CFL each one preserves. Right now the defining property of
  these three schemes is asserted by their names alone.
- [ ] Add a hyperbolic benchmark problem. All nine current problems are dissipative or
  Hamiltonian ODEs; `diffusion_problem` is the only PDE semi-discretization and there is
  nothing hyperbolic. A scalar advection or inviscid Burgers semi-discretization would
  give the SSP schemes something to be right about.
- [ ] Add total-variation / positivity assertions on that problem, in the same
  "numerical assertion, not just a figure" style Phase 8 used for the implicit side.

Validation gate:

- [ ] Each SSP scheme's measured SSP coefficient matches its published value.
- [ ] A non-SSP scheme of the same order visibly violates the TVD bound on the same
  problem where the SSP schemes hold it — otherwise the test is not measuring the
  property it claims to.

## Housekeeping

- [ ] `midPoint` is imported in `integration.py` and never registered — dead import.
- [ ] `nonLagrangian` agrees with `dissipation` for every registered scheme, carries no
  information, and is documented as a removal candidate. Remove it, or give it a
  meaning.
- [ ] Phase 6 contains an orphan `[x]` item (symplectic-form validation) inside an
  otherwise unstarted phase; it belongs in Phase 8.
- [ ] The Current Baseline's JFNK-default list omits ARK, which does default to
  `JFNKSolver`.

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
10. [ ] Phase 13 explicit-side nonlinear stability — small, and it closes the one place where a registered scheme advertises a property nothing measures.
11. [ ] Phase 11 adaptive step control, once the multistep-vs-variable-`dt` contract is decided.
12. [ ] Phase 12 missing families, RKC/RKL first: it is the one candidate that directly challenges the Phase 8 cost baseline on a benchmark that already exists.
13. [ ] Phase 6 coupled implicit RK only when a high-order symplectic or Radau use case justifies the block solver.
14. [ ] Phase 7 Rosenbrock/exponential methods only when their downstream structure makes them competitive.

## Completion Definition

- [x] Each registered scheme has verified coefficients, correct metadata, a documented RHS/history contract, and targeted regression tests. (Coefficients: SUNDIALS ARKODE v7.9.0 for TR-BDF2/ESDIRK/ARK pairs, exact order conditions for BDF4-5 and AM2-4, Taylor-model + local-error verification (Phases 3-5); metadata/tables reconciled in Phase 0; per-family regression tests in `tests/`.)
- [x] Every stiff-method claim is backed by a nonlinear stiff benchmark and a stability diagnostic appropriate to that method family. (Phase 8: van der Pol, Robertson, diffusion, Prothero-Robinson in both signs, stiff damped oscillator; per-family diagnostics: Dahlquist regions, A(α) cones, L-damping assertions, damped-oscillator amplification matrices, two-parameter IMEX slices.)
- [x] Solver diagnostics make failed or stalled nonlinear solves observable to callers. (Phase 1: `SolveDiagnostics` in `IntegrationResult` — residual, GMRES iterations, RHS evaluations, line-search backtracks, termination reason — with finite-value checks and structured failure paths.)
- [x] New methods do not regress copied fields, stage times, history invalidation, gradients, or existing public APIs. (Copied fields: Phase 3 gate; stage times: `forced`-problem coverage; history invalidation: Phase 4 gate + `StepHistory` dt/uid guards; public API: full suite green after every phase. Gradient-through-step **was** the exception and is now covered — see Phase 9 below: the "differentiable by construction" claim was false for every implicit scheme, and `tests/test_gradients.py` (57 tests) now pins it. The warp gradient path remains open, NOTES §2.2.)
- [x] The full test suite passes in the `warp` environment. (2320 passed / 187 skipped as of 2026-09-10.)
