# Implicit and Stiff Integration Roadmap

Status markers: `[x]` complete, `[>]` in progress, `[ ]` planned, `[?]` needs a downstream use case or design decision.

This roadmap covers the remaining implicit, stiff, IMEX, stability, and nonlinear-validation work. Adaptive timestep control is explicitly out of scope: it belongs to a problem-specific driving loop and only applies where a meaningful local error estimator is available.

## Current Baseline

- [x] Nonlinear-solver protocol with fixed Picard, relaxed Picard, and matrix-free JFNK.
- [x] JFNK is the default nonlinear closure for DIRK, Newmark, BDF, and IMEX Euler.
- [x] DIRK: Backward Euler, implicit midpoint, trapezoidal / Crank-Nicolson, SDIRK2, and TR-BDF2.
- [x] Newmark average-acceleration and linear-acceleration variants.
- [x] BDF1-BDF3 with state-bearing `StepHistory` and a safe Dormand-Prince cold start.
- [x] IMEX Euler with explicit `IMEXRHS(explicit=..., implicit=...)` callbacks; an ordinary RHS remains fully implicit.
- [x] Dahlquist stability-region plots and numerical boundary checks for tableau methods and BDF1-BDF3.
- [x] Nonlinear Kepler convergence coverage and nonlinear stiff relaxation coverage.

## Phase 0: Keep the Public Story Accurate

- [x] Update `README.md` Known Limitations to say BDF1/2 and IMEX Euler are available.
- [x] Update `NOTES.md` status and scheme tables to replace old Picard-default claims with JFNK-default behavior.
- [ ] Document the JFNK cost model: residual evaluations, GMRES matvecs, finite-difference versus exact JVP, and the need for preconditioning at scale.
- [x] Document which methods are A-stable, L-stable, symplectic, or only conditionally stable; distinguish the exact scheme from a truncated Picard override.
- [x] Add a concise supported-scheme table with conservative dynamics, stiff dissipation, split stiff terms, and long smooth stiff integrations.

Validation gate:

- [ ] README examples execute in the `warp` environment.
- [x] Scheme names, registry metadata, and documentation tables agree.

## Phase 1: Solver Observability and Robustness

- [x] Extend `SolveResult` with optional diagnostics: nonlinear residual, GMRES iterations, total RHS evaluations, and termination reason.
- [x] Define termination reasons: converged tolerance, stagnation floor, maximum Newton iterations, maximum GMRES iterations, invalid residual.
- [x] Add finite-value checks for nonlinear residuals before accepting a Newton update.
- [ ] Add optional line search / damping for JFNK Newton corrections when a full correction increases the nonlinear residual.
- [ ] Add a reusable solver-options dataclass or typed dictionary, while preserving current keyword compatibility.
- [x] Record diagnostics in `IntegrationResult` without changing existing `stages` semantics.
- [ ] Add tests for Newton convergence, stagnation, GMRES exhaustion, NaN/Inf detection, and line-search recovery.

Validation gate:

- [x] Existing JFNK stiff oscillator remains correct.
- [x] Failure paths report structured diagnostics instead of silently returning a poor iterate.
- [x] Full test suite remains green.

## Phase 2: Preconditioned JFNK

- [ ] Add an optional `preconditioner(v, state, context) -> vector` hook to JFNK.
- [ ] Upgrade GMRES to apply right or left preconditioning consistently and expose the selected mode.
- [ ] Provide an identity preconditioner baseline and preserve current behavior when none is supplied.
- [ ] Add diagonal / block-diagonal examples suitable for diffusion and relaxation terms.
- [ ] Add a sparse or operator-based SPH preconditioner integration point; do not form dense Jacobians.
- [ ] Measure residual evaluations and Krylov iterations as problem size grows.
- [ ] Add benchmark plots comparing unpreconditioned and preconditioned JFNK.

Validation gate:

- [ ] Preconditioned and unpreconditioned solutions agree on linear reference problems.
- [ ] A representative stiff diffusion/relaxation problem uses materially fewer GMRES iterations with a supplied preconditioner.
- [ ] The no-preconditioner path remains bitwise or tolerance-equivalent to the current solver.

## Phase 3: Higher-Quality Sequential DIRK

### TR-BDF2

- [x] Verify the TR-BDF2 SDIRK coefficients and independently check the second-order condition.
- [x] Add the three-stage tableau to `getDIRKTableau`.
- [ ] Add an embedded estimator, returning `IntegrationResult.error`.
- [x] Register order 2, L-stable, stiffly accurate metadata.
- [x] Test convergence through the generic oscillator, forced, damped, and Kepler suites plus nonlinear stiff relaxation.
- [ ] Add a method-specific L-stable damping test on the negative-real-axis scalar problem.

### ESDIRK3(2)4L[2]SA and ESDIRK4(3)6L[2]SA

- [ ] Verify coefficients and named variants from Kennedy-Carpenter or the original source.
- [ ] Extend the tableau model only if needed for explicit first stages and embedded weights.
- [ ] Register both schemes with accurate stage count, stability, stiff-accuracy, and error-estimator metadata.
- [ ] Add convergence and embedded-error tests on autonomous, non-autonomous, nonlinear, and stiff problems.
- [ ] Add stability-region overlays for all new DIRK methods.

Validation gate:

- [ ] Every new tableau passes hand-checked consistency/order conditions and empirical convergence tests.
- [ ] JFNK preserves copied-field lifecycle through every implicit stage.
- [ ] Stability plots match the advertised A/L-stability class numerically.

## Phase 4: Higher-Order BDF and True Adams-Moulton

### BDF3-BDF5

- [>] Generalize `bdf.py` to coefficient tables for orders 1-5. BDF3 is registered and verified; BDF4/5 need non-autonomous convergence work before registration.
- [ ] Extend `StepHistory` requirements and preserve state snapshots for all required previous states.
- [ ] Keep the safe high-order starter when history is insufficient or invalidated by timestep / particle identity changes.
- [>] Register BDF3-BDF5 with correct A(alpha) metadata, not A-stable metadata. BDF3 is registered as A(alpha).
- [ ] Add root-condition and numerical stability-boundary tests.
- [ ] Test order on oscillator, forced, damped, Kepler, and nonlinear stiff relaxation.

### True implicit Adams-Moulton 2-4

- [ ] Add a JFNK residual using known history derivatives plus the unknown endpoint derivative.
- [ ] Keep existing ABM PECE methods separate: predictor-corrector and fully implicit Adams-Moulton are different contracts.
- [ ] Add an optional predictor using the matching Adams-Bashforth formula.
- [ ] Register and test AM2-AM4 with explicit history requirements and startup behavior.

Validation gate:

- [ ] BDF3-BDF5 achieve their claimed order with threaded history.
- [ ] History resets safely on `dt` or particle-set changes.
- [ ] AM methods converge their nonlinear corrector and outperform or match PECE on a stiff nonlinear case.

## Phase 5: Higher-Order IMEX / Additive RK

- [ ] Define an additive-tableau representation with explicit and diagonally-implicit coefficients, stage times, propagated weights, and embedded weights.
- [ ] Generalize IMEX Euler into an additive RK driver using `IMEXRHS`.
- [ ] Preserve the compatibility rule: a normal RHS callable means fully implicit; only `IMEXRHS` activates a split.
- [ ] Decide and document copied-field semantics when explicit and implicit callbacks each preprocess a stage.
- [ ] Implement and verify ARK3(2)4L[2]SA.
- [ ] Implement and verify ARK4(3)6L[2]SA.
- [ ] Add split test problems: explicit transport plus implicit linear decay/diffusion; nonlinear explicit forcing plus implicit relaxation.
- [ ] Add two-parameter IMEX stability plots over `(z_explicit, z_implicit)` slices.
- [ ] Measure RHS cost: explicit evaluations, implicit residual evaluations, and GMRES iterations.

Validation gate:

- [ ] Pure-explicit and pure-implicit limits recover the corresponding component methods where the tableau guarantees it.
- [ ] Split linear test equations meet claimed order and remain stable in their published IMEX region.
- [ ] Existing ordinary RHS behavior remains fully implicit and cannot silently discard a split term.

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
- [ ] Extend the oscillator stability plots over damping ratio.
- [x] Add direct finite-difference phase-area checks across every registered scheme and nonlinear Kepler symplectic-form checks for the genuinely symplectic subset.
- [ ] Add two-parameter IMEX stability slices for IMEX Euler and each later ARK method.
- [ ] Add Prothero-Robinson with configurable smooth forcing and stiffness parameter.
- [ ] Add stiff van der Pol with a moderate parameter for deterministic CI and a large parameter for an opt-in benchmark.
- [ ] Add Robertson or Oregonator kinetics as a positive, nonlinear stiff chemistry benchmark.
- [ ] Add a semi-discrete diffusion or reaction-diffusion benchmark with a known spectral stiffness scale.
- [ ] Add a stiff damped oscillator separating high frequency from true dissipative stiffness.
- [ ] Produce figures comparing solution error, energy/dissipation, nonlinear iterations, Krylov iterations, and RHS cost.

Validation gate:

- [ ] CI tests remain small and deterministic.
- [ ] Longer benchmark scripts produce versioned PNGs under `images/` and clearly state parameter values and solver settings.
- [ ] Claimed stability advantages are demonstrated on a problem where the explicit step restriction is actually active.

## Recommended Execution Order

1. [ ] Phase 0 documentation reconciliation.
2. [ ] Phase 1 solver diagnostics and failure handling.
3. [ ] Phase 3 TR-BDF2, then ESDIRK3.
4. [ ] Phase 5 ARK3 IMEX after the existing `IMEXRHS` API is stress-tested.
5. [ ] Phase 2 preconditioning using the first real SPH diffusion/acoustic downstream.
6. [ ] Phase 4 BDF3-BDF5 and true Adams-Moulton.
7. [ ] Phase 8 broadened nonlinear/stiff benchmark and stability suite throughout.
8. [ ] Phase 6 coupled implicit RK only when a high-order symplectic or Radau use case justifies the block solver.
9. [ ] Phase 7 Rosenbrock/exponential methods only when their downstream structure makes them competitive.

## Completion Definition

- [ ] Each registered scheme has verified coefficients, correct metadata, a documented RHS/history contract, and targeted regression tests.
- [ ] Every stiff-method claim is backed by a nonlinear stiff benchmark and a stability diagnostic appropriate to that method family.
- [ ] Solver diagnostics make failed or stalled nonlinear solves observable to callers.
- [ ] New methods do not regress copied fields, stage times, history invalidation, gradients, or existing public APIs.
- [ ] The full test suite passes in the `warp` environment.
