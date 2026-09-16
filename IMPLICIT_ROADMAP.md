# Implicit and Stiff Integration Roadmap

Status markers: `[x]` complete, `[>]` in progress, `[ ]` planned, `[?]` needs a downstream use case or design decision.

**All 15 phases of this roadmap are complete** (landed 2026-09-09 → 2026-09-15;
index at the bottom). This document now tracks only the remaining work — the gated
and deferred items, each with the trigger that would activate it. The per-phase
record (coefficients, measurements, decisions, validation gates) lives in NOTES.md
and the git history; the pre-2026-09-15 text of this file remains in git.

## Remaining work

### Higher-order BDF (Phase 4)

- [?] **BDF6.** The A(α) cone shrinks below BDF5's 51.84°, so order 6 buys accuracy
  only while losing the stiff sector. **Trigger:** a downstream stiff problem that
  needs order ≥ 6 *and* whose stiff modes stay inside the narrower cone.
- [?] **BDF3-5 cold-start ceiling.** The Dormand-Prince bootstrap is explicit with a
  negative-real-axis boundary at |z| ≈ 3.3, so BDF3-5 cannot *start* at
  `dt·rate ≳ 3.3` on the stiff relaxation (the rate=100 case is out of reach for
  them — see the `tests/test_stiff.py` docstring). Options: document the per-order
  usable-rate ceiling, or add a JFNK-based startup (e.g. a BE or AM2 warm-up).
  **Trigger:** a downstream stiff problem where the ceiling binds.

### Coupled fully implicit RK (Phase 6)

- [?] **Gauss-Legendre s=3 and Radau IIA s=3.** Consider only after the block solver
  and preconditioner are proven at scale. **Trigger:** a downstream that needs the
  s=3 order/stability points *and* the `s·N`-unknown block solve clears the GMRES
  cost bar at that size (the s=2 scale gate at 256 DOF = 1536 unknowns is the
  current reference; NOTES §3.18 has the spectrum finding).
- [?] **Lobatto IIIA-IIIB.** Only for a concrete partitioned/separable Hamiltonian
  downstream — it needs the component partition (`f_q` / `f_p` accessors, which
  Phase 14 explicitly left out of scope), not this additive block solve.

### Gradients (Phase 9)

- [?] **Warp gradient path.** Still the open `torch.autograd` vs `wp.Tape` decision
  of NOTES §2.2; Phase 9 settled the torch half only. The adjoint solve needs
  *reverse*-mode VJPs, which warp has — so the implicit-differentiation design
  carries over to a warp backend more cleanly than an unrolled one would have.
- [?] **Gradients through the explicit cold start.** BDF/AM gradients currently flow
  through the Dormand-Prince bootstrap by unrolling it, which is correct but not
  free. **Trigger:** a downstream differentiates long multistep runs where the
  startup cost is measurable.

### First-stage reuse (Phase 10)

- [?] **Warm-starting implicit first stages.** For SDIRK2 and any other
  non-explicit-first-stage tableau, a reused `k` is still a better Newton *initial
  guess* than the current cold start. That is a softer, iteration-count win rather
  than an evaluation-count one; measure before building.

### Families not yet represented (Phase 12)

- [ ] **Generalized-alpha / HHT-alpha.** The natural sibling of the registered
  Newmark: adds controllable high-frequency numerical dissipation, and is close to
  a parameter generalization of the existing `newmark.py`. Relevant to
  structural/solid SPH. Gated on a concrete downstream need plus a cost comparison
  against the Phase 8 baseline.
- [ ] **SSPRK(5,4).** Butcher's 5-stage order-4 SSP method (Butcher 1987 / Shu
  2002) is the standard order-4 SSP workhorse. Still unimplemented — and not in the
  SUNDIALS ARKODE v7.9.0 bundle (only the LS-RK I/O layer references SSPRK
  tableaus), so the coefficients would come from the paper. Its niche overlaps the
  shipped SSPRK(10,4) (order 4, exact SSP coefficient 6.0, real-axis interval
  [−13.916, 0]): 5 evaluations per step instead of 10, at the cost of the smaller
  SSP coefficient.
- [?] **High-order symplectic composition.** Yoshida 4/6/8, Suzuki, Blanes-Moan, or
  a general `compose()` helper over the existing Verlet base. **Gated on a
  separable Hamiltonian downstream**, and that gate is real: NOTES already measures
  the whole Verlet/Forest-Ruth family dropping to first order under a
  velocity-dependent force, which is every actual SPH momentum equation. (For a
  *separable* Hamiltonian, composition reaches order 6-8 explicitly and far more
  cheaply than Gauss-Legendre s=3.)
- [?] **One-step Runge-Kutta-Nystrom.** NOTES §3.6 lists *multistep* Nystrom
  (Stormer-Cowell) and Gauss-Jackson as not-done, but not one-step RKN. Same
  separability caveat as the composition methods.

## Completed phases (index)

| Phase | Content | Landed | Record |
|---|---|---|---|
| 0 | Documentation reconciliation (scheme tables, JFNK-default story) | 2026-09-09 | README, NOTES §3.4 |
| 1 | Solver observability — `SolveDiagnostics` in `IntegrationResult`, termination reasons, line search | 2026-09-09 | NOTES §3.4, `tests/test_jfnk.py` |
| 2 | Preconditioned JFNK — left/right-preconditioned GMRES, `preconditioner(v, state, context)` hook, identity/diagonal helpers | 2026-09-09 | NOTES §3.4, `images/jfnk_preconditioner_benchmark.png` |
| 3 | TR-BDF2, ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA (SUNDIALS ARKODE coefficients + embedded pairs) | 2026-09-09 | NOTES §3.6 |
| 4 | BDF3-BDF5 (A(α) cones measured), JFNK-corrected Adams-Moulton AM2-AM4 | 2026-09-09 | NOTES §3.6, §3.7 |
| 5 | Additive IMEX RK — ARK3(2)4L[2]SA, ARK4(3)6L[2]SA, two-parameter stability slices | 2026-09-09 | NOTES §3.6 |
| 6 | Coupled fully implicit RK — Gauss-Legendre 2, Radau IIA s=2 (`fullyimplicit.py`, `fields.BlockState`) | 2026-09-15 | NOTES §3.18, `images/blockrk_benchmark.png` |
| 7 | Rosenbrock-W (ROS3P) + exponential integrators (ETD2RK, EXPRB32), family cost gate | 2026-09-11/12/13 | NOTES §3.14–3.16 |
| 8 | Nonlinear/stiff benchmark + stability suite (five problem factories, amplification matrices) | 2026-09-09 | NOTES §3.10, `images/stiff_benchmark_suite.png` |
| 9 | Gradients through every implicit scheme (implicit-function-theorem re-attachment) | 2026-09-10 | NOTES §2.2, `tests/test_gradients.py` |
| 10 | First-stage reuse for the stiffly accurate DIRK tableaus | 2026-09-10 | NOTES §3.6 (Phase 10 note), `tests/test_dirk.py` |
| 11 | Adaptive step control as helpers + DP5 dense output (multistep refused explicitly) | 2026-09-14 | NOTES §3.17, `images/adaptive_benchmark.png` |
| 12 | Missing families — RKC/RKL (09-11); IMEX linear multistep SBDF2/SBDF3/CNAB2 and SSPRK(10,4) (09-15) | 2026-09-11/15 | NOTES §3.13, §3.19, §3.20 |
| 13 | Explicit-side nonlinear stability — advection/Burgers problems, TVD/SSP classifier over the whole registry | 2026-09-10 | NOTES §3.11, `scripts/tvd_classifier.py` |
| 14 | Structured RHS interface (`RHS` / `IMEXRHS` / `SemilinearRHS`), viscous-Burgers semilinear benchmark | 2026-09-10 | NOTES §3.12, `tests/test_rhs.py` |

Plus the four roadmap housekeeping items (2026-09-15): NOTES §2.14.
