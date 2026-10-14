> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.11 Phase 13: TVD / SSP classification of the registered schemes — done 2026-09-10

The roadmap's Phase 13 (nonlinear stability for the explicit side) is complete.
`stability.py` answers the parabolic question — "does this scheme stay bounded on
`y' = λy`?"; this phase measures the hyperbolic one — "does one step increase
total variation?" — for **every** registered scheme, not just the TVD/SSP-named
ones, whose defining property was previously asserted by name alone.

What landed:

- **Two hyperbolic problem factories in `testing.py`** (the `PROBLEMS` registry
  now has eleven). `advection_problem(n, c, L, ic)` is 1D periodic first-order
  upwind advection `u_t + c u_x = 0`, `(Au)_i = c(u_{i-1} − u_i)/h`: the model
  problem for the classifier, because every Fourier mode is an exact eigenmode
  with eigenvalue `(c/h)(e^{−2πim/n} − 1)` and every RK step is a circulant map
  on the grid. `ic='mode'` starts on two eigenmodes (exact semi-discrete
  solution known); `ic='step'` is a periodic square wave (TV = 4) for the
  total-variation checks. `burgers_problem(n)` is semi-discrete inviscid Burgers
  with local Lax-Friedrichs fluxes on the smooth entropy wave `−sin(2πx)` —
  the *nonlinear* hyperbolic problem.
- **`tvd_analysis.py`** — the general TVD classifier, with two measurements:
  1. `convex_combination_cfl` (the **SSP coefficient** `r`): the stage-value
     maps and the final map of the scheme's tableau, evaluated on the upwind
     model over all `n` modes at once (`stage_and_final_maps`), must be convex
     combinations of periodic shifts (elementwise non-negative circulant first
     row, row sum 1) for every `μ' ≤ μ`. A convex-combination step is TVD by
     construction, so `r` is a *rigorous* TVD CFL for every scheme that
     exposes its tableau (explicit RK, DIRK; ARK in its pure-implicit limit;
     the hand-written TVD RK2/3 through their algebraically identical Butcher
     equivalents, pinned to the actual drivers by `tests/test_tvd.py`).
  2. `measure_tvd_cfl` (the **measured per-step TVD CFL**): run the registered
     driver on the step initial condition and require the per-step total-
     variation increase to stay within `1e-8 · TV0` over a post-burn-in window
     (burn-in 10, window 40 — the multistep cold-start TV jump is shared by
     all self-starting multistep schemes and is not a property of the step
     map). This one sees the actual driver, so it also classifies the
     schemes that expose no tableau (BDF/Adams family, semi-implicit Euler).
- **`benchmarks/tvd_classifier.py` → the verdict table below** — the maintained
  fact; re-run the script to refresh it (the numbers are measured, not
  asserted by name).
- **`tests/test_tvd.py`** — 48 tests: the published `r = 1` of the TVD-named
  schemes, the landmark coefficients, the TVD-driver-vs-tableau cross-check,
  per-step TV / positivity / Burgers assertions at CFL 1, the stage-level
  separation (below), and the full-registry verdict table.

The measured landscape (n = 64, step IC, CFL grid 0.25 … 5, convex-combination
scan to CFL 4):

| scheme | SSP r | measured TVD CFL |
| --- | --- | --- |
| Forward Euler, Midpoint, Heun 2, Ralston 2, Heun 3, Ralston 3, Wray 3, SSP RK3, Bogacki-Shampine, EPEC, EPEC Modified, **TVD RK2, TVD RK3** | 1 | 1 |
| RK3 | 0.5 | 1 |
| RK4 | 2/3 | 1.25 |
| RK4 (alternative) | 1/3 | 1.25 |
| Cash-Karp 5(4) | 5/12 | 1.5 |
| **Nystrom 5th order, Dormand-Prince 5(4)** | **0** | 1.5 |
| **SSPRK(10,4)** | 4 (scan cap; exact 6) | 5 (tested limit) |
| **SSPRK53, SSPRK63, SSPRK73, SSPRK83, SSPRK54, KYKSSPRK42, KYK2014DGSSPRK_3S2** (§3.21) | 2.651 / 3.518 / 4 (exact 4.2879) / 4 (exact 5.1071) / 1.769 / 2.284 / 1.894 | 2 / 3 / 3 / 5 / 2 / 2 / 1.5 |
| **SSPRK43, SSPRK432, SSPRK932** (§3.22) | 2.0 / 2.0 / 4 (scan cap; exact 6.0) | 2 / 2 / 5 (SSPRK932 unconditional) |
| Backward Euler (implicit) | unconditional (Fejer kernel) | 5 (tested limit) |
| Implicit Midpoint, Trapezoidal | 2 | 5 (tested limit) |
| SDIRK2, TR-BDF2 | 1 + √2 | 5 (tested limit) |
| ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA, ARK3(2)4L[2]SA, ARK4(3)6L[2]SA | 0 | 5 (tested limit) |
| BDF1, IMEX Euler, Semi-Implicit Euler | (no tableau) | 5 (tested limit) |
| BDF2-5, Adams-Bashforth 2-5, ABM 2-4 (PECE), Adams-Moulton 2-4 (implicit), SSPRKMSVS32/43 (§3.23) | (no tableau) | 1.5 |
| Leap Frog, Symplectic Euler, Velocity Verlet, PEFRL, VEFRL, Newmark | n/a — second-order-system schemes; the TVD question does not apply | |

Findings worth keeping:

- **TVD is strictly broader than SSP, and the registry shows it.** RK4's
  convex-combination limit is `r = 2/3` (stage 4's circulant entry
  `(μ²/4)(2 − 3μ)`), yet its *per-step* TVD CFL measures 1.25: the final map
  (the shared fourth-order stability function) is TVD past the point where the
  internal stages leave the convex hull. Same pattern, stronger: Nystrom 5th
  order and Dormand-Prince 5(4) have `r = 0` — not a convex-combination step
  at *any* CFL — yet are per-step TVD up to CFL 1.5.
- **Per-step TV cannot separate same-order schemes; the stage level can.**
  Classical RK3 at CFL 1 produces *zero* per-step TV increase on the step
  advection and on Burgers, identical to TVD RK3 — all third-order RK methods
  share the same final map, which is a convex combination up to CFL 1. The
  separation is at the stage level: RK3's stage-3 map is `1 + ζ + ζ²`, whose
  circulant row at CFL 1 is `[1, −1, 1]` (entry `μ(1 − 2μ) = −1`), while
  TVD RK3's stage rows are `[0, 1]` and `[3/4, 0, 1/4]`. The test runs the
  registered drivers from a delta initial condition, so each stage state *is*
  its circulant row (`tests/test_tvd.py::test_classical_rk3_stage_leaves_the_convex_hull`).
- **Nystrom 5th order, Dormand-Prince, ESDIRK3/4 (and the ARK limits) have
  `r = 0` exactly, not "tiny".** For the explicit ones this was checked by
  exact rational arithmetic: Nystrom's stage-5 entry at circulant offset 3 is
  `7.995e-18·μ³ − 0.1185·μ⁴` (negative for every `μ` above `~1e-16` — the
  method's notoriously ill-scaled coefficients), and Dormand-Prince's stage-5
  and stage-6 entries at offset 4 have lowest terms `−0.0465·μ⁴` /
  `−0.0509·μ⁴` (negative for all `μ > 0`). A naive tol-based scan reports
  where the (genuinely negative) entry's magnitude happens to cross the
  tolerance (0.0054 for Nystrom), which looks like a small-but-real
  coefficient; `convex_combination_cfl` now probes below the candidate
  boundary (a genuine zero-crossing grows in magnitude as the probe moves
  away from the root, an everywhere-negative entry shrinks like `(1/2)^d` per
  halving) and returns `0.0` in the latter case. Cash-Karp's `5/12` is a
  genuine crossing and survives the probe.
- **Implicit Midpoint and Trapezoidal are *not* unconditionally TVD — both
  have `r = 2`.** Their stage map is the unconditionally positive Poisson
  kernel, but the propagated map is the shared Cayley factor
  `(1 + ζ/2)/(1 − ζ/2)`, a convex combination only for `μ ≤ 2`. The earlier
  "implicit midpoint is unconditionally TVD" reading of its A-stability was
  wrong: A-stability bounds `|R(z)|`, TVD requires the convex-combination
  structure, and the final map is what propagates.
- **The L-stable implicits are TVD to the top of the sweep** (CFL 5): BE
  (unconditionally, by the Fejer kernel), SDIRK2 / TR-BDF2 (whose `r = 1+√2`
  bounds the stages, not the final map), and the ESDIRK/ARK pairs — whose
  stages leave the convex hull immediately (`r = 0`) but whose stiff-accurate
  final maps damp so strongly that no per-step TV increase is measurable.
- **The multistep family is TVD to a measured CFL of 1.5** (BDF2-5, AB2-5,
  ABM 2-4 PECE, implicit AM2-4), diverging by CFL 2 (per-step TV increase
  `~4e25 · TV0` — the solution itself leaves the O(1) band). BDF1 = BE is
  unconditional. The measured cold-start TV jump (`+1.333e-2 · TV0`, shared by
  every self-starting multistep scheme and gone after five steps) is why the
  sweep uses a burn-in.
