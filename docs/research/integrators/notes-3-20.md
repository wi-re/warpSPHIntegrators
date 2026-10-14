> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.20 Phase 12 follow-up: SSPRK(10,4), Shu's order-4 SSP method — done 2026-09-15

The bundled SSP/TVD set stopped at order 3 — precisely where the SSP barrier
for explicit RK bites (order 4 needs 5+ stages). This closes the gap with
Shu's 10-stage order-4 method (Shu, "Total variation diminishing Runge-Kutta
time discretizations", *J. Comput. Phys.* 169 (2001) 208–228, Sec. 4.2): a
5+5 stage cascade (rows 2–5 the full-triangular 1/6 cascade, rows 6–10 add
1/15 back-weights over the first five stages), full step c[-1] = 1, uniform
b = 1/10, ten RHS evaluations per step. It is the last of Phase 12's named
explicit families; SSPRK(5,4) (Butcher's 5-stage workhorse) stays gated.

**Coefficient verification.** The tableau is not in the SUNDIALS ARKODE v7.9.0
bundle (only the LS-RK I/O layer references SSPRK tableaus), so the
coefficients come from the paper and were verified empirically:

- Row sums equal c exactly.
- Order 4 measured on all three canonical problems: 4.005 (oscillator),
  4.000 (forced), 3.995 (Kepler) — the same registry-driven gate every other
  registered scheme passes (`tests/test_convergence.py`).
- **Exact SSP coefficient: 6.0.** In the upwind-advection convex-combination
  sense — the largest μ = dt·|λ| for which every stage map and the final map
  are convex combinations of the identity and the upwind shift. The registered
  scanner (`tvd_analysis.convex_combination_cfl`) reports 4.0 for this scheme
  (its grid stops there); exact shift-basis arithmetic (D = S − 1 on exact
  Fraction coefficients, scan to 100 plus bisection) pins the boundary at
  μ = 6: stage 2's constant coefficient 1 − μ/6 goes negative there, and no
  earlier stage or the final map breaks first.
- **Negative real-axis stability: [−13.916, 0]** (bisection to 1e-10 on the
  tenth-degree stability polynomial), consistent with OrdinaryDiffEq.jl's
  recorded 13.917 for SSPRK104.
- Both are pinned in `tests/test_tvd.py::test_ssprk104_ssp_boundary_and_real_axis_interval`.

**TVD verdict: (4.0, 5.0, True)** — "unconditionally TVD" up to the CFL cap
(the measured per-step TVD CFL hits the scanner's 5.0 cap; the certified SSP
coefficient from the stage maps is 4.0). Added to `EXPECTED_VERDICTS` in
`tests/test_tvd.py`; the registry-drift test enforces the entry's presence.

**First-stage reuse: order 3.** c[0] = 0 and c[-1] = 1, so the reuse analysis
applies: the a[-1] weights define an internal method of order q = 2, and the
stale k0 enters the update (b[0] = 1/10 ≠ 0), so reuse degrades 4 → 3. One
subtlety measured and pinned: the a[-1] weights satisfy the
stability-polynomial third-order condition w·(a·c) = 1/6 *exactly* — they
fail only the node-value condition w·c² = 11/36 ≠ 1/3. On the linear
autonomous oscillator the stability polynomial is all that matters, so the
stale k0 sits O(dt⁵) away and reuse measures a full 4.00 there; on the forced
non-autonomous problem (3.00) and Kepler (2.98) the node-value defect is
visible and the order is 3. `test_step_reuse.py` records this as a documented
exception (`REUSE_EXACT_ON_LINEAR_AUTONOMOUS`) rather than a prediction
change: 3 is the guaranteed order, so the degrading-reuse warning is not
needless.

**Cost and niche.** Ten RHS evaluations per step — the most expensive
fixed-cost registered explicit scheme. The order-4 SSP payoff is the largest
certified SSP coefficient among the bundled order-4 methods (6.0, versus 2/3
for RK4 and 5/12 for Cash-Karp 5(4)) plus the −13.9 real-axis interval; it
is explicit, so there is nothing to solve. On the viscous Burgers demo it
sits in the §5 explicit panel and the §10 convergence ladder (order 4
measured on the PDE, Gaussian IC), and the §11 cost table prices it: 800
work units at `dt = 5e-3`, `T = 0.4` — off the Pareto front there, where
its niche (order-4 TVD for hyperbolic problems) is not the point.
