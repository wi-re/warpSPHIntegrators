> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.17 Adaptive step control and dense output (roadmap Phase 11)

Phase 11 lands as **helpers, not a solver** — the decision the roadmap's open
contract item asked for. The library exposes two functions in
`src/warpSPHIntegrators/adaptive.py`, and the caller drives the accept/reject loop:

- `estimate_error_norm(result, rtol=1e-3, atol=1e-6)` — the embedded-pair
  difference as a dimensionless number: `state_norm(result.error,
  reference=result.state)`, the weighted-RMS convention the DIRK driver already
  uses ("< 1.0 means at the tolerance") — the roadmap's [?] error-norm item,
  resolved by reusing `fields.state_norm` rather than introducing a second
  scale. Raises a `ValueError` for a scheme that emits no estimate.
- `propose_dt(error_norm, dt, order, *, target=1.0, safety=0.9, growth_min=0.2,
  growth_max=5.0)` — the classic predictive controller,
  `dt * clamp(safety * (target/error_norm)**(1/order), growth_min,
  growth_max)`; a zero error proposes `dt * growth_max`. The controller itself
  never knows whether a step was rejected; "never grow on a rejection" is a
  property of the *loop* (the SciPy convention) and the demo loops apply it.

The demo loop — step, measure, accept/reject, propose; ~15 lines — lives in
`benchmarks/adaptive_benchmark.py::run_adaptive` and
`tests/test_adaptive.py::run_adaptive`, deliberately not in the library.

**Why helpers, not a driver.** §3.8's "no solver contract without a downstream"
caution: an adaptive driver would own the loop, the output times, and the event
handling, and none of that has a downstream here yet. The helpers are useful
on their own, and the loop they leave to the caller is small.

**The estimate emitters and their `q`.** Ten registered schemes emit
`IntegrationResult.error` (the roadmap's "eight" predates ROS3P and EXPRB32):
the (p, p−1) embedded pairs Bogacki-Shampine 3(2), Dormand-Prince 5(4), and
Cash-Karp 5(4); the SUNDIALS ARKODE v7.9.0 built-in estimates TR-BDF2,
ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA, ARK3(2)4L[2]SA, and ARK4(3)6L[2]SA; and
ROS3P and EXPRB32's published embedded pairs. The controller needs the power
`q` that the estimate scales as `h^q`: `q = scheme.order` for nine of the ten,
and **`q = order + 1 = 3` for TR-BDF2** — its published SUNDIALS pair (both
branches order 2) makes the estimate the propagated branch's true local error,
`O(h^3)`. The `estimate_order` helper in the two demo loops pins the exception.

**Multistep: refused explicitly.** Adaptive `dt` and the multistep family are
mutually exclusive, by design rather than accident: `StepHistory` restarts on
any `dt` change (pinned in
`tests/test_groundwork.py::test_step_history_restarts_on_dt_change`), so every
step of a variable-`dt` run would throw away the history a multistep scheme
exists to build — and the family emits no estimate to drive a controller in the
first place, so `estimate_error_norm` raises for it. Variable-step BDF/Adams
coefficients (the CVODE/LSODA route) remain the future work if a downstream
asks for it.

**Dense output: Shampine's 1986 quartic.** DP5's published continuous extension
(Shampine, "Some Practical Runge-Kutta Formulas", *Mathematics of Computation*
46(173) 1986, 135–150 — the optimum-`c_6` variant, as implemented in SciPy's
`RK45`) is

    y(θ) = y_n + h · Σ_{i=1..7} P_i(θ) k_i,   P_i(θ) = p₁θ + p₂θ² + p₃θ³ + p₄θ⁴

(no constant term; the `7 × 4` coefficient matrix `_DP5_DENSE_P` is transcribed
verbatim from `scipy/integrate/_ivp/rk.py`, verified against the local SciPy
1.18.0 source). The stage mapping is 1:1 in order with `RungeKuttaB`'s
`k1..k7` (SciPy's `K[i]` is the repo's `k_{i+1}`; `k7` is the FSAL stage at
`c = 1`, whose row is `b` itself). Verified:

- `P(1) == b` (max diff 7e-16) and `P(0) == 0`, so the interpolant is **exact
  at both step endpoints** — `θ = 1` re-applies the propagated weights through
  `butcher._weighted_update` and is bit-for-bit the step's returned state, and
  `θ = 0` returns the input state itself;
- the continuous order conditions hold **through order 4 only** (an 8-point
  Vandermonde fit of the extension on a linear problem matches `(θz)^m/m!` for
  `m ≤ 4` and deviates at `m = 5` by ~1.8e-4), so the interior interpolation
  error is `O(dt^5)`.

Measured on the linear oscillator (one step, queried at `θ = 0.4`, against the
analytic solution): rates 4.832 / 4.961 / 4.990 / 4.998 / 4.999 as `dt` halves
from 0.4 to 0.0125 — fifth-order interior accuracy, one order below the
propagator, exactly what the order conditions predict. Dense output is
**DP5-only**: the other nine emitters have no published continuous extension,
and `dormand_prince_dense_output` refuses their stage vectors (a seven-stage
check) rather than guess.

**The gate (van der Pol, μ = 10, T = 5, DP5, dt0 = 0.1** — deliberately far
outside the Phase 8 explicit wall, fast eigenvalue ~ μ(1 + x²) = O(10–20)):

| run | endpoint error | accepted steps |
|---|---|---|
| reference floor (dt = 5e-4 vs 2.5e-4) | 8.8e-14 | — |
| fixed dt = 0.016 | 1.3e-06 | 312 |
| fixed dt = 0.008 | 2.0e-08 | 625 |
| adaptive rtol = 1e-4 | 7.2e-05 | 54 (18 rejected) |
| adaptive rtol = 1e-5 | 3.7e-06 | 75 (17 rejected) |
| adaptive rtol = 1e-6 | 3.0e-08 | 107 (19 rejected) |

Both roadmap gates hold. Step rejection triggers where the wall is (17–19
rejections out of 72–126 attempts), and the stiff problem completes in
materially fewer steps under control than at the fixed `dt` needed for the same
final error: 75 vs 312 (4.2×) at rtol = 1e-5, and 107 vs 625 (5.8×) at rtol =
1e-6. The `tests/test_adaptive.py` gate pins the former with a margin
(`accepted < 0.75 ×` the coarsest matching fixed step count). No fixed-ladder
point diverged; dt = 0.016 is already deep inside the wall.

**New tests / artifacts:** 24 in `tests/test_adaptive.py` (controller math,
growth clamps, the zero-error and `target` paths, argument validation; the norm
convention and the `ValueError` for estimate-less schemes; finite-positive
norms on all ten emitters; the loop contract on the oscillator — accepted steps
at the target, rejections occur, the endpoint reached; the van der Pol gate;
dense-output endpoint exactness, order-4 interior rates, the half-step
cross-check, and the validation errors); `benchmarks/adaptive_benchmark.py` →
`docs/assets/adaptive_benchmark.png` (error vs step count on the ladder, the
controller in flight, the interior interpolation rate).

---
