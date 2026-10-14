> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.24 Scheme-expansion wave 4: pRRK22/33/54 -- parametric-relaxation SSPRK -- done 2026-10-13

Liu, Burchard, Kuzmin & Shang, "Parametric relaxation Runge-Kutta methods"
(2023). ODE source: `lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl`
(`_prrk{22,33,54}_coeffs` + the three `perform_step!`) and
`ssprk_caches.jl` (`pRRK{22,33,54}ConstantCache` base Shu-Osher coefficients).

**Mechanism.** pRRK starts from a Shu-Osher SSPRK method and applies a
per-step *relaxation* with parameter `z = kappa*dt`:
`psi_1 = 1 + z*beta_10`, `psi_i = alpha_i0 + psi_{i-1}(alpha + z*beta)`,
`alpha_hat_ij = (psi_j/psi_i)(alpha_ij + z*beta_ij)`,
`beta_hat_ij = (psi_j/psi_i)*beta_ij`, and the effective sub-step
`dt_hat = c_hat_last * dt` (last modified abscissa). The base methods are
Heun (pRRK22), SSPRK3 (pRRK33), and Spiteri-Ruuth SSPRK54 (pRRK54).

**Key finding -- it is dt-DEPENDENT, not a fixed tableau.** Because
`z = kappa*dt`, the modified `(a, b, c)` depend on BOTH `kappa` and `dt`.
A `butcherScheme(...)` builds its tableau once, so it cannot represent pRRK;
the one-step map *is* `u^{n+1} = u^n + dt*sum_j b_j k_j` (verified
algebraically: `alpha_hat20*u0 + alpha_hat21*u1 + beta_hat21*dt_hat*k2 =
u0 + dt*(c_last*alpha_hat21*beta_hat10*k1 + c_last*beta_hat21*k2)`), but the
`b_j` (and `a_ij`, `c_i`) are functions of `(kappa, dt)`. So the driver
(`prrk.py`) recomputes the tableau from the caller's `(kappa, dt)` on every
call and feeds it to the shared `RungeKuttaB`. The Shu-Osher relaxation map
unfolds to a standard explicit RK tableau because every modified alpha-row
sums to 1 (a direct consequence of the psi recurrence).

**Not registered (registry decision, SCHEME_EXPANSION_PLAN wave 4).** At
`kappa = 0` pRRK duplicates already-registered schemes; at `kappa > 0` there
is no single fixed order/identifier. So `pRRK22/33/54` are exposed
*alongside* (not inside) `IntegrationSchemes` -- imported in `__init__.py`,
pinned absent from the registry by
`tests/test_prrk.py::test_prrk_is_not_registered_in_integration_schemes`.
The scheme count stays 76.

**Exact kappa -> 0 reduction.** At `kappa = 0` the relaxation is the
identity: pRRK22 -> Heun, pRRK33 -> SSPRK3, pRRK54 -> SSPRK54. Pinned to
machine precision (one-step map AND tableau) in
`test_prrk_kappa_zero_equals_base_ssprk` (pRRK54 deviates at 3.9e-16, the
rest at 0.0).

**Why `sum(b) != 1` at kappa > 0 is not a bug.** For a fixed-tableau RK
method the order-1 condition is `sum(b) = 1`. For pRRK, `sum(b) =
c_last^2`, which is 1 only at `kappa = 0`. This does not break the order:
pRRK's coefficients depend on `z = kappa*dt`, and every alpha-row sums to 1
for ALL `z`, so the spurious `O(dt)` terms from the `O(dt)` coefficient
corrections cancel (the `kappa*dt*[alpha_hat20'(0)+alpha_hat21'(0)]*u0`
term vanishes because `alpha_hat20 + alpha_hat21 = 1` identically in `z`).
The method therefore retains its formal order for every `kappa` -- measured
on a 2D nonlinear non-autonomous ODE vs `solve_ivp` RK45 (rtol 1e-12):
pRRK22 -> 2, pRRK33 -> 3, pRRK54 -> 4, at `kappa = 0.5` and `1.0`.

**Stability-region enlargement (the selling point).** On the negative real
axis (`u' = -u`, `z = -dt`): at `kappa = 0` the interval is exactly the base
SSPRK one ([-2, 0] / [-2.51, 0] / [-5.33, 0] for 22/33/54, matching the
library's registered intervals). At `kappa >= 0.5` the interval is
UNBOUNDED -- A-stable on the real axis: |R| <= 1 for every dt, with
R -> 1 from below as dt -> inf (A-stable but **not** L-stable; the stiff
mode is bounded, not damped). There is no imaginary-axis stability at any
`kappa` (a known SSPRK limitation -- Heun's `|R(-is)|^2 = 1 + s^4/4 > 1`),
so the enlargement is a real-axis (stiff/parabolic) property, trading the
SSP (TVD) constant for stiffness tolerance.

`tests/test_prrk.py` (16 tests) pins: the exact kappa->0 reduction, tableau
consistency for all kappa, order preservation at kappa>0, the base real-axis
interval at kappa=0, A-stability at kappa>=0.5, and the not-registered
registry decision.

---
