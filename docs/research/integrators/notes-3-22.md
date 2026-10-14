> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.22 Scheme-expansion wave 2: the SSPRK 3(2) pairs and SSPRK932 — done 2026-10-12

Wave 2 of `docs/plans/scheme-expansion.md`: the remaining one-step SSPRK methods
present in local OrdinaryDiffEq v7.8.1 — SSPRK43 and SSPRK432 (4-stage
order-3 SSP with an embedded order-2 estimate; the registry's first SSP 3(2)
pairs) and SSPRK932 (9-stage order-3 SSP, registered fixed-step only). All
three ride the generic `butcher.RungeKuttaB` driver (complexity class C1);
none are FSAL. Registry: 71 -> 74 schemes, ExplicitRK 28 -> 31.

**Citation chain.** Coefficients read line-by-line on 2026-10-12 from
`~/.julia/packages/OrdinaryDiffEq/OnlvZ/lib/OrdinaryDiffEqSSPRK/src/`: stage
recurrences in `ssprk_perform_step.jl` (SSPRK43 lines 766-799, SSPRK432
862-899, SSPRK932 1201-1253), method references in the `@doc` blocks of
`algorithms.jl`: Kraaijevanger, *BIT* 31.3 (1991) 482-528 (SSPRK43 main
method), Conde, Fekete & Shadid, arXiv:1806.08693 (SSPRK43 embedded pair;
implementation: Ranocha et al., arXiv:2104.06836), Gottlieb, Ketcheson &
Shu, *Strong Stability Preserving Runge–Kutta and Multistep Time
Discretizations*, World Scientific 2011 (SSPRK432 as Example 6.1; SSPRK932).
Unlike wave 1 (Shu–Osher alphas/betas), all three are stored as stage
recurrences; the Butcher conversion is the stage-state increment
`S_i = (u_i - u^n)/dt`, `c_i` = row sum of `S_i`, `b` from the final update
line.

**Verification.** In exact Fraction arithmetic
(`.tmp/wave2_tableau_check.py`, `.tmp/wave2_verify.py`): all three tableaus
reproduce the verbatim ODE stage recurrences to at most 2.5e-16 on a
nonlinear time-dependent test ODE; the order conditions hold exactly — main
weights order 3, both embedded weights order 2 (`sum = 1`, `b·c = 1/2`);
`c[-1] != 1` (non-FSAL) for all three. The registry's own gates then measure
order 3 on all three canonical problems (`tests/test_convergence.py`), the
embedded-estimate scaling `O(dt^3)` on the oscillator
(`tests/test_embedded.py`), and the adaptive accept/reject loop
(`tests/test_adaptive.py::test_adaptive_loop_smoke_on_new_ssprk_pairs`).
One transcription error was caught this way: the first SSPRK932 draft
dropped the all-zero first row (`k_1 = f(u_0, t)`, `c_1 = 0`), which made
`b·c = 7/12` (order 1, contradicting the measured order ~3) — the
tableau-vs-recurrence comparison showed a 3e-3 mismatch and the missing row
was restored.

The ODE source comments on the embedded construction are algebraically
exact, not approximations: SSPRK43's `utilde` is computed after the third
stage (before the final update), where `u` is the 3-stage cascade output
`u0 + (dt/2)(k1 + k2 + k3)`, so `utilde = u0 + dt·(1/3, 1/3, 1/3, 0)·k`
(matches its "corresponds to bhat = (1/3, 1/3, 1/3, 0)" comment) and the
final `E = (1/2)(utilde - u_final)` = `(u0 + dt·(1/4, 1/4, 1/4, 1/4)·k) -
u_final` exactly (its "bhat = (1/4, 1/4, 1/4, 1/4)" comment). SSPRK432's
`utilde = (uprev + 2u)/3` is the same 3-stage cascade blend and
`E = utilde - u_final` gives `b_embed = (1/3, 1/3, 1/3, 0)` exactly. The
registered `b_embed` values are therefore ODE's own embedded solutions, not
a re-derivation.

**SSP coefficients.** ODE's published values match the library's
`convex_combination_cfl` (all stage maps + final map, upwind model) exactly
(no divergence, unlike wave 1's KYK/SSPRK54):

| method | ODE published | library, all-stage |
|---|---|---|
| SSPRK43 | 2.0 | 2.0 |
| SSPRK432 | 2.0 | 2.0 |
| SSPRK932 | 6.0 | 6.0 |

(SSPRK932's 6.0 is why `tests/test_tvd.py`'s boundary scan goes to CFL 8;
the default cap 4.0 would clip it.) A naive hand bound `r <= 1/max(row sum)`
gives 1.2 for SSPRK932 and is wrong — the calibrated upwind model is the
authority (it reproduces ODE's published C for all three).

**SSPRK932 is fixed-step only: ODE's adaptive estimate is not a 3(2) pair.**
ODE's SSPRK932 perform_step computes, after the six-stage cascade, an extra
evaluation `k = f(u6, t + dt)` (a 10th, tableau-external stage value), sets
`utilde = (uprev + 6*u6 + 6*dt*k)/7`, and uses `utilde - u_final` as the
error estimate. The three coefficients (1/7 on `uprev`, 6/7 on `u6`, 6/7 on
`dt*k`) sum to 13/7, so `utilde` is not of the form `u0 + dt*bhat*k` for any
probability vector `bhat`. Expanding around `t`: `u6 = u(t+dt) + O(dt^2)`
(the cascade is six forward-Euler substeps of `dt/6`, order 1),
`k = u'(t) + O(dt)`, so `utilde - u_final = (5/7)*dt*u'(t) + O(dt^2)` —
measured scaling 1.058 on the oscillator (`.tmp/wave2_probe.py`), i.e.
`O(dt)`: two orders too large for the `O(dt^3)` local error a 3(2) pair's
estimate must have. SSPRK932 is therefore registered without an embedded
`b`; the adaptive controller has no estimate to drive
(`result.error is None`, pinned by
tests/test_embedded.py::test_non_embedded_schemes_report_no_error). ODE's
own docstring says "Consider using `SSPRK43` instead, which uses the same
main method and an improved embedded method" (the "same main method" part is
ODE's loose phrasing — the methods differ; the recommendation is the
point). SSPRK43 / SSPRK432 provide the SSP 3(2) adaptive capability.

**Real-axis stability intervals** (linear scan over 20001 points, the
test's `abs=5e-3` pin): SSPRK43 and SSPRK432 share `[-5.149, 0]` (same
`(a, c, b)`), SSPRK932 `[-13.290, 0]`. Each is a single interval (the
amplification does not re-enter the unit disk for more negative z). Pinned
in `tests/test_tvd.py::test_expanded_ssprk_ssp_boundaries_and_real_axis_intervals`.
The real-axis pin needed a one-line fix for embedded pairs: the test's `b`
is the `(b_main, b_embedded)` tuple, and the interval is defined for the
propagated weight (`b = b[0]`), which the test now unwraps.

**TVD verdicts** (n = 64, step IC, CFL grid to 5, convex-combination scan to
CFL 4) — added to `EXPECTED_VERDICTS`: SSPRK43 (2.0, 2.0, False), SSPRK432
(2.0, 2.0, False), SSPRK932 (4.0, 5.0, True). SSPRK932's ssp value hits the
scan cap (true boundary 6.0); its per-step sweep is TVD to the CFL-5 cap
(unconditional) — its real axis `[-13.29, 0]` covers the whole sweep.
`tests/test_tvd.py` is now 48 tests.
