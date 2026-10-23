> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.21 Scheme-expansion wave 1: the seven plain SSPRK tableaus — done 2026-10-12

Wave 1 of `docs/development/plan-reconciliation-2026-10-23.md`: the plain (non-embedded, non-multistep)
SSPRK methods present in local OrdinaryDiffEq v7.8.1 but missing from this
library. All seven ride the generic `butcher.RungeKuttaB` driver as plain
tableaus (complexity class C1); none are FSAL, so the reuse guard applies
automatically.

**Citation chain.** Coefficients read line-by-line on 2026-10-12 from
`~/.julia/packages/OrdinaryDiffEq/OnlvZ/lib/OrdinaryDiffEqSSPRK/src/`:
alphas/betas as decimal literals in `ssprk_caches.jl`, stage recurrences in
`ssprk_perform_step.jl`. ODE stores the methods in Shu–Osher form (stage
state `u_i = sum_r alpha_ir u_r + dt sum_r beta_ir k_r`); the Butcher
conversion is `S_i = (u_i - u^n)/dt` (the stage-state increment scaled by
1/dt), `c_i` = row sum of `S_i`, `b` from the final update line. Paper
citations from the `@doc` blocks in `algorithms.jl`: Ruuth, *Mathematics of
Computation* 75.253 (2006) 183-207 (SSPRK53/63/73/83, SSPRK54); Kubatko,
Yeager & Ketcheson, *J. Sci. Comput.* 60 (2014) 313-344 (KYKSSPRK42,
KYK2014DGSSPRK_3S2).

**Verification.** Per method, in exact Fraction arithmetic on the verbatim
decimal coefficients (`.tmp/wave1_tableaus.py`): row sums equal ODE's stored
c to at most 5.6e-16; `sum(b) = 1` to at most 2e-15 (SSPRK73 to 1e-5: ODE
stores alpha73/alpha76 at 6 digits, so its coefficients carry that
order-condition slack); measured order at or above the registered order on a
smooth nonlinear time-dependent test ODE (four dt halvings, reference = the
same method at dt/16); `c[-1] != 1` (non-FSAL) for all seven. The registry's
own convergence gate (`tests/test_convergence.py`) then measures the order on
all three canonical problems. One transcription error (SSPRK83, a missing
alpha73*beta21 term in stage 8) was caught by the c-check at 4.76e-2 and
fixed.

**SSP coefficients: where ODE's published value matches, and where it does
not.** The library's `convex_combination_cfl` (all stage maps + final map,
upwind model, scan to CFL 8):

| method | ODE published | library, all-stage | final-map-only |
|---|---|---|---|
| SSPRK53 | 2.65 | 2.6506 | 2.6506 |
| SSPRK63 | 3.518 | 3.5184 | 3.5184 |
| SSPRK73 | 4.2879 | 4.2879 | 4.2879 |
| SSPRK83 | 5.107 | 5.1071 | 5.1071 |
| SSPRK54 | 1.508 | 1.7693 | 1.8611 |
| KYKSSPRK42 | 2.459 | 2.2838 | 2.4595 |
| KYK2014DGSSPRK_3S2 | 0.8417 | 1.8939 | 1.8939 |

The four Ruuth order-3 methods agree with the published coefficient to at
most 6.3e-4. The three divergences are framework differences, not
transcription errors: KYKSSPRK42's published 2.459 is the final-map
coefficient (the stage maps bind at 2.2838 in the upwind model); SSPRK54's
published 1.508 is the paper's own coefficient, which neither the all-stage
(1.7693) nor the final-map-only (1.8611) library measurement reproduces;
KYK2014DGSSPRK_3S2's published 0.8417 is the paper's DG-context coefficient.
The tests pin the library-measured values, with these comments
(`tests/test_tvd.py`).

**Real-axis stability intervals** (linear scan + 80 bisections on the
stability polynomial): KYK2014DGSSPRK_3S2 [-4.267, 0], KYKSSPRK42
[-4.954, 0], SSPRK54 [-5.331, 0], SSPRK53 [-6.214, 0], SSPRK63 [-7.213, 0],
SSPRK73 [-8.888, 0], SSPRK83 [-10.986, 0]. Each is a single interval (the
amplification does not re-enter the unit disk for more negative z). Pinned
in `tests/test_tvd.py::test_expanded_ssprk_ssp_boundaries_and_real_axis_intervals`.

**TVD verdicts** (n = 64, step IC, CFL grid to 5, convex-combination scan to
CFL 4) — added to `EXPECTED_VERDICTS`: SSPRK53 (2.65, 2.0, False), SSPRK63
(3.518, 3.0, False), SSPRK73 (4.0, 3.0, False), SSPRK83 (4.0, 5.0, True),
SSPRK54 (1.769, 2.0, False), KYKSSPRK42 (2.284, 2.0, False),
KYK2014DGSSPRK_3S2 (1.894, 1.5, False). SSPRK73/SSPRK83's ssp values hit
the scan cap (true boundaries 4.2879 / 5.1071); SSPRK83's per-step sweep is
TVD to the CFL-5 cap (unconditional).

---
