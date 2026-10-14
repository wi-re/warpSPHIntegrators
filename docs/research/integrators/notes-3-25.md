> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.25 Scheme-expansion wave 5: SSPRK53_2N1/_2N2/_H -- low-storage third-order SSP -- done 2026-10-13

Higueras & Roldan's new third-order low-storage SSP methods
(arXiv:1809.04807), from OrdinaryDiffEq v7.8.1 (`ssprk_caches.jl` constant
caches, `ssprk_perform_step.jl` recurrences). All three are 5-stage order 3;
ODE implements them with 1-2 working temporaries (vs 2 for SSPRK53).

**Screening gate (the plan's) -- all three are DISTINCT from SSPRK53.**
First check (before any registration work): is each variant algebraically
identical to the registered SSPRK53 as a one-step map? Derived each variant's
Butcher tableau by unfolding ODE's Shu-Osher recurrences (the base-method
derivation reproduces the registered SSPRK53 tableau to 3e-17, validating the
conversion), then compared one-step maps on a 2D nonlinear non-autonomous
ODE: the difference per unit dt^4 is 4.4e-3 (2N1), 2.0e-3 (2N2), 1.5e-4 (H)
-- O(dt^4), i.e. distinct same-order methods, not ~1e-16 storage
rearrangements. All three register via the standard checklist. (ODE's
storage saving does not apply to this driver: `RungeKuttaB` builds each
stage state fresh from the a-row.)

**Tableau check.** Row sums equal ODE's stored c to <=1.05e-15; the four
order-3 conditions (sum b = 1, b.c = 1/2, b.a.c = 1/6, b.c^2 = 1/3) hold to
<=5.3e-15. None FSAL. Note SSPRK53_2N1's stage 3 sits at c_3 = 1.0053 > 1
(evaluated just past the step end, as ODE does) -- harmless for the generic
driver, which takes the node time from c.

**Measured properties (pinned in `tests/test_tvd.py`).**

| Method | SSP r (convex_combination_cfl, mu_max=8) | ODE published | Real axis |
|---|---|---|---|
| SSPRK53 (reference) | 2.6506 | 2.65 | [-6.21361, 0] |
| SSPRK53_2N1 | 2.1808 | 2.18 | [-5.52779, 0] |
| SSPRK53_2N2 | 2.1487 | 2.148 | [-7.26303, 0] |
| SSPRK53_H | 2.6506 | 2.65 | [-6.21351, 0] |

ODE's published SSP coefficient agrees with the library's all-stage
measurement to <=8e-4 (as in wave 1). SSPRK53_2N2 has the largest
real-axis interval of the four despite the smallest SSP coefficient;
SSPRK53_H matches SSPRK53's interval to 1e-4 (it shares SSPRK53's first two
nodes c_1, c_2 but has distinct later stages). Classifier verdict
(`classify_all`, the test parameters): all three `(ssp, tvd=2.0,
unconditional=False)` -- the same per-step TVD CFL as SSPRK53.
