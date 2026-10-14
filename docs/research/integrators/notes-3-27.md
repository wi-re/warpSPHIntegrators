> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.27 Scheme-expansion wave 7: the high-order tableaus (Verner / HighOrderRK / Feagin / DP8) -- done 2026-10-13

Eleven high-order OrdinaryDiffEq v7.8.1 tableaus on the generic Butcher
driver (wave 7 of the expansion plan), registry 97 -> 108, ExplicitRK
52 -> 63: the OrdinaryDiffEqVerner pairs, the OrdinaryDiffEqHighOrderRK
methods and the OrdinaryDiffEqFeagin symmetric methods.

**Registration list (all one-step, all dissipative, all ExplicitRK).**

| Method | Order | Stages | Embedded | FSAL | SSP r | Real axis |
|---|---|---|---|---|---|---|
| Vern6 | 6(5) | 9 | yes | yes | 0.0877 | [-4.855, 0] |
| Vern7 | 7(6) | 10 | yes | no | 0.0514 | [-4.640, 0] |
| TanYam7 | 7(6) | 10 | yes | no | 0 | [-4.913, 0] |
| RKV76IIa | 7(6) | 10 | yes | no | 0 | [-4.911, 0] |
| Vern8 | 8(7) | 13 | yes | no | 0.1615 | [-13.425, 0] |
| TsitPap8 | 8(7) | 13 | yes | no | 0.1804 | [-10.388, 0] |
| DP8 | 8 | 12 | -- (dual 5/3) | no | 0.2371 | [-6.394, 0] |
| Vern9 | 9(8) | 16 | yes | no | 0.1272 | [-4.476, 0] |
| Feagin10 | 10 | 17 | -- (symmetric) | no | 0 | [-2.528, 0] |
| Feagin12 | 12 | 25 | -- (symmetric) | no | 0 | [-4.070, 0] |
| Feagin14 | 14 | 35 | -- (symmetric) | no | 0 | [-1.874, 0] |

Structural notes: TanYam7 (the 9-stage method of Tanaka-Muramatsu-Yamashita
1992) and RKV76IIa both store 9 working stages plus a companion stage at
c = 1; the Verner pairs likewise carry a c = 1 tail, and **Vern6 is the
wave's only FSAL tableau** (its last row IS the main weight vector). TsitPap8's
stage 8 sits at c_8 = 1.0070922, just past the step end (the SSPRK53_2N1 /
ESDIRK precedent; the stage-times test carries a per-scheme slack). The
Feagin c-nodes are mirror-symmetric about c = 1/2 up to the endpoint stages
-- the visible signature of time-reversibility.

**Cross-check against the actual ODE solver.** Every method ran the full
interval on a real (nonlinear, non-autonomous) problem and was compared
with the corresponding ODE v7.8.1 algorithm; max |dy| at T = 2: TanYam7
1.7e-14, TsitPap8 1.1e-16, DP8 0.0, Feagin10 0.0, Feagin12 0.0, Feagin14
1.7e-16, Vern6 4.1e-14, Vern7 2.8e-17, Vern8 8.9e-16, Vern9 0.0,
RKV76IIa 1.2e-14. (Unlike wave 6's exact-fraction sources, the Verner /
HighOrderRK / Feagin tableaus store decimal floats, so this is a
tolerance-level check, not a byte-level one.)

**Plain registrations: DP8 and the Feagin methods.** DP8's companion is the
dual 5th/3rd-order combination of Hairer's 8/5/3 adaptation (not a valid
(8,7) embedded solution), and the Feagin tableaus are plain in ODE
(symmetric methods, no embedded pair): the four register with no error
estimate (`result.error is None`). The seven embedded pairs expose the
pair difference; `tests/test_embedded.py` pins their estimate rates on
per-pair grids (`EMBEDDED_GRID` -- they floor out on the standard grid):
Vern6 6.08/6.04/6.00, Vern7 7.24/7.11/7.05, Vern8 8.48/8.32/8.19, Vern9
9.79/9.29/9.21, RKV76IIa 7.20/7.09/7.03, TanYam7 7.20/7.09/6.99, TsitPap8
9.03/8.73/8.51 (rates against dt^order on [0.2..0.025] / [0.4..0.05] /
[0.8..0.1] grids).

**Reuse: the first multi-stage refusal class.** Reuse turns the step into a
recurrence whose state is (y^{n-1}, y^n, k_s^{n-1}) -- the new step's k0 is
the previous step's last-stage derivative, which depends on the step before
that -- so its stability region is NOT the tableau's one-step region. On
y' = lambda y with z = lambda*dt the recurrence's matrix is
M = [[0, 1, 0], [0, G_y, G_k], [0, H_y, H_k]] with A2 = a[1:, 1:],
Inv = (I - z*A2)^{-1}, ones = 1: G_y = 1 + z*b[1:] Inv ones,
G_k = b[0] + z*b[1:] Inv a[:, 0][1:], H_y = z*(1 + z*last Inv ones),
H_k = z*(a[-1, 0] + z*last Inv a[:, 0][1:]), last = a[-1, 1:]; its two
nonzero eigenvalues are the roots of
x^2 - (G_y + H_k) x + (G_y H_k - G_k H_y). The model was verified to
~1e-15 against empirical extraction at basis states, and against a literal
step-by-step run (FSAL reuse reproduces the one-step R(z)^n to 1.5e-15,
where R(z) = 1 + z b^T (I - zA)^{-1} ONES -- the stage forcing is the ones
vector, every stage is forced by y^n; using e_0 is a bug that invalidates
the whole scan). Imaginary-axis scan (both eigenvalues in the unit disk):

| Method | One-step | Reuse |
|---|---|---|
| Vern6 (FSAL) | 2.582 | = one-step (R-mode) |
| Vern7 | 0.352 | **0.020** |
| Vern8 | 0.636 | 0.332 |
| Vern9 | 2.754 | 0.476 |
| RKV76IIa | 4.435 | **0.008** |
| TanYam7 | 4.435 | **0.008** |
| TsitPap8 | 2.914 | 0.112 |
| DP8 | 5.960 | 0.440 |
| Feagin10 | 1.153 | 1.565 |
| Feagin12 | 1.061 | 0.712 |
| Feagin14 | 1.157 | **0.108** |

(Reuse can *widen* the interval -- Feagin10 -- the two-step map's extra
state carries phase information; the refusal decision is about covering
the standard test step z = 2i*dt = 0.2i at dt = 0.1 on the omega = 2
oscillator, not about absolute width.) **Vern7, RKV76IIa, TanYam7 and
Feagin14 refuse reuse outright** (`reuse.REUSE_IMAGINARY_UNSTABLE`,
`reuse_order = None`, the single-stage refusal's precedent): their reuse
intervals do not reach 0.2, and the literal runs confirm -- Vern7 osc
std errors [2.8e13, 6.1e18, 1.7e19, 5.9e-3] at dt 0.1..0.0125,
RKV76IIa [3.4e19, 2.8e34, 1.4e53, 2.0e67], TanYam7
[7.6e18, 1.3e33, 2.9e50, 1.2e62], Feagin14 [4.8e-03, 2.9e-13, 5.9e-15,
7.8e-16] (the dt = 0.1 point is the instability, ten digits off the
predicted order-6 curve). The warning text names the stability bound and
the measured divergence.

**Reuse orders (measured, pinned in `tests/test_step_reuse.py`).** Vern6
reuses losslessly at 6 (bitwise, `test_fsal_reuse_is_free`). Non-FSAL,
prediction -> measured (oscillator std grid / forced coarse grid):
Vern8 5 -> 4.92 / 5.07, TsitPap8 5 -> 4.96 / 5.05, Vern9 6 -> 5.54 / 5.49,
DP8 5 -> 5.00 / 5.82, Feagin10 6 -> 6.01 / 5.67, Feagin12 6 -> 6.99 / 7.04.
The last four are **pre-asymptotic on every double-precision grid**: the
reuse recurrence's leading error term is one degree below the nominal
order, so the C7 term keeps the finest usable pair ~0.5 below the
asymptotic slope, and the finest pair is the same on any grid (the floor,
not the grid, sets it -- Vern9 forced measures 5.4871 pre-asymptotic on
every grid ending 0.5/0.25). They get the relaxed `HIGH_ORDER_REUSE` band
[_pred - 0.6, order + 1 - 0.15) instead of the standard
[pred - 0.15, pred + 0.85). The reuse-order oscillator test uses the
standard grid (the reuse order is asymptotic only there: Vern8 measures
4.34 on the forced-style coarse grid, 4.92 on the standard one); forced
uses the per-problem coarse grid.

**Roundoff floor: per-problem coarse grids (a test-grid issue).** The
wave-7 tableaus floor out on more than one problem at once: the standard
grid (dt 0.1 .. 0.0125) sits at or below the 1e-12 floor for the
oscillator (all but the Feagins), the damped problem (all), and forced
(all but Feagin12/14, which floor out on *every* dt that divides T = 2.0 --
Feagin12 at 1.8e-11 for the single dt = 2.0 step, Feagin14 at 5.8e-14 --
so their MAIN order is unmeasurable there and
`test_scheme_achieves_registered_order` skips forced for them;
`FORCED_FLOOR`). The main-order tests now use
`conftest.PROBLEM_COARSE_GRIDS` (per-problem dicts; every entry divides
T = 2.0). Measured main orders on the chosen grids (forced / oscillator /
kepler / damped): Vern6 6.85 / 7.03 / 7.45 / 7.99, Vern7 6.87 / 7.07 /
7.22 / 7.17, Vern8 8.72 / 7.98 / 7.87 / 8.62, Vern9 9.18 / 9.65 / 9.95 /
9.73, RKV76IIa 7.30 / 7.03 / 8.40 / 7.00, TanYam7 7.43 / 7.23 / 7.81 /
7.17, TsitPap8 9.10 / 8.01 / 7.86 / 8.22, DP8 8.04 / 8.04 / 7.93 / 7.88,
Feagin10 10.20 / 10.58 / 12.65 / 11.08, Feagin12 -- / 13.16 / 12.35 /
13.23, Feagin14 -- / 14.85 / 15.43 / 15.13. Three 3-point grids
(RKV76IIa / TanYam7 on forced, Feagin10 on kepler) keep the two finest
points above the floor; a 4th point at dt = 0.25 would be
floor-contaminated and drag the measured pair 0.5-1.5 below the true order.
Two more pairs need the g2 grid because their g4 grid's dt = 0.1 point
(1.2e-13 for TsitPap8 on kepler, 6.5e-13 for DP8 on damped) is floor-
dropped, leaving only the pre-asymptotic coarse pair (7.69 / 7.77, both
below the 7.85 lower bound): the (0.5, 0.25) / (0.25, 0.125) pairs on g2
measure 7.855 / 7.878.

**TVD / SSP pins (`tests/test_tvd.py`).** Verdicts (ssp cap4, tvd_cfl,
unconditional): Vern6 (0.0877, 2.0, F), Vern7 (0.0514, 2.0, F), Vern8
(0.1615, 2.0, F), Vern9 (0.1272, 2.0, F), RKV76IIa (0, 2.0, F), TanYam7
(0, 2.0, F), TsitPap8 (0.1804, 2.0, F), DP8 (0.2371, 3.0, F), Feagin10
(0, 1.0, F), Feagin12 (0, 1.25, F), Feagin14 (0, 0.75, F) -- none
unconditional. Real-axis intervals: the wave's widest is Vern8
[-13.425, 0]; the narrowest is Feagin14 [-1.874, 0] despite its order --
verified genuine (|R(-2.0)| = 3.51, |R(-1.8736)| = 0.9976, a single
interval): the 35-stage stability polynomial oscillates for |z| >~ 2,
smaller than the family trend (Feagin10 -2.53, Feagin12 -4.07). ODE's own
stored stability sizes agree (Vern6 4.8553, Vern7 4.64, Vern9 4.4762,
RKV76IIa 4.9108).

**Area-defect floor (`tests/test_hamiltonian.py`).** Five of the wave's
schemes -- Vern8, Vern9, TsitPap8, DP8, Feagin12 -- are non-symplectic, but
their one-step area defect on the linear oscillator at this test's step
(h = dt*omega = 1) sits below the 1e-7 discrimination floor: det(D Phi) - 1
= +9.6e-09 / -2.9e-08 / -8.4e-08 / -4.8e-08 / -1.9e-08. A non-symplectic
order-p method's defect on the linear oscillator is O(h^(p+2)) with a
method-dependent coefficient, and for these the coefficient is small
enough that the single-step test cannot see it. The area-property test
therefore skips the five (`AREA_DEFECT_BELOW_DISCRIMINATION`), and their
non-symplecticity is pinned by the growth of the defect with the step
instead: at h = 2.5 the same five measure +1.97e-05 / -3.74e-04 /
-9.76e-05 / -5.07e-04 / +7.97e-02
(`test_high_order_area_defect_grows_with_the_step`, which also asserts the
h = 1 values stay below the floor, so a future reclassification is forced
to be deliberate), with the nonlinear-problem drift pinned by the
dissipation-flag test. The wave's controls (Vern6, Vern7, Feagin10,
Feagin14) sit above the floor at h = 1 (-9.9e-06 / +1.1e-06 / -2.6e-06 /
-1.0e-05) and pass the original test unchanged.

---
