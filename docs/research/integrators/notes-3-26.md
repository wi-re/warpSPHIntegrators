> Erratum, 14 October 2026: the SplitEuler/Symplectic Euler equivalence stated
> in the historical text below is incorrect. OrdinaryDiffEq 7.8.1
> `lib/OrdinaryDiffEqLowOrderRK/src/split_perform_step.jl` advances
> `u = uprev + dt*(f1+f2)`, matching Forward Euler on a combined RHS.
> [Current disposition](../../plans/method-dispositions.md).

> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.26 Scheme-expansion wave 6: Tsit5 + the LowOrderRK workhorses -- done 2026-10-13

Eighteen OrdinaryDiffEq v7.8.1 tableaus on the generic Butcher driver
(wave 6 of the expansion plan), registry 79 -> 97: the Tsit5 5(4) pair
(ODE's recommended default for unknown stiffness, OrdinaryDiffEqTsit5) and
the OrdinaryDiffEqLowOrderRK workhorses.

**Registration list (all one-step, all dissipative).**

| Method | Order | Stages | Embedded | FSAL | SSP r | Real axis |
|---|---|---|---|---|---|---|
| Tsit5 | 5(4) | 7 | yes | yes | 0 | [-3.506, 0] |
| OwrenZen3 | 3(2) | 4 | yes | yes | 0.7813 | [-2.512, 0] |
| OwrenZen4 | 4(3) | 6 | yes | yes | 0.6444 | [-2.872, 0] |
| OwrenZen5 | 5(4) | 8 | yes | yes | 0 | [-3.192, 0] |
| BS5 | 5(4) | 8 | yes (bhat) | yes | 0.5237 | [-3.986, 0] |
| Stepanov5 | 5(4) | 7 | yes | yes | 0 | [-3.496, 0] |
| Alshina2 | 2(1) | 2 | yes (= Euler) | no | 1 | [-2, 0] |
| Alshina3 | 3 | 3 | -- | no | 1 | [-2.512, 0] |
| Alshina6 | 6 | 7 | -- | no | 0 | [-2.904, 0] |
| MSRK5 | 5 | 9 (8+FSAL) | -- | yes | 0.3226 | [-4.366, 0] |
| MSRK6 | 6 | 9 (8+FSAL) | -- | yes | 0.1009 | [-4.164, 0] |
| SIR54 | 5 | 8 (7+FSAL) | -- | yes | 0 | [-3.512, 0] |
| Ralston4 | 4 | 4 | -- | no | 0.7209 | [-2.784, 0] |
| RKO65 | 5 | 6 | -- | no (free) | 0 | [-4.164, 0] |
| RKM | 3 | 6 | -- | no | 0.7531 | [-3.224, 0] |
| PSRK4p7q6 | 4 | 6 | -- | no | 0 | [-11.41, 0] |
| PSRK3p6q5 | 3 | 5 | -- | no | 0 | [-3.548, 0] |
| PSRK3p5q4 | 3 | 4 | -- | no | 0 | [-2.784, 0] |

**Cross-check.** Every tableau was run one step against the corresponding ODE
algorithm on the same 2D nonlinear non-autonomous problem: max |diff| = 0.0
for all 18. (The ODE sources give exact fractions where printed decimals
would round, and the perform_step recurrences reproduce the Butcher rows
exactly, so this is a stronger check than a tolerance.)

**Fixed-step only (no valid embedded estimate).** Alshina3's ODE `b2tilde`
sums to 5/9 and SIR54's `btilde` sums to 7.854 -- both are derivative
estimates for step-size control, not embedded *solutions*, so both register
plain (`result.error is None`). BS5's embedded weights are ODE's `bhat`, not
`btilde` (which also fails to sum to 1); the pair is FSAL and the free
estimate measures the advertised 4th-order rate on both test problems.

**RKO65: the free first stage (the wave's only theory result).** RKO65
(Tsitouras 2019, Lane-Emden starter) has a_1 = b_1 = 0 with ODE storing
c_1 = 2/3. Claim: for any tableau with a[0] = b[0] = 0, *no* order
condition of any order involves c_0. Every order-2+ tree's elementary
differential at stage 1 needs an a-row factor (F(tau)_1 = 0 for all such
trees), and the order-1 condition is sum(b) = 1, in which b_1 = 0 drops c_0
entirely. ODE's c_1 = 2/3 is therefore decorative: the first stage may be
evaluated at any time. The driver's stage-0 convention (k0 at t^n, the
step start, like every registered scheme) is a valid 5th-order choice --
and the no-reuse measurement confirms it: 5.006 on the non-autonomous
`forced` problem (an order-1 stage-time error would show up there, not on
the autonomous oscillator). Row sums: rows 2..s satisfy c_i = sum_j a_ij to
~1e-16; row 1 structurally does not (c_1 = 2/3, a_1 = 0; residual 6.7e-01),
which is why RKO65 is not in the embedded-pair consistency test.

**RKO65 reuse: lossless, and the new gain-3 branch in `reuse.py`.** With
b[0] = 0 the stale k0 enters the update only through the stage equations
(gain 2, retained q + 2), but the one-stage paths from k0 to the update
carry the common coefficient (b^T a)_0 -- for RKO65, 6.1e-16, zero to
machine precision -- so the stale k0 reaches the update only through
two-stage paths: local O(dt^6), global O(dt^5) = lossless. Measured 5.00
(oscillator) / 5.01 (forced); the forced-reuse errors are bit-identical to
no-reuse. `tableau_reuse_analysis` gained the matching branch
(`b[0] == 0 and (b^T a)_0 == 0` -> gain 3); RKO65 is its first registered
consumer (Midpoint / EPEC have (b^T a)_0 = 0.5 and are unchanged, and no
other registered tableau has b[0] = 0 with (b^T a)_0 = 0).

**RKM registers as order 3.** ODE's RKM (Mead-Renaut 1999) is a
stability-optimized 6-stage method for first-order pseudospectral
operators; its formal accuracy is 3rd order (measured at the registered 3
within tolerance on both test problems), which is what it registers. Its
value is the stability shape (real axis [-3.224, 0]), not the accuracy.

**Reuse orders (measured, pinned in `tests/test_step_reuse.py`).** The nine
FSAL tableaus (the six embedded pairs plus MSRK5 / MSRK6 / SIR54, whose
last row IS the weight vector) reuse losslessly at the registered order.
Non-FSAL, prediction -> measured: Alshina2 1 -> 1.04, Alshina3 1 -> 0.98,
Alshina6 4 -> 4.01, Ralston4 2 -> 1.98, RKM 1 -> 0.99, RKO65 **5** -> 5.00
(the gain-3 case), PSRK4p7q6 4 -> 4.00, PSRK3p6q5 3 -> 2.98, PSRK3p5q4
3 -> 2.90.

**Roundoff floor on `forced` (a test-grid issue, not a scheme defect).**
BS5, Alshina6, MSRK6 and SIR54 are so accurate on the non-autonomous
`forced` problem that their standard-grid (dt 0.1 .. 0.0125) errors sit at
or below the 1e-12 double-precision floor: `measured_order` has no usable
pair (None), not a low number -- the same phenomenon as the
`EXACT_ON_LINEAR_PROBLEMS` skip, on a non-autonomous problem. The
convergence/reuse tests pin them on a coarser grid (`FORCED_COARSE_GRID` in
`tests/conftest.py`; every dt divides T = 2.0): Alshina6 / MSRK6 / SIR54 on
[1.0, 0.5, 0.25, 0.125], where they measure 6.00 / 5.99 / 5.07. BS5 on
[2.0, 1.0, 0.5, 0.25] (finest usable pair 6.95): its x-component error on
this problem is superconvergent -- the u error is oscillatory and its time
integral over [0, 2] nearly cancels (measured x errors: +2.66e-08 at
0.5, +2.16e-11 at 0.25, then -1.18e-11 at 0.125: a sign change, with a
~2^10.3 decay ratio between 0.5 and 0.25 while the u component decays at a
clean 5.16-5.43) -- so a higher-order term dominates at moderate dt and
crosses the true dt^5 term near dt ~ 0.17. The L1 apparent order therefore
drifts 6.95 (0.5..0.25) to 4.03 (0.125..0.0625) across grids, and no grid
resolves a clean 5 in the L1 norm; the registered 5 is pinned directly on
oscillator (5.02) / kepler instead.

**Deferred (documented, not registered).** Anas5(w) (Anastassi-Simos 2005)
and FRK65(omega) (Medvedev-Simos-Tsitouras 2018) are parameter-dependent
tableaus (a free w / omega trades SSP against accuracy) -- the pRRK
situation, not fixed tableaus. SplitEuler is algebraically the registered
`symplecticEuler` (same one-step map); registering it would duplicate the
scheme.
