> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.28 Scheme-expansion wave 8: the remaining DIRK tableaus (Kvaerno / KenCarp / ESDIRK / ARS / Cash / Hairer / SFSDIRK) -- implemented 2026-10-13; validation gate 2026-10-14

The remaining 29 OrdinaryDiffEq v7.8.1 DIRK-family tableaus on the DIRK
driver (wave 8 of the expansion plan), registry 108 -> 137, DIRK family
7 -> 36. Sub-batch A: the L-stability-oriented lines (Kvaerno3/4/5,
KenCarp3/4/47/5/58, the ESDIRK325-659 L[2]SA family, SSPSDIRK2). Sub-batch
B: the general-purpose L-stable methods (ARS222-443, BHR553, Cash4,
CFNLIRK3, Hairer4/42) and the SFSDIRK4-8 family.

**Registration list (all one-step, all DIRK; SSPSDIRK2 is symplectic).**

| Method | Order | Stages | 1st stage | Stiffly acc. | Embedded | Stability |
|---|---|---|---|---|---|---|
| Kvaerno3 | 3 | 4 | E | yes | (3,2) | L |
| Kvaerno4 | 4 | 5 | E | yes | (4,3) | A |
| Kvaerno5 | 5 | 7 | E | yes | (5,4) | L |
| KenCarp3 | 3 | 4 | E | yes | (3,2) | L |
| KenCarp4 | 4 | 6 | E | yes | (4,3) | L |
| KenCarp47 | 4 | 7 | E | yes | (4,3) | L |
| KenCarp5 | 5 | 8 | E | yes | (5,4) | L |
| KenCarp58 | 5 | 8 | E | yes | (5,4) | L |
| ESDIRK3(2)5L[2]SA | 3 | 5 | E | yes | (3,2) | L |
| ESDIRK4(3)6L[2]SA2 | 4 | 6 | E | yes | (4,3) | L |
| ESDIRK4(3)7L[2]SA | 4 | 7 | E | yes | (4,3) | L |
| ESDIRK5(4)7L[2]SA2 | 5 | 7 | E | yes | (5,4) | L |
| ESDIRK5(4)I8L[2]SA | 5 | 8 | E | yes | (5,4) | L |
| ESDIRK6(5)9L[2]SA | 6 | 9 | E | yes | (6,5) | L |
| SSPSDIRK2 | 2 | 2 | I (c0 = 0.25) | no | -- | A |
| ARS222 | 2 | 3 | E | yes | -- | L |
| ARS232 | 2 | 3 | E | yes | -- | L |
| ARS343 | 3 | 4 | E | yes | -- | L |
| ARS443 | 3 | 5 | E | yes | -- | L |
| BHR553 | 3 | 5 | E | yes | -- | L |
| Cash4 | 4 | 5 | I (c0 = 0.4359) | no | (4,3) | L |
| CFNLIRK3 | 3 | 4 | E | yes | -- | L |
| Hairer4 | 4 | 5 | I (c0 = 0.25) | yes | (4,3) | L |
| Hairer42 | 4 | 5 | I (c0 = 0.2667) | yes | (4,3) | L |
| SFSDIRK4 | 4 | 4 | I (c0 = 0.098) | no | -- | (none) |
| SFSDIRK5 | 4 | 5 | I (c0 = 0.079) | no | -- | (none) |
| SFSDIRK6 | 4 | 6 | I (c0 = 0.067) | no | -- | (none) |
| SFSDIRK7 | 4 | 7 | I (c0 = 0.057) | no | -- | (none) |
| SFSDIRK8 | 4 | 8 | I (c0 = 0.050) | no | -- | (none) |

Structural notes:

- Orders follow ODE's `alg_order` (alg_utils.jl), including the
  non-obvious ones: ARS443 and BHR553 are order 3 (not 4 and 5), and
  SFSDIRK5-8 are ALL order 4 -- ODE's own docstrings "Method of order
  5/6/7/8" name the stage count, not the order (pinned by the
  convergence-order tests).
- ODE's ESDIRK436L2SA2 is a *different* method from the pre-existing
  ESDIRK4(3)6L[2]SA; the two coexist (measured |R(-100)| 7.70e-2 vs
  7.57e-2). Identity clusters: Kvaerno3, KenCarp3, ARS343, CFNLIRK3 and
  BHR553 all share one stability function (the 3-stage L-stable method of
  Kvaerno 1997 / Kennedy-Carpenter 2001; the ARS343 implicit part is
  CFNLIRK3 at 12 digits; BHR553's extra stages do not move it) -- R(z)
  agrees to ~5.5e-11 across the five (stored-digit noise) -- and Hairer4
  shares the stability function of KenCarp4 (R agrees to 5.1e-14).
- b-tilde convention: ODE's `btilde` field is the *error* weight (sums to
  ~0); we register `b = (b_main, b_embed)` with `b_embed = b - btilde`
  (sums to 1 = the true lower-order weights). ODE ignores Ae/be on pure
  problems.
- **ODE's stored-c execution bug.** Five methods store c values that do not
  match the row sums of their own a -- Kvaerno3 |delta| = 3.13e-3,
  Kvaerno4 2.45e-4, Kvaerno5 7.99e-7, ESDIRK547L2SA2 6.52e-5, SSPSDIRK2
  2.36e-3 -- and ODE evaluates non-autonomous stages at the STORED c, i.e.
  its solver integrates a different tableau than the Butcher array it
  prints. We register the row-sum c (the mathematically consistent
  tableau). The 1e-13 row-sum test tolerance additionally forces row-sum c
  for Cash4 and SFSDIRK4, where ODE's stored c carries <= 1e-14
  transcription noise off the row sum.
- **SDIRK22 is deliberately absent.** In ODE v7.8.1 (and v7.4.0 and
  v6.9.0) its constructor is a stub -- `SDIRK22()` literally
  `return Trapezoid(...)` (algorithms.jl:314-330) -- and the SDIRK22Tableau
  fields (a = 1 - 1/sqrt(2), alpha = -sqrt(2), beta = 1 + sqrt(2)) are
  orphaned: no perform step references them. There is nothing executable
  to transcribe or cite.

**Cross-check against the actual ODE solver.** One step of each method on
autonomous and non-autonomous problems, our driver vs the corresponding
ODE v7.8.1 algorithm (`.tmp/wave8_dirk/onestep_wave8.jl`): the
16-digit-stored family agrees to 1e-10..1e-12, the 12-digit-stored family
(ARS222-443, BHR553, Cash4, CFNLIRK3, Hairer4/42) to 1e-9..1.4e-8 (the
stored-coefficient precision). For the five stored-c-bug methods the
autonomous diff is the same coefficient-level number, while the
non-autonomous diff is exactly their c-bug magnitude (8.0e-7 .. 3.1e-3)
-- ODE's own stored-c evaluation, which we do not reproduce.

**Stability classification (measured; pinned in `tests/test_stiff.py`).**
A fine left-half-plane grid (500 angles x 500 radii, |z| in [1e-4, 1e5],
plus a dense real axis) found no |R| > 1 for any of the 22 L-stable
tableaus:

| Class | Schemes |
|---|---|
| L (22) | Kvaerno3/5, KenCarp3/4/47/5/58, the six ESDIRK L[2]SA, ARS222-443, BHR553, Cash4, CFNLIRK3, Hairer4/42 |
| A, not L (2) | Kvaerno4 (|R(-100)| = 0.629; R(inf) a nonzero asymptote, |.| = 0.7175), SSPSDIRK2 (|R(-100)| = 0.852; R(inf) = +1) |
| none (5) | SFSDIRK4-8: NOT A-stable, no A(alpha) sector -- a bounded region of the real axis only (max |R| 16-123 on the LHP grid) |

The 19 L-stable tableaus with |R(-100)| < 0.1 join the existing
damping test (measured 2.6e-3 .. 7.9e-2). KenCarp47 (0.149),
ESDIRK4(3)7L[2]SA (0.125) and ESDIRK5(4)7L[2]SA2 (0.106) have a large
damping constant C in R(z) ~ C/|z| and are pinned separately at larger
|z| (|R(-1000)| = 1.5e-2..2.6e-2, |R(-1e6)| < 3e-5). Kvaerno4 and
SSPSDIRK2 sit strictly between the L-stable and the unit-magnitude rows
(0.1 < |R(-100)| < 0.9) and get their own band test. ESDIRK659L2SA's ODE
docstring carries a "STABILITY ISSUES" note (SciML#1933): that issue is
about ODE's adaptive error test for this method, not its one-step map --
on the fine grid it is plainly A- and L-stable.

**Reuse: 20 lossless, 9 refused.** The 20 with an explicit first stage
AND stiff accuracy (Kvaerno3/4/5, KenCarp3/4/47/5/58, the six ESDIRK
L[2]SA, ARS222-443, BHR553, CFNLIRK3) reuse the previous step's last stage
losslessly, exactly like the wave-6/7 ESDIRK/ARK precedent. The 9 with an
implicit first stage (SSPSDIRK2, Cash4, Hairer4, Hairer42, SFSDIRK4-8)
refuse `priorStep` with the standard warning. (Hairer4/42 are stiffly
accurate but cannot open from a reused k0, so accuracy there buys
nothing.)

**Embedded pairs: 17 (p, p-1) pairs.** Kvaerno3-5, KenCarp3-58, the six
ESDIRK L[2]SA, Cash4, Hairer4, Hairer42 carry an embedded weight vector
(`result.error` is the branch difference, O(dt^p) = dt^order, pinned in
`tests/test_embedded.py`). The other 12 register plain (no error
estimate). Three pairs sit above their asymptotic rate on the standard
grid (pre-asymptotic overshoot of the (p, p-1) difference; KenCarp5's
finest magnitude 2.7e-15 is also at the roundoff floor): KenCarp5
6.20/6.12/(floor), ESDIRK4(3)7L[2]SA 4.77/4.69/4.55, ESDIRK6(5)9L[2]SA
6.27/6.14/(6.09 at the floor) -- they run the standard grid under the
one-sided band (`EMBEDDED_GRID`), which admits the overshoot but nothing
below half an order.

**Convergence grids: ESDIRK6(5)9L[2]SA floors out.** The order-6 method's
standard-grid errors are roundoff-contaminated: on the oscillator the two
finest points (8.8e-14, 5.7e-15) drag the measured order to 5.84 (below
the 6 - 0.15 bound), and on forced the error is at the floor from
2.8e-13 down (no measurable order). `PROBLEM_COARSE_GRIDS` gives it
[0.5, 0.25, 0.125, 0.0625] on both, where all four points sit above the
floor and measure 5.89 / 6.00. Damped passes on the standard grid
(6.04).

**Test-machinery fallout from the new tableaus.** (1) The stage-time shift
test (`test_stage_times_shift_by_dt_between_steps`) compares the raw f-call
list between steps; JFNK re-evaluates f at the stage time once per solver
iteration, and the iteration count can differ between steps (a cold-start
step takes a different count -- KenCarp47/58 and SFSDIRK5-8 showed it:
5 vs 7 at one stage). The stage-time PATTERN is invariant; the iteration
count is a solver property, so `_stage_times` now collapses consecutive
repeats of one time. (2) Stage-time slacks for the wave-8 nodes outside
[0, 1]: ESDIRK4(3)6L[2]SA2 c4 = 1.046923, Kvaerno5 c2 = 1.230333,
BHR553 c3 = 1.5, Cash4 has a later node c1 = -0.7, while its first node is
c0 = 0.4358665 (the stage nodes are not chronological). (3) The nine implicit-first-stage schemes join
`DRIFT_BEFORE_FIRST_EVALUATION` (their first f evaluation is at
t^n + c[0]*dt, c[0] in [0.05, 0.436]).

**Registry: 108 -> 137; DIRK 7 -> 36** (`tests/test_family_enums.py`),
every scheme picked up by the generic `scheme` fixture (state, convergence,
hamiltonian, history, kwargs) with no per-scheme skips.

**Stored stage counts.** The registration table counts the upstream stored
stages, including explicit null stages and endpoint rows. Kvaerno3/4/5,
KenCarp3/4/5, ARS222/443, Cash4, CFNLIRK3 and Hairer4/42 have extra stored
stages relative to the published compact forms. Removing those extras leaves
the same scalar stability function. ARS222 and ARS232 have identical stored
implicit tableaus here; their distinct upstream additive methods are not
implemented by these pure DIRK registrations. KenCarp and ARS names also
refer to the implicit half only; the existing ARK drivers provide the split
explicit/implicit interface.

**Coefficient precision and test grids.** Cash4 and SFSDIRK5/7/8 have rounded
upstream main weights whose sums differ from 1 by up to 7.8e-12. Only the
last main weight is corrected to make the sum exactly 1, as recorded beside
the coefficient arrays; the a/c entries and companion weights are preserved.
The default-solve coarse grids in `tests/conftest.py` keep errors above the
solver/roundoff floor (Cash4 and SFSDIRK5/7/8 on affected problems).
ESDIRK4(3)7L[2]SA is superconvergent on forced (order 5): its DIRK order test
has a documented one-sided cell, while oscillator and Kepler pin order 4.

The final Kepler investigation separated tableau order from solver error:

| Scheme | Default solver, standard grid | Strict solver, standard grid | Default solver grid / order |
|---|---:|---:|---|
| ESDIRK4(3)7L[2]SA | 3.394 | 3.998 | 0.2, 0.1, 0.05, 0.025 / 4.009 |
| ESDIRK6(5)9L[2]SA | 5.003, nonmonotone | 5.998 | 0.4, 0.2, 0.1, 0.05 / 5.966 |
| SFSDIRK6 | 3.835 | 3.999 | 0.4, 0.2, 0.1, 0.05 / 3.990 |

Strict probes use `tol=gmres_tol=newton_tol=1e-14, fd_eps=1e-6`.
The standard grid is 0.1, 0.05, 0.025, 0.0125 over T=2. The order band is
unchanged; both default coarse-grid and strict standard-grid regressions run.
Monotonicity uses the same justified per-problem grids as order measurement.

**Deterministic reuse cost.** Total RHS counts vary with JFNK iterations
(the measured difference was -7 to +3 per step). The explicit first-stage
reuse gate counts only calls at the step-start time: 20 without reuse, 1
with reuse over 20 steps, saving exactly 19 initial evaluations.

**SSPSDIRK2 geometry.** Its tableau is `a=[[1/4,0],[1/2,1/4]]`,
`b=[1/2,1/2]`: two implicit-midpoint half-steps. The symplectic RK defect
`b_i*a_ij + b_j*a_ji - b_i*b_j` is exactly zero, and
`R(z)=((1+z/4)/(1-z/4))**2` tends to **+1**. It is A-stable, not L-stable,
and registers `dissipation=False`. Direct tableau, scalar-amplification,
phase-area and nonlinear Kepler-form tests support that classification.

**TVD/SSP measurements.** All 29 additions pass the trajectory sweep to the
CFL-5 cap on the step-advection model; this finite sweep is not a proof of
unconditional TVD or general SSP. The all-stage convex-combination scan to
CFL 4 measures 0 for most, 0.121673 for Kvaerno5, 0.111762 for
ESDIRK3(2)5L[2]SA, 2.414214 for ARS222/232, 0.136672 for Hairer4, and
0.092749 for Hairer42. SSPSDIRK2 and SFSDIRK4-8 reach the scan cap (lower
bounds). These model-specific verdicts are pinned per scheme in
`tests/test_tvd.py`, with an independent cheap registry-completeness gate.

**Adaptive and gradient gate.** Driver/tableau metadata selects all 45
estimate emitters for adaptive-loop tests. Moderate tolerances avoid the
float64 estimator floor; ROS3P/EXPRB32 use forced because their estimates
vanish on a linear autonomous problem. The implicit gradient inventory is
registry-driven with startup-aware run lengths. ROS3P is explicitly covered
by its dedicated frozen-operator gradient tests. All 36 DIRKs additionally
compare captured RHS-parameter gradients against strict-solve finite differences.
State roots enable the JFNK differentiable path in those parameter tests.

The 78 ordinary explicit registrations also receive state-gradient checks,
with metadata-sized trajectories that pass multistep startup. The stage-count
RKC1/RKC2/RKL2 drivers have diffusion finite-difference gradient checks;
the exponential drivers retain their dedicated Krylov parity checks.
Together with ROS3P's dedicated suite, every registered driver has coverage.

The fresh full gate initially found four additional failures: default-solve
noise in the exact FSAL comparisons for KenCarp5 (1.63e-11),
ESDIRK5(4)I8L[2]SA (1.20e-10) and ESDIRK6(5)9L[2]SA (1.50e-11), plus
SSPSDIRK2's linear phase-area defect (7.36e-7). Strict stage solves
(`tol=gmres_tol=newton_tol=1e-12, fd_eps=1e-6`) resolve all four without
changing the equality or geometry tolerances. Default-solve reuse order,
energy behavior and nonlinear geometry checks remain active. The focused
rerun passes 247 cases with eight existing property exclusions.

Implementation, documentation and validation are included in the bundled
wave 8 / Phase 1 commit. Package publication remains a separate release step.

---
