---
sidebar_label: DIRK
---

# Diagonally implicit RK (`DIRK`) — 37 schemes

One-step methods on a **triangular** Butcher tableau driven by `dirk.DIRK`:
each stage with a nonzero diagonal $a_{ii}$ is closed as one nonlinear solve

$$
Y_i = y^n + \mathrm{dt}\,\sum_{j<i} a_{ij}k_j \;+\; \mathrm{dt}\,a_{ii}\, k_i(Y_i), \qquad k_i = f(t^n + c_i\,\mathrm{dt},\, Y_i),
$$

i.e. the fixed point of $G(Y) = \text{base} + \mathrm{dt}\,a_{ii}\,f(\cdot, Y)$,
solved by `JFNKSolver` by default (a fixed-count Picard `FixedPointSolver` is
an explicit low-overhead override; see [solver](../solver)). Stages with
$a_{ii} = 0$ are explicit. The driver reuses the explicit-RK machinery for the
explicit part, the $b$-weighted final update, and embedded-pair error
estimation; the only new piece is the stage solve.

## The tableaus

### Parallel DIRK (`ParallelDIRK()`)

The additional `ParallelDIRK()` factory implements the pinned PDIRK44 map
outside the 37 fixed DIRK entries. It solves ordinary first-order ODEs with four
implicit stages, two independent pairs, and order four. Its stability function is
`R(z)=(z+3)*(9*z*z-20*z+12)/((z-2)**2*(2*z-3)**2)`: A-stable and L-stable.
It has no embedded estimate, FSAL reuse or SSP guarantee. Coupled first-order
quantities follow the same RK weights.

This backend executes the stages serially using JFNK, with the usual DIRK
solver and preconditioner options. Its independent groups remain available for
future execution backends; there is no internal-threading speed claim. A fifth,
zero-weight endpoint evaluation refreshes copied fields. Use fixed timesteps
and resupply solver options when continuing from a checkpoint.

On 4,096 nonlinear forced reaction modes, sixteen steps of size .0625 gave
maximum error 4.80e-6 with 790 actual RHS calls. Radau IIA with two stages gave
1.34e-5 with 422 calls; the six-stage ESDIRK4 method gave 1.75e-7 with 880 calls.
These are fixed-step comparisons, not equal-error speed rankings. They support
PDIRK as another four-stage L-stable option rather than a new default. See
[source and numerical validation](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/parallel-dirk-validation.md).

All 37 fixed records are in `dirk.getDIRKTableau`, covered by convergence, tableau consistency, reuse,
stability, adaptive and gradient tests (NOTES.md §§3.6 and 3.28).

### MEBDF2 (`mebdf2`)

The modified extended BDF2 step is a three-stage SDIRK with
`A=[[1,0,0],[1,1,0],[1/2,-1/2,1]]`, `b=A[-1]`, `c=[1,2,1]`.
It needs three nonlinear solves and no step history. Its second stage evaluates
at t+2h, beyond the accepted endpoint. Order two, stiffly accurate and L-stable:
`R(z)=(1-2*z+z*z/2)/(1-z)**3`. There is no embedded estimate and no SSP guarantee.
The pinned-source nonlinear step and known-consumer evidence are in
[the backend record](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/implicit-extensions-validation.md).

### Backward Euler (`backwardEuler`)

$$
\begin{array}{c|c}
1 & 1 \\
\hline
& 1
\end{array}
\qquad R(z) = \frac{1}{1-z}
$$

Order 1, **L-stable** ($R(z)\to 0$ as $z\to-\infty$). One solve per step. The
stiffness workhorse; on a stiff scalar problem with $z = -10$ it damps the
mode to $|R(-10)| = 0.0909$ in a single step. Unconditionally TVD (the Fejer
kernel; NOTES.md §3.11).

### Implicit Midpoint (`implicitMidpoint`)

$$
\begin{array}{c|c}
1/2 & 1/2 \\
\hline
& 1
\end{array}
\qquad R(z) = \frac{1 + z/2}{1 - z/2}
$$

Order 2, **A-stable**, symmetric. It is the Gauss-Legendre $s = 1$ collocation
method, so for separable Hamiltonians it is symplectic *once its stage equation
is solved* — the README's symplectic column marks it `yes` with the qualifier
"strict solve for geometry". One solve per step. Caveat (measured): under an
explicitly selected low-overhead Picard solve it measures dissipative — the symplecticity
is a property of the converged map, not of two Picard iterates (NOTES.md §3.6
finding).

### Trapezoidal / Crank-Nicolson (`trapezoidal`)

Lobatto IIIA-2: two stages, explicit first, one solve:

$$
\begin{array}{c|cc}
0 & 0 & 0 \\
1 & 1/2 & 1/2 \\
\hline
& 1/2 & 1/2
\end{array}
\qquad R(z) = \frac{1 + z/2}{1 - z/2}
$$

Order 2, **A-stable and symmetric** (the Cayley transform — the same stability
function as implicit midpoint, so no damping on the imaginary axis: oscillatory
stiff modes are *not* damped, only bounded). The linear oscillator's map
preserves area exactly (the "linear only" symplectic flag). **FSAL-eligible**:
explicit first stage + stiffly accurate, so `priorStep` reuse is lossless.
SSP coefficient $r = 2$ (NOTES.md §3.11).

### SDIRK2 (`SDIRK2`)

Ellsiepen's L-stable two-stage SDIRK, $\gamma = 1 - \sqrt{2}/2$:

$$
\begin{array}{c|cc}
\gamma & \gamma & 0 \\
1 & 1-\gamma & \gamma \\
\hline
& 1-\gamma & \gamma
\end{array}
$$

Order 2, **L-stable**, stiffly accurate; both stages implicit (two solves per
step). The order-2 condition $\sum b_i c_i = 1/2$ holds exactly for this
$\gamma$. SSP coefficient $r = 1+\sqrt{2}$.

### TR-BDF2 (`TRBDF2`)

A trapezoidal substep at $\gamma = 2 - \sqrt{2}$ followed by a variable-step
BDF2 endpoint solve, written as a 3-stage SDIRK (both implicit diagonals equal
$(1-\gamma)/(2-\gamma)$):

$$
\begin{array}{c|ccc}
0 & 0 & 0 & 0 \\
\gamma & \gamma/2 & \gamma/2 & 0 \\
1 & \frac{1}{2(2-\gamma)} & \frac{1}{2(2-\gamma)} & \frac{1-\gamma}{2-\gamma} \\
\hline
& \frac{1}{2(2-\gamma)} & \frac{1}{2(2-\gamma)} & \frac{1-\gamma}{2-\gamma}
\end{array}
$$

Order 2, **L-stable and stiffly accurate**. Carries an embedded (2, 3) pair —
SUNDIALS ARKODE's published `d` vector, an $O(\mathrm{dt}^3)$ estimate, so it
feeds the adaptive helpers (see [solver](../solver)). Explicit first
stage + stiffly accurate $\Rightarrow$ lossless `priorStep` reuse.

### ESDIRK3(2)4L[2]SA (`ESDIRK324L2SA`)

Kennedy-Carpenter's 4-stage ESDIRK, order 3 / embedded 2, explicit first
stage, **A- and L-stable with L[2] damping** (stiff modes decay like
$O(z^{-2})$ as $z \to -\infty$), stiffly accurate. The single $\gamma$-type
node is

$$
g = 0.4358665215084589994160194511935568425293,
$$

with $c = (0, 2g, 0.6, 1)$; the full $a$ matrix and the (3, 2) embedded pair
are in `dirk.py` (Kennedy-Carpenter coefficients as published in SUNDIALS
ARKODE v7.9.0, `arkode_butcher_dirk.def`). Cost: three implicit stage solves and one explicit
first-stage evaluation per step (the explicit evaluation is saved under reuse).

### ESDIRK4(3)6L[2]SA (`ESDIRK436L2SA`)

Kennedy-Carpenter's 6-stage ESDIRK, order 4 / embedded 3, same property set
(A- and L-stable, L[2], stiffly accurate, explicit first stage). Carried in
`dirk.py` in its exact radical/rational form:
$c = (0, 1/2, (2-\sqrt2)/4, 5/8, 26/25, 1)$, all $a_{ii} = 1/4$. Wave 8 adds
DIRK methods through order 6.

## Wave 8 tableaus

The additional 29 registrations come from OrdinaryDiffEq v7.8.1's SDIRK
family. Stage counts include upstream null/endpoint stages, and every entry
uses the same DIRK driver. `ESDIRK4(3)6L[2]SA2` is distinct from the existing
`ESDIRK4(3)6L[2]SA`. The coefficient/source records are in `dirk.py` and
NOTES.md §3.28.

| Scheme | Order | Stored stages | Pair | Stability | Lossless reuse |
|---|---:|---:|---|---|---|
| Kvaerno3 | 3 | 4 | 3(2) | L | yes |
| Kvaerno4 | 4 | 5 | 4(3) | A | yes |
| Kvaerno5 | 5 | 7 | 5(4) | L | yes |
| KenCarp3 | 3 | 4 | 3(2) | L | yes |
| KenCarp4 | 4 | 6 | 4(3) | L | yes |
| KenCarp47 | 4 | 7 | 4(3) | L | yes |
| KenCarp5 | 5 | 8 | 5(4) | L | yes |
| KenCarp58 | 5 | 8 | 5(4) | L | yes |
| ESDIRK3(2)5L[2]SA | 3 | 5 | 3(2) | L | yes |
| ESDIRK4(3)6L[2]SA2 | 4 | 6 | 4(3) | L | yes |
| ESDIRK4(3)7L[2]SA | 4 | 7 | 4(3) | L | yes |
| ESDIRK5(4)7L[2]SA2 | 5 | 7 | 5(4) | L | yes |
| ESDIRK5(4)I8L[2]SA | 5 | 8 | 5(4) | L | yes |
| ESDIRK6(5)9L[2]SA | 6 | 9 | 6(5) | L | yes |
| SSPSDIRK2 | 2 | 2 | — | A | no |
| ARS222 | 2 | 3 | — | L | yes |
| ARS232 | 2 | 3 | — | L | yes |
| ARS343 | 3 | 4 | — | L | yes |
| ARS443 | 3 | 5 | — | L | yes |
| BHR553 | 3 | 5 | — | L | yes |
| Cash4 | 4 | 5 | 4(3) | L | no |
| CFNLIRK3 | 3 | 4 | — | L | yes |
| Hairer4 | 4 | 5 | 4(3) | L | no |
| Hairer42 | 4 | 5 | 4(3) | L | no |
| SFSDIRK4 | 4 | 4 | — | bounded | no |
| SFSDIRK5 | 4 | 5 | — | bounded | no |
| SFSDIRK6 | 4 | 6 | — | bounded | no |
| SFSDIRK7 | 4 | 7 | — | bounded | no |
| SFSDIRK8 | 4 | 8 | — | bounded | no |

`L` means A- and L-stable; `A` means A-stable with a nonzero stiff limit.
`bounded` for SFSDIRK4–8 means a bounded stability region; these methods
have no A-stability sector. Kvaerno4 is A-stable, not L-stable.

SSPSDIRK2 is the composition of two implicit-midpoint half-steps:
`a=[[1/4,0],[1/2,1/4]]`, `b=[1/2,1/2]`. It is symplectic and A-stable,
with `R(z)=((1+z/4)/(1-z/4))**2` and stiff limit **+1**. Its geometry
requires accurate stage solves, as for implicit midpoint.

KenCarp and ARS registrations are the **pure implicit tableaus**. Use an
ARK scheme for the implemented additive explicit/implicit pair interface.
ARS222 and ARS232 have identical stored implicit tableaus in this catalog;
their public identifiers are preserved. SDIRK22 is omitted because the
inspected upstream constructor is a trapezoidal stub.

Five upstream stored node vectors disagree with their row sums (Kvaerno3/4/5,
ESDIRK5(4)7L[2]SA2, SSPSDIRK2). This library uses row-sum nodes; it therefore
intentionally differs from that upstream execution on time-dependent RHSs.
Cash4 starts at approximately 0.4359; its later -0.7 node is outside the
step interval. Rounded main weights of Cash4 and SFSDIRK5/7/8 have their
last entry adjusted to sum exactly to one.

All 17 new embedded pairs feed the caller-driven adaptive helpers. On
nonlinear problems, solve accuracy can dominate the truncation error of the
high-order tableaus. The convergence gate retains default-solve coarse grids
and strict-solve standard-grid checks for the three measured Kepler cases.

## First-stage reuse (the DIRK rule)

`priorStep` is lossless **iff** the tableau has an explicit first stage
($a_{11} = 0$, $c_1 = 0$) *and* is stiffly accurate ($c_s = 1$,
$a_{s\cdot} = b$) — then the previous step's last stage *is*
$f(t^{n+1}, y^{n+1})$, exactly what this step's first stage needs. That covers
24 DIRKs: Trapezoidal, TR-BDF2, the two original ESDIRKs, and 20 wave 8
entries marked above. The other 12 DIRKs (implicit first stages, including
backward Euler and implicit midpoint) **refuses** `priorStep` with the standard warning. The rule is derived
from the tableau itself (`_dirk_reuses_first_stage`), so it cannot drift from
the registered `stiffly_accurate` metadata.

## Stability and TVD

| Scheme | A-stable | L-stable | L[2] | Stiffly accurate |
|---|---|---|---|---|
| Backward Euler | yes | yes | — | yes |
| Implicit Midpoint | yes | no | — | no |
| Trapezoidal | yes | no | — | yes |
| SDIRK2 | yes | yes | — | yes |
| TR-BDF2 | yes | yes | — | yes |
| ESDIRK3(2)4L[2]SA | yes | yes | yes | yes |
| ESDIRK4(3)6L[2]SA | yes | yes | yes | yes |

TVD (NOTES.md §3.11): backward Euler is unconditionally TVD (Fejer kernel);
implicit midpoint and trapezoidal have $r = 2$ (the Cayley factor is a convex
combination only for $\mu \le 2$ — *not* unconditionally TVD, despite the
A-stability); SDIRK2/TR-BDF2 have $r = 1+\sqrt{2}$; the ESDIRKs have $r = 0$
(their stages leave the convex hull immediately) but their stiffly-accurate
final maps damp so strongly that per-step TVD measures to CFL 5.

The wave 8 step-advection trajectory sweep finds no TV growth through CFL 5
for any addition. This is a finite model-specific measurement. The all-stage
convex-combination scan finds 0 for most, 0.1217 for Kvaerno5, 0.1118 for
ESDIRK3(2)5L[2]SA, 2.4142 for ARS222/232, 0.1367 for Hairer4 and 0.0927
for Hairer42. SSPSDIRK2 and SFSDIRK4–8 reach the CFL-4 scan cap (lower bounds).
The verdicts are pinned per scheme in `tests/test_tvd.py`.

## References

- Kennedy & Carpenter, *Additive Runge-Kutta schemes for stiff problems*, J. Comput. Phys. 181 (2002) 362-415 — the ESDIRK/TR-BDF2 designs (as shipped in SUNDIALS ARKODE v7.9.0).
- Hairer & Wanner, *Solving ODEs II* (1996), Ch. 4 — DIRK theory, L-stability, L[$q$] damping.
- OrdinaryDiffEq v7.8.1, `OrdinaryDiffEqSDIRK/src/sdirk_tableaus.jl` and stage caches — wave 8 coefficient source; source-specific citations accompany each array in `dirk.py`.
- Lambert (2005) — TR-BDF2.

**Tests:** `tests/test_dirk.py` (order conditions, stability functions, the
reuse rule), `tests/test_stiff.py` (stiff damping), `tests/test_embedded.py`
(embedded-pair rates), `tests/test_tvd.py` (the TVD row above).
