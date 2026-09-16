# Diagonally implicit RK (`DIRK`) — 7 schemes

One-step methods on a **triangular** Butcher tableau driven by `dirk.DIRK`:
each stage with a nonzero diagonal $a_{ii}$ is closed as one nonlinear solve

$$Y_i = y^n + \mathrm{dt}\,\sum_{j<i} a_{ij}k_j \;+\; \mathrm{dt}\,a_{ii}\, k_i(Y_i), \qquad k_i = f(t^n + c_i\,\mathrm{dt},\, Y_i),$$

i.e. the fixed point of $G(Y) = \text{base} + \mathrm{dt}\,a_{ii}\,f(\cdot, Y)$,
solved by `JFNKSolver` by default (a fixed-count Picard `FixedPointSolver` is
an explicit low-overhead override; see [solver.md](../solver.md)). Stages with
$a_{ii} = 0$ are explicit. The driver reuses the explicit-RK machinery for the
explicit part, the $b$-weighted final update, and embedded-pair error
estimation; the only new piece is the stage solve.

## The tableaus

All seven are in `dirk.getDIRKTableau`, verified against their order
conditions (symbolic Taylor expansion) and, for the L-stable ones, their exact
stability functions (NOTES.md §3.6).

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
"strict solve for geometry". One solve per step. Caveat (measured): under the
*default* low-overhead Picard solve it measures dissipative — the symplecticity
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
feeds the adaptive helpers (see [solver.md](../solver.md)). Explicit first
stage + stiffly accurate $\Rightarrow$ lossless `priorStep` reuse.

### ESDIRK3(2)4L[2]SA (`ESDIRK324L2SA`)

Kennedy-Carpenter's 4-stage ESDIRK, order 3 / embedded 2, explicit first
stage, **A- and L-stable with L[2] damping** (stiff modes decay like
$O(z^{-2})$ as $z \to -\infty$), stiffly accurate. The single $\gamma$-type
node is

$$g = 0.4358665215084589994160194511935568425293,$$

with $c = (0, 2g, 0.6, 1)$; the full $a$ matrix and the (3, 2) embedded pair
are in `dirk.py` (Kennedy-Carpenter coefficients as published in SUNDIALS
ARKODE v7.9.0, `arkode_butcher_dirk.def`). Cost: one solve + three explicit
evaluations per step (two under reuse).

### ESDIRK4(3)6L[2]SA (`ESDIRK436L2SA`)

Kennedy-Carpenter's 6-stage ESDIRK, order 4 / embedded 3, same property set
(A- and L-stable, L[2], stiffly accurate, explicit first stage). Carried in
`dirk.py` in its exact radical/rational form:
$c = (0, 1/2, (2-\sqrt2)/4, 5/8, 26/25, 1)$, all $a_{ii} = 1/4$. The
highest-order implicit one-step scheme in the registry.

## First-stage reuse (the DIRK rule)

`priorStep` is lossless **iff** the tableau has an explicit first stage
($a_{11} = 0$, $c_1 = 0$) *and* is stiffly accurate ($c_s = 1$,
$a_{s\cdot} = b$) — then the previous step's last stage *is*
$f(t^{n+1}, y^{n+1})$, exactly what this step's first stage needs. That is
Trapezoidal, TR-BDF2, and the two ESDIRKs. Every other DIRK tableau (SDIRK2's
implicit first stage; backward Euler and implicit midpoint's single implicit
stage) **refuses** `priorStep` with the standard warning. The rule is derived
from the tableau itself (`_dirk_reuses_first_stage`), so it cannot drift from
the registered `stiffly_accurate` metadata.

## Stability and TVD

| Scheme | A-stable | L-stable | L[2] | Stiffly accurate |
|---|---|---|---|---|
| Backward Euler | yes | yes | — | no |
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

## References

- Kennedy & Carpenter, *Additive Runge-Kutta schemes for stiff problems*, J. Comput. Phys. 181 (2002) 362-415 — the ESDIRK/TR-BDF2 designs (as shipped in SUNDIALS ARKODE v7.9.0).
- Hairer & Wanner, *Solving ODEs II* (1996), Ch. 4 — DIRK theory, L-stability, L[$q$] damping.
- Lambert (2005) — TR-BDF2.

**Tests:** `tests/test_dirk.py` (order conditions, stability functions, the
reuse rule), `tests/test_stiff.py` (stiff damping), `tests/test_embedded.py`
(embedded-pair rates), `tests/test_tvd.py` (the TVD row above).
