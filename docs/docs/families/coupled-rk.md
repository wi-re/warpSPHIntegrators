---
sidebar_label: Coupled RK
---

# Coupled (block) fully implicit RK (`CoupledRK`) — 2 schemes

The fully implicit RK family, driven by `fullyimplicit.py`: Gauss-Legendre 2
and Radau IIA $s=2$. These are the collocation methods whose Butcher $a$
matrices are **coupled** ($a_{ij} \neq 0$ for $j > i$) — a DIRK driver cannot
take them, because no stage is a function of already-known quantities alone.

## The block driver

A step is one **$s \times s$ block solve**. The $s$ stage states are packed
into a `fields.BlockState` (substate-major), and the block residual

$$
Y_i - y^n - \mathrm{dt}\,\sum_{j=1}^{s} a_{ij}\, k_j(Y_1, \dots, Y_s) = 0,
\qquad k_j = f(t^n + c_j\,\mathrm{dt},\, Y_j)
$$

is solved by `JFNKSolver` in a single call per step (the default matvec is the
**exact JVP** of the block map, `matvec='jvp'` — one forward pass, no
`fd_eps`). One step costs $s$ RHS evaluations inside the solve plus the
Krylov matvecs.

Driver conventions (NOTES.md §3.18):

- `history=` is bookkeeping only; `priorStep=` is **refused** — a coupled
  tableau has no stage that can be consumed alone, and the reuse analyses in
  `reuse.py` both assume a stage that can be solved without the others.
  `warmStart=` accepts an explicit previous block iterate instead (a caller
  that can prove the warm start is valid).
- The step's `SolveDiagnostics` is scaled by $s$ (`rhs_evaluations` and
  `gmres_iterations` × $s$), so the work-unit invariant
  `total = rhs_evaluations + gmres_iterations` keeps counting physical work
  at the per-step level (the user's confirmed choice).
- Each scheme carries a **null-stage order-2 companion**: the 2-stage-only
  order-2 embedded pair is degenerate (it is $b$ itself), so the companion
  quadrature is taken over $(k_0, k_1, k_2)$ with the null stage $k_0 =
  f(t^n, y^n)$ at $c_0 = 0$ (min-norm choice). The companion's $b_0 \neq 0$
  costs one extra RHS evaluation per step; the embedded rate measures 3.0 for
  both schemes. `IntegrationResult.error` is the state-shaped difference
  `state_difference(y_main, y_embed)` (the same convention the adaptive
  helpers use; [solver](../solver)).

## Gauss-Legendre 2 (`Gauss-Legendre 2`)

The collocation method at the Gauss points $c = \tfrac12 \pm \tfrac{\sqrt3}{6}$:

$$
a = \begin{pmatrix} 1/4 & \frac{3 - 2\sqrt3}{12} \\[2pt] \frac{3 + 2\sqrt3}{12} & 1/4 \end{pmatrix}
\quad (\approx \begin{pmatrix} 0.25 & -0.0387 \\ 0.5387 & 0.25 \end{pmatrix}),
\qquad b = (1/2, 1/2).
$$

Order **4**. A-stable with $|R(iy)| = 1$ on the imaginary axis; **not
L-stable**:

$$
R(z) = \frac{1 + z/2 + z^2/12}{1 - z/2 + z^2/12}, \qquad
R(-100) = \frac{2353}{2653} \approx 0.887, \qquad R(z) \to +1 \text{ as } z \to -\infty.
$$

Not stiffly accurate ($c_2 \neq 1$). Companion:
$\bar b = (1/6,\ (5-\sqrt3)/12,\ (5+\sqrt3)/12)$.

**Symplectic for separable Hamiltonians — pinned by measurement, not by the
folklore argument**: the tableau is *not* A-symmetric ($a_{12} \neq a_{21}$),
so the standard "A-symmetric collocation ⇒ symplectic" shortcut does not
apply; the library measures it instead. Strict solve: phase-area defect
$3.05\times10^{-13}$ (oscillator), Kepler-form defect $4.87\times10^{-9}$
(default solve) — `tests/test_hamiltonian.py`. It is the library's only
symplectic method above order 2 (the position/velocity family tops out at
order 4 with PEFRL/VEFRL, which are geometric compositions rather than
collocation).

The negative $a_{12}$ makes the implicit solution **non-monotone**: on the
256-DOF advection scale gate (CFL 100, step IC) the true method value is
$\max|u| = 1.6903$ — a bounded Gibbs-like overshoot of the exact
$\max|u| = 1$, with every per-mode $|R(z)| \le 1$ (the matrix-free gate pins
this; NOTES.md §3.18).

## Radau IIA $s=2$ (`Radau IIA s=2`)

The collocation method at the Radau points $c = (1/3, 1)$ (roots of
$3x^2 - 2x - 1$; the right endpoint is pinned, so the $s=1$ limit is exactly
backward Euler):

$$
a = \begin{pmatrix} 5/12 & -1/12 \\ 3/4 & 1/4 \end{pmatrix},
\qquad b = (3/4, 1/4) \ (= \text{last row of } a).
$$

Order **3**, A- and **L-stable**, **stiffly accurate**:

$$
R(z) = \frac{1 + z/3}{1 - 2z/3 + z^2/6}, \qquad
|R(-10)| = 0.095890, \qquad R(-100) \approx 0.019.
$$

Companion: $\bar b = (2/7,\ 9/28,\ 11/28)$. Not symplectic (measured: linear
area defect $2.44\times10^{-2}$; absent from the symplectic sets in
`tests/test_hamiltonian.py`). On the stiff scalar $y' = -100y$ it damps the
mode to $7\times10^{-11}$ in a **single** step, versus $\sim 4$ steps for
Gauss-Legendre 2 (the L-stability gap in action; `tests/test_stiff.py`).

## Cost and the GMRES finding

The block system couples all $s$ substates, so its Jacobian spectrum is
**spread** (the 256-DOF advection gate: dense $1536\times1536$ block
Jacobian, condition number 88.6, eigenvalues in $[0.93, 82.5]$ including 128
complex pairs on a two-dimensional annulus). Consequence pinned in
`tests/test_fullyimplicit.py`: restarted GMRES (restart 30/60) **stagnates**
on the block system, while no-restart GMRES converges in 256 iterations to
$10^{-12}$ — the scale gate uses `gmres_restart` ≥ system size and asserts
`termination == 'tolerance'`. SciPy's `gmres` reproduces the same split
(restart 30: info 1536, residual $1.1\times10^{-3}$; no restart: info 0,
residual $8.3\times10^{-13}$).

Benchmark: `scripts/blockrk_benchmark.py` → `images/blockrk_benchmark.png`
— (1) energy drift vs. $T$ on the oscillator (GL2 flat at JFNK-noise level,
RK4 secular); (2) stiff damping at $z = -10$ across the implicit family
(Radau 0.0959, BDF5 0.65, GL2 0.30, trapezoidal 0.67); (3) the order ladder
(GL2 3.98, Radau 3.00, vs. the explicit/implicit references).

## References

- Hairer & Wanner, *Solving ODEs II* (1996), Ch. IV — Gauss and Radau IIA collocation, their stability functions, symplecticity of Gauss methods.
- Radau, I. (1985) — the Radau IIA construction (as summarized in Hairer-Wanner).
- The tableaus are **not** in the SUNDIALS ARKODE v7.9.0 extract this repo cites elsewhere; the derivations (order conditions, collocation identity, stability functions) are in NOTES.md §3.18.

**Tests:** `tests/test_fullyimplicit.py` (32 tests: tableaus, order, the block
JVP against the hand-computed dense Jacobian, the scale gate, companions,
warm-start, the GMRES finding), `tests/test_hamiltonian.py` (GL2
symplecticity), `tests/test_stiff.py` (Radau L-damping), `tests/test_tvd.py`
(both: no SSP coefficient — `blockTableau` is not read by the tableau
classifier — measured per-step TVD to CFL 5).
