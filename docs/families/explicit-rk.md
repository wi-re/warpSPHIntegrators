# Explicit Runge-Kutta (`ExplicitRK`) — 21 schemes

One-step explicit RK on the generic Butcher tableau, driven by the single
`butcher.RungeKuttaB` path: a stage $k_i = f(t^n + c_i\,\mathrm{dt},
y^n + \mathrm{dt}\,\sum_{j<i} a_{ij} k_j)$, then
$y^{n+1} = y^n + \mathrm{dt}\,\sum_i b_i k_i$. One step costs $s$ right-hand-side
evaluations ($s-1$ under first-stage reuse for the FSAL tableaus below). No
solve, no matrix — this is the family for non-stiff dynamics.

Tableaus are stored as `butcherTableau` NamedTuples (`.a`, `.b`, `.c`); embedded
pairs store `.b` as a tuple `(b_main, b_embedded)` and the step returns the pair
difference as `IntegrationResult.error` for step-size control (see
[solver.md](../solver.md)). Every tableau below is the one in `butcher.py`,
transcribed exactly.

## Order 1

### Forward Euler

$$
\begin{array}{c|c}
0 & \\
\hline
& 1
\end{array}
$$

Stability interval $[-2, 0]$ on the real axis. FSAL-eligible (the one stage
*is* the next step's first stage; reuse costs no order). The baseline.

### Explicit Euler (position/velocity form)

The first-order system form of the second-order-ODE update
$x^{n+1} = x^n + \mathrm{dt}\,v^n$, $v^{n+1} = v^n + \mathrm{dt}\,f(t^n, x^n)$
for systems in position/velocity form (`euler.integrateExplicitEuler`); the
"baseline alias" of Forward Euler for that state shape. One evaluation.

## Order 2 (three-stage-free: two evaluations)

All three two-stage order-2 tableaus below share the stability function
$R(z) = 1 + z + z^2/2$ (real-axis interval $[-2, 0]$); they differ only in
node placement.

### Midpoint (`RungeKutta2`)

$$
\begin{array}{c|cc}
0 & 0 & \\
1/2 & 1/2 & 0 \\
\hline
& 0 & 1
\end{array}
$$

### Heun 2 (`heunsMethod`)

$$
\begin{array}{c|cc}
0 & 0 & \\
1 & 1 & 0 \\
\hline
& 1/2 & 1/2
\end{array}
$$

### Ralston 2 (`ralston2nd`)

$$
\begin{array}{c|cc}
0 & 0 & \\
2/3 & 2/3 & 0 \\
\hline
& 1/4 & 3/4
\end{array}
$$

### EPEC / EPEC Modified

`EPEC` and `EPECmodified` (the pySPH *evaluate–predict–evaluate–correct*
spelling) are the same two tableaus under different names: EPEC ≡ Midpoint,
EPEC Modified ≡ Heun 2.

## Order 3 (three evaluations)

All three-stage order-3 tableaus share the stability function
$R(z) = 1 + z + z^2/2 + z^3/6$ (real axis $[-2, 0]$); the stage-3 map, which
separates them at the stage level, is what the TVD classification measures
(NOTES.md §3.11).

### Classical RK3 (`RungeKutta3`)

$$
\begin{array}{c|ccc}
0 & 0 & 0 & 0 \\
1/2 & 1/2 & 0 & 0 \\
1 & -1 & 2 & 0 \\
\hline
& 1/6 & 2/3 & 1/6
\end{array}
$$

### Heun 3 (`heunsMethod3rd`)

$$
\begin{array}{c|ccc}
0 & 0 & 0 & 0 \\
1/3 & 1/3 & 0 & 0 \\
2/3 & 0 & 2/3 & 0 \\
\hline
& 1/4 & 0 & 3/4
\end{array}
$$

### Ralston 3 (`ralston3rd`)

$$
\begin{array}{c|ccc}
0 & 0 & 0 & 0 \\
1/2 & 1/2 & 0 & 0 \\
3/4 & 0 & 3/4 & 0 \\
\hline
& 2/9 & 1/3 & 4/9
\end{array}
$$

### Wray 3 (`Wray3rd`)

$$
\begin{array}{c|ccc}
0 & 0 & 0 & 0 \\
8/15 & 8/15 & 0 & 0 \\
2/3 & 1/4 & 5/12 & 0 \\
\hline
& 1/4 & 0 & 3/4
\end{array}
$$

### SSP RK3 (`SSPRK3`)

Shu-Osher third-order SSP: $R(z) = 1 + z + z^2/2 + z^3/6$, SSP coefficient
$r = 1$ (the stage maps are convex combinations exactly up to CFL 1):

$$
\begin{array}{c|ccc}
0 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1/2 & 1/4 & 1/4 & 0 \\
\hline
& 1/6 & 1/6 & 2/3
\end{array}
$$

**TVD RK3** (`TVDRK3`) is the same Shu-Osher scheme written as the hand-rolled
conservative/TVD update (no tableau object; algebraically identical — pinned to
the actual driver by `tests/test_tvd.py`). **TVD RK2** is the two-stage
Shu-Osher form, likewise hand-rolled.

## Order 4

### Classical RK4 (`RungeKutta4`)

$$
\begin{array}{c|cccc}
0 & 0 & 0 & 0 & 0 \\
1/2 & 1/2 & 0 & 0 & 0 \\
1/2 & 0 & 1/2 & 0 & 0 \\
1 & 0 & 0 & 1 & 0 \\
\hline
& 1/6 & 1/3 & 1/3 & 1/6
\end{array}
$$

$R(z)$ = the 4th-degree Taylor polynomial; real-axis stability interval
$[-2.785, 0]$; SSP coefficient $r = 2/3$, per-step TVD measured to CFL 1.25.

### RK4 (3/8 rule, `RungeKutta4alt`)

$$
\begin{array}{c|cccc}
0 & 0 & 0 & 0 & 0 \\
1/3 & 1/3 & 0 & 0 & 0 \\
2/3 & -1/3 & 1 & 0 & 0 \\
1 & 1 & -1 & 1 & 0 \\
\hline
& 1/8 & 3/8 & 3/8 & 1/8
\end{array}
$$

Same stability function as RK4 (all order-4 4-stage RKs share it).

### SSPRK(10,4) (`SSPRK104`)

Shu's 10-stage order-4 SSP method (Shu, *J. Comput. Phys.* 169 (2001) 208-228,
Sec. 4.2) — the highest-order SSP scheme in the registry, added because the
SSP barrier for explicit RK bites exactly at order 4 (5+ stages required).
Measured: exact SSP coefficient $r = 6.0$ (stage 2's constant coefficient
$1 - \mu/6$ is the binding constraint) and real-axis interval
$[-13.916, 0]$. The tableau is the 5+5 cascade in `butcher.py`
(`getButcherTableau('SSPRK104')`); the final row is
$b = (1/10)\,\mathbf{1}$, $c = (0, 1/6, 1/3, 1/2, 2/3, 1/3, 1/2, 2/3, 5/6, 1)$.

## Order 5 and the embedded pairs

### Nystrom 5th (`Nystrom5th`)

The 6-stage 5th-order Runge-Kutta-Nystrom scheme for the position/velocity
form, $b = (23/192,\, 0,\, 125/192,\, 0,\, -27/64,\, 125/192)$,
$c = (0, 1/3, 2/5, 1, 2/3, 4/5)$ (the full $a$ matrix in `butcher.py`); six
force evaluations per step, one per stage. The fifth-order reference method;
notoriously ill-scaled coefficients (its SSP coefficient is exactly $r = 0$ —
not a convex-combination step at *any* CFL — yet it is per-step TVD up to
CFL 1.5, NOTES.md §3.11).

### Bogacki-Shampine 3(2) (`BogackiShampine`)

Order 3 / embedded 2, 4 stages, **FSAL** (last stage $= f(t^{n+1}, y^{n+1})$):
$y^{n+1} \leftarrow b = (2/9, 1/3, 4/9, 0)$,
$\hat b = (7/24, 1/4, 1/3, 1/8)$,
$a$-rows $(0), (1/2, 0), (0, 3/4), (2/9, 1/3, 4/9)$, $c = (0, 1/2, 3/4, 1)$.
scipy's RK23. 3 effective evaluations per step under `priorStep` reuse.

### Cash-Karp 5(4) (`CashKarp`)

Order 5 / embedded 4, 6 stages, **not FSAL** ($c_6 = 7/8$):
$b = (37/378, 0, 250/621, 125/594, 0, 512/1771)$,
$\hat b = (2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4)$,
$c = (0, 1/5, 3/10, 3/5, 1, 7/8)$; the full $a$ matrix in `butcher.py`.
A well-conditioned 6-stage alternative when Dormand-Prince is more than
needed.

### Dormand-Prince 5(4) (`DormandPrince`)

Order 5 / embedded 4, 7 stages, **FSAL**: $b = (35/384, 0, 500/1113, 125/192,
-2187/6784, 11/84, 0)$,
$\hat b = (5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40)$,
$c = (0, 1/5, 3/10, 4/5, 8/9, 1, 1)$; the full $a$ matrix in `butcher.py`.
scipy's RK45 / MATLAB's ode45. 6 effective evaluations per step under reuse.
Like Nystrom, $r = 0$ exactly (the stage-5/6 circulant entries are genuinely
negative for all $\mu > 0$; NOTES.md §3.11), yet per-step TVD to CFL 1.5.
This is also the **cold-start scheme** the multistep families bootstrap from
(NOTES.md §3.7).

## First-stage reuse

`priorStep=result.stages[-1]` feeds the previous step's last stage as this
step's first stage, saving one evaluation. It is lossless exactly for the FSAL
tableaus (Bogacki-Shampine, Dormand-Prince — and Forward Euler); for every
other tableau it costs order (the reuse analysis in `reuse.step_reuse_analysis`
reports the exact reuse order per scheme; pinned in `tests/test_step_reuse.py`).
Multistep and implicit families use different mechanisms — see their pages.

## SSP / TVD landscape (measured)

The registry-wide classification from `scripts/tvd_classifier.py`
(NOTES.md §3.11; $n = 64$, step IC, CFL grid to 5):

| Scheme | SSP coefficient $r$ (convex-combination CFL) | Measured per-step TVD CFL |
|---|---|---|
| Forward Euler, Midpoint, Heun 2, Ralston 2, Heun 3, Ralston 3, Wray 3, SSP RK3, Bogacki-Shampine, EPEC, EPEC Modified, TVD RK2, TVD RK3 | 1 | 1 |
| RK3 | 0.5 | 1 |
| RK4 | 2/3 | 1.25 |
| RK4 (alternative) | 1/3 | 1.25 |
| Cash-Karp 5(4) | 5/12 | 1.5 |
| Nystrom 5th, Dormand-Prince 5(4) | **0** | 1.5 |

TVD is strictly broader than SSP: RK4 is per-step TVD past where its internal
stages leave the convex hull, and Nystrom/Dormand-Prince are TVD to CFL 1.5
despite $r = 0$. The stage-level separation is what distinguishes RK3 from TVD
RK3 (RK3's stage-3 map $1 + \zeta + \zeta^2$ has a negative circulant entry at
CFL 1; TVD RK3's stage rows stay non-negative).

## References

- Hairer, Nørsett, Wanner, *Solving Ordinary Differential Equations I* (1993) — RK theory, stability, embedded pairs.
- Shu, *Total variation diminishing Runge-Kutta time discretizations*, J. Comput. Phys. 169 (2001) 208-228 — SSPRK(10,4).
- Shu & Osher, *Efficient implementation of essentially non-oscillatory shock capturing schemes*, J. Comput. Phys. 79 (1989) 500-512 — SSP/TVD RK2, RK3.
- Bogacki & Shampine, *A 3(2) pair of Runge-Kutta formulas*, BIT 35 (1995) 232-248.
- Dormand & Prince, J. Comput. Appl. Math. 8 (1981) 15-28; Cash & Karp, NASA TM-100035 (1990).

**Tests:** `tests/test_convergence.py` (orders), `tests/test_step_reuse.py`
(reuse orders), `tests/test_tvd.py` (the table above),
`tests/test_embedded.py` (embedded-pair rates), `tests/test_hamiltonian.py`
(energy drift on the oscillator).
