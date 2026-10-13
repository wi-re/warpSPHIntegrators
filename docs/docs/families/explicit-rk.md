---
sidebar_label: Explicit RK
---

# Explicit Runge-Kutta (`ExplicitRK`) — 31 schemes

One-step explicit RK on the generic Butcher tableau, driven by the single
`butcher.RungeKuttaB` path: a stage $k_i = f(t^n + c_i\,\mathrm{dt},
y^n + \mathrm{dt}\,\sum_{j<i} a_{ij} k_j)$, then
$y^{n+1} = y^n + \mathrm{dt}\,\sum_i b_i k_i$. One step costs $s$ right-hand-side
evaluations ($s-1$ under first-stage reuse for the FSAL tableaus below). No
solve, no matrix — this is the family for non-stiff dynamics.

Tableaus are stored as `butcherTableau` NamedTuples (`.a`, `.b`, `.c`); embedded
pairs store `.b` as a tuple `(b_main, b_embedded)` and the step returns the pair
difference as `IntegrationResult.error` for step-size control (see
[solver](../solver)). Every tableau below is the one in `butcher.py`,
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

## Order 2

The three two-stage order-2 tableaus below share the stability function
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

### KYKSSPRK42 and KYK2014DGSSPRK_3S2

The two Kubatko–Yeager–Ketcheson DG-optimized SSP methods (Kubatko, Yeager &
Ketcheson, *J. Sci. Comput.* 60 (2014) 313-344; coefficients as stored by
OrdinaryDiffEq v7.8.1). 3–4 stages, so their stability functions are not the
shared order-2 one above; ODE's published SSP coefficients are the
DG-context values of the paper, and the library's upwind-model measurements
(which the tests pin) differ:

| Method | Stages | SSP $r$ (measured) | Real axis |
|---|---|---|---|
| KYKSSPRK42 | 4 | 2.2838 (ODE publishes 2.459 = the final-map-only CFL) | $[-4.954, 0]$ |
| KYK2014DGSSPRK_3S2 | 3 | 1.8939 (ODE publishes 0.8417, the paper's DG-context value) | $[-4.267, 0]$ |

Full tableaus in `butcher.py`; not FSAL.

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

### Ruuth's optimized SSP methods (SSPRK53, SSPRK63, SSPRK73, SSPRK83)

Ruuth's globally optimized order-3 SSP methods (Ruuth, *Mathematics of
Computation* 75.253 (2006) 183-207; coefficients as stored by OrdinaryDiffEq
v7.8.1). 5–8 stages — more than the three-evaluation tableaus above — with
the largest SSP coefficients in the registry short of SSPRK(10,4); ODE's
published `ssp_coefficient` agrees with the measured $r$ to within 6.3e-4:

| Method | Stages | SSP $r$ (measured) | Real axis |
|---|---|---|---|
| SSPRK53 | 5 | 2.6506 | $[-6.214, 0]$ |
| SSPRK63 | 6 | 3.5184 | $[-7.213, 0]$ |
| SSPRK73 | 7 | 4.2879 (beyond the landscape scan cap; the table below reports 4) | $[-8.888, 0]$ |
| SSPRK83 | 8 | 5.1071 (beyond the landscape scan cap; the table below reports 4) | $[-10.986, 0]$ |

Full tableaus in `butcher.py`; not FSAL. SSPRK73's coefficients inherit a
1e-6 slack: ODE stores alpha73/alpha76 at 6 digits.

### SSPRK43 and SSPRK432 (the 3(2) pairs)

The 4-stage order-3 SSP method of Kraaijevanger (Kraaijevanger, *BIT* 31.3
(1991) 482-528) with an embedded order-2 estimate (Conde, Fekete & Shadid,
arXiv:1806.08693; coefficients as stored by OrdinaryDiffEq v7.8.1). One
$(a, c, b)$, two choices of embedded weight:

$$
\begin{array}{c|cccc}
0 & 0 & 0 & 0 & 0 \\
1/2 & 1/2 & 0 & 0 & 0 \\
1 & 1/2 & 1/2 & 0 & 0 \\
1/2 & 1/6 & 1/6 & 1/6 & 0 \\
\hline
& 1/6 & 1/6 & 1/6 & 1/2
\end{array}
$$

SSPRK43's embedded weight is $\hat b = (1/4, 1/4, 1/4, 1/4)$ and
SSPRK432's $\hat b = (1/3, 1/3, 1/3, 0)$ — both order 2
($\sum \hat b_i = 1$, $\sum_i \hat b_i c_i = 1/2$), so in both the
propagated-minus-embedded difference is the embedded branch's
$O(\mathrm{dt}^3)$ local error and both drive the adaptive controller
(`tests/test_adaptive.py::test_adaptive_loop_smoke_on_new_ssprk_pairs`).
Stage 3 is evaluated at $t + \mathrm{dt}$ — the node ordering is
$c = (0, 1/2, 1, 1/2)$, not monotone. Measured: exact SSP coefficient
$r = 2.0$ (matching ODE's published value) and real-axis interval
$[-5.149, 0]$ for both; not FSAL. These are the registry's SSP 3(2) pairs —
adaptive step-size control on a strongly-stability-preserving step, for
problems where Dormand-Prince's $r = 0$ is unacceptable.

### SSPRK932 (9-stage order 3, fixed step)

The 9-stage order-3 SSP method of Gottlieb–Ketcheson–Shu (Gottlieb,
Ketcheson & Shu, *Strong Stability Preserving Runge–Kutta and Multistep
Time Discretizations*, World Scientific, 2011; coefficients as stored by
OrdinaryDiffEq v7.8.1): a six-stage $dt/6$ forward-Euler cascade
($c = 0, 1/6, 1/3, 1/2, 2/3, 5/6$), a blend
$u_6^* = (3u_0 + (dt/2)k_1 + 2u_6)/5$, then three more $dt/6$ steps
(stages 7-9 at $c = 1/2, 2/3, 5/6$). Measured: exact SSP coefficient
$r = 6.0$ (matching ODE's published value; it is why the boundary scan in
`tests/test_tvd.py` goes to CFL 8 — the default cap 4.0 would clip it) and
real-axis interval $[-13.290, 0]$; not FSAL ($c_9 = 5/6$). Registered
**fixed-step only**: ODE's adaptive estimate computes an extra evaluation
$k = f(u_6, t + dt)$, sets $\tilde u = (u_0 + 6u_6 + 6\,\mathrm{dt}\, k)/7$, and
uses $\tilde u - u^{n+1}$ as the error — but that estimate is
$(5/7)\,\mathrm{dt}\,u'(t) + O(\mathrm{dt}^2)$ (the coefficients $1/7, 6/7, 6/7$
do not sum to 1; measured scaling 1.058, NOTES.md §3.22), $O(\mathrm{dt})$
instead of the $O(\mathrm{dt}^3)$ a 3(2) pair must have. ODE's own docstring
says "Consider using `SSPRK43` instead". Full tableau in `butcher.py`.

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

### SSPRK54

Ruuth's 5-stage order-4 SSP method (Ruuth, *Mathematics of Computation*
75.253 (2006) 183-207; coefficients as stored by OrdinaryDiffEq v7.8.1).
Measured: SSP coefficient $r = 1.7693$ (ODE publishes the paper's 1.508,
which the library's all-stage upwind-model measurement does not reproduce;
the final map alone is convex to 1.8611) and real-axis interval
$[-5.331, 0]$; not FSAL. The full tableau is in `butcher.py`.

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
| SSPRK(10,4) | 4 (scan cap; exact 6) | 5 (tested limit) |
| SSPRK53 / SSPRK63 / SSPRK73 / SSPRK83 / SSPRK54 / KYKSSPRK42 / KYK2014DGSSPRK_3S2 | 2.651 / 3.518 / 4 / 4 / 1.769 / 2.284 / 1.894 (SSPRK73/83 at the scan cap; exact 4.2879 / 5.1071) | 2 / 3 / 3 / 5 / 2 / 2 / 1.5 |
| SSPRK43 / SSPRK432 / SSPRK932 | 2.0 / 2.0 / 4 (scan cap; exact 6.0) | 2 / 2 / 5 (SSPRK932 unconditional) |

TVD is strictly broader than SSP: RK4 is per-step TVD past where its internal
stages leave the convex hull, and Nystrom/Dormand-Prince are TVD to CFL 1.5
despite $r = 0$. The stage-level separation is what distinguishes RK3 from TVD
RK3 (RK3's stage-3 map $1 + \zeta + \zeta^2$ has a negative circulant entry at
CFL 1; TVD RK3's stage rows stay non-negative).

## Parametric-relaxation pRRK (`prrk.py`) — not registered

pRRK22 / pRRK33 / pRRK54 (Liu, Burchard, Kuzmin & Shang 2023) are
*parametric-relaxation* variants of Heun / SSP RK3 / SSPRK54. A free
relaxation parameter $\kappa$ trades the SSP (TVD) constant for a much larger
negative-real-axis stability interval. They are **not registered** in
`IntegrationSchemes` (see below) and are exposed as standalone drivers:

```python
from warpSPHIntegrators import pRRK33
result = pRRK33(system, dt, rhs, kappa=1.0)
```

**Why not registered:** the relaxation parameter is $z = \kappa \Delta t$, so
the Butcher tableau $(a, b, c)$ depends on *both* $\kappa$ and $\Delta t$ —
it is not a fixed tableau, and at $\kappa = 0$ the methods duplicate
already-registered schemes. There is no single fixed order or identifier to
register, so they live alongside the registry (NOTES.md §3.24).

**Behaviour (measured, `tests/test_prrk.py`):**
- $\kappa = 0$ reduces *exactly* to the base method (Heun / SSP RK3 /
  SSPRK54), to machine precision.
- The formal order (2 / 3 / 4) is preserved for every $\kappa$.
- Negative-real-axis stability interval: bounded at $\kappa = 0$
  ($[-2, 0]$ / $[-2.51, 0]$ / $[-5.33, 0]$), **unbounded (A-stable)** for
  $\kappa \ge 0.5$ — $R \to 1$ from below as $\Delta t \to \infty$ (A-stable,
  not L-stable). There is no imaginary-axis stability (a general SSPRK
  limitation), so the enlargement is a stiff/parabolic property.

## References

- Hairer, Nørsett, Wanner, *Solving Ordinary Differential Equations I* (1993) — RK theory, stability, embedded pairs.
- Shu, *Total variation diminishing Runge-Kutta time discretizations*, J. Comput. Phys. 169 (2001) 208-228 — SSPRK(10,4).
- Ruuth, *Global optimization of explicit strong-stability-preserving Runge-Kutta methods*, Mathematics of Computation 75.253 (2006) 183-207 — SSPRK53/63/73/83, SSPRK54.
- Kubatko, Yeager & Ketcheson, *Optimal strong-stability-preserving Runge-Kutta time discretizations for discontinuous Galerkin methods*, J. Sci. Comput. 60 (2014) 313-344 — KYKSSPRK42, KYK2014DGSSPRK_3S2.
- Kraaijevanger, *Contractivity of Runge-Kutta methods*, BIT 31.3 (1991) 482-528 — SSPRK43 main method; Conde, Fekete & Shadid, arXiv:1806.08693 (2018) — the SSPRK43 embedded pair.
- Gottlieb, Ketcheson & Shu, *Strong Stability Preserving Runge–Kutta and Multistep Time Discretizations*, World Scientific (2011) — SSPRK432 (Example 6.1), SSPRK932.
- Shu & Osher, *Efficient implementation of essentially non-oscillatory shock capturing schemes*, J. Comput. Phys. 79 (1989) 500-512 — SSP/TVD RK2, RK3.
- Bogacki & Shampine, *A 3(2) pair of Runge-Kutta formulas*, BIT 35 (1995) 232-248.
- Dormand & Prince, J. Comput. Appl. Math. 8 (1981) 15-28; Cash & Karp, NASA TM-100035 (1990).
- Liu, Burchard, Kuzmin & Shang, *Parametric relaxation Runge-Kutta methods* (2023) — pRRK22/33/54 (not registered).

**Tests:** `tests/test_convergence.py` (orders), `tests/test_step_reuse.py`
(reuse orders), `tests/test_tvd.py` (the table above),
`tests/test_embedded.py` (embedded-pair rates), `tests/test_hamiltonian.py`
(energy drift on the oscillator).
