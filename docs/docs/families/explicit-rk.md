---
sidebar_label: Explicit RK
---

# Explicit Runge-Kutta (`ExplicitRK`) — 63 schemes

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

### Alshina 2(1) (`Alshina2`)

Order 2 / embedded 1, 2 stages (Alshina, Zaks & Kalitkin, *Computational
Mathematics and Mathematical Physics* 48.3 (2008) 395-405; coefficients as
stored by OrdinaryDiffEq v7.8.1): $a$-rows $(0), (2/3, 0)$,
$b = (1/4, 3/4)$, $\hat b = (1, 0)$, $c = (0, 2/3)$. The embedded branch is
exactly Euler, so the estimate is the embedded branch's $O(\mathrm{dt}^2)$
local error. Measured: SSP coefficient $r = 1$ and real-axis interval
$[-2, 0]$; not FSAL ($c_2 = 2/3$).

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

### SSPRK53_2N1, SSPRK53_2N2 and SSPRK53_H (Higueras–Roldán low-storage variants)

Higueras & Roldán's new third-order low-storage SSP methods (Higueras &
Roldán, *New third order low-storage SSP explicit Runge–Kutta methods*,
arXiv:1809.04807; coefficients as stored by OrdinaryDiffEq v7.8.1). All three
are 5-stage order 3, written in ODE's Shu–Osher α/β form with 1–2 working
temporaries (vs 2 for SSPRK53). **Screened before registration** (NOTES.md
§3.25): all three are algebraically *distinct* one-step maps from SSPRK53 —
different stages and a different final b-row, with the one-step map differing
by $O(\mathrm{dt}^4) \approx 1.5\times 10^{-4}$–$4.6\times 10^{-3}$ per unit
dt⁴, not $\sim 10^{-16}$ — so they are their own methods, not storage
rearrangements of it. ODE's storage saving does not apply to this driver,
which builds each stage state fresh from the a-row. Full tableaus in
`butcher.py`; not FSAL.

| Method | Stages | SSP $r$ (measured) | Real axis |
|---|---|---|---|
| SSPRK53_2N1 | 5 | 2.1808 (ODE publishes 2.18) | $[-5.528, 0]$ |
| SSPRK53_2N2 | 5 | 2.1487 (ODE publishes 2.148) | $[-7.263, 0]$ |
| SSPRK53_H | 5 | 2.6506 (ODE publishes 2.65) | $[-6.213, 0]$ |

SSPRK53_2N2 has the largest real-axis interval of the four despite the
smallest SSP coefficient; SSPRK53_H matches SSPRK53's interval to 1e-4 (it
shares SSPRK53's first two nodes $c_1, c_2$ but has distinct later stages).
SSPRK53_2N1's stage 3 sits at $c_3 = 1.0053 > 1$ (evaluated just past the
step end, as ODE does).

### OwrenZen 3(2) and the wave-6 order-3 workhorses

| Method | Stages | Embedded | SSP $r$ (measured) | Real axis | FSAL |
|---|---|---|---|---|---|
| OwrenZen3 (`OwrenZen3`) | 4 | 2nd (free interpolant) | 0.7813 | $[-2.512, 0]$ | yes |
| Alshina3 (`Alshina3`) | 3 | — (fixed step) | 1.0 | $[-2.512, 0]$ | no |
| RKM (`RKM`) | 6 | — (fixed step) | 0.7531 | $[-3.224, 0]$ | no |
| PSRK3p6q5 (`PSRK3p6q5`) | 5 | — (fixed step) | 0.0 | $[-3.548, 0]$ | no |
| PSRK3p5q4 (`PSRK3p5q4`) | 4 | — (fixed step) | 0.0 | $[-2.784, 0]$ | no |

**OwrenZen3** is the Owren–Zennaro optimized interpolation 3/2 pair (Owren &
Zennaro, *SIAM J. Sci. Stat. Comput.* 13.6 (1992) 1488-1501) with a free
order-3 interpolant. **Alshina3** is the 3rd-order member of Alshina, Zaks &
Kalitkin's optimal 1st–6th order family; registered **fixed-step only**:
ODE's stored $\tilde b$ sums to $5/9 \ne 1$, so ODE's own estimate is not a
valid companion (NOTES.md §3.26). **RKM** is the Mead–Renaut method designed
for good stability on pseudospectral discretizations of hyperbolic PDEs
(Mead & Renaut, *J. Comput. Phys.* 152.1 (1999) 404-419); the registered
order is the measured 3. **PSRK3p6q5 / PSRK3p5q4** are Aubry–Chartier
pseudo-symplectic order-3 methods (Aubry & Chartier, *BIT* 38 (1998) 439-461;
Capuano et al., *J. Comput. Phys.* 328 (2017) 86-94) with improved
energy-conservation properties; $r = 0$ (not convex-combination steps).
Full tableaus in `butcher.py`.

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

### OwrenZen 4(3), Ralston 4 and PSRK4p7q6

| Method | Stages | Embedded | SSP $r$ (measured) | Real axis | FSAL |
|---|---|---|---|---|---|
| OwrenZen4 (`OwrenZen4`) | 6 | 3rd (free interpolant) | 0.6444 | $[-2.872, 0]$ | yes |
| Ralston4 (`Ralston4`) | 4 | — | 0.7209 | $[-2.784, 0]$ | no |
| PSRK4p7q6 (`PSRK4p7q6`) | 6 | — | 0.0 | $[-11.41, 0]$ | no |

**OwrenZen4** is the Owren–Zennaro optimized interpolation 4/3 pair (Owren &
Zennaro 1992, the same paper as OwrenZen3). **Ralston4** is Ralston's
4-stage minimum-truncation-error order-4 method (Ralston, *Mathematics of
Computation* 16.80 (1962) 431-437). **PSRK4p7q6** is the 6-stage
pseudo-symplectic order-4 method (Aubry & Chartier 1998; Capuano et al.
2017) — the largest real-axis interval of the whole wave 6, $[-11.41, 0]$,
at the cost of $r = 0$. Full tableaus in `butcher.py`.

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

### Tsit5 5(4) and the wave-6 order-5 methods

| Method | Stages | Embedded | SSP $r$ (measured) | Real axis | FSAL |
|---|---|---|---|---|---|
| Tsit5 (`Tsit5`) | 7 | 4th (free interpolant) | 0.0 | $[-3.506, 0]$ | yes |
| OwrenZen5 (`OwrenZen5`) | 8 | 4th (free interpolant) | 0.0 | $[-3.192, 0]$ | yes |
| BS5 (`BS5`) | 8 | 4th (ODE's $\hat b$, not $\tilde b$) | 0.5237 | $[-3.986, 0]$ | yes |
| Stepanov5 (`Stepanov5`) | 7 | 4th | 0.0 | $[-3.496, 0]$ | yes |
| MSRK5 (`MSRK5`) | 9 | — (8 + FSAL) | 0.3226 | $[-4.366, 0]$ | yes |
| SIR54 (`SIR54`) | 8 | — (fixed step) | 0.0 | $[-3.512, 0]$ | yes |
| RKO65 (`RKO65`) | 6 | — | 0.0 | $[-4.164, 0]$ | no (free first stage) |

- **Tsit5** — Tsitouras 5(4) (Tsitouras, *Computers & Mathematics with
  Applications* 62.2 (2011) 770-775): a 5(4) pair satisfying only the first
  column simplifying assumption, with a free order-4 interpolant — ODE's
  recommended default for non-stiff problems of unknown stiffness.
- **OwrenZen5** — the Owren–Zennaro optimized interpolation 5/4 pair (Owren
  & Zennaro 1992); ODE stores $a_{43} = 1$ in-code.
- **BS5** — Bogacki–Shampine's efficient (4, 5) pair (Bogacki & Shampine,
  *Computers & Mathematics with Applications* 32.6 (1996) 15-28) — a
  different algorithm from Dormand-Prince above. Registered with ODE's
  $\hat b$ as the embedded branch: $\tilde b$ is invalid (it does not sum
  to 1), so $\hat b$ is the order-4 companion.
- **Stepanov5** — Stepanov's embedded (4, 5) pair of 7-stage FSAL methods
  (Stepanov, *Calcolo* 59 (2021)).
- **MSRK5** — Stepanov's 9-stage order-5 method (Stepanov,
  arXiv:2202.08443, Figure 3): 8 working stages + the FSAL stage.
- **SIR54** — the Kovalnogov–Simos–Tsitouras 5(4) pair for SIR-type
  epidemic models (Math. Methods Appl. Sci. 44 (2020) 5210-5216); FSAL
  (last row $= b$). Registered **fixed-step only**: ODE's stored
  $\tilde b$ sums to 7.854, not 1, so ODE's own estimate is not a valid
  companion (NOTES.md §3.26).
- **RKO65** — Tsitouras' Runge–Kutta–Oliver 6-stage 5th-order "starting
  integration" method (Tsitouras, *Applied Mathematics and Computation*
  354 (2019) 353-364). Its first stage is **free**: $c_1 = 2/3$ with a zero
  $a$-row and $b_1 = 0$, so no order condition involves $c_1$ — the stage
  may be evaluated at any time (the driver's $t^n$ convention, shared with
  every other registered stage 0, is one valid choice; ODE's stored
  $c_1 = 2/3$ is another). Row-sum consistency therefore holds for rows
  $2..s$ as usual but not for row 1 (structural, pinned by the residual),
  and reuse is **lossless** — see the First-stage reuse section.

Full tableaus in `butcher.py`.

## Order 6

### Alshina 6 and MSRK6

| Method | Stages | Embedded | SSP $r$ (measured) | Real axis | FSAL |
|---|---|---|---|---|---|
| Alshina6 (`Alshina6`) | 7 | — (fixed step) | 0.0 | $[-2.904, 0]$ | no |
| MSRK6 (`MSRK6`) | 9 | — (8 + FSAL) | 0.1009 | $[-4.164, 0]$ | yes |

**Alshina6** is the 6th-order member of Alshina, Zaks & Kalitkin's optimal
family (2008). **MSRK6** is Stepanov's 9-stage order-6 method
(arXiv:2202.08443, Table 4): 8 working stages + the FSAL stage. Full
tableaus in `butcher.py`.

## Order 7 and up (wave 7)

The eleven high-order tableaus of wave 7 (SCHEME_EXPANSION_PLAN, 2026-10-13):
Verner's optimal pairs, the two 10-stage order-7 methods, the Tsitouras–
Papakostas 8(7) pair, Hairer's DP8, and Feagin's symmetric methods up to
order 14. The registry's highest-order one-step explicit methods are now
**Feagin14** (order 14) and **Vern9** (order 9 with a free embedded branch).

| Method | Stages | Embedded | SSP $r$ (measured) | Real axis | FSAL |
|---|---|---|---|---|---|
| Vern6 (`Vern6`) | 9 | 5th | 0.0877 | $[-4.855, 0]$ | yes |
| Vern7 (`Vern7`) | 10 | 6th | 0.0514 | $[-4.640, 0]$ | no |
| TanYam7 (`TanYam7`) | 10 | 6th | 0.0 | $[-4.913, 0]$ | no |
| RKV76IIa (`RKV76IIa`) | 10 | 6th | 0.0 | $[-4.911, 0]$ | no |
| Vern8 (`Vern8`) | 13 | 7th | 0.1615 | $[-13.425, 0]$ | no |
| TsitPap8 (`TsitPap8`) | 13 | 7th | 0.1804 | $[-10.388, 0]$ | no |
| DP8 (`DP8`) | 12 | — (dual 5/3 companion, registered plain) | 0.2371 | $[-6.394, 0]$ | no |
| Vern9 (`Vern9`) | 16 | 8th | 0.1272 | $[-4.476, 0]$ | no |
| Feagin10 (`Feagin10`) | 17 | — (symmetric, fixed step) | 0.0 | $[-2.528, 0]$ | no |
| Feagin12 (`Feagin12`) | 25 | — (symmetric, fixed step) | 0.0 | $[-4.070, 0]$ | no |
| Feagin14 (`Feagin14`) | 35 | — (symmetric, fixed step) | 0.0 | $[-1.874, 0]$ | no |

- **Vern6–Vern9** — Verner's numerically optimal pairs with interpolants
  (Verner, *Numerical Algorithms* 53.2-3 (2010) 383-396; coefficients as
  stored by OrdinaryDiffEq v7.8.1). Each is stored with a tail stage at
  $c = 1$; **Vern6 is the wave's only FSAL tableau** (its last row is the
  main weight vector), so its reuse is lossless. Vern8 carries the
  wave's widest real-axis interval, $[-13.425, 0]$.
- **TanYam7** — the 9-stage order-7 method of Tanaka, Muramatsu & Yamashita
  (*IPSJ* 33.12 (1992) 1512-1526), stored by ODE with a 10th stage at
  $c = 1$ for the order-6 companion. ODE's published real stability interval
  $[-4.9125, 0]$ is reproduced by the measured scan.
- **RKV76IIa** — Verner's "most efficient" 10-stage (10:7(6)) pair (Verner,
  *RKV76.IIa — a 'most efficient' Runge–Kutta pair*, 2024); same 9-work-
  stage + $c = 1$ companion structure as TanYam7.
- **TsitPap8** — Tsitouras–Papakostas' 8(7) pair designed for cheap error
  estimation (Tsitouras & Papakostas, *SIAM J. Sci. Comput.* 20.6 (1999));
  its stage 8 sits at $c_8 = 1.0070922$, just past the step end (like
  SSPRK53_2N1's $c_3$ and ESDIRK's $c_4$ — the stage-times test carries a
  per-scheme slack for it).
- **DP8** — Hairer's 8/5/3 adaptation of the Dormand–Prince method (Hairer,
  Nørsett & Wanner, *Solving Ordinary Differential Equations I*, 2nd ed.,
  1993). Registered **plain**: the companion is a dual 5th/3rd-order
  combination, not a valid (8,7) pair, so no `IntegrationResult.error` is
  exposed. The wave's largest SSP coefficient ($r = 0.2371$).
- **Feagin10 / Feagin12 / Feagin14** — Feagin's symmetric (time-reversible)
  10th/12th/14th-order methods for molecular dynamics and celestial mechanics
  (Feagin, *The Computer Journal* 26.5 (1983) 351-357; coefficients as
  stored by ODE v7.8.1). Plain and fixed-step in ODE; the c-node structure
  is the visible signature of the symmetry (mirror-symmetric about
  $c = 1/2$ up to the endpoint stages). Feagin14's real-axis interval
  $[-1.874, 0]$ is the narrowest of the wave despite its order — verified
  genuine ($|R(-2.0)| = 3.51$), its 35-stage stability polynomial
  oscillates for $|z| \gtrsim 2$.

Cross-checked against the actual ODE v7.8.1 solver on a real (nonlinear,
non-autonomous) problem: $\max|\Delta y|$ at $T = 2$ is
$\le 8.9\times 10^{-16}$ for every member of the wave (NOTES.md §3.27).

## First-stage reuse

`priorStep=result.stages[-1]` feeds the previous step's last stage as this
step's first stage, saving one evaluation. It is lossless exactly for the FSAL
tableaus — Bogacki-Shampine, Dormand-Prince, the wave-6
Tsit5 / OwrenZen3 / OwrenZen4 / OwrenZen5 / BS5 / Stepanov5 pairs plus
MSRK5 / MSRK6 / SIR54, and the wave-7 **Vern6** — and, one case beyond FSAL,
for RKO65: with $b_1 = 0$
its stale k0 reaches the update only through the stage equations, and the
one-stage paths carry the common coefficient $(b^\top a)_1 = 0$ as well, so
`reuse.step_reuse_analysis` reports a lossless order 5 for it (NOTES.md
§3.26). For every other tableau reuse costs order (the analysis reports the
exact reuse order per scheme; pinned in `tests/test_step_reuse.py`);
single-stage tableaus (Forward Euler) refuse it outright, since the reuse
would be the lagged two-step map with no imaginary-axis stability region.
Wave 7 adds a second refusal class (NOTES.md §3.27): the multi-stage
**Vern7**, **RKV76IIa**, **TanYam7** and **Feagin14** refuse reuse outright,
because their reuse-modified recurrence — a three-dimensional two-step map in
the state $(y^{n-1}, y^n, k_s^{n-1})$ — has no stability interval on the
imaginary axis that covers the standard test step ($|\mathrm{Im}(z)| \lesssim$
0.02 / 0.008 / 0.008 / 0.108 respectively, against one-step intervals to
0.35 / 4.44 / 4.44 / 1.16). The remaining wave-7 schemes keep reuse at the
reported (lower) order. Multistep and implicit families use different
mechanisms — see their pages.

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
| SSPRK53_2N1 / SSPRK53_2N2 / SSPRK53_H | 2.181 / 2.149 / 2.651 | 2 / 2 / 2 |
| Tsit5 5(4) / OwrenZen3 3(2) / OwrenZen4 4(3) / OwrenZen5 5(4) / BS5 5(4) / Stepanov5 5(4) / Alshina2 2(1) / Alshina3 / Alshina6 / MSRK5 / MSRK6 / SIR54 / Ralston4 / RKO65 / RKM / PSRK4p7q6 / PSRK3p6q5 / PSRK3p5q4 (wave 6) | 0 / 0.7813 / 0.6444 / 0 / 0.5237 / 0 / 1 / 1 / 0 / 0.3226 / 0.1009 / 0 / 0.7209 / 0 / 0.7531 / 0 / 0 / 0 | 1.5 / 1 / 1.25 / 1.5 / 1.5 / 1.5 / 1 / 1 / 1.25 / 2 / 2 / 1.5 / 1.25 / 2 / 1.5 / 2 / 1.5 / 1.25 |
| Vern6 6(5) / Vern7 7(6) / TanYam7 7(6) / RKV76IIa 7(6) / Vern8 8(7) / TsitPap8 8(7) / DP8 / Vern9 9(8) / Feagin10 / Feagin12 / Feagin14 (wave 7) | 0.0877 / 0.0514 / 0 / 0 / 0.1615 / 0.1804 / 0.2371 / 0.1272 / 0 / 0 / 0 | 2 / 2 / 2 / 2 / 2 / 2 / 3 / 2 / 1 / 1.25 / 0.75 |

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
- Higueras & Roldán, *New third order low-storage SSP explicit Runge–Kutta methods*, arXiv:1809.04807 (2018) — SSPRK53_2N1 / SSPRK53_2N2 / SSPRK53_H.
- Tsitouras, *Runge–Kutta pairs of order 5 (4) satisfying only the first column simplifying assumption*, Computers & Mathematics with Applications 62.2 (2011) 770-775 — Tsit5.
- Owren & Zennaro, *Derivation of efficient, continuous, explicit Runge–Kutta methods*, SIAM Journal on Scientific Computing 13.6 (1992) 1488-1501 — OwrenZen3 / OwrenZen4 / OwrenZen5.
- Bogacki & Shampine, *An efficient runge-kutta (4, 5) pair*, Computers & Mathematics with Applications 32.6 (1996) 15-28 — BS5.
- Stepanov, *Embedded (4, 5) pairs of explicit 7-stage Runge–Kutta methods with FSAL property*, Calcolo 59 (2021) — Stepanov5; Stepanov, arXiv:2202.08443 (2022) — MSRK5 (Figure 3) / MSRK6 (Table 4).
- Kovalnogov, Simos & Tsitouras, *Runge–Kutta pairs suited for SIR‐type epidemic models*, Mathematical Methods in the Applied Sciences 44 (2020) 5210-5216 — SIR54.
- Alshina, Zaks & Kalitkin, *Optimal first- to sixth-order accurate Runge–Kutta schemes*, Computational Mathematics and Mathematical Physics 48.3 (2008) 395-405 — Alshina2 / Alshina3 / Alshina6.
- Ralston, *Runge-Kutta methods with minimum error bounds*, Mathematics of Computation 16.80 (1962) 431-437 — Ralston4.
- Tsitouras, *Explicit Runge–Kutta methods for starting integration of Lane–Emden problem*, Applied Mathematics and Computation 354 (2019) 353-364 — RKO65.
- Mead & Renaut, *Optimal Runge–Kutta methods for first order pseudospectral operators*, Journal of Computational Physics 152.1 (1999) 404-419 — RKM.
- Aubry & Chartier, *Pseudo-symplectic Runge–Kutta methods*, BIT 38 (1998) 439-461; Capuano et al., *Pseudo-symplectic Runge–Kutta methods: revisiting the optimal coefficients*, Journal of Computational Physics 328 (2017) 86-94 — PSRK4p7q6 / PSRK3p6q5 / PSRK3p5q4.
- Verner, *Numerically optimal Runge–Kutta pairs with interpolants*, Numerical Algorithms 53.2-3 (2010) 383-396 — Vern6 / Vern7 / Vern8 / Vern9.
- Verner, *RKV76.IIa — a 'most efficient' Runge–Kutta (10:7(6)) pair* (2024, sfu.ca) — RKV76IIa.
- Tanaka, Muramatsu & Yamashita, *On the Optimization of Some Nine-Stage Seventh-order Runge-Kutta Method*, Information Processing Society of Japan 33.12 (1992) 1512-1526 — TanYam7.
- Tsitouras & Papakostas, *Cheap error estimation for Runge–Kutta methods*, SIAM Journal on Scientific Computing 20.6 (1999) — TsitPap8.
- Feagin, *A family of symmetric integration algorithms for molecular dynamics, celestial mechanics, and cosmology*, The Computer Journal 26.5 (1983) 351-357 — Feagin10 / Feagin12 / Feagin14 (DP8: Hairer, Nørsett & Wanner 1993, the first entry above).
- Shu & Osher, *Efficient implementation of essentially non-oscillatory shock capturing schemes*, J. Comput. Phys. 79 (1989) 500-512 — SSP/TVD RK2, RK3.
- Bogacki & Shampine, *A 3(2) pair of Runge-Kutta formulas*, BIT 35 (1995) 232-248.
- Dormand & Prince, J. Comput. Appl. Math. 8 (1981) 15-28; Cash & Karp, NASA TM-100035 (1990).
- Liu, Burchard, Kuzmin & Shang, *Parametric relaxation Runge-Kutta methods* (2023) — pRRK22/33/54 (not registered).

**Tests:** `tests/test_convergence.py` (orders), `tests/test_step_reuse.py`
(reuse orders), `tests/test_tvd.py` (the table above),
`tests/test_embedded.py` (embedded-pair rates), `tests/test_hamiltonian.py`
(energy drift on the oscillator).
