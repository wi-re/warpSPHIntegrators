# Relaxed Chebyshev / Lobatto super-timestepping (`RelaxedChebyshev`) — 3 schemes

Stabilised **explicit** methods for stiff *parabolic* (diffusive) right-hand
sides, driven by `rkc.py`. A step takes $s$ cheap, matrix-free right-hand-side
evaluations and buys a real-axis stability interval that grows like $O(s^2)$ —
so the step size can be set by the physics (the nonlinear convective scale),
not by the stiffest diffusive eigenvalue. That is what makes them attractive
for the viscous term in SPH. No solve, no matrix.

## Construction

Each method's one-step stability polynomial is a scaled, shifted orthogonal
polynomial in $z = \mathrm{dt}\,\lambda$:

$$
\text{RKC1: } R_s(z) = T_s(1 + z/s^2) \qquad
\text{(Chebyshev, order 1)}
$$
$$
\text{RKC2: } R_s(z) = a_s + b_s\, T_s(1 + w_1 z) \qquad
\text{RKL2: } R_s(z) = a_s + b_s\, P_s(1 + w_1 z)
$$

with $T_s$ the Chebyshev and $P_s$ the Legendre polynomial. RKC2 and RKL2 add
the constant offset $a_s$ (and scaling $b_s$) that makes the polynomial match
$e^z$ through $z^2$, lifting it to order 2. For RKC2:
$w_1 = 3/(s^2-1)$, $b_s = (s^2-1)/(3s^2)$, $a_s = 1 - b_s$.

The real-axis stability intervals ($|R_s| \le 1$ on $[-K, 0]$):

| Scheme | Order | $K(s)$ | Min stages |
|---|---|---|---|
| RKC1 | 1 | $2s^2$ | 1 |
| RKC2 | 2 | $2(s^2-1)/3$ | 3 |
| RKL2 | 2 | $(s^2+s-2)/2$ | 3 |

All three are realised by a three-term (plus a cached $Y_0$) recurrence, so a
step costs $s$ RHS evaluations and a few state buffers:

$$
Y_0 = u,\ f_0 = f(Y_0), \qquad
Y_1 = Y_0 + \tilde m_1\, \mathrm{dt}\, f_0,
$$
$$
Y_j = (1-\mu_j-\nu_j) Y_0 + \mu_j Y_{j-1} + \nu_j Y_{j-2}
      + \mathrm{dt}\,(\tilde m_j f_{j-1} + \tilde g_j f_0), \quad 2 \le j \le s,
\qquad u^{n+1} = Y_s.
$$

The stage weights come from the Chebyshev/Legendre three-term recurrences,
chosen so that **every intermediate stage polynomial** is itself bounded by 1
on $[-K, 0]$ (no stage overshoots).

## Stage count: per-step, physics-driven

$s$ is a per-step choice: pass `s=` directly, or pass `lambda_max=` (the
magnitude of the stiffest RHS eigenvalue) and the smallest admissible $s$ with
$K(s) \ge \mathrm{dt}\,\lambda_{\max}$ is chosen. RKL2 and RKC2 prefer an
**odd** stage count: for even $s$ the smallest-wavelength mode is only weakly
damped (Meyer et al., 2014). The driver clamps $s$ to the method's minimum
(1 for RKC1, 3 for RKC2/RKL2) but does not force parity.

These schemes are explicit one-step methods: `priorStep` is rejected; stage
times are set to $t + \mathrm{dt}\, j/s$ as a reasonable intermediate time (an
approximation for non-autonomous RHS).

## Why not an implicit method? (measured)

On the diffusion benchmark, RKL2 holds the error flat across the stiff range
with *fewer* total RHS evaluations than the implicit baselines at the default
JFNK settings — the implicit methods pay for a Newton+GMRES solve per step,
while RKL2 pays $s$ cheap evaluations (NOTES.md §3.13,
`scripts/rkl2_benchmark.py`, `images/rkl2_benchmark.png`;
`scripts/rkc_benchmark.py`, `images/rkc_benchmark.png`). The crossover: once
the problem is stiff enough that the implicit solve converges in a small number
of Krylov iterations, the implicit methods win; the benchmark plots both sides.

TVD classification (NOTES.md §3.11): RKC2/RKL2 expose no tableau and are
classified by measurement — per-step TVD to CFL 1.5 (tested to CFL 5).

## References

- Ruuth, *An analysis of a relaxed Chebyshev method*, J. Comput. Phys. 169 (2001) 162-175 — the RKC construction.
- Meyer, Balsara, Aslam, *Relaxed Lobatto schemes for advection-dominated PDEs*, J. Comput. Phys. 257 (2014) 594-626 (Eqs. 15-19) — RKL2.

**Tests:** `tests/test_rkc.py` (order, stability-interval endpoints, odd-stage
damping, the $s$-clamping and `lambda_max` selection).
