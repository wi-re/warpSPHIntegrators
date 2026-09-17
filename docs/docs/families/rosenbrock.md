---
sidebar_label: Rosenbrock-W
---

# Linearly implicit — Rosenbrock-W (`LinearlyImplicit`) — ROS3P

The single linearly-implicit scheme: **ROS3P**, a 3-stage, order-3,
**A-stable** (not L-stable) Rosenbrock-W method with a built-in order-2
embedded estimate, from the 3-stage Rosenbrock-W construction in **Lang &
Verwer, BIT 41(4) (2001) 731-738** (the citable coefficient source). Driven by
`rosenbrock.py`.

## The one-step form

Rosenbrock-W methods freeze the Jacobian (or the `linear` part) once per step,
$W = F_y(t^n, y^n)$, and close each stage with a linear solve in the common
operator $M = I - \tau\gamma W$ — **identical for all three stages**, so the
three stage solves share one Krylov operator:

$$
M\, K_i = F(t^n + a_i\tau, \; y^n + \tau \textstyle\sum_{j<i} a_{ij} K_j)
+ \tau W \sum_{j<i} g_{ij} K_j
+ \tau\, \gamma_i\, \tfrac{dF}{dt}(t^n, y^n),
\qquad
y^{n+1} = y^n + \tau \sum_i b_i K_i.
$$

## Coefficients (exact in $\sqrt3$)

$$
\gamma = \tfrac12 + \tfrac{\sqrt3}{6} \approx 0.7886751345948129
$$

| quantity | value |
|---|---|
| stage nodes $a$ | $(0, 1, 1)$ |
| $a_{21}, a_{31}, a_{32}$ | $1,\ 1,\ 0$ |
| $g_{21},\ g_{31},\ g_{32}$ | $-1,\ -\gamma,\ \tfrac12 - 2\gamma$ |
| $\gamma$-row | $(\gamma,\ \gamma - 1,\ \tfrac12 - 2\gamma)$ |
| $b$ (propagated) | $(2/3,\ 0,\ 1/3)$ |
| $\hat b$ (embedded order 2) | $(1/3,\ 1/3,\ 1/3)$ |

The coefficients satisfy the Rosenbrock-W order conditions exactly (the
order-3 condition $\gamma^2 - \gamma + 1/6 = 0$ holds in $Q(\sqrt3)$). The
stability function satisfies $R(\infty) = 1 - \sqrt{3} \approx -0.732$: stiff
modes are damped to $|\!1-\sqrt3\!| \approx 0.732$ **per step**, not killed —
A-stable, not L-stable.

**Cost.** The three stage *states* collapse to **two** distinct RHS points —
$(t^n, y^n)$ and $(t^n + \tau, y^n + \tau K_1)$ (stage 3 reuses stage 2's
state and time) — so a step costs **two** $f$ evaluations (plus one more when
$f_t\text{='fd'}$) and **three** GMRES solves in the shared $M$. The embedded
error estimate is $y^{n+1} - \hat y = \tau (K_1 - K_2)/3$ — free, no extra
stage.

## Options

**`w=`** — the frozen operator:

- `'jvp'` (default): the exact Jacobian action $Wv = J_f(t^n, y^n)v$ by
  forward-mode AD. Exact for RHS built from warpSPHCore's wrapped operators;
  fails loudly otherwise.
- `'fd'`: finite-difference Jacobian action; works for any $f$.
- `'linear'`: only the `linear` part of a `SemilinearRHS` ($L$) — the fast
  matrix-free option, but it **drops the nonlinear Jacobian**. Exact (order 3)
  only when the RHS is linear in the state; on a genuinely semilinear problem
  it is a **first-order** method (measured on viscous Burgers: asymptotic
  order drops 3 → 1). A cheap low-order option, not a stand-in.

**`f_t=`** — the explicit time derivative for non-autonomous RHS:

- `'fd'` (default): one-sided finite difference, reusing the stage-1
  evaluation (one extra $f$ per step; order 3 retained for $\mathrm{eps}$ near
  $\mathrm{machine\_eps}^{1/3}$).
- `'none'`: treat the RHS as autonomous (exact and cheaper when it is).

## Gradient

Forward under `no_grad`, result re-attached to the autograd tape: each stage
solve $K_i = M^{-1}b_i$ and each frozen-operator application $Wx$ is re-attached
as a constant linear map whose adjoint is the **transposed** operator
($M^{-T}$, $W^T$), by backward hooks rather than unrolling the Krylov
iterations. The frozen operator is held constant in the adjoint (its own
derivative is a Hessian term that is dropped): the adjoint error is
$O(\tau)$ per step — the standard Rosenbrock-adjoint approximation — so the
gradient is **exact on linear problems** ($J_N = 0$, nothing dropped) and
accurate to a few percent on nonlinear ones.

## Benchmarks

`scripts/rosenbrock_benchmark.py` → `images/rosenbrock_benchmark.png` (wave
equation and viscous Burgers, NOTES.md §3.14): order 3 measured on both
semilinear problems with `w='jvp'`/`'fd'`, the order drop to 1 with
`w='linear'`, and the per-step cost (2-3 $f$ + 3 GMRES in the shared operator)
vs. the DIRK family at matched accuracy.

## References

- Lang & Verwer, *A family of explicit and implicit partitioned Rosenbrock methods*, BIT 41 (2001) 731-738 — the 3-stage construction and coefficients.
- Hochbrueck & Ostermann, *Exponential integrators*, Acta Numerica 19 (2010) — the surrounding linearly-implicit context (frozen-operator methods).
- Lang, Verwer, *On the construction and application of explicit additively partitioned Runge-Kutta methods* (2004) — AP-RK context.

**Tests:** `tests/test_rosenbrock.py` (order conditions, the $R(\infty)$
damping, the `w`/`f_t` options, the gradient re-attachment on linear
problems).
