---
sidebar_label: Rosenbrock-W
---

# Linearly implicit — Rosenbrock-W (`LinearlyImplicit`) — 1 scheme

The fixed catalog contains **ROS3P**, a 3-stage, order-3,
**A-stable** (not L-stable) Rosenbrock-W method with a built-in order-2
embedded estimate, from the 3-stage Rosenbrock-W construction in **Lang &
Verwer, BIT 41(4) (2001) 731-738** (the citable coefficient source). Driven by
`rosenbrock.py`.

## Additional Rosenbrock factories

`Rosenbrock(method)` supplies three additional methods outside the fixed catalog.
They freeze the full RHS Jacobian at the start of each step, apply it through
automatic differentiation, and solve linear stage equations with GMRES. They
need no outer Newton iteration or user-supplied semilinear split.

| Method | Order | Difference exponent | Linear stability guidance |
|---|---:|---:|---|
| `Rosenbrock23` | 2 | 3 | A- and L-stable; a low-order option for stiff decay |
| `Rosenbrock32` | 3 | 3 | Bounded stiff region; stiff modes can grow |
| `RosenbrockW6S4OS` | 4 | none | Damps very stiff real modes, but is not A-stable |

For Rosenbrock23/32, the package corrects a pinned-source time-derivative term
that otherwise spoils Rosenbrock32's nonautonomous order and the pair's error
estimate. The correction is verified by adding time as an extra state variable
with t'=1. It leaves autonomous maps unchanged. W6S4OS uses the source coefficients.

Use `solver_opts={'gmres_tol': 1e-12, 'gmres_maxiter': 100}` for linear-solve
controls. A preconditioner has the usual `(vector, state, context)` interface;
`context['stage_coefficient']` is h*gamma. All stages share that frozen operator.
`f_t` can supply a callable returning an exact partial-time update. The default
uses isolated central time probes; `ft_eps` sets their spacing. Choose spacing
for the forcing timescale, or supply the derivative for strict accuracy work.
Solver and time-probe errors limit accuracy below their numerical floors.

These factories retain state and RHS-parameter derivatives through the Jacobian,
including the Hessian terms dropped by the legacy ROS3P frozen-operator adjoint.
Extra integrated quantities join the same coupled stage solve. JVP/time probes
use isolated lifecycle state; the physical endpoint is refreshed for copied fields.
Custom finalize sets `rosenbrock_compatible=True` and consumes
`RosenbrockQuadrature`: its weighted, already h-scaled increments reconstruct
the update, while `result.stages` retains the physical RHS rates. Those are
different objects. No mass-matrix, FSAL or source cache-controller policy is implied.

Rosenbrock23/32 support adaptive runs and portable tensor continuation; resupply
solver options and RHS callbacks on restart. W6S4OS requires fixed timesteps.
All three support physical composition and algorithm switching.

On 4,096 nonlinear forced reaction modes, sixteen steps of size .0625 gave
W6S4OS error 1.90e-7 with 618 actual RHS calls and .289 seconds, versus
1.75e-7, 880 calls and .382 seconds for six-stage ESDIRK4. This is a measured
similar-error benefit on this known problem, including JVP/time probes. On stiff
linear decay, ESDIRK4 was faster. Rosenbrock32 amplified the fastest stiff modes
to about 1,445 despite converged stage solves: successful linear algebra alone
does not establish numerical stability. W6S4OS also slightly amplifies purely
oscillatory modes; it is not an A/L-stable replacement for Rosenbrock23.
See [source, gradient and consumer validation](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/linearized-validation.md).

## ROS3P

The following coefficients and options describe the fixed ROS3P record.

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

`benchmarks/rosenbrock_benchmark.py` → `docs/assets/rosenbrock_benchmark.png` (wave
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
