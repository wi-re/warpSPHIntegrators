---
sidebar_position: 2
title: Choosing a Scheme
---

# Choosing a Scheme

Sixty-four schemes in eleven families. This page is the decision path; the
[families](families/explicit-rk) pages have the details, the
[solver](solver) page the JFNK mechanics, and the README's
[Supported Integrators table](https://github.com/wi-re/warpSPHIntegrators#supported-integrators)
the full registry reference.

## The decision path

1. **Is the problem stiff?** — i.e. does stability, not accuracy, cap the step
   size (must you take steps far larger than the fast timescale)?
2. **Does it have exploitable structure?** — a Hamiltonian, a semilinear split
   $f = Ly + N$, an additive fast/stiff split, or a parabolic (real-negative-eigenvalue) spectrum.
3. **Order and cost** — evaluations per step, history threading, reuse.

## Non-stiff problems

| Situation | Start with | Why / caveat |
|---|---|---|
| General purpose | **RK4 (classic)** | Order 4 in 4 evaluations; reuse keeps order 3 |
| Adaptive step size | **Dormand–Prince 5(4)** | FSAL: 6 evaluations/step under reuse; embedded error **and** the only published dense output (`dormand_prince_dense_output`) |
| Cheapest adaptive | **Bogacki–Shampine 3(2)** | FSAL: 3 evaluations/step under reuse |
| Conservation laws / SSP | **SSP-RK3** or **TVD-RK2/3**; **SSPRK(10,4)** for order 4 | Strong-stability-preserving; SSPRK(10,4) has the exact SSP coefficient 6.0 at the cost of 10 evaluations |
| Hamiltonian, long runs | **Symplectic Euler** (velocity-dependent force) / **Velocity Verlet** (separable) | Energy error stays in an $O(dt^p)$ band instead of drifting. **Trap:** Verlet, Leap-Frog and the Forest–Ruth variants drop to *first* order under a velocity-dependent force — every real SPH momentum equation. Symplectic Euler keeps second order either way. |
| Hamiltonian, order > 2 | **Gauss-Legendre 2** | The only symplectic scheme above order 2 (order 4, symplecticity pinned by measurement); one block JFNK solve per step |
| Long runs, cheap RHS | **Adams-Bashforth 4/5** | One force evaluation per step — *if* you thread `history=` (see below) |
| Second-order structural system | **Newmark** | $\beta/\gamma$ parametrization, A-stable for $\gamma \ge \tfrac12$ |

## Stiff problems

| Situation | Start with | Why / caveat |
|---|---|---|
| General stiff, no structure | **Radau IIA s=2** or **ESDIRK3(2)4L[2]SA** | Radau: L-stable, damps the whole negative real axis (no A(α) cone). ESDIRK: order 3, L-stable, *stiffly accurate* so first-stage reuse is lossless; embedded (3,2) pair |
| High order, moderate stiffness | **BDF3–BDF5** | The A(α) cone shrinks with order (BDF3: 86.03°, BDF5: 51.84°) — stiff *oscillatory* modes outside the cone are not damped. Cold-start ceiling: the Dormand–Prince bootstrap is explicit with a boundary at $\lvert z\rvert \approx 3.3$, so BDF3–5 cannot *start* at $dt\cdot\text{rate} \gtrsim 3.3$ |
| Strong damping, low order fine | **BDF1/BDF2**, **TR-BDF2**, **SDIRK2** | L-stable; BDF2 is the one-step-history workhorse |
| Parabolic, but no solves allowed | **RKC2** / **RKL2** | Chebyshev/Legendre super-timestepping: stage count $s$ grows like $\sqrt{dt\cdot\lvert\lambda_{max}\rvert}$, so a huge $dt$ costs many *cheap* stages. Built for real (negative) eigenvalues — not oscillatory. Measured fewer total RHS evaluations than BE/BDF2/TR-BDF2 on the diffusion benchmark |
| Stiff, Jacobian–vector products available | **ROS3P** | Order-3 Rosenbrock-W: A-stable (not L), linearly implicit — one JVP per stage, no nonlinear solve; built-in order-2 estimator |
| Semilinear $f = Ly + N$ | **ETD2RK** / **EXPRB32** | Integrate the linear part *exactly* (matrix exponential + $\phi$-functions). EXPRB32 (order 3) freezes the full Jacobian $J_n$; on a linear autonomous problem the step is then the exact exponential flow |
| Additive fast + stiff split | **ARK3(2)4L[2]SA** / **ARK4(3)6L[2]SA** | Order 3/4 additive pairs (Kennedy–Carpenter): explicit half + stiffly-accurate implicit half, L-stable implicit part, embedded pairs. Pass an `IMEXRHS` |
| Same split, one solve per step | **SBDF2 / SBDF3 / CNAB2** | IMEX linear multistep: BDF2/BDF3/trapezoidal backbone, explicit part extrapolated. **The explicit half caps $dt$** — real-axis boundaries 4/3, 20/21, 1, so a convective explicit part needs $dt\cdot\max\lvert\lambda_E\rvert \lesssim 0.5$ (below the ARK pairs' allowance) |
| Mildly stiff, one solve per step | **Adams-Moulton 2–4 (implicit)** | JFNK-corrected AM: AM2 (trapezoidal) is A-stable, AM3/4 only have a bounded region. Iterated correction beats same-order PECE by an order of magnitude on a stiff nonlinear relaxation |
| Stiff + symplectic (separable Hamiltonian) | **Gauss-Legendre 2** | Order 4, symplectic (measured, phase-area defect ~1e-13); not L-stable |

## Cost notes

- **Evaluations per step.** AB: 1 (with `history=`); BDF/AM: 1 RHS + one nonlinear
  solve per step (the solve itself costs GMRES matvecs); explicit RK: the stage
  count; RKC/RKL: $s$ stages chosen per step from $dt\cdot\lvert\lambda_{max}\rvert$.
- **Reuse.** `priorStep=` is lossless for the stiffly accurate DIRKs (the
  previous step's last stage *is* $f(t^{n+1}, y^{n+1})$) and exact for FSAL
  pairs; `warmStart=` takes the previous converged stage *states* as the block
  solve's initial guess for the coupled tableaus. Everything else refuses with
  a warning.
- **History.** The multistep schemes (AB, ABM-PECE, BDF, AM, SBDF/CNAB) need
  `history=` threaded across calls (`StepHistory(maxlen=order - 1)`). Forgetting
  it does **not** corrupt the trajectory — they bootstrap with Dormand–Prince
  5(4) instead, which is *more* accurate, so you silently pay DP5's cost every
  step.
- **Gradients.** Every scheme differentiates through `torch.autograd`; implicit
  schemes flow through their JFNK solves via implicit-function-theorem
  re-attachment (Phase 9), so the adjoint cost is one backward solve per
  Newton iteration, not an unroll.

## When in doubt

`getPreferredScheme(order)` returns a sensible default for a given order;
otherwise: **RK4** for the non-stiff branch, **ESDIRK3(2)4L[2]SA** for the
stiff one, and let the embedded-pair error estimate (plus the adaptive
helpers on the [API page](api)) tell you whether the step size is the right
one.
