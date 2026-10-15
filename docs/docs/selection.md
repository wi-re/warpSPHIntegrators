---
sidebar_position: 2
title: Choosing a Scheme
---

# Choosing a Scheme

[catalog:summary:start]: #

**142 registered schemes across 11 families**, with orders through 14 and 47 estimate emitters.

[catalog:summary:end]: #

This page is the decision path; the
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
| Hamiltonian, order > 2 | **Gauss-Legendre 2/3** | Symplectic collocation at orders 4/6; one block JFNK solve per step |
| Long runs, cheap RHS | **Adams-Bashforth 4/5** | One force evaluation per step — *if* you thread `history=` (see below) |
| Second-order structural system | **Newmark** | Default $\beta=1/4$, $\gamma=1/2$ is unconditionally stable for linear mechanics; other parameters need their stability conditions |
| Controllable damping of fast structural modes | **GeneralizedAlpha(rho_inf)** factory | Fixed-step constant-mass mechanics; accepted acceleration memory; see [the known consumer](structural-demo) |

## Stiff problems

| Situation | Start with | Why / caveat |
|---|---|---|
| General stiff, no structure | **Radau IIA s=2** or **ESDIRK3(2)4L[2]SA** | Radau: L-stable, damps the whole negative real axis (no A(α) cone). ESDIRK: order 3, L-stable, *stiffly accurate* so first-stage reuse is lossless; embedded (3,2) pair |
| High order, moderate stiffness | **BDF3–BDF5** | The A(α) cone shrinks with order (BDF3: 86.03°, BDF5: 51.84°) — stiff *oscillatory* modes outside the cone are not damped. Order-matched implicit starters remove the old explicit cold-start ceiling; stiffness-dependent accuracy and the steady-state cone still apply |
| Strong damping, low order fine | **BDF1/BDF2**, **TR-BDF2**, **SDIRK2** | L-stable; BDF2 is the one-step-history workhorse |
| Parabolic, but no solves allowed | **RKC2** / **RKL2** | Chebyshev/Legendre super-timestepping: stage count $s$ grows like $\sqrt{dt\cdot\lvert\lambda_{max}\rvert}$, so a huge $dt$ costs many *cheap* stages. Built for real (negative) eigenvalues — not oscillatory. Measured fewer total RHS evaluations than BE/BDF2/TR-BDF2 on the diffusion benchmark |
| Stiff, Jacobian–vector products available | **Rosenbrock 2(3) or ROS3P** | Rosenbrock 2(3) is A/L-stable and supports full Jacobian derivatives; legacy ROS3P is order three, A-stable (not L), linearly implicit — one linear solve per stage, potentially many JVPs in GMRES; built-in order-2 estimator |
| Semilinear $f = Ly + N$ | **ExponentialRK('ETDRK4')** / **ExponentialRK('Exprb43')** | Integrate the linear part *exactly* (matrix exponential + $\phi$-functions). Exprb43 freezes the full Jacobian $J_n$ and augments time; its dense operator-AD limit and large-action budget are explicit; on a linear autonomous problem the step is then the exact exponential flow |
| Additive fast + stiff split | **ARK3(2)4L[2]SA** / **ARK4(3)6L[2]SA** | Order 3/4 additive pairs (Kennedy–Carpenter): explicit half + stiffly-accurate implicit half, L-stable implicit part, embedded pairs. Pass an `IMEXRHS` |
| Same split, one solve per step | **SBDF2 / SBDF3 / CNAB2** | IMEX linear multistep: BDF2/BDF3/trapezoidal backbone, explicit part extrapolated. **The explicit half caps $dt$** — real-axis boundaries 4/3, 20/21, 1, so a convective explicit part needs $dt\cdot\max\lvert\lambda_E\rvert \lesssim 0.5$ (below the ARK pairs' allowance) |
| Mildly stiff, one solve per step | **Adams-Moulton 2–4 (implicit)** | JFNK-corrected AM: AM2 (trapezoidal) is A-stable, AM3/4 only have a bounded region. Iterated correction beats same-order PECE by an order of magnitude on a stiff nonlinear relaxation |
| Stiff + symplectic (separable Hamiltonian) | **Gauss-Legendre 2** | Order 4, symplectic (measured, phase-area defect ~1e-13); not L-stable |

The [DIRK catalog](families/dirk) now includes Kvaerno, KenCarp, ARS,
Cash/Hairer and ESDIRK methods through order 6. Kvaerno4 and SSPSDIRK2
are A-stable with nonzero stiff limits; SFSDIRK4–8 have bounded stability
regions. The KenCarp/ARS names here provide the implicit half only.

## Cost notes

- **Evaluations per step.** AB: 1 (with `history=`); BDF/AM: 1 RHS + one nonlinear
  solve per step (the solve itself costs GMRES matvecs); explicit RK: the stage
  count; RKC/RKL: $s$ stages chosen per step from $dt\cdot\lvert\lambda_{max}\rvert$.
- **Reuse.** `priorStep=` is lossless for the stiffly accurate DIRKs with an explicit first stage (the
  previous step's last stage *is* $f(t^{n+1}, y^{n+1})$) and exact for FSAL
  pairs; `warmStart=` takes the previous converged stage *states* as the block
  solve's initial guess for the coupled tableaus. Everything else refuses with
  a warning.
- **History.** Manual multistep loops must thread accepted `result.history` into
  the next step. Missing history repeatedly invokes startup, changing cost and
  potentially effective trajectory order. Implicit BDF uses order-matched
  implicit starters; other families have their own startup contracts. `solve`
  manages history and compatible reuse. Fixed-coefficient drivers restart on
  changed timesteps; VariableAdams/NDF provide distinct variable-history policies.
- **Gradients.** Ordinary explicit drivers use `torch.autograd`; nonlinear
  implicit drivers reattach the converged solution using the implicit function
  theorem, with an adjoint solve for each forward stage/block solve. The Newton
  iterations run without a tape. Rosenbrock and exponential drivers have their
  own frozen-operator adjoint paths. See the dedicated gradient tests.

## When in doubt

`get_scheme` selects an explicit enum identity;
otherwise: **RK4** for the non-stiff branch, **ESDIRK3(2)4L[2]SA** for the
stiff one, and let the embedded-pair error estimate (plus the adaptive
helpers on the [API page](api)) tell you whether the step size is the right
one.
