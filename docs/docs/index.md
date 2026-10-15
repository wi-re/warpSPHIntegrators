---
sidebar_label: Overview
---

# Time integration for tensors and simulation states

warpSPHIntegrators advances differential equations in PyTorch. Use an ordinary
`f(t, y)` callback for a tensor ODE, or connect a particle/system state whose
positions, velocities and other quantities need coordinated stage updates.
Implicit methods use matrix-free solvers, and differentiable methods retain
state and parameter gradients within their documented limits.

[catalog:summary:start]: #

**142 registered schemes across 11 families**, with orders through 14 and 47 estimate emitters.

[catalog:summary:end]: #

A **scheme** specifies when to evaluate the equations and how to combine those
values into the next state. Classical Runge–Kutta 4 uses four evaluations;
Dormand–Prince 5(4) also provides a comparison solution to control step size.
Higher order can reduce error at a given step size, but each stage and each
implicit solve costs work. Start with a method suited to the equations, then
compare accuracy and total cost on a problem you understand.

## A complete example: damped oscillation

For x'' + .4 x' + x = 0, the oscillations decay inside an exp(-.2 t) envelope.
Writing y=(x,v) turns this into a first-order tensor ODE. This example checks
both position and velocity against the exact solution.

[example:run overview-oscillator]: #
```python
import math
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, solve
from warpSPHIntegrators.schemes import ExplicitRK

rhs = tensor_rhs(lambda t, y: torch.stack((y[1], -.4*y[1] - y[0])))
initial = TensorState(torch.tensor([1., 0.], dtype=torch.float64))
run = solve(ExplicitRK.dormandPrince, initial, 10., rhs, dt=.1,
            adaptive=True, rtol=1e-8, atol=1e-10)
w = math.sqrt(1 - .2**2)
exact = math.exp(-2) * torch.tensor([
    math.cos(10*w) + .2/w*math.sin(10*w),
    -math.sin(10*w)/w,
], dtype=torch.float64)
assert run.successful
assert torch.allclose(run.state.y, exact, rtol=0, atol=2e-7)
```

For the same equation, a step can be accurate yet unstable if it amplifies the
oscillatory modes. A method's **stability region** contains the values h*lambda
for which its linear test solution does not grow. L-stable implicit methods also
damp very fast decaying modes; symplectic methods instead target the geometry of
undamped mechanics. These are different reasons to choose a scheme.

## Where to start

| Your problem | Useful starting point | What to check |
|---|---|---|
| Smooth ordinary ODE | Classical RK4; adaptive Dormand–Prince or Tsitouras 5(4) | Accuracy versus RHS evaluations; dense output currently needs Dormand–Prince |
| Fast decaying modes force tiny explicit steps | L-stable ESDIRK or Radau IIA; Rosenbrock 2(3) | Nonlinear/linear solve convergence and total RHS work |
| Separable undamped mechanics | Velocity Verlet or Gauss–Legendre collocation | Force assumptions and long-time energy behavior |
| Shock-sensitive spatial discretization | SSP/TVD Runge–Kutta | The spatial forward-Euler stability bound still applies |
| Diffusion-dominated spectrum | Stabilized RKL/RKG/ROCK configurations | A negative-real spectral bound, enough stages and acceptable damping |
| A useful constant linear split | Exponential RK | Operator-action cost, nonlinear accuracy and gradient limits |

**TVD** means total variation diminishing: a suitable discretization should not
create extra spatial oscillation. **SSP** methods preserve an appropriate
forward-Euler stability property under a method-specific step bound. They do
not turn an unsuitable spatial operator into a monotone one.

**JFNK** means Jacobian-free Newton–Krylov. Newton solves an implicit stage;
GMRES solves each linear correction using Jacobian–vector products instead of
building a dense Jacobian. Matrix-free storage does not make every solve cheap:
conditioning, tolerances and a suitable preconditioner matter.

Read the [quick start](/docs/quickstart) for the ordinary workflow,
[method selection](/docs/selection) for tradeoffs, [runs](/docs/runs) for acceptance and
restart, and [solvers](/docs/solver) when implicit work dominates. The
[API reference](/docs/api) shows the focused import paths. Existing users should read
[the 0.7 migration guide](/docs/migration).

## The 11 families

[catalog:families:start]: #

| Family | Enum | Schemes | Page |
|---|---|---:|---|
| Explicit RK | `ExplicitRK` | 63 | [explicit-rk](/docs/families/explicit-rk) |
| Symplectic | `Symplectic` | 6 | [symplectic](/docs/families/symplectic) |
| Relaxed Chebyshev | `RelaxedChebyshev` | 3 | [relaxed-chebyshev](/docs/families/relaxed-chebyshev) |
| Multistep explicit | `MultistepExplicit` | 10 | [multistep-explicit](/docs/families/multistep-explicit) |
| DIRK | `DIRK` | 37 | [dirk](/docs/families/dirk) |
| Coupled RK | `CoupledRK` | 4 | [coupled-rk](/docs/families/coupled-rk) |
| Multistep implicit | `MultistepImplicit` | 8 | [multistep-implicit](/docs/families/multistep-implicit) |
| IMEX | `IMEX` | 7 | [imex](/docs/families/imex) |
| Linearly implicit | `LinearlyImplicit` | 1 | [rosenbrock](/docs/families/rosenbrock) |
| Exponential | `Exponential` | 2 | [exponential](/docs/families/exponential) |
| Newmark | `Newmark` | 1 | [newmark](/docs/families/newmark) |

[catalog:families:end]: #

The [searchable catalog](/docs/catalog) retains friendly method names alongside exact
enum identifiers. Parameterized configurations such as collocation, adaptive
Adams/NDF, Nyström and exponential RK extend the fixed catalog without adding an
enum member for every possible setting.

## Examples and measured behavior

The [README](https://github.com/wi-re/warpSPHIntegrators#readme) includes the full
scheme list, oscillator examples, stability plots and benchmark summaries.
[Example scripts and notebooks](https://github.com/wi-re/warpSPHIntegrators/blob/main/examples/README.md)
cover decay, damped oscillation, mechanics, reaction–diffusion and the wave equation.
Use their known solutions to distinguish stability, local error and solver cost.

For example, the measured 1,024-entry manufactured reaction–diffusion problem at
h=.1 gives ETDRK2/3/4 errors about 6.36e-4 / 4.51e-6 / 4.34e-8. Higher order
improves accuracy here, while the older second-order implementation is faster at
its larger error. This is an accuracy/work comparison, not a universal speed
ranking. [The measurement record](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/exponential-validation.md)
keeps the hardware, operator and gradient qualifications.
