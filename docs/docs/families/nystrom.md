---
sidebar_label: Nyström mechanics
---

# Generalized Runge–Kutta–Nyström backends

Runge–Kutta–Nyström methods exploit x′=v and v′=a(t,x,v). Some configurations require velocity-independent acceleration; others support drag and coupled quantities. Choose the force structure first, then compare order and force-call cost on a known oscillator.

`Nystrom(method)` supplies explicit second-order ODE methods outside the fixed
catalog. The system declares matching floating position/velocity fields with
`q'=v` and returns explicit acceleration. Fields are assembled through their
integration metadata. Constraints, mass-matrix solves and custom update
projectors are not part of this contract.

| Method | Order | Estimate exponent | Acceleration contract | Force stages |
|---|---|---|---|---|
| DPRKN4 | 4 | 4 | velocity independent | 4 |
| DPRKN5 | 5 | 5 | velocity independent | 6 |
| DPRKN6 / DPRKN6FM | 6 | 5 | velocity independent | 6 |
| DPRKN8 | 8 | 7 | velocity independent | 9 |
| DPRKN12 | 12 | 11 | velocity independent | 17 |
| ERKN4 | 4 | 4 | velocity independent | 4 |
| ERKN5 | 5 | 5, position only | velocity independent | 4 |
| ERKN7 | 7 | 6 | velocity independent | 7 |
| Nystrom4VelocityIndependent | 4 | none | velocity independent | 3 |
| Nystrom5VelocityIndependent | 5 | none | velocity independent | 4 |
| Nystrom4 | 4 | none | velocity dependent allowed | 4 |
| FineRKN4 | 4 | 4 | velocity dependent allowed | 5 |
| FineRKN5 | 5 | 5 | velocity dependent allowed | 7 |

Each step also evaluates the physical endpoint to refresh copied fields, so it
costs s+1 full RHS calls before structural probes. There is no FSAL saving.
`nystrom_tableau` returns immutable coefficient tuples. `ERKN` entries use the
pinned constant coefficients; no runtime frequency or exact phase-fitting claim
is made. These methods are not automatically symplectic.

## A damped oscillator that converts mechanical energy into heat

For q''=−4q−0.4q', define heat by e'=0.4(q')². The exact underdamped solution
provides all three references, and total energy 2q²+(q')²/2+e remains constant.
FineRKN5 advances the coupled mechanical and first-order thermal quantities.

[example:run nystrom-heat]: #
```python
import math
import torch
from warpSPHIntegrators.schemes import Nystrom
from warpSPHIntegrators.schemes.families.nystrom import NystromWeights
import warpSPHIntegrators.testing as testing
from warpSPHIntegrators import solve

initial = testing.ParticleSystem(testing.ParticleState(
    x=torch.ones(1, dtype=torch.float64), u=torch.zeros(1, dtype=torch.float64),
    e=torch.zeros(1, dtype=torch.float64), m=torch.ones(1, dtype=torch.float64)))

def rhs(system, dt, **kwargs):
    s = system.state
    return testing.ParticleUpdate(s.u, -4*s.x - .4*s.u, .4*s.u.square()), None

quadratures = []
result = solve(Nystrom('FineRKN5'), initial, 1., rhs, dt=.05,
               callback=lambda record: quadratures.append(record.result.quadrature))
w = math.sqrt(4 - .2**2)
q = math.exp(-.2) * (math.cos(w) + .2/w * math.sin(w))
v = -4/w * math.exp(-.2) * math.sin(w)
heat = 2 - (2*q*q + v*v/2)
s = result.state.state
assert result.successful
assert max(abs(s.x.item()-q), abs(s.u.item()-v), abs(s.e.item()-heat)) < 2e-8
assert abs(2*s.x.item()**2 + s.u.item()**2/2 + s.e.item() - 2) < 2e-8
assert isinstance(quadratures[-1], NystromWeights)
```

The standalone `examples/nystrom_demo.py` extends this to vectors with known
frequencies and reports component errors and the total-energy defect. Its
benchmark compares direct Nyström, RK and physical Strang steps. These are toy
mechanics/heat checks, not SPH energy-conservation claims.

## Structural and quantity contracts

Velocity-independent methods require passive extra fields. By default,
`structure='probe'` checks local velocity-independence using AD and perturbed
velocities on owned preprocessed buffers. This costs six extra RHS entries per
step. A caller with an independently verified contract can explicitly use
`structure='declared'`; actual stages still check `q'=v` and passive quantities.
Local probes do not prove independence everywhere. Unsupported AD is reported,
not treated as a positive certificate.

The velocity-dependent methods allow evolving, coupled extra first-order fields.
Their stage values use the velocity matrix `abar`, and final/error values use
the velocity weights `bp`/`bptilde`. This is a mathematical lifting: an extra
quantity acts as the velocity of an unused dummy coordinate, which the RHS
ignores. It avoids attaching thermal evolution to unrelated position weights.

`IntegrationResult.quadrature` carries `NystromWeights`, paired with the stages.
Position weights multiply h² times acceleration, with a separate h*v0 term;
velocity/quantity weights multiply h. The endpoint refresh has zero weight.
Custom finalize hooks set `nystrom_compatible=True` and consume this component
structure. In physical compositions, inner finalize is suppressed and each
substep result retains its quadrature for the single outer hook.

Fixed/adaptive `solve` and in-memory restart preserve ownership and gradients.
Three nonembedded methods refuse adaptation; ERKN5 controls position only.
The runner's local estimator does not guarantee global error equal to rtol.
Portable custom-system serialization remains application-owned.

The pinned `RKN4` formula is excluded: its velocity-only third-order condition
is 1/12 instead of 1/6 and measured velocity convergence is order two. Under
velocity-independent acceleration its position/final coefficients match
Nystrom4VelocityIndependent.
See [source, numerical and consumer evidence](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/nystrom-backend-validation.md).

## Improved Nyström: reusing force history

`ImprovedNystrom(3)` and `ImprovedNystrom(4)` use the previous velocity,
physical acceleration and internal forces to reduce the work of a fixed step.
They require velocity-independent acceleration and passive extra quantities.
Use them for a smooth mechanical problem where evaluating forces is expensive
and a fixed timestep is appropriate. They do not supply an error estimate or
claim symplecticity. Damping through velocity-dependent forces needs another method.
IRKN4 slightly amplifies undamped oscillations even at small steps; choose a
symplectic method when long-time mechanical energy behavior is the priority.

[example:run improved-nystrom]: #
```python
import math
from warpSPHIntegrators.schemes import ImprovedNystrom
import warpSPHIntegrators.testing as testing
from warpSPHIntegrators import solve

problem = testing.oscillator_problem()
method = ImprovedNystrom(4, structure='declared')
first = solve(method, problem.initial(), .5, problem.rhs, dt=.03125)
continued = solve(method, None, 1., problem.rhs, restart=first.checkpoint)
assert continued.successful
assert abs(continued.state.state.x.item() - problem.exact(1.)[0][0]) < 1e-6
```

With declared structure, the first step costs five full RHS calls for order
three, six for order four; warm steps cost two and three respectively. The
default probe adds the same structural checks as velocity-independent Nyström.
The package corrects the pinned source's startup-history timing defect, rather
than reproducing its observed order-two error on time-dependent forces.

`solve` owns history. Direct callers feed each accepted `result.history` into
the next call. A change in timestep, time, integrated values, tensor layout,
method configuration or particle IDs restarts safely. If RHS parameters change,
pass `restart_history=True`. Repeated restarts reduce the force-saving benefit.
Checkpoints for these custom mechanical systems are supported in memory;
portable serialization remains the application's responsibility.

`ImprovedNystromWeights` pairs the current and retained old stage rates with
separate position/velocity weights and an explicit previous-velocity contribution.
A custom finalize hook must set `irkn_compatible=True`, handle those weights and
preserve integrated values, time and IDs. These history methods cannot act as
inner physical splitting flows.

On 4,096 nonlinear forced oscillator modes, sixteen steps of size .0625 used
51 RHS calls for order four versus 64 for Nyström's fourth-order one-step
method. Maximum position/velocity errors were 1.33e-6/4.91e-6 versus
1.08e-5/3.32e-5. Cheap tensor RHS timings favored the one-step method: retaining
and checking history has overhead. This is a measured force/accuracy niche,
not a general wall-clock speed claim.
See [corrected-source and consumer validation](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/improved-nystrom-validation.md).
