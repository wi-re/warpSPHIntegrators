# Quick start

Start with a tensor ODE if your equations fit f(t,y). Use a custom state/system
when a simulation needs positions, velocities, auxiliary quantities and stage
hooks to move together. Both paths use the same scheme selection and run control.

From a checkout, install with `python -m pip install -e .`. Contributors can use
`python -m pip install -e '.[test]'` for the test dependencies.

## Solve a known equation

The equation y'=-y has the exact solution y(t)=y(0)*exp(-t). It is a useful first
check because numerical error, time and the expected decay are all explicit.

[example:run quickstart-decay]: #
```python
import math
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, solve
from warpSPHIntegrators.schemes import ExplicitRK

initial = TensorState(torch.tensor([1., 2.], dtype=torch.float64))
rhs = tensor_rhs(lambda t, y: -y)
run = solve(ExplicitRK.tsit5, initial, 1., rhs, dt=.1,
            adaptive=True, rtol=1e-7, atol=1e-9)
assert run.successful
assert torch.allclose(run.state.y, initial.y*math.exp(-1), rtol=1e-6)
assert initial.t == 0.  # solve owns the accepted/trial states
```

`ExplicitRK.tsit5` selects **Tsitouras 5(4)**. The main solution has order five;
a fourth-order comparison estimates local error so the runner can adjust the
step. An enum is an unambiguous configuration key. The catalog also accepts its
exact stored lookup names; friendly display labels are not additional aliases.

Use `float64` while checking a new problem against a known solution. Smaller
steps eventually encounter floating-point or solver accuracy limits; they do
not improve the result indefinitely.

## Take a step yourself

[example:run quickstart-step]: #
```python
from warpSPHIntegrators import get_scheme, TensorState, tensor_rhs
from warpSPHIntegrators.schemes import ExplicitRK
import torch

method = get_scheme(ExplicitRK.rungeKutta4)  # classical Runge–Kutta 4
state = TensorState(torch.ones(1, dtype=torch.float64))
step = method(state, .01, tensor_rhs(lambda t, y: -y))
assert step.step_status.successful
assert step.state.t == .01
```

A step returns `IntegrationResult(state, stages, ...)`. Each `StageResult`
pairs the physical update with optional auxiliary data. Some methods also
return an error estimate and accepted-history candidate. In a manual loop, keep
`step.state` and thread `step.history` into the next call for history methods.
The [run API](runs) handles acceptance, rejection and history for complete
trajectories; direct stepping leaves those decisions to the caller.

## Connect a complete simulation state

The state protocol distinguishes fields by what should happen at a stage:

| Field declaration | Behavior |
|---|---|
| `integrated(update_key, ...)` | Advance using a declared derivative, including every integrated quantity |
| `constant(...)` | Carry a physical/configuration value without integrating it |
| `copied(...)` | Refresh a value calculated at a physical stage, such as a density estimate |
| `reference_state(...)` | Identify the state inside a larger system |
| `tagged(...)` | Give fields physical roles used by typed component-update helpers |

A system supplies new-stage construction and typed update operations. Its RHS
receives `(system, dt, ...)` and returns an update, optionally followed by
auxiliary values. Hooks can prepare stage data, process RHS results and finalize
the accepted step. The integrator owns the stage clock.

This complete oscillator example shows those pieces without a particle-neighbor
backend. The exact solution has position cos(2t) and velocity -2sin(2t).

[example:include examples/harmonic_oscillator.py]: #
```python
"""Complete CPU example: python examples/harmonic_oscillator.py."""

from dataclasses import dataclass

import torch

from warpSPHIntegrators.state import (
    BaseIntegrationSystem, BaseState, ComponentUpdateSpec, PositionUpdateSpec,
    constant, get_reference_state, integrated, reference_state, tagged,
)
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.state.fields import update_component, update_position


@dataclass
class OscillatorState(BaseState):
    position: torch.Tensor = integrated('dx_dt', tags=('position',))
    velocity: torch.Tensor = integrated('dv_dt', tags=('velocity',))
    mass: torch.Tensor = constant()
    spring_constant: torch.Tensor = constant()


@dataclass
class OscillatorUpdate:
    dx_dt: torch.Tensor = tagged(tags=('position_derivative',))
    dv_dt: torch.Tensor = tagged(tags=('velocity_derivative',))


@dataclass
class OscillatorSystem(BaseIntegrationSystem):
    state: OscillatorState = reference_state()
    t: float = 0.0

    def initializeNewState(self, *args, **kwargs):
        return OscillatorSystem(get_reference_state(self).initializeNewState(), self.t)

    def apply_position_update(self, update, spec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative',
                               'velocity', 'velocity_derivative')

    def apply_velocity_update(self, update, spec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')

    def apply_state_update(self, update, spec, **kwargs):
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        return self


def rhs(system, dt, **kwargs):
    state = get_reference_state(system)
    return OscillatorUpdate(
        dx_dt=state.velocity,
        dv_dt=-(state.spring_constant / state.mass) * state.position,
    ), None


def run_example():
    system = OscillatorSystem(OscillatorState(
        position=torch.tensor([1.0], dtype=torch.float64),
        velocity=torch.tensor([0.0], dtype=torch.float64),
        mass=torch.tensor([1.0], dtype=torch.float64),
        spring_constant=torch.tensor([4.0], dtype=torch.float64),
    ))
    scheme = get_scheme('RK4')
    for _ in range(100):
        result = scheme(system, dt=0.01, f=rhs)
        system = result.state
    return result


if __name__ == '__main__':
    result = run_example()
    state = get_reference_state(result.state)
    print(f't={result.state.t:.2f}, x={state.position.item():.6f}, '
          f'v={state.velocity.item():.6f}, stages={len(result.stages)}')
```

The example's `update_position` and `update_component` helpers use field tags,
so additional quantities do not need one scheme-specific code path each.
A custom finalizer must understand the selected family's quadrature. Ordinary RK
uses scalar derivative weights; Nyström and exponential families can carry
separate component or operator-weighted increments. Keep finalization compatible
with the family's contract rather than treating all weights as interchangeable.

## Add implicit solves when the equations need them

Stiff equations can require tiny explicit steps for stability even when the
visible solution varies slowly. ESDIRK and Radau methods solve stage equations;
JFNK uses Newton corrections and matrix-free GMRES linear solves.

[example:run quickstart-implicit]: #
```python
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, solve
from warpSPHIntegrators.schemes import DIRK

run = solve(DIRK.esdirk324l2sa, TensorState(torch.ones(1, dtype=torch.float64)),
            .5, tensor_rhs(lambda t, y: -10*y), dt=.05,
            solver_opts={'matvec': 'jvp', 'tol': 1e-12, 'newton_tol': 1e-9})
assert run.successful
assert torch.isfinite(run.state.y).all()
```

Inspect convergence before interpreting cost or accuracy. A fixed Picard
iteration budget can diverge on a stiff problem even when the underlying
implicit method is stable. The [solver guide](solver) explains the diagnostics,
JVP/finite-difference choices and preconditioners.

Next, use [method selection](selection) to choose the right family, [runs](runs)
for tolerances/events/restart, and [options](options) for the exact step controls.
The [API reference](api) lists the focused imports for custom systems and backends.
