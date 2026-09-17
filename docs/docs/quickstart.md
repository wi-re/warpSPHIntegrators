---
sidebar_position: 1
title: Quick Start
---

A minimal but complete integration: define a state, define the update
structure, implement the integration system protocol, define the
right-hand side, and pick a scheme. This page mirrors the
[Quick Start in the README](https://github.com/wi-re/warpSPHIntegrators#quick-start)
(keep the two in sync when either changes).


## 1. Define Your State

Create a state class that inherits from `BaseState` and annotate fields with their integration behavior:

```python
from dataclasses import dataclass
import torch
from warpSPHIntegrators import BaseState, integrated, constant

@dataclass
class MyState(BaseState):
    # Fields that evolve over time
    position: torch.Tensor = integrated('dx_dt', tags=('position',))
    velocity: torch.Tensor = integrated('dv_dt', tags=('velocity',))
    
    # Constant parameters (don't change during integration)
    mass: torch.Tensor = constant(tags=('mass',))
    spring_constant: torch.Tensor = constant(tags=('spring_constant',))
```

**Field Behavior Options:**

| Behavior | `initializeNewState()` | End of step | For |
|---|---|---|---|
| `integrated('dfield', ...)` | cloned | advanced by the scheme | position, velocity, energy |
| `constant(...)` | cloned | unchanged | mass, material parameters |
| `copied(...)` | nulled | **taken from the last stage** | quantities recomputed per stage: summation density, pressure, smoothing length |
| `ephemeral(...)` | nulled | stays null | stage-local scratch, neighbour lists |
| `custom(...)` | nulled | your responsibility | anything the generic paths must not touch |

A field with no behavior declared is treated as `constant`. That is the conservative
default: it survives both `clone()` and `initializeNewState()`, so a forgotten decorator
cannot silently turn a tensor into `None` halfway through a step.

**Field value types.** Cloning is dispatched on the value's type, not hard-coded to
`torch.Tensor`. Out of the box: torch tensors, `wp.array` (detected without importing
warp), and lists / tuples / dicts recursed into. Immutable scalars are shared, which is
safe. Anything else is shared **by reference** and warns once — that would mean every
stage writing into the same object — so register a handler for it:

```python
from warpSPHIntegrators import register_clone_handler

register_clone_handler(
    'mylib.Buffer',
    matches=lambda v: isinstance(v, mylib.Buffer),
    clone=lambda v, detach: v.copy(),
    to_device=lambda v, device: v.to(device),   # optional, drives BaseState.to()
)
```

## 2. Define Your Update Structure

Create a dataclass that holds all derivatives and auxiliary values:

```python
from dataclasses import dataclass
from warpSPHIntegrators import tagged

@dataclass
class MyUpdate:
    dx_dt: torch.Tensor = tagged(tags=('position_derivative',))
    dv_dt: torch.Tensor = tagged(tags=('velocity_derivative',))
```

## 3. Implement the Integration System Protocol

Create an `IntegrationSystem` that knows how to apply updates to your state:

```python
from warpSPHIntegrators import BaseIntegrationSystem, IntegrationSystem, PositionUpdateSpec, ComponentUpdateSpec
from dataclasses import dataclass

@dataclass
class MySystem(BaseIntegrationSystem):
    state: MyState = reference_state(tags=('my_state',))
    t: float = 0.0  # Current time
    
    def initializeNewState(self, *args, **kwargs):
        """Create a fresh state copy for the next integration step."""
        current_state = get_reference_state(self)
        return MySystem(
            state=current_state.initializeNewState(),
            t=self.t
        )
    
    def apply_position_update(self, update: MyUpdate, spec: PositionUpdateSpec, **kwargs):
        """Apply a position update using velocity (semi-implicit or explicit)."""
        # Implementation handles velocity drift and derivative steps
        return update_position(self, update, spec, 
                             'position', 'position_derivative',
                             'velocity', 'velocity_derivative')
    
    def apply_velocity_update(self, update: MyUpdate, spec: ComponentUpdateSpec, **kwargs):
        """Apply a velocity update."""
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')
    
    def apply_state_update(self, update: MyUpdate, spec: ComponentUpdateSpec, **kwargs):
        """Apply all state updates in sequence."""
        # Note: DO NOT advance self.t here. Time is managed by the integrator.
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        return self
```

**Protocol Methods:**
- `initializeNewState()` — Create a fresh state for the next step
- `apply_position_update()` — Position update (often semi-implicit with velocity)
- `apply_velocity_update()` — Velocity update
- `apply_quantity_update()` — Other field updates
- `apply_state_update()` — General state update (calls the above in sequence)

## 4. Define Your Right-Hand Side (RHS) Function

The RHS function receives the current system and returns an update structure plus optional auxiliary values:

```python
def my_rhs(system: MySystem, dt: float, verbose: bool = False) -> tuple:
    """
    Compute derivatives and auxiliary values.
    
    Returns:
        (update: MyUpdate, aux: tuple)
    """
    state = get_reference_state(system)
    
    # Compute derivatives
    dx_dt = state.velocity
    dv_dt = -(state.spring_constant / state.mass) * state.position
    
    # Auxiliary values (optional, for logging/analysis)
    kinetic_energy = 0.5 * state.mass * state.velocity**2
    potential_energy = 0.5 * state.spring_constant * state.position**2
    
    return (
        MyUpdate(dx_dt=dx_dt, dv_dt=dv_dt),
        (kinetic_energy, potential_energy)
    )
```

This plain callable is all most schemes need. Split methods take a
**structured RHS** instead: `IMEXRHS(explicit=, implicit=)` for the additive
split `f = f_E + f_I` (IMEX Euler, ARK3/ARK4, SBDF2/3, CNAB2), or `SemilinearRHS(linear=,
nonlinear=)` for the semilinear split `f = L·y + N`. Both remain callable as
the combined `f`, so a scheme that only needs the full right-hand side treats
the split as fully implicit; a scheme that needs a split the RHS does not
declare fails before the solve with an error naming the missing capability.
See [NOTES.md §3.12](https://github.com/wi-re/warpSPHIntegrators/blob/main/NOTES.md#312-phase-14-the-structured-rhs-interface--done-2026-09-10).

## 5. Choose an Integration Scheme and Integrate

```python
from warpSPHIntegrators import getIntegrator, IntegrationSchemeType

# Create initial system
initial_state = MyState(
    position=torch.tensor([1.0]),
    velocity=torch.tensor([0.0]),
    mass=torch.tensor([1.0]),
    spring_constant=torch.tensor([5.0])
)
system = MySystem(state=initial_state, t=0.0)

# Get an integrator
integrator = getIntegrator(IntegrationSchemeType.rungeKutta4)

# Integrate one step
dt = 0.01
result = integrator.function(
    system,
    dt=dt,
    f=my_rhs,
    verbose=False
)

# Access results
next_system = result.state  # IntegrationResult.state
last_stage = result.stages[-1]  # Last StageResult
aux_values = last_stage.aux  # Auxiliary output from RHS
```

## Return Type: `IntegrationResult`

All integrator functions return an `IntegrationResult` named tuple:

```python
from warpSPHIntegrators import IntegrationResult, StageResult

result: IntegrationResult = integrator.function(...)

# Fields:
result.state        # The new system state after integration
result.stages       # List[StageResult] — all intermediate stages

# Each stage:
stage: StageResult = result.stages[-1]
stage.aux           # Auxiliary values returned from RHS
stage.update        # The k-value (derivative) used at this stage
```

This replaces the older tuple return format and provides a clear, self-documenting API.