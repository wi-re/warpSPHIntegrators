# sphWarpIntegrators — Differentiable ODE Integration with PyTorch

A flexible, fully differentiable numerical ODE integration library for PyTorch. Implements multiple integration schemes with support for complex state management, custom field behavior, and both typed and legacy APIs.

> **Status.** Today this is a pure-PyTorch library; there is no NVIDIA Warp code in it
> yet. The name describes where it is going, not what it does. State cloning is now
> backend-dispatched, so `wp.array` fields are copied correctly and a Warp state is safe
> to hold — but a Warp *backend* still needs a preallocated-buffer path instead of the
> allocate-per-stage design, and a decision between torch autograd and `wp.Tape` for
> gradients. See `NOTES.md`.

## Overview

This library provides a set of time integration schemes for solving systems of ordinary differential equations (ODEs) where states may be complex objects with multiple field types and integration behaviors. All computations are differentiable through PyTorch, making the library suitable for physics-informed machine learning, neural ODEs, and scientific computing.

The integrators never refer to `position` / `velocity` / `density` by name. They only say
"advance the component tagged X by `c·dt` times the derivative tagged Y, optionally
blended with a reference state", and your system object decides what that means.

### Key Features

- **Multiple Integration Schemes**: Runge-Kutta up to 5th order, embedded FSAL pairs (Bogacki–Shampine, Dormand–Prince, Cash–Karp), TVD-RK2/3, symplectic Verlet, Forest–Ruth high-order, and Euler methods
- **Flexible State Management**: Custom state objects with metadata-driven field behavior (integrated, constant, copied, ephemeral, custom)
- **Type-Safe Protocol**: Structural typing for integration systems with clear separation of concerns
- **Fully Differentiable**: All operations preserve gradient flow for end-to-end learning
- **Extensible**: Easy to add new schemes or state types

## Installation

```bash
pip install sphWarpIntegrators
```

The distribution is `sphWarpIntegrators`; the import name is `integrators`.

For a checkout, with the test dependencies:

```bash
pip install -e ".[test]"
pytest
```

## Quick Start

### 1. Define Your State

Create a state class that inherits from `BaseState` and annotate fields with their integration behavior:

```python
from dataclasses import dataclass
import torch
from integrators import BaseState, integrated, constant

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
from integrators import register_clone_handler

register_clone_handler(
    'mylib.Buffer',
    matches=lambda v: isinstance(v, mylib.Buffer),
    clone=lambda v, detach: v.copy(),
    to_device=lambda v, device: v.to(device),   # optional, drives BaseState.to()
)
```

### 2. Define Your Update Structure

Create a dataclass that holds all derivatives and auxiliary values:

```python
from dataclasses import dataclass
from integrators import tagged

@dataclass
class MyUpdate:
    dx_dt: torch.Tensor = tagged(tags=('position_derivative',))
    dv_dt: torch.Tensor = tagged(tags=('velocity_derivative',))
```

### 3. Implement the Integration System Protocol

Create an `IntegrationSystem` that knows how to apply updates to your state:

```python
from integrators import BaseIntegrationSystem, IntegrationSystem, PositionUpdateSpec, ComponentUpdateSpec
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

### 4. Define Your Right-Hand Side (RHS) Function

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

### 5. Choose an Integration Scheme and Integrate

```python
from integrators import getIntegrator, IntegrationSchemeType

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
from integrators import IntegrationResult, StageResult

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

## Available Integration Schemes

`Order` throughout is the **global** convergence order, verified by `tests/test_convergence.py`
on an autonomous, a time-dependent, and a nonlinear problem. `Reuse` is the order retained
when the previous step's last stage is fed back in as `k0` — see
[First-stage reuse](#first-stage-reuse-priorstep).

### Explicit Runge-Kutta Methods

| Scheme | Order | Stages | Reuse | Use Case |
|--------|-------|--------|-------|----------|
| **Forward Euler** / **Explicit Euler** | 1 | 1 | refused | Quick, low-accuracy tests; baseline |
| **Midpoint** / **RK2** / **EPEC** | 2 | 2 | 2 ✓ | Moderate accuracy in 2 evaluations; reuse is free |
| **Heun 2nd** / **EPEC Modified** | 2 | 2 | 2 ✓ | As above |
| **Ralston 2nd** | 2 | 2 | 1 | Minimises the local error constant |
| **RK3**, **Heun 3rd**, **Ralston 3rd**, **Wray 3rd** | 3 | 3 | 1–2 | Third order in 3 evaluations |
| **SSP-RK3** | 3 | 3 | 1 | Conservation laws; strong-stability-preserving |
| **RK4 (Classic)** | 4 | 4 | 3 | High accuracy; the usual default |
| **RK4 (alternative)** | 4 | 4 | 2 | 3/8 rule |
| **Nyström 5th** | 5 | 6 | 1 | Fifth order without step control |

### Embedded pairs (error estimate + adaptive `dt`)

These return a step-size-control estimate in `IntegrationResult.error`. The first two are
**FSAL**, so first-stage reuse is exact and costs one evaluation less per step.

| Scheme | Order | Stages | Reuse | Use Case |
|--------|-------|--------|-------|----------|
| **Bogacki–Shampine 3(2)** | 3 | 4 (3 under reuse) | 3 ✓ FSAL | Cheapest useful pair; `scipy`'s `RK23` |
| **Dormand–Prince 5(4)** | 5 | 7 (6 under reuse) | 5 ✓ FSAL | The workhorse; `scipy`'s `RK45`, MATLAB's `ode45` |
| **Cash–Karp 5(4)** | 5 | 6 | 1 | Well-conditioned pair when DP5 is more than needed |

### Symplectic Methods

Energy error stays inside an `O(dt^p)` band however long the run, rather than drifting
secularly — this is what `dissipation=False` on the scheme records, and
`tests/test_hamiltonian.py` holds them to it.

**These schemes assume a separable Hamiltonian**, i.e. a force depending on position
alone. With a velocity-dependent force — artificial viscosity, drag, any real SPH
momentum equation — Leap Frog, Velocity Verlet, PEFRL and VEFRL all drop to **first
order**. Symplectic Euler does not; it keeps second order either way.

| Scheme | Order | Reuse | Use Case |
|--------|-------|-------|----------|
| **Semi-Implicit Euler** | 1 | refused | Cheapest symplectic scheme |
| **Symplectic Euler** | 2 | 1 | Kick-drift-kick; second order even for velocity-dependent forces |
| **Velocity Verlet** | 2 (1 if `f` sees velocity) | 2 ✓ | Standard Verlet; reuse is free (FSAL property) |
| **Leap-Frog** | 2 (1 if `f` sees velocity) | 1 | Synchronised form |
| **PEFRL** | 4 (1 if `f` sees velocity) | refused | High-order Forest–Ruth, position-first |
| **VEFRL** | 4 (1 if `f` sees velocity) | refused | Velocity-first variant |

### TVD and Conservative Schemes

| Scheme | Order | Reuse | Use Case |
|--------|-------|-------|----------|
| **TVD-RK2** | 2 | refused | Conservation laws with the TVD property |
| **TVD-RK3** | 3 | refused | Shu–Osher form of SSP-RK3 |

## First-stage reuse (`priorStep`)

Passing `priorStep=result.stages[-1]` feeds the last stage of step *n* in as the first
stage of step *n+1*, saving one right-hand-side evaluation. It is standard practice in
the SPH literature, but **whether it is valid is a property of the tableau, not of the
caller** — so ask before opting in:

```python
from integrators import getIntegrator, step_reuse_order, supports_step_reuse

scheme = getIntegrator('Dormand-Prince 5(4)')
supports_step_reuse(scheme)   # True  -- FSAL, reuse is exact
step_reuse_order(scheme)      # 5

scheme = getIntegrator('SSP RK3')
supports_step_reuse(scheme)   # False
step_reuse_order(scheme)      # 1  -- third order becomes first
```

Supplying `priorStep` to a scheme that loses order still works — trading order for half
the evaluations is a legitimate choice — but warns once, naming the order you will
actually get. `step_reuse_order` returns `None` for schemes that do not implement reuse
at all; they warn and ignore it. Single-stage schemes refuse outright, because reuse
turns them into `y^{n+1} = y^n + dt·f(y^{n-1})`, which has no stability region on the
imaginary axis.

```python
result = scheme(system, dt=dt, f=rhs)
result.error        # embedded pairs only: y_high - y_low, as a state
```

## Example: Damped Harmonic Oscillator with Feedback

See [integrators.ipynb](integrators.ipynb) for the complete interactive example. This example demonstrates:

- Custom state with multiple field types
- Energy-dependent feedback (spring constant varies with stored energy)
- Comparison of three integration schemes (RK4, RK2, Forward Euler)
- Visualization of phase plot, energy evolution, and trajectories

**Example Output:**

![Integration schemes comparison: Phase plot and energy evolution](images/example.png)

The visualization shows:
- **Phase Plot (top-left)**: How position and velocity evolve together
- **Energy (top-right & bottom-right)**: Kinetic, potential, and total energy over time
- **Position/Velocity Trajectories**: Individual component evolution for each scheme

## Integrator Image Gallery

| Order | Modified Harmonic Oscillator | Integrator Comparison |
|------:|-------------------------------|-----------------------|
| 1 | ![Modified harmonic oscillator: order 1 integrators](images/modified_harmonic_oscillator_order_1_integrators.png) | ![Integrator comparison for order 1](images/integrator_comparison_order_1.png) |
| 2 | ![Modified harmonic oscillator: order 2 integrators](images/modified_harmonic_oscillator_order_2_integrators.png) | ![Integrator comparison for order 2](images/integrator_comparison_order_2.png) |
| 3 | ![Modified harmonic oscillator: order 3 integrators](images/modified_harmonic_oscillator_order_3_integrators.png) | ![Integrator comparison for order 3](images/integrator_comparison_order_3.png) |
| 4 | ![Modified harmonic oscillator: order 4 integrators](images/modified_harmonic_oscillator_order_4_integrators.png) | ![Integrator comparison for order 4](images/integrator_comparison_order_4.png) |
| 5 | ![Modified harmonic oscillator: order 5 integrators](images/modified_harmonic_oscillator_order_5_integrators.png) | ![Integrator comparison for order 5](images/integrator_comparison_order_5.png) |

## API Reference

### State Definition

```python
from integrators import BaseState, integrated, constant, copied, ephemeral, custom

@dataclass
class MyState(BaseState):
    # Integrated: evolves using derivative field
    x: torch.Tensor = integrated('dx_dt', tags=('position',))
    
    # Constant: fixed throughout integration
    m: torch.Tensor = constant(tags=('mass',))
    
    # Copied: recomputed each stage; the final state inherits the last stage's value
    density: torch.Tensor = copied(default=None)
    
    # Ephemeral: stage-local scratch, never carried out of the step
    neighbours: Any = ephemeral(default=None)
    
    # Custom: the library will not touch it; your system owns it entirely
    custom_field: Any = custom(default=None)
```

### Update Specifications

```python
from integrators import (
    PositionUpdateSpec, ComponentUpdateSpec, StateBlend,
    blend_state, explicit_step, semi_implicit_position_step, verlet_position_step,
)

# `StateBlend` is a three-field dataclass describing how the existing value is folded
# in:  value <- self_scale * value + reference_weight * reference_state_value + ...
# `reference_state` and `reference_weight` must be given together.
blend = blend_state(self_scale=3/4, reference_state=y_1, reference_weight=1/4)

# Component (velocity, quantity, ...) update: value += derivative_dt * k
comp_spec = ComponentUpdateSpec(derivative_dt=0.5 * dt, blend=blend)

# Position update, plus optionally one of the two special position modes:
#   current_velocity_dt=s  ->  x += s * v_current       (semi-implicit drift)
#   update_velocity_dt=s   ->  x += s * update.velocity (Verlet correction)
pos_spec = PositionUpdateSpec(derivative_dt=0.5 * dt)

# The constructors below are the preferred spelling:
explicit_step(dt)                                    # x += dt * k
semi_implicit_position_step(dt)                      # x += dt * v_current
verlet_position_step(dt, update_velocity_dt=dt**2/2) # x += dt*k.x + (dt^2/2)*k.v
```

### Integration Functions

```python
from integrators import (
    IntegrationSchemeType,
    getIntegrator,
    IntegrationResult,
    StageResult,
)

# Get an integrator
integrator = getIntegrator(IntegrationSchemeType.rungeKutta4)

# Call it
result: IntegrationResult = integrator.function(
    state,           # Current system state
    dt=0.01,         # Time step
    f=rhs_function,  # RHS function: (system, dt, **kwargs) -> (update, aux)
    verbose=False,   # Enable verbose output
    **kwargs         # Additional args passed to RHS
)

# Access results
next_state = result.state
stages = result.stages  # List of StageResult namedtuples
```

### Helper Functions

```python
from integrators import (
    get_reference_state,        # Extract state from system
    get_tagged_attr,            # Access fields by tag
    update_position,            # Apply position update
    update_component,           # Apply component update
)

# Extract state from system
state = get_reference_state(system)

# Get field by tag
position = get_tagged_attr(state, tag='position')

# Apply typed updates
system = update_position(system, update, spec, 
                        'position', 'position_derivative',
                        'velocity', 'velocity_derivative')
```

## Advanced Usage

### Multi-Stage Systems

For systems with multiple coupled components, use multiple field types:

```python
@dataclass
class ComplexState(BaseState):
    # Position-velocity pair
    position: torch.Tensor = integrated('dp_dt', tags=('position',))
    velocity: torch.Tensor = integrated('dv_dt', tags=('velocity',))
    
    # Additional quantities
    energy: torch.Tensor = integrated('de_dt', tags=('energy',))
    temperature: torch.Tensor = constant(tags=('temperature',))
    
    # Metadata
    particle_id: int = copied()
```

### Custom Update Logic

`custom()` marks a field as one the library will not touch: it is neither integrated,
cloned, nor nulled by the generic paths, and your system is responsible for it entirely.
There is no `apply_fn` hook — the field is handled by your own `apply_*_update` methods
and lifecycle hooks.

```python
@dataclass
class MyState(BaseState):
    custom_field: Any = custom(default=None)

@dataclass
class MySystem(BaseIntegrationSystem):
    state: MyState = reference_state()
    t: float = 0.0

    def initializeNewState(self, *args, **kwargs):
        fresh = get_reference_state(self).initializeNewState()
        fresh.custom_field = my_own_rule(get_reference_state(self).custom_field)
        return MySystem(state=fresh, t=self.t)
```

If what you need is "recomputed every stage, and the final state should keep the last
stage's value" — summation density, pressure, a smoothing length — use `copied()`
instead. The integrator carries those over for you when the step is finalized.

### Passing Extra Arguments to RHS

```python
def advanced_rhs(system, dt, config: dict, verbose: bool = False):
    """RHS with extra parameters."""
    state = get_reference_state(system)
    # Use config['param'] as needed
    return update, aux

# Call with extra kwargs
result = integrator.function(
    system,
    dt=0.01,
    f=advanced_rhs,
    config={'param': 42},
    verbose=True
)
```

## Extending the Library

### Adding a New Integration Scheme

If the scheme has a Butcher tableau, that is the whole job — add it to
`getButcherTableau` and register `butcherScheme('yourTableau')`. The stage loop, the
embedded-pair handling, and the reuse analysis all fall out of the tableau:

```python
# in butcher.py
elif scheme == 'myTableau':
    return butcherTableau(a=..., b=..., c=...)   # b may be a (main, embedded) tuple

myScheme = butcherScheme('myTableau')

# in enums.py, then integration.py
IntegrationSchemes.append(IntegrationScheme(myScheme, 'My Scheme',
                                            IntegrationSchemeType.myScheme, order,
                                            dissipation, nonLagrangian))
```

`reuse_order` and `fsal` are derived automatically from the tableau; do not set them by
hand. Running `pytest` then holds the new scheme to its declared order on three problems,
checks its reuse order against the prediction, and checks its `dissipation` flag against
its actual energy behaviour.

For a hand-rolled scheme with no tableau:

1. Implement the scheme function in a new module or existing one
2. `kwargs.pop('priorStep')` — either reuse it or pass it to `reject_prior_step`, but
   never let it flow through into `f`
3. Call `initializeSystem` once, and set `.t` explicitly on every stage buffer
4. Evaluate `f` on a fresh `initializeNewState()` buffer, never on the caller's state
5. Return `IntegrationResult(state=..., stages=[StageResult(...), ...])`
6. Register in the `IntegrationSchemeType` enum and record its reuse behaviour in
   `HANDROLLED_REUSE` in [reuse.py](src/integrators/reuse.py)

Example template:

```python
def myCustomScheme(state, dt, f, *args, **kwargs):
    """Custom O(h^p) integrator."""
    # Compute stages
    k1, aux1 = f(state, dt, *args, **kwargs)
    # ... more stages ...
    
    # Apply updates
    new_state = apply_state_update(state, ...)
    
    # Return consistent result
    return IntegrationResult(
        state=new_state,
        stages=[
            StageResult(aux=aux1, update=k1),
            # ... more stages
        ]
    )
```

### Implementing Custom State Behaviors

There are exactly four dispatch points — `apply_position_update`, `apply_velocity_update`,
`apply_quantity_update`, `apply_state_update`. There is no dispatch on arbitrary tag
names, so an extra field is routed through one of those four (`apply_quantity_update`
is the usual home) rather than through a method named after its tag:

```python
@dataclass
class MyState(BaseState):
    energy:  torch.Tensor = integrated('dedt', tags=('quantity',))
    entropy: torch.Tensor = integrated('dsdt', tags=('entropy',))

class MySystem(BaseIntegrationSystem):
    def apply_quantity_update(self, update, spec, **kwargs):
        update_component(self, update, spec, 'quantity', 'quantity_derivative')
        update_component(self, update, spec, 'entropy', 'entropy_derivative')
        return self
```

Tags select *which field* `update_component` reads and writes; they do not select which
method runs.

## Performance Notes

- **Memory**: Named tuples have minimal overhead vs raw tuples
- **Gradients**: All operations preserve gradient flow; use `torch.no_grad()` if differentiation isn't needed
- **GPU**: Works transparently with GPU tensors (CUDA/MPS)
- **Benchmarking**: See [integrators.ipynb](integrators.ipynb) for timing comparisons

## Known Limitations

- Implicit methods not yet implemented (only explicit schemes)
- Adaptive step-size control not built-in (use external error estimators)
- No multistep methods (BDF, Adams) yet

The last two are scoped and costed in [NOTES.md §3](NOTES.md#3-multistep-and-implicit-methods),
including which schemes are worth adding and what each one costs.

## Contributing

Contributions welcome! Areas of interest:

- Implicit and multistep schemes
- Adaptive time stepping
- Better documentation and examples
- Performance optimizations
- Additional state field behaviors

## License

Apache 2.0 — See [license.md](license.md)

## Citation

If you use this library in research, please cite:

```bibtex
@software{integrators2024,
  title={Differentiable ODE Integration with PyTorch},
  author={Winchenbach, Rene},
  url={https://github.com/wi-re/integrators},
  year={2024}
}
```

## References

- Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving ordinary differential equations I: Nonstiff problems*
- Ruth, R. D. (1983). A canonical integration technique. IEEE Trans. Nucl. Sci., 24(2), 2669–2671
- Verlet, L. (1967). Computer "experiments" on classical fluids. I. Thermodynamical properties by a molecular dynamics method. Phys. Rev., 159(1), 98
