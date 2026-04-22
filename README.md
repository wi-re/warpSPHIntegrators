# Integrators — Differentiable ODE Integration with PyTorch

A flexible, fully differentiable numerical ODE integration library for PyTorch. Implements multiple integration schemes with support for complex state management, custom field behavior, and both typed and legacy APIs.

## Overview

This library provides a set of time integration schemes for solving systems of ordinary differential equations (ODEs) where states may be complex objects with multiple field types and integration behaviors. All computations are differentiable through PyTorch, making the library suitable for physics-informed machine learning, neural ODEs, and scientific computing.

### Key Features

- **Multiple Integration Schemes**: Runge-Kutta (2–4th order), TVD-RK3, symplectic Verlet, Ruth-Forest high-order, and Euler methods
- **Flexible State Management**: Custom state objects with metadata-driven field behavior (integrated, constant, copied, ephemeral, custom)
- **Type-Safe Protocol**: Structural typing for integration systems with clear separation of concerns
- **Fully Differentiable**: All operations preserve gradient flow for end-to-end learning
- **Extensible**: Easy to add new schemes or state types

## Installation

```bash
pip install diffSPH_integrators
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
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
- `integrated('derivative_field', ...)` — Evolves using the provided derivative field
- `constant(...)` — Remains fixed throughout integration  
- `copied(...)` — Automatically copied to new state instances
- `ephemeral(...)` — Created fresh each step (not carried forward)
- `custom(...)` — Custom update logic via user functions

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

### Explicit Runge-Kutta Methods

| Scheme | Order | Error | Use Case |
|--------|-------|-------|----------|
| **Forward Euler** | 1 | O(h²) | Quick, low-accuracy tests; baseline |
| **RK2 (Heun)** | 2 | O(h³) | Moderate accuracy with 2 function evals |
| **RK4 (Classic)** | 4 | O(h⁵) | High accuracy; most common choice |
| **SSP-RK3** | 3 | TVD | Conservation laws; non-oscillatory |

### Symplectic Methods

| Scheme | Type | Use Case |
|--------|------|----------|
| **Symplectic Euler** | Order 1 | Fast, symplectic for Hamiltonian systems |
| **Velocity Verlet** | Order 2 | Standard Verlet; good energy conservation |
| **Leap-Frog** | Order 2 | Alternative symplectic scheme |

### High-Order Hamiltonian Methods

| Scheme | Order | Use Case |
|--------|-------|----------|
| **PEFRL** | 4 | High-order, efficient Hamiltonian integrator |
| **VEFRL** | 4 | Variant with different coefficient pattern |

### TVD and Conservative Schemes

| Scheme | Type | Use Case |
|--------|------|----------|
| **TVD-RK2** | 2 | Conservation laws with TVD property |
| **TVD-RK3** | 3 | Higher-order TVD for PDEs |

## Example: Damped Harmonic Oscillator with Feedback

See [integrators.ipynb](integrators.ipynb) for the complete interactive example. This example demonstrates:

- Custom state with multiple field types
- Energy-dependent feedback (spring constant varies with stored energy)
- Comparison of three integration schemes (RK4, RK2, Forward Euler)
- Visualization of phase plot, energy evolution, and trajectories

**Example Output:**

![Integration schemes comparison: Phase plot and energy evolution](example.png)

The visualization shows:
- **Phase Plot (top-left)**: How position and velocity evolve together
- **Energy (top-right & bottom-right)**: Kinetic, potential, and total energy over time
- **Position/Velocity Trajectories**: Individual component evolution for each scheme

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
    
    # Copied: automatically copied to new states
    name: str = copied()
    
    # Ephemeral: created fresh each step
    temp: float = ephemeral()
    
    # Custom: user-defined update logic
    custom_field: Any = custom()
```

### Update Specifications

```python
from integrators import PositionUpdateSpec, ComponentUpdateSpec, StateBlend

# Position update: can include velocity drift and derivative step
pos_spec = PositionUpdateSpec(
    derivative_dt=0.5,  # Fraction of dt for the k-value derivative term
    blend=StateBlend.IMPLICIT,  # How to blend position and velocity updates
)

# Component (velocity, quantity, etc.) update
comp_spec = ComponentUpdateSpec(
    derivative_dt=0.5,
    blend=StateBlend.IMPLICIT,
)
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

For cases where standard field behaviors don't suffice:

```python
def custom_apply(system, update, **kwargs):
    """Custom logic that manipulates the system in non-standard ways."""
    # Implement domain-specific logic here
    return system

# Then declare in state:
custom_field: Any = custom(apply_fn=custom_apply)
```

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

1. Implement the scheme function in a new module or existing one
2. Add to the `IntegrationSchemes` list in [integration.py](src/integrators/integration.py)
3. Return `IntegrationResult(state=..., stages=[StageResult(...), ...])`
4. Register in the `IntegrationSchemeType` enum

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

Define new field behavior tags in your state and implement corresponding update methods in your system:

```python
@dataclass
class MyState(BaseState):
    my_field: torch.Tensor = custom(tags=('my_behavior',))

class MySystem(BaseIntegrationSystem):
    def apply_my_behavior_update(self, update, spec, **kwargs):
        """Handle 'my_behavior' field updates."""
        # Custom logic here
        return self
```

## Performance Notes

- **Memory**: Named tuples have minimal overhead vs raw tuples
- **Gradients**: All operations preserve gradient flow; use `torch.no_grad()` if differentiation isn't needed
- **GPU**: Works transparently with GPU tensors (CUDA/MPS)
- **Benchmarking**: See [integrators.ipynb](integrators.ipynb) for timing comparisons

## Known Limitations

- Implicit methods not yet implemented (only explicit schemes)
- Adaptive step-size control not built-in (use external error estimators)
- No multistep methods (BDF, Adams) yet

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
