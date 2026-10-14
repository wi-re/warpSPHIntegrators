# warpSPHIntegrators

Solver-agnostic, differentiable ODE time integrators for PyTorch, designed for
systems with several physical fields and auxiliary buffers, including SPH.

[catalog:summary:start]: #

**137 registered schemes across 11 families**, with orders through 14 and 45 estimate emitters.

[catalog:summary:end]: #

## Install

Python 3.11 or newer:

```bash
python -m pip install warpSPHIntegrators
# For the current checkout and test tools:
python -m pip install -e '.[test]'
```

The catalog includes explicit, implicit, geometric, multistep, IMEX, stabilized
and exponential methods. [The scheme catalog](docs/docs/catalog.md) and
[selection guide](docs/docs/selection.md) explain capabilities and assumptions.

## A complete example

This float64 CPU example implements the state/update protocol, integrates a
harmonic oscillator with RK4, and checks against the exact solution. It is also
available as [a standalone script](examples/harmonic_oscillator.py).

[example:include examples/harmonic_oscillator.py]: #
```python
"""Complete CPU example: python examples/harmonic_oscillator.py."""

from dataclasses import dataclass

import torch

from warpSPHIntegrators import (
    BaseIntegrationSystem, BaseState, ComponentUpdateSpec, PositionUpdateSpec,
    constant, getIntegrator, get_reference_state, integrated, reference_state,
    tagged, update_component, update_position,
)


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
    scheme = getIntegrator('RK4')
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

## Documentation and development

- [Documentation site](https://fluids.dev/warpSPHIntegrators/),
  [quick start](docs/docs/quickstart.md), [API](docs/docs/api.md) and
  [solver stack](docs/docs/solver.md).
- [Examples and notebooks](examples/README.md), including downstream setup.
- [Contributor instructions](CONTRIBUTING.md): environment, multicore tests,
  registration, coefficient provenance and release checks.
- [Current execution plan](REPOSITORY_PLAN.md) and [developer index](docs/development/README.md).
- [Numerical evidence](docs/research/integrators/notes-index.md),
  [method dispositions](docs/plans/method-dispositions.md) and [benchmarks](benchmarks/README.md).
- [Detailed protocol examples and galleries](docs/development/usage-reference.md).

Stepping is PyTorch based; native Warp buffers remain future work. Adaptive
helpers require a caller-owned acceptance loop, multistep history is fixed-step,
particle masking belongs to the user's system, and geometric methods retain
their force/separability assumptions. See [architecture](docs/development/architecture.md)
for the current boundaries.

## License and citation

[Apache 2.0](license.md). If used in research:

```bibtex
@software{warpSPHIntegrators2024,
  title={Differentiable ODE Integration with PyTorch},
  author={Winchenbach, Rene},
  url={https://github.com/wi-re/warpSPHIntegrators},
  year={2024}
}
```
