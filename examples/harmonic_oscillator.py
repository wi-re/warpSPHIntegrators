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
