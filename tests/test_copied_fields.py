"""`copied()` fields must arrive at the caller holding the last stage's value.

`copied` promises "not integrated; copied from last substep in finalize", and
`_state_finalize` implemented exactly that -- but it was never called anywhere in the
package, so `copied` behaved identically to `ephemeral`: nulled by
`initializeNewState` and never restored (NOTES.md 2.7).

This is the behaviour SPH needs for a quantity recomputed per stage rather than
integrated -- summation density, pressure, a smoothing length. `preprocess` writes it
into the stage buffer; the final state has to inherit it from the last stage.
"""

from dataclasses import dataclass

import pytest
import torch

from warpSPHIntegrators import (
    BaseIntegrationSystem,
    BaseState,
    ComponentUpdateSpec,
    PositionUpdateSpec,
    constant,
    copied,
    ephemeral,
    get_reference_state,
    integrated,
    reference_state,
    tagged,
)
from warpSPHIntegrators.fields import update_component, update_position


@dataclass
class DensityState(BaseState):
    x: torch.Tensor = integrated('dxdt', tags=('position',))
    u: torch.Tensor = integrated('dudt', tags=('velocity',))
    e: torch.Tensor = integrated('dedt', tags=('quantity',))
    m: torch.Tensor = constant(tags=('mass',))
    #: Recomputed by preprocess at every stage, never integrated.
    density: torch.Tensor = copied(default=None)
    #: The control: stage-local, must NOT survive the step.
    scratch: torch.Tensor = ephemeral(default=None)


@dataclass
class DensityUpdate:
    dxdt: torch.Tensor = tagged(tags=('position_derivative',))
    dudt: torch.Tensor = tagged(tags=('velocity_derivative',))
    dedt: torch.Tensor = tagged(tags=('quantity_derivative',))


@dataclass
class DensitySystem(BaseIntegrationSystem):
    state: DensityState = reference_state(tags=('particles',))
    t: float = 0.0
    stage_counter: list = None

    def initializeNewState(self, *args, **kwargs):
        return DensitySystem(state=get_reference_state(self).initializeNewState(),
                             t=self.t, stage_counter=self.stage_counter)

    def preprocess(self, initialState, dt, *args, **kwargs):
        """Stand-in for a summation-density evaluation: writes into the stage buffer."""
        self.stage_counter[0] += 1
        s = get_reference_state(self)
        s.density = torch.full_like(s.x, float(self.stage_counter[0]))
        s.scratch = torch.full_like(s.x, -float(self.stage_counter[0]))
        return self

    def apply_position_update(self, update, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec, 'position', 'position_derivative',
                               'velocity', 'velocity_derivative')

    def apply_velocity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')

    def apply_quantity_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'quantity', 'quantity_derivative')

    def apply_state_update(self, update, spec: ComponentUpdateSpec, **kwargs):
        self.apply_position_update(
            update, PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend), **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        self.apply_quantity_update(update, spec, **kwargs)
        return self


def _rhs(system, dt, **kwargs):
    s = get_reference_state(system)
    assert s.density is not None, 'preprocess should have filled density before f ran'
    return DensityUpdate(dxdt=s.u.clone(), dudt=-4.0 * s.x, dedt=torch.zeros_like(s.e)), None


def _system():
    one = torch.ones(1, dtype=torch.float64)
    return DensitySystem(
        state=DensityState(x=one.clone(), u=torch.zeros(1, dtype=torch.float64),
                           e=torch.zeros(1, dtype=torch.float64), m=one.clone()),
        t=0.0, stage_counter=[0])


def test_copied_field_survives_the_step(scheme):
    system = _system()
    result = scheme(system, dt=0.05, f=_rhs)
    final = get_reference_state(result.state)
    assert final.density is not None, (
        f'{scheme.name}: a copied() field arrived as None. It is behaving like ephemeral.'
    )


def test_copied_field_holds_the_last_stage_value(scheme):
    """Not just non-None: it must be the value from the *final* evaluation."""
    system = _system()
    result = scheme(system, dt=0.05, f=_rhs)
    stages_run = system.stage_counter[0]
    final = get_reference_state(result.state)
    assert float(final.density[0]) == pytest.approx(float(stages_run)), (
        f'{scheme.name}: density is from stage {float(final.density[0])} of {stages_run}'
    )


def test_ephemeral_field_does_not_survive(scheme):
    """The control. If this also survived, `copied` would carry no information."""
    result = scheme(_system(), dt=0.05, f=_rhs)
    assert get_reference_state(result.state).scratch is None, (
        f'{scheme.name}: an ephemeral() field leaked out of the step'
    )


def test_copied_field_is_carried_across_repeated_steps(scheme):
    system = _system()
    for _ in range(3):
        system = scheme(system, dt=0.05, f=_rhs).state
    assert get_reference_state(system).density is not None
