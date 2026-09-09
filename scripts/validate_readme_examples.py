"""Execute every runnable code example in README.md, in the `warp` environment.

Roadmap Phase 0 validation gate: "README examples execute in the `warp`
environment." Each check below mirrors one README code block as written (the
quick-start spring oscillator is reused as `system`/`rhs` by the examples that
build on it); examples that reference caller-defined names the README does not
define itself (e.g. the IMEX `transport_rhs`/`diffusion_rhs`) get the smallest
definition that makes the shown call shape work.

Run:  conda run -n warp python scripts/validate_readme_examples.py
Exits nonzero if any example fails.
"""

from __future__ import annotations

import dataclasses
import sys
import traceback

import torch

import warpSPHIntegrators
from warpSPHIntegrators import (
    BaseIntegrationSystem,
    BaseState,
    ComponentUpdateSpec,
    IMEXRHS,
    IntegrationResult,
    IntegrationSchemeType,
    PositionUpdateSpec,
    StageResult,
    StepHistory,
    FixedPointSolver,
    blend_state,
    constant,
    copied,
    custom,
    ephemeral,
    explicit_step,
    getIntegrator,
    get_reference_state,
    get_tagged_attr,
    integrated,
    reference_state,
    register_clone_handler,
    semi_implicit_position_step,
    state_difference,
    state_norm,
    supports_step_reuse,
    step_reuse_order,
    tagged,
    update_component,
    update_position,
    verlet_position_step,
)


# --------------------------------------------------------------------------- #
# Quick start (README sections 1-5): the spring oscillator, reused below      #
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class MyState(BaseState):
    position: torch.Tensor = integrated('dx_dt', tags=('position',))
    velocity: torch.Tensor = integrated('dv_dt', tags=('velocity',))
    mass: torch.Tensor = constant(tags=('mass',))
    spring_constant: torch.Tensor = constant(tags=('spring_constant',))


@dataclasses.dataclass
class MyUpdate:
    dx_dt: torch.Tensor = tagged(tags=('position_derivative',))
    dv_dt: torch.Tensor = tagged(tags=('velocity_derivative',))


@dataclasses.dataclass
class MySystem(BaseIntegrationSystem):
    state: MyState = reference_state(tags=('my_state',))
    t: float = 0.0

    def initializeNewState(self, *args, **kwargs):
        current_state = get_reference_state(self)
        return MySystem(state=current_state.initializeNewState(), t=self.t)

    def apply_position_update(self, update: MyUpdate, spec: PositionUpdateSpec, **kwargs):
        return update_position(self, update, spec,
                               'position', 'position_derivative',
                               'velocity', 'velocity_derivative')

    def apply_velocity_update(self, update: MyUpdate, spec: ComponentUpdateSpec, **kwargs):
        return update_component(self, update, spec, 'velocity', 'velocity_derivative')

    def apply_state_update(self, update: MyUpdate, spec: ComponentUpdateSpec, **kwargs):
        position_spec = PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend)
        self.apply_position_update(update, position_spec, **kwargs)
        self.apply_velocity_update(update, spec, **kwargs)
        return self


def my_rhs(system: MySystem, dt: float, verbose: bool = False) -> tuple:
    state = get_reference_state(system)
    dx_dt = state.velocity
    dv_dt = -(state.spring_constant / state.mass) * state.position
    kinetic_energy = 0.5 * state.mass * state.velocity**2
    potential_energy = 0.5 * state.spring_constant * state.position**2
    return MyUpdate(dx_dt=dx_dt, dv_dt=dv_dt), (kinetic_energy, potential_energy)


def make_system():
    initial_state = MyState(
        position=torch.tensor([1.0]),
        velocity=torch.tensor([0.0]),
        mass=torch.tensor([1.0]),
        spring_constant=torch.tensor([5.0]),
    )
    return MySystem(state=initial_state, t=0.0)


def example_quick_start():
    integrator = getIntegrator(IntegrationSchemeType.rungeKutta4)
    system = make_system()
    dt = 0.01
    result = integrator.function(system, dt=dt, f=my_rhs, verbose=False)
    next_system = result.state
    last_stage = result.stages[-1]
    aux_values = last_stage.aux
    assert isinstance(result, IntegrationResult)
    assert isinstance(next_system, MySystem)
    assert isinstance(last_stage, StageResult)
    # split_return wraps aux as rv[1:], so the (ke, pe) pair sits at index 0
    energies = aux_values[0]
    assert len(energies) == 2
    assert torch.isfinite(energies[0]).all() and torch.isfinite(energies[1]).all()
    assert torch.isfinite(get_reference_state(next_system).position).all()


def example_register_clone_handler():
    class MyBuffer:
        def __init__(self, value):
            self.value = value

        def copy(self):
            return MyBuffer(self.value.clone() if torch.is_tensor(self.value) else self.value)

    @dataclasses.dataclass
    class BufferState(BaseState):
        buffer: MyBuffer = custom(default=None)

    state = BufferState(buffer=MyBuffer(torch.tensor([1.0, 2.0])))
    register_clone_handler(
        'MyBuffer',
        matches=lambda v: isinstance(v, MyBuffer),
        clone=lambda v, detach: v.copy(),
        to_device=lambda v, device: v,
    )
    # custom fields are nulled by initializeNewState (README field-behavior
    # table); the handler is exercised on the clone() path instead.
    fresh = state.initializeNewState()
    assert fresh.buffer is None
    cloned = state.clone()
    assert cloned.buffer is not state.buffer
    assert torch.equal(cloned.buffer.value, state.buffer.value)


def example_newmark():
    # Newmark hard-codes x/u state fields (structural-dynamics convention,
    # matching the testing.py harness), so this example uses an x/u state
    # rather than the quick-start MyState.
    @dataclasses.dataclass
    class NMState(BaseState):
        x: torch.Tensor = integrated('dxdt', tags=('position',))
        u: torch.Tensor = integrated('dudt', tags=('velocity',))
        m: torch.Tensor = constant(tags=('mass',))
        k: torch.Tensor = constant(tags=('spring_constant',))

    @dataclasses.dataclass
    class NMUpdate:
        dxdt: torch.Tensor = tagged(tags=('position_derivative',))
        dudt: torch.Tensor = tagged(tags=('velocity_derivative',))

    @dataclasses.dataclass
    class NMSystem(BaseIntegrationSystem):
        state: NMState = reference_state(tags=('nm_state',))
        t: float = 0.0

        def initializeNewState(self, *args, **kwargs):
            return NMSystem(state=get_reference_state(self).initializeNewState(), t=self.t)

        def apply_position_update(self, update, spec, **kwargs):
            return update_position(self, update, spec, 'position', 'position_derivative',
                                   'velocity', 'velocity_derivative')

        def apply_velocity_update(self, update, spec, **kwargs):
            return update_component(self, update, spec, 'velocity', 'velocity_derivative')

        def apply_quantity_update(self, update, spec, **kwargs):
            return self

        def apply_state_update(self, update, spec, **kwargs):
            self.apply_position_update(
                update, PositionUpdateSpec(derivative_dt=spec.derivative_dt, blend=spec.blend),
                **kwargs)
            self.apply_velocity_update(update, spec, **kwargs)
            return self

    def nm_rhs(system, dt, **kwargs):
        state = get_reference_state(system)
        return NMUpdate(dxdt=state.u, dudt=-(state.k / state.m) * state.x), None

    scheme = getIntegrator('Newmark')
    initial = NMSystem(state=NMState(x=torch.tensor([1.0]), u=torch.tensor([0.0]),
                                     m=torch.tensor([1.0]), k=torch.tensor([5.0])))
    result = scheme(initial, dt=0.01, f=nm_rhs, beta=0.25, gamma=0.5)
    assert torch.isfinite(get_reference_state(result.state).x).all()


def example_implicit_midpoint_picard_override():
    scheme = getIntegrator('Implicit Midpoint')
    result = scheme(make_system(), dt=0.01, f=my_rhs, solver=FixedPointSolver(iterations=16))
    assert torch.isfinite(get_reference_state(result.state).position).all()


def example_adams_bashforth_4():
    scheme = getIntegrator('Adams-Bashforth 4')
    history = StepHistory(maxlen=3)
    system = make_system()
    dt = 0.01
    for _ in range(6):
        result = scheme(system, dt=dt, f=my_rhs, history=history)
        system, history = result.state, result.history
    assert torch.isfinite(get_reference_state(system).position).all()


def example_bdf3():
    scheme = getIntegrator('BDF3')
    history = StepHistory(maxlen=2)
    system = make_system()
    dt = 0.01
    for _ in range(4):
        result = scheme(system, dt=dt, f=my_rhs, history=history)
        system, history = result.state, result.history
    assert torch.isfinite(get_reference_state(system).position).all()


def example_imex_euler():
    # The README shows `IMEXRHS(explicit=transport_rhs, implicit=diffusion_rhs)`
    # with caller-defined callbacks; the smallest pair that exercises the shown
    # call shape is transport (explicit dx/dt = v) plus a stiff decay (implicit
    # dv/dt = -rate * v), which runs the JFNK closure on the implicit side.
    def transport_rhs(system, dt, **kwargs):
        state = get_reference_state(system)
        return MyUpdate(dx_dt=state.velocity, dv_dt=torch.zeros_like(state.velocity)), None

    def diffusion_rhs(system, dt, **kwargs):
        state = get_reference_state(system)
        return MyUpdate(dx_dt=torch.zeros_like(state.velocity),
                        dv_dt=-100.0 * state.velocity), None

    scheme = getIntegrator('IMEX Euler')
    rhs = IMEXRHS(explicit=transport_rhs, implicit=diffusion_rhs)
    result = scheme(make_system(), dt=0.01, f=rhs)
    assert torch.isfinite(get_reference_state(result.state).position).all()


def example_first_stage_reuse():
    scheme = getIntegrator('Dormand-Prince 5(4)')
    assert supports_step_reuse(scheme) is True
    assert step_reuse_order(scheme) == 5

    no_reuse = getIntegrator('SSP RK3')
    assert supports_step_reuse(no_reuse) is False
    assert step_reuse_order(no_reuse) == 1

    system = make_system()
    result = scheme(system, dt=0.01, f=my_rhs)
    error = result.error  # embedded pairs only: y_high - y_low, as a state
    assert error is not None
    reused = scheme(result.state, dt=0.01, f=my_rhs, priorStep=result.stages[-1])
    assert torch.isfinite(get_reference_state(reused.state).position).all()


def example_state_definition():
    from typing import Any

    @dataclasses.dataclass
    class ApiState(BaseState):
        x: torch.Tensor = integrated('dxdt', tags=('position',))
        m: torch.Tensor = constant(tags=('mass',))
        density: torch.Tensor = copied(default=None)
        neighbours: Any = ephemeral(default=None)
        custom_field: Any = custom(default=None)

    state = ApiState(x=torch.tensor([1.0]), m=torch.tensor([1.0]))
    # a bare state has no reference_state role; get_reference_state needs a system
    assert state.x is not None
    assert state.density is None


def example_update_specifications():
    y_1 = make_system().state.initializeNewState()
    blend = blend_state(self_scale=3 / 4, reference_state=y_1, reference_weight=1 / 4)
    comp_spec = ComponentUpdateSpec(derivative_dt=0.5 * 0.01, blend=blend)
    pos_spec = PositionUpdateSpec(derivative_dt=0.5 * 0.01)
    assert comp_spec.blend.self_scale == 3 / 4
    assert pos_spec.derivative_dt == 0.5 * 0.01
    assert explicit_step(0.01).derivative_dt == 0.01
    assert semi_implicit_position_step(0.01).current_velocity_dt == 0.01
    assert verlet_position_step(0.01, update_velocity_dt=0.01**2 / 2).update_velocity_dt == 0.01**2 / 2


def example_integration_functions():
    integrator = getIntegrator(IntegrationSchemeType.rungeKutta4)
    system = make_system()
    result: IntegrationResult = integrator.function(system, dt=0.01, f=my_rhs, verbose=False)
    next_state = result.state
    stages = result.stages
    assert next_state is not None and len(stages) == 4


def example_helper_functions():
    system = make_system()
    state = get_reference_state(system)
    position = get_tagged_attr(state, tag='position')
    assert torch.equal(position, state.position)
    update = MyUpdate(dx_dt=torch.zeros_like(state.position),
                      dv_dt=torch.zeros_like(state.velocity))
    clone = get_reference_state(system.initializeNewState())
    clone_system = MySystem(state=clone, t=system.t)
    clone_system = update_position(clone_system, update, PositionUpdateSpec(derivative_dt=0.01),
                                   'position', 'position_derivative',
                                   'velocity', 'velocity_derivative')
    clone_system = update_component(clone_system, update, explicit_step(0.01),
                                    'velocity', 'velocity_derivative')
    diff = state_difference(clone_system.state, system.state)
    assert state_norm(diff, rtol=1e-3, atol=1e-6) >= 0.0


def example_passing_extra_arguments():
    def advanced_rhs(system, dt, config: dict, verbose: bool = False):
        state = get_reference_state(system)
        update = MyUpdate(dx_dt=state.velocity,
                          dv_dt=-config['param'] * state.position)
        return update, None

    integrator = getIntegrator(IntegrationSchemeType.rungeKutta4)
    result = integrator.function(make_system(), dt=0.01, f=advanced_rhs,
                                 config={'param': 42})
    assert torch.isfinite(get_reference_state(result.state).position).all()


CHECKS = [
    ('quick start (RK4 step + result access)', example_quick_start),
    ('register_clone_handler', example_register_clone_handler),
    ('Newmark example', example_newmark),
    ('Implicit Midpoint Picard override', example_implicit_midpoint_picard_override),
    ('Adams-Bashforth 4 with StepHistory', example_adams_bashforth_4),
    ('BDF3 with StepHistory', example_bdf3),
    ('IMEX Euler with IMEXRHS split', example_imex_euler),
    ('first-stage reuse queries + priorStep run', example_first_stage_reuse),
    ('API: state definition', example_state_definition),
    ('API: update specifications', example_update_specifications),
    ('API: integration functions', example_integration_functions),
    ('API: helper functions', example_helper_functions),
    ('advanced: passing extra arguments to RHS', example_passing_extra_arguments),
]


def main() -> int:
    failures = []
    for name, check in CHECKS:
        try:
            check()
        except Exception:
            failures.append(name)
            print(f'FAIL: {name}')
            traceback.print_exc()
        else:
            print(f'ok:   {name}')
    print(f'\n{len(CHECKS) - len(failures)}/{len(CHECKS)} README examples executed '
          f'(warpSPHIntegrators {warpSPHIntegrators.__version__})')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
