"""State cloning, field behaviours, stage times, and registry lookup.

These cover the defects in NOTES.md 2.3, 2.5, 2.6, 2.13 and 2.15 -- all of them silent
in the sense that they produce a wrong answer or a crash far from their cause.
"""

from dataclasses import dataclass

import pytest
import torch

from warpSPHIntegrators import (
    BaseState,
    IntegrationSchemeType,
    StateBlend,
    constant,
    ephemeral,
    getIntegrationEnum,
    getIntegrator,
    integrated,
    testing,
)
from warpSPHIntegrators.integration import IntegrationSchemes


@dataclass
class MixedState(BaseState):
    """A state with the field kinds a real system carries, including non-tensors."""

    x: torch.Tensor = integrated('dxdt', tags=('position',))
    rho0: float = constant(default=1000.0)
    label: str = constant(default='fluid')
    scratch: torch.Tensor = ephemeral(default=None)
    untagged: torch.Tensor = None   # deliberately carries no behavior metadata


def _mixed():
    return MixedState(x=torch.arange(3, dtype=torch.float64), scratch=torch.ones(3))


# --------------------------------------------------------------------------- #
# NOTES 2.5 -- nograd() on a state with non-tensor fields                      #
# --------------------------------------------------------------------------- #

def test_nograd_survives_non_tensor_fields():
    """`_op` used to parse as `A if detach else (B if isinstance(...) else C)`.

    The isinstance guard therefore protected only the non-detach branch, so
    `nograd()` called `.detach()` on floats and strings. Every realistic SPH state
    carries scalar parameters, which made the method unusable.
    """
    state = _mixed()
    detached = state.nograd()
    assert detached.rho0 == 1000.0
    assert detached.label == 'fluid'
    assert torch.equal(detached.x, state.x)
    assert not detached.x.requires_grad


def test_nograd_detaches_from_the_graph():
    x = torch.arange(3, dtype=torch.float64, requires_grad=True)
    state = MixedState(x=x, scratch=None)
    assert state.clone().x.requires_grad
    assert not state.nograd().x.requires_grad


# --------------------------------------------------------------------------- #
# NOTES 2.6 -- clone() and initializeNewState() disagreed on untagged fields   #
# --------------------------------------------------------------------------- #

def test_untagged_fields_survive_both_clone_paths():
    """A missing `constant(...)` used to turn a tensor into None mid-step."""
    state = _mixed()
    state.untagged = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

    assert state.clone().untagged is not None
    assert state.initializeNewState().untagged is not None
    assert torch.equal(state.clone().untagged, state.initializeNewState().untagged)


def test_ephemeral_is_dropped_and_integrated_is_carried():
    state = _mixed()
    fresh = state.initializeNewState()
    assert fresh.scratch is None
    assert torch.equal(fresh.x, state.x)
    assert fresh.x is not state.x, 'integrated fields must be cloned, not aliased'


def test_clone_keeps_ephemeral_but_still_copies():
    state = _mixed()
    cloned = state.clone()
    assert cloned.scratch is not None
    assert cloned.scratch is not state.scratch


# --------------------------------------------------------------------------- #
# NOTES 2.3 -- stage times                                                     #
# --------------------------------------------------------------------------- #

def _stage_times(scheme, dt, steps, t0=0.0):
    """Times at which the scheme evaluates f, per step."""
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    system.t = t0
    per_step = []

    def recording_rhs(state, step_dt, **kwargs):
        per_step[-1].append(float(state.t))
        return prob.rhs(state, step_dt, **kwargs)

    for _ in range(steps):
        per_step.append([])
        system = scheme(system, dt=dt, f=recording_rhs).state
    return per_step


#: Schemes whose first evaluation legitimately does not sit at t^n. PEFRL opens with a
#: position drift *before* its first force evaluation. The DIRK schemes with a nonzero
#: diagonal on their first stage (`tableau.a[0, 0] != 0`, i.e. `tableau.c[0] != 0`) are
#: implicit *at* that first stage, so it is solved at t^n + c[0]*dt, not at t^n -- true
#: by construction for backward Euler (c[0]=1), implicit midpoint (c[0]=1/2) and SDIRK2
#: (c[0]=gamma); trapezoidal's first stage has a[0,0]=0 (explicit) so it is exempt from
#: this set, same as every explicit RK scheme.
DRIFT_BEFORE_FIRST_EVALUATION = {
    'PEFRL', 'Backward Euler (implicit)', 'Implicit Midpoint', 'SDIRK2',
}


def test_stage_times_shift_by_dt_between_steps(scheme):
    """The whole stage-time pattern must translate by dt each step.

    This is the invariant that `RungeKuttaB` broke: it set `currentState.t` for stages
    1..s-1 but never for stage 0, so stage 0 inherited whatever the user's
    `initializeNewState` returned -- t=0 in the README's own example -- and stayed
    pinned there forever while the other stages marched on.
    """
    dt = 0.1
    times = _stage_times(scheme, dt, steps=3, t0=1.0)
    first = times[0]
    for step, stage_times in enumerate(times[1:], start=1):
        assert len(stage_times) == len(first), f'{scheme.name} changed stage count between steps'
        for i, (t_now, t_first) in enumerate(zip(stage_times, first)):
            assert t_now == pytest.approx(t_first + step * dt), (
                f'{scheme.name} step {step}, stage {i}: evaluated at t={t_now}, expected '
                f'{t_first + step * dt}. Stage times: step 0 {first}, step {step} {stage_times}.'
            )


def test_first_evaluation_is_at_the_start_of_the_step(scheme):
    if scheme.name in DRIFT_BEFORE_FIRST_EVALUATION:
        pytest.skip(f'{scheme.name} drifts before its first evaluation')
    dt = 0.1
    for step, stage_times in enumerate(_stage_times(scheme, dt, steps=3, t0=1.0)):
        assert stage_times[0] == pytest.approx(1.0 + step * dt), (
            f'{scheme.name} step {step}: first evaluation at t={stage_times[0]}, '
            f'expected {1.0 + step * dt}. Stage times were {stage_times}.'
        )


def test_stage_times_stay_within_the_step(scheme):
    """No stage may be evaluated outside [t^n, t^{n+1}] beyond the tableau's own nodes.

    PEFRL and VEFRL have coefficients outside [0, 1] by construction, so they are
    allowed a wider window; everything else must stay inside the step.
    """
    dt = 0.1
    slack = 1.5 if scheme.name in ('PEFRL', 'VEFRL', 'RK3', 'RK4 (alternative)') else 0.0
    for step, stage_times in enumerate(_stage_times(scheme, dt, steps=2, t0=1.0)):
        lo, hi = 1.0 + step * dt, 1.0 + (step + 1) * dt
        for t in stage_times:
            assert lo - slack * dt - 1e-12 <= t <= hi + slack * dt + 1e-12, (
                f'{scheme.name} evaluated f at t={t}, outside [{lo}, {hi}]'
            )


def test_final_time_advances_by_exactly_dt(scheme):
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    system.t = 0.75
    for step in range(4):
        system = scheme(system, dt=0.1, f=prob.rhs).state
        assert system.t == pytest.approx(0.75 + 0.1 * (step + 1))


def test_time_stays_a_python_float(scheme):
    """Tableau nodes are numpy scalars; `t` used to silently become np.float64."""
    prob = testing.PROBLEMS['oscillator']()
    result = scheme(prob.initial(), dt=0.1, f=prob.rhs)
    assert type(result.state.t) is float, f'{scheme.name} produced t of type {type(result.state.t)}'


# --------------------------------------------------------------------------- #
# NOTES 2.15 -- who owns `t`                                                   #
# --------------------------------------------------------------------------- #

def test_update_helpers_do_not_advance_time():
    """updateStateEuler never advanced `t`; updateStateSemiImplicitEuler used to.

    Composing the two therefore double-advanced time. Time belongs to the integrator.
    """
    from warpSPHIntegrators import updateStateEuler, updateStateSemiImplicitEuler

    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    system.t = 2.0
    update, _ = prob.rhs(system, 0.1)

    assert updateStateEuler(system, update, 0.1).t == pytest.approx(2.0)
    assert updateStateSemiImplicitEuler(system, update, 0.1).t == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# NOTES 2.13 -- registry metadata and lookup                                   #
# --------------------------------------------------------------------------- #

def test_getIntegrator_accepts_all_three_spellings():
    """`getIntegrator('rungeKutta4')` used to raise: it compared an enum to a string."""
    by_name = getIntegrator('RK4')
    assert getIntegrator(IntegrationSchemeType.rungeKutta4) is by_name
    assert getIntegrator('rungeKutta4') is by_name


def test_getIntegrationEnum_matches_getIntegrator():
    for s in IntegrationSchemes:
        assert getIntegrationEnum(s.name) is s.identifier
        assert getIntegrationEnum(s.identifier.name) is s.identifier


def test_unknown_integrator_raises():
    with pytest.raises(ValueError):
        getIntegrator('no such scheme')


def test_semi_implicit_euler_is_registered_as_first_order():
    """It was registered as order 2 and measures 1.0. It is one force evaluation."""
    assert getIntegrator('Semi-Implicit Euler').order == 1


def test_registry_identifiers_are_unique():
    identifiers = [s.identifier for s in IntegrationSchemes]
    assert len(identifiers) == len(set(identifiers))
    names = [s.name for s in IntegrationSchemes]
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------- #
# NOTES 2.16 -- StateBlend validation                                          #
# --------------------------------------------------------------------------- #

def test_state_blend_rejects_a_reference_without_a_weight():
    """Otherwise it fails much later, as `None * tensor` inside update_component."""
    with pytest.raises(ValueError, match='reference_state and reference_weight'):
        StateBlend(reference_state=object(), reference_weight=None)
    with pytest.raises(ValueError, match='reference_state and reference_weight'):
        StateBlend(reference_state=None, reference_weight=0.5)
    StateBlend(self_scale=0.5)  # fine


# --------------------------------------------------------------------------- #
# NOTES 2.12 -- schemes must not scribble on the caller's state                #
# --------------------------------------------------------------------------- #

def test_scheme_does_not_mutate_the_caller_state(scheme):
    """`preprocess` writes scratch into whatever state it is handed."""
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    before_x = system.state.x.clone()
    before_u = system.state.u.clone()
    before_t = system.t

    scheme(system, dt=0.1, f=prob.rhs)

    assert torch.equal(system.state.x, before_x), f'{scheme.name} mutated the caller position'
    assert torch.equal(system.state.u, before_u), f'{scheme.name} mutated the caller velocity'
    assert system.t == before_t, f'{scheme.name} mutated the caller time'
