"""Stiff nonlinear regression cases for first-order state-space implicit methods."""

import math

import pytest
import torch

from warpSPHIntegrators import StepHistory, getIntegrator, get_reference_state, testing


STATE_SPACE_IMPLICIT = [
    'Backward Euler (implicit)',
    'Implicit Midpoint',
    'Trapezoidal (Crank-Nicolson)',
    'SDIRK2',
    'BDF1',
    'BDF2',
    'IMEX Euler',
]


def _stiff_relaxation_rhs(rate):
    """Nonlinear Prothero-Robinson form with exact solution x(t)=tanh(t)."""
    def rhs(system, dt, **kwargs):
        state = get_reference_state(system)
        time = float(system.t)
        target = math.tanh(time)
        target_derivative = 1.0 / math.cosh(time) ** 2
        return testing.ParticleUpdate(
            dxdt=-rate * (state.x - target) + target_derivative,
            dudt=torch.zeros_like(state.u),
            dedt=torch.zeros_like(state.e),
        ), None
    return rhs


@pytest.mark.parametrize('scheme_name', STATE_SPACE_IMPLICIT)
def test_implicit_state_space_schemes_remain_bounded_on_nonlinear_stiff_relaxation(scheme_name):
    scheme = getIntegrator(scheme_name)
    system = testing.PROBLEMS['oscillator']().initial()
    history = StepHistory(maxlen=1)
    for _ in range(10):
        result = scheme(system, dt=0.1, f=_stiff_relaxation_rhs(rate=100.0), history=history)
        system, history = result.state, result.history or history
    x = float(get_reference_state(system).x[0])
    assert math.isfinite(x)
    assert abs(x - math.tanh(1.0)) < 0.2