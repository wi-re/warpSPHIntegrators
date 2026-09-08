"""Stiff nonlinear regression cases for first-order state-space implicit methods."""

import math

import pytest
import torch

from warpSPHIntegrators import StepHistory, getIntegrator, get_reference_state, testing
from warpSPHIntegrators.stability import bdf_is_stable, oscillator_is_stable, oscillator_spectral_radius, rk_is_stable
from warpSPHIntegrators.integration import IntegrationSchemes


STATE_SPACE_IMPLICIT = [
    'Backward Euler (implicit)',
    'Implicit Midpoint',
    'Trapezoidal (Crank-Nicolson)',
    'SDIRK2',
    'TR-BDF2',
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


def test_every_tableau_is_consistent_at_the_dahlquist_origin():
    for scheme in IntegrationSchemes:
        tableau = getattr(scheme.function, 'butcherTableau', getattr(scheme.function, 'dirkTableau', None))
        if tableau is not None:
            assert rk_is_stable(tableau, 0.0), scheme.name


@pytest.mark.parametrize('name', [
    'Backward Euler (implicit)', 'Implicit Midpoint', 'Trapezoidal (Crank-Nicolson)',
    'SDIRK2', 'TR-BDF2',
])
@pytest.mark.parametrize('z', [-10.0, -10.0 + 4.0j, -0.25 + 3.0j])
def test_a_stable_dirk_schemes_contain_known_left_half_plane_points(name, z):
    scheme = getIntegrator(name)
    assert rk_is_stable(scheme.function.dirkTableau, z), f'{name} rejected z={z}'


@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('z', [-10.0, -10.0 + 4.0j, -0.25 + 3.0j])
def test_a_stable_bdf_schemes_contain_known_left_half_plane_points(order, z):
    assert bdf_is_stable(order, z), f'BDF{order} rejected z={z}'


def test_explicit_euler_stability_boundary_is_detected():
    tableau = getIntegrator('Forward Euler').function.butcherTableau
    assert rk_is_stable(tableau, -1.0)
    assert not rk_is_stable(tableau, -2.1)


def test_bdf3_retains_the_negative_real_axis_but_not_full_a_stability():
    assert bdf_is_stable(3, -10.0)
    assert not bdf_is_stable(3, -0.01 + 1.0j)


@pytest.mark.parametrize('method', ['leapfrog', 'velocity_verlet', 'symplectic_euler'])
def test_verlet_family_is_stable_through_the_h_omega_two_boundary(method):
    assert oscillator_is_stable(method, 2.0)
    assert not oscillator_is_stable(method, 2.01)


def test_newmark_average_acceleration_is_unconditionally_stable_on_the_oscillator():
    assert oscillator_is_stable('newmark_average_acceleration', 100.0)
    assert oscillator_spectral_radius('newmark_average_acceleration', 100.0) == pytest.approx(1.0)


def test_newmark_linear_acceleration_has_the_sqrt_twelve_stability_boundary():
    assert oscillator_is_stable('newmark_linear_acceleration', math.sqrt(12.0))
    assert not oscillator_is_stable('newmark_linear_acceleration', math.sqrt(12.0) + 0.01)