"""Stiff nonlinear regression cases for first-order state-space implicit methods."""

import math

import pytest
import torch

from warpSPHIntegrators import StepHistory, getIntegrator, get_reference_state, testing
from warpSPHIntegrators.stability import bdf_is_stable, oscillator_is_stable, oscillator_spectral_radius, rk_is_stable, rk_stability_function
from warpSPHIntegrators.integration import IntegrationSchemes


STATE_SPACE_IMPLICIT = [
    'Backward Euler (implicit)',
    'Implicit Midpoint',
    'Trapezoidal (Crank-Nicolson)',
    'SDIRK2',
    'TR-BDF2',
    'ESDIRK3(2)4L[2]SA',
    'ESDIRK4(3)6L[2]SA',
    'ARK3(2)4L[2]SA',
    'ARK4(3)6L[2]SA',
    'BDF1',
    'BDF2',
    'IMEX Euler',
    # Both coupled fully implicit block schemes are A-stable: at z = -10
    # (rate 100, dt 0.1) the stiff transient decays as |R(-10)|^n -- L-stable
    # Radau IIA to ~1e-10, Gauss-Legendre (|R(-10)| ~= 0.30) in ~4 steps --
    # and the slow tanh component is integrated at their full order.
    'Gauss-Legendre 2',
    'Radau IIA s=2',
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


@pytest.mark.parametrize('scheme_name', ['BDF3', 'BDF4', 'BDF5'])
def test_higher_order_bdf_with_threaded_history_tracks_stiff_relaxation(scheme_name):
    """BDF3-5 at a rate their *explicit* Dormand-Prince startup can absorb.

    rate=10 (stiff scale z = 1): the startup steps are stable and the threaded
    history drives the full BDF formula. rate=100 (like the test above) is out of
    reach for a different reason: the startup is explicit, so it needs `order - 1`
    consecutive stable *explicit* steps at dt=0.1 (DP5 amplifies the stiff
    transient by ~1e3 per step at z = -10), and BDF3-5's own A(alpha) damping
    along the negative real axis (|R(-10)| ~= 0.2-0.3 per step) cannot recover
    from two or more compounded startup steps (BDF2's single startup step can:
    |R_BDF2(-10)| ~= 0.21 damps it away). That is a
    property of the shared DP5 startup design (NOTES.md S3.7), not of the BDF
    formulas -- which is why this test threads history with maxlen = scheme.steps
    instead of the maxlen=1 the one-step schemes above use.
    """
    scheme = getIntegrator(scheme_name)
    system = testing.PROBLEMS['oscillator']().initial()
    history = StepHistory(maxlen=max(1, scheme.steps))
    for _ in range(10):
        result = scheme(system, dt=0.1, f=_stiff_relaxation_rhs(rate=10.0), history=history)
        system, history = result.state, result.history
    x = float(get_reference_state(system).x[0])
    assert math.isfinite(x)
    assert abs(x - math.tanh(1.0)) < 0.2


def test_every_tableau_is_consistent_at_the_dahlquist_origin():
    for scheme in IntegrationSchemes:
        tableau = getattr(scheme.function, 'butcherTableau',
                          getattr(scheme.function, 'dirkTableau',
                                  getattr(scheme.function, 'blockTableau', None)))
        if tableau is not None:
            assert rk_is_stable(tableau, 0.0), scheme.name


@pytest.mark.parametrize('name', [
    'Backward Euler (implicit)', 'Implicit Midpoint', 'Trapezoidal (Crank-Nicolson)',
    'SDIRK2', 'TR-BDF2', 'ESDIRK3(2)4L[2]SA', 'ESDIRK4(3)6L[2]SA',
])
@pytest.mark.parametrize('z', [-10.0, -10.0 + 4.0j, -0.25 + 3.0j])
def test_a_stable_dirk_schemes_contain_known_left_half_plane_points(name, z):
    scheme = getIntegrator(name)
    assert rk_is_stable(scheme.function.dirkTableau, z), f'{name} rejected z={z}'


@pytest.mark.parametrize('name', [
    'Backward Euler (implicit)', 'SDIRK2', 'TR-BDF2',
    'ESDIRK3(2)4L[2]SA', 'ESDIRK4(3)6L[2]SA',
])
def test_l_stable_dirk_schemes_damp_the_negative_real_axis(name):
    """L-stability, method-specifically: the stability function decays along the
    negative real axis (|R(z)| -> 0 as z -> -inf), not merely stays bounded.

    Hand-checked values at z = -100: backward Euler R(z) = 1/(1-z) gives
    1/101 ~= 0.0099; SDIRK2 and TR-BDF2 both land near 0.044-0.049; the ESDIRK
    pair sits at 2.65e-2 (order 3) and 7.57e-2 (order 4). All show the L-stable
    decay ~ C/|z|, far under the 0.1 bound below, so the bound pins the class,
    not one tableau's exact constant."""
    tab = getIntegrator(name).function.dirkTableau
    r = abs(rk_stability_function(tab, -100.0))
    assert r < 0.1, f'{name}: |R(-100)| = {r:.4f}, expected strong L-stable damping'


@pytest.mark.parametrize('name', ['Implicit Midpoint', 'Trapezoidal (Crank-Nicolson)'])
def test_a_stable_but_not_l_dirk_schemes_approach_unit_magnitude(name):
    """The companion: A-stable but not L-stable tableaus have R(z) -> -1 along
    the negative real axis (|R(-100)| = 49/51 ~= 0.961 by the (1+z/2)/(1-z/2)
    closed form for both), i.e. they bound the solution but do not damp it.
    This is what separates the two rows of the stability='L' flag."""
    tab = getIntegrator(name).function.dirkTableau
    r = abs(rk_stability_function(tab, -100.0))
    assert r > 0.9, f'{name}: |R(-100)| = {r:.4f}, expected approach to 1 (A-stable, not L)'


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