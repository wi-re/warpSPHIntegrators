"""Stiff nonlinear regression cases for first-order state-space implicit methods."""

import math

import pytest
import torch

from warpSPHIntegrators import StepHistory, getIntegrator, get_reference_state, testing
from warpSPHIntegrators.stability import bdf_is_stable, oscillator_is_stable, oscillator_spectral_radius, rk_is_stable, rk_stability_function
from warpSPHIntegrators.integration import IntegrationSchemes
from warpSPHIntegrators.enums import DIRK


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
    # SBDF2: one Dormand-Prince startup step (like BDF2) plus the BDF2
    # backbone, so it damps the z = -10 transient the same way BDF2 does
    # (measured |x - tanh(1)| = 8e-4). SBDF3 and CNAB2 need two and one
    # startup steps respectively, and are pinned at the rates their shared
    # explicit startup can absorb (the tests below).
    'SBDF2',
]
STATE_SPACE_IMPLICIT += [
    s.name for s in IntegrationSchemes
    if s.family is DIRK and s.stability in {'A', 'L'}
    and s.name not in STATE_SPACE_IMPLICIT
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


@pytest.mark.parametrize('scheme_name', ['BDF3', 'BDF4', 'BDF5', 'SBDF3'])
def test_higher_order_bdf_with_threaded_history_tracks_stiff_relaxation(scheme_name):
    """BDF3-5 and the pure-implicit SBDF3 limit at rate 10.

    BDF3-5 now have implicit starters; the rate-100 bootstrap/order regression
    is in test_stiff_startup.py. Actual IMEX splits retain their DP5 starter.
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


def test_cnab2_with_threaded_history_tracks_stiff_relaxation():
    """CNAB2 at rate=10 (z = 1): one Dormand-Prince startup step (the
    trapezoidal backbone damps |R(-1)| = 1/3 per step, so the startup
    transient is gone quickly) plus the full CNAB2 formula. It is out of the
    z = -10 list above for a startup-only reason: the one-step trapezoidal
    DIRK scheme there skips the startup entirely, while CNAB2 pays one DP5
    step (~1e3 amplification at z = -10) that the trapezoidal rule's mild
    real-axis damping (|R(-10)| = 2/3) cannot undo in nine more steps.
    Measured |x - tanh(1)| = 6e-5 at rate=10."""
    scheme = getIntegrator('CNAB2')
    system = testing.PROBLEMS['oscillator']().initial()
    history = StepHistory(maxlen=1)
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
    # Wave 8 (NOTES.md 3.28): the 24 A-stable new tableaus (22 L-stable plus
    # Kvaerno4 and SSPSDIRK2, A-stable not L). All contain these three LHP
    # probe points; SFSDIRK4-8 are NOT A-stable (bounded real-axis region only)
    # and stay out of this list.
    'Kvaerno3', 'Kvaerno4', 'Kvaerno5', 'KenCarp3', 'KenCarp4', 'KenCarp47',
    'KenCarp5', 'KenCarp58', 'ESDIRK3(2)5L[2]SA', 'ESDIRK4(3)6L[2]SA2',
    'ESDIRK4(3)7L[2]SA', 'ESDIRK5(4)7L[2]SA2', 'ESDIRK5(4)I8L[2]SA',
    'ESDIRK6(5)9L[2]SA', 'SSPSDIRK2', 'ARS222', 'ARS232', 'ARS343', 'ARS443',
    'BHR553', 'Cash4', 'CFNLIRK3', 'Hairer4', 'Hairer42',
])
@pytest.mark.parametrize('z', [-10.0, -10.0 + 4.0j, -0.25 + 3.0j])
def test_a_stable_dirk_schemes_contain_known_left_half_plane_points(name, z):
    scheme = getIntegrator(name)
    assert rk_is_stable(scheme.function.dirkTableau, z), f'{name} rejected z={z}'


@pytest.mark.parametrize('name', [
    'Backward Euler (implicit)', 'SDIRK2', 'TR-BDF2',
    'ESDIRK3(2)4L[2]SA', 'ESDIRK4(3)6L[2]SA',
    # Wave 8 (NOTES.md 3.28): the 19 new L-stable tableaus whose measured
    # |R(-100)| lands under 0.1 (2.6e-3 .. 7.9e-2). The three new L-stable
    # tableaus with a large damping constant (|R(-100)| = 0.149/0.125/0.106)
    # are pinned at larger |z| in the test below.
    'Kvaerno3', 'Kvaerno5', 'KenCarp3', 'KenCarp4', 'KenCarp5', 'KenCarp58',
    'ESDIRK3(2)5L[2]SA', 'ESDIRK4(3)6L[2]SA2', 'ESDIRK5(4)I8L[2]SA',
    'ESDIRK6(5)9L[2]SA', 'ARS222', 'ARS232', 'ARS343', 'ARS443', 'BHR553',
    'Cash4', 'CFNLIRK3', 'Hairer4', 'Hairer42',
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


@pytest.mark.parametrize('name', [
    'KenCarp47', 'ESDIRK4(3)7L[2]SA', 'ESDIRK5(4)7L[2]SA2',
])
def test_l_stable_dirk_schemes_with_large_damping_constants(name):
    """The complement of the < 0.1 test: L-stable tableaus whose C in
    R(z) ~= C/|z| is large enough that |R(-100)| still exceeds 0.1 (measured
    KenCarp47 0.149, ESDIRK4(3)7L[2]SA 0.125, ESDIRK5(4)7L[2]SA2 0.106).
    L-stability is pinned at larger |z|: |R(-1000)| < 0.1 (measured
    2.6e-2/2.2e-2/1.5e-2) and |R(-1e6)| < 1e-4 (measured 2.8e-5/2.4e-5/1.6e-5).
    A fine left-half-plane grid (NOTES.md 3.28) found no |R| > 1 for any of
    the three, so the classification stays 'L' despite the large constant."""
    tab = getIntegrator(name).function.dirkTableau
    r100 = abs(rk_stability_function(tab, -100.0))
    r1000 = abs(rk_stability_function(tab, -1000.0))
    r1e6 = abs(rk_stability_function(tab, -1e6))
    assert r100 > 0.1, (
        f'{name}: |R(-100)| = {r100:.4f}; this test covers the large-constant '
        f'L-stable row (the < 0.1 test covers the rest)')
    assert r1000 < 0.1, f'{name}: |R(-1000)| = {r1000:.4f}, expected L-stable decay'
    assert r1e6 < 1e-4, f'{name}: |R(-1e6)| = {r1e6:.3e}, expected decay to 0'


@pytest.mark.parametrize('name', ['Implicit Midpoint', 'Trapezoidal (Crank-Nicolson)'])
def test_a_stable_but_not_l_dirk_schemes_approach_unit_magnitude(name):
    """The companion: A-stable but not L-stable tableaus have R(z) -> -1 along
    the negative real axis (|R(-100)| = 49/51 ~= 0.961 by the (1+z/2)/(1-z/2)
    closed form for both), i.e. they bound the solution but do not damp it.
    This is what separates the two rows of the stability='L' flag."""
    tab = getIntegrator(name).function.dirkTableau
    r = abs(rk_stability_function(tab, -100.0))
    assert r > 0.9, f'{name}: |R(-100)| = {r:.4f}, expected approach to 1 (A-stable, not L)'


@pytest.mark.parametrize('name', ['Kvaerno4', 'SSPSDIRK2'])
def test_a_stable_not_l_dirk_schemes_damp_sub_l(name):
    """The middle class of the stability='A' flag (wave 8, NOTES.md 3.28):
    A-stable (they pass the probe-point test above) but not L-stable, so R(z)
    does not decay to 0 along the negative real axis.

    Measured at z = -100: Kvaerno4 |R| = 0.629, SSPSDIRK2 |R| = 0.852 -- more
    damping than the unit-magnitude class (|R(-100)| = 49/51 ~= 0.961 there)
    but far from the L-stable < 0.1. As z -> -inf, Kvaerno4 settles at a
    nonzero A-stable asymptote |R(inf)| = 0.7175 while SSPSDIRK2 settles at
    |R(inf)| ~= 1 (R(inf) = -1), so neither decays ~ C/|z| like the L-stable
    row: both sit strictly between the two tests above."""
    tab = getIntegrator(name).function.dirkTableau
    r = abs(rk_stability_function(tab, -100.0))
    assert 0.1 < r < 0.9, (
        f'{name}: |R(-100)| = {r:.4f}, expected sub-L damping (strictly between '
        f'the L-stable < 0.1 row and the unit-magnitude > 0.9 row)')


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
