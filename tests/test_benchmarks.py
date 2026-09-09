"""Phase 8: broadened nonlinear/stiff benchmark and stability suite.

These tests exercise the Phase 8 problem registry (Prothero-Robinson in both
signs, stiff damped oscillator in all three damping regimes, van der Pol,
Robertson kinetics, semi-discrete diffusion) at small deterministic sizes, and
they pin down the damped-oscillator amplification matrices added to
``stability.py`` against the registered schemes.

The long cost/accuracy sweeps live in ``scripts/stiff_benchmark_suite.py``
(they produce ``images/stiff_benchmark_suite.png``); the damping-ratio
stability contours live in ``scripts/oscillator_stability_gallery.py``.
"""

import math

import numpy as np
import pytest
import torch

from warpSPHIntegrators import StepHistory, getIntegrator, get_reference_state, testing
from warpSPHIntegrators.newmark import newmark
from warpSPHIntegrators.stability import (
    damped_oscillator_amplification_matrix,
    damped_oscillator_spectral_radius,
    oscillator_spectral_radius,
)


def _run(scheme_name, problem, dt, T, history_len=0, **kw):
    """Fixed-step run of ``T`` with the registered scheme; returns the final system."""
    scheme = getIntegrator(scheme_name)
    system = problem.initial()
    history = StepHistory(maxlen=history_len) if history_len else None
    for _ in range(int(round(T / dt))):
        result = scheme(system, dt=dt, f=problem.rhs, history=history, **kw)
        system, history = result.state, result.history or history
    return system


# --------------------------------------------------------------------------- #
# Problem registry                                                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', [
    'oscillator', 'forced', 'damped', 'kepler',
    'stiffRelaxation', 'stiffDampedOscillator', 'vanDerPol', 'robertson', 'diffusion',
])
def test_problem_registry_factories_produce_finite_states(name):
    problem = testing.PROBLEMS[name]()
    system = problem.initial()
    state = get_reference_state(system)
    assert math.isfinite(float(state.x.sum()))
    assert math.isfinite(float(state.u.sum()))
    update, _ = problem.rhs(system, dt=0.01)
    assert math.isfinite(float(update.dxdt.sum()))
    assert math.isfinite(float(update.dudt.sum()))


# --------------------------------------------------------------------------- #
# Prothero-Robinson (stable and unstable variants)                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('forcing', ['tanh', 'sin', 'cos'])
def test_stable_pr_backward_euler_tracks_all_forcings(forcing):
    """rate=100, dt=0.1 (stiff scale z = -10): explicit methods cannot take this step,
    backward Euler is unconditionally stable and tracks the moving target."""
    problem = testing.stiff_relaxation_problem(rate=100.0, forcing=forcing)
    system = _run('Backward Euler (implicit)', problem, 0.1, 1.0)
    exact, _ = problem.exact(1.0)
    assert abs(float(get_reference_state(system).x[0]) - exact[0]) < 1e-2


def test_unstable_pr_backward_euler_tracks_positive_eigenvalue():
    """Unstable PR (eigenvalue +50, z = +5): the exact solution is still x = sin(t).
    Backward Euler is stable on the positive real axis (R(z) = 1/(1-z) -> 1/2 as z -> +inf)."""
    problem = testing.stiff_relaxation_problem(rate=50.0, forcing='sin', sign=-1)
    system = _run('Backward Euler (implicit)', problem, 0.1, 1.0)
    assert abs(float(get_reference_state(system).x[0]) - math.sin(1.0)) < 1e-2


def test_unstable_pr_explicit_and_esdirk_diverge():
    """On the unstable PR the error is amplified by |R(+5)| ~= 117 per step for DP5, and
    ESDIRK6's stability function has a pole at z = 1 (stiffly-accurate gamma = 1 stage),
    so both leave the O(1) band within a handful of steps while BE (test above) tracks."""
    problem = testing.stiff_relaxation_problem(rate=50.0, forcing='sin', sign=-1)
    for name in ('Dormand-Prince 5(4)', 'ESDIRK4(3)6L[2]SA'):
        system = _run(name, problem, 0.1, 0.5)
        x = float(get_reference_state(system).x[0])
        assert not math.isfinite(x) or abs(x) > 10.0


# --------------------------------------------------------------------------- #
# Stiff damped oscillator: three regimes                                      #
# --------------------------------------------------------------------------- #

def test_stiff_damped_oscillator_overdamped_backward_euler():
    """Dissipative stiffness (omega=1, c=50): the step restriction dt < 2/c is
    violated (dt=0.1 -> c*h = 5) but backward Euler resolves the relaxation."""
    problem = testing.stiff_damped_oscillator_problem(omega=1.0, c=50.0)
    system = _run('Backward Euler (implicit)', problem, 0.1, 1.0)
    exact_x, exact_u = problem.exact(1.0)
    state = get_reference_state(system)
    err = abs(float(state.x[0]) - exact_x[0]) + abs(float(state.u[0]) - exact_u[0])
    assert err < 1e-3


def test_stiff_damped_oscillator_high_frequency_velocity_verlet():
    """High-frequency stiffness (omega=50, c=1): dt=0.005 gives h*omega = 0.25, inside
    the explicit restriction, and velocity Verlet is second order."""
    problem = testing.stiff_damped_oscillator_problem(omega=50.0, c=1.0)
    system = _run('Velocity Verlet', problem, 0.005, 1.0)
    exact_x, exact_u = problem.exact(1.0)
    state = get_reference_state(system)
    assert abs(float(state.x[0]) - exact_x[0]) < 0.05
    assert abs(float(state.u[0]) - exact_u[0]) / 50.0 < 0.15


def test_stiff_damped_oscillator_critically_damped_esdirk():
    """Critically damped branch of the closed-form exact (omega=1, c=2)."""
    problem = testing.stiff_damped_oscillator_problem(omega=1.0, c=2.0)
    system = _run('ESDIRK4(3)6L[2]SA', problem, 0.01, 0.5)
    exact_x, exact_u = problem.exact(0.5)
    state = get_reference_state(system)
    err = abs(float(state.x[0]) - exact_x[0]) + abs(float(state.u[0]) - exact_u[0])
    assert err < 1e-8


# --------------------------------------------------------------------------- #
# Van der Pol: limit cycle + explicit wall                                    #
# --------------------------------------------------------------------------- #

def test_van_der_pol_moderate_mu_limit_cycle_amplitude():
    """mu=2: the attractor is a limit cycle of amplitude ~2; ESDIRK6 at dt=0.05 sits
    on the cycle in the second half of the run."""
    problem = testing.van_der_pol_problem(mu=2.0)
    scheme = getIntegrator('ESDIRK4(3)6L[2]SA')
    system = problem.initial()
    amplitude = []
    for i in range(800):  # T = 40
        system = scheme(system, dt=0.05, f=problem.rhs).state
        if i >= 400:
            amplitude.append(abs(float(get_reference_state(system).x[0])))
    assert 1.8 < max(amplitude) < 2.2


def test_van_der_pol_large_mu_limit_cycle_amplitude():
    """mu=10: the cycle is stiffer in u (|dudt_u| ~ mu*(1 + amplitude^2)); ESDIRK6
    stays on the amplitude-~2 cycle at dt=0.1 where coarser steps land on spurious
    large-amplitude discrete attractors (see scripts/stiff_benchmark_suite.py)."""
    problem = testing.van_der_pol_problem(mu=10.0)
    scheme = getIntegrator('ESDIRK4(3)6L[2]SA')
    system = problem.initial()
    amplitude = []
    for i in range(500):  # T = 50
        system = scheme(system, dt=0.1, f=problem.rhs).state
        if i >= 250:
            amplitude.append(abs(float(get_reference_state(system).x[0])))
    assert 1.8 < max(amplitude) < 2.3


def test_van_der_pol_moderate_mu_explicit_tracks():
    """mu=2 is not stiff at dt=0.05 (fast eigenvalue ~ mu): DP5 tracks the cycle,
    i.e. the explicit restriction is not active here."""
    problem = testing.van_der_pol_problem(mu=2.0)
    system = _run('Dormand-Prince 5(4)', problem, 0.05, 40.0)
    assert 1.5 < abs(float(get_reference_state(system).x[0])) < 2.5


def test_van_der_pol_large_mu_explicit_diverges():
    """mu=10, dt=0.5: the fast eigenvalue is ~ mu*(1 + x^2) ~ O(10-20), z is far outside
    DP5's stability region, and the solution leaves the O(1) band within a few steps."""
    problem = testing.van_der_pol_problem(mu=10.0)
    scheme = getIntegrator('Dormand-Prince 5(4)')
    system = problem.initial()
    for _ in range(5):
        system = scheme(system, dt=0.5, f=problem.rhs).state
    x = float(get_reference_state(system).x[0])
    assert not math.isfinite(x) or abs(x) > 100.0


# --------------------------------------------------------------------------- #
# Robertson kinetics: invariants + scheme agreement                           #
# --------------------------------------------------------------------------- #

def _run_robertson_bootstrapped(scheme_name, dt, T, history_len=0, **kw):
    """Robertson starts on a stiff quasi-steady layer (width ~1e-3) that a fixed step
    of dt >= 0.02 cannot cross from y(0) = (1, 0, 0): ten dt=0.001 bootstrap steps put
    the solution on the quasi-steady manifold, then the main dt takes over."""
    problem = testing.robertson_problem()
    scheme = getIntegrator(scheme_name)
    system = problem.initial()
    history = StepHistory(maxlen=history_len) if history_len else None
    for _ in range(10):
        result = scheme(system, dt=0.001, f=problem.rhs, history=history, **kw)
        system, history = result.state, result.history or history
    for _ in range(int(round((T - 0.01) / dt))):
        result = scheme(system, dt=dt, f=problem.rhs, history=history, **kw)
        system, history = result.state, result.history or history
    return system


def test_robertson_invariants_on_long_run():
    """BE with dt=0.5 over T=100 (bootstrapped) must respect the structural invariants:
    mass conservation, positivity, monotone y3, and y2 on its O(1e-5) quasi-steady
    plateau. (dt >= 2 instead converges each step to an unstable discrete fixed point
    of the stiff quadratic and diverges; that is BE inaccuracy at dt above the fast
    scale, not a solver failure.)"""
    scheme = getIntegrator('Backward Euler (implicit)')
    problem = testing.robertson_problem()
    system = problem.initial()
    for _ in range(10):
        system = scheme(system, dt=0.001, f=problem.rhs).state
    y3_samples = []
    min_component = 1.0
    for i in range(500):  # dt=0.2, T = 100
        system = scheme(system, dt=0.2, f=problem.rhs).state
        y = get_reference_state(system).x.tolist()
        min_component = min(min_component, *y)
        if i in (49, 499) or (i + 1) % 250 == 0:
            y3_samples.append((0.2 * (i + 1), y[2]))
    y = get_reference_state(system).x.tolist()
    assert abs(sum(y) - 1.0) < 1e-12
    assert min_component > 0.0
    assert 0.55 < y[0] < 0.70
    assert 0.3 < y[2] < 0.45
    assert y[1] < 1e-3
    y3_values = [v for _, v in y3_samples]
    assert all(a < b for a, b in zip(y3_values, y3_values[1:]))


def test_robertson_implicit_scheme_agreement():
    """Against a bootstrapped ESDIRK6 reference (dt=0.02, T=10): BE is first order in
    the slow mode, TR-BDF2 is second order, and BDF2 needs dt=0.001 because the fast
    eigenvalue is ~2.2e3 (z = 2.19 sits just inside BDF2's 2.414 negative-real boundary)."""
    reference = _run_robertson_bootstrapped('ESDIRK4(3)6L[2]SA', 0.02, 10.0)
    y_ref = get_reference_state(reference).x.tolist()
    for name, dt, history_len, tol in (
        ('Backward Euler (implicit)', 0.02, 0, 1e-3),
        ('TR-BDF2', 0.02, 0, 1e-6),
        ('BDF2', 0.001, 1, 1e-3),
    ):
        system = _run_robertson_bootstrapped(name, dt, 10.0, history_len=history_len)
        y = get_reference_state(system).x.tolist()
        assert max(abs(a - b) for a, b in zip(y, y_ref)) < tol


# --------------------------------------------------------------------------- #
# Semi-discrete diffusion                                                     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('scheme_name, history_len, tol', [
    ('Backward Euler (implicit)', 0, 5e-2),
    ('BDF2', 1, 5e-3),
])
def test_diffusion_spectral_stiffness_accuracy(scheme_name, history_len, tol):
    """n=32, D=1: the fastest discrete Laplacian eigenvalue is ~-246, so dt=0.01 puts
    the top mode at z ~= -2.46 (outside BDF2's A-stable interval, inside BE's).
    The error is dominated by the slowest mode's first/second-order time error."""
    problem = testing.diffusion_problem(n=32)
    system = _run(scheme_name, problem, 0.01, 0.1, history_len=history_len)
    exact, _ = problem.exact(0.1)
    err = max(abs(a - b) for a, b in zip(get_reference_state(system).x.tolist(), exact))
    assert err < tol


# --------------------------------------------------------------------------- #
# Explicit step-restriction walls (roadmap gate: the restriction is active)   #
# --------------------------------------------------------------------------- #

def test_explicit_wall_dormand_prince_stable_pr():
    """rate=100, dt=0.1 (z = -10, far outside DP5's ~3.3 negative-real boundary):
    three steps already leave the O(1) band, while BE at the same dt tracks
    (test_stable_pr_backward_euler_tracks_all_forcings)."""
    problem = testing.stiff_relaxation_problem(rate=100.0, forcing='tanh')
    system = _run('Dormand-Prince 5(4)', problem, 0.1, 0.3)
    assert abs(float(get_reference_state(system).x[0])) > 10.0


def test_explicit_wall_leapfrog_dissipative_stiffness():
    """omega=1, c=50, dt=0.1: the dissipative scale c*h = 5 exceeds leapfrog's c < 2
    bound for its explicit old-velocity damping term; ten steps blow up while BE at
    the same dt tracks (test_stiff_damped_oscillator_overdamped_backward_euler)."""
    problem = testing.stiff_damped_oscillator_problem(omega=1.0, c=50.0)
    system = _run('Leap Frog', problem, 0.1, 1.0)
    assert abs(float(get_reference_state(system).x[0])) > 100.0


def test_explicit_wall_leapfrog_high_frequency():
    """omega=50, c=1, dt=0.05: h*omega = 2.5 is outside the oscillator stability
    boundary (~1.28 for leapfrog); twenty steps blow up while velocity Verlet at
    dt=0.005 tracks (test_stiff_damped_oscillator_high_frequency_velocity_verlet)."""
    problem = testing.stiff_damped_oscillator_problem(omega=50.0, c=1.0)
    system = _run('Leap Frog', problem, 0.05, 1.0)
    assert abs(float(get_reference_state(system).x[0])) > 2.0


# --------------------------------------------------------------------------- #
# Damped oscillator amplification matrices (stability.py)                     #
# --------------------------------------------------------------------------- #

DAMPED_METHODS = [
    'leapfrog',
    'velocity_verlet',
    'symplectic_euler',
    'newmark_average_acceleration',
    'newmark_linear_acceleration',
]


@pytest.mark.parametrize('method', DAMPED_METHODS)
def test_damped_matrix_reduces_to_undamped_at_zero_damping(method):
    """Every damped formula must reproduce the registered undamped matrix exactly
    at zeta = 0 (the existing Phase 6 oscillator stability results)."""
    h_omegas = np.concatenate([np.linspace(0.0, 0.05, 5), np.logspace(-2, math.log10(5.0), 240)])
    for h_omega in h_omegas:
        damped = damped_oscillator_spectral_radius(method, float(h_omega), 0.0)
        undamped = oscillator_spectral_radius(method, float(h_omega))
        assert abs(damped - undamped) < 1e-12


def _one_step_matrix(scheme, omega, c, h, **kw):
    """Apply the scheme once to the two basis states of the (x, h*v) variables
    and assemble the numerical amplification matrix."""
    problem = testing.stiff_damped_oscillator_problem(omega=omega, c=c)
    solver_opts = {'matvec': 'jvp', 'tol': 1e-14, 'newton_tol': 1e-14}
    columns = []
    for x0, v0 in ((1.0, 0.0), (0.0, 1.0 / h)):
        system = testing.ParticleSystem(
            state=testing.ParticleState(
                x=torch.tensor([x0], dtype=torch.float64),
                u=torch.tensor([v0], dtype=torch.float64),
                e=torch.tensor([0.0], dtype=torch.float64),
                m=torch.tensor([1.0], dtype=torch.float64)),
            t=0.0)
        result = scheme(system, dt=h, f=problem.rhs, solver_opts=solver_opts, **kw)
        state = get_reference_state(result.state)
        columns.append((float(state.x[0]), h * float(state.u[0])))
    return np.array(columns).T


@pytest.mark.parametrize('method, scheme_name, scheme_kw', [
    ('leapfrog', 'Leap Frog', {}),
    ('velocity_verlet', 'Velocity Verlet', {}),
    ('symplectic_euler', 'Symplectic Euler', {}),
    # the registry's 'Newmark' is the beta = 1/4 average-acceleration default;
    # the linear-acceleration variant (beta = 1/6) is the unregistered alias
    ('newmark_average_acceleration', 'Newmark', {}),
    ('newmark_linear_acceleration', None, {'beta': 1.0 / 6.0}),
])
@pytest.mark.parametrize('h_omega, zeta', [(0.3, 0.05), (1.0, 0.2), (2.5, 0.5)])
def test_damped_matrix_matches_registered_one_step(method, scheme_name, scheme_kw, h_omega, zeta):
    """The closed-form matrix in stability.py must equal the scheme's actual
    one-step map (basis-state cross-check). omega = 1, h = h_omega,
    c = 2*zeta*omega."""
    scheme = getIntegrator(scheme_name) if scheme_name else newmark
    numerical = _one_step_matrix(scheme, 1.0, 2.0 * zeta, h_omega, **scheme_kw)
    closed = damped_oscillator_amplification_matrix(method, h_omega, zeta)
    tol = 1e-8 if method.startswith('newmark') else 1e-10
    assert np.max(np.abs(numerical - closed)) < tol


def test_damping_extends_stable_region_velocity_verlet():
    """h*omega = 2.5 is outside velocity Verlet's undamped boundary (real eigenvalues
    -0.25 and -4.0). Mild damping (zeta = 0.5) is not enough (rho ~= 1.23); at
    zeta = 0.6 the eigenvalues turn complex with |lambda| = sqrt(det) = 0.5, strictly
    inside the unit circle. The Schur-Cohn conditions give the stable window
    0.5055 < zeta < 0.8 at this h*omega."""
    assert damped_oscillator_spectral_radius('velocity_verlet', 2.5, 0.0) > 3.0
    assert damped_oscillator_spectral_radius('velocity_verlet', 2.5, 0.5) > 1.0
    assert damped_oscillator_spectral_radius('velocity_verlet', 2.5, 0.6) < 1.0


def test_damping_extends_stable_region_newmark():
    """Average-acceleration Newmark is neutrally stable undamped for every h*omega
    (rho = 1, a complex conjugate pair on the unit circle); zeta = 0.1 moves the pair
    strictly inside at h*omega = 10."""
    assert damped_oscillator_spectral_radius('newmark_average_acceleration', 10.0, 0.0) <= 1.0 + 1e-7
    assert damped_oscillator_spectral_radius('newmark_average_acceleration', 10.0, 0.1) < 1.0


def test_damping_extends_stable_region_symplectic_euler():
    """Symplectic Euler is neutrally stable at h*omega = 0.8 undamped; zeta = 0.5
    damps the pair to rho ~= 0.4."""
    assert damped_oscillator_spectral_radius('symplectic_euler', 0.8, 0.0) <= 1.0 + 1e-7
    assert damped_oscillator_spectral_radius('symplectic_euler', 0.8, 0.5) < 1.0


def test_leapfrog_old_velocity_damping_does_not_stabilize():
    """Leapfrog's second force evaluation sits at the old velocity, so its damping is
    an explicit term in c = 2*zeta*h*omega that is itself unstable for c > 2: unlike
    velocity Verlet, no damping ratio rescues h*omega = 2.5."""
    assert damped_oscillator_spectral_radius('leapfrog', 2.5, 0.5) > 1.0
    # c = 2*zeta*h*omega = 4 > 2: the explicit old-velocity update alone is unstable
    assert damped_oscillator_spectral_radius('leapfrog', 1.0, 2.0) > 1.5
