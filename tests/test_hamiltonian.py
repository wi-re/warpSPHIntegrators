"""Symplecticity: which schemes conserve energy over long runs, and which drift.

This is what the `dissipation` flag on `IntegrationScheme` is *for*. NOTES.md 2.13
observed that the flag was copy-pasted and self-inconsistent (symplecticEuler was True
while leapFrog was False; PEFRL was False while VEFRL was True) and that nothing read
it. It now has one precise meaning -- "the energy error on a Hamiltonian problem grows
secularly rather than staying bounded" -- and this file is what holds it to that.

The distinction is not visible in the endpoint energy, which merely samples the phase
of an oscillation. It is visible in the *maximum* error over the run: a symplectic
integrator conserves a shadow Hamiltonian, so its energy error stays inside an
O(dt^p) band however long you integrate, while a dissipative one accumulates.
"""

import pytest
import torch

from conftest import NEEDS_STAGE_COUNT
from warpSPHIntegrators import JFNKSolver, getIntegrator, get_reference_state, testing
from warpSPHIntegrators.integration import IntegrationSchemes

#: Short and long horizons over the same problem. The long one is 8x the short one,
#: so a scheme with secular drift shows a ratio near 8 and a symplectic one near 1.
T_SHORT, T_LONG, DT = 10.0, 80.0, 0.1
ENERGY_NOISE_FLOOR = 1e-8

#: Blows up on a Hamiltonian problem at any step size; nothing to measure.
UNSTABLE = {'Forward Euler', 'Explicit Euler'}

#: L-stable schemes damp so hard that the *relative* energy error saturates near its
#: ceiling (all of the initial energy gone) well inside T_SHORT, at which point it
#: cannot grow any further -- the growth ratio reads "bounded", the same signature a
#: symplectic scheme gives, for the opposite reason. Backward Euler measures
#: short=0.98, long=1.00 on `oscillator` (already fully damped by T_SHORT=10) and a
#: flat 1.0 on `kepler`. The dissipative *direction* is not in question -- L-stability
#: is what backward Euler is for -- only this particular growth-ratio test doesn't fit
#: a scheme fast enough to hit its own floor.
SATURATES_EARLY = {'Backward Euler (implicit)', 'BDF1', 'BDF2', 'IMEX Euler', 'TR-BDF2'}


@pytest.mark.parametrize('problem_name', ['oscillator', 'kepler'])
def test_dissipation_flag_predicts_energy_behaviour(scheme, problem_name):
    if scheme.name in UNSTABLE:
        pytest.skip(f'{scheme.name} is unstable on an oscillatory problem')
    if scheme.name in SATURATES_EARLY:
        pytest.skip(f'{scheme.name} saturates near total dissipation before T_SHORT; '
                    f'growth ratio cannot distinguish that from "bounded"')

    problem = testing.PROBLEMS[problem_name]()
    short = testing.max_energy_drift(scheme, problem, DT, T_SHORT)
    long = testing.max_energy_drift(scheme, problem, DT, T_LONG)
    if max(short, long) < ENERGY_NOISE_FLOOR:
        return
    growth = long / short

    if scheme.dissipation:
        assert growth > 2.0, (
            f'{scheme.name} is registered as dissipation=True but its energy error is '
            f'bounded ({short:.3e} -> {long:.3e} over 8x the time). It looks symplectic; '
            f'the flag should be False.'
        )
    else:
        assert growth < 1.5, (
            f'{scheme.name} is registered as dissipation=False (symplectic) but its energy '
            f'error grew {growth:.1f}x ({short:.3e} -> {long:.3e}) when the run got 8x '
            f'longer. A symplectic scheme conserves a shadow Hamiltonian and stays bounded.'
        )


def test_bounded_energy_schemes_are_exactly_the_expected_set():
    """A named list, so that adding a scheme forces a deliberate choice of flag."""
    bounded_energy = {s.name for s in IntegrationSchemes if not s.dissipation}
    assert bounded_energy == {
        'Leap Frog', 'Symplectic Euler', 'Velocity Verlet',
        'PEFRL', 'VEFRL', 'Semi-Implicit Euler', 'Implicit Midpoint',
    }


@pytest.mark.parametrize('name', ['Leap Frog', 'Velocity Verlet', 'PEFRL', 'VEFRL'])
def test_energy_band_narrows_with_the_step_size(name):
    """The bound a symplectic scheme respects is O(dt^p), not just "some constant"."""
    s = getIntegrator(name)
    problem = testing.PROBLEMS['oscillator']()
    coarse = testing.max_energy_drift(s, problem, 0.1, T_SHORT)
    fine = testing.max_energy_drift(s, problem, 0.05, T_SHORT)
    assert fine < coarse / 2 ** (s.order - 0.5), (
        f'{name}: halving dt took the energy band from {coarse:.3e} to {fine:.3e}, '
        f'which is not the O(dt^{s.order}) expected for an order-{s.order} scheme'
    )


# In one degree of freedom, preserving dq wedge dp is exactly det(D Phi_h) = 1.
# This is a direct property check rather than the indirect long-run energy proxy
# above. A larger step separates high-order but non-symplectic RK methods from the
# exactly area-preserving methods without pushing the explicit schemes unstable.
LINEARLY_SYMPLECTIC = {
    'Leap Frog', 'Symplectic Euler', 'Velocity Verlet', 'PEFRL', 'VEFRL',
    'Semi-Implicit Euler', 'Implicit Midpoint', 'Trapezoidal (Crank-Nicolson)',
    'Newmark',
}
NONLINEARLY_SYMPLECTIC = LINEARLY_SYMPLECTIC - {'Newmark', 'Trapezoidal (Crank-Nicolson)'}


def _phase_map(scheme, problem, phase, dt, solver=None):
    system = problem.initial()
    state = get_reference_state(system)
    state.x = torch.tensor([phase[0]], dtype=state.x.dtype)
    state.u = torch.tensor([phase[1]], dtype=state.u.dtype)
    result = scheme(system, dt=dt, f=problem.rhs, solver=solver) if solver else scheme(system, dt=dt, f=problem.rhs)
    final = get_reference_state(result.state)
    return torch.stack((final.x[0], final.u[0]))


def _phase_jacobian(scheme, problem, phase=(0.7, -0.4), dt=0.5, epsilon=1e-4, solver=None):
    columns = []
    for index in range(2):
        plus = list(phase)
        minus = list(phase)
        plus[index] += epsilon
        minus[index] -= epsilon
        columns.append((_phase_map(scheme, problem, plus, dt, solver) -
                _phase_map(scheme, problem, minus, dt, solver)) / (2 * epsilon))
    return torch.stack(columns, dim=1)


@pytest.mark.parametrize('scheme', IntegrationSchemes, ids=lambda scheme: scheme.name)
def test_every_scheme_has_the_expected_linear_oscillator_area_property(scheme):
    if scheme.name in NEEDS_STAGE_COUNT:
        pytest.skip(f'{scheme.name} needs a per-step stage count (s=); '
                    f'covered in tests/test_rkc.py')
    solver = None
    if scheme.name in {'Implicit Midpoint', 'Trapezoidal (Crank-Nicolson)', 'Newmark'}:
        solver = JFNKSolver(tol=1e-12, gmres_tol=1e-12, newton_tol=1e-12, fd_eps=1e-6)
    determinant = float(torch.linalg.det(_phase_jacobian(
        scheme, testing.PROBLEMS['oscillator'](), solver=solver)))
    defect = abs(determinant - 1.0)
    if scheme.name in LINEARLY_SYMPLECTIC:
        assert defect < 1e-7, (
            f'{scheme.name} is expected to preserve oscillator phase area, '
            f'but det(D Phi)={determinant:.9f}'
        )
    else:
        assert defect > 1e-7, (
            f'{scheme.name} unexpectedly looks symplectic on the oscillator; '
            f'det(D Phi)={determinant:.9f}. Reclassify it deliberately if this is intended.'
        )


@pytest.mark.parametrize('name', ['Implicit Midpoint', 'Newmark'])
def test_strict_jfnk_restores_linear_symplecticity(name):
    solver = JFNKSolver(tol=1e-12, gmres_tol=1e-12, newton_tol=1e-12, fd_eps=1e-6)
    determinant = float(torch.linalg.det(_phase_jacobian(
        getIntegrator(name), testing.PROBLEMS['oscillator'](), solver=solver)))
    assert abs(determinant - 1.0) < 1e-7


def _kepler_map(scheme, phase, dt, epsilon=None):
    problem = testing.PROBLEMS['kepler']()
    system = problem.initial()
    state = get_reference_state(system)
    state.x = torch.tensor(phase[:2], dtype=state.x.dtype)
    state.u = torch.tensor(phase[2:], dtype=state.u.dtype)
    result = scheme(system, dt=dt, f=problem.rhs)
    final = get_reference_state(result.state)
    return torch.cat((final.x, final.u))


def _kepler_symplectic_defect(scheme, dt=0.1, epsilon=1e-6):
    phase = (1.0, 0.1, -0.1, 0.95)
    columns = []
    for index in range(4):
        plus = list(phase)
        minus = list(phase)
        plus[index] += epsilon
        minus[index] -= epsilon
        columns.append((_kepler_map(scheme, plus, dt) - _kepler_map(scheme, minus, dt)) /
                       (2 * epsilon))
    jacobian = torch.stack(columns, dim=1)
    zero = torch.zeros((2, 2), dtype=jacobian.dtype)
    identity = torch.eye(2, dtype=jacobian.dtype)
    omega = torch.cat((torch.cat((zero, identity), dim=1),
                       torch.cat((-identity, zero), dim=1)), dim=0)
    return float(torch.linalg.norm(jacobian.T @ omega @ jacobian - omega, ord=float('inf')))


@pytest.mark.parametrize('name', sorted(NONLINEARLY_SYMPLECTIC))
def test_symplectic_schemes_preserve_the_kepler_symplectic_form(name):
    defect = _kepler_symplectic_defect(getIntegrator(name))
    assert defect < 2e-6, f'{name}: ||D Phi^T Omega D Phi - Omega||_inf={defect:.3e}'
