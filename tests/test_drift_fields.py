"""Drift fields (drift.py): configuration quantities with a velocity-linear rate
(the continuity density) advanced with the drifts, not the kicks.

What is pinned here, all on `testing.DRIFT_PROBLEMS`:

* the Verlet family keeps the linear acoustic pair's energy bounded with the
  drift path, and every one of them grows it without (the defect this fixes --
  symplecticEuler at (omega dt)^4/4 per step, velocityVerlet at ~h^2/2,
  leapFrog / semiImplicitEuler / PEFRL / VEFRL at ~h^2);
* the drift path is second order on a problem whose velocity-linear rate
  depends on the position (the kick-attached update is first order in
  velocityVerlet, leapFrog, PEFRL and VEFRL);
* schemes outside the Verlet family ignore the declaration (bitwise), and a
  system that does not opt in never takes the drift path.
"""

import math

import pytest
import torch

from warpSPHIntegrators.state.drift import drift_fields_active
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.state import get_reference_state
import warpSPHIntegrators.testing as testing

VERLET_FAMILY = ['Symplectic Euler', 'Velocity Verlet', 'Leap Frog', 'Semi-Implicit Euler', 'PEFRL', 'VEFRL']

#: omega dt on the acoustic mode; steps for the short and the 8x longer run
H, N_SHORT, N_LONG = 0.3, 200, 1600


def _acoustic_drift(name, drift, steps):
    scheme = get_scheme(name)
    dt = H / testing.acoustic_omega()
    return testing.max_energy_drift(scheme, testing.acoustic_problem(drift=drift), dt, steps * dt)


@pytest.mark.parametrize('name', VERLET_FAMILY)
def test_acoustic_energy_stays_bounded_with_drift_fields(name):
    short, long = _acoustic_drift(name, True, N_SHORT), _acoustic_drift(name, True, N_LONG)
    assert long < 0.25, f'{name}: acoustic energy error {long:.3e} with the drift path'
    assert long / max(short, 1e-12) < 1.5, (
        f'{name}: energy error grew {long / short:.2f}x over an 8x longer run with the '
        f'drift path ({short:.3e} -> {long:.3e}); it should be a bounded band')


@pytest.mark.parametrize('name', VERLET_FAMILY)
def test_kick_attached_density_pumps_acoustic_energy(name):
    """The pre-drift-field behaviour, kept as a measured fact: with the density
    updated alongside the kicks every Verlet scheme amplifies the acoustic mode."""
    short, long = _acoustic_drift(name, False, N_SHORT), _acoustic_drift(name, False, N_LONG)
    assert long > 2.0 * short, (
        f'{name}: the kick-attached update no longer grows the acoustic energy '
        f'({short:.3e} -> {long:.3e}); if that is intended, update drift.py\'s table')


def test_symplectic_euler_growth_is_the_explicit_midpoint_rate():
    """|lambda|^2 = 1 + h^4/4 per step for the (rho, v) pair: the energy of the
    un-dissipated mode grows by that factor, to within a few percent."""
    scheme = get_scheme('Symplectic Euler')
    dt = H / testing.acoustic_omega()
    problem = testing.acoustic_problem(drift=False)
    system = problem.initial()
    energies = []
    for _ in range(N_LONG):
        system = scheme(system, dt=dt, f=problem.rhs).state
        energies.append(problem.energy(get_reference_state(system)))
    steps = torch.arange(len(energies), dtype=torch.float64)
    slope = float(torch.linalg.lstsq(torch.stack([steps, torch.ones_like(steps)], 1),
                                     torch.log(torch.tensor(energies, dtype=torch.float64)).unsqueeze(1)).solution[0])
    assert slope == pytest.approx(H ** 4 / 4, rel=0.15)


def _order(name, gamma, drift):
    scheme = get_scheme(name)
    reference = get_reference_state(testing.run(get_scheme('RK4'),
                                                 testing.drift_oscillator_problem(gamma=gamma), 1e-4, 2.0))
    errors = []
    dts = [0.05, 0.025, 0.0125]
    for dt in dts:
        s = get_reference_state(testing.run(scheme, testing.drift_oscillator_problem(gamma=gamma, drift=drift), dt, 2.0))
        errors.append(float((s.x - reference.x).abs().sum() + (s.u - reference.u).abs().sum()
                            + (s.q - reference.q).abs().sum()))
    return math.log(errors[-2] / errors[-1], 2)


@pytest.mark.parametrize('name', ['Symplectic Euler', 'Velocity Verlet', 'Leap Frog'])
def test_drift_fields_are_second_order_with_a_non_velocity_rate(name):
    """gamma > 0: part of q's rate is not velocity-driven and stays with the kicks."""
    assert _order(name, gamma=0.3, drift=True) > 1.9


@pytest.mark.parametrize('name', ['PEFRL', 'VEFRL'])
def test_ruth_compositions_are_second_order_for_a_purely_kinematic_drift_field(name):
    """The per-drift trapezoid caps the drift field at order 2 (the compositions
    are order 4 only for exactly solved sub-flows). With a non-velocity rate
    (gamma > 0) PEFRL/VEFRL are first order in either mode: the kicks then change
    q while the force depends on q, which one evaluation per kick does not solve
    -- a property of the compositions' quantity handling, not of drift fields."""
    assert _order(name, gamma=0.0, drift=True) > 1.9


@pytest.mark.parametrize('name', ['Velocity Verlet', 'Leap Frog', 'PEFRL', 'VEFRL'])
def test_kick_attached_update_is_first_order(name):
    assert _order(name, gamma=0.0, drift=False) < 1.2


@pytest.mark.parametrize('name', ['RK4', 'Midpoint', "Heun's Method (2nd order)"])
def test_other_schemes_ignore_the_declaration(name):
    scheme = get_scheme(name)
    finals = []
    for drift in (False, True):
        s = get_reference_state(testing.run(scheme, testing.drift_oscillator_problem(gamma=0.3, drift=drift), 0.05, 1.0))
        finals.append(torch.cat([s.x, s.u, s.q]))
    assert torch.equal(finals[0], finals[1])


def test_only_opted_in_systems_take_the_drift_path():
    assert not drift_fields_active(testing.oscillator_problem().initial())
    assert not drift_fields_active(testing.drift_oscillator_problem(drift=False).initial())
    assert drift_fields_active(testing.drift_oscillator_problem(drift=True).initial())
