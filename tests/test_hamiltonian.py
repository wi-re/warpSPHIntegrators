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

from warpSPHIntegrators import getIntegrator, testing
from warpSPHIntegrators.integration import IntegrationSchemes

#: Short and long horizons over the same problem. The long one is 8x the short one,
#: so a scheme with secular drift shows a ratio near 8 and a symplectic one near 1.
T_SHORT, T_LONG, DT = 10.0, 80.0, 0.1

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
SATURATES_EARLY = {'Backward Euler (implicit)'}


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


def test_symplectic_schemes_are_exactly_the_expected_set():
    """A named list, so that adding a scheme forces a deliberate choice of flag."""
    symplectic = {s.name for s in IntegrationSchemes if not s.dissipation}
    assert symplectic == {
        'Leap Frog', 'Symplectic Euler', 'Velocity Verlet',
        'PEFRL', 'VEFRL', 'Semi-Implicit Euler',
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
