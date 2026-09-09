"""DIRK driver (NOTES.md S3.6 Phase 2): the four shipped tableaus and the FixedPointSolver.

Most of the generic behaviour (state cloning, copied/ephemeral fields, stage times,
kwargs passthrough) is already covered for these schemes by the existing parametrized
`scheme` fixture in every other test file -- adding four entries to `IntegrationSchemes`
pulled them into all of that for free. This file covers what's specific to DIRK: the
convergence order of each tableau, the Picard solver's own behaviour (stiff divergence,
iteration-count control, pluggability), and the two-tier "converged vs. shipped default"
distinction the energy-drift investigation in `integration.py`'s registration comment
turned up.
"""

import numpy as np
import pytest
import torch

from warpSPHIntegrators import FixedPointSolver, get_reference_state, getIntegrator, testing
from warpSPHIntegrators.dirk import DIRK, getDIRKTableau

DIRK_SCHEMES = [
    'Backward Euler (implicit)', 'Implicit Midpoint', 'Trapezoidal (Crank-Nicolson)',
    'SDIRK2', 'TR-BDF2', 'ESDIRK3(2)4L[2]SA', 'ESDIRK4(3)6L[2]SA',
]


def test_newmark_reaches_second_order_on_the_oscillator():
    s = getIntegrator('Newmark')
    prob = testing.PROBLEMS['oscillator']()
    dts = testing.default_step_sizes(0.1, 5)
    order, errors = testing.convergence(s, prob, dts, T=2.0)
    assert order == pytest.approx(2.0, abs=0.15), (
        f'Newmark on oscillator: measured order {order:.3f}, expected 2. errors={errors}'
    )


@pytest.mark.parametrize('kwargs', [
    {'beta': -0.1},
    {'gamma': -0.1},
    {'beta': 1.1},
    {'gamma': 1.1},
])
def test_newmark_rejects_invalid_beta_or_gamma(kwargs):
    prob = testing.PROBLEMS['oscillator']()
    with pytest.raises(ValueError, match='beta|gamma'):
        getIntegrator('Newmark')(prob.initial(), dt=0.1, f=prob.rhs, **kwargs)


# --------------------------------------------------------------------------- #
# Tableau consistency (mirrors test_embedded.py's check for the explicit ones) #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', [
    'backwardEuler', 'implicitMidpoint', 'trapezoidal', 'SDIRK2', 'TRBDF2',
    'ESDIRK324L2SA', 'ESDIRK436L2SA',
])
def test_dirk_tableau_row_sums_match_c(name):
    tab = getDIRKTableau(name)
    for row, c in zip(tab.a, tab.c):
        assert row.sum() == pytest.approx(c, abs=1e-13)
    weights = tab.b[0] if isinstance(tab.b, tuple) else tab.b
    assert weights.sum() == pytest.approx(1.0, abs=1e-13)


def test_sdirk2_is_l_stable_by_construction():
    """The defining relation for this parametrization: gamma solves gamma^2-2gamma+1/2=0."""
    tab = getDIRKTableau('SDIRK2')
    gamma = tab.a[0, 0]
    assert gamma ** 2 - 2 * gamma + 0.5 == pytest.approx(0.0, abs=1e-13)


def test_trbdf2_satisfies_the_second_order_condition_and_is_stiffly_accurate():
    tab = getDIRKTableau('TRBDF2')
    b_main, b_embedded = tab.b
    assert b_main @ tab.c == pytest.approx(0.5, abs=1e-13)
    assert tab.a[-1] == pytest.approx(b_main, abs=1e-13)


def test_trbdf2_embedded_pair_is_the_published_third_order_pair():
    """The embedded weights are SUNDIALS ARKODE's ARKODE_TRBDF2_3_3_2 published
    (2, 3) pair, so the estimate is O(dt^3) -- the propagated branch's own true
    local error. The three identities are the branch's order-3 conditions; the old
    (0, 0, 1) backward-Euler branch, for comparison, satisfies the first two but
    gives d.c^2 = 1, not 1/3."""
    tab = getDIRKTableau('TRBDF2')
    b_main, b_embedded = tab.b
    gamma = 2.0 - np.sqrt(2.0)
    expected = np.array([
        (1.0 - np.sqrt(2.0) / 4.0) / 3.0,
        (1.0 + 3.0 * np.sqrt(2.0) / 4.0) / 3.0,
        gamma / 6.0,
    ])
    assert b_embedded == pytest.approx(expected, abs=1e-15)
    assert b_embedded.sum() == pytest.approx(1.0, abs=1e-15)
    assert b_embedded @ tab.c == pytest.approx(0.5, abs=1e-15)
    assert b_embedded @ tab.c ** 2 == pytest.approx(1.0 / 3.0, abs=1e-15)


@pytest.mark.parametrize('name', ['trapezoidal', 'ESDIRK324L2SA', 'ESDIRK436L2SA'])
def test_esdirk_like_tableaus_have_an_explicit_first_stage(name):
    """a[0,0] == 0: exercises the DIRK driver's non-Picard branch on a real
    tableau. For the ESDIRKs this is the 'E' in the name -- the first stage
    evaluates f explicitly and needs no solve."""
    tab = getDIRKTableau(name)
    assert tab.a[0, 0] == 0.0
    assert tab.c[0] == 0.0


# --------------------------------------------------------------------------- #
# Convergence order (empirical, matching NOTES.md S3.1's own verified table)   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', DIRK_SCHEMES)
@pytest.mark.parametrize('problem_name', ['oscillator', 'forced', 'damped'])
def test_dirk_scheme_reaches_its_claimed_order(name, problem_name):
    s = getIntegrator(name)
    dts = testing.default_step_sizes(0.1, 5)
    order, errors = testing.convergence(s, testing.PROBLEMS[problem_name](), dts, T=2.0)
    assert order == pytest.approx(s.order, abs=0.15), (
        f'{name} on {problem_name}: measured order {order:.3f}, claimed {s.order}. '
        f'errors={errors}'
    )


# --------------------------------------------------------------------------- #
# Picard solver behaviour (NOTES.md S3.2, S3.4)                               #
# --------------------------------------------------------------------------- #

def test_picard_diverges_on_a_stiff_problem_regardless_of_tableau_stability():
    """L-stability is a property of the *exact* method, not of a truncated Picard solve.

    NOTES.md S3.2 measured this for backward Euler specifically with Picard(20): a
    bounded answer up to dt*omega=1, then explosive, iteration-count-dependent
    divergence past it -- 3.7e+239 at dt*omega=10 -- even though the exact method is
    unconditionally stable. Reproduced directly: at this stiffness (dt*omega=100) the
    per-iteration blowup is already visible growing with iteration count (10^4 at 2
    iterations, 10^39 at 20), the signature of a divergent fixed-point map, not of
    "wrong but settling down". The shipped 2-iteration default does not run enough
    iterations to reach an astronomical value by itself -- it just returns something
    finite and badly wrong -- which is exactly why the ladder in NOTES.md S3.4 treats
    Picard as only the non-stiff rung, not something to run more iterations of here.
    """
    k = 1e6  # dt*omega = 0.1 * 1000 = 100, well past the Picard breakdown point (S3.2: ~1)
    prob = testing.PROBLEMS['oscillator'](k=k)
    tab = getDIRKTableau('backwardEuler')

    xs = []
    for iters in [2, 5, 10, 20]:
        result = DIRK(prob.initial(), dt=0.1, f=prob.rhs, tableau=tab,
                      solver=FixedPointSolver(),
                      solver_opts={'iterations': iters})
        xs.append(abs(float(get_reference_state(result.state).x[0])))

    assert all(b > a * 100 for a, b in zip(xs, xs[1:])), (
        f'expected each extra batch of Picard iterations to blow the answer up further '
        f'at this stiffness, got |x| sequence {xs}'
    )
    assert xs[-1] > 1e30, f'expected Picard(20) to be astronomically wrong here, got |x|={xs[-1]:.3e}'


def test_more_picard_iterations_recovers_the_converged_solution():
    """A converged Picard solve should approach the same answer regardless of dt*a_ii*L.

    Non-stiff regime (small k), where Picard converges at any iteration count -- more
    iterations should move the answer *less*, i.e. the sequence should be Cauchy.
    """
    prob = testing.PROBLEMS['oscillator']()
    tab = getDIRKTableau('implicitMidpoint')
    xs = []
    for iters in [2, 4, 8, 16, 32]:
        solver = FixedPointSolver(iterations=iters)
        result = DIRK(prob.initial(), dt=0.1, f=prob.rhs, tableau=tab, solver=solver)
        xs.append(float(get_reference_state(result.state).x[0]))
    diffs = [abs(a - b) for a, b in zip(xs, xs[1:])]
    assert all(d2 <= d1 * 0.5 + 1e-14 for d1, d2 in zip(diffs, diffs[1:])), (
        f'Picard iterates did not converge monotonically: x sequence {xs}'
    )


def test_implicit_midpoint_energy_drift_bound_recovers_with_more_iterations():
    """The finding behind integration.py registering Implicit Midpoint dissipation=True.

    At the shipped 2-iteration default the energy error grows with T (confirmed by
    test_hamiltonian.py's dissipation-flag test); solving the stage equation to
    near-machine-precision recovers the textbook symplectic bound.
    """
    prob = testing.PROBLEMS['oscillator']()
    tab = getDIRKTableau('implicitMidpoint')

    def scheme(iterations):
        solver = FixedPointSolver(iterations=iterations)
        def s(state, dt, f, *args, **kwargs):
            return DIRK(state, dt, f, tab, *args, solver=solver, **kwargs)
        return s

    short_2 = testing.max_energy_drift(scheme(2), prob, dt=0.05, T=20)
    long_2 = testing.max_energy_drift(scheme(2), prob, dt=0.05, T=160)
    assert long_2 / short_2 > 2.0, 'expected the 2-iteration default to drift secularly'

    short_16 = testing.max_energy_drift(scheme(16), prob, dt=0.05, T=20)
    long_16 = testing.max_energy_drift(scheme(16), prob, dt=0.05, T=160)
    assert long_16 < 1e-9, f'expected a near-machine-precision bound, got {long_16:.3e}'
    assert long_16 / short_16 < 20, (
        f'expected the bound to stay flat with T once converged, got a {long_16/short_16:.1f}x growth'
    )


def test_fixed_point_iteration_count_is_configurable_per_call():
    """solver_opts={'iterations': N} overrides an explicitly selected Picard solver."""
    prob = testing.PROBLEMS['oscillator']()
    tab = getDIRKTableau('implicitMidpoint')
    r_default = DIRK(prob.initial(), dt=0.1, f=prob.rhs, tableau=tab, solver=FixedPointSolver())
    r_more = DIRK(prob.initial(), dt=0.1, f=prob.rhs, tableau=tab,
                  solver=FixedPointSolver(), solver_opts={'iterations': 20})
    x_default = float(get_reference_state(r_default.state).x[0])
    x_more = float(get_reference_state(r_more.state).x[0])
    assert x_default != pytest.approx(x_more, abs=1e-12), (
        'expected more Picard iterations to change the (non-stiff but nonzero-residual) answer'
    )


def test_dirk_defaults_to_jfnk_on_a_stiff_backward_euler_stage():
    """The registered default must solve the stage that fixed-count Picard cannot."""
    k = 1e6
    dt = 0.1
    prob = testing.PROBLEMS['oscillator'](k=k)
    result = DIRK(prob.initial(), dt=dt, f=prob.rhs, tableau=getDIRKTableau('backwardEuler'))
    expected = 1.0 / (1.0 + dt ** 2 * k)
    assert float(get_reference_state(result.state).x[0]) == pytest.approx(expected, rel=1e-4)
    assert result.solver_diagnostics[0].termination in {'tolerance', 'stagnation'}


# --------------------------------------------------------------------------- #
# priorStep / history                                                         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', DIRK_SCHEMES)
def test_dirk_rejects_prior_step_with_a_warning(name):
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    first = s(prob.initial(), dt=0.1, f=prob.rhs)
    with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
        s(prob.initial(), dt=0.1, f=prob.rhs, priorStep=first.stages[-1])


@pytest.mark.parametrize('name', DIRK_SCHEMES)
def test_dirk_history_is_bookkeeping_only(name):
    """Same guarantee as the RK path: history= must not silently enable reuse."""
    from warpSPHIntegrators import StepHistory

    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    plain = s(prob.initial(), dt=0.1, f=prob.rhs).state
    with_history = s(prob.initial(), dt=0.1, f=prob.rhs, history=StepHistory(maxlen=2)).state
    assert torch.equal(get_reference_state(plain).x, get_reference_state(with_history).x)


@pytest.mark.parametrize('name', DIRK_SCHEMES)
def test_dirk_scheme_does_not_mutate_the_caller_state(name):
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    before_x = system.state.x.clone()
    s(system, dt=0.1, f=prob.rhs)
    assert torch.equal(system.state.x, before_x), f'{name} mutated the caller position'
