"""DIRK driver (docs/research/integrators/notes-index.md §§3.6, 3.28): 36 tableaus and pluggable stage solves.

Most of the generic behaviour (state cloning, copied/ephemeral fields, stage times,
kwargs passthrough) is already covered for these schemes by the existing parametrized
`scheme` fixture in every other test file -- adding entries to `IntegrationSchemes`
pulled them into all of that for free. This file covers what's specific to DIRK: the
convergence order of each tableau, the Picard solver's own behaviour (stiff divergence,
iteration-count control, pluggability), and the two-tier "converged vs. shipped default"
distinction the energy-drift investigation in `integration.py`'s registration comment
turned up.
"""

import warnings

import numpy as np
import pytest
import torch

from conftest import dts_for
from warpSPHIntegrators.solvers import FixedPointSolver
from warpSPHIntegrators.state import get_reference_state
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.analysis import is_fsal, step_reuse_order, supports_step_reuse
import warpSPHIntegrators.testing as testing
from warpSPHIntegrators.drivers.implicit.dirk import DIRK
from warpSPHIntegrators.drivers.implicit.dirk import _dirk_reuses_first_stage
from warpSPHIntegrators.schemes.coefficients.implicit import getDIRKTableau
from warpSPHIntegrators.solvers import JFNKSolver
from warpSPHIntegrators.analysis.stability import rk_stability_function

DIRK_SCHEMES = [
    'Backward Euler (implicit)', 'Implicit Midpoint', 'Trapezoidal (Crank-Nicolson)',
    'SDIRK2', 'TR-BDF2', 'ESDIRK3(2)4L[2]SA', 'ESDIRK4(3)6L[2]SA',
    # Wave 8 (docs/development/plan-reconciliation-2026-10-23.md, 2026-10-13; docs/research/integrators/notes-3-28.md): the remaining
    # ODE v7.8.1 DIRK tableaus.
    'Kvaerno3', 'Kvaerno4', 'Kvaerno5', 'KenCarp3', 'KenCarp4', 'KenCarp47',
    'KenCarp5', 'KenCarp58', 'ESDIRK3(2)5L[2]SA', 'ESDIRK4(3)6L[2]SA2',
    'ESDIRK4(3)7L[2]SA', 'ESDIRK5(4)7L[2]SA2', 'ESDIRK5(4)I8L[2]SA',
    'ESDIRK6(5)9L[2]SA', 'SSPSDIRK2', 'ARS222', 'ARS232', 'ARS343', 'ARS443',
    'BHR553', 'Cash4', 'CFNLIRK3', 'Hairer4', 'Hairer42', 'SFSDIRK4', 'SFSDIRK5',
    'SFSDIRK6', 'SFSDIRK7', 'SFSDIRK8',
]

# Stiffly accurate (a[-1] == b, c[-1] == 1) WITH an explicit first stage
# (a[0,0] == 0): the previous step's last stage IS y^{n+1}, so its converged
# derivative is exactly the f(t^{n+1}, y^{n+1}) the next step's first stage needs.
# Reuse is lossless for exactly these (Phase 10; wave-8 additions per docs/research/integrators/notes-index.md
# 3.28). The no-reuse schemes either have an implicit first stage (SDIRK2,
# SSPSDIRK2, Cash4, Hairer4, Hairer42, SFSDIRK4-8) or are single-stage
# (backward Euler, implicit midpoint), so they cannot consume a reused k0 and
# reject priorStep.
DIRK_REUSE_SCHEMES = [
    'Trapezoidal (Crank-Nicolson)', 'TR-BDF2', 'ESDIRK3(2)4L[2]SA', 'ESDIRK4(3)6L[2]SA',
    'Kvaerno3', 'Kvaerno4', 'Kvaerno5', 'KenCarp3', 'KenCarp4', 'KenCarp47',
    'KenCarp5', 'KenCarp58', 'ESDIRK3(2)5L[2]SA', 'ESDIRK4(3)6L[2]SA2',
    'ESDIRK4(3)7L[2]SA', 'ESDIRK5(4)7L[2]SA2', 'ESDIRK5(4)I8L[2]SA',
    'ESDIRK6(5)9L[2]SA', 'ARS222', 'ARS232', 'ARS343', 'ARS443', 'BHR553',
    'CFNLIRK3',
]
DIRK_NO_REUSE_SCHEMES = [
    'Backward Euler (implicit)', 'Implicit Midpoint', 'SDIRK2', 'SSPSDIRK2',
    'Cash4', 'Hairer4', 'Hairer42', 'SFSDIRK4', 'SFSDIRK5', 'SFSDIRK6',
    'SFSDIRK7', 'SFSDIRK8',
]


def test_sspsdirk2_is_symplectic_midpoint_composition():
    tab = getDIRKTableau('SSPSDIRK2')
    assert np.array_equal(tab.a, [[0.25, 0.0], [0.5, 0.25]])
    assert np.array_equal(tab.b, [0.5, 0.5])
    b = tab.b
    assert np.max(np.abs(b[:, None] * tab.a + b[None, :] * tab.a.T
                         - b[:, None] * b[None, :])) == 0.0
    for z in [-100.0, -1.0, 1j, -2.0 + 3j]:
        assert rk_stability_function(tab, z) == pytest.approx(((1 + z/4) / (1 - z/4))**2)
    assert rk_stability_function(tab, -1e8) == pytest.approx(1.0, abs=2e-7)
    assert not get_scheme('SSPSDIRK2').dissipation


@pytest.mark.parametrize('name', ['ESDIRK4(3)7L[2]SA', 'ESDIRK6(5)9L[2]SA', 'SFSDIRK6'])
def test_strict_solve_recovers_kepler_order_on_standard_grid(name):
    """Distinguish stage-solver noise from tableau defects without loosening order."""
    s = get_scheme(name)
    solver = JFNKSolver(tol=1e-14, gmres_tol=1e-14, newton_tol=1e-14, fd_eps=1e-6)

    def step(state, dt, f, **kwargs):
        return s(state, dt, f, solver=solver, **kwargs)

    grid = testing.default_step_sizes()
    order, errors = testing.convergence(step, testing.kepler_problem(), grid, T=2.0)
    assert order == pytest.approx(s.order, abs=0.15), errors
    assert all(b < a for a, b in zip(errors, errors[1:]) if b >= 1e-13), errors


@pytest.fixture(scope='module')
def semilinear_reference():
    """A nonlinear semilinear vector problem; the reference is step-refined."""
    prob = testing.viscous_burgers_problem(n=16, nu=0.1)
    rk4 = get_scheme('RK4')
    coarse = get_reference_state(testing.run(rk4, prob, 0.0005, T=0.1)).x
    fine = get_reference_state(testing.run(rk4, prob, 0.00025, T=0.1)).x
    assert torch.max(torch.abs(coarse - fine)) < 1e-9
    return prob, fine


@pytest.mark.parametrize('name', DIRK_SCHEMES)
def test_dirk_converges_on_semilinear_burgers(name, semilinear_reference):
    """Every registered DIRK solves the full L*y+N problem with its default JFNK.

    Stability-limited SFSDIRK methods use the same small grid as the others;
    their stronger stiff-step limitations are pinned by the stability tests.
    """
    prob, reference = semilinear_reference
    s = get_scheme(name)
    errors = []
    for dt in [0.02, 0.01]:
        state = prob.initial()
        for _ in range(round(0.1 / dt)):
            result = s(state, dt, prob.rhs)
            state = result.state
            assert all(d is None or d.termination in {'tolerance', 'stagnation'}
                       for d in result.solver_diagnostics)
        final = get_reference_state(state).x
        assert torch.isfinite(final).all()
        errors.append(float(torch.max(torch.abs(final - reference))))
    assert errors[1] < errors[0], f'{name}: {errors}'
    assert errors[1] < 0.03, f'{name}: {errors}'


def test_newmark_reaches_second_order_on_the_oscillator():
    s = get_scheme('Newmark')
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
        get_scheme('Newmark')(prob.initial(), dt=0.1, f=prob.rhs, **kwargs)


# --------------------------------------------------------------------------- #
# Tableau consistency (mirrors test_embedded.py's check for the explicit ones) #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', [
    'backwardEuler', 'implicitMidpoint', 'trapezoidal', 'SDIRK2', 'TRBDF2',
    'ESDIRK324L2SA', 'ESDIRK436L2SA',
    'Kvaerno3', 'Kvaerno4', 'Kvaerno5', 'KenCarp3', 'KenCarp4', 'KenCarp47',
    'KenCarp5', 'KenCarp58', 'ESDIRK325L2SA', 'ESDIRK436L2SA2', 'ESDIRK437L2SA',
    'ESDIRK547L2SA2', 'ESDIRK54I8L2SA', 'ESDIRK659L2SA', 'SSPSDIRK2', 'ARS222',
    'ARS232', 'ARS343', 'ARS443', 'BHR553', 'Cash4', 'CFNLIRK3', 'Hairer4',
    'Hairer42', 'SFSDIRK4', 'SFSDIRK5', 'SFSDIRK6', 'SFSDIRK7', 'SFSDIRK8',
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


@pytest.mark.parametrize('name', [
    'trapezoidal', 'ESDIRK324L2SA', 'ESDIRK436L2SA',
    'Kvaerno3', 'Kvaerno4', 'Kvaerno5', 'KenCarp3', 'KenCarp4', 'KenCarp47',
    'KenCarp5', 'KenCarp58', 'ESDIRK325L2SA', 'ESDIRK436L2SA2', 'ESDIRK437L2SA',
    'ESDIRK547L2SA2', 'ESDIRK54I8L2SA', 'ESDIRK659L2SA', 'ARS222', 'ARS232',
    'ARS343', 'ARS443', 'BHR553', 'CFNLIRK3',
])
def test_esdirk_like_tableaus_have_an_explicit_first_stage(name):
    """a[0,0] == 0: exercises the DIRK driver's non-Picard branch on a real
    tableau. For the ESDIRKs this is the 'E' in the name -- the first stage
    evaluates f explicitly and needs no solve. The wave-8 Kvaerno, KenCarp,
    ARS, BHR and CFNLIRK tableaus open the same way (docs/research/integrators/notes-3-28.md)."""
    tab = getDIRKTableau(name)
    assert tab.a[0, 0] == 0.0
    assert tab.c[0] == 0.0


# --------------------------------------------------------------------------- #
# Convergence order (empirical, matching docs/research/integrators/notes-3-1.md's own verified table)   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', DIRK_SCHEMES)
@pytest.mark.parametrize('problem_name', ['oscillator', 'forced', 'damped'])
def test_dirk_scheme_reaches_its_claimed_order(name, problem_name):
    s = get_scheme(name)
    # ESDIRK6(5)9L[2]SA's standard-grid errors sit at the roundoff floor on
    # oscillator/forced (docs/research/integrators/notes-3-28.md); dts_for gives it the coarser grid.
    dts = dts_for(name, problem_name, testing.default_step_sizes(0.1, 5))
    order, errors = testing.convergence(s, testing.PROBLEMS[problem_name](), dts, T=2.0)
    if (name, problem_name) == ('ESDIRK4(3)7L[2]SA', 'forced'):
        # Superconvergent cell: this (scheme, problem) pair measures ~5.0 on
        # EVERY grid above the roundoff floor, so the two-sided band can never
        # be met -- the order-4 error term nearly vanishes on this problem and
        # a higher-order term dominates (docs/research/integrators/notes-3-28.md, the same phenomenon as
        # BS5 on forced, which test_convergence.py treats one-sided). Assert
        # the one-sided lower bound instead.
        assert order is not None and order >= s.order - 0.15, (
            f'{name} on {problem_name}: measured order {order}, claimed {s.order}. '
            f'errors={errors}'
        )
        return
    assert order == pytest.approx(s.order, abs=0.15), (
        f'{name} on {problem_name}: measured order {order:.3f}, claimed {s.order}. '
        f'errors={errors}'
    )


# --------------------------------------------------------------------------- #
# Picard solver behaviour (docs/research/integrators/notes-3-2.md, S3.4)                               #
# --------------------------------------------------------------------------- #

def test_picard_diverges_on_a_stiff_problem_regardless_of_tableau_stability():
    """L-stability is a property of the *exact* method, not of a truncated Picard solve.

    docs/research/integrators/notes-3-2.md measured this for backward Euler specifically with Picard(20): a
    bounded answer up to dt*omega=1, then explosive, iteration-count-dependent
    divergence past it -- 3.7e+239 at dt*omega=10 -- even though the exact method is
    unconditionally stable. Reproduced directly: at this stiffness (dt*omega=100) the
    per-iteration blowup is already visible growing with iteration count (10^4 at 2
    iterations, 10^39 at 20), the signature of a divergent fixed-point map, not of
    "wrong but settling down". The shipped 2-iteration default does not run enough
    iterations to reach an astronomical value by itself -- it just returns something
    finite and badly wrong -- which is exactly why the ladder in docs/research/integrators/notes-3-4.md treats
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

@pytest.mark.parametrize('name', DIRK_NO_REUSE_SCHEMES)
def test_dirk_rejects_prior_step_with_a_warning(name):
    """Implicit first stage (SDIRK2) or single-stage (BE / midpoint): no reuse."""
    s = get_scheme(name)
    prob = testing.PROBLEMS['oscillator']()
    first = s(prob.initial(), dt=0.1, f=prob.rhs)
    with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
        s(prob.initial(), dt=0.1, f=prob.rhs, priorStep=first.stages[-1])


@pytest.mark.parametrize('name', DIRK_REUSE_SCHEMES)
def test_dirk_accepts_prior_step_without_a_warning(name):
    """Stiffly accurate + explicit first stage: reuse is lossless, so no warning."""
    s = get_scheme(name)
    prob = testing.PROBLEMS['oscillator']()
    first = s(prob.initial(), dt=0.1, f=prob.rhs)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        s(prob.initial(), dt=0.1, f=prob.rhs, priorStep=first.stages[-1])


@pytest.mark.parametrize('name', DIRK_REUSE_SCHEMES)
def test_dirk_reuse_preserves_the_measured_order(name, step_sizes):
    """The reuse gate: order with reuse must equal the scheme's order, not drop."""
    from conftest import ORDER_TOLERANCE, order_of
    s = get_scheme(name)
    assert step_reuse_order(s) == s.order
    assert supports_step_reuse(s) and is_fsal(s)
    for problem_name in ['oscillator', 'forced']:
        order, errors = order_of(s, problem_name, dts_for(name, problem_name, step_sizes), reuse=True)
        assert order is not None, f'{name} produced no usable errors: {errors}'
        assert order >= s.order - ORDER_TOLERANCE, (
            f'{name} on {problem_name}: reuse dropped the order to {order:.2f}, '
            f'expected {s.order}')


@pytest.mark.parametrize('name', DIRK_REUSE_SCHEMES)
def test_dirk_reuse_saves_an_explicit_evaluation_per_step(name):
    """One saved RHS per reusable step (the explicit first stage), counted exactly.

    The saving is isolated to evaluations AT the step-start time t^n: stage 0 sits
    exactly there (c[0] = 0, so the driver sets the stage time to t^n + 0*dt, which
    is t^n bit-for-bit), and no other stage of any of these tableaus shares that
    time. Without reuse, every step evaluates f at its start; with reuse, only the
    first step does, because the reused k0 IS the previous step's final-stage
    evaluation, which sits at t^n since every one of these schemes is stiffly
    accurate. Counting total f calls instead is unreliable: JFNK's iteration count
    on the implicit stages varies by a few from step to step (docs/research/integrators/notes-3-28.md measured
    per-step savings of -7 to +3 for KenCarp47), so the total is a band, not a count.
    """
    s = get_scheme(name)
    prob = testing.PROBLEMS['oscillator']()

    def count(reuse):
        at_step_start = {'n': 0}
        t_start = float(prob.initial().t)

        def f(system, dt, *args, **kwargs):
            if float(system.t) == t_start:
                at_step_start['n'] += 1
            return prob.rhs(system, dt)

        system = prob.initial()
        prior = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for _ in range(20):
                result = s(system, dt=0.05, f=f, priorStep=prior if reuse else None)
                system, prior = result.state, result.stages[-1]
                t_start = float(result.state.t)
        return at_step_start['n']

    saved = count(False) - count(True)
    # 20 steps; the first has no prior in either mode. No-reuse therefore counts
    # 20 step-start evaluations, reuse counts 1, and the difference is exactly the
    # 19 reusable steps' explicit first stages.
    assert saved == 19, (
        f'{name}: saved {saved} step-start evaluations over 19 reusable steps, '
        f'expected exactly 19 (one explicit stage per step)')


@pytest.mark.parametrize('name', DIRK_SCHEMES)
def test_dirk_stiffly_accurate_metadata_matches_the_tableau(name):
    """`stiffly_accurate` is now read (reuse.dirk_reuse_analysis), so pin it to the tableau.

    The driver decides reuse from the tableau alone (`_dirk_reuses_first_stage`); the
    analysis gates on the registered `stiffly_accurate` flag. They must agree, or the
    driver would accept a priorStep the analysis (and the registration warning) says is
    ignored -- a silent divergence. This test is what makes the metadata a real consumer
    rather than an unverified flag.
    """
    s = get_scheme(name)
    # The tableau the scheme actually carries (same object the driver decides reuse
    # from), not a re-lookup by enum name -- the enum names are lowercase and are not
    # `getDIRKTableau`'s keys.
    tab = s.function.dirkTableau
    a = np.asarray(tab.a, dtype=float)
    c = np.asarray(tab.c, dtype=float)
    b = np.asarray(tab.b[0] if isinstance(tab.b, tuple) else tab.b, dtype=float)
    tableau_sa = abs(c[-1] - 1.0) < 1e-12 and np.allclose(a[-1], b, atol=1e-12)
    assert s.stiffly_accurate == tableau_sa, (
        f'{name}: registered stiffly_accurate={s.stiffly_accurate} but the tableau says '
        f'{tableau_sa} (a[-1] == b and c[-1] == 1)')

    # And the driver's call-time predicate agrees with the analysis for this scheme.
    explicit_first = abs(a[0, 0]) < 1e-12 and abs(c[0]) < 1e-12
    assert _dirk_reuses_first_stage(tab) == bool(explicit_first and s.stiffly_accurate), (
        f'{name}: driver reuse predicate disagrees with the registered metadata')


@pytest.mark.parametrize('name', DIRK_SCHEMES)
def test_dirk_history_is_bookkeeping_only(name):
    """Same guarantee as the RK path: history= must not silently enable reuse."""
    from warpSPHIntegrators.state import StepHistory

    s = get_scheme(name)
    prob = testing.PROBLEMS['oscillator']()
    plain = s(prob.initial(), dt=0.1, f=prob.rhs).state
    with_history = s(prob.initial(), dt=0.1, f=prob.rhs, history=StepHistory(maxlen=2)).state
    assert torch.equal(get_reference_state(plain).x, get_reference_state(with_history).x)


@pytest.mark.parametrize('name', DIRK_SCHEMES)
def test_dirk_scheme_does_not_mutate_the_caller_state(name):
    s = get_scheme(name)
    prob = testing.PROBLEMS['oscillator']()
    system = prob.initial()
    before_x = system.state.x.clone()
    s(system, dt=0.1, f=prob.rhs)
    assert torch.equal(system.state.x, before_x), f'{name} mutated the caller position'
