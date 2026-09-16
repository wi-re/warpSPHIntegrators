"""TVD / SSP classification of the registered schemes (NOTES.md S3.11).

The TVD/SSP-named schemes advertise a property nothing else in the test suite
measured: that one step does not increase total variation (and, for the SSP
ones, is a convex combination of shifts). `tvd_analysis` measures that on the
model hyperbolic problems (`testing.advection_problem`, `testing.burgers_problem`)
for every registered scheme. This file pins the load-bearing numbers:

* the SSP (convex-combination) coefficient of the TVD-named schemes is the
  published r = 1, and the landmark coefficients of the rest of the registry
  (RK4's 2/3, Cash-Karp's 5/12, Nystrom/Dormand-Prince's 0 -- their stage maps
  leave the convex hull for every resolvable CFL);
* the hand-written TVD RK2 / TVD RK3 drivers agree with their algebraically
  equivalent Butcher tableaus (the classifier's tableaus are pinned to the
  actual registered drivers, not to the names);
* the TVD-named schemes do not increase total variation at CFL 1 and preserve
  positivity of a non-negative initial condition;
* the stage-level separation: classical RK3's third stage leaves the convex
  hull at CFL 1 (a circulant row entry of -1) while TVD RK3's stages stay in
  it -- per-step TV cannot see this, because all third-order RK methods share
  the same final map, which is a convex combination up to CFL 1;
* the full-registry verdict table of `tvd_analysis.classify_all`, the
  maintained fact `scripts/tvd_classifier.py` prints (re-run that script to
  refresh it).
"""

import math

import pytest
import torch

from warpSPHIntegrators import getIntegrator, get_reference_state, testing
from warpSPHIntegrators import tvd_analysis
from warpSPHIntegrators.butcher import getButcherTableau
from warpSPHIntegrators.dirk import getDIRKTableau
from warpSPHIntegrators.integration import IntegrationSchemes
from warpSPHIntegrators.testing import ParticleState, ParticleSystem

N = 64
DT_SCALE = 1.0 / N  # h / c for the advection problem (L = 1, c = 1)
TV_TOL = 1e-8       # relative per-step TV-increase tolerance (of the initial TV)


def advection(ic, n=N):
    return testing.PROBLEMS['advection'](n=n, ic=ic)


def upwind(u, h):
    """The advection semi-discretisation as a plain tensor function (c = 1)."""
    return (torch.roll(u, 1) - u) / h


def make_system(u0):
    """A particle system carrying the field `u0` in `x` (testing.py's state)."""
    x = u0.clone() if torch.is_tensor(u0) else torch.tensor(u0, dtype=torch.float64)
    n = len(u0)
    return ParticleSystem(
        state=ParticleState(
            x=x,
            u=torch.zeros(n, dtype=torch.float64),
            e=torch.zeros(n, dtype=torch.float64),
            m=torch.ones(n, dtype=torch.float64)),
        t=0.0)


# --------------------------------------------------------------------------- #
# SSP (convex-combination) coefficients                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('tableau_name', ['TVD_RK2_TABLEAU', 'TVD_RK3_TABLEAU', 'SSPRK3'])
def test_named_tvd_schemes_have_ssp_coefficient_one(tableau_name):
    """The roadmap's Phase 13 gate: the measured SSP coefficient of the
    TVD-named schemes is the published r = 1."""
    if tableau_name.startswith('TVD_'):
        tableau = getattr(tvd_analysis, tableau_name)
    else:
        tableau = getButcherTableau(tableau_name)
    r = tvd_analysis.convex_combination_cfl(tableau)
    assert r == pytest.approx(1.0, abs=2e-3), (
        f'{tableau_name}: measured SSP coefficient {r:.4g}, expected 1')


@pytest.mark.parametrize('name,expected', [
    ('RK3', 0.5),
    ('RK4', 2.0 / 3.0),
    ('RK4alt', 1.0 / 3.0),
    ('BogackiShampine', 1.0),
    ('CashKarp', 5.0 / 12.0),
    ('Nystrom5', 0.0),
    ('DormandPrince', 0.0),
])
def test_explicit_ssp_coefficients(name, expected):
    r = tvd_analysis.convex_combination_cfl(getButcherTableau(name))
    assert r == pytest.approx(expected, abs=2e-3), (
        f'{name}: measured SSP coefficient {r:.4g}, expected {expected:g}')


@pytest.mark.parametrize('name,expected', [
    ('backwardEuler', 4.0),          # mu_max: convex up to at least 4
    ('implicitMidpoint', 2.0),
    ('trapezoidal', 2.0),
    ('SDIRK2', 1.0 + math.sqrt(2.0)),
    ('TRBDF2', 1.0 + math.sqrt(2.0)),
    ('ESDIRK324L2SA', 0.0),
    ('ESDIRK436L2SA', 0.0),
])
def test_implicit_ssp_coefficients(name, expected):
    r = tvd_analysis.convex_combination_cfl(getDIRKTableau(name))
    assert r == pytest.approx(expected, abs=2e-3), (
        f'{name}: measured SSP coefficient {r:.4g}, expected {expected:g}')


# --------------------------------------------------------------------------- #
# The hand-written TVD drivers vs. their Butcher equivalents                   #
# --------------------------------------------------------------------------- #

def test_tvd_rk_drivers_match_their_tableaus():
    """The classifier measures TVD RK2 / TVD RK3 through the tableaus in
    `tvd_analysis`; this pins those tableaus to the actual registered drivers:
    final state and every stage update must agree with a manual Butcher
    computation on the advection semi-discretisation."""
    prob = advection('mode', n=32)
    h = 1.0 / 32
    dt = 2.0 * DT_SCALE
    system = prob.initial()
    u0 = get_reference_state(system).x.clone()

    def manual(tableau):
        a = torch.as_tensor(tableau.a, dtype=torch.float64)
        b = torch.as_tensor(
            tableau.b[0] if isinstance(tableau.b, tuple) else tableau.b,
            dtype=torch.float64)
        ks, ys = [], []
        for j in range(len(tableau.c)):
            Y = u0.clone()
            for i in range(j):
                Y = Y + a[j, i] * dt * ks[i]
            ks.append(upwind(Y, h))
            ys.append(Y)
        uplus = u0.clone()
        for j in range(len(tableau.c)):
            uplus = uplus + b[j] * dt * ks[j]
        return ys, ks, uplus

    for name, tableau in [('TVD RK2', tvd_analysis.TVD_RK2_TABLEAU),
                          ('TVD RK3', tvd_analysis.TVD_RK3_TABLEAU)]:
        scheme = getIntegrator(name)
        result = scheme(system, dt=dt, f=prob.rhs)
        ys, ks, uplus = manual(tableau)
        assert len(result.stages) == len(ks), f'{name}: stage count'
        for j, (stage, k) in enumerate(zip(result.stages, ks)):
            err = (stage.update.dxdt - k).abs().max()
            assert err < 1e-12, f'{name}: stage {j + 1} update differs ({err:.3e})'
        err = (get_reference_state(result.state).x - uplus).abs().max()
        assert err < 1e-12, f'{name}: final state differs ({err:.3e})'


# --------------------------------------------------------------------------- #
# TVD-named schemes: total variation and positivity at CFL 1                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', ['TVD RK2', 'TVD RK3'])
def test_tvd_schemes_do_not_increase_total_variation_at_cfl_one(name):
    """Per-step TV increases of the registered driver on the step initial
    condition, at exactly CFL 1 (one-step schemes: no burn-in needed)."""
    scheme = getIntegrator(name)
    prob = advection('step')
    tv0 = tvd_analysis.total_variation(prob.initial().state.x)
    increases = tvd_analysis.tv_per_step_increases(
        scheme, prob, dt=1.0 * DT_SCALE, n_steps=25, burn_in=0)
    worst = max(increases) / tv0
    assert worst <= TV_TOL, (
        f'{name} at CFL 1: per-step TV increase {worst:.3e} * TV0 exceeds {TV_TOL:g}')


def test_tvd_rk3_preserves_positivity_at_cfl_one():
    """A non-negative initial condition (half step) stays non-negative under
    TVD RK3 at CFL 1: a convex-combination step maps non-negative data to
    non-negative data."""
    scheme = getIntegrator('TVD RK3')
    prob = advection('step')
    u0 = torch.zeros(N, dtype=torch.float64)
    u0[:N // 2] = 1.0
    system = make_system(u0)
    for _ in range(10):
        system = scheme(system, dt=1.0 * DT_SCALE, f=prob.rhs).state
        u = get_reference_state(system).x
        assert u.min() >= -1e-12, (
            f'TVD RK3 at CFL 1: positivity violated (min {u.min():.3e})')


def test_burgers_tvd_rk3_does_not_increase_tv_at_cfl_one():
    """The nonlinear hyperbolic problem: TVD RK3 on Burgers at CFL 1
    (alpha = max|u0| = 1) does not increase total variation per step."""
    scheme = getIntegrator('TVD RK3')
    prob = testing.PROBLEMS['burgers'](n=N)
    h = 1.0 / N
    tv0 = tvd_analysis.total_variation(prob.initial().state.x)
    increases = tvd_analysis.tv_per_step_increases(
        scheme, prob, dt=1.0 * h, n_steps=25, burn_in=0)
    worst = max(increases) / tv0
    assert worst <= TV_TOL, (
        f'TVD RK3 on Burgers at CFL 1: per-step TV increase {worst:.3e} * TV0')


# --------------------------------------------------------------------------- #
# The stage-level separation: RK3 vs. TVD RK3 at CFL 1                         #
# --------------------------------------------------------------------------- #

def test_classical_rk3_stage_leaves_the_convex_hull():
    """Per-step TV cannot separate classical RK3 from TVD RK3 at CFL 1 -- all
    third-order RK methods share the same final map, a convex combination up
    to CFL 1. The separation is at the stage level: run the registered drivers
    from a delta initial condition (the stage states ARE the circulant rows of
    the stage maps) and check convexity of each stage state.

    RK3's stage-3 row at CFL 1 is [1, -1, 1] (entry mu(1 - 2 mu) = -1);
    TVD RK3's stage rows are [0, 1] and [3/4, 0, 1/4] -- both convex.
    """
    u0 = torch.zeros(N, dtype=torch.float64)
    u0[0] = 1.0
    dt = 1.0 * DT_SCALE  # CFL 1
    prob = advection('step')

    def stage_states(scheme, a_last):
        """Reconstruct the stage states from the driver's stage updates:
        Y_{j+1} = y + dt * sum_{i < j} a_{ji} k_i."""
        result = scheme(make_system(u0), dt=dt, f=prob.rhs)
        ks = [stage.update.dxdt for stage in result.stages]
        states = []
        for j in range(len(ks)):
            Y = u0.clone()
            for i in range(j):
                Y = Y + a_last[j][i] * dt * ks[i]
            states.append(Y)
        return states, get_reference_state(result.state).x

    # Classical RK3: stage 3 of the tableau is a[2] = (-1, 2, 0).
    states, _ = stage_states(
        getIntegrator('RK3'), [[], [1.0 / 2.0], [-1.0, 2.0]])
    y3 = states[2]
    expected = torch.zeros(N, dtype=torch.float64)
    expected[0], expected[1], expected[2] = 1.0, -1.0, 1.0
    assert (y3 - expected).abs().max() < 1e-12, (
        f'RK3 stage-3 state from delta IC: expected row [1, -1, 1], '
        f'got min {y3.min():.3e} at {(y3.argmin().item())}')
    assert y3.min() < -0.5, (
        f'RK3 stage-3 state at CFL 1 should leave the convex hull, '
        f'got min {y3.min():.3e}')

    # TVD RK3: stage rows of its Butcher equivalent, a[1] = (1, 0),
    # a[2] = (1/4, 1/4). Every stage state and the final state is a convex
    # combination of the (non-negative) initial condition.
    states, final = stage_states(
        getIntegrator('TVD RK3'), [[], [1.0], [0.25, 0.25]])
    for j, Y in enumerate(states + [final]):
        assert Y.min() >= -1e-12, (
            f'TVD RK3 stage {j + 1} state at CFL 1 left the convex hull '
            f'(min {Y.min():.3e})')
        assert abs(Y.sum() - 1.0) <= 1e-12, (
            f'TVD RK3 stage {j + 1} state lost the row sum (sum {Y.sum():.3e})')


# --------------------------------------------------------------------------- #
# Unconditional TVD at large CFL                                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('name', ['Backward Euler (implicit)', 'BDF1'])
def test_backward_euler_families_are_tv_to_large_cfl(name):
    """BE (and BDF1 = BE) is the Fejer kernel: a convex combination of shifts
    at every CFL. The trajectory sweep must pass at the top of the grid."""
    scheme = getIntegrator(name)
    prob = advection('step')
    measurement = tvd_analysis.measure_tvd_cfl(
        scheme, prob, cfls=[1.0, 2.0, 5.0], dt_scale=DT_SCALE)
    assert measurement.unconditional, (
        f'{name}: expected TVD up to CFL 5, got {measurement.detail}')


# --------------------------------------------------------------------------- #
# The full-registry verdict table (the maintained fact of NOTES.md S3.11)      #
# --------------------------------------------------------------------------- #

#: (ssp_cfl, tvd_cfl, unconditional) as measured by the Phase 13 sweep
#: (n = 64, step IC, CFL grid 0.25..5, burn-in 10, window 40, tol 1e-8,
#: convex-combination scan to CFL 4). None: the scheme exposes no tableau
#: (ssp) or the TVD question does not apply (tvd, second-order-system schemes).
#: ssp values at the scan cap (4.0) are lower bounds.
EXPECTED_VERDICTS = {
    'Forward Euler': (1.0, 1.0, False),
    'Midpoint': (1.0, 1.0, False),
    "Heun's Method (2nd order)": (1.0, 1.0, False),
    "Ralston's Method (2nd order)": (1.0, 1.0, False),
    'RK3': (0.5, 1.0, False),
    "Heun's Method (3rd order)": (1.0, 1.0, False),
    "Ralston's Method (3rd order)": (1.0, 1.0, False),
    "Wray's Method (3rd order)": (1.0, 1.0, False),
    'SSP RK3': (1.0, 1.0, False),
    'RK4': (2.0 / 3.0, 1.25, False),
    'RK4 (alternative)': (1.0 / 3.0, 1.25, False),
    'Nystrom 5th order': (0.0, 1.5, False),
    'Bogacki-Shampine 3(2)': (1.0, 1.0, False),
    'Dormand-Prince 5(4)': (0.0, 1.5, False),
    'Cash-Karp 5(4)': (5.0 / 12.0, 1.5, False),
    'Leap Frog': (None, None, False),
    'Symplectic Euler': (None, None, False),
    'Velocity Verlet': (None, None, False),
    'PEFRL': (None, None, False),
    'VEFRL': (None, None, False),
    'EPEC': (1.0, 1.0, False),
    'EPEC Modified': (1.0, 1.0, False),
    'TVD RK3': (1.0, 1.0, False),
    'TVD RK2': (1.0, 1.0, False),
    'Semi-Implicit Euler': (None, 5.0, True),
    'Explicit Euler': (None, 1.0, False),
    'Backward Euler (implicit)': (4.0, 5.0, True),
    'Implicit Midpoint': (2.0, 5.0, True),
    'Trapezoidal (Crank-Nicolson)': (2.0, 5.0, True),
    'SDIRK2': (1.0 + math.sqrt(2.0), 5.0, True),
    'TR-BDF2': (1.0 + math.sqrt(2.0), 5.0, True),
    'ESDIRK3(2)4L[2]SA': (0.0, 5.0, True),
    'ESDIRK4(3)6L[2]SA': (0.0, 5.0, True),
    'ARK3(2)4L[2]SA': (0.0, 5.0, True),
    'ARK4(3)6L[2]SA': (0.0, 5.0, True),
    'Newmark': (None, None, False),
    'BDF1': (None, 5.0, True),
    'BDF2': (None, 1.5, False),
    'BDF3': (None, 1.5, False),
    'BDF4': (None, 1.5, False),
    'BDF5': (None, 1.5, False),
    'IMEX Euler': (None, 5.0, True),
    'Adams-Bashforth 2': (None, 1.5, False),
    'Adams-Bashforth 3': (None, 1.5, False),
    'Adams-Bashforth 4': (None, 1.5, False),
    'Adams-Bashforth 5': (None, 1.5, False),
    'Adams-Bashforth-Moulton 2 (PECE)': (None, 1.5, False),
    'Adams-Bashforth-Moulton 3 (PECE)': (None, 1.5, False),
    'Adams-Bashforth-Moulton 4 (PECE)': (None, 1.5, False),
    'Adams-Moulton 2 (implicit)': (None, 1.5, False),
    'Adams-Moulton 3 (implicit)': (None, 1.5, False),
    'Adams-Moulton 4 (implicit)': (None, 1.5, False),
    'RKC1': (None, None, False),
    'RKC2': (None, None, False),
    'RKL2': (None, None, False),
    # Rosenbrock-W: no tableau (so no SSP convex-combination coefficient), but
    # A-stable, so the measured sweep finds no TV increase up to the CFL cap.
    'ROS3P': (None, 5.0, True),
    # Exponential integrator: needs the `linear` accessor of a SemilinearRHS to
    # integrate the linear part exactly, but the model advection problem is a
    # plain callable, so the TVD/SSP measurement does not apply to it here.
    'ETD2RK': (None, None, False),
    # Exponential Rosenbrock: no tableau (so no SSP coefficient), and -- unlike
    # ETD2RK -- it runs on the plain-callable advection problem (it freezes the
    # FULL right-hand-side Jacobian, no `linear` accessor needed). Advection is
    # linear, so the step is the exact per-mode exponential (L-stable: the
    # amplification is e^{-h lambda} per mode, a contraction at every CFL) and
    # the sweep finds no TV increase up to the CFL cap.
    'EXPRB32': (None, 5.0, True),
    # The coupled fully implicit block pair: no tableau attribute the SSP scan
    # reads (blockTableau is not butcher/DIRK/ARK), so no ssp_cfl. Both are
    # A-stable (|R(z)| <= 1 on the whole left half plane, |R(iy)| = 1 for
    # Gauss-Legendre), so the per-mode amplification never exceeds 1 and the
    # measured sweep finds no TV increase up to the CFL cap.
    'Gauss-Legendre 2': (None, 5.0, True),
    # Radau IIA is L-stable: |R(z)| -> 0 along the negative real axis, an even
    # stronger contraction than A-stability gives.
    'Radau IIA s=2': (None, 5.0, True),
    # The IMEX multistep family (NOTES.md S3.19): no tableau attribute the SSP
    # scan reads, so no ssp_cfl. The measured 1.5 is the shared Dormand-Prince
    # cold start's imaginary-axis boundary (the bootstrap is explicit and
    # unstable once CFL*2 >~ 3.4), NOT the backbones' TVD limits -- exactly
    # the cap the BDF/AB/AM family carries. The one-step trapezoidal DIRK
    # scheme skips the startup and reaches 5.0.
    'SBDF2': (None, 1.5, False),
    'SBDF3': (None, 1.5, False),
    'CNAB2': (None, 1.5, False),
}

CFLS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]


def test_classifier_verdicts_match_the_recorded_landscape():
    """The Phase 13 gate: a verdict is recorded for every registered scheme.
    Runs the same sweep `scripts/tvd_classifier.py` prints (~1 minute)."""
    prob = advection('step')
    verdicts = tvd_analysis.classify_all(
        IntegrationSchemes, prob, CFLS, DT_SCALE, mu_max=4.0)
    got = {v.scheme for v in verdicts}
    assert got == set(EXPECTED_VERDICTS), (
        f'registry drift: missing {set(EXPECTED_VERDICTS) - got}, '
        f'unexpected {got - set(EXPECTED_VERDICTS)}')
    failures = []
    for v in verdicts:
        ssp, tvd, uncond = EXPECTED_VERDICTS[v.scheme]
        if v.ssp_cfl is None or ssp is None:
            ssp_ok = v.ssp_cfl is None and ssp is None
        else:
            ssp_ok = abs(v.ssp_cfl - ssp) <= 2e-3
        tvd_ok = (v.tvd_cfl is None and tvd is None) or \
            (v.tvd_cfl is not None and v.tvd_cfl == tvd)
        if not (ssp_ok and tvd_ok and v.unconditional == uncond):
            failures.append(
                f'{v.scheme}: got (ssp={v.ssp_cfl}, tvd={v.tvd_cfl}, '
                f'uncond={v.unconditional}), expected (ssp={ssp}, tvd={tvd}, '
                f'uncond={uncond})')
    assert not failures, 'verdict table drift:\n' + '\n'.join(failures)
