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

import numpy as np
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
    # Shu's 10-stage order-4 SSP method: the true SSP coefficient is 6, beyond
    # the scan's default cap, so it is pinned with an explicit mu_max in
    # test_ssprk104_ssp_boundary_and_real_axis_interval.
    ('SSPRK104', 4.0),
    # The wave-1 ODE SSPRK tableaus (SCHEME_EXPANSION_PLAN, 2026-10-12): the
    # library's upwind-model all-stage convex-combination CFL at the default
    # scan cap 4.0. SSPRK53/63/73/83 agree with ODE's published
    # ssp_coefficient (2.65/3.518/4.2879/5.107) to <= 6.3e-4.
    ('SSPRK53', 2.65),
    ('SSPRK63', 3.518),
    # True SSP boundary 4.2879 is beyond the scan cap, so the scan reports
    # the cap (a lower bound); the exact boundary is pinned in
    # test_expanded_ssprk_ssp_boundaries_and_real_axis_intervals.
    ('SSPRK73', 4.0),
    ('SSPRK83', 4.0),
    # ODE's published 1.508 is the paper's coefficient; the library's
    # all-stage measurement is 1.7693.
    ('SSPRK54', 1.7693),
    # ODE's published 2.459 matches the final-map-only convexity CFL (2.4595);
    # the stage maps bind at 2.2838 in the upwind model.
    ('KYKSSPRK42', 2.2838),
    # ODE's published 0.8417 is the paper's DG-context coefficient, not the
    # upwind-model stage-map convexity CFL (1.8939).
    ('KYK2014DGSSPRK_3S2', 1.8939),
    # Wave-2 ODE SSPRK tableaus (SCHEME_EXPANSION_PLAN, 2026-10-12): the 3(2)
    # pairs SSPRK43/SSPRK432 agree with ODE's published ssp_coefficient (2.0)
    # in the upwind model. SSPRK932's true SSP boundary 6.0 is beyond the
    # scan cap, so the scan reports the cap (a lower bound); the exact
    # boundary is pinned in
    # test_expanded_ssprk_ssp_boundaries_and_real_axis_intervals.
    ('SSPRK43', 2.0),
    ('SSPRK432', 2.0),
    ('SSPRK932', 4.0),
    # Wave-5 ODE SSPRK tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13): the
    # Higueras-Roldan low-storage third-order variants; ODE's published
    # ssp_coefficient (2.18/2.148/2.65) agrees with the library's all-stage
    # measurement (2.1808/2.1487/2.6506) to <= 8e-4.
    ('SSPRK53_2N1', 2.18),
    ('SSPRK53_2N2', 2.148),
    ('SSPRK53_H', 2.65),
    # Wave-6 ODE tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13): Tsit5 and the
    # LowOrderRK workhorses, measured on the propagated branch (the embedded
    # pairs store `b` as a tuple; the scan reads the main weights).
    ('Tsit5', 0.0),
    ('OwrenZen3', 0.7813),
    ('OwrenZen4', 0.6444),
    ('OwrenZen5', 0.0),
    ('BS5', 0.5237),
    ('Stepanov5', 0.0),
    ('Alshina2', 1.0),
    ('Alshina3', 1.0),
    ('Alshina6', 0.0),
    ('MSRK5', 0.3226),
    ('MSRK6', 0.1009),
    ('SIR54', 0.0),
    ('Ralston4', 0.7209),
    ('RKO65', 0.0),
    ('RKM', 0.7531),
    ('PSRK4p7q6', 0.0),
    ('PSRK3p6q5', 0.0),
    ('PSRK3p5q4', 0.0),
    # Wave 7 (SCHEME_EXPANSION_PLAN, 2026-10-13): the high-order Verner /
    # TsitPap / DP8 / Feagin tableaus. None is SSP in the upwind model beyond
    # the small convex-combination coefficients below (the Feagin and
    # RKV76IIa/TanYam7 stage maps leave the convex hull at every resolvable
    # CFL). DP8 has the largest of the wave (0.2371).
    ('Vern6', 0.0877),
    ('Vern7', 0.0514),
    ('Vern8', 0.1615),
    ('Vern9', 0.1272),
    ('RKV76IIa', 0.0),
    ('TanYam7', 0.0),
    ('TsitPap8', 0.1804),
    ('DP8', 0.2371),
    ('Feagin10', 0.0),
    ('Feagin12', 0.0),
    ('Feagin14', 0.0),
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


def test_ssprk104_ssp_boundary_and_real_axis_interval():
    """SSPRK(10,4)'s exact SSP boundary and negative real-axis stability limit.

    The true SSP coefficient is exactly 6: stage 2's shift-basis constant
    coefficient is 1 - mu/6, and every later stage map stays non-negative up to
    the same point (verified in exact rational arithmetic). The real-axis
    stability interval is [-13.916, 0] (OrdinaryDiffEq.jl records 13.917).
    """
    tableau = getButcherTableau('SSPRK104')
    r = tvd_analysis.convex_combination_cfl(tableau, mu_max=10.0)
    assert r == pytest.approx(6.0, abs=5e-3), (
        f'SSPRK(10,4): exact SSP boundary {r:.4g}, expected 6')

    a, b = tableau.a, tableau.b
    z = np.linspace(-14.5, 0.0, 20001)
    identity = np.eye(len(a))
    solves = np.linalg.solve(np.stack([identity - zz * a for zz in z]),
                             np.ones((len(z), len(a), 1)))
    amplification = np.abs(1.0 + z * (solves[:, :, 0] @ b))
    left = z[amplification <= 1.0 + 1e-9].min()
    assert left == pytest.approx(-13.916, abs=5e-3), (
        f'SSPRK(10,4) real-axis interval {left:.4g}, expected -13.916')


@pytest.mark.parametrize('name,ssp_boundary,real_axis_left', [
    # (name, exact SSP convex-combination boundary, real-axis interval left
    # end) as measured by tvd_analysis. The boundary is scanned to CFL 8, past
    # the default cap 4.0 that clips SSPRK73/SSPRK83/SSPRK932 in the landscape
    # table (the same cap the SSPRK(10,4) pin above uses). For every method
    # the negative real-axis interval is a single [left, 0] interval (|R(z)|
    # does not re-enter the unit disk for z < left).
    ('SSPRK53', 2.6506, -6.21361),
    ('SSPRK63', 3.5184, -7.21303),
    ('SSPRK73', 4.2879, -8.88769),
    ('SSPRK83', 5.1071, -10.98619),
    ('SSPRK54', 1.7693, -5.33147),
    ('KYKSSPRK42', 2.2838, -4.95391),
    ('KYK2014DGSSPRK_3S2', 1.8939, -4.26688),
    # Wave 2 (SCHEME_EXPANSION_PLAN, 2026-10-12): SSPRK43/SSPRK432 share the
    # same (a, c, b_main), so the same boundary and interval; SSPRK932's exact
    # boundary 6.0 is the reason this scan goes to CFL 8 (the default cap 4.0
    # would clip it).
    ('SSPRK43', 2.0, -5.14940),
    ('SSPRK432', 2.0, -5.14940),
    ('SSPRK932', 6.0, -13.28960),
    # Wave 5 (SCHEME_EXPANSION_PLAN, 2026-10-13): the Higueras-Roldan
    # low-storage variants. SSPRK53_2N2 has the largest real-axis interval of
    # the four despite the smallest SSP coefficient; SSPRK53_H matches
    # SSPRK53's interval to 1e-4 (it shares SSPRK53's first two nodes but has
    # distinct later stages).
    ('SSPRK53_2N1', 2.1808, -5.52779),
    ('SSPRK53_2N2', 2.1487, -7.26303),
    ('SSPRK53_H', 2.6506, -6.21351),
    # Wave 6 (SCHEME_EXPANSION_PLAN, 2026-10-13): the Tsit5 + LowOrderRK
    # workhorses. PSRK4p7q6 (Aubry-Chartier pseudo-symplectic) has the
    # largest real-axis interval of the wave ([-11.41, 0]).
    ('Tsit5', 0.0, -3.506),
    ('OwrenZen3', 0.7813, -2.512),
    ('OwrenZen4', 0.6444, -2.872),
    ('OwrenZen5', 0.0, -3.192),
    ('BS5', 0.5237, -3.986),
    ('Stepanov5', 0.0, -3.496),
    ('Alshina2', 1.0, -2.0),
    ('Alshina3', 1.0, -2.512),
    ('Alshina6', 0.0, -2.904),
    ('MSRK5', 0.3226, -4.366),
    ('MSRK6', 0.1009, -4.164),
    ('SIR54', 0.0, -3.512),
    ('Ralston4', 0.7209, -2.784),
    ('RKO65', 0.0, -4.164),
    ('RKM', 0.7531, -3.224),
    ('PSRK4p7q6', 0.0, -11.41),
    ('PSRK3p6q5', 0.0, -3.548),
    ('PSRK3p5q4', 0.0, -2.784),
    # Wave 7 (SCHEME_EXPANSION_PLAN, 2026-10-13): the high-order tableaus.
    # Vern8 has the widest real-axis interval of the wave ([-13.42, 0]);
    # Feagin14 the narrowest ([-1.874, 0] -- verified genuine, |R(-2.0)| = 3.51:
    # its 35-stage stability polynomial oscillates for |z| >~ 2, smaller than
    # the family trend).
    ('Vern6', 0.0877, -4.855),
    ('Vern7', 0.0514, -4.640),
    ('Vern8', 0.1615, -13.425),
    ('Vern9', 0.1272, -4.476),
    ('RKV76IIa', 0.0, -4.911),
    ('TanYam7', 0.0, -4.913),
    ('TsitPap8', 0.1804, -10.388),
    ('DP8', 0.2371, -6.394),
    ('Feagin10', 0.0, -2.528),
    ('Feagin12', 0.0, -4.070),
    ('Feagin14', 0.0, -1.874),
])
def test_expanded_ssprk_ssp_boundaries_and_real_axis_intervals(
        name, ssp_boundary, real_axis_left):
    """The wave 1/2/6 ODE SSPRK + Tsit5/LowOrderRK tableaus
    (SCHEME_EXPANSION_PLAN, 2026-10-12/13): the exact SSP convex-combination
    boundary and the negative real-axis stability limit of each, mirroring the
    SSPRK(10,4) pin above."""
    tableau = getButcherTableau(name)
    r = tvd_analysis.convex_combination_cfl(tableau, mu_max=8.0)
    assert r == pytest.approx(ssp_boundary, abs=5e-3), (
        f'{name}: SSP boundary {r:.4g}, expected {ssp_boundary:g}')

    a, b = tableau.a, tableau.b
    # Embedded pairs (wave-2 SSPRK43/SSPRK432) store `b` as a (main, embedded)
    # tuple; the real-axis interval is defined for the propagated branch.
    b = b[0] if isinstance(b, tuple) else b
    z = np.linspace(real_axis_left - 1.5, 0.0, 20001)
    identity = np.eye(len(a))
    solves = np.linalg.solve(np.stack([identity - zz * a for zz in z]),
                             np.ones((len(z), len(a), 1)))
    amplification = np.abs(1.0 + z * (solves[:, :, 0] @ b))
    left = z[amplification <= 1.0 + 1e-9].min()
    assert left == pytest.approx(real_axis_left, abs=5e-3), (
        f'{name} real-axis interval {left:.4g}, expected {real_axis_left:g}')


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
    # Shu's 10-stage order-4 SSP method: the convex-combination scan hits the
    # registered cap 4.0 (the true boundary is exactly 6 -- stage 2's constant
    # coefficient, 1 - mu/6, crosses zero there, pinned in
    # test_ssprk104_ssp_boundary_and_real_axis_interval below). The negative
    # real-axis stability interval [-13.916, 0] keeps every per-mode
    # amplification <= 1, so the sweep finds no TV increase up to the CFL cap.
    'SSPRK(10,4)': (4.0, 5.0, True),
    # The wave-1 ODE SSPRK tableaus (SCHEME_EXPANSION_PLAN, 2026-10-12): the
    # upwind-model all-stage convex-combination CFL at the scan cap 4.0.
    # SSPRK73/SSPRK83 sit beyond the cap (exact boundaries 4.2879 / 5.1071,
    # pinned in test_expanded_ssprk_ssp_boundaries_and_real_axis_intervals),
    # so their ssp values are lower bounds.
    'SSPRK53': (2.65, 2.0, False),
    'SSPRK63': (3.518, 3.0, False),
    'SSPRK73': (4.0, 3.0, False),
    'SSPRK83': (4.0, 5.0, True),
    'SSPRK54': (1.7693, 2.0, False),
    'KYKSSPRK42': (2.2838, 2.0, False),
    'KYK2014DGSSPRK_3S2': (1.8939, 1.5, False),
    # The wave-2 ODE SSPRK tableaus (SCHEME_EXPANSION_PLAN, 2026-10-12).
    # SSPRK932's ssp is the scan cap 4.0 (the exact boundary is 6.0, pinned in
    # test_expanded_ssprk_ssp_boundaries_and_real_axis_intervals); its real-axis
    # interval [-13.290, 0] keeps every per-mode amplification <= 1, so the
    # sweep finds no TV increase up to the CFL cap (unconditional).
    'SSPRK43': (2.0, 2.0, False),
    'SSPRK432': (2.0, 2.0, False),
    'SSPRK932': (4.0, 5.0, True),
    # The wave-5 ODE SSPRK tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13): the
    # Higueras-Roldan low-storage third-order variants. All three measure the
    # same per-step TVD CFL 2.0 as SSPRK53 in the upwind-model sweep, and none
    # is unconditionally TVD (the stage maps leave the convex hull past CFL 2).
    'SSPRK53_2N1': (2.18, 2.0, False),
    'SSPRK53_2N2': (2.148, 2.0, False),
    'SSPRK53_H': (2.65, 2.0, False),
    'RK4': (2.0 / 3.0, 1.25, False),
    'RK4 (alternative)': (1.0 / 3.0, 1.25, False),
    'Nystrom 5th order': (0.0, 1.5, False),
    'Bogacki-Shampine 3(2)': (1.0, 1.0, False),
    'Dormand-Prince 5(4)': (0.0, 1.5, False),
    'Cash-Karp 5(4)': (5.0 / 12.0, 1.5, False),
    # The wave-6 ODE tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13): Tsit5 and
    # the LowOrderRK workhorses. The embedded pairs' ssp values are measured
    # on the propagated (main) branch, as in test_explicit_ssp_coefficients.
    'Tsit5 5(4)': (0.0, 1.5, False),
    'OwrenZen3 3(2)': (0.7813, 1.0, False),
    'OwrenZen4 4(3)': (0.6444, 1.25, False),
    'OwrenZen5 5(4)': (0.0, 1.5, False),
    'BS5 5(4)': (0.5237, 1.5, False),
    'Stepanov5 5(4)': (0.0, 1.5, False),
    'Alshina2 2(1)': (1.0, 1.0, False),
    'Alshina3': (1.0, 1.0, False),
    'Alshina6': (0.0, 1.25, False),
    'MSRK5': (0.3226, 2.0, False),
    'MSRK6': (0.1009, 2.0, False),
    'SIR54': (0.0, 1.5, False),
    'Ralston4': (0.7209, 1.25, False),
    'RKO65': (0.0, 2.0, False),
    'RKM': (0.7531, 1.5, False),
    'PSRK4p7q6': (0.0, 2.0, False),
    'PSRK3p6q5': (0.0, 1.5, False),
    'PSRK3p5q4': (0.0, 1.25, False),
    # The wave-7 high-order tableaus (SCHEME_EXPANSION_PLAN, 2026-10-13).
    # None is unconditionally TVD: every one of them grows TV past its
    # measured per-step CFL (DP8 is the best of the wave at 3.0; Feagin14
    # the worst at 0.75).
    'Vern6 6(5)': (0.0877, 2.0, False),
    'Vern7 7(6)': (0.0514, 2.0, False),
    'Vern8 8(7)': (0.1615, 2.0, False),
    'Vern9 9(8)': (0.1272, 2.0, False),
    'RKV76IIa 7(6)': (0.0, 2.0, False),
    'TanYam7 7(6)': (0.0, 2.0, False),
    'TsitPap8 8(7)': (0.1804, 2.0, False),
    'DP8': (0.2371, 3.0, False),
    'Feagin10': (0.0, 1.0, False),
    'Feagin12': (0.0, 1.25, False),
    'Feagin14': (0.0, 0.75, False),
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
    # Wave 8: measured with the same CFL grid and scan caps below.
    'Kvaerno3': (0.0, 5.0, True),
    'Kvaerno4': (0.0, 5.0, True),
    'Kvaerno5': (0.12167340688544225, 5.0, True),
    'KenCarp3': (0.0, 5.0, True),
    'KenCarp4': (0.0, 5.0, True),
    'KenCarp47': (0.0, 5.0, True),
    'KenCarp5': (0.0, 5.0, True),
    'KenCarp58': (0.0, 5.0, True),
    'ESDIRK3(2)5L[2]SA': (0.11176233817083416, 5.0, True),
    'ESDIRK4(3)6L[2]SA2': (0.0, 5.0, True),
    'ESDIRK4(3)7L[2]SA': (0.0, 5.0, True),
    'ESDIRK5(4)7L[2]SA2': (0.0, 5.0, True),
    'ESDIRK5(4)I8L[2]SA': (0.0, 5.0, True),
    'ESDIRK6(5)9L[2]SA': (0.0, 5.0, True),
    'SSPSDIRK2': (4.0, 5.0, True),
    'ARS222': (2.4142135630766486, 5.0, True),
    'ARS232': (2.4142135630766486, 5.0, True),
    'ARS343': (0.0, 5.0, True),
    'ARS443': (0.0, 5.0, True),
    'BHR553': (0.0, 5.0, True),
    'Cash4': (0.0, 5.0, True),
    'CFNLIRK3': (0.0, 5.0, True),
    'Hairer4': (0.13667198545180206, 5.0, True),
    'Hairer42': (0.09274881117186362, 5.0, True),
    'SFSDIRK4': (4.0, 5.0, True),
    'SFSDIRK5': (4.0, 5.0, True),
    'SFSDIRK6': (4.0, 5.0, True),
    'SFSDIRK7': (4.0, 5.0, True),
    'SFSDIRK8': (4.0, 5.0, True),
    'ARK3(2)4L[2]SA': (0.0, 5.0, True),
    'ARK4(3)6L[2]SA': (0.0, 5.0, True),
    'Newmark': (None, None, False),
    'BDF1': (None, 5.0, True),
    'BDF2': (None, 1.5, False),
    # Phase 6: the cold-call classifier now measures the implicit starters,
    # not the full history-dependent BDF formula. TVD through CFL 5 is finite
    # measured coverage, not a proof of unconditional multistep TVD.
    'BDF3': (None, 5.0, True),
    'BDF4': (None, 5.0, True),
    'BDF5': (None, 5.0, True),
    'IMEX Euler': (None, 5.0, True),
    'Adams-Bashforth 2': (None, 1.5, False),
    'Adams-Bashforth 3': (None, 1.5, False),
    'Adams-Bashforth 4': (None, 1.5, False),
    'Adams-Bashforth 5': (None, 1.5, False),
    'Adams-Bashforth-Moulton 2 (PECE)': (None, 1.5, False),
    'Adams-Bashforth-Moulton 3 (PECE)': (None, 1.5, False),
    'Adams-Bashforth-Moulton 4 (PECE)': (None, 1.5, False),
    'Adams-Bashforth-Moulton 5 (PECE)': (None, 1.5, False),
    # Shu's SSP multistep (wave 3): no tableau the SSP scan reads, so no ssp_cfl.
    # The measured 1.5 is the shared Dormand-Prince cold start's limit, NOT the
    # methods' steady-state SSP bounds (C = 1/2 for SSPRKMSVS32, C = 1/3 for
    # SSPRKMSVS43) -- the trajectory sweep does not thread history, so the driver
    # re-runs its DP5 starter every step, exactly the cap the Adams/BDF/AM family
    # carries. The tight SSP coefficients are a structural property of the
    # steady-state convex combination (NOTES.md S3.23), not a trajectory limit.
    'SSPRKMSVS32': (None, 1.5, False),
    'SSPRKMSVS43': (None, 1.5, False),
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
    'Gauss-Legendre 3': (None, 5.0, True),
    'Radau IIA s=3': (None, 5.0, True),
    # The IMEX multistep family (NOTES.md S3.19): no tableau attribute the SSP
    # scan reads, so no ssp_cfl. The measured 1.5 is the shared Dormand-Prince
    # cold start's imaginary-axis boundary (the bootstrap is explicit and
    # unstable once CFL*2 >~ 3.4), NOT the backbones' TVD limits -- exactly
    # the cap the BDF/AB/AM family carries. The one-step trapezoidal DIRK
    # scheme skips the startup and reaches 5.0.
    'SBDF2': (None, 1.5, False),
    # Bare-RHS SBDF3 delegates to BDF3's new implicit startup. Splits retain DP5.
    'SBDF3': (None, 5.0, True),
    'CNAB2': (None, 1.5, False),
    'MEBDF2': (0.0, 0.0, False),
    # Existing classifier calls are cold one-step trajectories: CNLF2 repeats
    # backward Euler here. This is not a TVD claim for its warm two-step map.
    'CNLF2': (None, 5.0, True),
}

CFLS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]


def test_recorded_landscape_covers_the_registry():
    """Cheap completeness gate: no sweep is needed to detect registry drift."""
    assert {s.name for s in IntegrationSchemes} == set(EXPECTED_VERDICTS)


@pytest.mark.parametrize('s', IntegrationSchemes, ids=lambda s: s.name)
def test_classifier_verdicts_match_the_recorded_landscape(s):
    """One sweep per scheme, so xdist distributes the expensive measurements."""
    assert s.name in EXPECTED_VERDICTS, f'missing verdict for {s.name}'
    v = tvd_analysis.classify_tvd(s, advection('step'), CFLS, DT_SCALE, mu_max=4.0)
    ssp, tvd, uncond = EXPECTED_VERDICTS[s.name]
    if ssp is None:
        assert v.ssp_cfl is None
    else:
        assert v.ssp_cfl == pytest.approx(ssp, abs=2e-3)
    assert v.tvd_cfl == tvd
    assert v.unconditional == uncond


@pytest.mark.parametrize('name',['BDF3','BDF4','BDF5'])
def test_legacy_explicit_bdf_startup_retains_its_measured_cfl_cap(name):
    """The Phase 6 opt-in legacy path retains the previous numerical landscape."""
    from functools import partial
    scheme=getIntegrator(name)
    legacy=scheme._replace(function=partial(scheme.function,startup='explicit'))
    verdict=tvd_analysis.classify_tvd(legacy,advection('step'),CFLS,DT_SCALE,mu_max=4.)
    assert verdict.tvd_cfl==1.5 and not verdict.unconditional
