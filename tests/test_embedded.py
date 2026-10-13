"""Embedded pairs: the second weight vector must reach the caller as an error estimate.

Before this, `RungeKuttaB` computed both solutions, finalized every one of them with a
`b_` leaked from the loop variable, and returned only the last -- so the estimate that
is the entire point of an embedded pair was thrown away (NOTES.md 2.8).
"""

import math

import pytest
import torch

from conftest import EXACT_ON_LINEAR_PROBLEMS, dts_for
from warpSPHIntegrators import getIntegrator, get_reference_state, testing
from warpSPHIntegrators.butcher import getButcherTableau

# TR-BDF2 carries SUNDIALS ARKODE's published (2, 3) embedded pair: the estimate
# is O(dt^3) = dt^(order+1), the propagated branch's own true local error. The
# ESDIRK and ARK pairs are (3, 2) and (4, 3): the estimate is the embedded branch's
# local error, O(dt^p) = dt^order.
EMBEDDED = ['Bogacki-Shampine 3(2)', 'Dormand-Prince 5(4)', 'Cash-Karp 5(4)',
            # The wave-2 ODE SSPRK 3(2) pairs (SCHEME_EXPANSION_PLAN, 2026-10-12):
            # order-3 main, order-2 estimate, so the estimate is the embedded
            # branch's local error, O(dt^3) = dt^order.
            'SSPRK43', 'SSPRK432',
            # The wave-6 ODE embedded pairs (SCHEME_EXPANSION_PLAN, 2026-10-13):
            # each is (p, p-1), so the estimate is the embedded branch's local
            # error, O(dt^(p-1+1)) = dt^order. BS5's embedded branch is ODE's
            # bhat (order 4), not btilde (invalid).
            'Tsit5 5(4)', 'OwrenZen3 3(2)', 'OwrenZen4 4(3)', 'OwrenZen5 5(4)',
            'BS5 5(4)', 'Stepanov5 5(4)', 'Alshina2 2(1)',
            'TR-BDF2', 'ESDIRK3(2)4L[2]SA', 'ESDIRK4(3)6L[2]SA',
            'ARK3(2)4L[2]SA', 'ARK4(3)6L[2]SA',
            # Rosenbrock-W carries the built-in order-2 embedded estimate
            # y - y_hat = tau (K1 - K2) / 3 (local error O(dt^3)).
            'ROS3P',
            # Exponential Rosenbrock carries the embedded order-2 estimate: the
            # stage U2 (exp-Rosenbrock-Euler), so the estimate is the corrector
            # correction y - y_hat = 2 h phi_3(hJn) D2 (local error O(dt^3)).
            'EXPRB32',
            # The coupled fully implicit block pair (NOTES.md S3.18): the
            # estimate is the null-stage order-2 companion (k0 = f(t^n, y^n)
            # plus the two stages). Both tableaus carry an order-2 companion,
            # so main-minus-companion is O(dt^3) for both -- one order below
            # the propagated branch for Gauss-Legendre 2 (order 4), exactly
            # the propagated order for Radau IIA s=2 (order 3).
            'Gauss-Legendre 2',
            'Radau IIA s=2',
            # The wave-7 Verner / TsitPap embedded pairs (SCHEME_EXPANSION_PLAN,
            # 2026-10-13): each is (p, p-1), so the estimate is the embedded
            # branch's local error, O(dt^(p-1+1)) = dt^order, like the wave-6
            # pairs. DP8 and the Feagin tableaus are PLAIN (no embedded weight
            # vector) and stay out of this list.
            'Vern6 6(5)', 'Vern7 7(6)', 'Vern8 8(7)', 'Vern9 9(8)',
            'RKV76IIa 7(6)', 'TanYam7 7(6)', 'TsitPap8 8(7)']


#: Wave 7 (SCHEME_EXPANSION_PLAN, 2026-10-13): per-pair grids for the new (p, p-1)
#: estimates, which floor out (below 1e-13) at the finest points of the standard
#: grid. Measured single-step estimate rates against dt^order on these grids:
#: Vern6 6.08/6.04/6.00, Vern7 7.24/7.11/7.05, Vern8 8.48/8.32/8.19, Vern9
#: 9.79/9.29/9.21, RKV76IIa 7.20/7.09/7.03, TanYam7 7.20/7.09/6.99, TsitPap8
#: 9.03/8.73/8.51.
EMBEDDED_GRID = {
    'Vern6 6(5)': [0.2, 0.1, 0.05, 0.025],
    'Vern7 7(6)': [0.4, 0.2, 0.1, 0.05],
    'Vern8 8(7)': [0.8, 0.4, 0.2, 0.1],
    'Vern9 9(8)': [0.8, 0.4, 0.2, 0.1],
    'RKV76IIa 7(6)': [0.4, 0.2, 0.1, 0.05],
    'TanYam7 7(6)': [0.4, 0.2, 0.1, 0.05],
    'TsitPap8 8(7)': [0.8, 0.4, 0.2, 0.1],
}


@pytest.mark.parametrize('name', EMBEDDED)
def test_error_estimate_is_returned(name):
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    result = s(prob.initial(), dt=0.1, f=prob.rhs)
    assert result.error is not None, f'{name} returned no error estimate'
    err = get_reference_state(result.error)
    assert err.x.shape == get_reference_state(result.state).x.shape


@pytest.mark.parametrize('name', EMBEDDED)
def test_error_estimate_has_the_embedded_order(name):
    """|y_high - y_low| must scale as dt^(p_low + 1), the local error of the low branch.

    That is what makes it usable for step size control.

    ROS3P's (3, 2) embedded pair is degenerate on a linear *autonomous* problem:
    there the frozen operator is the constant Jacobian ``A`` and the stage-2 right
    hand side ``A(z_n + tau K1) - tau A K1`` collapses to ``A z_n`` -- stage 1's --
    so ``K1 == K2`` and the estimate ``tau (K1 - K2) / 3`` is exactly zero (0/0 here).
    EXPRB32's pair degenerates the same way: on a linear autonomous problem the
    nonlinear remainder ``D2 = f - f_n - Jn (u - un)`` vanishes, so the estimate
    ``2 h phi_3(hJn) D2`` is exactly zero. The non-autonomous forced oscillator is
    the smallest problem where the pair is active, so it is the one both are
    measured on.
    """
    s = getIntegrator(name)
    prob_name = 'forced' if name in ('ROS3P', 'EXPRB32') else 'oscillator'
    prob = testing.PROBLEMS[prob_name]()

    # The wave-7 pairs (EMBEDDED_GRID) floor out on the standard grid, so each
    # runs on its own coarser grid; a pair whose finer magnitude is at the
    # roundoff floor carries no rate.
    dts = EMBEDDED_GRID.get(name, [0.1, 0.05, 0.025, 0.0125])
    magnitudes = []
    for dt in dts:
        result = s(prob.initial(), dt=dt, f=prob.rhs)
        err = get_reference_state(result.error)
        magnitudes.append(float(err.x.abs().sum() + err.u.abs().sum()))

    rates = []
    for a, b in zip(magnitudes, magnitudes[1:]):
        if name in EMBEDDED_GRID and b < 1e-13:
            continue  # at the roundoff floor, the rate is meaningless
        rates.append(math.log(a / b) / math.log(2))
    # Local error of the (p-1)-order branch is O(dt^p). TR-BDF2 is the exception:
    # its published (2, 3) pair embeds a *higher*-order branch, so the difference
    # is the propagated branch's own O(dt^3) local error. Gauss-Legendre 2's
    # null-stage companion is only order 2 (the 2-stage-only order-2 pair is
    # degenerate -- b itself -- for both block tableaus), so its estimate is
    # O(dt^3) = dt^(order-1), not dt^order; Radau IIA s=2 (order 3) lands on
    # the default dt^order either way.
    expected = (s.order + 1 if s.name == 'TR-BDF2'
                else s.order - 1 if s.name == 'Gauss-Legendre 2'
                else s.order)
    if name in EMBEDDED_GRID:
        # One-sided band: the coarsest pair of the coarser grids can still be
        # pre-asymptotic (overshooting the asymptotic rate), so only the low
        # side is held tight; nothing may undershoot by more than half an order.
        assert all(expected - 0.5 <= r <= expected + 2.0 for r in rates), (
            f'{name}: error estimate scales as dt^{rates}, expected dt^{expected}'
        )
    else:
        assert all(abs(r - expected) < 0.3 for r in rates), (
            f'{name}: error estimate scales as dt^{rates}, expected dt^{expected}'
        )


@pytest.mark.parametrize('name', [
    'Bogacki-Shampine 3(2)',
    'Dormand-Prince 5(4)',
    'Cash-Karp 5(4)',
    'SSPRK43',
    'SSPRK432',
    'TR-BDF2',
    'ESDIRK3(2)4L[2]SA',
    'ESDIRK4(3)6L[2]SA',
    'ARK3(2)4L[2]SA',
    'ARK4(3)6L[2]SA',
])
def test_error_estimate_is_the_right_size(name):
    """The estimate must be the right size -- within the (lo, hi) bracket of the
    propagated solution's true local error. TR-BDF2's published (2, 3) pair makes
    the estimate the propagated branch's own O(dt^3) error (ratio ~ 0.6 at
    dt = 0.05). For a (p, p-1) ESDIRK pair the estimate is the embedded branch's
    local error, one order coarser, so the ratio grows ~ 1/dt (34x at dt = 0.05
    for ESDIRK4(3)6) -- conservative in the right direction for step-size control,
    absorbed by the loose upper bound."""
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()
    dt = 0.05

    result = s(prob.initial(), dt=dt, f=prob.rhs)
    estimate = float(get_reference_state(result.error).x.abs().sum())

    exact_x, _ = prob.exact(dt)
    reference_value = abs(float(get_reference_state(result.state).x[0]) - exact_x[0])
    lo, hi = 0.02, 500.0

    assert estimate > 0
    assert lo < estimate / max(reference_value, 1e-300) < hi, (
        f'{name}: estimate {estimate:.3e} vs reference {reference_value:.3e}'
    )


@pytest.mark.parametrize('name', EMBEDDED)
def test_propagated_solution_is_the_high_order_branch(name):
    """b[0] propagates. Returning b[-1] instead would silently cost a whole order."""
    s = getIntegrator(name)
    # EXPRB32 is exact on the linear autonomous oscillator (roundoff-floor error,
    # no measurable order); the forced problem is the smallest one where its order
    # 3 shows up. The wave-7 pairs floor out on the standard oscillator grid too,
    # so use the per-problem coarse grid where they have a measurable order.
    prob_name = 'forced' if name in EXACT_ON_LINEAR_PROBLEMS else 'oscillator'
    order, _ = testing.convergence(s, testing.PROBLEMS[prob_name](),
                                   dts_for(name, prob_name, testing.default_step_sizes()), 2.0)
    assert order >= s.order - 0.15


@pytest.mark.parametrize('name', ['BogackiShampine', 'DormandPrince', 'CashKarp',
                                  'SSPRK43', 'SSPRK432',
                                  'Tsit5', 'OwrenZen3', 'OwrenZen4', 'OwrenZen5',
                                  'BS5', 'Stepanov5', 'Alshina2',
                                  'Vern6', 'Vern7', 'Vern8', 'Vern9',
                                  'RKV76IIa', 'TanYam7', 'TsitPap8'])
def test_tableau_is_consistent(name):
    """Row sums of `a` must equal `c`, and both weight vectors must sum to 1."""
    tab = getButcherTableau(name)
    assert isinstance(tab.b, tuple) and len(tab.b) == 2
    for row, c in zip(tab.a, tab.c):
        assert row.sum() == pytest.approx(c, abs=1e-13)
    for weights in tab.b:
        assert weights.sum() == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize('name', ['BogackiShampine', 'DormandPrince',
                                  'Tsit5', 'OwrenZen3', 'OwrenZen4', 'OwrenZen5',
                                  'BS5', 'Stepanov5', 'Vern6'])
def test_fsal_property_holds_in_the_tableau(name):
    tab = getButcherTableau(name)
    assert tab.c[-1] == pytest.approx(1.0)
    assert tab.a[-1] == pytest.approx(tab.b[0])


def test_non_embedded_schemes_report_no_error(scheme):
    if scheme.name in EMBEDDED:
        return
    prob = testing.PROBLEMS['oscillator']()
    result = scheme(prob.initial(), dt=0.1, f=prob.rhs)
    assert result.error is None
