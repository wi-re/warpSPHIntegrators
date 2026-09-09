"""Embedded pairs: the second weight vector must reach the caller as an error estimate.

Before this, `RungeKuttaB` computed both solutions, finalized every one of them with a
`b_` leaked from the loop variable, and returned only the last -- so the estimate that
is the entire point of an embedded pair was thrown away (NOTES.md 2.8).
"""

import math

import pytest
import torch

from warpSPHIntegrators import getIntegrator, get_reference_state, testing
from warpSPHIntegrators.butcher import getButcherTableau

# TR-BDF2 carries SUNDIALS ARKODE's published (2, 3) embedded pair: the estimate
# is O(dt^3) = dt^(order+1), the propagated branch's own true local error. The
# ESDIRK and ARK pairs are (3, 2) and (4, 3): the estimate is the embedded branch's
# local error, O(dt^p) = dt^order.
EMBEDDED = ['Bogacki-Shampine 3(2)', 'Dormand-Prince 5(4)', 'Cash-Karp 5(4)',
            'TR-BDF2', 'ESDIRK3(2)4L[2]SA', 'ESDIRK4(3)6L[2]SA',
            'ARK3(2)4L[2]SA', 'ARK4(3)6L[2]SA']


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
    """
    s = getIntegrator(name)
    prob = testing.PROBLEMS['oscillator']()

    dts = [0.1, 0.05, 0.025, 0.0125]
    magnitudes = []
    for dt in dts:
        result = s(prob.initial(), dt=dt, f=prob.rhs)
        err = get_reference_state(result.error)
        magnitudes.append(float(err.x.abs().sum() + err.u.abs().sum()))

    rates = [math.log(a / b) / math.log(2) for a, b in zip(magnitudes, magnitudes[1:])]
    # Local error of the (p-1)-order branch is O(dt^p). TR-BDF2 is the exception:
    # its published (2, 3) pair embeds a *higher*-order branch, so the difference
    # is the propagated branch's own O(dt^3) local error.
    expected = s.order + 1 if s.name == 'TR-BDF2' else s.order
    assert all(abs(r - expected) < 0.3 for r in rates), (
        f'{name}: error estimate scales as dt^{rates}, expected dt^{expected}'
    )


@pytest.mark.parametrize('name', [
    'Bogacki-Shampine 3(2)',
    'Dormand-Prince 5(4)',
    'Cash-Karp 5(4)',
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
    order, _ = testing.convergence(s, testing.PROBLEMS['oscillator'](),
                                   testing.default_step_sizes(), 2.0)
    assert order >= s.order - 0.15


@pytest.mark.parametrize('name', ['BogackiShampine', 'DormandPrince', 'CashKarp'])
def test_tableau_is_consistent(name):
    """Row sums of `a` must equal `c`, and both weight vectors must sum to 1."""
    tab = getButcherTableau(name)
    assert isinstance(tab.b, tuple) and len(tab.b) == 2
    for row, c in zip(tab.a, tab.c):
        assert row.sum() == pytest.approx(c, abs=1e-13)
    for weights in tab.b:
        assert weights.sum() == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize('name', ['BogackiShampine', 'DormandPrince'])
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
