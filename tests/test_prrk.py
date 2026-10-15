"""Wave-4 pRRK (parametric-relaxation Runge-Kutta) tests.

pRRK22/33/54 (Liu, Burchard, Kuzmin & Shang 2023) apply a per-step relaxation
with parameter z = kappa*dt to a base Shu-Osher SSPRK method. Because z =
kappa*dt, the modified Butcher tableau depends on BOTH kappa and dt -- it is
not a fixed tableau, so these are exposed as standalone per-step drivers
(`prrk.py`), NOT registered in `IntegrationSchemes` (SCHEME_EXPANSION_PLAN
wave 4 registry decision). The gates pinned here:

  (a) exact kappa -> 0 reduction to the base SSPRK methods;
  (b) the stability-region enlargement at kappa > 0 (negative real axis:
      bounded at kappa = 0, A-stable at kappa >= 0.5);
  plus order preservation for every kappa, tableau consistency, and the
  "not registered" registry decision.
"""

import math

import numpy as np
import pytest

from warpSPHIntegrators.state import get_reference_state
from warpSPHIntegrators import get_scheme
import warpSPHIntegrators.testing as testing
from warpSPHIntegrators.schemes import IntegrationSchemes
from warpSPHIntegrators.drivers.explicit.relaxation import pRRK22, pRRK33, pRRK54, _prrk22_tableau, _prrk33_tableau, _prrk54_tableau

# (driver, tableau fn, registered base-scheme name, formal order)
PARRK = [
    (pRRK22, _prrk22_tableau, "Heun's Method (2nd order)", 2),
    (pRRK33, _prrk33_tableau, 'SSP RK3', 3),
    (pRRK54, _prrk54_tableau, 'SSPRK54', 4),
]
IDS = ['pRRK22', 'pRRK33', 'pRRK54']


def test_prrk_is_not_registered_in_integration_schemes():
    """Registry decision (SCHEME_EXPANSION_PLAN wave 4): at kappa = 0 pRRK
    duplicates already-registered schemes, and at kappa > 0 there is no single
    fixed order/identifier (the tableau depends on dt). So pRRK is exposed
    alongside -- not inside -- IntegrationSchemes, and must not appear in the
    registry (which would break the scheme-count pins and the auto fixtures)."""
    names = {s.name for s in IntegrationSchemes}
    for bad in ('pRRK22', 'pRRK33', 'pRRK54'):
        assert bad not in names, f'{bad} must not be a registered IntegrationScheme'


# --------------------------------------------------------------------------- #
# (a) exact kappa -> 0 reduction                                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('fn,tab_func,base_name,order', PARRK, ids=IDS)
def test_prrk_kappa_zero_equals_base_ssprk(fn, tab_func, base_name, order):
    """At kappa = 0 the relaxation is the identity: the one-step map AND the
    tableau must match the registered base SSPRK method to machine precision."""
    prob = testing.PROBLEMS['oscillator']()
    s0 = prob.initial()
    for dt in (0.07, 0.13):
        r_p = fn(s0, dt, prob.rhs, kappa=0.0)
        r_b = get_scheme(base_name)(s0, dt, prob.rhs)
        sp, sb = get_reference_state(r_p.state), get_reference_state(r_b.state)
        d = max(float((sp.x - sb.x).abs().max()), float((sp.u - sb.u).abs().max()))
        assert d < 1e-13, f'{fn.__name__} @kappa=0 deviates from {base_name} by {d}'
    tab0, base = tab_func(0.0, 0.07), get_scheme(base_name).function.butcherTableau
    assert np.allclose(tab0.a, base.a, atol=1e-14)
    assert np.allclose(tab0.b, base.b, atol=1e-14)
    assert np.allclose(tab0.c, base.c, atol=1e-14)


# --------------------------------------------------------------------------- #
# tableau consistency (at any kappa)                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('tab_func,order', [(f, o) for _, f, _, o in PARRK], ids=IDS)
def test_prrk_tableau_is_consistent_for_all_kappa(tab_func, order):
    """The relaxed tableau is a valid explicit RK tableau: c = row-sums of a
    and a strictly lower-triangular, for every kappa.

    NOTE: sum(b) is NOT 1 at kappa > 0 (it equals c_last^2). That does not
    break the order: pRRK's coefficients depend on z = kappa*dt, and every
    alpha-row sums to 1 for ALL z, so the spurious O(dt) terms cancel and the
    formal order is retained (pinned in the order test below). The standard
    sum(b) = 1 order-1 condition applies only to fixed-tableau RK methods.
    """
    for kappa in (0.0, 0.3, 1.0, 3.0):
        a, b, c = tab_func(kappa, 0.11)
        assert np.allclose(c, a.sum(axis=1), atol=1e-12), f'c != row sums at kappa={kappa}'
        assert np.allclose(a, np.tril(a, -1), atol=1e-15), f'a not strictly lower at kappa={kappa}'
    # at kappa = 0 it is a genuine standard RK tableau, so sum(b) = 1 there
    a, b, c = tab_func(0.0, 0.11)
    assert math.isclose(float(b.sum()), 1.0, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# order preservation at kappa > 0                                             #
# --------------------------------------------------------------------------- #

def _nonautonomous(t, y):
    """2D nonlinear, non-autonomous test ODE (same shape as the wave-3 probe)."""
    return np.array([y[1], -y[0] + 0.1 * np.sin(t) * y[1] ** 2])


def _run_tableau(tab_func, kappa, y0, dt, T, fun):
    y = np.array(y0, dtype=float); t = 0.0
    for _ in range(int(round(T / dt))):
        a, b, c = tab_func(kappa, dt)
        s = a.shape[0]
        ks = [fun(t, y)]
        for i in range(1, s):
            yi = y + dt * np.dot(a[i, :i], ks[:i])
            ks.append(fun(t + c[i] * dt, yi))
        y = y + dt * np.dot(b, ks); t += dt
    return y


@pytest.mark.parametrize('tab_func,order', [(f, o) for _, f, _, o in PARRK], ids=IDS)
def test_prrk_preserves_order_at_positive_kappa(tab_func, order):
    """The formal order is preserved for every kappa (the relaxation does not
    degrade accuracy on smooth problems)."""
    from scipy.integrate import solve_ivp
    y0 = np.array([1.0, 0.0]); T = 1.0
    ref = solve_ivp(_nonautonomous, [0, T], y0, method='RK45',
                    rtol=1e-12, atol=1e-14).y[:, -1]
    for kappa in (0.5, 1.0):
        errs = [np.max(np.abs(_run_tableau(tab_func, kappa, y0, dt, T, _nonautonomous) - ref))
                for dt in (0.2, 0.1, 0.05)]
        rate = math.log(errs[0] / errs[2]) / math.log(4.0)
        assert rate > order - 0.25, f'kappa={kappa}: measured rate {rate:.2f} < {order}'


# --------------------------------------------------------------------------- #
# (b) stability-region enlargement                                            #
# --------------------------------------------------------------------------- #

def _prrk_R_real(tab_func, kappa, dt, lam=-1.0):
    """Stability function R for the linear test u' = lam*u at (kappa, dt):
    R = u1/u0 after one pRRK step with the (kappa, dt)-dependent tableau."""
    a, b, c = tab_func(kappa, dt)
    s = a.shape[0]
    ks = [lam * 1.0]
    for i in range(1, s):
        ui = 1.0 + dt * np.dot(a[i, :i], ks[:i])
        ks.append(lam * ui)
    return 1.0 + dt * np.dot(b, ks)


def _real_axis_boundary(tab_func, kappa, dtmax=20.0, n=4000):
    """Largest contiguous stable dt on the negative real axis (z = -dt)."""
    ds = np.linspace(0.005, dtmax, n)
    last = 0.0
    for dt in ds:
        if abs(_prrk_R_real(tab_func, kappa, float(dt))) <= 1.0 + 1e-12:
            last = float(dt)
        else:
            break
    return last


@pytest.mark.parametrize('tab_func,order,base_interval',
                         [(_prrk22_tableau, 2, 2.0),
                          (_prrk33_tableau, 3, 2.509),
                          (_prrk54_tableau, 4, 5.331)],
                         ids=IDS)
def test_prrk_kappa_zero_real_axis_is_the_base_interval(tab_func, order, base_interval):
    """At kappa = 0 the real-axis stability interval is exactly the base SSPRK
    one ([-2, 0] / [-2.51, 0] / [-5.33, 0])."""
    boundary = _real_axis_boundary(tab_func, 0.0)
    assert abs(boundary - base_interval) < 0.05, f'kappa=0 interval {boundary} != {base_interval}'


@pytest.mark.parametrize('tab_func,order', [(f, o) for _, f, _, o in PARRK], ids=IDS)
def test_prrk_kappa_positive_is_astable_on_real_axis(tab_func, order):
    """The pRRK selling point: at kappa >= 0.5 the negative-real-axis interval
    is UNBOUNDED (A-stable) -- |R| <= 1 for arbitrarily large dt, R -> 1 from
    below (A-stable but not L-stable). This is the enlargement vs kappa = 0."""
    for dt in (1.0, 10.0, 50.0, 200.0):
        r = _prrk_R_real(tab_func, 0.5, dt)
        assert abs(r) <= 1.0 + 1e-12, f'dt={dt}: |R|={abs(r):.4f} > 1 (not A-stable)'
    # and it is a genuine enlargement: the kappa=0 boundary is finite
    assert _real_axis_boundary(tab_func, 0.0) < 10.0
    # while at kappa=0.5 we are stable well past it (sweep to dtmax=300)
    assert _real_axis_boundary(tab_func, 0.5, dtmax=300.0) >= 200.0
