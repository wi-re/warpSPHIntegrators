"""Parametric-relaxation Runge-Kutta methods (pRRK).

Liu, Burchard, Kuzmin & Shang, "Parametric relaxation Runge-Kutta methods"
(2023). Coefficients transcribed from OrdinaryDiffEq v7.8.1
(``lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl`` --
``_prrk{22,33,54}_coeffs`` -- and ``ssprk_caches.jl`` -- the
``pRRK{22,33,54}ConstantCache`` base Shu-Osher coefficients).

Mechanism -- and why these are NOT registered in ``IntegrationSchemes``:
    pRRK starts from a Shu-Osher SSPRK method and applies a per-step
    *relaxation* with parameter ``z = kappa*dt``::

        psi_1 = 1 + z*beta_10,   psi_i = alpha_i0 + psi_{i-1}(alpha_i? + z*beta_i?)
        alpha_hat_ij = (psi_j/psi_i) * (alpha_ij + z*beta_ij)
        beta_hat_ij  = (psi_j/psi_i) * beta_ij
        dt_hat       = c_hat_s * dt          # c_hat_s = last modified abscissa

    Because ``z = kappa*dt``, the modified tableau ``(a, b, c)`` depends on
    BOTH ``kappa`` and ``dt`` -- it is **not** a fixed Butcher tableau. A
    ``butcherScheme(...)`` builds its tableau once, so it cannot represent
    pRRK; each step must recompute ``(a, b, c)`` from the caller's
    ``(kappa, dt)``, which is what the drivers below do. The Shu-Osher
    relaxation map unfolds to a standard explicit RK tableau because every
    modified alpha-row sums to 1 (a direct consequence of the psi
    recurrence), so the shared ``RungeKuttaB`` reproduces the pRRK one-step
    map exactly.

    At ``kappa = 0`` the relaxation is the identity and the method reduces
    EXACTLY to the base SSPRK -- pRRK22 -> Heun, pRRK33 -> SSPRK3,
    pRRK54 -> SSPRK54 (pinned to machine precision in tests/test_prrk.py).
    So at ``kappa = 0`` these duplicate already-registered schemes; their
    value is only at ``kappa > 0``, where there is no single fixed
    order/identifier. Hence they are exposed *alongside* (not inside)
    ``IntegrationSchemes`` -- see SCHEME_EXPANSION_PLAN.md wave 4.

Stability (measured; NOTES.md 3.24): the relaxation enlarges the
negative-real-axis stability interval. At ``kappa = 0`` it is the base
SSPRK interval ([-2, 0] / [-2.51, 0] / [-5.33, 0] for 22/33/54); at
``kappa >= 0.5`` the interval is UNBOUNDED -- A-stable on the real axis
(|R| <= 1 for every dt, with R -> 1 from below as dt -> inf, i.e.
A-stable but not L-stable). The formal order is preserved for every
kappa. The trade-off: larger kappa buys stiffness tolerance at the cost of
the SSP (TVD) constant.

The tableau is recomputed from the caller's dt on every call, so a dt
change needs no history and no restart. These are plain one-step methods:
no error estimate, so adaptive step control does not apply (there is no
embedded pair).
"""

import numpy as np

from warpSPHIntegrators.drivers.runge_kutta import RungeKuttaB
from warpSPHIntegrators.schemes.coefficients.tableau import ButcherTableau as butcherTableau


# --------------------------------------------------------------------------- #
# kappa-, dt-dependent Butcher tableaus                                        #
# --------------------------------------------------------------------------- #

def _prrk22_tableau(kappa, dt):
    """Butcher (a, b, c) for pRRK22 at (kappa, dt). Base: Heun / SSPRK(2,2)."""
    z = kappa * dt
    # Base Shu-Osher: a10=1, b10=1, a20=1/2, a21=1/2, b21=1/2
    psi1 = 1.0 + z
    psi2 = 0.5 + psi1 * (0.5 + z * 0.5)
    bhat10 = 1.0 / psi1
    ahat21 = psi1 * (0.5 + z * 0.5) / psi2
    bhat21 = psi1 * 0.5 / psi2
    c_last = ahat21 * bhat10 + bhat21            # = c_hat_2, dt_hat = c_last*dt
    a21 = bhat10 * c_last
    return butcherTableau(
        a=np.array([[0.0, 0.0], [a21, 0.0]]),
        b=np.array([ahat21 * a21, bhat21 * c_last]),
        c=np.array([0.0, a21]),
    )


def _prrk33_tableau(kappa, dt):
    """Butcher (a, b, c) for pRRK33 at (kappa, dt). Base: SSPRK(3,3)."""
    z = kappa * dt
    # Base Shu-Osher: a10=1, b10=1, a20=3/4, a21=1/4, b21=1/4,
    #                 a30=1/3, a32=2/3, b32=2/3
    psi1 = 1.0 + z
    psi2 = 0.75 + psi1 * (0.25 + z * 0.25)
    psi3 = (1.0 / 3.0) + psi2 * ((2.0 / 3.0) + z * (2.0 / 3.0))
    bhat10 = 1.0 / psi1
    ahat21 = psi1 * (0.25 + z * 0.25) / psi2
    bhat21 = psi1 * 0.25 / psi2
    ahat32 = psi2 * ((2.0 / 3.0) + z * (2.0 / 3.0)) / psi3
    bhat32 = psi2 * (2.0 / 3.0) / psi3
    c_last = ahat32 * (ahat21 * bhat10 + bhat21) + bhat32   # = c_hat_3
    a21 = bhat10 * c_last
    a31 = ahat21 * a21
    a32 = bhat21 * c_last
    return butcherTableau(
        a=np.array([[0.0, 0.0, 0.0], [a21, 0.0, 0.0], [a31, a32, 0.0]]),
        b=np.array([ahat32 * a31, ahat32 * a32, bhat32 * c_last]),
        c=np.array([0.0, a21, a31 + a32]),
    )


def _prrk54_tableau(kappa, dt):
    """Butcher (a, b, c) for pRRK54 at (kappa, dt). Base: SSPRK(5,4) (Spiteri-Ruuth)."""
    z = kappa * dt
    # Base Shu-Osher coefficients (ODE pRRK54ConstantCache); a10 = 1 implicit.
    b10 = 0.39175222657189
    a20, a21, b21 = 0.444370493651235, 0.555629506348765, 0.368410593050371
    a30, a32, b32 = 0.620101851488403, 0.379898148511597, 0.251891774271694
    a40, a43, b43 = 0.178079954393132, 0.821920045606868, 0.544974750228521
    a52, a53, b53 = 0.517231671970585, 0.096059710526147, 0.06369246866629
    a54, b54 = 0.386708617503269, 0.226007483236906
    psi1 = 1.0 + z * b10
    psi2 = a20 + psi1 * (a21 + z * b21)
    psi3 = a30 + psi2 * (a32 + z * b32)
    psi4 = a40 + psi3 * (a43 + z * b43)
    psi5 = psi2 * a52 + psi3 * (a53 + z * b53) + psi4 * (a54 + z * b54)
    bhat10 = b10 / psi1
    ahat21 = psi1 * (a21 + z * b21) / psi2
    bhat21 = psi1 * b21 / psi2
    ahat32 = psi2 * (a32 + z * b32) / psi3
    bhat32 = psi2 * b32 / psi3
    ahat43 = psi3 * (a43 + z * b43) / psi4
    bhat43 = psi3 * b43 / psi4
    ahat52 = psi2 * a52 / psi5
    ahat53 = psi3 * (a53 + z * b53) / psi5
    bhat53 = psi3 * b53 / psi5
    ahat54 = psi4 * (a54 + z * b54) / psi5
    bhat54 = psi4 * b54 / psi5
    c1 = bhat10
    c2 = ahat21 * c1 + bhat21
    c3 = ahat32 * c2 + bhat32
    c4 = ahat43 * c3 + bhat43
    c_last = ahat52 * c2 + ahat53 * c3 + bhat53 + ahat54 * c4 + bhat54   # c_hat_5
    # Stage arguments as u^n + dt*sum a_ij k_j (unfold the relaxation).
    a21 = bhat10 * c_last
    a31, a32 = ahat21 * a21, bhat21 * c_last
    a41, a42, a43 = ahat32 * a31, ahat32 * a32, bhat32 * c_last
    a51, a52, a53, a54 = ahat43 * a41, ahat43 * a42, ahat43 * a43, bhat43 * c_last
    return butcherTableau(
        a=np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                    [a21, 0.0, 0.0, 0.0, 0.0],
                    [a31, a32, 0.0, 0.0, 0.0],
                    [a41, a42, a43, 0.0, 0.0],
                    [a51, a52, a53, a54, 0.0]]),
        b=np.array([ahat52 * a31 + ahat53 * a41 + ahat54 * a51,
                    ahat52 * a32 + ahat53 * a42 + ahat54 * a52,
                    ahat53 * a43 + ahat54 * a53,
                    bhat53 * c_last + ahat54 * a54,
                    bhat54 * c_last]),
        c=np.array([0.0, a21, a31 + a32, a41 + a42 + a43, a51 + a52 + a53 + a54]),
    )


# --------------------------------------------------------------------------- #
# Drivers                                                                      #
# --------------------------------------------------------------------------- #

def _prrk_step(tableau_func, initialState, dt, f, kappa, *args, **kwargs):
    """Recompute the (kappa, dt)-dependent tableau, then run the shared RK driver."""
    tableau = tableau_func(kappa, float(dt))
    return RungeKuttaB(initialState, dt, f, tableau, *args, **kwargs)


def pRRK22(initialState, dt, f, *args, kappa=0.0, **kwargs):
    """pRRK22 -- parametric-relaxation SSPRK(2,2); base method is Heun.

    ``kappa`` is the free relaxation parameter (z = kappa*dt). At
    ``kappa = 0`` this is exactly Heun; for ``kappa >= 0.5`` the method is
    A-stable on the negative real axis. Order 2 for every kappa.
    """
    return _prrk_step(_prrk22_tableau, initialState, dt, f, kappa, *args, **kwargs)


def pRRK33(initialState, dt, f, *args, kappa=0.0, **kwargs):
    """pRRK33 -- parametric-relaxation SSPRK(3,3); base method is SSPRK3.

    ``kappa`` is the free relaxation parameter (z = kappa*dt). At
    ``kappa = 0`` this is exactly SSPRK3; for ``kappa >= 0.5`` the method is
    A-stable on the negative real axis. Order 3 for every kappa.
    """
    return _prrk_step(_prrk33_tableau, initialState, dt, f, kappa, *args, **kwargs)


def pRRK54(initialState, dt, f, *args, kappa=0.0, **kwargs):
    """pRRK54 -- parametric-relaxation SSPRK(5,4); base method is SSPRK54.

    ``kappa`` is the free relaxation parameter (z = kappa*dt). At
    ``kappa = 0`` this is exactly SSPRK54; for ``kappa >= 0.5`` the method is
    A-stable on the negative real axis. Order 4 for every kappa.
    """
    return _prrk_step(_prrk54_tableau, initialState, dt, f, kappa, *args, **kwargs)
