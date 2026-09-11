"""Relaxed Chebyshev / Lobatto super-timestepping: RKC1, RKC2, RKL2.

Stabilised explicit methods for stiff *parabolic* (diffusive) right-hand sides.
They take ``s`` cheap, matrix-free right-hand-side evaluations per step and buy a
real-axis stability interval that grows like ``O(s^2)`` -- so the step size can be
set by the physics (the nonlinear convective scale), not by the stiffest diffusive
eigenvalue. This is what makes them attractive for the viscous term in SPH and for
the semilinear benchmarks in ``testing``.

Construction
------------
Each method's one-step stability polynomial is a scaled, shifted orthogonal
polynomial in ``z = dt * lambda``:

    RKC1  R_s(z) = T_s(1 + z/s^2)                     (Chebyshev)
    RKC2  R_s(z) = a_s + b_s T_s(1 + w_1 z)           (relaxed Chebyshev)
    RKL2  R_s(z) = a_s + b_s P_s(1 + w_1 z)           (relaxed Lobatto)

with ``T_s`` the Chebyshev and ``P_s`` the Legendre polynomial. RKC1 is order 1;
RKC2 and RKL2 add the constant offset ``a_s`` (and the scaling ``b_s``) that makes
the polynomial match ``e^z`` through ``z^2``, lifting it to order 2. The real-axis
stability intervals (``|R_s| <= 1`` on ``[-K, 0]``) are

    RKC1  K = 2 s^2
    RKC2  K = 2 (s^2 - 1) / 3
    RKL2  K = (s^2 + s - 2) / 2

All three are realised by a three-term (plus a cached ``Y_0``) recurrence, so a step
costs ``s`` right-hand-side evaluations and a few state buffers -- no linear solve,
no matrix. The recurrences (per component, ``z = dt*lambda``) are

    Y_0 = u,   f_0 = f(Y_0)
    Y_1 = Y_0 + mtilde_1 dt f_0
    Y_j = (1 - mu_j - nu_j) Y_0 + mu_j Y_{j-1} + nu_j Y_{j-2}
          + dt (mtilde_j f_{j-1} + gtilde_j f_0),      2 <= j <= s
    u^{n+1} = Y_s

The coefficients come from the polynomial identities
``T_j = 2x T_{j-1} - T_{j-2}`` (Chebyshev) and the Legendre three-term recurrence,
with the stage weights chosen so that every intermediate stage polynomial
``R_j(z) = a_j + b_j Phi_j(1 + w_1 z)`` is itself bounded by 1 on ``[-K, 0]``
(intermediate stability, so no stage overshoots). RKL2 is the scheme of
Meyer, Balsara & Aslam, *J. Comput. Phys.* 257 (2014) 594-626 (Eqs. 15-19); RKC2
is the identical construction with ``T_j`` in place of ``P_j``; RKC1 is the order-1
Chebyshev limit (Ruuth, *J. Comput. Phys.* 169 (2001) 162-175).

Stage count
-----------
The number of stages ``s`` is a per-step choice. Pass ``s=`` directly, or pass
``lambda_max=`` (the magnitude of the stiffest right-hand-side eigenvalue) and the
smallest admissible ``s`` is chosen so that ``K(s) >= dt * lambda_max``. RKL2 and
RKC2 prefer an *odd* stage count: for even ``s`` the smallest-wavelength mode is
only weakly damped (Meyer et al., 2014). The driver clamps ``s`` to the method's
minimum (1 for RKC1, 3 for RKC2/RKL2) but does not force parity -- pick ``s`` (or
a slightly larger ``lambda_max``) if you want odd.

These schemes are explicit one-step methods: they do not implement first-stage
reuse (``priorStep`` is rejected) and are intended for autonomous, parabolic
right-hand sides. Stage times are set to ``t + dt * j / s`` as a reasonable
intermediate time; for a non-autonomous right-hand side this is an approximation.
"""

import dataclasses

from .util import (
    finalizeSystem,
    initializeSystem,
    reject_prior_step,
    updateStateEuler,
    updateStep,
)
from .fields import get_reference_state, field_behavior
from .specs import IntegrationResult, StageResult
from torch.profiler import record_function


# --------------------------------------------------------------------------- #
# Stage coefficients                                                          #
# --------------------------------------------------------------------------- #
#
# Each returns ``(mtilde1, mu, nu, mtilde, gtilde)`` for a given stage count ``s``:
# ``mu[j]``/``nu[j]``/``mtilde[j]``/``gtilde[j]`` are the recurrence weights used at
# stage ``j`` (``2 <= j <= s``), and ``mtilde1`` the first-stage weight. The lists
# are indexed 0..s with entries below the method's minimum left at zero.

def _rkc1_coeffs(s):
    """RKC1 (order 1, K = 2 s^2): pure Chebyshev, no Y_0 correction."""
    mu = [0.0] * (s + 1)
    nu = [0.0] * (s + 1)
    mtilde = [0.0] * (s + 1)
    gtilde = [0.0] * (s + 1)
    mtilde[1] = 1.0 / (s * s)
    for j in range(2, s + 1):
        mu[j] = 2.0
        nu[j] = -1.0
        mtilde[j] = 2.0 / (s * s)
    return mtilde[1], mu, nu, mtilde, gtilde


def _rkc2_coeffs(s):
    """RKC2 (order 2, K = 2(s^2-1)/3): relaxed Chebyshev.

    ``R_s = a_s + b_s T_s(1 + w_1 z)`` with ``w_1 = 3/(s^2-1)``,
    ``b_s = (s^2-1)/(3 s^2)``, ``a_s = 1 - b_s``. The stage weights follow from the
    Chebyshev three-term recurrence; the per-stage amplitude ``b_j`` runs geometrically
    from ``b_0 = b_1 = b_2 = 1/3`` to the final ``b_s``, which keeps every intermediate
    stage ``R_j = a_j + b_j T_j(1 + w_1 z)`` bounded by 1 on the stability interval.
    """
    w1 = 3.0 / (s * s - 1)
    b_s = (s * s - 1) / (3.0 * s * s)
    b = [0.0] * (s + 1)
    b[0] = b[1] = b[2] = 1.0 / 3
    if s >= 3:
        ratio = (b_s / (1.0 / 3)) ** (1.0 / (s - 2))
        for j in range(3, s + 1):
            b[j] = (1.0 / 3) * ratio ** (j - 2)
    b[s] = b_s  # pin the final stage exactly
    a = [1.0 - bj for bj in b]
    mu = [0.0] * (s + 1)
    nu = [0.0] * (s + 1)
    mtilde = [0.0] * (s + 1)
    gtilde = [0.0] * (s + 1)
    mtilde[1] = b[1] * w1
    for j in range(2, s + 1):
        mu[j] = 2.0 * (b[j] / b[j - 1])
        nu[j] = -(b[j] / b[j - 2])
        mtilde[j] = mu[j] * w1
        gtilde[j] = -a[j - 1] * mtilde[j]
    return mtilde[1], mu, nu, mtilde, gtilde


def _rkl2_coeffs(s):
    """RKL2 (order 2, K = (s^2+s-2)/2): relaxed Lobatto (Meyer et al. 2014, Eqs 15-19).

    ``R_s = a_s + b_s P_s(1 + w_1 z)`` with ``w_1 = 4/(s^2+s-2)``,
    ``b_s = (s^2+s-2)/(2 s (s+1))``, ``a_s = 1 - b_s``. The stage weights follow from
    the Legendre three-term recurrence with ``b_j/b_{j-1} = (j+2)(j-1)^2 /
    ((j-2)(j+1)^2)`` for ``j >= 3`` and ``b_0 = b_1 = b_2 = 1/3``.
    """
    w1 = 4.0 / (s * s + s - 2)
    b = [0.0] * (s + 1)
    b[0] = b[1] = b[2] = 1.0 / 3
    for j in range(3, s + 1):
        b[j] = b[j - 1] * (j + 2) * (j - 1) ** 2 / ((j - 2) * (j + 1) ** 2)
    a = [1.0 - bj for bj in b]
    mu = [0.0] * (s + 1)
    nu = [0.0] * (s + 1)
    mtilde = [0.0] * (s + 1)
    gtilde = [0.0] * (s + 1)
    mtilde[1] = b[1] * w1
    for j in range(2, s + 1):
        mu[j] = (2 * j - 1) / j * (b[j] / b[j - 1])
        nu[j] = -(j - 1) / j * (b[j] / b[j - 2])
        mtilde[j] = mu[j] * w1
        gtilde[j] = -a[j - 1] * mtilde[j]
    return mtilde[1], mu, nu, mtilde, gtilde


_COEFFS = {'rkc1': _rkc1_coeffs, 'rkc2': _rkc2_coeffs, 'rkl2': _rkl2_coeffs}
_NAMES = {'rkc1': 'RKC1', 'rkc2': 'RKC2', 'rkl2': 'RKL2'}
#: Real-axis stability interval ``K(s)``: stable for ``dt * |lambda_max| <= K(s)``.
_K = {
    'rkc1': lambda s: 2.0 * s * s,
    'rkc2': lambda s: 2.0 * (s * s - 1) / 3.0,
    'rkl2': lambda s: (s * s + s - 2) / 2.0,
}
#: Minimum stage count the coefficient construction is valid for.
_SMIN = {'rkc1': 1, 'rkc2': 3, 'rkl2': 3}


def stage_count(dt_lambda_max, family):
    """Smallest stage count ``s`` with ``K(s) >= dt_lambda_max``.

    ``dt_lambda_max`` is ``dt * |lambda_max| >= 0``. Clamped to the method's minimum
    stage count. Does not enforce odd parity (see the module docstring).
    """
    K = _K[family]
    s = _SMIN[family]
    while K(s) < dt_lambda_max:
        s += 1
    return s


def _reference_lincomb(template_ref, terms):
    """``sum weight_i * ref_i`` over the integrated fields of a reference state.

    ``terms`` is ``[(ref_state, weight), ...]``. Returns ``{field_name: value}`` for
    every integrated field of ``template_ref``. The linear combination of the three
    prior stage states in the RKC/RKL recurrence is the one operation the standard
    RK drivers never need (they only ever form ``Y_0 + dt * sum a_ij f_j``), so it
    lives here once.
    """
    out = {}
    for f in dataclasses.fields(template_ref):
        if field_behavior(f) != 'integrated':
            continue
        name = f.name
        acc = None
        for ref, weight in terms:
            if weight == 0.0:
                continue
            v = getattr(ref, name)
            acc = weight * v if acc is None else acc + weight * v
        if acc is None:
            base = getattr(template_ref, name)
            acc = base * 0.0
        out[name] = acc
    return out


def integrateRelaxedChebyshev(state, dt, f, family, *args, **kwargs):
    """One step of RKC1 / RKC2 / RKL2 (selected by ``family``).

    ``s=`` sets the stage count; if omitted, ``lambda_max=`` (the magnitude of the
    stiffest right-hand-side eigenvalue) selects the smallest admissible count. Both
    are consumed here and never forwarded to the right-hand side.
    """
    name = _NAMES[family]
    reject_prior_step(name, kwargs.pop('priorStep', None))
    s = kwargs.pop('s', None)
    lambda_max = kwargs.pop('lambda_max', None)
    if s is None:
        if lambda_max is None:
            raise ValueError(
                f"{name}: pass s= (stage count) or lambda_max= (stiffest eigenvalue "
                f"magnitude) to size the super-time-step")
        s = stage_count(abs(dt) * abs(lambda_max), family)
    s = max(int(s), _SMIN[family])
    mtilde1, mu, nu, mtilde, gtilde = _COEFFS[family](s)

    initializeSystem(state, dt, *args, **kwargs)
    with record_function(f"[Integration] {name}"):
        # Stage 0: evaluate f at the initial state (this is the Y_0 correction the
        # order-2 methods carry through every later stage).
        y0_sys = state.initializeNewState(*args, **kwargs)
        y0_sys.t = float(state.t)
        f0, r0 = updateStep(state, y0_sys, dt, f, *args, **kwargs)
        ks = [f0]
        rs = [r0]
        y0_ref = get_reference_state(y0_sys)

        # Stage 1: Y_1 = Y_0 + mtilde_1 dt f_0.
        y1_sys = state.initializeNewState(*args, **kwargs)
        y1_sys.t = float(state.t + dt / s)
        y1_sys = updateStateEuler(y1_sys, f0, mtilde1 * dt, copyState=False, **kwargs)

        last_eval_sys = y1_sys
        y_final_sys = y1_sys
        if s == 1:
            # Only f_0 was evaluated; the last evaluated stage is Y_0.
            last_eval_sys = y0_sys
        else:
            f1, r1 = updateStep(state, y1_sys, dt, f, *args, **kwargs)
            ks.append(f1)
            rs.append(r1)
            y1_ref = get_reference_state(y1_sys)
            yp2_ref = y0_ref
            yp1_ref = y1_ref
            fprev = f1
            for j in range(2, s + 1):
                yj_sys = state.initializeNewState(*args, **kwargs)
                yj_sys.t = float(state.t + dt * j / s)
                ref_j = get_reference_state(yj_sys)
                combined = _reference_lincomb(
                    y0_ref,
                    [(y0_ref, 1.0 - mu[j] - nu[j]),
                     (yp1_ref, mu[j]),
                     (yp2_ref, nu[j])])
                for field_name, value in combined.items():
                    setattr(ref_j, field_name, value)
                yj_sys = updateStateEuler(yj_sys, fprev, mtilde[j] * dt, copyState=False, **kwargs)
                if gtilde[j] != 0.0:
                    yj_sys = updateStateEuler(yj_sys, f0, gtilde[j] * dt, copyState=False, **kwargs)
                if j < s:
                    # The final stage Y_s is not evaluated (the step costs s
                    # evaluations, f_0 .. f_{s-1}); its copied fields come from the
                    # last stage that was.
                    fj, rj = updateStep(state, yj_sys, dt, f, *args, **kwargs)
                    ks.append(fj)
                    rs.append(rj)
                    fprev = fj
                    last_eval_sys = yj_sys
                y_final_sys = yj_sys
                yp2_ref = yp1_ref
                yp1_ref = get_reference_state(yj_sys)

        new_state = y_final_sys
        new_state.t = float(state.t + dt)
        stages = [StageResult(aux=r, update=k) for r, k in zip(rs, ks)]
        finalizeSystem(new_state, state, dt, rs, ks, [],
                       *args, lastStageSystem=last_eval_sys, **kwargs)
        return IntegrationResult(state=new_state, stages=stages)


def _make_scheme(family, name):
    def scheme(state, dt, f, *args, **kwargs):
        return integrateRelaxedChebyshev(state, dt, f, family, *args, **kwargs)

    scheme.__name__ = name
    scheme.family = family
    return scheme


RKC1 = _make_scheme('rkc1', 'RKC1')
RKC2 = _make_scheme('rkc2', 'RKC2')
RKL2 = _make_scheme('rkl2', 'RKL2')
