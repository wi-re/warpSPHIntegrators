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

These explicit one-step methods reject first-stage reuse. Recurrence-derived
stage times preserve their order on non-autonomous problems. Their useful
stability region still concerns diffusion-like negative-real spectra. Affine
combinations are anchored at the initial state to preserve zero-rate fields;
finalization receives the derived derivative weights without extra RHS calls.
"""

from warpSPHIntegrators.schemes.coefficients.chebyshev import _COEFFS, _K, _NAMES, _SMIN, _recurrence_clock_weights, _rkc1_coeffs, _rkc2_coeffs, _rkl2_coeffs, stage_count

import dataclasses

from warpSPHIntegrators.state.lifecycle import finalizeSystem, initializeSystem, reject_prior_step, updateStateEuler, updateStep
from warpSPHIntegrators.state.fields import get_reference_state, field_behavior
from warpSPHIntegrators.state.specs import IntegrationResult, StageResult
from torch.profiler import record_function


# --------------------------------------------------------------------------- #
# Stage coefficients                                                          #
# --------------------------------------------------------------------------- #
#
# Each returns ``(mtilde1, mu, nu, mtilde, gtilde)`` for a given stage count ``s``:
# ``mu[j]``/``nu[j]``/``mtilde[j]``/``gtilde[j]`` are the recurrence weights used at
# stage ``j`` (``2 <= j <= s``), and ``mtilde1`` the first-stage weight. The lists
# are indexed 0..s with entries below the method's minimum left at zero.


#: Real-axis stability interval ``K(s)``: stable for ``dt * |lambda_max| <= K(s)``.
#: Minimum stage count the coefficient construction is valid for.


def _reference_lincomb(template_ref, terms):
    """Affine combination around the initial reference, preserving zero rates.

    ``terms`` is ``[(ref_state, weight), ...]``. Returns ``{field_name: value}`` for
    every integrated field of ``template_ref``. Weights sum to one, so computing
    ``base + sum(weight * (value-base))`` preserves a passive field exactly.
    """
    out = {}
    for f in dataclasses.fields(template_ref):
        if field_behavior(f) != 'integrated':
            continue
        name = f.name
        base = getattr(template_ref, name)
        acc = base
        for ref, weight in terms:
            if weight == 0.0:
                continue
            acc = acc + weight * (getattr(ref, name) - base)
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
    if s is not None:
        from numbers import Integral
        if isinstance(s, bool) or not isinstance(s, Integral) or s < 1:
            raise ValueError(f'{name}: s must be a positive integer')
    if lambda_max is not None:
        import math
        if not math.isfinite(float(lambda_max)) or lambda_max < 0:
            raise ValueError(f'{name}: lambda_max must be finite and nonnegative')
    if s is None:
        if lambda_max is None:
            raise ValueError(
                f"{name}: pass s= (stage count) or lambda_max= (stiffest eigenvalue "
                f"magnitude) to size the super-time-step")
        s = stage_count(abs(dt) * abs(lambda_max), family)
    s = max(int(s), _SMIN[family])
    mtilde1, mu, nu, mtilde, gtilde = _COEFFS[family](s)
    c, weights = _recurrence_clock_weights(s, mtilde1, mu, nu, mtilde, gtilde)

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
        y1_sys.t = float(state.t + dt * c[1])
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
                yj_sys.t = float(state.t + dt * c[j])
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
        finalizeSystem(new_state, state, dt, rs, ks, weights,
                       *args, lastStageSystem=last_eval_sys, **kwargs)
        return IntegrationResult(state=new_state, stages=stages, quadrature=weights)


def _make_scheme(family, name):
    def scheme(state, dt, f, *args, **kwargs):
        return integrateRelaxedChebyshev(state, dt, f, family, *args, **kwargs)

    scheme.__name__ = name
    scheme.family = family
    return scheme


RKC1 = _make_scheme('rkc1', 'RKC1')
RKC2 = _make_scheme('rkc2', 'RKC2')
RKL2 = _make_scheme('rkl2', 'RKL2')
