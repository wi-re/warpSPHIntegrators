"""Adaptive step-size control and dense output (roadmap Phase 11, NOTES.md S3.17).

This module is a *set of helpers, not a solver*: nothing here owns a step loop.
The caller drives the accept/reject loop (NOTES.md S3.8's "no solver contract
without a downstream" caution), and the two primitives it needs are:

- :func:`estimate_error_norm` -- turn the embedded-pair estimate carried on an
  :class:`~warpSPHIntegrators.specs.IntegrationResult` into a single
  dimensionless number, where ``1.0`` means "the error is exactly at the
  tolerance".
- :func:`propose_dt` -- the classic predictive step-size controller (Hairer,
  Noerssett & Wanner, *Solving Ordinary Differential Equations I*, Sec. II.4;
  the same formula SciPy's ``RungeKutta._select_step`` applies):

      dt_new = dt * clamp( safety * (target / error_norm) ** (1 / order),
                           growth_min, growth_max ),

  with the zero-error limit ``dt_new = dt * growth_max``. ``order`` is the power
  ``q`` that the error estimate scales as ``h^q``. For the (p, p-1) embedded
  pairs and the built-in estimates (ROS3P, EXPRB32) that is the scheme's
  ``order``; TR-BDF2 is the one exception -- its published SUNDIALS pair makes
  the estimate ``O(h**(order+1))``, so it gets ``order + 1`` (see
  ``tests/test_embedded.py::test_error_estimate_has_the_embedded_order``).

Dense output: :func:`dormand_prince_dense_output`, Shampine's quartic
continuous extension of the Dormand-Prince 5(4) pair -- the only published
dense output in this library (the other estimate-emitting schemes have no
citable coefficients). The coefficients are those implemented in SciPy 1.18.0
(``scipy/integrate/_ivp/rk.py``, class ``RK45``, matrix ``P``; Shampine,
"Some Practical Runge-Kutta Formulas", Mathematics of Computation 46 (1986),
135-150, optimum ``c_6`` choice). Row ``i`` of ``P`` weights stage derivative
``k_i`` of the ``DormandPrince`` tableau in the order ``RungeKuttaB`` emits
them (``k7`` is the FSAL stage at ``c = 1``). Verified against the continuous
order conditions through order 4 (interior error ``O(dt**5)``) and exact
endpoint reproduction; see NOTES.md S3.17.

Multistep schemes (BDF1-5, AM2-4, AB2-5, ABM2-4) emit no error estimate and
``StepHistory`` restarts on any ``dt`` change, so adaptive stepping with them
is *refused by construction*, not just unimplemented: the controller applies to
the one-step estimate-emitting schemes only (NOTES.md S3.17).
"""

import numpy as np

from .butcher import _weighted_update
from .fields import state_norm
from .specs import IntegrationResult

__all__ = ['estimate_error_norm', 'propose_dt', 'dormand_prince_dense_output']


def estimate_error_norm(result: IntegrationResult, rtol: float = 1e-3,
                        atol: float = 1e-6) -> float:
    """The embedded-pair estimate on ``result`` as one dimensionless number.

    The Hairer-Wanner weighted RMS that ``fields.state_norm`` implements,
    applied to the propagated-minus-embedded difference (``result.error``)
    scaled against the propagated solution (``result.state``):
    ``sqrt(mean((e_i / (atol + rtol*|y_i|))**2))`` over the integrated fields.
    ``1.0`` means the error is exactly at the tolerance, which is the ``target``
    :func:`propose_dt` aims for by default.

    Raises ``ValueError`` when ``result.error`` is ``None``: the scheme emits no
    estimate (every non-embedded one-step scheme, and the whole multistep
    family -- which a variable step would invalidate anyway).
    """
    if result.error is None:
        raise ValueError(
            'estimate_error_norm needs an estimate-emitting scheme '
            '(result.error is not None); this result carries none. The estimate '
            'emitters are the embedded pairs (Bogacki-Shampine 3(2), '
            'Dormand-Prince 5(4), Cash-Karp 5(4), TR-BDF2, ESDIRK3(2)4L[2]SA, '
            'ESDIRK4(3)6L[2]SA, ARK3(2)4L[2]SA, ARK4(3)6L[2]SA) and the '
            'built-in estimates (ROS3P, EXPRB32); see tests/test_embedded.py.')
    return state_norm(result.error, rtol=rtol, atol=atol, reference=result.state)


def propose_dt(error_norm: float, dt: float, order: float, *, target: float = 1.0,
               safety: float = 0.9, growth_min: float = 0.2,
               growth_max: float = 5.0) -> float:
    """Predictive step-size controller: ``dt`` scaled by
    ``clamp(safety * (target / error_norm) ** (1 / order))``.

    ``error_norm`` is :func:`estimate_error_norm` (``1.0`` = at the tolerance),
    ``dt`` the step that just produced it, and ``order`` the power ``q`` that
    the estimate scales as ``h^q`` -- the scheme's ``order`` for every
    estimate emitter except TR-BDF2 (whose published pair makes the estimate
    ``O(h**(order+1))``; pass ``order + 1`` there).

    ``error_norm == 0`` (the pair sees no error, e.g. a zero right-hand side)
    grows to ``dt * growth_max`` rather than dividing by zero. The clamp keeps
    one pathological step from rescaling the run by more than an order of
    magnitude in either direction. After a *rejected* step a caller should
    additionally cap the factor at 1 (never grow on a rejection) -- that is a
    property of the loop, not of this helper.
    """
    if dt <= 0:
        raise ValueError(f'dt must be positive, got {dt}')
    if order <= 0:
        raise ValueError(f'order must be positive, got {order}')
    if error_norm < 0:
        raise ValueError(f'error_norm must be non-negative, got {error_norm}')
    if target <= 0:
        raise ValueError(f'target must be positive, got {target}')
    if safety <= 0:
        raise ValueError(f'safety must be positive, got {safety}')
    if growth_min <= 0 or growth_max < growth_min:
        raise ValueError(
            f'need 0 < growth_min <= growth_max, got {growth_min} / {growth_max}')
    if error_norm == 0.0:
        return dt * growth_max
    factor = safety * (target / error_norm) ** (1.0 / order)
    return dt * min(growth_max, max(growth_min, factor))


#: Propagated (fifth-order) weights of the DormandPrince tableau (butcher.py),
#: restated here so the ``theta = 1`` endpoint can be re-applied bit-for-bit
#: without importing the scheme object. tests/test_adaptive.py pins equality
#: against ``getButcherTableau('DormandPrince').b[0]`` so the two cannot drift.
_DP5_B_MAIN = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])

#: Shampine (1986) quartic continuous extension of the Dormand-Prince 5(4)
#: pair, as implemented in SciPy 1.18.0 (scipy/integrate/_ivp/rk.py, RK45.P).
#: Row i: P_i(theta) = p1*theta + p2*theta**2 + p3*theta**3 + p4*theta**4,
#: weighting stage derivative k_i (i = 1..7, k7 the FSAL stage at c = 1).
_DP5_DENSE_P = np.array([
    [1, -8048581381/2820520608, 8663915743/2820520608,
     -12715105075/11282082432],
    [0, 0, 0, 0],
    [0, 131558114200/32700410799, -68118460800/10900136933,
     87487479700/32700410799],
    [0, -1754552775/470086768, 14199869525/1410260304,
     -10690763975/1880347072],
    [0, 127303824393/49829197408, -318862633887/49829197408,
     701980252875/199316789632],
    [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
    [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])


def dormand_prince_dense_output(state, stages, dt: float, theta: float,
                                *args, **kwargs):
    """Quartic dense output for one Dormand-Prince 5(4) step (Shampine 1986).

    ``state`` is the step's initial state, ``stages`` the
    ``IntegrationResult.stages`` of that step (seven ``StageResult``s whose
    ``.update`` is ``k1``..``k7``, ``k7`` the FSAL stage at ``c = 1``), ``dt``
    the step size, and ``theta`` the query position in ``[0, 1]`` relative to
    the step (``t = state.t + theta*dt``). Returns a state of the same type:

    - ``theta = 0`` returns ``state`` itself (no copy, no arithmetic);
    - ``theta = 1`` re-applies the propagated weights and is bit-for-bit
      identical to the step's returned state, given the same ``*args,
      **kwargs`` the step was run with;
    - ``0 < theta < 1`` evaluates Shampine's quartic
      ``state + dt * sum_i P_i(theta) * k_i`` through the same additive-update
      path the step itself uses (``butcher._weighted_update``).

    The quartic satisfies the continuous order conditions through order 4, so
    the interior error is ``O(dt**5)`` for a fixed ``theta`` (NOTES.md S3.17).
    This is the only published dense output in the library: the other
    estimate-emitting schemes have no citable coefficients, so they get none.
    """
    if len(stages) != 7:
        raise ValueError(
            f'dormand_prince_dense_output needs the seven stages of a '
            f'DormandPrince step, got {len(stages)} -- this is the only '
            'published dense output in the library (the other estimate-'
            'emitting schemes have no citable coefficients)')
    if any(s.update is None for s in stages):
        raise ValueError('every stage must carry its derivative (StageResult.update)')
    if dt == 0:
        raise ValueError(f'dt must be nonzero, got {dt}')
    if theta < 0 or theta > 1:
        raise ValueError(f'theta must be in [0, 1], got {theta}')
    if theta == 0:
        return state
    if theta == 1:
        weights = _DP5_B_MAIN
    else:
        weights = (_DP5_DENSE_P[:, 0] * theta
                   + _DP5_DENSE_P[:, 1] * theta ** 2
                   + _DP5_DENSE_P[:, 2] * theta ** 3
                   + _DP5_DENSE_P[:, 3] * theta ** 4)
    ks = [s.update for s in stages]
    new_state = _weighted_update(state, ks, weights, dt, *args, **kwargs)
    new_state.t = float(state.t + theta * dt)
    return new_state
