"""First-stage reuse (``priorStep``): what it costs, per scheme.

Passing ``priorStep=result.stages[-1]`` feeds the last stage of step *n* in as the
first stage ``k0`` of step *n+1*, halving the number of right-hand-side evaluations
for a two-stage scheme. It is standard practice in the SPH literature -- CRKSPH does
exactly this for its second-order scheme -- but **its validity is a property of the
tableau, not of the caller**.

Reuse substitutes ``k0`` with a derivative evaluated at ``(t^n + c_s*dt, Y_s)``
instead of at ``(t^{n+1}, y^{n+1})``. Two things set the damage:

  * If ``c_s == 1`` the times agree and ``Y_s`` differs from ``y^{n+1}`` by
    ``O(dt^{q+1})``, where ``q`` is the order of the method whose weights are the
    last row of ``a``. If ``c_s != 1`` the times disagree at ``O(dt)`` and ``q = 0``.
    When additionally ``a[-1] == b`` the stage *is* ``y^{n+1}``: the tableau is
    **FSAL** and reuse is exact.
  * That perturbation of ``k0`` reaches the update directly if ``b[0] != 0``
    (costing one power of ``dt``), or only through the stage equations if
    ``b[0] == 0`` (costing two).

So the retained global order is ``min(p, q + 1)``, or ``min(p, q + 2)`` when
``b[0] == 0``. This predicts *consistency* order only; reuse also changes the
stability region, because the method becomes a two-step one. A single-stage tableau
degenerates to ``y^{n+1} = y^n + dt*f(y^{n-1})``, which is formally first order but
has no stability region on the imaginary axis, so single-stage schemes refuse reuse
outright rather than reporting an order.

That substitution argument is for *explicit* tableaus, where every stage is a plain
function of already-known states. An implicit (DIRK) stage is not: its value depends
on ``dt`` through the very solve reuse would try to skip. DIRK tableaus are therefore
analysed separately (``dirk_reuse_analysis``), on the implicit FSAL condition -- an
explicit first stage *and* stiff accuracy -- rather than with the formula above.

Public API::

    step_reuse_order(scheme)     -> int | None   order retained, None if unsupported
    supports_step_reuse(scheme)  -> bool         True iff no order is lost
    step_reuse_analysis(scheme)  -> ReuseAnalysis   the above plus `fsal` and a reason
"""

from typing import NamedTuple, Optional

import numpy as np

from .enums import IntegrationSchemeType


class ReuseAnalysis(NamedTuple):
    """What ``priorStep`` reuse costs for one scheme."""

    #: Global convergence order retained under reuse, or None if the scheme does not
    #: implement reuse at all (it will warn and ignore ``priorStep``).
    order: Optional[int]
    #: True when the tableau is FSAL, i.e. the reused stage is exactly
    #: f(t^{n+1}, y^{n+1}) and reuse is not an approximation at all.
    fsal: bool
    #: Human-readable justification, suitable for a warning message.
    reason: str


#: Schemes with no Butcher tableau, where the answer has to be recorded by hand.
#: ``None`` means the scheme does not implement reuse and warns if it is offered one.
HANDROLLED_REUSE = {
    IntegrationSchemeType.symplecticEuler: (
        1, 'the reused stage is f(y^n + (dt/2) k0) at t^n + dt/2, which is O(dt) from y^{n+1}, '
           'and k0 enters the position update directly'),
    IntegrationSchemeType.leapFrog: (
        1, 'the reused stage carries the full-step position x^{n+1} but the old velocity v^n, '
           'so it is O(dt) from y^{n+1}'),
    # Velocity Verlet consumes only the *acceleration* of the reused stage -- the
    # position drift uses the state's own velocity, not the stage's -- and that
    # acceleration was evaluated at the exact x^{n+1}. So for a force depending on
    # position alone, reuse is exact: the classic velocity-Verlet FSAL property.
    # For a velocity-dependent force the reused stage carries v^{n+1/2} instead of
    # v^{n+1}, but velocity Verlet is only first order for such a force to begin
    # with (the scheme is implicit in that case and this is the explicit shortcut),
    # so reuse still costs nothing relative to what the scheme delivers. Measured
    # both ways in tests/test_step_reuse.py.
    IntegrationSchemeType.velocityVerlet: (
        2, 'only the acceleration of the reused stage is consumed, and it was evaluated at the '
           'exact x^{n+1}, so reuse is lossless (velocity-Verlet FSAL property)'),
    IntegrationSchemeType.pefrl: (None, 'PEFRL does not implement first-stage reuse'),
    IntegrationSchemeType.vefrl: (None, 'VEFRL does not implement first-stage reuse'),
    IntegrationSchemeType.tvdRK3: (None, 'TVD RK3 does not implement first-stage reuse'),
    IntegrationSchemeType.tvdRK2: (None, 'TVD RK2 does not implement first-stage reuse'),
    IntegrationSchemeType.semiImplicitEuler: (None, 'Semi-Implicit Euler does not implement first-stage reuse'),
    IntegrationSchemeType.explicitEuler: (None, 'Explicit Euler does not implement first-stage reuse'),
    # Super-timestepping: the stage count `s` (hence the whole recurrence) is a
    # per-step choice, so the previous step's last stage cannot be spliced into this
    # step's first stage. None of the three implement first-stage reuse.
    IntegrationSchemeType.rkc1: (None, 'RKC1 does not implement first-stage reuse (its stage count is a per-step parameter)'),
    IntegrationSchemeType.rkc2: (None, 'RKC2 does not implement first-stage reuse (its stage count is a per-step parameter)'),
    IntegrationSchemeType.rkl2: (None, 'RKL2 does not implement first-stage reuse (its stage count is a per-step parameter)'),
    # Rosenbrock-W: the stages are coupled linear solves against a frozen operator
    # (each K_i = M^{-1} b_i with b_i built from the earlier K_j), and the method is
    # not stiffly accurate (the last stage is not y^{n+1}), so there is no ready-made
    # derivative to splice in.
    IntegrationSchemeType.ros3p: (None, 'ROS3P does not implement first-stage reuse (its stages are coupled linear solves and it is not stiffly accurate)'),
    # ETD2RK: a two-stage exponential integrator (exp-Euler predictor + corrector);
    # it is not stiffly accurate (the last stage is not y^{n+1}), so there is no
    # ready-made derivative to splice in as a first stage.
    IntegrationSchemeType.etd2rk: (None, 'ETD2RK does not implement first-stage reuse (it is not stiffly accurate)'),
    # EXPRB32: a two-stage exponential Rosenbrock method; it is not stiffly
    # accurate (the stage U2 is not y^{n+1} -- the corrector corrects it), so
    # there is no ready-made derivative to splice in as a first stage.
    IntegrationSchemeType.exprb32: (None, 'EXPRB32 does not implement first-stage reuse (it is not stiffly accurate)'),
}

_TOL = 1e-12


def _method_order(a: np.ndarray, c: np.ndarray, w: np.ndarray, max_order: int = 5) -> int:
    """Largest p for which weights ``w`` over nodes ``c`` satisfy the RK order conditions."""
    ac = a @ c
    conds = [
        [(w.sum(), 1.0)],
        [(w @ c, 1 / 2)],
        [(w @ c ** 2, 1 / 3), (w @ ac, 1 / 6)],
        [(w @ c ** 3, 1 / 4), (w @ (c * ac), 1 / 8),
         (w @ (a @ c ** 2), 1 / 12), (w @ (a @ ac), 1 / 24)],
        [(w @ c ** 4, 1 / 5), (w @ (c ** 2 * ac), 1 / 10),
         (w @ (c * (a @ c ** 2)), 1 / 15), (w @ (ac ** 2), 1 / 20),
         (w @ (a @ c ** 3), 1 / 20), (w @ (c * (a @ ac)), 1 / 30),
         (w @ (a @ (c * ac)), 1 / 40), (w @ (a @ (a @ c ** 2)), 1 / 60),
         (w @ (a @ (a @ ac)), 1 / 120)],
    ]
    for p, group in enumerate(conds[:max_order], start=1):
        if any(abs(got - want) > _TOL for got, want in group):
            return p - 1
    return max_order


def tableau_reuse_analysis(tableau, nominal_order: int) -> ReuseAnalysis:
    """Predict the order retained under reuse from a Butcher tableau alone."""
    a = np.asarray(tableau.a, dtype=float)
    c = np.asarray(tableau.c, dtype=float)
    # For an embedded pair, the propagated solution is the first weight vector.
    b = np.asarray(tableau.b[0] if isinstance(tableau.b, tuple) else tableau.b, dtype=float)

    if a.shape[0] == 1:
        return ReuseAnalysis(
            None, False,
            'single-stage tableau: reuse degenerates to the lagged two-step method '
            'y^{n+1} = y^n + dt f(y^{n-1}), which is consistent but has no stability region '
            'on the imaginary axis, so it is refused rather than merely degraded')

    last_at_end = abs(c[-1] - 1.0) < _TOL
    if last_at_end and np.allclose(a[-1], b, atol=_TOL):
        return ReuseAnalysis(
            nominal_order, True,
            'FSAL: c_s == 1 and a[-1] == b, so the reused stage is exactly f(t^{n+1}, y^{n+1})')

    if not last_at_end:
        q = 0
        why = f'c_s = {c[-1]:.4g} != 1, so the reused stage is O(dt) away in time'
    else:
        q = _method_order(a, c, a[-1])
        why = (f'c_s == 1 but a[-1] != b; a[-1] is a method of order {q}, '
               f'so the reused stage sits at y^{{n+1}} + O(dt^{q + 1})')

    gain = 2 if abs(b[0]) < _TOL else 1
    if gain == 2:
        why += '; b[0] == 0, so the stale k0 only enters through the stage equations'
    return ReuseAnalysis(min(nominal_order, q + gain), False, why)


def _dirk_reuse_refusal(a: np.ndarray, explicit_first: bool) -> str:
    """Why a DIRK tableau that fails the FSAL test does not implement reuse here."""
    if a.shape[0] == 1:
        return ('single-stage DIRK: the one stage is implicit (a[0,0] != 0), so reusing it '
                'degenerates to y^{n+1} = y^n + dt f(y^{n-1}) -- consistent but with no '
                'stability region on the imaginary axis, refused rather than merely degraded')
    return ('implicit first stage (a[0,0] != 0): the first stage requires a nonlinear solve, '
            'so a reused k0 cannot be consumed directly')


def dirk_reuse_analysis(tableau, nominal_order: int, stiffly_accurate: bool) -> ReuseAnalysis:
    """Predict the order retained under reuse for a DIRK tableau.

    A DIRK's last stage is the step's unknown endpoint, not a stage of an already-known
    trajectory, so the *explicit*-tableau substitution analysis does not apply. Reuse is
    exact -- and is the only reuse this driver implements -- when both halves of the
    (implicit) FSAL condition hold: an explicit first stage (``a[0,0] == 0``,
    ``c[0] == 0``), so it can consume a ready-made derivative, *and* stiff accuracy
    (``c[-1] == 1``, ``a[-1] == b``), so the previous step's last stage IS ``y^{n+1}``
    and its converged derivative is exactly the ``f(t^{n+1}, y^{n+1})`` this stage needs.

    ``stiffly_accurate`` is the registered metadata -- this analysis is its first
    consumer. The ``c[-1] == 1 and a[-1] == b`` check on the actual tableau is the
    drift guard: a stale ``True`` flag cannot grant reuse on a non-stiffly-accurate
    tableau, and a ``False`` flag cannot withhold it from one that is.
    """
    a = np.asarray(tableau.a, dtype=float)
    c = np.asarray(tableau.c, dtype=float)
    b = np.asarray(tableau.b[0] if isinstance(tableau.b, tuple) else tableau.b, dtype=float)

    explicit_first = abs(a[0, 0]) < _TOL and abs(c[0]) < _TOL
    if not explicit_first:
        return ReuseAnalysis(None, False, _dirk_reuse_refusal(a, explicit_first))

    tableau_sa = abs(c[-1] - 1.0) < _TOL and np.allclose(a[-1], b, atol=_TOL)
    if not stiffly_accurate:
        return ReuseAnalysis(
            None, False,
            'not stiffly accurate per its registered metadata: the last stage is not y^{n+1}, '
            'so a reused k0 would be stale')
    if not tableau_sa:
        return ReuseAnalysis(
            None, False,
            'registered stiffly_accurate but the tableau disagrees (a[-1] != b or c[-1] != 1): '
            'stale metadata, reuse refused')
    return ReuseAnalysis(
        nominal_order, True,
        'stiffly accurate with an explicit first stage: the last stage IS y^{n+1}, so the '
        'reused k0 is exactly f(t^{n+1}, y^{n+1})')


def _tableau_of(scheme):
    """The Butcher tableau behind an explicit registered scheme, or None if it has none."""
    fn = getattr(scheme, 'function', scheme)
    return getattr(fn, 'butcherTableau', None)


def _dirk_tableau_of(scheme):
    """The DIRK tableau behind an implicit registered scheme, or None if it is not a DIRK."""
    fn = getattr(scheme, 'function', scheme)
    return getattr(fn, 'dirkTableau', None)


def _block_tableau_of(scheme):
    """The coupled (block) fully implicit tableau, or None if the scheme is not one."""
    fn = getattr(scheme, 'function', scheme)
    return getattr(fn, 'blockTableau', None)


def step_reuse_analysis(scheme) -> ReuseAnalysis:
    """Full reuse analysis for a registered ``IntegrationScheme``."""
    tableau = _tableau_of(scheme)
    if tableau is not None:
        return tableau_reuse_analysis(tableau, scheme.order)

    dirk_tableau = _dirk_tableau_of(scheme)
    if dirk_tableau is not None:
        return dirk_reuse_analysis(
            dirk_tableau, scheme.order, getattr(scheme, 'stiffly_accurate', False))

    block_tableau = _block_tableau_of(scheme)
    if block_tableau is not None:
        return ReuseAnalysis(
            None, False,
            'coupled (block) tableau: the stage equations are solved jointly as one '
            's-by-s system, so there is no independent first stage to splice a previous '
            "step's derivative into (pass warmStart=result.stages for the block "
            'initial-guess analogue)')

    identifier = getattr(scheme, 'identifier', None)
    if identifier in HANDROLLED_REUSE:
        order, reason = HANDROLLED_REUSE[identifier]
        return ReuseAnalysis(order, False, reason)

    return ReuseAnalysis(None, False, 'no tableau and no recorded reuse behaviour for this scheme')


def step_reuse_order(scheme) -> Optional[int]:
    """Global convergence order retained when ``priorStep`` is supplied to ``scheme``.

    ``None`` means the scheme does not implement reuse and will ignore ``priorStep``
    (with a warning). A value below ``scheme.order`` means reuse is accepted but
    costs convergence order -- which may still be the right trade, since it also
    halves the number of right-hand-side evaluations for a two-stage scheme.
    """
    return step_reuse_analysis(scheme).order


def supports_step_reuse(scheme) -> bool:
    """True iff ``priorStep`` reuse costs this scheme no convergence order."""
    analysis = step_reuse_analysis(scheme)
    return analysis.order is not None and analysis.order >= scheme.order


def is_fsal(scheme) -> bool:
    """True iff the tableau is FSAL, so the reused stage is exact rather than merely harmless."""
    return step_reuse_analysis(scheme).fsal
