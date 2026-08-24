"""Explicit linear multistep: Adams-Bashforth and Adams-Bashforth-Moulton (NOTES.md
S3.6 Phase 1).

`y^{n+1} = y^n + dt * sum_j beta_j * f^{n-j}` is exactly the shape
`butcher._weighted_update` already builds for a Runge-Kutta `b`-weight accumulation --
the "list of updates weighted by a list of coefficients, times dt" primitive does not
care whether the updates came from stages of *this* step or evaluations kept from
*past* steps. The only genuinely new piece is `StepHistory` (NOTES.md S2, Phase 0)
supplying those past evaluations, and a startup strategy for the first `order - 1`
steps, before there is enough history to run the real formula.

**These schemes require `history=` to be threaded across calls to get their claimed
cost.** Unlike every one-step scheme in this library, where `history=`/`priorStep=`
are pure opt-in bookkeeping, a multistep scheme's `history` argument is not optional
plumbing -- it is the only channel through which past derivatives reach it at all.
Call one of these the way every other scheme is called, discarding
`IntegrationResult.history` between steps, and it will transparently keep re-running
its high-order starter forever: not numerically *wrong* (the starter, Dormand-Prince
5(4), is more accurate than any of these, not less), just none of the one-evaluation-
per-step saving that is the entire point of choosing Adams-Bashforth over a
Runge-Kutta method. `tests/test_multistep.py` pins this "safe but expensive, not
silently wrong" fallback down as a tested property, not an accident. The correct
calling convention is exactly what `testing.run(..., history=True)` already does
(NOTES.md S5): thread `result.history` back in as `history=` on the next call.
"""

from typing import Optional

import numpy as np
from torch.profiler import record_function

from .butcher import DormandPrince, _weighted_update
from .history import HistoryEntry, StepHistory
from .specs import IntegrationResult, StageResult
from .util import finalizeSystem, initializeSystem, reject_prior_step, updateStep


def getABCoefficients(order: int) -> np.ndarray:
    """Adams-Bashforth beta weights, newest point first: beta[0] weights f^n."""
    if order == 1:
        return np.array([1.0])
    elif order == 2:
        return np.array([3 / 2, -1 / 2])
    elif order == 3:
        return np.array([23 / 12, -16 / 12, 5 / 12])
    elif order == 4:
        return np.array([55 / 24, -59 / 24, 37 / 24, -9 / 24])
    elif order == 5:
        return np.array([1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720])
    else:
        raise ValueError(f"No Adams-Bashforth coefficients recorded for order {order}")


def getAMCoefficients(order: int) -> np.ndarray:
    """Adams-Moulton corrector weights for an ABM scheme of this `order`.

    `order` points total: the predicted point first, then `order - 1` already-known
    ones (f^n, then progressively older history) -- the pairing that makes ABM_order's
    corrector the same order as its AB_order predictor (NOTES.md S3.6).
    """
    if order == 1:
        return np.array([1.0])
    elif order == 2:
        return np.array([1 / 2, 1 / 2])
    elif order == 3:
        return np.array([5 / 12, 8 / 12, -1 / 12])
    elif order == 4:
        return np.array([9 / 24, 19 / 24, -5 / 24, 1 / 24])
    elif order == 5:
        return np.array([251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720])
    else:
        raise ValueError(f"No Adams-Moulton coefficients recorded for order {order}")


def _bootstrap(initialState, dt, f, history: StepHistory, starter, *args, **kwargs):
    """Run one step of `starter` and record *its* f(t^n, y^n) as this step's history entry.

    Every explicit RK scheme in this library evaluates its first stage at (t^n, y^n)
    (`tableau.c[0] == 0`, the invariant `test_state.py::test_first_evaluation_is_at_the
    _start_of_the_step` holds for every non-implicit scheme) -- so `stages[0]` is
    exactly the Adams-family history entry a multistep formula needs for this step,
    regardless of how many further stages the starter itself takes internally to reach
    its own (much higher) order. Deliberately does *not* pass `history=` through to the
    starter: `butcher.RungeKuttaB`'s own `history=` records `stages[-1]` (built for
    first-stage *reuse*, a different, LAST-stage-shaped need), which is the wrong end
    of the stage list for this purpose.
    """
    result = starter(initialState, dt, f, *args, **kwargs)
    first = result.stages[0]
    new_history = history.pushed(HistoryEntry(t=float(initialState.t), dt=dt, update=first.update, aux=first.aux))
    return IntegrationResult(state=result.state, stages=result.stages, history=new_history)


def AdamsBashforth(initialState, dt, f, order: int, *args,
                    history: Optional[StepHistory] = None, **kwargs):
    """One Adams-Bashforth step of the given `order` (2-5).

    Needs `order - 1` past derivatives; `history=None` (or one with fewer entries
    than that) bootstraps from Dormand-Prince 5(4) instead, exactly the "high-order
    starter" NOTES.md S3.7 pain point 1 found necessary -- a low-order self-starter
    (e.g. one Euler step) was measured to cap AB3/AB4 at order 2 no matter how many
    later steps ran at full order. Not a `starter=` parameter: that pain point's own
    conclusion was "the starter must be registered scheme metadata, not caller
    policy" -- a caller-suppliable starter is exactly the footgun that warns against
    (pick one lower-order than this scheme's own claim, or one whose first stage
    isn't at `t^n`, and the whole run silently caps at the starter's order instead of
    this scheme's).
    """
    reject_prior_step(f'AB{order}', kwargs.pop('priorStep', None))
    starter = DormandPrince
    needed = order - 1
    history = history if history is not None else StepHistory(maxlen=needed)

    with record_function(f"[Integration] AB{order}"):
        if len(history) < needed:
            return _bootstrap(initialState, dt, f, history, starter, *args, **kwargs)

        initializeSystem(initialState, dt, *args, **kwargs)
        beta = getABCoefficients(order)

        currentState = initialState.initializeNewState(*args, **kwargs)
        currentState.t = float(initialState.t)
        k_n, r_n = updateStep(initialState, currentState, dt, f, *args, **kwargs)

        hist_ks = [e.update for e in reversed(history.entries)][:needed]
        ks = [k_n] + hist_ks

        new_state = _weighted_update(initialState, ks, beta, dt, *args, **kwargs)
        new_state.t = float(initialState.t + dt)
        finalizeSystem(new_state, initialState, dt, [r_n], ks, beta,
                       *args, lastStageSystem=currentState, **kwargs)

        new_history = history.pushed(HistoryEntry(t=float(initialState.t), dt=dt, update=k_n, aux=r_n))
        return IntegrationResult(state=new_state, stages=[StageResult(aux=r_n, update=k_n)], history=new_history)


def AdamsBashforthMoulton(initialState, dt, f, order: int, *args,
                          history: Optional[StepHistory] = None, **kwargs):
    """One Adams-Bashforth-Moulton (PECE) step of the given `order` (2-4).

    Predict-Evaluate-Correct-Evaluate with a *fixed* (uniterated) correction, per
    NOTES.md S3.6: the AM corrector uses the predictor's new evaluation in place of the
    true implicit f(t^{n+1}, y^{n+1}), rather than iterating to convergence the way the
    DIRK driver's Picard solve does. 2 evaluations per step (f^n, f at the predicted
    point) against AB's 1 -- the corrector step itself needs no further evaluation,
    since it only recombines values already computed. See `AdamsBashforth`'s
    docstring for why the bootstrap starter (Dormand-Prince 5(4)) is not a
    caller-configurable parameter.
    """
    reject_prior_step(f'ABM{order}', kwargs.pop('priorStep', None))
    starter = DormandPrince
    needed = order - 1
    history = history if history is not None else StepHistory(maxlen=needed)

    with record_function(f"[Integration] ABM{order}"):
        if len(history) < needed:
            return _bootstrap(initialState, dt, f, history, starter, *args, **kwargs)

        initializeSystem(initialState, dt, *args, **kwargs)
        beta = getABCoefficients(order)
        gamma = getAMCoefficients(order)

        currentState = initialState.initializeNewState(*args, **kwargs)
        currentState.t = float(initialState.t)
        k_n, r_n = updateStep(initialState, currentState, dt, f, *args, **kwargs)

        hist_ks = [e.update for e in reversed(history.entries)][:needed]

        with record_function(f"[Integration] ABM{order}: Predict"):
            y_pred = _weighted_update(initialState, [k_n] + hist_ks, beta, dt, *args, **kwargs)
            y_pred.t = float(initialState.t + dt)

        with record_function(f"[Integration] ABM{order}: Evaluate"):
            k_pred, r_pred = updateStep(initialState, y_pred, dt, f, *args, **kwargs)

        with record_function(f"[Integration] ABM{order}: Correct"):
            # order points total: k_pred (new), k_n (current), then order-2 older ones.
            ks_correct = [k_pred, k_n] + hist_ks[:order - 2]
            new_state = _weighted_update(initialState, ks_correct, gamma, dt, *args, **kwargs)
            new_state.t = float(initialState.t + dt)
            # y_pred, not currentState: it is the buffer the *last* evaluation ran on,
            # so it carries this step's `copied` fields (see butcher.RungeKuttaB's own
            # `lastStageState` convention).
            finalizeSystem(new_state, initialState, dt, [r_n, r_pred], ks_correct, gamma,
                           *args, lastStageSystem=y_pred, **kwargs)

        new_history = history.pushed(HistoryEntry(t=float(initialState.t), dt=dt, update=k_n, aux=r_n))
        return IntegrationResult(state=new_state, stages=[StageResult(aux=r_n, update=k_n)], history=new_history)


def adamsBashforthScheme(order: int):
    def scheme(state, dt, f, *args, **kwargs):
        return AdamsBashforth(state, dt, f, order, *args, **kwargs)
    scheme.__name__ = f'AB{order}'
    return scheme


def adamsBashforthMoultonScheme(order: int):
    def scheme(state, dt, f, *args, **kwargs):
        return AdamsBashforthMoulton(state, dt, f, order, *args, **kwargs)
    scheme.__name__ = f'ABM{order}'
    return scheme


AB2 = adamsBashforthScheme(2)
AB3 = adamsBashforthScheme(3)
AB4 = adamsBashforthScheme(4)
AB5 = adamsBashforthScheme(5)
ABM2 = adamsBashforthMoultonScheme(2)
ABM3 = adamsBashforthMoultonScheme(3)
ABM4 = adamsBashforthMoultonScheme(4)
