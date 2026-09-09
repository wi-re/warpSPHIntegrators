"""Implicit backward differentiation formulas with JFNK stage closure.

BDF1 is backward Euler. BDF2-BDF5 store one prior-state snapshot per past step in
``StepHistory`` and solve the full multistep residual using the same matrix-free
Newton interface as DIRK: the history enters only as state values, the right-hand
side is evaluated once, at the new grid time, so the standard BDF formula keeps its
claimed order on non-autonomous problems. Cold calls (history shorter than
``order - 1``) use Dormand-Prince 5(4), matching the existing multistep family's
safe high-order fallback until the caller threads a history object.

The coefficients are derived from the BDF order conditions (verified in
``tests/test_bdf.py``); BDF1/2 are A-stable, BDF3-BDF5 are A(alpha)-stable with
cone half-angles 86.03 deg / 73.35 deg / 51.84 deg.
"""

from typing import Optional

from .butcher import DormandPrince
from .fields import (
    flatten_integrated,
    get_reference_state,
    integrated_field_names,
    state_difference,
    state_norm,
)
from .history import HistoryEntry, StepHistory
from .jfnk import JFNKSolver
from .solvers import NonlinearSolver
from .specs import IntegrationResult, StageResult
from .util import finalizeSystem, initializeSystem, reject_prior_step, updateStateEuler, updateStep


def _copy_integrated(destination, source):
    destination_ref = get_reference_state(destination)
    source_ref = get_reference_state(source)
    for field_name in integrated_field_names(source):
        setattr(destination_ref, field_name, getattr(source_ref, field_name))
    return destination


def _linear_combination(template, terms):
    """Return an initialized state whose integrated fields are ``sum(a_i * y_i)``."""
    terms = tuple(terms)
    out = template.initializeNewState()
    out_ref = get_reference_state(out)
    names = integrated_field_names(template)
    for name in names:
        setattr(
            out_ref,
            name,
            sum(weight * getattr(get_reference_state(state), name) for weight, state in terms),
        )
    return out


def getBDFCoefficients(order: int):
    """Return ``(state_weights, derivative_weight)`` newest state first.

    ``y^n = sum_j c_j y^{n-j} + beta * dt * f(t^n, y^n)``; the weights are the
    unique solution of the BDF order conditions (``tests/test_bdf.py`` re-derives
    and cross-checks them), so the standard formula -- history as state values, the
    right-hand side only at the new grid time -- is order ``order`` for
    non-autonomous problems as well.
    """
    coefficients = {
        1: ((1.0,), 1.0),
        2: ((4.0 / 3.0, -1.0 / 3.0), 2.0 / 3.0),
        3: ((18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0), 6.0 / 11.0),
        4: ((48.0 / 25.0, -36.0 / 25.0, 16.0 / 25.0, -3.0 / 25.0), 12.0 / 25.0),
        5: ((300.0 / 137.0, -300.0 / 137.0, 200.0 / 137.0,
             -75.0 / 137.0, 12.0 / 137.0), 60.0 / 137.0),
    }
    try:
        return coefficients[order]
    except KeyError as exc:
        raise ValueError(f'Only BDF1 through BDF5 are implemented, got order={order}') from exc


def BDF(initial_state, dt, f, order: int, *args, history: Optional[StepHistory] = None,
        solver: Optional[NonlinearSolver] = None, **kwargs):
    """One BDF1-BDF5 step, using a high-order starter until history is ready."""
    state_weights, coefficient = getBDFCoefficients(order)
    reject_prior_step(f'BDF{order}', kwargs.pop('priorStep', None))
    needed = order - 1
    history = history if history is not None else StepHistory(maxlen=max(1, needed))
    if len(history) < needed:
        starter_result = DormandPrince(initial_state, dt, f, *args, **kwargs)
        first_stage = starter_result.stages[0]
        starter_history = history.pushed(HistoryEntry(
            t=float(initial_state.t), dt=dt, update=first_stage.update,
            aux=first_stage.aux, state=initial_state.initializeNewState(*args, **kwargs),
        ))
        return IntegrationResult(
            state=starter_result.state,
            stages=starter_result.stages,
            history=starter_history,
        )
    initializeSystem(initial_state, dt, *args, **kwargs)

    previous_states = [entry.state for entry in reversed(history.entries)][:needed]
    base_state = _linear_combination(initial_state, zip(state_weights, [initial_state, *previous_states]))
    base_state.t = float(initial_state.t + dt)

    def step(stage):
        stage.t = float(initial_state.t + dt)
        update, _ = updateStep(initial_state, stage, dt, f, *args, **kwargs)
        return updateStateEuler(base_state, update, coefficient * dt, copyState=True, **kwargs)

    def norm(y_new, y_old):
        return state_norm(state_difference(y_new, y_old), 1e-3, 1e-6, reference=y_old)

    solve_result = (solver or JFNKSolver()).solve(step, base_state, norm, **kwargs.get('solver_opts', {}))
    stage_state = _copy_integrated(base_state, solve_result.y)
    stage_state.t = float(initial_state.t + dt)
    last_update, last_aux = updateStep(initial_state, stage_state, dt, f, *args, **kwargs)

    new_state = solve_result.y
    new_state.t = float(initial_state.t + dt)
    finalizeSystem(new_state, initial_state, dt, last_aux, last_update,
                   lastStageSystem=stage_state, **kwargs)
    new_history = history.pushed(HistoryEntry(
        t=float(initial_state.t), dt=dt, update=last_update, aux=last_aux,
        state=initial_state.initializeNewState(*args, **kwargs),
    ))
    return IntegrationResult(
        state=new_state,
        stages=[StageResult(aux=last_aux, update=last_update)],
        history=new_history,
        solver_diagnostics=solve_result.diagnostics,
    )


def bdfScheme(order: int):
    def scheme(state, dt, f, *args, **kwargs):
        return BDF(state, dt, f, order, *args, **kwargs)
    scheme.__name__ = f'BDF{order}'
    return scheme


BDF1 = bdfScheme(1)
BDF2 = bdfScheme(2)
BDF3 = bdfScheme(3)
BDF4 = bdfScheme(4)
BDF5 = bdfScheme(5)