"""Implicit backward differentiation formulas with JFNK stage closure.

BDF1 is backward Euler. BDF2 stores one prior-state snapshot in ``StepHistory`` and
solves its full multistep residual using the same matrix-free Newton interface as
DIRK. Cold BDF2 calls use Dormand-Prince 5(4), matching the existing multistep
family's safe high-order fallback until the caller threads a history object.
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


def BDF(initial_state, dt, f, order: int, *args, history: Optional[StepHistory] = None,
        solver: Optional[NonlinearSolver] = None, **kwargs):
    """One BDF1 or BDF2 step, using BDF1 until BDF2 has a previous state."""
    if order not in (1, 2):
        raise ValueError(f'Only BDF1 and BDF2 are implemented, got order={order}')
    reject_prior_step(f'BDF{order}', kwargs.pop('priorStep', None))
    history = history if history is not None else StepHistory(maxlen=1)
    if order == 2 and history.latest is None:
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

    previous = history.latest.state if order == 2 and history.latest is not None else None
    effective_order = 2 if previous is not None else 1
    if effective_order == 1:
        base_state = initial_state.initializeNewState(*args, **kwargs)
        coefficient = 1.0
    else:
        base_state = _linear_combination(initial_state, ((4.0 / 3.0, initial_state), (-1.0 / 3.0, previous)))
        coefficient = 2.0 / 3.0
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
    )


def bdfScheme(order: int):
    def scheme(state, dt, f, *args, **kwargs):
        return BDF(state, dt, f, order, *args, **kwargs)
    scheme.__name__ = f'BDF{order}'
    return scheme


BDF1 = bdfScheme(1)
BDF2 = bdfScheme(2)