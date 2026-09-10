"""First-order IMEX Euler with explicit RHS splitting and JFNK closure."""

from typing import Optional

from .fields import get_reference_state, integrated_field_names, state_difference, state_norm
from .jfnk import JFNKSolver
from .rhs import resolve
from .solvers import NonlinearSolver
from .specs import IntegrationResult, StageResult
from .util import finalizeSystem, initializeSystem, reject_prior_step, updateStateEuler, updateStep


def _copy_integrated(destination, source):
    destination_ref = get_reference_state(destination)
    source_ref = get_reference_state(source)
    for field_name in integrated_field_names(source):
        setattr(destination_ref, field_name, getattr(source_ref, field_name))
    return destination


def IMEXEuler(initial_state, dt, f, *args, solver: Optional[NonlinearSolver] = None, **kwargs):
    """One IMEX Euler step.

    Pass ``IMEXRHS(explicit=..., implicit=...)`` as ``f`` to split the right-hand
    side. Passing an ordinary existing RHS callable treats its complete update as
    implicit, so the standard integrator call convention remains valid.
    """
    reject_prior_step('IMEX Euler', kwargs.pop('priorStep', None))
    kwargs.pop('history', None)
    initializeSystem(initial_state, dt, *args, **kwargs)

    # The additive split is read by capability (NOTES S3.12), not by type: a plain
    # callable resolves to (explicit=None, implicit=f) -- fully implicit -- exactly
    # as the old isinstance dispatch did, so bare-callable trajectories are unchanged.
    resolved = resolve(f, scheme_name='IMEX Euler')
    explicit_rhs, implicit_rhs = resolved.explicit, resolved.implicit

    explicit_aux = None
    base_state = initial_state.initializeNewState(*args, **kwargs)
    if explicit_rhs is not None:
        explicit_stage = initial_state.initializeNewState(*args, **kwargs)
        explicit_stage.t = float(initial_state.t)
        explicit_update, explicit_aux = updateStep(
            initial_state, explicit_stage, dt, explicit_rhs, *args, **kwargs
        )
        base_state = updateStateEuler(initial_state, explicit_update, dt, copyState=True, **kwargs)
    base_state.t = float(initial_state.t + dt)

    def step(stage):
        stage.t = float(initial_state.t + dt)
        implicit_update, _ = updateStep(initial_state, stage, dt, implicit_rhs, *args, **kwargs)
        return updateStateEuler(base_state, implicit_update, dt, copyState=True, **kwargs)

    def norm(y_new, y_old):
        return state_norm(state_difference(y_new, y_old), 1e-3, 1e-6, reference=y_old)

    solve_result = (solver or JFNKSolver()).solve(step, base_state, norm, **kwargs.get('solver_opts', {}))
    last_stage = _copy_integrated(base_state, solve_result.y)
    last_stage.t = float(initial_state.t + dt)
    implicit_update, implicit_aux = updateStep(initial_state, last_stage, dt, implicit_rhs, *args, **kwargs)

    new_state = solve_result.y
    new_state.t = float(initial_state.t + dt)
    aux = (explicit_aux, implicit_aux)
    finalizeSystem(new_state, initial_state, dt, aux, implicit_update,
                   lastStageSystem=last_stage, **kwargs)
    return IntegrationResult(
        state=new_state, stages=[StageResult(aux=aux, update=implicit_update)],
        solver_diagnostics=solve_result.diagnostics)


IMEXEuler.__name__ = 'IMEXEuler'