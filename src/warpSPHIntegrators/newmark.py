"""Newmark-beta implementation for second-order systems.

This is a small, repo-native version of the classical Newmark scheme:

    x_{n+1} = x_n + dt * v_n + dt^2 * ((1/2 - beta) * a_n + beta * a_{n+1})
    v_{n+1} = v_n + dt * ((1 - gamma) * a_n + gamma * a_{n+1})

with the implicit unknown `a_{n+1}` closed by the same fixed-point iteration pattern
used elsewhere in the repo's implicit-stage machinery.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from .fields import get_reference_state, get_tagged_attr, state_difference, state_norm
from .history import HistoryEntry
from .jfnk import JFNKSolver
from .solvers import FixedPointSolver, NonlinearSolver
from .specs import IntegrationResult, StageResult, explicit_step
from .util import (
    applyQuantityUpdate,
    finalizeSystem,
    initializeSystem,
    reject_prior_step,
    updateStep,
    verbosePrint,
)


def _validate_newmark_params(beta: float, gamma: float) -> None:
    if beta < 0.0 or beta > 1.0:
        raise ValueError(f'Newmark beta must lie in [0, 1], got {beta!r}')
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f'Newmark gamma must lie in [0, 1], got {gamma!r}')


def _acceleration_of(update: Any) -> Any:
    """Return the acceleration-like term from an rhs update object."""
    if update is None:
        raise ValueError('Newmark requires a rhs that returns a velocity derivative')
    if hasattr(update, 'dudt'):
        return getattr(update, 'dudt')
    try:
        return get_tagged_attr(update, tag='velocity_derivative')
    except LookupError:
        pass
    try:
        return get_tagged_attr(update, tag='acceleration')
    except LookupError as exc:
        raise ValueError(
            'Newmark expects the rhs update to expose a velocity derivative under either '
            '`dudt` or the `velocity_derivative` tag.'
        ) from exc


def _rhs_update(initial_state: Any, state: Any, dt: float, f, *args, **kwargs) -> Tuple[Any, Any]:
    """Evaluate the rhs and return its full update plus auxiliary payload."""
    return updateStep(initial_state, state, dt, f, *args, **kwargs)


def _rhs_acceleration(initial_state: Any, state: Any, dt: float, f, *args, **kwargs) -> Tuple[Any, Any]:
    """Evaluate the rhs and return the acceleration plus auxiliary payload."""
    update, aux = _rhs_update(initial_state, state, dt, f, *args, **kwargs)
    return _acceleration_of(update), aux


def _newmark_step(base_state: Any, guess: Any, dt: float, f,
                 beta: float, gamma: float, previous_update: Any, *args, **kwargs):
    """One fixed-point update for the Newmark stage equation."""
    out = base_state.initializeNewState(*args, **kwargs)
    s_prev = get_reference_state(base_state)
    out_ref = get_reference_state(out)

    guess.t = float(base_state.t + dt)
    guess_update, _ = _rhs_update(base_state, guess, dt, f, *args, **kwargs)
    a_prev = _acceleration_of(previous_update)
    a_guess = _acceleration_of(guess_update)

    x_next = s_prev.x + dt * s_prev.u + dt * dt * ((0.5 - beta) * a_prev + beta * a_guess)
    u_next = s_prev.u + dt * ((1.0 - gamma) * a_prev + gamma * a_guess)

    out_ref.x = x_next
    out_ref.u = u_next
    # Newmark supplies the position/velocity update. Other integrated quantities
    # still need a second-order endpoint update so fields such as energy-driven
    # stiffness stay coupled to the nonlinear solve.
    applyQuantityUpdate(out, previous_update, explicit_step(0.5 * dt), **kwargs)
    applyQuantityUpdate(out, guess_update, explicit_step(0.5 * dt), **kwargs)
    out.t = float(base_state.t + dt)
    return out


def newmark(state, dt, f, *args,
            beta: float = 0.25,
            gamma: float = 0.5,
            solver: Optional[NonlinearSolver] = None,
            **kwargs):
    """One Newmark-beta step for second-order dynamics.

    The default choice `beta=0.25, gamma=0.5` is the constant-average-acceleration
    scheme; other values are accepted as long as they lie in the standard admissible
    range for a second-order Newmark method.
    """
    _validate_newmark_params(beta, gamma)

    history = kwargs.pop('history', None)
    reject_prior_step('Newmark', kwargs.pop('priorStep', None))
    initializeSystem(state, dt, *args, **kwargs)

    previous_stage = state.initializeNewState(*args, **kwargs)
    previous_stage.t = float(state.t)
    previous_update, aux = _rhs_update(state, previous_stage, dt, f, *args, **kwargs)
    a_prev = _acceleration_of(previous_update)

    def step(Y):
        return _newmark_step(state, Y, dt, f, beta, gamma, previous_update, *args, **kwargs)

    def norm(y_new, y_old):
        return state_norm(state_difference(y_new, y_old), 1e-3, 1e-6, reference=y_old)

    solver = solver or JFNKSolver()
    solver_opts = kwargs.get('solver_opts', {})
    if kwargs.get('verbose', False):
        verbosePrint(True, '[Integrator] Newmark: solving endpoint acceleration')

    y0 = state.initializeNewState(*args, **kwargs)
    s0 = get_reference_state(y0)
    s_state = get_reference_state(state)
    s0.x = s_state.x + dt * s_state.u
    s0.u = s_state.u + dt * a_prev
    y0.t = float(state.t + dt)

    result = solver.solve(step, y0, norm, **solver_opts)
    new_state = result.y
    new_state.t = float(state.t + dt)

    last_stage = new_state.initializeNewState(*args, **kwargs)
    last_stage.t = float(new_state.t)
    last_update, last_aux = _rhs_update(state, last_stage, dt, f, *args, **kwargs)
    last_r = last_aux if last_aux is not None else aux
    finalizeSystem(new_state, state, dt, last_r, last_update,
                   lastStageSystem=last_stage, **kwargs)

    if history is not None:
        entry = HistoryEntry(t=float(state.t), dt=dt, update=last_update, aux=last_r)
        history = history.pushed(entry)

    return IntegrationResult(
        state=new_state,
        stages=[StageResult(aux=last_r, update=last_update)],
        history=history,
        solver_diagnostics=result.diagnostics,
    )


def newmark_average_acceleration(state, dt, f, *args, **kwargs):
    """Convenience alias for the classic average-acceleration Newmark method."""
    return newmark(state, dt, f, *args, beta=0.25, gamma=0.5, **kwargs)


def newmark_linear_acceleration(state, dt, f, *args, **kwargs):
    """Convenience alias for the linear-acceleration Newmark variant."""
    return newmark(state, dt, f, *args, beta=1.0 / 6.0, gamma=0.5, **kwargs)


__all__ = ['newmark', 'newmark_average_acceleration', 'newmark_linear_acceleration']
