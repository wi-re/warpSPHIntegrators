"""Semi-implicit BDF (SBDF2/3) and Crank-Nicolson/Adams-Bashforth (CNAB2).

IMEX linear multistep methods (NOTES.md S3.19, Phase 12): a BDF or
trapezoidal implicit backbone for the stiff part, explicit endpoint
extrapolation for the smooth part. The split is read by capability through
``resolve`` (NOTES S3.12): an ``IMEXRHS`` (or a structured ``RHS`` providing
``explicit``/``implicit``) activates the split, and an ordinary RHS callable
degenerates to the pure-implicit limit -- BDF2 / BDF3 / the trapezoidal rule --
so the standard integrator call convention remains valid, exactly as in
``imex.IMEXEuler`` and ``ark.py``.

SBDFp (order p = 2, 3) is the BDFp backbone with the BDF derivative weight
beta scaling the WHOLE endpoint derivative --

    y^{k+1} = sum_j c_j y^{k-j} + beta*dt* [f_I(t^{k+1}, y^{k+1}) + E(f_E)]

where E is the p-point Lagrange polynomial through the known states,
extrapolated to t^{k+1} --

    SBDF2: E(f_E) =  2 f_E(t^k, y^k)        -    f_E(t^{k-1}, y^{k-1})         (O(h^2))
    SBDF3: E(f_E) =  3 f_E(t^k, y^k)        - 3 f_E(t^{k-1}, y^{k-1})
                                  + f_E(t^{k-2}, y^{k-2})                      (O(h^3))

-- so the combined method is order p. The extrapolation weights are the
Lagrange coefficients through (t^k, t^{k-1}, ...) at t^{k+1}; the order
defects are O(h^3) (SBDF2) and O(h^4) (SBDF3), verified by Taylor expansion
in ``tests/test_imexmultistep.py``, which also pins the measured orders.
Dropping beta on the explicit part amplifies it by 1/beta and makes the
defect O(h) -- a first-order, O(1)-error scheme (measured, and pinned as a
regression guard). Pure-implicit limit: BDFp (SBDF2 A- and L-stable, SBDF3
A(alpha) with the 86.03 deg cone); the pure-explicit limit is the matching
zero-stable explicit p-step method (NOT the Adams-Bashforth method of the
same order -- its stability region is narrower, measured in the same file).

CNAB2 (order 2) is the trapezoidal rule on the implicit part plus the AB2
increment on the explicit part:

    y^{k+1} = y^k + (dt/2) [f_I(t^k, y^k) + f_I(t^{k+1}, y^{k+1})]
                    + dt [ (3/2) f_E(t^k, y^k) - (1/2) f_E(t^{k-1}, y^{k-1}) ]

so its pure-implicit limit is the registered trapezoidal (A-stable, not L)
scheme and its pure-explicit limit is the registered Adams-Bashforth 2.

History and cold start follow the BDF family exactly: ``history=`` is threaded
across calls (``testing.run(..., history=True)``); a cold call (fewer entries
than ``order - 1``) runs Dormand-Prince 5(4) and records a state snapshot, so
the run is never silently wrong -- just at the starter's cost until the
history fills (NOTES S3.7 pain point 1). The explicit part is re-evaluated at
the history state snapshots every step rather than stored: one extra explicit
evaluation per past point, no history-format change.

Diagnostics: the step's one solve produces the ``SolveDiagnostics`` entry,
with ``rhs_evaluations`` increased by the non-solve evaluations of the step
(the explicit-part evaluations, the CNAB2 implicit start-point evaluation,
and the endpoint evaluation recorded for the history) so the Phase 1
work-unit invariant ``total = rhs_evaluations + gmres_iterations`` covers the
full step cost, not just the solve. (The plain BDF family counts the solve
only; SBDF/CNAB pay for the split, and this makes that visible.)
"""

import dataclasses
from typing import Optional

import numpy as np
from torch.profiler import record_function

from .bdf import _copy_integrated, _linear_combination, getBDFCoefficients
from .butcher import DormandPrince
from .fields import get_reference_state, state_difference, state_norm
from .history import HistoryEntry, StepHistory
from .jfnk import JFNKSolver
from .rhs import resolve
from .solvers import NonlinearSolver
from .specs import IntegrationResult, StageResult
from .util import (finalizeSystem, initializeSystem, reject_prior_step,
                  updateStateEuler, updateStep)


def _extrapolation_weights(order: int) -> np.ndarray:
    """Lagrange endpoint extrapolation, newest point first.

    ``f_E(t^{k+1}) ~ sum_j w_j f_E(t^{k-j}, y^{k-j})``: the p-point Lagrange
    polynomial through the known states, evaluated one point ahead. Order p-1
    accurate in the extrapolated value, which is exactly what the order-p
    multistep backbone needs.
    """
    if order == 2:
        return np.array([2.0, -1.0])
    elif order == 3:
        return np.array([3.0, -3.0, 1.0])
    else:
        raise ValueError(f'SBDF is implemented for order 2 and 3, got {order}')


def _starter(initial_state, dt, f, history: StepHistory, *args, **kwargs):
    """One Dormand-Prince 5(4) step recording a BDF-style entry (state snapshot).

    Same cold-start convention as ``bdf.BDF``: the starter is registered-scheme
    metadata, not caller policy (NOTES S3.7 pain point 1), and its first stage
    sits at (t^n, y^n) like every explicit RK scheme here.
    """
    result = DormandPrince(initial_state, dt, f, *args, **kwargs)
    first = result.stages[0]
    new_history = history.pushed(HistoryEntry(
        t=float(initial_state.t), dt=dt, update=first.update, aux=first.aux,
        state=initial_state.initializeNewState(*args, **kwargs)))
    return IntegrationResult(state=result.state, stages=result.stages,
                             history=new_history)


def _explicit_evaluations(initial_state, dt, explicit_rhs, states, *args, **kwargs):
    """Evaluate the explicit part at each known state (newest first).

    Returns ``(updates, auxes, n_evals)``. Each evaluation runs on a fresh
    buffer seeded with the snapshot's integrated fields, at the snapshot's own
    grid time -- the standard stage-buffer lifecycle.
    """
    updates, auxes = [], []
    for state in states:
        buffer = initial_state.initializeNewState(*args, **kwargs)
        _copy_integrated(buffer, state)
        buffer.t = float(state.t)
        update, aux = updateStep(initial_state, buffer, dt, explicit_rhs, *args, **kwargs)
        updates.append(update)
        auxes.append(aux)
    return updates, auxes, len(states)


def SBDF(initial_state, dt, f, order: int, *args, history: Optional[StepHistory] = None,
         solver: Optional[NonlinearSolver] = None, **kwargs):
    """One SBDF2/SBDF3 step: BDFp backbone, explicit endpoint extrapolation."""
    name = f'SBDF{order}'
    state_weights, beta = getBDFCoefficients(order)
    extrap = _extrapolation_weights(order)
    reject_prior_step(name, kwargs.pop('priorStep', None))
    needed = order - 1
    history = history if history is not None else StepHistory(maxlen=max(1, needed))
    if len(history) < needed:
        return _starter(initial_state, dt, f, history, *args, **kwargs)

    with record_function(f"[Integration] {name}"):
        initializeSystem(initial_state, dt, *args, **kwargs)
        resolved = resolve(f, scheme_name=name)
        explicit_rhs, implicit_rhs = resolved.explicit, resolved.implicit

        # Known part: the BDFp state combination, plus the explicit part
        # extrapolated to the endpoint from known points (newest first).
        previous_states = [entry.state for entry in reversed(history.entries)][:needed]
        base_state = _linear_combination(
            initial_state, zip(state_weights, [initial_state, *previous_states]))
        extra_evals = 0
        explicit_auxes = []
        if explicit_rhs is not None:
            updates, auxes, extra_evals = _explicit_evaluations(
                initial_state, dt, explicit_rhs,
                [initial_state, *previous_states], *args, **kwargs)
            explicit_auxes = auxes
            # The BDF derivative weight beta scales the WHOLE endpoint
            # derivative, implicit and explicit alike: without it the explicit
            # part is amplified by 1/beta and the defect is O(h) (measured as
            # an O(1) error plateau, tests/test_imexmultistep.py).
            for weight, update in zip(extrap, updates):
                base_state = updateStateEuler(base_state, update, weight * beta * dt,
                                              copyState=False, **kwargs)
        base_state.t = float(initial_state.t + dt)

        def step(stage):
            stage.t = float(initial_state.t + dt)
            update, _ = updateStep(initial_state, stage, dt, implicit_rhs, *args, **kwargs)
            return updateStateEuler(base_state, update, beta * dt, copyState=True, **kwargs)

        def norm(y_new, y_old):
            return state_norm(state_difference(y_new, y_old), 1e-3, 1e-6, reference=y_old)

        solve_result = (solver or JFNKSolver()).solve(
            step, base_state, norm, **kwargs.get('solver_opts', {}))

        # Endpoint evaluation of the combined f (history entry + StageResult),
        # the same single evaluation the BDF driver makes.
        stage_state = _copy_integrated(base_state, solve_result.y)
        stage_state.t = float(initial_state.t + dt)
        last_update, last_aux = updateStep(initial_state, stage_state, dt, f, *args, **kwargs)
        extra_evals += 1

        new_state = solve_result.y
        new_state.t = float(initial_state.t + dt)
        diag = solve_result.diagnostics
        if diag is not None:
            diag = dataclasses.replace(diag, rhs_evaluations=diag.rhs_evaluations + extra_evals)
        finalizeSystem(new_state, initial_state, dt, (explicit_auxes, last_aux),
                       last_update, lastStageSystem=stage_state, **kwargs)
        new_history = history.pushed(HistoryEntry(
            t=float(initial_state.t), dt=dt, update=last_update, aux=last_aux,
            state=initial_state.initializeNewState(*args, **kwargs)))
        return IntegrationResult(
            state=new_state,
            stages=[StageResult(aux=(explicit_auxes, last_aux), update=last_update)],
            history=new_history, solver_diagnostics=diag)


def CNAB2(initial_state, dt, f, *args, history: Optional[StepHistory] = None,
          solver: Optional[NonlinearSolver] = None, **kwargs):
    """One CNAB2 step: trapezoidal on the implicit part, AB2 on the explicit part."""
    name = 'CNAB2'
    reject_prior_step(name, kwargs.pop('priorStep', None))
    needed = 1
    history = history if history is not None else StepHistory(maxlen=1)
    if len(history) < needed:
        return _starter(initial_state, dt, f, history, *args, **kwargs)

    with record_function(f"[Integration] {name}"):
        initializeSystem(initial_state, dt, *args, **kwargs)
        resolved = resolve(f, scheme_name=name)
        explicit_rhs, implicit_rhs = resolved.explicit, resolved.implicit

        # Known part: y^k + (dt/2) f_I(t^k, y^k) + dt[(3/2) f_E^k - (1/2) f_E^{k-1}].
        previous_states = [entry.state for entry in reversed(history.entries)][:needed]
        base_state = _copy_integrated(
            initial_state.initializeNewState(*args, **kwargs), initial_state)
        extra_evals = 0
        explicit_auxes = []
        start_buf = initial_state.initializeNewState(*args, **kwargs)
        start_buf.t = float(initial_state.t)
        implicit_start, _ = updateStep(initial_state, start_buf, dt, implicit_rhs, *args, **kwargs)
        base_state = updateStateEuler(base_state, implicit_start, 0.5 * dt,
                                      copyState=False, **kwargs)
        extra_evals += 1
        if explicit_rhs is not None:
            updates, auxes, extra_evals = _explicit_evaluations(
                initial_state, dt, explicit_rhs,
                [initial_state, *previous_states], *args, **kwargs)
            explicit_auxes = auxes
            for weight, update in zip([3.0 / 2.0, -1.0 / 2.0], updates):
                base_state = updateStateEuler(base_state, update, weight * dt,
                                              copyState=False, **kwargs)
        base_state.t = float(initial_state.t + dt)

        def step(stage):
            stage.t = float(initial_state.t + dt)
            update, _ = updateStep(initial_state, stage, dt, implicit_rhs, *args, **kwargs)
            return updateStateEuler(base_state, update, 0.5 * dt, copyState=True, **kwargs)

        def norm(y_new, y_old):
            return state_norm(state_difference(y_new, y_old), 1e-3, 1e-6, reference=y_old)

        solve_result = (solver or JFNKSolver()).solve(
            step, base_state, norm, **kwargs.get('solver_opts', {}))

        stage_state = _copy_integrated(base_state, solve_result.y)
        stage_state.t = float(initial_state.t + dt)
        last_update, last_aux = updateStep(initial_state, stage_state, dt, f, *args, **kwargs)
        extra_evals += 1

        new_state = solve_result.y
        new_state.t = float(initial_state.t + dt)
        diag = solve_result.diagnostics
        if diag is not None:
            diag = dataclasses.replace(diag, rhs_evaluations=diag.rhs_evaluations + extra_evals)
        finalizeSystem(new_state, initial_state, dt, (explicit_auxes, last_aux),
                       last_update, lastStageSystem=stage_state, **kwargs)
        new_history = history.pushed(HistoryEntry(
            t=float(initial_state.t), dt=dt, update=last_update, aux=last_aux,
            state=initial_state.initializeNewState(*args, **kwargs)))
        return IntegrationResult(
            state=new_state,
            stages=[StageResult(aux=(explicit_auxes, last_aux), update=last_update)],
            history=new_history, solver_diagnostics=diag)


def sbdfScheme(order: int):
    def scheme(state, dt, f, *args, **kwargs):
        return SBDF(state, dt, f, order, *args, **kwargs)
    scheme.__name__ = f'SBDF{order}'
    return scheme


SBDF2 = sbdfScheme(2)
SBDF3 = sbdfScheme(3)

CNAB2.__name__ = 'CNAB2'
