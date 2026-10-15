"""Shu's strong-stability-preserving multistep methods (SSPRKMSVS), fixed-step.

``SSPRKMSVS32`` (3-step, order 2, SSP coefficient ``C = 1/2``) and
``SSPRKMSVS43`` (4-step, order 3, SSP coefficient ``C = 1/3``), from Shu,
*Efficient time integration for parabolic and hyperbolic PDEs*, J. Comput.
Phys. 169 (2001) 201-228, pulled in from OrdinaryDiffEq (SCHEME_EXPANSION_PLAN
wave 3). The steady-state step is a convex combination of forward-Euler
substeps -- the exact structure that makes the methods strongly stability
preserving:

    SSPRKMSVS32:  u^{n+1} = (3/4) (u^n + 2 dt f^n) + (1/4) u^{n-2}
    SSPRKMSVS43:  u^{n+1} = (16/27) (u^n + 3 dt f^n)
                    + (11/27) (u^{n-3} + (12/11) dt f^{n-3})

Each bracketed term is one Euler substep; the binding substep sets the SSP
coefficient -- ``2 dt`` for SSPRKMSVS32 (so ``C = 1/2``) and ``3 dt`` for
SSPRKMSVS43 (so ``C = 1/3``), the other substeps being shorter and non-binding.
Unlike the Adams family, the formula consumes past *states* (``u^{n-2}`` /
``u^{n-3}``), so the history entries store a state snapshot in addition to the
step-start derivative -- the same ``HistoryEntry.state`` mechanism BDF uses.
SSPRKMSVS43 also reuses the old state's derivative (``f^{n-3}``), so each entry
carries both the snapshot and the derivative.

**History must be threaded** to get the one-evaluation-per-step steady state,
exactly as for the Adams family (``testing.run(..., history=True)`` is the
blessed convention). Called without ``history=``, the driver transparently
re-runs its cold start every step: not numerically wrong (the starter,
Dormand-Prince 5(4), is more accurate than either method), just none of the
multistep saving. ``tests/test_multistep.py`` pins this "safe but expensive,
never silently wrong" fallback as a tested property.

Cold start: Dormand-Prince 5(4) for the first ``steps`` calls, the library's
uniform multistep bootstrap (NOTES.md S3.7 pain point 1). OrdinaryDiffEq
self-starts these with a 2nd-order Heun step; a fixed ``(k - 1)``-step
bootstrap of a ``q``-order starter accumulates only ``O(dt^{q+1})`` starting
error (``k - 1`` is a constant, not ``T/dt``, steps), so a ``p``-order method
needs a starter of order at least ``p - 1``. Heun (``q = 2``) is exactly the
minimum for the 3rd-order method and more than enough for the 2nd-order one,
so OrdinaryDiffEq's choice is correct; the Dormand-Prince starter used here is
a higher-order superset of that minimum, matching every other multistep scheme
in this library. The only difference is a smaller cold-start error constant,
not the order (verified in ``tests/test_multistep.py``).

Fixed-step only: there is no embedded error estimate, so ``result.error is
None`` and adaptive stepping is refused by construction.
"""

from typing import Optional

from torch.profiler import record_function

from warpSPHIntegrators.drivers.history.bdf import _linear_combination
from warpSPHIntegrators.schemes.definitions.explicit import DormandPrince
from warpSPHIntegrators.drivers.runge_kutta import _weighted_update
from warpSPHIntegrators.state.history import HistoryEntry, StepHistory
from warpSPHIntegrators.state.specs import IntegrationResult, StageResult
from warpSPHIntegrators.state.lifecycle import finalizeSystem, initializeSystem, reject_prior_step, updateStep


def _bootstrap_with_state(initialState, dt, f, history: StepHistory, starter, *args, **kwargs):
    """Run one step of `starter` and record the step-start state + derivative.

    Like `multistep._bootstrap`, but the history entry additionally stores
    `state` (a snapshot of `initialState`), because the SSPRKMSVS formulas
    consume past *states*, not just past derivatives. `stages[0]` is
    ``f(t^n, y^n)`` (the starter's first stage, at `t^n` -- the invariant
    `test_state.py::test_first_evaluation_is_at_the_start_of_the_step` holds for
    Dormand-Prince), which is the `update` the entry needs.
    """
    result = starter(initialState, dt, f, *args, **kwargs)
    first = result.stages[0]
    new_history = history.pushed(HistoryEntry(
        t=float(initialState.t), dt=dt, update=first.update, aux=first.aux,
        state=initialState.initializeNewState(*args, **kwargs),
    ))
    return IntegrationResult(state=result.state, stages=result.stages, history=new_history)


def SSPRKMSVS32(initialState, dt, f, *args, history: Optional[StepHistory] = None, **kwargs):
    """One fixed-step SSPRKMSVS32 step (3-step, order 2, SSP ``C = 1/2``).

    Needs the state two steps back; ``history=None`` (or one with fewer than two
    entries) bootstraps from Dormand-Prince 5(4) instead -- see the module
    docstring for why the starter is registered scheme behaviour, not a
    caller-configurable parameter.
    """
    reject_prior_step('SSPRKMSVS32', kwargs.pop('priorStep', None))
    starter = DormandPrince
    steps = 2
    history = history if history is not None else StepHistory(maxlen=steps)

    with record_function('[Integration] SSPRKMSVS32'):
        if len(history) < steps:
            return _bootstrap_with_state(initialState, dt, f, history, starter, *args, **kwargs)

        initializeSystem(initialState, dt, *args, **kwargs)

        currentState = initialState.initializeNewState(*args, **kwargs)
        currentState.t = float(initialState.t)
        k_n, r_n = updateStep(initialState, currentState, dt, f, *args, **kwargs)

        # u^{n-2}: the state snapshot two steps back (entries are newest-last).
        state_nm2 = history.entries[-2].state

        # base = (3/4) u^n + (1/4) u^{n-2}, then add (3/2) dt f^n.
        base = _linear_combination(
            initialState, [(3.0 / 4.0, initialState), (1.0 / 4.0, state_nm2)])
        new_state = _weighted_update(base, [k_n], [3.0 / 2.0], dt, *args, **kwargs)
        new_state.t = float(initialState.t + dt)
        finalizeSystem(new_state, initialState, dt, [r_n], [k_n], [3.0 / 2.0],
                       *args, lastStageSystem=currentState, **kwargs)

        new_history = history.pushed(HistoryEntry(
            t=float(initialState.t), dt=dt, update=k_n, aux=r_n,
            state=initialState.initializeNewState(*args, **kwargs),
        ))
        return IntegrationResult(
            state=new_state, stages=[StageResult(aux=r_n, update=k_n)], history=new_history)


def SSPRKMSVS43(initialState, dt, f, *args, history: Optional[StepHistory] = None, **kwargs):
    """One fixed-step SSPRKMSVS43 step (4-step, order 3, SSP ``C = 1/3``).

    Needs the state and derivative three steps back; ``history=None`` (or one
    with fewer than three entries) bootstraps from Dormand-Prince 5(4) instead
    -- see the module docstring for the starter choice.
    """
    reject_prior_step('SSPRKMSVS43', kwargs.pop('priorStep', None))
    starter = DormandPrince
    steps = 3
    history = history if history is not None else StepHistory(maxlen=steps)

    with record_function('[Integration] SSPRKMSVS43'):
        if len(history) < steps:
            return _bootstrap_with_state(initialState, dt, f, history, starter, *args, **kwargs)

        initializeSystem(initialState, dt, *args, **kwargs)

        currentState = initialState.initializeNewState(*args, **kwargs)
        currentState.t = float(initialState.t)
        k_n, r_n = updateStep(initialState, currentState, dt, f, *args, **kwargs)

        # The entry three steps back carries both u^{n-3} (snapshot) and
        # f^{n-3} (the derivative recorded at that step's start).
        entry_nm3 = history.entries[-3]
        state_nm3 = entry_nm3.state
        k_nm3 = entry_nm3.update

        # base = (16/27) u^n + (11/27) u^{n-3}, then add (16/9) dt f^n and
        # (4/9) dt f^{n-3}  [ (11/27) * (12/11) = 4/9 ].
        base = _linear_combination(
            initialState, [(16.0 / 27.0, initialState), (11.0 / 27.0, state_nm3)])
        new_state = _weighted_update(
            base, [k_n, k_nm3], [16.0 / 9.0, 4.0 / 9.0], dt, *args, **kwargs)
        new_state.t = float(initialState.t + dt)
        finalizeSystem(new_state, initialState, dt, [r_n], [k_n, k_nm3],
                       [16.0 / 9.0, 4.0 / 9.0], *args, lastStageSystem=currentState, **kwargs)

        new_history = history.pushed(HistoryEntry(
            t=float(initialState.t), dt=dt, update=k_n, aux=r_n,
            state=initialState.initializeNewState(*args, **kwargs),
        ))
        return IntegrationResult(
            state=new_state, stages=[StageResult(aux=r_n, update=k_n)], history=new_history)
