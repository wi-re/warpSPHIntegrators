from .util import (
    finalizeSystem,
    initializeSystem,
    reject_prior_step,
    updateStateEuler,
    updateStateSemiImplicitEuler,
    updateStep,
)
from .specs import IntegrationResult, StageResult
from .drift import drift_fields_active, drift_rates, nonkinematic_part, set_drift_fields, velocity_of
from torch.profiler import record_function

def integrateExplicitEuler(state, dt, f, *args, **kwargs):
    reject_prior_step('Explicit Euler', kwargs.pop('priorStep', None))
    initializeSystem(state, dt, *args, **kwargs)
    with record_function("[Integration] Explicit Euler"):
        with record_function("[Integration] Explicit Euler: Eval"):
            currentState = state.initializeNewState(*args, **kwargs)
            currentState.t = float(state.t)
            k1, r1 = updateStep(state, currentState, dt, f, *args, **kwargs)
        with record_function("[Integration] Explicit Euler: Update"):
            newState = updateStateEuler(state, k1, dt, **kwargs)
            newState.t = float(state.t + dt)
            finalizeSystem(newState, state, dt, [r1], [k1], [1],
                           *args, lastStageSystem=currentState, **kwargs)
        return IntegrationResult(state=newState, stages=[StageResult(aux=r1, update=k1)])

def integrateSemiImplicitEuler(state, dt, f, *args, **kwargs):
    reject_prior_step('Semi-Implicit Euler', kwargs.pop('priorStep', None))
    initializeSystem(state, dt, *args, **kwargs)
    with record_function("[Integration] Semi-Implicit Euler"):
        with record_function("[Integration] Semi-Implicit Euler: Eval"):
            currentState = state.initializeNewState(*args, **kwargs)
            currentState.t = float(state.t)
            k1, r1 = updateStep(state, currentState, dt, f, *args, **kwargs)
        with record_function("[Integration] Semi-Implicit Euler: Update"):
            newState = updateStateSemiImplicitEuler(state, k1, dt, **kwargs)
            if drift_fields_active(state, **kwargs):
                # Drift fields (drift.py) move with the velocity the positions
                # drift with, v^{n+1}: symplectic Euler on the (q, v) pair instead
                # of forward Euler (one velocity-linear evaluation at x^n).
                set_drift_fields(newState, state,
                                 (dt, drift_rates(currentState, velocity_of(newState), r1, **kwargs)),
                                 (dt, nonkinematic_part(state, k1)))
            newState.t = float(state.t + dt)
            finalizeSystem(newState, state, dt, [r1], [k1], [1],
                           *args, lastStageSystem=currentState, **kwargs)
        return IntegrationResult(state=newState, stages=[StageResult(aux=r1, update=k1)])
