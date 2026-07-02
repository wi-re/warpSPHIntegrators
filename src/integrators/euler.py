from .util import updateStateEuler, updateStateSemiImplicitEuler
import torch
import copy
from .util import split_return, preprocessSystem, postprocessSystem, finalizeSystem, updateStep, initializeSystem
from .specs import IntegrationResult, StageResult
from torch.profiler import record_function
import warnings

def integrateExplicitEuler(state, dt, f, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    if priorStep is not None:
        warnings.warn("Prior step is not used in Explicit Euler integration. Ignoring it.")
    initializeSystem(state, dt, *args, **kwargs)
    with record_function("[Integration] Explicit Euler"):
        with record_function("[Integration] Explicit Euler: Eval"):
            currentState = state.initializeNewState(*args, **kwargs)
            k1, r1 = updateStep(state, currentState, dt, f, *args, **kwargs)
        with record_function("[Integration] Explicit Euler: Update"):
            newState = updateStateEuler(state, k1, dt, **kwargs)
            newState.t = state.t + dt
            finalizeSystem(newState, state, dt, [r1], [k1], [1], *args, **kwargs)
        return IntegrationResult(state=newState, stages=[StageResult(aux=r1, update=k1)])

def integrateSemiImplicitEuler(state, dt, f, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    if priorStep is not None:
        warnings.warn("Prior step is not used in Semi-Implicit Euler integration. Ignoring it.")
    initializeSystem(state, dt, *args, **kwargs)
    with record_function("[Integration] Semi-Implicit Euler"):
        with record_function("[Integration] Semi-Implicit Euler: Eval"):
            currentState = state.initializeNewState(*args, **kwargs)
            k1, r1 = updateStep(state, currentState, dt, f, *args, **kwargs)
        with record_function("[Integration] Semi-Implicit Euler: Update"):
            newState = updateStateSemiImplicitEuler(state, k1, dt, **kwargs)
            newState.t = state.t + dt
            finalizeSystem(newState, state, dt, [r1], [k1], [1], *args, **kwargs)
        return IntegrationResult(state=newState, stages=[StageResult(aux=r1, update=k1)])
