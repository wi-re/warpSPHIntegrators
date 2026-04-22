import copy
import torch
from typing import NamedTuple, Optional, List, Union

from integrators.fields import get_tagged_attr

from .specs import (
    ComponentUpdateSpec,
    PositionUpdateSpec,
    StateBlend,
    blend_state,
    explicit_step,
    semi_implicit_position_step,
    verlet_position_step,
)


def verbosePrint(verbose, *args):
    if verbose:
        print(*args)

def updateStateEuler(systemState_, systemUpdate, dt, copyState = True, **kwargs):
    if copyState:
        systemState = systemState_.initializeNewState(**kwargs)
    else:
        systemState = systemState_
    return applyStateUpdate(systemState, systemUpdate, explicit_step(dt), **kwargs)

def updateStateSemiImplicitEuler(systemState_, systemUpdate, dt, copyState = True, **kwargs):
    if copyState:
        systemState = systemState_.initializeNewState(**kwargs)
    else:
        systemState = systemState_
    applyVelocityUpdate(systemState, systemUpdate, explicit_step(dt), semiImplicit = True, **kwargs)
    applyPositionUpdate(systemState, systemUpdate, semi_implicit_position_step(dt), **kwargs)
    applyQuantityUpdate(systemState, systemUpdate, explicit_step(dt), **kwargs)
    systemState.t = systemState.t + dt
    return systemState


def _normalize_blend(blend: Optional[StateBlend]) -> StateBlend:
    if blend is None:
        return StateBlend()
    return blend


def applyStateUpdate(systemState, systemUpdate, spec: ComponentUpdateSpec, **kwargs):
    if hasattr(systemState, 'apply_state_update'):
        return systemState.apply_state_update(systemUpdate, spec, **kwargs)

    blend = _normalize_blend(spec.blend)
    return systemState.integrate(
        systemUpdate,
        spec.derivative_dt,
        selfScale=blend.self_scale,
        referenceState=blend.reference_state,
        referenceWeight=blend.reference_weight,
        **kwargs,
    )


def applyPositionUpdate(systemState, systemUpdate, spec: PositionUpdateSpec, **kwargs):
    if hasattr(systemState, 'apply_position_update'):
        return systemState.apply_position_update(systemUpdate, spec, **kwargs)

    blend = _normalize_blend(spec.blend)
    legacy_kwargs = {}
    if spec.current_velocity_dt is not None:
        legacy_kwargs['semiImplicitScale'] = spec.current_velocity_dt
    if spec.update_velocity_dt is not None:
        legacy_kwargs['verletScale'] = spec.update_velocity_dt

    return systemState.integratePosition(
        systemUpdate,
        spec.derivative_dt,
        selfScale=blend.self_scale,
        referenceState=blend.reference_state,
        referenceWeight=blend.reference_weight,
        **legacy_kwargs,
        **kwargs,
    )


def applyVelocityUpdate(systemState, systemUpdate, spec: ComponentUpdateSpec, **kwargs):
    if hasattr(systemState, 'apply_velocity_update'):
        return systemState.apply_velocity_update(systemUpdate, spec, **kwargs)

    blend = _normalize_blend(spec.blend)
    return systemState.integrateVelocity(
        systemUpdate,
        spec.derivative_dt,
        selfScale=blend.self_scale,
        referenceState=blend.reference_state,
        referenceWeight=blend.reference_weight,
        **kwargs,
    )


def applyQuantityUpdate(systemState, systemUpdate, spec: ComponentUpdateSpec, **kwargs):
    if hasattr(systemState, 'apply_quantity_update'):
        return systemState.apply_quantity_update(systemUpdate, spec, **kwargs)

    blend = _normalize_blend(spec.blend)
    return systemState.integrateQuantities(
        systemUpdate,
        spec.derivative_dt,
        selfScale=blend.self_scale,
        referenceState=blend.reference_state,
        referenceWeight=blend.reference_weight,
        **kwargs,
    )


from enum import Enum
import torch

from .enums import IntegrationSchemeType
from typing import Callable
class IntegrationScheme(NamedTuple):
    function: Callable
    name: str    
    identifier: IntegrationSchemeType
    order: int
    
    dissipation: bool = False
    nonLagrangian: bool = False
    
    def __call__(self, state, dt, f, *args, **kwargs):
        return self.function(state, dt, f, *args, **kwargs)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    
IntegrationSchemes = []

def integrateQ(q : torch.Tensor, dqdt : Union[torch.Tensor, List[torch.Tensor]], dt : Union[float, List[float]],
                selfScale: Optional[float] = None,
                verletScale: Optional[float] = None,
                verletValue: Optional[torch.Tensor] = None,
                referenceValue: Optional[torch.Tensor] = None,
                referenceWeight: Optional[float] = None,
                integrateSpecies: Optional[List[int]] = None,
                species: Optional[torch.Tensor] = None):
    if integrateSpecies is not None:
        mask = torch.zeros(q.shape[0], device=q.device, dtype=torch.bool)
        for i in integrateSpecies:
            mask = torch.logical_or(mask, species == i)
        newValue = q.clone()
        if selfScale is not None:
            newValue[mask] = newValue[mask] * selfScale
        if referenceValue is not None:
            newValue[mask] = newValue[mask] + referenceValue[mask] * referenceWeight
        if verletScale is not None:
            newValue[mask] = newValue[mask] + verletScale * verletValue[mask]
        if isinstance(dt, list):
            for i in range(len(dt)):
                newValue[mask] = newValue[mask] + dt[i] * dqdt[mask]
        else:
            newValue[mask] = newValue[mask] + dt * dqdt[mask]
        return newValue
    else:        
        newValue = q.clone()
        if selfScale is not None:
            newValue = newValue * selfScale
        if referenceValue is not None:
            newValue = newValue + referenceValue * referenceWeight
        if verletScale is not None:
            newValue = newValue + verletScale * verletValue
            
        if isinstance(dt, list):
            for i in range(len(dt)):
                newValue = newValue + dt[i] * dqdt[i]
        else:
            newValue = newValue + dt * dqdt
        return newValue

def is_multiple_return_values(func, *args, **kwargs):
    result = func(*args, **kwargs)
    return isinstance(result, tuple) and len(result) > 1
def split_return(rv):
    if isinstance(rv, tuple):
        return rv[0], rv[1:]
    return rv, None

from torch.profiler import record_function

def initializeSystem(currentSystem, dt, *args, **kwargs):
    # print('Arguments: ', args)
    # print('Kwargs: ', kwargs)
    with record_function("[Integration] Preprocess"):
        return currentSystem.initialize(dt, *args, **kwargs)
    return currentSystem

def preprocessSystem(currentSystem , initialSystem , dt, *args, **kwargs):
    with record_function("[Integration] Preprocess"):
        return currentSystem.preprocess(initialSystem, dt, *args, **kwargs)
    return currentSystem

def postprocessSystem(initialSystem , currentSystem , dt, r, *args, **kwargs):
    with record_function("[Integration] Postprocess"):
        return initialSystem.postprocess(currentSystem, dt, r, *args, **kwargs)
    return initialSystem

def finalizeSystem(currentSystem , initialSystem , dt, *args, **kwargs):
    with record_function("[Integration] Finalize"):
        return currentSystem.finalize(initialSystem, dt, *args, **kwargs)
    return currentSystem


def updateStep(initialState, currentState, dt, f, *args, **kwargs):
    with record_function("[Integration] Update Step"):
        preprocessSystem(currentState, initialState, dt, *args, **kwargs)
        k, r = split_return(f(currentState, dt, *args, **kwargs))
        postprocessSystem(initialState, currentState, dt, r, *args, **kwargs)
        return k, r
    

# from integrators.integration import *
