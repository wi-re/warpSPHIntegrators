from typing import Union, Tuple, NamedTuple
from .util import (
    applyStateUpdate,
    updateStateEuler,
    split_return,
    preprocessSystem,
    postprocessSystem,
    finalizeSystem,
)
from .specs import blend_state, explicit_step, IntegrationResult, StageResult
import torch
import copy
import numpy as np
from torch.profiler import record_function


# TVD RK3 scheme, uses the following integration scheme:
# $y^{n + \frac{1}{3}} = y^n + \Delta t F( y^n )$
# $y^{n + \frac{2}{3}} = \frac{3}{4}y^n + \frac{1}{4}(y^{n + \frac{1}{3}} + \Delta t F(y^{n + \frac{1}{3}}))$
# $y^{n + 1} = \frac{1}{3}y^n + \frac{2}{3}(y^{n + \frac{2}{3}}+ \Delta t F(y^{n + \frac{2}{3}}))$        
def TVDRK3(state, dt, f, *args, **kwargs):
    with record_function("[Integration] TVD RK3"):
        with record_function("[Integration] TVD RK3: k0"):
            preprocessSystem(state, state, dt, *args, **kwargs)
            k0, r0 = split_return(f(state, dt, *args, **kwargs))
            postprocessSystem(state, state, dt, r0, *args, **kwargs)
            y_1_3 = updateStateEuler(state, k0, dt)
        with record_function("[Integration] TVD RK3: k1"):
            preprocessSystem(y_1_3, state, 1/3 * dt, *args, **kwargs)
            k_1_3, r_1_3 = split_return(f(y_1_3, 1/3 * dt, *args, **kwargs))
            postprocessSystem(state, y_1_3, 1/3 * dt, r_1_3, *args, **kwargs)
            y_2_3 = copy.deepcopy(state)
            applyStateUpdate(
                y_2_3, k_1_3,
                explicit_step(1/4 * dt, blend=blend_state(
                    self_scale=3/4,
                    reference_state=y_1_3,
                    reference_weight=1/4,
                )),
            )
            y_2_3.t = state.t + dt * 1/3
        with record_function("[Integration] TVD RK3: k2"):
            preprocessSystem(y_2_3, state, 1/3 * dt, *args, **kwargs)
            k_2_3, r_2_3 = split_return(f(y_2_3, 1/3 * dt, *args, **kwargs))
            postprocessSystem(state, y_2_3, 1/3 * dt, r_2_3, *args, **kwargs)
        with record_function("[Integration] TVD RK3: Update"):
            finalState = copy.deepcopy(state)
            applyStateUpdate(
                finalState, k_2_3,
                explicit_step(2/3 * dt, blend=blend_state(
                    self_scale=1/3,
                    reference_state=y_2_3,
                    reference_weight=2/3,
                )),
            )
            finalState.t = state.t + dt
            rs = [r0, r_1_3, r_2_3]
            ks = [k0, k_1_3, k_2_3]
            finalizeSystem(finalState, state, dt, rs, ks, *args, **kwargs)
    return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])
    
    # return state._replace(
    #     position = finalPosition,
    #     velocity = finalVelocity,
    #     energy = finalEnergy,
    #     t = state.t + dt
    # )
    
def TVDRK2(state, dt, f, *args, **kwargs):
    with record_function("[Integration] TVD RK2"):
        with record_function("[Integration] TVD RK2: k0"):
            preprocessSystem(state, state, dt, *args, **kwargs)
            k0, r0 = split_return(f(state, dt, *args, **kwargs))
            postprocessSystem(state, state, dt, r0, *args, **kwargs)
            state1 = updateStateEuler(state, k0, dt)
        with record_function("[Integration] TVD RK2: k1"):
            preprocessSystem(state1, state, 1/2 * dt, *args, **kwargs)
            k1, r1 = split_return(f(state1, 1/2 * dt, *args, **kwargs))
            postprocessSystem(state, state1, 1/2 * dt, r1, *args, **kwargs)
        with record_function("[Integration] TVD RK2: Update"):
            finalState = copy.deepcopy(state)
            applyStateUpdate(
                finalState, k1,
                explicit_step(1/2 * dt, blend=blend_state(
                    self_scale=1/2,
                    reference_state=state1,
                    reference_weight=1/2,
                )),
            )
            finalState.t = state.t + dt
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, state, dt, rs, ks, [], *args, **kwargs)
        return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])
    
    # finalPosition = 1/2 * state.position + 1/2 * (state1.position + dt * k1.position)
    # finalVelocity = 1/2 * state.velocity + 1/2 * (state1.velocity + dt * k1.velocity)
    # finalEnergy   = 1/2 * state.energy   + 1/2 * (state1.energy + dt * k1.energy) if hasattr(k1, 'energy') else state.energy
    
    # return state._replace(
    #     position = finalPosition,
    #     velocity = finalVelocity,
    #     energy = finalEnergy,
    #     t = state.t + dt
    # )
    