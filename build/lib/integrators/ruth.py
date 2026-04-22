from .util import (
    applyPositionUpdate,
    applyVelocityUpdate,
    applyQuantityUpdate,
    split_return,
    preprocessSystem,
    postprocessSystem,
    finalizeSystem,
)
from .specs import semi_implicit_position_step, explicit_step
import torch
import copy
from torch.profiler import record_function

    
# Based on PySPH based on
# [Omeylan2002] I.M. Omelyan, I.M. Mryglod and R. Folk, "Optimized
# Forest-Ruth- and Suzuki-like algorithms for integration of motion
# in many-body systems", Computer Physics Communications 146, 188 (2002)
# http://arxiv.org/abs/cond-mat/0110585
# Probably won't work well with energy or other integration terms due to relying on an initial drift step
def PEFRL(state, dt, f, *args, **kwargs):
    with record_function("[Integration] PEFRL"):
        lamda = -0.2123418310626054
        xi = +0.1786178958448091
        chi = -0.06626458266981849
        with record_function("[Integration] PEFRL: Step 1"):
            state1 = copy.deepcopy(state)
            applyPositionUpdate(state1, [], semi_implicit_position_step(xi * dt))
            state1.t = state.t + xi * dt
            preprocessSystem(state1, state, xi * dt, *args, **kwargs)
            k0, r0 = split_return(f(state1, xi * dt, *args, **kwargs))
            postprocessSystem(state, state1, xi * dt, r0, *args, **kwargs)
            applyVelocityUpdate(state1, k0, explicit_step((1 - 2 * lamda) * dt / 2))
            applyQuantityUpdate(state1, k0, explicit_step((1 - 2 * lamda) * dt / 2))

        with record_function("[Integration] PEFRL: Step 2"):
            state2 = copy.deepcopy(state1)
            applyPositionUpdate(state2, [], semi_implicit_position_step(chi * dt))
            state2.t = state1.t + chi * dt
            preprocessSystem(state2, state, chi * dt, *args, **kwargs)
            k1, r1 = split_return(f(state2, chi * dt, *args, **kwargs))
            postprocessSystem(state, state2, chi * dt, r1, *args, **kwargs)
            applyVelocityUpdate(state2, k1, explicit_step(lamda * dt))
            applyQuantityUpdate(state2, k1, explicit_step(lamda * dt))

        with record_function("[Integration] PEFRL: Step 3"):
            state3 = copy.deepcopy(state2)
            applyPositionUpdate(state3, [], semi_implicit_position_step((1 - 2 * (chi + xi)) * dt))
            state3.t = state2.t + (1 - 2 * (chi + xi)) * dt
            preprocessSystem(state3, state, (1 - 2 * (chi + xi)) * dt, *args, **kwargs)
            k2, r2 = split_return(f(state3, (1 - 2 * (chi + xi)) * dt, *args, **kwargs))
            postprocessSystem(state, state3, (1 - 2 * (chi + xi)) * dt, r2, *args, **kwargs)
            applyVelocityUpdate(state3, k2, explicit_step(lamda * dt))
            applyQuantityUpdate(state3, k2, explicit_step(lamda * dt))

        with record_function("[Integration] PEFRL: Step 4"):
            finalState = copy.deepcopy(state3)
            applyPositionUpdate(finalState, [], semi_implicit_position_step(chi * dt))
            finalState.t = state3.t + chi * dt
            preprocessSystem(finalState, state, chi * dt, *args, **kwargs)
            k3, r3 = split_return(f(finalState, chi * dt, *args, **kwargs))
            postprocessSystem(state, finalState, chi * dt, r3, *args, **kwargs)

        with record_function("[Integration] PEFRL: Update"):
            applyVelocityUpdate(finalState, k3, explicit_step((1 - 2 * lamda) * dt / 2))
            applyQuantityUpdate(finalState, k3, explicit_step((1 - 2 * lamda) * dt / 2))
            applyPositionUpdate(finalState, [], semi_implicit_position_step(xi * dt))
            finalState.t = state.t + dt
            rs = [r0, r1, r2, r3]
            ks = [k0, k1, k2, k3]
            finalizeSystem(finalState, state, dt, rs, ks, [], *args, **kwargs)
    if any([t is not None for t in rs]):
        return finalState, rs, ks
    return finalState, ks
    
    # r4 = r3 + chi * dt * v3
    # k3 = f(state3._replace(position = r4, t = state3.t + chi * dt))
    
    # finalVelocity = v3 + (1 - 2 * lamda) * dt / 2 * k3.velocity
    # finalPosition = r4 + xi * dt * finalVelocity
    # finalEnergy = e3 + (1 - 2 * lamda) * dt / 2 * k3.energy if hasattr(k3, 'energy') else e3
    
    # return state._replace(
    #     position = finalPosition,
    #     velocity = finalVelocity,
    #     energy = finalEnergy,
    #     t = state.t + dt
    # )
def VEFRL(state, dt, f, *args, **kwargs):
    with record_function("[Integration] VEFRL"):
        lamda = -0.2123418310626054
        xi = +0.1786178958448091
        chi = -0.06626458266981849
        with record_function("[Integration] VEFRL: Step 1"):
            preprocessSystem(state, state, xi * dt, *args, **kwargs)
            k0, r0 = split_return(f(state, xi * dt, *args, **kwargs))
            postprocessSystem(state, state, xi * dt, r0, *args, **kwargs)
            state1 = copy.deepcopy(state)
            applyVelocityUpdate(state1, k0, explicit_step(xi * dt))
            applyQuantityUpdate(state1, k0, explicit_step(xi * dt))
            applyPositionUpdate(state1, k0, semi_implicit_position_step((1 - 2 * lamda) * dt / 2))

        with record_function("[Integration] VEFRL: Step 2"):
            preprocessSystem(state1, state, chi * dt, *args, **kwargs)
            k1, r1 = split_return(f(state1, chi * dt, *args, **kwargs))
            postprocessSystem(state, state1, chi * dt, r1, *args, **kwargs)
            state2 = copy.deepcopy(state1)
            applyVelocityUpdate(state2, k1, explicit_step(chi * dt))
            applyQuantityUpdate(state2, k1, explicit_step(chi * dt))
            applyPositionUpdate(state2, k1, semi_implicit_position_step(lamda * dt))

        with record_function("[Integration] VEFRL: Step 3"):
            preprocessSystem(state2, state, (1 - 2 * (chi + xi)) * dt, *args, **kwargs)
            k2, r2 = split_return(f(state2, (1 - 2 * (chi + xi)) * dt, *args, **kwargs))
            postprocessSystem(state, state2, (1 - 2 * (chi + xi)) * dt, r2, *args, **kwargs)
            state3 = copy.deepcopy(state2)
            applyVelocityUpdate(state3, k2, explicit_step((1 - 2 * (chi + xi)) * dt))
            applyQuantityUpdate(state3, k2, explicit_step((1 - 2 * (chi + xi)) * dt))
            applyPositionUpdate(state3, k2, semi_implicit_position_step(lamda * dt))

        with record_function("[Integration] VEFRL: Step 4"):
            preprocessSystem(state3, state, chi * dt, *args, **kwargs)
            k3, r3 = split_return(f(state3, chi * dt, *args, **kwargs))
            postprocessSystem(state, state3, chi * dt, r3, *args, **kwargs)
            state4 = copy.deepcopy(state3)
            applyVelocityUpdate(state4, k3, explicit_step(chi * dt))
            applyQuantityUpdate(state4, k3, explicit_step(chi * dt))
            applyPositionUpdate(state4, k3, semi_implicit_position_step((1 - 2 * lamda) * dt / 2))

        with record_function("[Integration] VEFRL: Step 5"):
            preprocessSystem(state4, state, xi * dt, *args, **kwargs)
            k4, r4 = split_return(f(state4, xi * dt, *args, **kwargs))
            postprocessSystem(state, state4, xi * dt, r4, *args, **kwargs)
        with record_function("[Integration] VEFRL: Update"):
            finalState = copy.deepcopy(state4)
            applyVelocityUpdate(finalState, k4, explicit_step(xi * dt))
            applyQuantityUpdate(finalState, k4, explicit_step(xi * dt))
            finalState.t = state.t + dt
            rs = [r0, r1, r2, r3, r4]
            ks = [k0, k1, k2, k3, k4]
            finalizeSystem(finalState, state, dt, rs, ks, [], *args, **kwargs)
    if any([t is not None for t in rs]):
        return finalState, rs, ks
    return finalState, ks
    
    
    # finalVelocity = v4 + xi * dt * k4.velocity
    # finalPosition = r4
    # finalEnergy = e4 + xi * dt * k4.energy if hasattr(k4, 'energy') else e4
    
    # return state._replace(
    #     position = finalPosition,
    #     velocity = finalVelocity,
    #     energy = finalEnergy,
    #     t = state.t + dt
    # )