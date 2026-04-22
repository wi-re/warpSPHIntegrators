from typing import Union, Tuple, NamedTuple
from .util import updateStateEuler, updateStateSemiImplicitEuler
import torch
import copy
import numpy as np
from .util import applyPositionUpdate, applyQuantityUpdate, applyStateUpdate, applyVelocityUpdate, split_return, preprocessSystem, postprocessSystem, finalizeSystem, updateStep, initializeSystem
from .specs import blend_state, explicit_step, semi_implicit_position_step, verlet_position_step, IntegrationResult, StageResult
from torch.profiler import record_function


# Also known as synchronized form
def leapFrog(initialState, dt, f, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    initializeSystem(initialState, dt, *args, **kwargs)

    with record_function("[Integration] Leap Frog"):
        with record_function("[Integration] Leap Frog: k0"):
            currentState = initialState.initializeNewState(*args, **kwargs)
            k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs) if priorStep is None else priorStep
            # k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs)

            halfState = initialState.initializeNewState(*args, **kwargs)
            applyPositionUpdate(
                halfState,
                k0,
                verlet_position_step(dt, update_velocity_dt=0.5 * dt**2),
            )
            halfState.t = initialState.t + 0.5 * dt

        with record_function("[Integration] Leap Frog: k1"):
            k1, r1 = updateStep(initialState, halfState, dt / 2, f, *args, **kwargs)
    
        with record_function("[Integration] Leap Frog: Update"):
            finalState = initialState.initializeNewState(*args, **kwargs)
            applyPositionUpdate(
                finalState,
                k0,
                verlet_position_step(dt, update_velocity_dt=0.5 * dt**2),
            )
            applyVelocityUpdate(finalState, [k0, k1], explicit_step([0.5 * dt, 0.5 * dt]))
            applyQuantityUpdate(finalState, [k0, k1], explicit_step([0.5 * dt, 0.5 * dt]))
            finalState.t = initialState.t + dt
            
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, initialState, dt, rs, ks, [0.5, 0.5], *args, **kwargs)
            return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])

# Also known as kick-drift-kick form and position verlet
# see 'Improvements in SPH method by means of interparticle
# contact algorithm and analysis of perforation tests at moderate
# projectile velocities.'
def symplecticEuler(initialState, dt, f, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    initializeSystem(initialState, dt, *args, **kwargs)
    with record_function("[Integration] Symplectic Euler"):
        with record_function("[Integration] Symplectic Euler: k0"):
            currentState = initialState.initializeNewState(*args, **kwargs)
            k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs) if priorStep is None else priorStep
            # k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs)
            halfState = initialState.initializeNewState(*args, **kwargs)
            applyStateUpdate(halfState, k0, explicit_step(dt / 2), **kwargs)

        with record_function("[Integration] Symplectic Euler: k1"):
            k1, r1 = updateStep(initialState, halfState, dt / 2, f, *args, **kwargs)

        with record_function("[Integration] Symplectic Euler: Update"):
            finalState = initialState.initializeNewState(*args, **kwargs)
            # Based on the DualSPHysics wiki:
            # r^n+1/2 = r^n + 0.5 * dt * v^n
            # v^n+1/2 = v^n + dt * a^n
            # v^n+1 = v^n + dt * a^n+1/2
            # r^n+1 = r^n + dt/2 * (v^n + v^n+1)
            # at this point we have k0 = a^n, k1 = a^n+1/2, and we want to compute the final state at n+1

            applyVelocityUpdate(finalState, k1, explicit_step(dt), **kwargs) # update velocity first using the half-step acceleration

            # We can apply the position update in two stages, however, the seconmd stage does not use the returned update value, just the result of applying the velocity update in the previous op
            applyPositionUpdate(finalState, k0, explicit_step(dt / 2), **kwargs) # update position using the initial velocity
            applyPositionUpdate(finalState, [], semi_implicit_position_step(dt / 2), **kwargs) # update position using the initial velocity



            if hasattr(finalState, 'integrateDensity'):
                finalState.integrateDensity(k1, dt, **kwargs)
                applyQuantityUpdate(finalState, k1, explicit_step(dt), densitySwitch = True, **kwargs)
            else:
                applyQuantityUpdate(finalState, k1, explicit_step(dt), **kwargs)
            finalState.t = initialState.t + dt
            
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, initialState, dt, rs, ks, [0,1], *args, **kwargs)
            return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])

    # finalVelocity = state.velocity + dt * k1.velocity
    # finalPosition = state.position + dt * (k0.position + finalVelocity) / 2
    # finalEnergy   = state.energy + dt * k1.energy if hasattr(k1, 'energy') else state.energy

    # return state._replace(
    #     position = finalPosition,
    #     velocity = finalVelocity,
    #     energy = finalEnergy,
    #     t = state.t + dt
    # )

# velocity verlet, doesn't work well for systems with energy, also known as drift-kick-drift
def velocityVerlet(initialState, dt, f, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    initializeSystem(initialState, dt, *args, **kwargs)
    with record_function("[Integration] Velocity Verlet"):
        with record_function("[Integration] Velocity Verlet: k0"):
            currentState = initialState.initializeNewState(*args, **kwargs)
            k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs) if priorStep is None else priorStep
            # k0, r0 = updateStep(initialState, currentState, dt/2, f, *args, **kwargs)
            halfState = initialState.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(halfState, k0, explicit_step(dt / 2))
            applyPositionUpdate(halfState, k0, semi_implicit_position_step(dt))
            halfState.t = initialState.t + dt / 2

        with record_function("[Integration] Velocity Verlet: k1"):
            k1, r1 = updateStep(initialState, halfState, dt / 2, f, *args, **kwargs)

        with record_function("[Integration] Velocity Verlet: Update"):
            finalState = halfState.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(finalState, k1, explicit_step(dt / 2))
            applyQuantityUpdate(finalState, k1, explicit_step(dt))
            
            finalState.t = initialState.t + dt
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, initialState, dt, rs, ks, [1/2, 1/2], *args, **kwargs)
            return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])
    
    k0 = f(state)
    halfStateVelocity = state.velocity + k0.velocity * dt / 2
    finalPosition = state.position + halfStateVelocity * dt
    halfState = state._replace(position = finalPosition, velocity = halfStateVelocity)

    k1 = f(halfState)
    finalVelocity = halfStateVelocity + k1.velocity * dt / 2
    finalEnergy   = state.energy + dt * k0.energy if hasattr(k0, 'energy') else state.energy

    return state._replace(
        position = finalPosition,
        velocity = finalVelocity,
        energy = finalEnergy,
        t = state.t + dt
    )

