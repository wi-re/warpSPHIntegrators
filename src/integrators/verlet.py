from typing import Union, Tuple, NamedTuple
from .util import updateStateEuler, updateStateSemiImplicitEuler
import torch
import copy
import numpy as np
from .util import applyPositionUpdate, applyQuantityUpdate, applyStateUpdate, applyVelocityUpdate, split_return, preprocessSystem, postprocessSystem, finalizeSystem, updateStep, initializeSystem
from .specs import blend_state, explicit_step, semi_implicit_position_step, verlet_position_step, IntegrationResult, StageResult
from torch.profiler import record_function


# Also known as synchronized form
def leapFrog(state, dt, f, *args, **kwargs):
    verbose = True if 'verbose' in kwargs and kwargs['verbose'] else False
    priorStep = kwargs.pop('priorStep', None)
    initializeSystem(state, dt, *args, **kwargs)

    with record_function("[Integration] Leap Frog"):
        with record_function("[Integration] Leap Frog: k0"):
            currentState = state.initializeNewState(*args, **kwargs)
            if priorStep is None:
                if verbose:
                    print(f"[Integrator] No priorStep provided, computing k0 and r0 using updateStep.")
                k0, r0 = updateStep(state, currentState, dt/2, f, *args, **kwargs)
            else:
                if isinstance(priorStep, StageResult):
                    if verbose:
                        print(f"[Integrator] Extracting k0 and r0 from priorStep NamedTuple with fields: {priorStep._fields} and values: {priorStep}")
                    k0, r0 = priorStep.update, priorStep.aux
                elif isinstance(priorStep, Tuple):
                    if verbose:
                        print(f"[Integrator] Extracting k0 and r0 from priorStep tuple with values: {priorStep}")
                    k0, r0 = priorStep
                else:
                    raise ValueError(f"Invalid priorStep format: {priorStep}")
            # k0, r0 = updateStep(state, currentState, dt/2, f, *args, **kwargs)

            halfState = state.initializeNewState(*args, **kwargs)
            applyPositionUpdate(
                halfState,
                k0,
                verlet_position_step(dt, update_velocity_dt=0.5 * dt**2),
            )
            halfState.t = state.t + 0.5 * dt

        with record_function("[Integration] Leap Frog: k1"):
            k1, r1 = updateStep(state, halfState, dt / 2, f, *args, **kwargs)
    
        with record_function("[Integration] Leap Frog: Update"):
            finalState = state.initializeNewState(*args, **kwargs)
            applyPositionUpdate(
                finalState,
                k0,
                verlet_position_step(dt, update_velocity_dt=0.5 * dt**2),
            )
            applyVelocityUpdate(finalState, [k0, k1], explicit_step([0.5 * dt, 0.5 * dt]))
            applyQuantityUpdate(finalState, [k0, k1], explicit_step([0.5 * dt, 0.5 * dt]))
            finalState.t = state.t + dt
            
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, state, dt, rs, ks, [0.5, 0.5], *args, **kwargs)
            return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])

# Also known as kick-drift-kick form and position verlet
# see 'Improvements in SPH method by means of interparticle
# contact algorithm and analysis of perforation tests at moderate
# projectile velocities.'
def symplecticEuler(state, dt, f, *args, **kwargs):
    with record_function("[Integration] Symplectic Euler"):
        verbose = True if 'verbose' in kwargs and kwargs['verbose'] else False
        priorStep = kwargs.pop('priorStep', None)
        initializeSystem(state, dt, *args, **kwargs)

        with record_function("[Integration] Symplectic Euler: k0"):
            if priorStep is None:
                with record_function("[Integration] Symplectic Euler: Current State Initialization"):
                    currentState = state.initializeNewState(*args, **kwargs)
                if verbose:
                    print(f"[Integrator] No priorStep provided, computing k0 and r0 using updateStep.")
                k0, r0 = updateStep(state, currentState, dt/2, f, *args, **kwargs)
            else:
                if isinstance(priorStep, StageResult):
                    if verbose:
                        print(f"[Integrator] Extracting k0 and r0 from priorStep NamedTuple with fields: {priorStep._fields} and values: {priorStep}")
                    k0, r0 = priorStep.update, priorStep.aux
                elif isinstance(priorStep, Tuple):
                    if verbose:
                        print(f"[Integrator] Extracting k0 and r0 from priorStep tuple with values: {priorStep}")
                    k0, r0 = priorStep
                else:
                    raise ValueError(f"Invalid priorStep format: {priorStep}")
            # k0, r0 = updateStep(state, currentState, dt/2, f, *args, **kwargs) if priorStep is None else priorStep
            # k0, r0 = updateStep(state, currentState, dt/2, f, *args, **kwargs)
            with record_function("[Integration] Symplectic Euler: Half State Initialization"):
                halfState = state.initializeNewState(*args, **kwargs)
            applyStateUpdate(halfState, k0, explicit_step(dt / 2), **kwargs)

        with record_function("[Integration] Symplectic Euler: k1"):
            k1, r1 = updateStep(state, halfState, dt / 2, f, *args, **kwargs)

        with record_function("[Integration] Symplectic Euler: Update"):
            with record_function("[Integration] Symplectic Euler: Final State Initialization"):
                finalState = state.initializeNewState(*args, **kwargs)
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
            finalState.t = state.t + dt
            
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, state, dt, rs, ks, [0,1], *args, **kwargs)
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
def velocityVerlet(state, dt, f, *args, **kwargs):
    priorStep = kwargs.pop('priorStep', None)
    initializeSystem(state, dt, *args, **kwargs)
    with record_function("[Integration] Velocity Verlet"):
        with record_function("[Integration] Velocity Verlet: k0"):
            currentState = state.initializeNewState(*args, **kwargs)
            k0, r0 = updateStep(state, currentState, dt/2, f, *args, **kwargs) if priorStep is None else priorStep
            # k0, r0 = updateStep(state, currentState, dt/2, f, *args, **kwargs)
            halfState = state.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(halfState, k0, explicit_step(dt / 2))
            applyPositionUpdate(halfState, k0, semi_implicit_position_step(dt))
            halfState.t = state.t + dt / 2

        with record_function("[Integration] Velocity Verlet: k1"):
            k1, r1 = updateStep(state, halfState, dt / 2, f, *args, **kwargs)

        with record_function("[Integration] Velocity Verlet: Update"):
            finalState = halfState.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(finalState, k1, explicit_step(dt / 2))
            applyQuantityUpdate(finalState, k1, explicit_step(dt))
            
            finalState.t = state.t + dt
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, state, dt, rs, ks, [1/2, 1/2], *args, **kwargs)
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

