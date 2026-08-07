from .util import (
    applyPositionUpdate,
    applyQuantityUpdate,
    applyStateUpdate,
    applyVelocityUpdate,
    finalizeSystem,
    initializeSystem,
    unpack_prior_step,
    updateStep,
)
from .specs import explicit_step, semi_implicit_position_step, verlet_position_step, IntegrationResult, StageResult
from torch.profiler import record_function


# Also known as synchronized form
def leapFrog(state, dt, f, *args, **kwargs):
    verbose = True if 'verbose' in kwargs and kwargs['verbose'] else False
    priorStep = kwargs.pop('priorStep', None)
    initializeSystem(state, dt, *args, **kwargs)

    with record_function("[Integration] Leap Frog"):
        with record_function("[Integration] Leap Frog: k0"):
            currentState = state.initializeNewState(*args, **kwargs)
            currentState.t = float(state.t)
            if priorStep is None:
                if verbose:
                    print(f"[Integrator] No priorStep provided, computing k0 and r0 using updateStep.")
                k0, r0 = updateStep(state, currentState, dt/2, f, *args, **kwargs)
            else:
                k0, r0 = unpack_prior_step(priorStep, verbose)

            # x^{n+1} = x^n + dt v^n + (dt^2/2) a^n -- this is the *full step*
            # position, so the second evaluation sits at t^n + dt, not t^n + dt/2.
            driftedState = state.initializeNewState(*args, **kwargs)
            applyPositionUpdate(
                driftedState,
                k0,
                verlet_position_step(dt, update_velocity_dt=0.5 * dt**2),
            )
            driftedState.t = float(state.t + dt)

        with record_function("[Integration] Leap Frog: k1"):
            k1, r1 = updateStep(state, driftedState, dt / 2, f, *args, **kwargs)

        with record_function("[Integration] Leap Frog: Update"):
            finalState = state.initializeNewState(*args, **kwargs)
            applyPositionUpdate(
                finalState,
                k0,
                verlet_position_step(dt, update_velocity_dt=0.5 * dt**2),
            )
            applyVelocityUpdate(finalState, [k0, k1], explicit_step([0.5 * dt, 0.5 * dt]))
            applyQuantityUpdate(finalState, [k0, k1], explicit_step([0.5 * dt, 0.5 * dt]))
            finalState.t = float(state.t + dt)
            
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, state, dt, rs, ks, [0.5, 0.5],
                           *args, lastStageSystem=driftedState, **kwargs)
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
                    currentState.t = float(state.t)
                if verbose:
                    print(f"[Integrator] No priorStep provided, computing k0 and r0 using updateStep.")
                k0, r0 = updateStep(state, currentState, dt/2, f, *args, **kwargs)
            else:
                k0, r0 = unpack_prior_step(priorStep, verbose)
            with record_function("[Integration] Symplectic Euler: Half State Initialization"):
                halfState = state.initializeNewState(*args, **kwargs)
            applyStateUpdate(halfState, k0, explicit_step(dt / 2), **kwargs)
            halfState.t = float(state.t + dt / 2)

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



            # NOTE: this used to branch on `hasattr(finalState, 'integrateDensity')`
            # and, when present, update density twice -- once through that bespoke
            # method and once through the generic path with `densitySwitch=True`
            # injected into the user's apply_quantity_update signature. That was a
            # solver-specific hook leaked in from diffSPH (NOTES.md 2.14); a system
            # that needs density handled differently should express it through the
            # field-behavior metadata or its own apply_quantity_update.
            applyQuantityUpdate(finalState, k1, explicit_step(dt), **kwargs)
            finalState.t = float(state.t + dt)

            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, state, dt, rs, ks, [0, 1],
                           *args, lastStageSystem=halfState, **kwargs)
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
    verbose = True if 'verbose' in kwargs and kwargs['verbose'] else False
    priorStep = kwargs.pop('priorStep', None)
    initializeSystem(state, dt, *args, **kwargs)
    with record_function("[Integration] Velocity Verlet"):
        with record_function("[Integration] Velocity Verlet: k0"):
            if priorStep is None:
                currentState = state.initializeNewState(*args, **kwargs)
                currentState.t = float(state.t)
                k0, r0 = updateStep(state, currentState, dt/2, f, *args, **kwargs)
            else:
                k0, r0 = unpack_prior_step(priorStep, verbose)
            # v^{n+1/2} = v^n + (dt/2) a^n, then x^{n+1} = x^n + dt v^{n+1/2}.
            # The position drift is a full dt, so the evaluation point is t^n + dt.
            driftedState = state.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(driftedState, k0, explicit_step(dt / 2))
            applyPositionUpdate(driftedState, k0, semi_implicit_position_step(dt))
            driftedState.t = float(state.t + dt)

        with record_function("[Integration] Velocity Verlet: k1"):
            k1, r1 = updateStep(state, driftedState, dt / 2, f, *args, **kwargs)

        with record_function("[Integration] Velocity Verlet: Update"):
            finalState = driftedState.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(finalState, k1, explicit_step(dt / 2))
            applyQuantityUpdate(finalState, k1, explicit_step(dt))

            finalState.t = float(state.t + dt)
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, state, dt, rs, ks, [1/2, 1/2],
                           *args, lastStageSystem=driftedState, **kwargs)
            return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])

