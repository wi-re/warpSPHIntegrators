from .util import (
    applyPositionUpdate,
    applyVelocityUpdate,
    applyQuantityUpdate,
    finalizeSystem,
    initializeSystem,
    reject_prior_step,
    updateStep,
)
from .fields import clear_ephemeral_fields
from .specs import semi_implicit_position_step, explicit_step, IntegrationResult, StageResult
from torch.profiler import record_function


# Based on PySPH based on
# [Omeylan2002] I.M. Omelyan, I.M. Mryglod and R. Folk, "Optimized
# Forest-Ruth- and Suzuki-like algorithms for integration of motion
# in many-body systems", Computer Physics Communications 146, 188 (2002)
# http://arxiv.org/abs/cond-mat/0110585
# Probably won't work well with energy or other integration terms due to relying on an initial drift step
#
# Time is treated as an extra coordinate that advances with the drift (position)
# substeps, which is what staggers the force evaluations across the step. The drift
# coefficients xi, chi, 1-2(chi+xi), chi, xi sum to 1.
def PEFRL(state, dt, f, *args, **kwargs):
    with record_function("[Integration] PEFRL"):
        reject_prior_step('PEFRL', kwargs.pop('priorStep', None))
        initializeSystem(state, dt, *args, **kwargs)

        lamda = -0.2123418310626054
        xi = +0.1786178958448091
        chi = -0.06626458266981849
        with record_function("[Integration] PEFRL: Step 1"):
            state1 = state.initializeNewState(*args, **kwargs)
            applyPositionUpdate(state1, [], semi_implicit_position_step(xi * dt))
            state1.t = float(state.t + xi * dt)
            k0, r0 = updateStep(state, state1, xi * dt, f, *args, **kwargs)
            applyVelocityUpdate(state1, k0, explicit_step((1 - 2 * lamda) * dt / 2))
            applyQuantityUpdate(state1, k0, explicit_step((1 - 2 * lamda) * dt / 2))

        with record_function("[Integration] PEFRL: Step 2"):
            state2 = state1.initializeNewState(*args, **kwargs)
            applyPositionUpdate(state2, [], semi_implicit_position_step(chi * dt))
            state2.t = float(state1.t + chi * dt)
            k1, r1 = updateStep(state, state2, chi * dt, f, *args, **kwargs)
            applyVelocityUpdate(state2, k1, explicit_step(lamda * dt))
            applyQuantityUpdate(state2, k1, explicit_step(lamda * dt))

        with record_function("[Integration] PEFRL: Step 3"):
            state3 = state2.initializeNewState(*args, **kwargs)
            applyPositionUpdate(state3, [], semi_implicit_position_step((1 - 2 * (chi + xi)) * dt))
            state3.t = float(state2.t + (1 - 2 * (chi + xi)) * dt)
            k2, r2 = updateStep(state, state3, (1 - 2 * (chi + xi)) * dt, f, *args, **kwargs)
            applyVelocityUpdate(state3, k2, explicit_step(lamda * dt))
            applyQuantityUpdate(state3, k2, explicit_step(lamda * dt))

        with record_function("[Integration] PEFRL: Step 4"):
            finalState = state3.initializeNewState(*args, **kwargs)
            applyPositionUpdate(finalState, [], semi_implicit_position_step(chi * dt))
            finalState.t = float(state3.t + chi * dt)
            k3, r3 = updateStep(state, finalState, chi * dt, f, *args, **kwargs)

        with record_function("[Integration] PEFRL: Update"):
            applyVelocityUpdate(finalState, k3, explicit_step((1 - 2 * lamda) * dt / 2))
            applyQuantityUpdate(finalState, k3, explicit_step((1 - 2 * lamda) * dt / 2))
            applyPositionUpdate(finalState, [], semi_implicit_position_step(xi * dt))
            finalState.t = float(state.t + dt)
            rs = [r0, r1, r2, r3]
            ks = [k0, k1, k2, k3]
            # No lastStageSystem: PEFRL's final state *is* the buffer the last
            # evaluation ran on, so `copied` fields are already in place. For the same
            # reason its stage-local scratch is still attached and has to be dropped.
            clear_ephemeral_fields(finalState)
            finalizeSystem(finalState, state, dt, rs, ks, [], *args, **kwargs)
    return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])


# Velocity-extended FRL: the mirror image of PEFRL, starting with a kick rather than
# a drift. The drift coefficients are (1-2*lamda)/2, lamda, lamda, (1-2*lamda)/2,
# which is what stages the force evaluations in time.
def VEFRL(state, dt, f, *args, **kwargs):
    with record_function("[Integration] VEFRL"):
        reject_prior_step('VEFRL', kwargs.pop('priorStep', None))
        initializeSystem(state, dt, *args, **kwargs)

        # lamda = -0.2094333910398989e-01
        # xi = +0.1644986515575760e+00
        # chi = +0.1235692651138917e+01

        xi = +0.1720865590295143e+00
        lamda = -0.9156203075515678e-01
        chi = -0.1616217622107222e+00
        half_drift = (1 - 2 * lamda) * dt / 2
        with record_function("[Integration] VEFRL: Step 1"):
            state0 = state.initializeNewState(*args, **kwargs)
            state0.t = float(state.t)
            k0, r0 = updateStep(state, state0, xi * dt, f, *args, **kwargs)
            state1 = state.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(state1, k0, explicit_step(xi * dt))
            applyQuantityUpdate(state1, k0, explicit_step(xi * dt))
            applyPositionUpdate(state1, [], semi_implicit_position_step(half_drift))
            state1.t = float(state.t + half_drift)

        with record_function("[Integration] VEFRL: Step 2"):
            k1, r1 = updateStep(state, state1, chi * dt, f, *args, **kwargs)
            state2 = state1.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(state2, k1, explicit_step(chi * dt))
            applyQuantityUpdate(state2, k1, explicit_step(chi * dt))
            applyPositionUpdate(state2, [], semi_implicit_position_step(lamda * dt))
            state2.t = float(state1.t + lamda * dt)

        with record_function("[Integration] VEFRL: Step 3"):
            k2, r2 = updateStep(state, state2, (1 - 2 * (chi + xi)) * dt, f, *args, **kwargs)
            state3 = state2.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(state3, k2, explicit_step((1 - 2 * (chi + xi)) * dt))
            applyQuantityUpdate(state3, k2, explicit_step((1 - 2 * (chi + xi)) * dt))
            applyPositionUpdate(state3, [], semi_implicit_position_step(lamda * dt))
            state3.t = float(state2.t + lamda * dt)

        with record_function("[Integration] VEFRL: Step 4"):
            k3, r3 = updateStep(state, state3, chi * dt, f, *args, **kwargs)
            state4 = state3.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(state4, k3, explicit_step(chi * dt))
            applyQuantityUpdate(state4, k3, explicit_step(chi * dt))
            applyPositionUpdate(state4, [], semi_implicit_position_step(half_drift))
            state4.t = float(state3.t + half_drift)

        with record_function("[Integration] VEFRL: Step 5"):
            k4, r4 = updateStep(state, state4, xi * dt, f, *args, **kwargs)
        with record_function("[Integration] VEFRL: Update"):
            finalState = state4.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(finalState, k4, explicit_step(xi * dt))
            applyQuantityUpdate(finalState, k4, explicit_step(xi * dt))
            finalState.t = float(state.t + dt)
            rs = [r0, r1, r2, r3, r4]
            ks = [k0, k1, k2, k3, k4]
            finalizeSystem(finalState, state, dt, rs, ks, [],
                           *args, lastStageSystem=state4, **kwargs)
    return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])
