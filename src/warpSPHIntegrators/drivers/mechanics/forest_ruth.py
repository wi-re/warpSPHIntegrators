"""Numerical stepping for forest_ruth; configuration and coefficients are supplied separately."""
from warpSPHIntegrators.state.lifecycle import applyPositionUpdate, applyVelocityUpdate, applyQuantityUpdate, finalizeSystem, initializeSystem, reject_prior_step, updateStep
from warpSPHIntegrators.state.fields import clear_ephemeral_fields
from warpSPHIntegrators.state.specs import semi_implicit_position_step, explicit_step, IntegrationResult, StageResult
from warpSPHIntegrators.state.drift import add_to_drift_fields, drift_fields_active, drift_rates, kinematic_part, velocity_of, without_kinematic
from torch.profiler import record_function


def _kick(state, k, drift):
    """The stage update a kick applies to the quantities: without the drift
    fields' velocity-linear part when those are moved by the drifts (drift.py)."""
    return without_kinematic(state, k) if drift else k


def _driftStart(target, geometry, aux, tau, **kwargs):
    """Predictor for the drift fields over a drift of length `tau` with `target`'s
    current velocity, from the configuration of `geometry` (the start of the drift).
    Returns the start rates for `_driftEnd`'s trapezoid."""
    kin = drift_rates(geometry, velocity_of(target), aux, **kwargs)
    add_to_drift_fields(target, (tau, kin))
    return kin


def _driftEnd(target, tau, kinStart, kinEnd):
    """Complete the trapezoid once the end-of-drift rate is known."""
    add_to_drift_fields(target, (tau / 2, kinEnd), (-tau / 2, kinStart))


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
        # Drift fields (drift.py): a trapezoid per drift. Each stage runs right
        # after a drift with the drift's velocity, so its velocity-linear rate is
        # the end point for free; the start point is one evaluation per drift,
        # plus one for the end of the closing drift (6 per step).
        drift = drift_fields_active(state, **kwargs)
        with record_function("[Integration] PEFRL: Step 1"):
            state1 = state.initializeNewState(*args, **kwargs)
            if drift:
                kin = _driftStart(state1, state, None, xi * dt, **kwargs)
            applyPositionUpdate(state1, [], semi_implicit_position_step(xi * dt))
            state1.t = float(state.t + xi * dt)
            k0, r0 = updateStep(state, state1, xi * dt, f, *args, **kwargs)
            if drift:
                _driftEnd(state1, xi * dt, kin, kinematic_part(state, k0))
            applyVelocityUpdate(state1, k0, explicit_step((1 - 2 * lamda) * dt / 2))
            applyQuantityUpdate(state1, _kick(state, k0, drift), explicit_step((1 - 2 * lamda) * dt / 2))

        with record_function("[Integration] PEFRL: Step 2"):
            state2 = state1.initializeNewState(*args, **kwargs)
            if drift:
                kin = _driftStart(state2, state1, r0, chi * dt, **kwargs)
            applyPositionUpdate(state2, [], semi_implicit_position_step(chi * dt))
            state2.t = float(state1.t + chi * dt)
            k1, r1 = updateStep(state, state2, chi * dt, f, *args, **kwargs)
            if drift:
                _driftEnd(state2, chi * dt, kin, kinematic_part(state, k1))
            applyVelocityUpdate(state2, k1, explicit_step(lamda * dt))
            applyQuantityUpdate(state2, _kick(state, k1, drift), explicit_step(lamda * dt))

        with record_function("[Integration] PEFRL: Step 3"):
            state3 = state2.initializeNewState(*args, **kwargs)
            if drift:
                kin = _driftStart(state3, state2, r1, (1 - 2 * (chi + xi)) * dt, **kwargs)
            applyPositionUpdate(state3, [], semi_implicit_position_step((1 - 2 * (chi + xi)) * dt))
            state3.t = float(state2.t + (1 - 2 * (chi + xi)) * dt)
            k2, r2 = updateStep(state, state3, (1 - 2 * (chi + xi)) * dt, f, *args, **kwargs)
            if drift:
                _driftEnd(state3, (1 - 2 * (chi + xi)) * dt, kin, kinematic_part(state, k2))
            applyVelocityUpdate(state3, k2, explicit_step(lamda * dt))
            applyQuantityUpdate(state3, _kick(state, k2, drift), explicit_step(lamda * dt))

        with record_function("[Integration] PEFRL: Step 4"):
            finalState = state3.initializeNewState(*args, **kwargs)
            if drift:
                kin = _driftStart(finalState, state3, r2, chi * dt, **kwargs)
            applyPositionUpdate(finalState, [], semi_implicit_position_step(chi * dt))
            finalState.t = float(state3.t + chi * dt)
            k3, r3 = updateStep(state, finalState, chi * dt, f, *args, **kwargs)
            if drift:
                _driftEnd(finalState, chi * dt, kin, kinematic_part(state, k3))

        with record_function("[Integration] PEFRL: Update"):
            applyVelocityUpdate(finalState, k3, explicit_step((1 - 2 * lamda) * dt / 2))
            applyQuantityUpdate(finalState, _kick(state, k3, drift), explicit_step((1 - 2 * lamda) * dt / 2))
            if drift:
                kin = _driftStart(finalState, finalState, r3, xi * dt, **kwargs)
            applyPositionUpdate(finalState, [], semi_implicit_position_step(xi * dt))
            if drift:
                _driftEnd(finalState, xi * dt, kin,
                          drift_rates(finalState, velocity_of(finalState), r3, **kwargs))
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
        # Drift fields (drift.py): a trapezoid per drift; every drift is followed
        # by a stage at its own velocity (end point free), the start point is one
        # evaluation per drift (4 per step).
        drift = drift_fields_active(state, **kwargs)
        with record_function("[Integration] VEFRL: Step 1"):
            state0 = state.initializeNewState(*args, **kwargs)
            state0.t = float(state.t)
            k0, r0 = updateStep(state, state0, xi * dt, f, *args, **kwargs)
            state1 = state.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(state1, k0, explicit_step(xi * dt))
            applyQuantityUpdate(state1, _kick(state, k0, drift), explicit_step(xi * dt))
            if drift:
                kin = _driftStart(state1, state0, r0, half_drift, **kwargs)
            applyPositionUpdate(state1, [], semi_implicit_position_step(half_drift))
            state1.t = float(state.t + half_drift)

        with record_function("[Integration] VEFRL: Step 2"):
            k1, r1 = updateStep(state, state1, chi * dt, f, *args, **kwargs)
            if drift:
                _driftEnd(state1, half_drift, kin, kinematic_part(state, k1))
            state2 = state1.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(state2, k1, explicit_step(chi * dt))
            applyQuantityUpdate(state2, _kick(state, k1, drift), explicit_step(chi * dt))
            if drift:
                kin = _driftStart(state2, state1, r1, lamda * dt, **kwargs)
            applyPositionUpdate(state2, [], semi_implicit_position_step(lamda * dt))
            state2.t = float(state1.t + lamda * dt)

        with record_function("[Integration] VEFRL: Step 3"):
            k2, r2 = updateStep(state, state2, (1 - 2 * (chi + xi)) * dt, f, *args, **kwargs)
            if drift:
                _driftEnd(state2, lamda * dt, kin, kinematic_part(state, k2))
            state3 = state2.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(state3, k2, explicit_step((1 - 2 * (chi + xi)) * dt))
            applyQuantityUpdate(state3, _kick(state, k2, drift), explicit_step((1 - 2 * (chi + xi)) * dt))
            if drift:
                kin = _driftStart(state3, state2, r2, lamda * dt, **kwargs)
            applyPositionUpdate(state3, [], semi_implicit_position_step(lamda * dt))
            state3.t = float(state2.t + lamda * dt)

        with record_function("[Integration] VEFRL: Step 4"):
            k3, r3 = updateStep(state, state3, chi * dt, f, *args, **kwargs)
            if drift:
                _driftEnd(state3, lamda * dt, kin, kinematic_part(state, k3))
            state4 = state3.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(state4, k3, explicit_step(chi * dt))
            applyQuantityUpdate(state4, _kick(state, k3, drift), explicit_step(chi * dt))
            if drift:
                kin = _driftStart(state4, state3, r3, half_drift, **kwargs)
            applyPositionUpdate(state4, [], semi_implicit_position_step(half_drift))
            state4.t = float(state3.t + half_drift)

        with record_function("[Integration] VEFRL: Step 5"):
            k4, r4 = updateStep(state, state4, xi * dt, f, *args, **kwargs)
            if drift:
                _driftEnd(state4, half_drift, kin, kinematic_part(state, k4))
        with record_function("[Integration] VEFRL: Update"):
            finalState = state4.initializeNewState(*args, **kwargs)
            applyVelocityUpdate(finalState, k4, explicit_step(xi * dt))
            applyQuantityUpdate(finalState, _kick(state, k4, drift), explicit_step(xi * dt))
            finalState.t = float(state.t + dt)
            rs = [r0, r1, r2, r3, r4]
            ks = [k0, k1, k2, k3, k4]
            finalizeSystem(finalState, state, dt, rs, ks, [],
                           *args, lastStageSystem=state4, **kwargs)
    return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])
