from .util import (
    applyStateUpdate,
    finalizeSystem,
    initializeSystem,
    reject_prior_step,
    updateStateEuler,
    updateStep,
)
from .specs import blend_state, explicit_step, IntegrationResult, StageResult
from torch.profiler import record_function


# TVD RK3 (Shu-Osher form of SSP-RK3):
# $y^{(1)} = y^n + \Delta t F(t^n, y^n)$
# $y^{(2)} = \frac{3}{4}y^n + \frac{1}{4}\left(y^{(1)} + \Delta t F(t^n + \Delta t, y^{(1)})\right)$
# $y^{n+1} = \frac{1}{3}y^n + \frac{2}{3}\left(y^{(2)} + \Delta t F(t^n + \tfrac{1}{2}\Delta t, y^{(2)})\right)$
#
# Note the stage times: y^{(1)} is a *full* Euler step, so it sits at t^n + dt, and
# y^{(2)} is the second-order combination, which sits at t^n + dt/2. Those are the
# nodes c = (0, 1, 1/2) of the equivalent Butcher tableau.
def TVDRK3(state, dt, f, *args, **kwargs):
    with record_function("[Integration] TVD RK3"):
        reject_prior_step('TVD RK3', kwargs.pop('priorStep', None))
        initializeSystem(state, dt, *args, **kwargs)

        with record_function("[Integration] TVD RK3: k0"):
            # Evaluate on a stage buffer, never on the caller's state: preprocess
            # typically writes neighbour lists and scratch into whatever it is given.
            y_0 = state.initializeNewState(*args, **kwargs)
            y_0.t = float(state.t)
            k0, r0 = updateStep(state, y_0, dt, f, *args, **kwargs)
            y_1 = updateStateEuler(state, k0, dt, **kwargs)
            y_1.t = float(state.t + dt)

        with record_function("[Integration] TVD RK3: k1"):
            k_1, r_1 = updateStep(state, y_1, 1/3 * dt, f, *args, **kwargs)
            y_2 = state.initializeNewState(*args, **kwargs)
            applyStateUpdate(
                y_2, k_1,
                explicit_step(1/4 * dt, blend=blend_state(
                    self_scale=3/4,
                    reference_state=y_1,
                    reference_weight=1/4,
                )),
                **kwargs,
            )
            y_2.t = float(state.t + dt / 2)

        with record_function("[Integration] TVD RK3: k2"):
            k_2, r_2 = updateStep(state, y_2, 1/3 * dt, f, *args, **kwargs)

        with record_function("[Integration] TVD RK3: Update"):
            finalState = state.initializeNewState(*args, **kwargs)
            applyStateUpdate(
                finalState, k_2,
                explicit_step(2/3 * dt, blend=blend_state(
                    self_scale=1/3,
                    reference_state=y_2,
                    reference_weight=2/3,
                )),
                **kwargs,
            )
            finalState.t = float(state.t + dt)
            rs = [r0, r_1, r_2]
            ks = [k0, k_1, k_2]
            # Equivalent Butcher weights for the Shu-Osher form above.
            finalizeSystem(finalState, state, dt, rs, ks, [1/6, 1/6, 2/3],
                           *args, lastStageSystem=y_2, **kwargs)
    return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])


# TVD RK2 (Shu-Osher form of Heun's method):
# $y^{(1)} = y^n + \Delta t F(t^n, y^n)$
# $y^{n+1} = \frac{1}{2}y^n + \frac{1}{2}\left(y^{(1)} + \Delta t F(t^n + \Delta t, y^{(1)})\right)$
def TVDRK2(state, dt, f, *args, **kwargs):
    with record_function("[Integration] TVD RK2"):
        reject_prior_step('TVD RK2', kwargs.pop('priorStep', None))
        initializeSystem(state, dt, *args, **kwargs)

        with record_function("[Integration] TVD RK2: k0"):
            y_0 = state.initializeNewState(*args, **kwargs)
            y_0.t = float(state.t)
            k0, r0 = updateStep(state, y_0, dt, f, *args, **kwargs)
            state1 = updateStateEuler(state, k0, dt, **kwargs)
            state1.t = float(state.t + dt)

        with record_function("[Integration] TVD RK2: k1"):
            k1, r1 = updateStep(state, state1, 1/2 * dt, f, *args, **kwargs)

        with record_function("[Integration] TVD RK2: Update"):
            finalState = state.initializeNewState(*args, **kwargs)
            applyStateUpdate(
                finalState, k1,
                explicit_step(1/2 * dt, blend=blend_state(
                    self_scale=1/2,
                    reference_state=state1,
                    reference_weight=1/2,
                )),
                **kwargs,
            )
            finalState.t = float(state.t + dt)
            rs = [r0, r1]
            ks = [k0, k1]
            finalizeSystem(finalState, state, dt, rs, ks, [1/2, 1/2],
                           *args, lastStageSystem=state1, **kwargs)
        return IntegrationResult(state=finalState, stages=[StageResult(aux=r, update=k) for r, k in zip(rs, ks)])
