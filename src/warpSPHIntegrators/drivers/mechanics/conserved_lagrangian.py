"""Time staging for conserved moving-quadrature systems, independent of spatial physics."""
from dataclasses import dataclass
from torch.profiler import record_function
from warpSPHIntegrators.state.fields import get_reference_state, get_tagged_attr, set_tagged_attr
from warpSPHIntegrators.state.lifecycle import initializeSystem, updateStep, finalizeSystem, hostTime, reject_prior_step
from warpSPHIntegrators.state.specs import IntegrationResult, StageResult


@dataclass
class Advance:
    dQdt: object
    dxdt: object
    stage: object


def conservedEuler(Q, dt, rate):
    return rate, Q + dt * rate


def advance(temporal, Q, pos, vel, dt, initialStage, *, rates, buildStage, velocity, wrap, hancockUpdate=None):
    """Exact historical recurrences; callbacks evaluate fresh spatial stages."""
    if temporal == 'hancock':
        dQdt, Qn = hancockUpdate() if hancockUpdate is not None else conservedEuler(Q, dt, rates(initialStage, Q, dt))
        return Advance(dQdt, .5 * (vel + velocity(Qn)), initialStage)
    if temporal == 'ssprk':
        f1 = rates(initialStage, Q, 0.)
        Qs = Q + dt * f1
        poss = wrap(pos + .5 * dt * (vel + velocity(Qs)))
        stage2 = buildStage(poss, initialStage)
        f2 = rates(stage2, Qs, 0.)
        Qn = .5 * Qs + .5 * (Q + dt * f2)
        return Advance((Qn - Q) / dt, .5 * (vel + velocity(Qn)), stage2)
    if temporal == 'kdk':
        f1 = rates(initialStage, Q, 0.)
        Qh = Q + .5 * dt * f1
        velh = velocity(Qh)
        stage2 = buildStage(wrap(pos + dt * velh), initialStage)
        f2 = rates(stage2, Qh, 0.)
        Qn = Qh + .5 * dt * f2
        return Advance((Qn - Q) / dt, velh, stage2)
    raise ValueError(f'unknown temporal scheme {temporal!r}')


def _field(system, tag):
    return get_tagged_attr(get_reference_state(system), tag=tag)


def step(method, state, dt, rhs, *args, **kwargs):
    reject_prior_step(method.name, kwargs.pop('priorStep', None))
    if not callable(getattr(state, 'conserved_velocity', None)) or not callable(getattr(state, 'wrap_positions', None)):
        raise TypeError('ConservedLagrangian needs conserved_velocity(Q) and wrap_positions(x) system hooks')
    kw = dict(kwargs, conservedIntegration=True, temporalScheme=method.temporal)
    weights = [1.] if method.temporal == 'hancock' else [.5, .5]
    initializeSystem(state, dt, *args, stageWeights=weights, **kw)
    stages, returns, updates = [], [], []
    predictionDt = dt if method.temporal == 'hancock' else 0.
    first = state.initializeNewState(*args, **kw)
    first.t = state.t
    k1, r1 = updateStep(state, first, predictionDt, rhs, *args, stageIndex=0, **kw)
    stages.append(first); returns.append(r1); updates.append(k1)
    Q, pos = _field(state, 'conserved'), _field(state, 'position')
    vel = get_tagged_attr(k1, tag='position_derivative')

    def build(positions, previous):
        stage = previous.initializeNewState(*args, **kw)
        set_tagged_attr(get_reference_state(stage), positions, tag='position')
        stage.t = hostTime(state.t + dt)
        return stage

    def rates(stage, quantity, prediction):
        if stage is first:
            return get_tagged_attr(k1, tag='conserved_derivative')
        set_tagged_attr(get_reference_state(stage), quantity, tag='conserved')
        k, r = updateStep(state, stage, prediction, rhs, *args, stageIndex=len(stages), **kw)
        stages.append(stage); returns.append(r); updates.append(k)
        return get_tagged_attr(k, tag='conserved_derivative')

    with record_function('[Integration] Conserved Lagrangian ' + method.temporal):
        result = advance(method.temporal, Q, pos, vel, dt, first, rates=rates, buildStage=build,
            velocity=state.conserved_velocity, wrap=state.wrap_positions)
        output = state.initializeNewState(*args, **kw)
        set_tagged_attr(get_reference_state(output), state.wrap_positions(pos + dt * result.dxdt), tag='position')
        set_tagged_attr(get_reference_state(output), Q + dt * result.dQdt, tag='conserved')
        output.t = hostTime(state.t + dt)
        finalizeSystem(output, state, dt, returns, updates, weights, *args, lastStageSystem=stages[-1], **kw)
    return IntegrationResult(output, stages=[StageResult(aux=r, update=k) for r, k in zip(returns, updates)])
