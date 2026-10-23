"""Moving-quadrature stepping: independent oscillator algebra and lifecycle contracts."""
from dataclasses import dataclass, field
import pytest
import torch
from warpSPHIntegrators.schemes import ConservedLagrangian, get_scheme
from warpSPHIntegrators import solve
from warpSPHIntegrators.runtime import save_checkpoint
from warpSPHIntegrators.state import BaseState, BaseIntegrationSystem, integrated, reference_state, tagged


@dataclass
class State(BaseState):
    x: torch.Tensor = integrated('dx', tags=('position',))
    q: torch.Tensor = integrated('dq', tags=('conserved',))


@dataclass
class Update:
    dx: torch.Tensor = tagged(tags=('position_derivative',))
    dq: torch.Tensor = tagged(tags=('conserved_derivative',))


@dataclass
class Oscillator(BaseIntegrationSystem):
    state: State = reference_state()
    t: float = 2.
    events: list = field(default_factory=list)

    def initializeNewState(self, **kwargs):
        return Oscillator(self.state.initializeNewState(), self.t, self.events)

    def conserved_velocity(self, q):
        return q[:, 1:2] / q[:, 0:1]

    def wrap_positions(self, x):
        return x

    def preprocess(self, initial, dt, **kwargs):
        self.events.append(('pre', kwargs['stageIndex'], self.t, dt, self.state.x.detach().clone()))
        return self

    def postprocess(self, stage, dt, r, **kwargs):
        self.events.append(('post', kwargs['stageIndex']))
        return self

    def finalize(self, initial, dt, returns, updates, weights, **kwargs):
        self.events.append(('final', weights))
        return self


def rhs(system, dt, **kwargs):
    x, q = system.state.x, system.state.q
    dq = torch.cat([torch.zeros_like(x), -x, torch.zeros_like(x)], -1)
    return Update(system.conserved_velocity(q), dq), 'aux'


@pytest.mark.parametrize('temporal', ['hancock', 'ssprk', 'kdk'])
def test_oscillatorAndFreshStageLifecycle(temporal):
    h = .125
    initial = Oscillator(State(torch.tensor([[1.]], dtype=torch.float64),
        torch.tensor([[1., .25, 3.]], dtype=torch.float64)))
    method = get_scheme(ConservedLagrangian(temporal))
    result = method(initial, h, rhs)
    v = .25 - h
    x = 1. + h * (.25 + v) / 2.
    if temporal != 'hancock':
        stageX = 1. + .25 * h - .5 * h * h
        v = .25 - .5 * h * (1. + stageX)
        x = stageX if temporal == 'kdk' else 1. + .5 * h * (.25 + v)
        assert initial.events[2][0:3] == ('pre', 1, 2. + h)
        assert initial.events[2][4].item() == pytest.approx(stageX)
    assert result.state.state.q[0, 1].item() == pytest.approx(v)
    assert result.state.state.x.item() == pytest.approx(x)
    assert result.state.t == 2. + h
    assert initial.state.x.item() == 1. and initial.state.q[0, 1].item() == .25
    assert len(result.stages) == method.metadata.stage_count
    assert [event[0] for event in initial.events] == (['pre', 'post', 'final'] if temporal == 'hancock' else ['pre', 'post', 'pre', 'post', 'final'])
    assert initial.events[0][3] == (h if temporal == 'hancock' else 0.)
    assert initial.events[-1] == ('final', [1.] if temporal == 'hancock' else [.5, .5])
    torch.testing.assert_close(result.state.state.q[:, [0, 2]], initial.state.q[:, [0, 2]], rtol=0, atol=0)


@pytest.mark.parametrize('temporal', ['hancock', 'ssprk', 'kdk'])
def test_stateGradients(temporal):
    def solution(values):
        initial = Oscillator(State(values[:1].reshape(1, 1), values[1:].reshape(1, 3)))
        state = ConservedLagrangian(temporal)(initial, .03, rhs).state.state
        return torch.cat([state.x.flatten(), state.q.flatten()])
    assert torch.autograd.gradcheck(solution, (torch.tensor([1., 1., .25, 3.], dtype=torch.float64, requires_grad=True),))


def test_invalidTemporalAndSystemFailClearly():
    with pytest.raises(ValueError, match='temporal'):
        ConservedLagrangian('bogus')
    with pytest.raises(TypeError, match='system hooks'):
        ConservedLagrangian()(object(), .1, rhs)
    initial = Oscillator(State(torch.ones(1, 1), torch.tensor([[1., .25, 3.]])))
    with pytest.warns(RuntimeWarning, match='does not support first-stage reuse'):
        result = ConservedLagrangian()(initial, .1, rhs, priorStep=object())
    assert len(result.stages) == 1


@pytest.mark.parametrize('temporal', ['hancock', 'ssprk', 'kdk'])
def test_runContinuationPreservesTheConsumerMap(temporal, tmp_path):
    initial = Oscillator(State(torch.tensor([[1.]], dtype=torch.float64),
        torch.tensor([[1., .25, 3.]], dtype=torch.float64)))
    method = ConservedLagrangian(temporal)
    whole = solve(method, initial, 2.25, rhs, dt=.03125)
    first = solve(method, initial, 2.125, rhs, dt=.03125)
    saved = first.checkpoint.state.state.q.clone()
    resumed = solve(method, None, 2.25, rhs, restart=first.checkpoint)
    assert whole.successful and resumed.successful
    torch.testing.assert_close(resumed.state.state.x, whole.state.state.x, rtol=0, atol=0)
    torch.testing.assert_close(resumed.state.state.q, whole.state.state.q, rtol=0, atol=0)
    assert torch.equal(first.checkpoint.state.state.q, saved)
    assert initial.state.x.item() == 1. and initial.state.q[0, 1].item() == .25
    with pytest.raises(TypeError, match='plain TensorState'):
        save_checkpoint(first.checkpoint, tmp_path / 'custom-state.pt')
