"""Full projected maps: single evaluation, temporal order, constraints and AD."""

from dataclasses import dataclass, field
import pytest
import torch
from warpSPHIntegrators.state import BaseState, BaseIntegrationSystem, constant, reference_state
from warpSPHIntegrators.schemes import ProjectedStep, get_scheme
from warpSPHIntegrators.drivers.mechanics.projection import kick, drift, symplecticProjection
from warpSPHIntegrators import solve
from warpSPHIntegrators.runtime import save_checkpoint
from warpSPHIntegrators.state import integrated


@dataclass
class State(BaseState):
    positions: torch.Tensor = constant()
    velocities: torch.Tensor = constant()


@dataclass
class MapSystem(BaseIntegrationSystem):
    state: State = reference_state()
    t: float = 2.0
    events: list = field(default_factory=list)
    projected_map = True

    def initializeNewState(self, **kwargs):
        return MapSystem(self.state.initializeNewState(), self.t, self.events)

    def initialize(self, dt, **kwargs):
        self.events.append("init")
        return self

    def preprocess(self, initial, dt, **kwargs):
        self.events.append(("pre", self.t))
        return self

    def postprocess(self, stage, dt, aux, **kwargs):
        self.events.append("post")
        return self

    def finalize(self, initial, dt, returns, updates, weights, **kwargs):
        self.events.append(("final", weights))
        stage = returns[-1][0]
        self.state = stage.state
        return self


def map_rhs(system, dt, **kwargs):
    system.events.append("map")
    symplecticProjection(system.state, dt, torch.full_like(system.state.velocities, -2.0))
    return None, system


@pytest.mark.parametrize("dt", [0.01, 0.125])
def test_oneFullMapLifecycleAndPhysicalTime(dt):
    initial = MapSystem(
        State(
            torch.tensor([[1.0]], dtype=torch.float64), torch.tensor([[3.0]], dtype=torch.float64)
        )
    )
    method = get_scheme(ProjectedStep())
    out = method(initial, dt, map_rhs)
    assert out.state.t == 2.0 + dt
    assert len(out.stages) == 1
    assert initial.events == ["init", ("pre", 2.0), "map", "post", ("final", [1.0])]
    assert out.state.state.velocities.item() == pytest.approx(3.0 - 2.0 * dt)
    assert out.state.state.positions.item() == pytest.approx(1.0 + dt * (3.0 - 2.0 * dt))
    assert initial.state.positions.item() == 1.0
    assert initial.state.velocities.item() == 3.0


def test_maskedKickDriftAndShiftEvaluationOrder():
    state = State(
        torch.tensor([[1.0], [8.0]], dtype=torch.float64),
        torch.tensor([[2.0], [6.0]], dtype=torch.float64),
    )
    seen = []

    def shift(st):
        seen.append((st.positions.clone(), st.velocities.clone()))
        return torch.tensor([[0.1], [0.0]], dtype=torch.float64)

    symplecticProjection(
        state,
        0.5,
        torch.ones_like(state.velocities),
        mask=torch.tensor([[True], [False]]),
        beforeDrift=shift,
    )
    assert torch.equal(seen[0][0], torch.tensor([[1.0], [8.0]], dtype=torch.float64))
    assert torch.equal(seen[0][1], torch.tensor([[2.5], [6.0]], dtype=torch.float64))
    assert torch.equal(state.positions, torch.tensor([[2.35], [8.0]], dtype=torch.float64))
    assert torch.equal(state.velocities, seen[0][1])


def test_projectedMapGradientsAndZeroStep():
    def advance(xv):
        initial = MapSystem(State(xv[:1].reshape(1, 1), xv[1:].reshape(1, 1)))
        result = ProjectedStep()(initial, 0.125, map_rhs).state.state
        return torch.cat((result.positions.flatten(), result.velocities.flatten()))

    assert torch.autograd.gradcheck(
        advance, (torch.tensor([1.0, 3.0], dtype=torch.float64, requires_grad=True),)
    )
    x, v, a = (torch.tensor([[n]], dtype=torch.float64) for n in (1.0, 3.0, 4.0))
    assert torch.equal(kick(v, 0.0, a), v)
    assert torch.equal(drift(x, 0.0, v), x)


def test_requiresProjectedSystemAndIgnoresPriorStage():
    with pytest.raises(TypeError, match="projected_map"):
        ProjectedStep()(object(), 0.1, map_rhs)
    initial = MapSystem(State(torch.ones(1, 1), torch.zeros(1, 1)))
    with pytest.warns(RuntimeWarning, match="does not support first-stage reuse"):
        out = ProjectedStep()(initial, 0.1, map_rhs, priorStep=object())
    assert len(out.stages) == 1


def test_integratedProjectionStateContinuesWithoutRepeatingTheMap(tmp_path):
    @dataclass
    class IntegratedState(BaseState):
        positions: torch.Tensor = integrated('dx', tags=('position',))
        velocities: torch.Tensor = integrated('dv', tags=('velocity',))

    initial = MapSystem(IntegratedState(
        torch.tensor([[1.]], dtype=torch.float64), torch.tensor([[3.]], dtype=torch.float64)))
    whole = solve(ProjectedStep(), initial, 2.25, map_rhs, dt=.03125)
    first = solve(ProjectedStep(), initial, 2.125, map_rhs, dt=.03125)
    saved = first.checkpoint.state.state.positions.clone()
    resumed = solve(ProjectedStep(), None, 2.25, map_rhs, restart=first.checkpoint)
    assert whole.successful and resumed.successful
    assert resumed.state.state.positions.item() == 1.6796875
    assert resumed.state.state.velocities.item() == 2.5
    torch.testing.assert_close(resumed.state.state.positions, whole.state.state.positions, rtol=0, atol=0)
    torch.testing.assert_close(resumed.state.state.velocities, whole.state.state.velocities, rtol=0, atol=0)
    assert sum(e == 'map' for e in whole.state.events) == 8
    assert torch.equal(first.checkpoint.state.state.positions, saved)
    assert initial.state.positions.item() == 1. and initial.state.velocities.item() == 3.
    with pytest.raises(TypeError, match='plain TensorState'):
        save_checkpoint(first.checkpoint, tmp_path / 'custom-projection.pt')


def test_constantOnlyProjectionStateUsesDirectStepping():
    initial = MapSystem(State(torch.ones(1, 1), torch.zeros(1, 1)))
    with pytest.raises(ValueError, match='requires integrated tensor fields'):
        solve(ProjectedStep(), initial, 2.25, map_rhs, dt=.03125)
