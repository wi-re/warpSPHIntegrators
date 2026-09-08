import pytest
import torch

from warpSPHIntegrators import IMEXRHS, getIntegrator, testing


def test_imex_euler_accepts_an_ordinary_rhs_as_fully_implicit():
    imex = getIntegrator('IMEX Euler')
    backward_euler = getIntegrator('Backward Euler (implicit)')
    problem = testing.PROBLEMS['oscillator']()
    imex_result = imex(problem.initial(), dt=0.1, f=problem.rhs)
    backward_euler_result = backward_euler(problem.initial(), dt=0.1, f=problem.rhs)
    imex_state = testing.get_reference_state(imex_result.state)
    backward_euler_state = testing.get_reference_state(backward_euler_result.state)
    assert torch.allclose(imex_state.x, backward_euler_state.x)
    assert torch.allclose(imex_state.u, backward_euler_state.u)


def test_imex_euler_uses_explicit_rhs_once_and_implicit_rhs_in_the_solve():
    imex = getIntegrator('IMEX Euler')
    problem = testing.PROBLEMS['oscillator']()
    explicit_calls = [0]
    implicit_calls = [0]

    def explicit_rhs(system, dt, **kwargs):
        explicit_calls[0] += 1
        state = testing.get_reference_state(system)
        return testing.ParticleUpdate(
            dxdt=state.u.clone(),
            dudt=torch.zeros_like(state.u),
            dedt=torch.zeros_like(state.e),
        ), None

    def implicit_rhs(system, dt, **kwargs):
        implicit_calls[0] += 1
        state = testing.get_reference_state(system)
        return testing.ParticleUpdate(
            dxdt=torch.zeros_like(state.x),
            dudt=-4.0 * state.x,
            dedt=torch.zeros_like(state.e),
        ), None

    result = imex(problem.initial(), dt=0.1, f=IMEXRHS(explicit_rhs, implicit_rhs))
    state = testing.get_reference_state(result.state)
    assert explicit_calls[0] == 1
    assert implicit_calls[0] > 1
    assert state.x.item() == pytest.approx(1.0)
    assert state.u.item() == pytest.approx(-0.4)