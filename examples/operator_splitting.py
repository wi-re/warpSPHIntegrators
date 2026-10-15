"""Physical operator splitting: python examples/operator_splitting.py."""

import math
import torch
from warpSPHIntegrators.wrappers import (
    ConservativeDissipativeRHS, Strang, LinearExponentialFlow,
)
from warpSPHIntegrators.adapters import SemilinearRHS
from warpSPHIntegrators import TensorState, tensor_rhs, solve


def run_example():
    # q' = v, v' = -4*q - .5*v: motion and damping are separate maps.
    initial = TensorState(torch.tensor([1., 0.], dtype=torch.float64))
    damping = torch.diag(torch.tensor([0., -.5], dtype=torch.float64))
    rhs = tensor_rhs(ConservativeDissipativeRHS(
        conservative=lambda t, y: torch.stack((y[1], -4*y[0])),
        dissipative=SemilinearRHS(linear=lambda t, y: damping@y)))
    method = Strang('RK4', LinearExponentialFlow(damping))
    result = solve(method, initial, 1., rhs, dt=.01)
    omega = math.sqrt(4 - .25**2)
    exact = torch.tensor([
        math.exp(-.25)*(math.cos(omega) + .25/omega*math.sin(omega)),
        -4/omega*math.exp(-.25)*math.sin(omega)], dtype=torch.float64)
    assert method.order == 2
    assert result.state.t == 1.
    assert torch.linalg.vector_norm(result.state.y-exact) < 1e-4
    assert torch.equal(initial.y, torch.tensor([1., 0.], dtype=torch.float64))
    return result


if __name__ == '__main__':
    result = run_example()
    print(f't={result.state.t}, [q, v]={result.state.y.tolist()}, steps={result.accepted_steps}')
