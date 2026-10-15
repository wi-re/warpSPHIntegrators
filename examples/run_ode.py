"""Controlled tensor ODE run: python examples/run_ode.py."""

import math
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, solve
from warpSPHIntegrators.runtime import Event


def run_example():
    initial = TensorState(torch.ones(2, dtype=torch.float64))
    rhs = tensor_rhs(lambda t, y: -y)
    result = solve('Dormand-Prince 5(4)', initial, 1., rhs, dt=.3,
                   adaptive=True, rtol=1e-8, atol=1e-10,
                   sample_times=[0., .25, .5, .75, 1.])
    assert result.state.t == 1.
    assert torch.allclose(result.state.y, initial.y * math.exp(-1), rtol=3e-8)
    assert torch.equal(initial.y, torch.ones_like(initial.y))
    resumed = solve('Dormand-Prince 5(4)', None, 2., rhs,
                    restart=result.checkpoint,
                    events=[Event(lambda t, s: s.y[0].item() - .2,
                                  name='threshold', terminal=True)])
    assert resumed.termination == 'event'
    assert abs(resumed.state.t - math.log(5)) < 1e-7
    return result, resumed


if __name__ == '__main__':
    result, resumed = run_example()
    print(f'accepted={result.accepted_steps}, rejected={result.rejected_steps}, '
          f'event={resumed.events[-1].name} at t={resumed.state.t:.8f}')
