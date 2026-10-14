"""Ordinary tensor ODE: python examples/tensor_ode.py."""

import math
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, filter_schemes, getPreferredIntegrator


def run_example():
    # Scalar, vector or batched floating-point tensors use the same interface.
    initial = torch.tensor([1., 2.], dtype=torch.float64)
    state = TensorState(initial, t=0.)
    scheme = getPreferredIntegrator(5)  # registered Dormand-Prince 5(4)
    rhs = tensor_rhs(lambda t, y, rate: -rate*y, rate=2.)
    for _ in range(100):
        result = scheme(state, .01, rhs)
        state = result.state
    assert torch.allclose(state.y, initial * math.exp(-2), rtol=1e-8, atol=1e-10)
    return result


if __name__ == '__main__':
    result = run_example()
    pairs = filter_schemes(family='ExplicitRK', has_error_estimate=True, supports_reuse=True)
    print(f't={result.state.t:.2f}, y={result.state.y.tolist()}, reusable pairs={len(pairs)}')
