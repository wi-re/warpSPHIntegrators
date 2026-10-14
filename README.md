# warpSPHIntegrators

Solver-agnostic, differentiable ODE time integrators for PyTorch, designed for
systems with several physical fields and auxiliary buffers, including SPH.

[catalog:summary:start]: #

**142 registered schemes across 11 families**, with orders through 14 and 47 estimate emitters.

[catalog:summary:end]: #

## Install

Python 3.11 or newer:

```bash
python -m pip install warpSPHIntegrators
# For the current checkout and test tools:
python -m pip install -e '.[test]'
```

The catalog includes explicit, implicit, geometric, multistep, IMEX, stabilized
and exponential methods. [The scheme catalog](docs/docs/catalog.md) and
[selection guide](docs/docs/selection.md) explain capabilities and assumptions.

## A complete example

For a tensor ODE `dy/dt = f(t, y)`, use this complete float64 CPU example.
It is also available as [a standalone script](examples/tensor_ode.py).

[example:include examples/tensor_ode.py]: #
```python
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
```

[Tensor adapter and catalog filters](docs/docs/tensor-ode.md) cover ordinary ODEs.
[The run API](docs/docs/runs.md) adds fixed/adaptive trajectories, rejection,
sampling, events and checkpoints; see [the complete run example](examples/run_ode.py).
[Physical splitting](docs/docs/operator-splitting.md) covers autonomous compositions.
[The structural demo](docs/docs/structural-demo.md) shows slow/fast modes and controlled numerical damping on two coupled masses.
The [full protocol example](examples/harmonic_oscillator.py) covers tagged
position/velocity states and custom SPH lifecycle behavior.

## Documentation and development

- [Documentation site](https://fluids.dev/warpSPHIntegrators/),
  [quick start](docs/docs/quickstart.md), [API](docs/docs/api.md) and
  [solver stack](docs/docs/solver.md).
- [Examples and notebooks](examples/README.md), including downstream setup.
- [Contributor instructions](CONTRIBUTING.md): environment, multicore tests,
  registration, coefficient provenance and release checks.
- [Current execution plan](REPOSITORY_PLAN.md) and [developer index](docs/development/README.md).
- [Numerical evidence](docs/research/integrators/notes-index.md),
  [method dispositions](docs/plans/method-dispositions.md) and [benchmarks](benchmarks/README.md).
- [Detailed protocol examples and galleries](docs/development/usage-reference.md).

Stepping is PyTorch based; native Warp buffers remain future work. The run API owns acceptance and isolates rejected attempts; external effects
require application transaction hooks. Multistep history is fixed-step,
particle masking belongs to the user's system, and geometric methods retain
their force/separability assumptions. See [architecture](docs/development/architecture.md)
for the current boundaries.

## License and citation

[Apache 2.0](license.md). If used in research:

```bibtex
@software{warpSPHIntegrators2024,
  title={Differentiable ODE Integration with PyTorch},
  author={Winchenbach, Rene},
  url={https://github.com/wi-re/warpSPHIntegrators},
  year={2024}
}
```
