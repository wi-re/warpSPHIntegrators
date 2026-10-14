# Ordinary tensor ODEs

For `dy/dt = f(t, y)`, use `TensorState` and `tensor_rhs` without writing
lifecycle hooks. The adapter runs on the existing drivers and returns their
ordinary `IntegrationResult`, with the new tensor available as `result.state.y`.

This complete float64 CPU example is also saved in `examples/tensor_ode.py`.

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

Scalar, vector and batched tensors preserve shape, dtype, device and autograd
connections. Supply a nonempty floating-point tensor; each derivative must have
exactly the same shape, dtype and device. Treat RHS inputs as read-only. Integer
and complex states are not supported by this adapter. Bind named parameters in
`tensor_rhs(f, rate=...)` or close over them; solver/logging kwargs do not reach
`f(t, y)`. Positional step arguments do reach `f`.

## Discovery and driver requirements

`filter_schemes` returns registered records in catalog order. Filters combine
with AND; no matches returns an empty tuple, and invalid criteria raise ValueError.

[example:run selection]: #
```python
from warpSPHIntegrators import filter_schemes

pairs = filter_schemes(family='ExplicitRK', min_order=5,
                       has_error_estimate=True, supports_reuse=True,
                       state_kind='tensor')
assert pairs
assert all(s.order >= 5 and s.has_error_estimate for s in pairs)
```

Other filters include exact/max order, implicit solves, dense output, history,
separability, drift-field support and available RHS capabilities. An empty
`available_capabilities` set excludes ETD2RK, whose linear action is mandatory;
optional split accessors do not restrict combined-RHS methods.

The adapter supports the 135 catalog records whose `metadata.state_kind` is
`tensor`. Geometric and Newmark methods require a position/velocity system;
using their records with TensorState fails before stepping. Use the
[full state protocol](quickstart) for those methods and SPH fields. The catalog
also retains separability/force assumptions and history requirements; consult
the family page before choosing a method.

## Splits, history and estimates

`tensor_rhs` accepts `IMEXRHS` and `SemilinearRHS` whose callbacks use `(t, y)`;
their declared capabilities remain intact. For example, a linear stiff mode:

[example:run splits]: #
```python
import math
import torch
from warpSPHIntegrators import TensorState, tensor_rhs, SemilinearRHS, getIntegrator

state = TensorState(torch.ones(2, dtype=torch.float64))
rhs = tensor_rhs(SemilinearRHS(linear=lambda t, y: -10*y))
result = getIntegrator('ETD2RK')(state, .1, rhs)
assert torch.allclose(result.state.y, torch.full_like(state.y, math.exp(-1)))
```

For multistep methods, pass `history=StepHistory(maxlen=...)` and thread
`result.history` into subsequent calls; `metadata.history_length` gives the
required number of retained previous entries. A missing history repeatedly uses
startup, and changing `dt` restarts history. `estimate_error_norm(result)` and
`propose_dt(..., order=scheme.estimate_order)` remain caller-driven helpers;
DP5 dense output takes the initial state and `result.stages`.

The adapter preserves the chosen driver's gradient semantics. Explicit and
implicit tensor state/parameter paths are tested. ROS3P/EXPRB32 freeze operators;
their existing gradient qualifications still apply, including incomplete
differentiation through parameter-dependent frozen operators. Use [the run API](runs) for fixed/adaptive trajectories, acceptance, sampling,
events and restart checkpoints.

## Selecting a conventional default

`getPreferredIntegrator(order)` returns catalog records for orders 1–5:
Forward Euler, Midpoint, SSP RK3, RK4 and Dormand-Prince 5(4). These are explicit
starting points for ordinary ODEs. Stiffness/geometry-aware selection uses the
catalog and family guidance. `getPreferredScheme` preserves its historical raw
callable choices, some of which require position/velocity fields; it is retained
for existing callers and is not the ordinary tensor-ODE selector.

See [step and solver options](options) for the common call surface.
