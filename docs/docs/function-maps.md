# Direct state maps

`FunctionMap()` applies one discrete map per fixed step. The declared map
returns the complete next state; it returns neither a rate nor a correction to
be integrated. `order` is None and there is no adaptive error estimate, FSAL,
dense interpolation or event localization. The fixed catalog remains separate.

`tensor_map(function, **parameters)` adapts
`function(t_next, y_old, h, *args, **parameters) -> y_next` for TensorState.
Shape, dtype and device stay unchanged. Positional arguments and bound map
parameters are forwarded; solver/step options are not passed to that function.

[example:run a known discrete recurrence]: #
```python
import torch
from warpSPHIntegrators import FunctionMap, TensorState, tensor_map, solve, MapTransition

records = []
result = solve(FunctionMap(), TensorState(torch.ones(4, dtype=torch.float64)),
               1., tensor_map(lambda t, y, h, ratio: ratio*y, ratio=.9),
               dt=.1, callback=records.append)
assert result.successful and result.accepted_steps == 10
assert torch.allclose(result.state.y, torch.full((4,), .9**10, dtype=torch.float64),
                      atol=1e-14, rtol=0)
assert isinstance(records[-1].result.transition, MapTransition)
assert records[-1].result.quadrature is None
```

The source clock is explicit: the map sees initialized old values with time
`t_next=t+h`. The returned state has that endpoint time. Each call applies one
map. For an h-independent recurrence, h labels the clock and endpoints should be
chosen for the desired number of iterations. The runner clips a final partial
step as usual, which still applies one map. An h-aware analytic flow can instead
use the supplied h. This does not make FunctionMap a generic exact ODE solver.

Use `StateMap(function)` for full-state systems. The callback receives an owned,
preprocessed system at t+h and returns a mapped system, optionally with auxiliary
values. It may mutate that owned input. The output preserves integrated field
layout/dtype/device, time and particle IDs; it is snapshotted before lifecycle
hooks so externally returned buffers remain owned. All numerical transitions,
including coupled quantities, belong in this callback.

Initialize establishes the old input once. Preprocess runs before the map and
again on the mapped output to refresh copied endpoint fields. The map itself is
called once. Postprocess receives the mapped state and source auxiliaries, then
finalize runs once. Preprocess/postprocess/finalize preserve the numerical map
values, clock and IDs. A custom finalize opts into `map_compatible=True` and
receives MapTransition in place of derivative weights; its update list is `[None]`.
The initial integrated anchor remains unchanged by map lifecycle evaluation.

`result.transition` contains owned `input_state`, `evaluation_time` and
`mapped_state`. It describes a complete nonlinear transition, separately from
`result.quadrature`. The one StageResult records map auxiliaries and has
`update=None`; there is no derivative stage. Nonfinite maps fail before endpoint
hooks/finalize, and the runner rolls back failed trials or caller exceptions.

Portable tensor checkpoints and differentiable in-memory continuation use the
ordinary runner. Reconstruct the same map and its parameters when resuming;
executable callbacks are not serialized. The map interface is separate from
ODE algorithm switching and autonomous physical flow composition.

The source `scale_by_time` branch is expressible explicitly as
`tensor_map(lambda t,y,h: y+h*rate(t,y))`. That evaluates the rate at t_next with
old values, as the pinned step body does. There is no overloaded derivative/map
flag. Source default-identity and step-limiter machinery are not part of this API.
Known affine recurrences and a damped rotation that generates heat provide
analytic consumer checks; their exact map accuracy is specific to those models.
