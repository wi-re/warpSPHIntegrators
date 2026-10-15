---
sidebar_label: Extrapolation
---

# Explicit extrapolation backends

Extrapolation combines several coarse/fine integrations to cancel leading errors. It can reach high order with a simple base method, but the extra columns cost RHS work and retain full stages here. The production factory below uses explicit Euler or midpoint columns.

`Extrapolation(base='midpoint', order=4)` binds a fixed target order outside the
142-method catalog. It advances independent subdivision columns and combines
their endpoint increments. Euler supports orders 2–8 with Romberg subdivisions
1,2,4,…; midpoint supports even orders 2–10 with subdivisions 2,4,6,… .
The midpoint recurrence is the source's unsmoothed even-subdivision form.
Both order-two main maps equal explicit midpoint, with an Euler comparison.

[example:run explicit extrapolation]: #
```python
import math
import torch
from warpSPHIntegrators.schemes import Extrapolation
from warpSPHIntegrators import TensorState, tensor_rhs, solve

records = []
result = solve(Extrapolation('midpoint', 6),
               TensorState(torch.ones(4, dtype=torch.float64)), 1.,
               tensor_rhs(lambda t, y: -y), dt=.2, adaptive=True,
               rtol=1e-8, atol=1e-10, callback=records.append)
assert result.successful
assert abs(result.state.y[0].item() - math.exp(-1)) < 2e-8
assert records[-1].result.quadrature.subdivisions == (2, 4, 6)
```

`extrapolation_coefficients(base, order)` returns immutable subdivision, main
and comparison column weights. Weights are derived using exact fractions before
conversion to binary64. Euler cancels powers of 1/n; midpoint cancels powers
of 1/n². The comparison omits the coarsest column. Euler's error exponent is
the target order; midpoint's is order minus one, except order two uses exponent
two. The runner uses `result.estimate_order`; adaptation changes h, not order.

The initial derivative is shared across columns. A physical endpoint evaluation
refreshes copied fields and has zero final weight. The total full RHS count is
`2 + sum(n - 1 for n in subdivisions)`. There is no cross-step FSAL saving.
For example, midpoint orders 4/6/8/10 cost 6/11/18/27 evaluations per step;
Euler order eight costs 249. All columns execute serially. Returned stages and
gradient graphs retain full tensors; no low-storage or internal parallelism
claim is made. High orders can reach floating-point floors on fine grids.

All integrated fields, including coupled quantities, use the same affine step.
Custom finalize opts into `extrapolation_compatible=True` and reads
`ExtrapolationQuadrature` on the result. `weights` and `comparison_weights`
correspond to returned stages, including the zero-weight endpoint.
`subdivisions` and `columns` identify each column's stage indices; the initial
stage can occur in several columns. Finalize must preserve integrated values,
time and particle IDs. Stage times follow the actual subdivision grids and the
RHS/lifecycle receives the outer dt. Initial state and stage buffers are owned.

The factory works in physical flow compositions. Inner lifecycle hooks are
suppressed and the outer hook can inspect each substep's quadrature. This does
not confer symplecticity or conservation on the extrapolated method.
Tensor continuation uses the normal portable checkpoint path without history.

These are explicit nonstiff methods. The fixed-column main maps and comparison
formulas are checked against pinned AitkenNeville and midpoint source kernels;
source adaptive-order/work controllers and threaded columns are separate.
Implicit extrapolation also remains separate: the inspected source uses frozen
Jacobian linear corrections, which require a different solver/operator contract.
Use measured accuracy and work to choose a method; extrapolation does not
automatically outperform the existing embedded RK families.
