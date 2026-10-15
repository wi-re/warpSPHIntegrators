# Algorithm switching

`AlgorithmSwitch(methods, selector, policy_id)` chooses one complete integrator
for each trial. It supports registered methods and the Collocation, Extrapolation,
Nystrom, StabilizedRK, GeneralizedAlpha, VariableAdams, AdaptiveAdams, NDF, AdaptiveNDF, ExpandedRK and
physical SplitScheme factories. This is algorithm selection, separate from
splitting a physical RHS into subflows.

The pure `selector(state, previous_index)` returns a zero-based Python integer.
It receives an owned state probe and the last accepted index, or None after a
reset. Integrated-value, time and ID mutations are rejected. Other probe changes
are discarded. A policy can use declared problem information and hysteresis;
the package does not supply an automatic stiffness estimator.

[example:run switching with a declared stiffness bound]: #
```python
import math
import torch
from warpSPHIntegrators.wrappers import AlgorithmSwitch
from warpSPHIntegrators.schemes import NDF
from warpSPHIntegrators import TensorState, tensor_rhs, solve

def rate(t):
    return 1 + 999 * math.sin(math.pi * t)**2

def choose(state, previous):
    return int(rate(state.t) > (10 if previous == 1 else 30))

method = AlgorithmSwitch(('dormandPrince', NDF(3)), choose, 'declared-rate-v1')
rhs = tensor_rhs(lambda t, y: -rate(t) * (y - math.exp(-t)) - math.exp(-t))
records = []
result = solve(method, TensorState(torch.ones(16, dtype=torch.float64)), 2., rhs,
               dt=.02, dt_max=.1, adaptive=True, rtol=1e-6, atol=1e-8,
               solver_opts={'matvec': 'jvp', 'tol': 1e-13, 'newton_tol': 1e-8},
               callback=records.append)
assert result.successful
assert abs(result.state.y[0].item() - math.exp(-2)) < 1e-5
assert {r.result.history.context['index'] for r in records} == {0, 1}
```

Same-method steps retain owned numeric history. Switching cold-starts the next
method; returning to an earlier method also cold-starts it. History never crosses
method boundaries. Constant-step histories clear before a changed h is consumed.
State-value, layout, dtype, ID or time changes invalidate stateful memory. Call
with `restart_history=True` after changing RHS parameters or a derivative
discontinuity. Warm numeric methods must preserve the initial integrated anchor.

The runner uses the selected child's per-result estimator exponent and delegates
accepted step-size control to that child. NDF holds and AdaptiveNDF/AdaptiveAdams order changes
operate on unwrapped owned history before external commit. Rejections discard
the trial's selection/history; the policy sees the last accepted index again.
Solver/RHS work diagnostics come from the actual selected method.

Every candidate must emit an estimate for `adaptive=True`. A fixed run may use
methods without estimates. All candidates must support the state's kind and
their RHS/options contracts. Normal RHS positional arguments and step options
are passed to the selected child. Its initialize, preprocess, postprocess and
finalize behavior applies once; the wrapper adds no lifecycle boundary or extra
RHS evaluation. Copied-field behavior remains that of the selected child.

The factory's `order` is None: frequent switches and repeated cold starts can
change trajectory order. In particular, methods with low-order startup can lose
their uninterrupted-history order under frequent restarts. Per-child formal
orders do not prove an overall order for an arbitrary policy.
FSAL reuse is disabled, including consecutive selections of the same method.
Dense interpolation, off-grid samples and event localization are unavailable.
No upstream CompositeAlgorithm cache transfer or controller equivalence is claimed.

`result.history.context` records `index`, canonical `selected` identifier, time,
stateful validity guards and nested `child_context`. The entries belong to the
active child. Portable tensor checkpoints save this memory, not executable policy
code. Reconstruct the same methods and pure selector with the same `policy_id`;
change that ID when its semantics change. In-memory checkpoints retain gradients;
discrete policy decisions are not differentiated. External policy state is not
automatically checkpointed or rolled back.

The varying-stiffness benchmark uses analytic manufactured solutions and declared
diagonal bounds. Switching avoids explicit stability restrictions in stiff regions;
an always-implicit baseline can still be cheaper. This interface does not promise
automatic speedups or infer a suitable SPH spatial preconditioner.
