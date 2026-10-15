---
sidebar_label: Expanded RK
---

# Source-expanded Runge–Kutta

These are familiar low-storage-source Runge–Kutta constructions, executed through the full-stage engine. The method name describes its source construction, not a memory-saving promise in this backend. Choose a qualified map for its numerical properties and measure allocation separately.

`ExpandedRK(method)` executes 54 qualified main maps from a pinned low-storage
source family as ordinary full-stage Runge–Kutta. Orders range from two to five;
19 configurations expose a checked comparison estimate for adaptive timesteps.
The factory stays outside the fixed 142-scheme registry.

[example:run expanded-rk]: #
```python
import math
import torch
from warpSPHIntegrators.schemes import ExpandedRK
from warpSPHIntegrators.schemes.coefficients.expanded import expanded_rk_methods
from warpSPHIntegrators import TensorState, tensor_rhs, solve

assert len(expanded_rk_methods()) == 54
scheme = ExpandedRK('RDPK3Sp510')
trajectory = solve(scheme, TensorState(torch.ones(4, dtype=torch.float64)),
                   1., tensor_rhs(lambda t, y: -y),
                   dt=.1, adaptive=True, rtol=1e-8, atol=1e-10)
assert trajectory.successful
assert torch.allclose(trajectory.state.y,
                      torch.full((4,), math.exp(-1), dtype=torch.float64),
                      atol=1e-8, rtol=0)
```

`expanded_rk_methods()` lists the supported source names.
`expanded_rk_tableau(name)` returns immutable A, b and c arrays. Pair weights use
`(b_main, b_comparison)`; `estimate_order` is the power q for timestep control.
The default is `CarpenterKennedy2N54`. Names with an FSAL suffix retain their
source name, but this execution path does not reuse derivatives.

The pinned source is OrdinaryDiffEq 7.8.1 / LowStorageRK 3.4.0. Actual source
recurrences are expanded into affine stage maps; duplicate stages are merged and
a physical endpoint is reconstructed for copied fields. Main order conditions,
time nodes, nonlinear forced maps and source companions are screened separately.
Six source methods fail this binary64 gate and are omitted. The qualified
CKLLSRK85_4FM_4R main map is available without its failed source companion.
The [screening record](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/expanded-rk-validation.md)
explains the individual dispositions.

All integrated fields use the standard RK weights, including coupled quantities.
The standard stageWeights, stageIndex and compatible-energy rerun lifecycle is
preserved on owned buffers. Fixed/adaptive runs, portable tensor continuation,
physical composition and AlgorithmSwitch are supported. There is no dense output.

This is full-stage storage O(sN), retaining the normal PyTorch graph. It supplies
qualified numerical maps and a baseline for future allocation work; it does not
supply native low-storage buffers, source cache/controller equivalence, fused
stages or FSAL savings. Source low-dissipation names do not establish SSP or
conservation for a new physical problem.
