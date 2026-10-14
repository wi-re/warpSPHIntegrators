---
sidebar_label: Multistep (explicit)
---

# Explicit linear multistep — Adams-Bashforth, PECE & SSPRKMSVS (`MultistepExplicit`) — 10 schemes

The explicit linear multistep family, driven by `multistep.py` (the Adams
family) and `ssprkmsvs.py` (Shu's SSP multistep, below):

$$
y^{n+1} = y^n + \mathrm{dt}\, \sum_{j=0}^{p-1} \beta_j\, f^{n-j},
$$

where $f^{n-j} = f(t^{n-j}, y^{n-j})$ are **stored past right-hand-side
evaluations** carried in the scheme's `StepHistory` (derivative history,
threaded across calls with `history=`). No solves.

## Variable-step backend

`VariableAdams(order=3, kind='abm')` supplies fixed-order variable-step Adams at
orders 3–5, with fixed or adaptive `solve`. Choose `kind='ab'` for the pure
predictor or `kind='abm'` for the source's order-(p-1) predictor and order-p
corrector. These six parameter choices are separate from the fixed catalog.
Coefficients integrate the Lagrange basis on the actual time grid and match the
pinned source's divided-difference helpers on constant and irregular histories.

[example:run variable_adams]: #
```python
import math
import torch
from warpSPHIntegrators import VariableAdams, TensorState, tensor_rhs, solve

method = VariableAdams(4, 'abm')
result = solve(method, TensorState(torch.ones(1, dtype=torch.float64)), 1.,
               tensor_rhs(lambda t, y: -y), dt=.1,
               adaptive=True, rtol=1e-7, atol=1e-10)
assert result.successful
assert abs(result.state.y.item() - math.exp(-1)) < 1e-5
```

AB compares with an order-(p-1) predictor; ABM uses the source same-order
difference estimate with exponent p+1. Neither a same-order difference nor a
requested local tolerance guarantees that exact global error. DP5 startup uses
an available-history Adams comparison with exponents 2,3,... before switching
to the steady exponent. The runner uses the exponent on each result and respects
the method's maximum step growth, so an optimistic starter does not cause
repeated resets. History stays valid across ordinary step changes. Large ratio
changes, time discontinuities, incompatible history/layouts and changed particle
IDs restart safely. Pass `restart_history=True` for RHS/parameter discontinuities.

Direct callers must thread accepted `result.history`; `solve` owns that threading,
rollback and checkpointing. Tensor checkpoints preserve primitive/tensor history
context through the weights-only format. Input state/history are cloned. Endpoint
copied fields come from the corrected physical state; warm AB/ABM therefore cost
two/three RHS calls, including re-evaluation of the start and corrected endpoint.
These are explicit nonstiff methods. See [backend evidence](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/variable-adams-validation.md).

## Let Adams select its target order

`AdaptiveAdams(max_order=5)` adapts the target order between three and the
configured maximum (3–5). It starts at target three with the same DP5 bootstrap
and per-step estimate exponents as VariableAdams. Once history is mature, it
compares source-derived neighboring-order defects using the runner's field norm.
It keeps accepted derivatives when changing order and holds the timestep for
one step. Direct calls and fixed-step runs keep their current target order.

[example:run adaptive_adams]: #
```python
import math
import torch
from warpSPHIntegrators import AdaptiveAdams, TensorState, tensor_rhs, solve

records = []
method = AdaptiveAdams(max_order=5)
result = solve(method, TensorState(torch.ones(4, dtype=torch.float64)), 4.,
               tensor_rhs(lambda t, y: -y), dt=.01,
               adaptive=True, rtol=1e-8, atol=1e-10, callback=records.append)
assert result.successful
assert {r.result.history.context['last_order'] for r in records} == {3, 4, 5}
assert abs(result.state.y[0].item() - math.exp(-4)) < 1e-7
```

`order` is None because target order changes. The history context records
`last_order` for the completed step and `order` for the next target. The actual
controller exponent is on `IntegrationResult.estimate_order`, including startup.
`adams_order_choice` exposes the bounded mature neighbor comparison. Selection
waits for p+1 warm accepted steps; a zero defect does not promote order. Adjacent
defects add state arithmetic but no RHS evaluations. Warm steps retain the
VariableAdams three-call lifecycle and corrected endpoint refresh.

The source is pinned VCABM, but this is a bounded package controller, not its
complete order-1–12 startup/FSAL/retry policy. `solve` owns rejection and commits
order/history together. Manual loops must pass accepted history. Edited values,
layout/dtype/time/IDs, foreign history or excessive step ratios reset before
history is consumed. Pass `restart_history=True` after changing RHS parameters.
Tensor-file continuation and AlgorithmSwitch delegation are supported. Physical
splitting, dense output and events are not added by this controller.

On a 4,096-component known reaction vector at local rtol=1e-8, target-order
selection uses 336 RHS entries with 6.40e-9 final max error, versus fixed target
three's 951/4.43e-8 and fixed target five's 338/6.40e-9. On a damped rotation/
heat vector, however, the same tolerance gives 8.09e-6 max error versus DP5's
1.47e-8. Requested local tolerances do not guarantee global observable error.
The current controller also costs more wall time than fixed target five on
these cheap RHS functions; fewer RHS entries alone do not establish a speedup.
See [source and consumer evidence](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/adaptive-adams-validation.md).

## Adams-Bashforth (pure predictor)

`AB2`–`AB5` evaluate the RHS **once** per step (at the new point) and apply the
order-$p$ Adams-Bashforth weights to it and the stored history. Weights
(newest first, $\beta_0$ weights $f^n$), from `getABCoefficients`:

| Order | $\beta$ weights (newest first) |
|---|---|
| 2 | $\tfrac{3}{2},\ -\tfrac{1}{2}$ |
| 3 | $\tfrac{23}{12},\ -\tfrac{16}{12},\ \tfrac{5}{12}$ |
| 4 | $\tfrac{55}{24},\ -\tfrac{59}{24},\ \tfrac{37}{24},\ -\tfrac{9}{24}$ |
| 5 | $\tfrac{1901}{720},\ -\tfrac{2774}{720},\ \tfrac{2616}{720},\ -\tfrac{1274}{720},\ \tfrac{251}{720}$ |

Cost: **1 evaluation per step** after startup — the cheapest order-2+ method in
the registry. The trade: zero-stable but not A-stable, with a bounded (and
order-dependent) stability region on the real axis; and a cold start.

## Adams-Bashforth-Moulton (PECE predictor-corrector)

`ABM2`–`ABM5` evaluate the current derivative, then run a **predictor–corrector** pair per step:

1. **P**redict $y^{n+1}$ with the order-$p$ Adams-Bashforth weights (the table
   above) — no new evaluation;
2. **E**valuate $f^{n+1} = f(t^{n+1}, y^{n+1}_{\text{pred}})$ — one evaluation;
3. **C**orrect with the order-$p$ Adams-Moulton weights (newest first,
   predicted point first):

| Order | $\gamma$ corrector weights |
|---|---|
| 2 | $\tfrac{1}{2},\ \tfrac{1}{2}$ |
| 3 | $\tfrac{5}{12},\ \tfrac{8}{12},\ -\tfrac{1}{12}$ |
| 4 | $\tfrac{9}{24},\ \tfrac{19}{24},\ -\tfrac{5}{24},\ \tfrac{1}{24}$ |
| 5 | $\tfrac{251}{720},\ \tfrac{646}{720},\ -\tfrac{264}{720},\ \tfrac{106}{720},\ -\tfrac{19}{720}$ |

4. Store the step-start derivative in history. The next step evaluates the
   corrected endpoint as its new current point, completing PECE across steps.

So ABM costs **2 evaluations per step** after startup: current and predicted
endpoint. Its explicit correction avoids the nonlinear solve of
[implicit Adams-Moulton](multistep-implicit) and has conditional stability. The pairing of the
order-$p$ AB predictor with the order-$p$ AM corrector is what makes the PECE
scheme order $p$ (NOTES.md §3.6).

## SSP multistep (SSPRKMSVS32 / SSPRKMSVS43)

Shu's strong-stability-preserving multistep methods (`ssprkmsvs.py`, NOTES.md
§3.23), wave 3 of the scheme expansion. Unlike the Adams methods, the
steady-state step is a **convex combination of forward-Euler substeps across
past states**, so the `StepHistory` carries a state snapshot plus the
step-start derivative (the same `HistoryEntry.state` mechanism BDF uses):

| Scheme | `steps` | order | SSP $C$ | steady-state step |
|---|---|---|---|---|
| `SSPRKMSVS32` | 2 | 2 | 1/2 | $u^{n+1} = \tfrac{3}{4}(u^n + 2\,\mathrm{dt}\, f^n) + \tfrac{1}{4} u^{n-2}$ |
| `SSPRKMSVS43` | 3 | 3 | 1/3 | $u^{n+1} = \tfrac{16}{27}(u^n + 3\,\mathrm{dt}\, f^n) + \tfrac{11}{27}(u^{n-3} + \tfrac{12}{11}\,\mathrm{dt}\, f^{n-3})$ |

Cost: **1 evaluation per step** after the cold start (the past derivative is
reused from the history). Each bracketed term is one Euler substep, so the
step is a convex combination of stable maps — strongly stability preserving,
with $C$ set by the longest substep. Note `steps == order` here (2 / 3), not
`order - 1` as for the Adams family. Fixed-step only: there is no embedded
error estimate.

The tight SSP bounds ($C = 1/2$, $1/3$) are a *structural* property of the
steady-state convex combination. The measured per-step TVD sweep records
$1.5$ for both — the shared Dormand-Prince cold-start limit (the sweep does
not thread history, so the driver re-runs its starter every step), the same
cap the whole multistep family carries (NOTES.md §3.23).

## History and cold start

The schemes carry derivative history in a `StepHistory` (NOTES.md S2): a cold
call (fewer than $p-1$ entries) runs **Dormand-Prince 5(4)** for that step and
records the history — a high-order, order-faithful startup, so the run is never
silently wrong, just at the starter's cost until the history fills (NOTES.md
§3.7). `priorStep` reuse is not the mechanism here — the history *is* the
reuse: each step's stored $f^{n}$ is exactly what the next step's formula
needs.

The measured cold-start TV jump ($+1.333\times10^{-2}\cdot\mathrm{TV}_0$, gone
after five steps) is shared by every self-starting multistep scheme and is not
a property of the step map (NOTES.md §3.11).

## Stability and TVD

Zero-stable, with bounded real-axis stability regions that shrink with order
(AB2: $[-1, 0]$; AB3: $\approx[-0.54, 0]$; AB4: $\approx[-0.32, 0]$;
AB5: $\approx[-0.17, 0]$) — nothing here is A-stable. Per-step TVD measured to
CFL 1.5 (tested to CFL 5), shared by the whole multistep family, explicit and
implicit (NOTES.md §3.11).

## When to use

- **AB2-AB5**: the cheapest way to buy order 2-5 on a non-stiff problem whose
  RHS is expensive to evaluate (one evaluation per step).
- **ABM2-5**: two RHS calls after startup, an explicit correction and bounded
  stability region — the default choice when an implicit solve is not worth it.
- **SSPRKMSVS32/43**: order 2 / 3 at one evaluation per step *and* a
  strong-stability-preserving (convex-combination) step — the choice when the
  step size is set by a CFL/TVD limit and you want an SSP multistep method
  (fixed-step; the tight SSP bound is $C = 1/2$ / $1/3$).
- Not for stiff problems: the real-axis stability intervals shrink with order,
  and nothing here is A-stable.

## References

- Hairer, Nørsett, Wanner, *Solving ODEs I* (1993), Ch. 4 — Adams methods, PECE.
- Lambert, *Computational Methods in Ordinary Differential Equations* (1991).
- Shu, *Efficient time integration for parabolic and hyperbolic PDEs*,
  J. Comput. Phys. 169 (2001) 201-228 — SSPRKMSVS32/43.

**Tests:** `tests/test_multistep.py` (orders, the startup, the PECE cost
accounting, the SSPRKMSVS one-evaluation steady state and fixed-step contract,
stability).
