# Phase 8 — bounded adaptive Adams target order

AdaptiveAdams supplies accepted target-order selection over variable-step PECE
orders 3–5. It starts at target three and builds on the existing VariableAdams
backend; maximum three/four/five are explicit factory configurations outside the
fixed 142-scheme registry. This closes a bounded adaptive-order backend increment,
not the complete VCABM order-1–12/FSAL/startup controller or other Nordsieck drivers.

## Independent source checks

The source pin is OrdinaryDiffEq 7.8.1 tree
bae9b8e5f1d00e51c554648e5094e215164a19f8, AdamsBashforthMoulton 2.2.0.
The stdlib Julia oracle executes the actual constant-cache divided-difference,
coefficient and history-expansion helpers. Mutable branches are unexecuted; a
broadcast annotation macro shim is provided only so those branches parse.
The oracle also extracts and executes the unchanged mature accepted-order branch
from the source constant-cache VCABM step, with minimal mutable policy scaffolding.
Complete source file hashes are recorded. No full SciML integration/controller
or native cache-equivalence claim follows from this bounded oracle.

Six basis-history records cover p=3/4/5 on uniform and irregular grids, using
distinct predicted and corrected endpoint derivative bases. All current/lower
and upper defect weights agree within 2.78e-16. The current/lower source defects
are AM(q+1)-AM(q) differences evaluated with the same predicted endpoint rate.
The upper source defect uses the actual corrected endpoint rate and one extra
history rate, with source gamma-star times the scaled highest divided difference.
The package evaluates this directly on declared derivative fields, avoiding
subtraction of large integrated-state offsets for neighboring defects.

There are 243 independent nonzero normalized-error policy records. Source lowers
when both p-2/p-1 errors are no greater than the current error, otherwise raises
when p+1 is smaller. Package selection matches after clamping the lower boundary
to three. Source order-two startup, orders six through twelve, FSAL transfer,
rejection restoration, source timestep policy and variable-order cache recurrence
are separate. Zero-error promotion is explicitly suppressed by package policy.

## Accepted trajectory and memory

DP5 startup retains VariableAdams's available-history comparison and per-result
q=2,3,..., then steady q=p+1. The runner consumes that actual exponent. Selection
waits for p+1 warm accepted steps, changes at most one target at a time, holds h
once on an order change and retains compatible derivative history. Direct calls
and fixed runs keep their target; adapt_step is the accepted adaptive hook.
Neighbor arithmetic adds no RHS or lifecycle call. Warm PECE still re-evaluates
start, predictor endpoint and corrected endpoint: three RHS entries, without
source FSAL savings.

The wrapper snapshots inputs/history, retains bounded derivative memory and
stores numeric/tensor-only target state. Values, layout, dtype/device, time and
UID guards reset before history is consumed. Excessive timestep ratios and
explicit RHS/parameter restarts also reset to target three. Foreign histories
cannot silently become accepted controller memory. Rejected trials and policy
exceptions leave accepted state/history unchanged and roll back external
transactions. AlgorithmSwitch delegates this accepted policy on unwrapped child
history. Portable TensorState restart reconstructs order/history without code
serialization. Physical splitting, dense output and event localization are not
added by this controller.

There are 278 focused tests; including fixed Adams and algorithm-switch coverage
there are 335 passes. These cover source weights/policy, startup exponents and
mature order bounds, nonlinear time forcing, exact parity with fixed-order PECE
under a fixed schedule and no extra RHS entries, copied endpoint fields,
full-state heat budgets and lifecycle counters, state/parameter gradients,
backward integration, zero rates at a large offset, trajectory edits, rejection,
policy rollback, foreign/capacity guards and portable continuation. Discrete
order/timestep decisions are not differentiated.

## Known consumers and cost limits

The 56-case benchmark uses float64 CPU vectors of N=64/4096, rates .1–3,
T=4 and local rtol=1e-5/1e-8 with atol=.01*rtol. It compares three maximum
orders, fixed-target PECE 3/4/5 and DP5. Reaction is nonlinear and time-forced,
with exact exp(-rate*t). Rotation has physical damping .2 and generated heat,
with known analytic state and constant mechanical-plus-thermal energy. RHS
entries include rejected attempts; all runs finish.

At N=4096 and rtol=1e-8, CPU model 'Intel(R) Core(TM) Ultra 9 285K':

| Problem/method | RHS entries | Final max error | Seconds on this run |
|---|---:|---:|---:|
| Reaction, adaptive target 3–5 | 336 | 6.4e-09 | 0.350 |
| Reaction, fixed target 3 | 951 | 4.43e-08 | 0.387 |
| Reaction, fixed target 5 | 338 | 6.4e-09 | 0.206 |
| Reaction, DP5 | 553 | 4.04e-11 | 0.076 |
| Rotation/heat, adaptive target 3–5 | 403 | 8.09e-06 | 0.484 |
| Rotation/heat, fixed target 5 | 405 | 8.23e-06 | 0.285 |
| Rotation/heat, DP5 | 868 | 1.47e-08 | 0.160 |

The adaptive reaction run accepts six target-three, five target-four and 87
target-five steps. Order selection avoids staying at an unnecessarily low target,
while approaching the work/error of a manually chosen target five. This is the
measured reason to retain the bounded order-controller baseline. It does not
beat fixed target five on cheap-RHS wall time; candidate arithmetic, snapshots
and value guards cost work. The source same-order estimator can substantially
understate global rotation/quantity error: the adaptive rotation/heat energy
defect is 9.56e-6 at local rtol=1e-8. At rtol=1e-5 its max error is 2.29e-3.
These are recorded limitations, not hidden by an equal-tolerance speed claim.
DP5 is considerably more accurate in these comparisons. Confirm observable
convergence before applying this nonstiff backend to another physical problem.
No stiff, nonnormal spatial or SPH scale claim follows from these vector toys.

## Reproduction and validation

The installed 0.6.1 wheel passes module/export checks, 135 fixed tensor drivers
and the adaptive-Adams target-five smoke outside the checkout, with isolated
project imports and shared numerical dependencies. Historical editable metadata
is still 0.5.0; source and wheel are 0.6.1. All 52 published Python blocks execute,
with 13 explicitly labeled sketches. The final CPU gate passes **8,417 tests,
639 skipped**, no failures/errors, in 323.47 seconds of pytest time / 324.7 seconds
with orchestration. Eight workers use one numerical thread each. The stored
source/test/config fingerprint matches the final backend batch. Catalog,
repository links/anchors and production site build pass.

Evidence: [source oracle](../../scripts/coefficients/wave10/adaptive_adams.json),
[consumer measurements](evidence/adaptive-adams-backend.json),
[installed wheel](evidence/adaptive-adams-wheel.json),
[full CPU gate](evidence/adaptive-adams-full.json).

```bash
JULIA_DEPOT_PATH=/tmp/integrators-julia-oracle:$HOME/.julia julia --startup-file=no scripts/coefficients/wave10/adaptive_adams.jl "$HOME/.julia/packages/OrdinaryDiffEq/OnlvZ"
python -m pytest -q tests/test_adaptive_adams.py tests/test_variable_adams.py tests/test_algorithm_switch.py
python benchmarks/adaptive_adams_backend.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/adaptive-adams-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/adaptive-adams-wheel
```
