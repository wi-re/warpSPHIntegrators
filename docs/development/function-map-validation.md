# Phase 8 — direct state-map backend

FunctionMap applies an explicit StateMap once per fixed step. tensor_map adapts
(t_next,y_old,h)->y_next. This is a direct discrete transition, not an ODE rate
or an ODE step followed by a projection. It has order=None, no error estimate,
FSAL or dense output, and remains outside the fixed 142-method catalog.

Primary source: OrdinaryDiffEq 7.8.1 tree
bae9b8e5f1d00e51c554648e5094e215164a19f8, FunctionMap 2.2.1. The stdlib Julia
oracle executes the unchanged constant-cache step with minimal map/stats/default
sentinel scaffolding and removed broadcast/muladd annotations. Eight records
cover direct/scaled modes, affine/nonlinear maps and h=.05/.2. All main values
match within 2e-15. The callback receives old values at t+h, as the source step
body actually does. The scaled branch is expressed as an explicit map wrapper
y+h*rate(t_next,y_old); no overloaded rate/map flag is added. It is not generally
the same nonautonomous map as forward Euler evaluated at the old clock.
Re-running the promoted script reproduces the complete tracked JSON.

Source default-identity/limiter machinery is not claimed. StateMap deliberately
declares a complete integration-system return, while tensor_map checks tensor
shape/dtype/device. Returned systems are snapshotted before hooks. Initialize
sets the old input once. Preprocess runs on the map input and again on its output
to refresh copied fields; the map is not called twice. Postprocess sees the
mapped state/auxiliary values and finalize runs once. Hooks preserve numerical
map values, endpoint time and IDs. The initial integrated anchor remains intact.
Custom finalize opts into map_compatible and receives MapTransition, with an
update list containing None rather than invented derivative weights.

IntegrationResult.transition is a new optional trailing field. It carries the
owned input/evaluation-time/mapped-state record independently of quadrature.
Existing result construction remains compatible. StageResult records map
auxiliaries with update=None. Nonfinite transitions return before endpoint
hooks/finalize. The transactional runner preserves accepted state on failure,
and portable tensor checkpoints need no numeric map history or executable code.

All 23 focused tests pass: eight source cases, scalar/vector/batched recurrence
values and parameter/state gradients, portable continuation, backward clocks,
no-estimate/rate-adapter refusal, coupled rotation/thermal state and constant
energy budget, copied endpoint fields and lifecycle counters, one map call per
step, nonfinite rollback, layout/dtype/time/UID/bad-return guards, custom-hook
opt-in and projection refusal. The final map/switching/runner subset has 76
passing cases. The earlier gate used generic quadrature metadata; after API review
the map was given its distinct transition field and all final gates were rerun.

The understood consumer benchmark has 18 spiral/heat comparisons and six affine
recurrence cases at N=64/4096 and h=.2/.1/.05 to T=1. Damped rotations have a
known analytic map, generated heat and constant total quadratic energy. RK4/DP5
integrate the corresponding rates. The affine map has an independently evaluated
geometric-series reference including its time-dependent exp(-t_next) forcing.
Every run completes; affine error is at most 2.22e-15.

At N=4096,h=.1 the analytic spiral map uses ten map evaluations with 1.11e-15
state error, 9.71e-16 heat error and 2.22e-16 energy defect. RK4 uses 40 RHS
entries with 3.27e-2 state error; DP5 uses 61 with 4.35e-3 state/heat error on
this fixed grid with angular rates up to ten. The analytic map includes trig
and exponential work, so map-entry and derivative-entry counts are not equivalent
cost units. Exactness belongs to this supplied model-specific map, not a generic
exact ODE solver. The toy budget does not establish arbitrary SPH conservation.

The final full CPU gate passes **7,896 tests, 639 skipped**, zero failures/errors,
in 342.73 seconds of pytest time (344.1 seconds with orchestration). Eight workers
use one numerical thread each; the stored fingerprint matches final source, tests
and shard/config files. The new test file belongs to the core CI category. The
installed 0.6.1 wheel passes all modules/exports, 135 fixed tensor drivers, direct
map transition metadata and portable map continuation outside the checkout with
isolated project imports and shared numerical dependencies. All 48 published
Python blocks execute, with 13 labeled sketches. Catalog, repository links and
production site build pass. Editable metadata remains historical 0.5.0;
source/wheel versions are 0.6.1.

Evidence: [consumer measurements](evidence/function-map-backend.json),
[full CPU gate](evidence/function-map-full.json),
[installed wheel](evidence/function-map-wheel.json).

Reproduce:

```bash
python -m pytest -q tests/test_function_map.py
python benchmarks/function_map_backend.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/function-map-final-v2
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/function-map-wheel-v2
JULIA_DEPOT_PATH=/tmp/integrators-julia-oracle:$HOME/.julia julia --startup-file=no scripts/coefficients/wave13/function_map.jl "$HOME/.julia/packages/OrdinaryDiffEq/OnlvZ"
```
