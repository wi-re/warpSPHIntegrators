# Phase 8 — adaptive NDF target order

`AdaptiveNDF(max_order=5)` adds accepted target-order selection over the NDF
difference backend. It is outside the fixed 142-method catalog. Actual target
orders range from one through max_order; the factory's order is None. Each
result carries the estimator exponent and history records both the last target
order used and the next order. The initial target is one. Fixed runs/direct
steps retain the current order; automatic selection runs through solve's
adaptive mode.

Candidate mathematics comes from the pinned OrdinaryDiffEq 7.8.1 tree
bae9b8e5f1d00e51c554648e5094e215164a19f8, BDF 2.4.5 controllers.jl. A stdlib
Julia oracle executes the unchanged QNDF accepted controller for 135 mature
cases: five orders, three main errors and nine combinations of lower/upper
errors. The standalone ndf_order_proposals reproduces selected order and
candidate timestep after applying the source steady deadband. Re-running the
promoted source script reproduces the complete tracked JSON. The warm kernel
and defect constants have their separate thirty-case source oracle.

The public driver deliberately has a package policy rather than claiming full
QNDF controller equivalence. It waits for p+2 equal accepted steps before
comparing neighboring defects, changes at most one order at a time and holds
h after an order change. Main/lower/upper candidate safety factors are
1.2/1.3/1.4. It uses the NDF 1–1.5 growth deadband and declared ratio cap;
zero main defect does not raise order. Retry shrinking remains transactional
runner behavior, and rapid history resampling can safely restart through the
order-preserving implicit RK starter. It does not reproduce upstream QNDF's
rejection-order controller or derivative-discontinuity detector. Changed RHS
parameters require restart_history=True.

The runner supplies error_norm and norm_estimate(error_state) using the same
old/new, per-field acceptance scales. Order proposals update owned trial memory
before external commit. Invalid returns or policy exceptions roll back external
transactions; dt_max and method ratio caps apply after the policy. Existing
three-argument next_step_size hooks continue to work. Normalized controller
values explicitly detach before conversion to Python scalars: gradients follow
the numerical path without differentiating discrete controller decisions.

All 159 focused cases pass, covering the independent source candidates, order
ceilings 1–5, mature promotion and reduction, zero defects, nonlinear forced
tolerance refinement, backward adaptation, capacity and foreign-history guards,
explicit restart, protected maximum dt, policy failure/invalid-return rollback,
and portable dynamic-order continuation. Coupled mechanics/quantity fields use
different relative tolerances and match a fine implicit reference. Lifecycle
counters match accepted steps and copied fields match physical endpoint values.
Parameter gradients through order transitions agree with finite differences.
The pre-final combined NDF/runner subset had 261 passing cases; the focused
suite then added five backward/invalid-candidate cases.

The consumer benchmark has 42 cases: N=64/4096, nonlinear manufactured reaction
and stiff linear decay, three tolerances and AdaptiveNDF/fixed NDF5/KenCarp5
baselines, with DP5 on the reaction problem. Analytic references are exp(-rate*t).
All cases complete, actual target-order counts show promotion through 1–5,
and reported warm counts show that measurements exercise the multistep kernel.
RHS work includes rejected attempts; GMRES counters in this report cover accepted
solves. JVP and exact diagonal forward preconditioning isolate driver behavior.

At N=4096 and rtol=1e-6, nonlinear reaction uses 764 RHS entries with maximum
error 1.98e-8, versus fixed NDF5's 776/2.44e-8, KenCarp5's 1,051/9.91e-8 and
DP5's 392/6.32e-9. Explicit DP5 is the more economical choice for this mild
problem. Stiff decay uses 1,758 entries with error 1.98e-8, versus fixed
NDF5's 1,570/5.15e-8 and KenCarp5's 1,537/1.33e-7. At rtol=1e-8 AdaptiveNDF
uses 794 accepted steps with 792 warm steps on stiff decay. These are measured
accuracy/work tradeoffs, not equal-accuracy speed claims or evidence that
automatic order selection universally improves on a chosen fixed target.
Local defect tolerances do not bound final global error; diagonal toy scaling
does not establish nonnormal spatial/SPH performance.

The final full CPU gate passes **7,789 tests, 639 skipped**, zero failures/errors,
in 314.01 seconds of pytest time (315.2 seconds with orchestration). Eight workers
use one numerical thread each; its fingerprint matches final source, tests and
shard/config files. The installed 0.6.1 wheel passes all modules/exports, 135 fixed
tensor drivers and adaptive-order portable continuation outside the checkout
with isolated project imports and shared numerical dependencies. All 45 published
Python blocks execute, with 13 labeled sketches. Catalog, repository links and
production site build pass. Editable test metadata remains historical 0.5.0;
source/wheel versions are 0.6.1.

Evidence: [consumer measurements](evidence/adaptive-ndf-backend.json),
[full CPU gate](evidence/adaptive-ndf-full.json),
[installed wheel](evidence/adaptive-ndf-wheel.json).

Reproduce:

```bash
python -m pytest -q tests/test_adaptive_ndf.py
python benchmarks/adaptive_ndf_backend.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/adaptive-ndf-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/adaptive-ndf-wheel
JULIA_DEPOT_PATH=/tmp/integrators-julia-oracle:$HOME/.julia julia --startup-file=no scripts/coefficients/wave10/ndf_controller.jl "$HOME/.julia/packages/OrdinaryDiffEq/OnlvZ"
```
