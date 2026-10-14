# Phase 8 — algorithm switching with owned child memory

AlgorithmSwitch chooses a complete method per trial, retains active-child
history and cold-starts when the selected method changes. It is outside the
142-method fixed catalog. Supported children are registered methods and the
Collocation, Extrapolation, Nystrom, StabilizedRK, GeneralizedAlpha, VariableAdams,
NDF, AdaptiveNDF and SplitScheme factories. The selector receives an owned state
probe and last accepted zero-based index; policy_id identifies its semantics.

Primary source: OrdinaryDiffEq 7.8.1 tree
bae9b8e5f1d00e51c554648e5094e215164a19f8, Core 4.15.1 algorithms.jl and
perform_step/composite_perform_step.jl. CompositeAlgorithm is algorithm choice,
not physical operator splitting. Its choice function receives the integrator,
uses Julia's one-based indices and initializes/transfers caches on transitions.
Our API is an independently designed control layer over already verified child
methods. It uses a pure state/previous-index policy, zero-based indices and cold
transitions. Source hashes and these differences are recorded separately; no
complete source CompositeAlgorithm cache/controller equivalence is claimed.
Numerical tests compare actual complete trajectories against manual child
selection and explicit history reset, rather than transcribing a new tableau.

The wrapper keeps the child's normal entries and nests its tensor/primitive
context in an envelope. Stateful children also store accepted value/layout/UID
guards; edits restart before use. This adds one owned flattened value guard
for stateful selections. Stateless selections avoid that copy. Constant-step
histories clear before a changed dt is consumed. Same-child trials retain the
numeric cache, and failed/rejected trials retain the prior accepted selection
and memory. Returning to an earlier method cold-starts again. Parameter/RHS
discontinuities require restart_history=True.

Accepted adaptive control unwraps the selected child's result/history and
delegates its policy with the runner's field norm, then retains the updated
child context. NDF's hold policy and AdaptiveNDF's order transitions therefore
survive the wrapper. The selected per-result estimator exponent controls retries
and acceptance proposals. Invalid child policy returns and exceptions roll back
before external commit. Child lifecycle, stage quadrature and solve diagnostics
remain intact; the wrapper adds no RHS or lifecycle call. Copied endpoint behavior
is that of the selected method. FSAL reuse is disabled even on unchanged choices.

All 22 focused tests pass: five manual fixed-trajectory/cache comparisons,
accepted-only alternating choices under rejection, child estimator exponents,
adaptive NDF order delegation, portable nested restart and policy identity,
constant-history dt resets, failed implicit transitions, invalid/mutating policy
rollback, explicit/foreign/capacity guards, field/layout/dtype/UID invalidation,
nonfinite bounded retries, backward nonautonomous stepping, invalid child policy
rollback, coupled quantities/lifecycle/copied fields and gradients through both
implicit one-step and NDF transitions. The pre-final switching/NDF/runner subset
has 287 passing cases; the final focused test adds UID/gradient transition coverage.

The 60-case known consumer has smooth time-varying diagonal stiffness, analytic
exp(-slow*t) solutions and a declared maximum rate. N=64/4096, peak rates
100/1,000/10,000 and tolerances 1e-4/1e-7 compare DP5/TR-BDF2 and DP5/NDF3
switching with all three individual methods. The policy has 30/10 hysteresis
thresholds and reads the declared bound; it does not infer stiffness from RHS
samples. JVP and exact diagonal forward preconditioning isolate driver behavior.
All cases complete. RHS work includes retries; GMRES counts cover accepted solves.

At N=4096, peak 10,000 and rtol=1e-7, DP5/NDF3 makes four switches, uses 665
RHS entries and has 9.48e-9 maximum error. DP5 alone uses 21,924 with 2.47e-9
error; fixed NDF3 uses 558 with 8.88e-9. DP5/TR-BDF2 uses 1,858 with 7.78e-8
error, close to TR-BDF2's 1,863/7.86e-8. This demonstrates removal of the explicit
stability restriction while exposing switching/cold-start overhead. It does not
show that switching beats a suitable always-implicit choice. At coarse tolerance
the policy may skip narrow low-stiffness windows and stay implicit, which is safe
but changes its selection counts. Equal tolerances are not equal-accuracy timing.
No nonnormal spatial/SPH scaling claim follows from a diagonal manufactured system.

The driver declares order=None: arbitrary frequent switching and low-order cold
starts can reduce trajectory order. Adaptive runs require estimates from every
candidate. Dense output/events and cross-method cache/FSAL transfer are excluded.
Portable checkpoints retain numeric/controller data, not executable selector code;
the caller reconstructs the same pure policy and parameters. Gradients follow the
selected path without differentiating discrete decisions. External policy state
is not automatically checkpointed or rolled back.

The final full CPU gate passes **7,873 tests, 639 skipped**, zero failures/errors,
in 324.74 seconds of pytest time (325.9 seconds with orchestration). Eight workers
use one numerical thread each; the stored fingerprint matches final source, tests
and shard/config files. The new test file belongs to the core CI category. The
installed 0.6.1 wheel passes all modules/exports, 135 fixed tensor drivers and
portable switched NDF continuation outside the checkout with isolated project
imports and shared numerical dependencies. All 47 published Python blocks execute,
with 13 labeled sketches. Catalog, repository links and production site build pass.
Editable metadata remains historical 0.5.0; source/wheel versions are 0.6.1.

Evidence: [primary source audit](evidence/algorithm-switch-source.json),
[consumer measurements](evidence/algorithm-switch-backend.json),
[full CPU gate](evidence/algorithm-switch-full.json),
[installed wheel](evidence/algorithm-switch-wheel.json).

Reproduce:

```bash
python -m pytest -q tests/test_algorithm_switch.py
python benchmarks/algorithm_switch_backend.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/algorithm-switch-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/algorithm-switch-wheel
```
