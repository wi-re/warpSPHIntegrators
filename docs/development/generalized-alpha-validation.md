# Phase 8 — generalized-alpha driver, 14 October 2026

The known two-mass consumer now evaluates a parameterized package driver:
`GeneralizedAlpha(rho_inf=.5)`. It uses the classical second-order old-state
weights and an augmented physical-state/algorithmic-acceleration nonlinear solve.
Implementation commit: `49b99fe`. The full gate records its starting base commit
and separately fingerprints the tested working-tree implementation.
It is separate from the 139 fixed catalog entries; waves 10–16 and larger
structural/SPH scaling remain independently gated. The original
[prototype/source audit](structural-demo-validation.md) remains the independent
reference and historical validation record.

## State and force contract

RHS supplies x'=v and explicit acceleration for constant-mass mechanics. The
force is evaluated at `(1-alpha_f)*endpoint + alpha_f*initial` and the
corresponding weighted time. Algorithmic acceleration satisfies
`(1-alpha_m)*a1 + alpha_m*a0 = acceleration(weighted_state)`; Newmark kinematics
advance position/velocity. Nonlinear and velocity/time-dependent force is supported.
Other integrated quantities use a coupled endpoint trapezoid and their weighted
values enter the acceleration equation. There is no adaptive estimator, dense
output, event localization, variable-mass residual or DAE constraint support.
At rho=1 equivalence with average-acceleration Newmark is a **linear-force** anchor.

The factory initializes physical acceleration once when history is empty. Every
accepted direct result returns StepHistory with GeneralizedAlphaMemory in the
latest entry's state slot. Direct callers must thread accepted history. The
runner owns this automatically and preserves the tensor graph in in-memory
checkpoints. Time, rho, integrated layout and particle IDs are checked before
consuming memory. State and history must stay paired; manually editing state
requires explicit reinitialization. Custom second-order systems still need
application-owned portable serialization.

Single-step input state/history are cloned; output memory does not alias caller
tensors. Accepted algorithmic acceleration survives shortened steps. A failed
solve/nonfinite endpoint has no finalization or new memory entry; the runner
rolls back owned lifecycle data and external transaction hooks as usual.
Initialization may establish the initial state but must preserve integrated
values when continuing memory. Finalize must preserve integrated values, so a
consumer cannot silently leave stale acceleration after an endpoint projection.
Copied fields are refreshed from a final physical endpoint RHS evaluation.
Finalize receives paired initial/endpoint auxiliary values and updates, with
trapezoidal weights .5/.5 for the extra quantities. Those weights do not replace
the separate weighted acceleration equation governing mechanics.

The runner also clips its last step using a bound on accumulated time rounding.
Previously, 120 additions of h=.1 could leave a near-zero 121st step at T=12.
That extra step hardly moved q/v but changed algorithmic acceleration. A long-grid
regression now checks exactly 120 steps and parity with a manually threaded run.
Constant-dt reset semantics for existing multistep drivers are unchanged;
SchemeMetadata.history_requires_constant_dt defaults true and the factory sets false.

## Consumer and cost gates

On the original 1/40 rad/s two-mass problem, all rho=0/.5/.8/1 behavior runs
complete 120 steps to T=12 at h=.1 with tolerance-converged solves. Displacement
matches the independent dense linear reference within 7e-14; velocity within
7e-13 and final algorithmic acceleration within 1.4e-11. At rho=.5 slow energy
retention is 99.947% and fast energy about 1.53e-20 of its initial value. Filtered
energies near numerical floors are suppression diagnostics, not precise physical
signals. The resolved 1/3 rad/s refinement reproduces orders 1.987/1.998/2.000
for rho=0/.5/1 without resetting acceleration.

Work is counted by an RHS wrapper, including Krylov evaluations and endpoint
evaluations, rather than comparing differently scoped solver counters. At rho=.5
the driver uses 1,680 RHS calls and 480 GMRES iterations versus Newmark's 840/360,
with identical physical force, h, horizon and JVP/tolerance options. The augmented
map performs two RHS calls per residual/Krylov map; reported RHS diagnostics
include residual-map calls plus startup/final endpoint work, and the corresponding
full JFNK cost is `reported_rhs + 2*gmres_iterations`. Timings are retained in
evidence but this tiny consumer does not establish large sparse/SPH performance.
The benefit is controllable high-frequency damping, with an explicit solve cost.

Focused tests cover dense parity, second-order refinement, selective damping,
shortened steps, restart/parameter mismatch, caller/history ownership, rejected
solve rollback/resumption, a nonlinear forced velocity/quantity-coupled exact
solution, initial-state and force-parameter gradients across restart, parameter
guards, unsupported runner modes, arbitrary field tags, copied endpoint values,
the full finalize signature and paired quantity weights,
particle-ID changes, unsafe finalize rejection, nonfinite endpoints and default/
fixed-depth solver paths. The original prototype's spectral-radius and source
negative controls remain separate checks.

## Reproduction

The final full CPU gate passes **7,133 tests, 621 skipped**, with zero failures
or errors, in 312.05 seconds of pytest time (313.2 seconds for the gate process).
Eight work-stealing workers use one numerical thread each. The stored
source/tests/config fingerprint matches the final implementation. All 24 focused
driver cases pass; the combined driver/catalog check passes 315 cases, 4 skipped.

The installed-wheel gate passes outside the checkout: all modules/exports,
132 tensor drivers, existing advanced consumers and generalized-alpha restart.
Dependencies are shared from the numerical environment; project imports are
isolated to the installed 0.6.1 wheel. The editable full-test environment retains
historical installed metadata 0.5.0 while source/package version remains 0.6.1.
All 39 published Python blocks execute; 13 sketches remain labeled. Generated
catalog, repository links and documentation-site build pass. No new runtime
dependency or fixed catalog entry is added.

```bash
python benchmarks/generalized_alpha.py
python -m pytest -q tests/test_generalized_alpha.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/alpha-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/alpha-wheel
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
python scripts/validate_repository.py
cd docs
npm run build
```

The benchmark records float64 CPU execution with one numerical thread. The wheel
gate installs the project outside the checkout and checks factory continuation
in addition to all modules/exports and the existing numerical consumers.

Evidence: [consumer/work measurements](evidence/generalized-alpha.json),
[full CPU gate](evidence/generalized-alpha-full.json),
[installed wheel gate](evidence/generalized-alpha-wheel.json).
