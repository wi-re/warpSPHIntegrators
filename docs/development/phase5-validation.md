# Phase 5 run orchestration — 14 October 2026

Phase 5 adds an optional run API over the existing 137 registrations. Numerical
coefficients and single-step calling conventions remain unchanged. The first
increment, `0f520a0`, adds fixed/adaptive acceptance, step status and sampling;
`8045506` adds continuous events. The final increment adds portable checkpoint
persistence, output/restart tests, runnable docs and installed-wheel coverage.

## Run and failure contracts

- `solve` owns fixed/adaptive stepping, positive step magnitudes in either time
  direction, exact requested endpoints, limits and accepted-step callbacks.
  It resets multistep history before a changed step consumes it, including the
  clipped final step; this invokes startup and retains its accuracy qualification.
  Only equal-step FSAL reuse is enabled automatically.
- Native implicit drivers retain `SolveResult.converged` in diagnostics. Custom
  solvers can return failure without evaluating their map or providing diagnostics;
  DIRK/ARK/coupled driver caches now handle this case. `step_status` distinguishes
  a finite candidate, actual tolerance convergence, fixed scheduled completion,
  failed solves and Newton stagnation. Direct calls keep candidate-return behavior.
  The run API rejects precision-floor stagnation unless deliberately opted into.
- Adaptive acceptance uses field-specific weighted RMS scaling against both old
  and candidate magnitudes, catalog estimate exponents, nonfinite checks and
  bounded retries. History-dependent methods are refused in adaptive mode.
- Each trial clones the owned state graph and history, retaining tensor gradients.
  Rejected lifecycle/buffer changes are discarded. External closure variables,
  I/O and device services require application `begin`/`commit`/`rollback` hooks;
  the runner cannot undo these implicitly. Snapshots, callbacks and outputs have
  documented ownership and resource-type requirements.

## Sampling, events and restart qualifications

Sampling never changes the grid. Fixed-grid outputs work on other methods;
interior interpolation and continuous sign-change events require DP5's published
quartic. Interpolation advances integrated fields without refreshing auxiliary
physical buffers. Event direction is defined in increasing physical time, also
for backward runs. Initial/endpoint roots are not repeatedly reported. Terminal
crossings roll back the full trial and recompute to the root, including lifecycle
and history. Tangencies/multiple crossings within one step can be missed.

In-memory checkpoints preserve accepted state, history, FSAL stage and controller
configuration, with gradient connections. Callback stops on existing accepted
endpoints restart bit-for-bit identically to uninterrupted fixed/adaptive runs.
Portable files support plain TensorState and tensor/empty-auxiliary history only;
Torch weights-only loading avoids arbitrary object deserialization and saving
truncates gradients. Full protocol/SPH persistence and external restart context
remain application-owned. RHS/options/events/hooks are supplied again on restart.

The [run guide](../docs/runs.md) and [standalone example](../../examples/run_ode.py)
explain the controls and these qualifications. No new GPU, low-precision or native
Warp execution claim is established by this CPU gate.

## Validation

Focused new run/output tests: **31 passed**. They cover caller ownership, callback
isolation, multistep clipping, FSAL invalidation, adaptive/external/lifecycle
rollback, nonfinite limits, custom implicit-solver failures, stagnation policy,
field tolerances, trajectory gradients, dense grid invariance, forward/backward
event direction, terminal recomputation and in-memory/disk restart parity.

The final full CPU gate passes: **6,938 passed, 609 skipped**, **267.93 seconds**
with eight work-stealing worker processes and one numerical-library thread each.
This retains the 6,907 Phase 4 passing cases and adds 31. Environment: Python
3.13.14, Torch 2.13.0+cu130 on CPU, NumPy 2.5.2, pytest 9.1.1.

The isolated installed 0.6.1 wheel passes all module/export/type-marker checks,
all 130 tensor registrations, the advanced oscillator protocol, adaptive samples,
terminal events and portable restart. Local wheel validation shares numerical
dependencies while installing project code only in its isolated venv outside the
checkout. Fresh dependency installation remains a CI job; its remote outcome is
not claimed. The base conda distribution metadata still reports 0.5.0 in the full
gate; wheel metadata verifies the current package version independently.

All **34** published runnable Python blocks execute; **13** caller-defined
sketches remain labeled. Catalog generation, local Markdown/notebook paths and
strict production site build pass. The full test gate verifies the unchanged CI
category partition during collection. Raw logs are under `.tmp/phase5`; compact
reports are retained: [full CPU gate](evidence/phase5-full.json) and
[installed wheel](evidence/phase5-wheel.json).

Reproduction:

```bash
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase5/final-full
python scripts/validate_repository.py
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase5/wheel
# Fresh dependencies / CI:
python scripts/validate_wheel.py --cpu-torch
cd docs
npm run build
```
