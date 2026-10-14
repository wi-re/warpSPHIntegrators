# Phase 4 usability and packaging — 14 October 2026

Phase 4 adds ordinary tensor ODEs and catalog discovery while preserving the
tagged state/update protocol and twelve-field scheme tuple. The unused particle
flag was removed separately in `3a64a58`; its consumer audit remains linked from
the execution plan.

## API and contracts

- `TensorState(y0, t=...)` and `tensor_rhs(f, **parameters)` wrap `(t,y)`
  callbacks without lifecycle implementation. Scalar/vector/batch shape, dtype,
  device, state ownership and autograd connections are preserved. Split RHS
  capabilities, history, estimates and DP5 dense output use the existing drivers.
- All 130 compatible registrations advance the same batched decay problem,
  including multistep startup/steady-state, semilinear and implicit paths.
  Position/velocity methods reject TensorState before stepping.
- `filter_schemes` composes family/order/estimate/reuse/dense-output/state/history/
  separability/drift/RHS-capability criteria and returns actual registered records.
  Metadata derives history length from startup/history registrations rather than
  treating the legacy `steps` field as a universal order-minus-one formula.
- `getPreferredIntegrator` returns conventional explicit order-1–5 tensor records;
  `getPreferredScheme` preserves the historical raw-callable choices.
- Known solver overrides now reject invalid finite ranges, enums and integer
  counts before stage evaluation; standalone GMRES, stabilized-stage counts,
  Krylov controls and Newmark nonfinite parameters have matching guards.
  Valid RKC counts below a method's minimum retain their documented clamp.
  Custom solver extension keys and arbitrary full-protocol RHS kwargs remain
  supported. Tensor callbacks bind named parameters separately from driver options.

## Gradient and numerical qualifications

Explicit and implicit tensor state/parameter gradients match finite differences.
The adapter does not change frozen-operator differentiation in ROS3P/EXPRB32:
strict parameter checks exposed the existing operator-parameter qualification,
now documented in the tensor guide. RKC non-autonomous stage-time approximation,
fixed-step history/startup and geometric force assumptions remain documented.
No coefficients, registry identifiers or existing driver algorithms were replaced.
Device preservation is checked on CPU; this gate does not establish new GPU or
low-precision numerical coverage. Caller-owned stepping/rejection remains Phase 5.

## Wheel isolation and environments

The `dev` extra declares test/build/release tooling; Phase 3 already declared
example/notebook and plotting dependencies. `validate_wheel.py` builds a wheel,
creates a separate venv, copies only its smoke/catalog inputs outside the checkout,
and runs isolated Python (`-I`). Every imported project module must reside in
that venv; installed metadata/version, exports, catalog and `py.typed` are checked.
The gate advances all compatible tensor drivers and a full-protocol oscillator.

Local validation uses `--reuse-dependencies`: the existing numerical dependencies
are shared, while the project comes only from the installed wheel. This catches
the old base environment's stale 0.5.0 distribution metadata instead of allowing
it to masquerade as the new 0.6.1 wheel. CI has a separate wheel job with fresh
dependency installation and CPU Torch; its remote result is not claimed here.
Minimum numerical dependency versions are not inferred from this one environment.

## Validation

The focused final checks pass: **532 passed, 4 skipped**, 18 warnings, **8.11s**.
The first full sweep had one failure in an existing error-message assertion;
the original GMRES diagnostic was restored, and that regression check passes.
The final full CPU gate passes: **6,907 passed, 609 skipped**, 175 warnings,
**272.92 seconds** with eight work-stealing processes and one numerical-library
thread per worker. This retains the 6,742 previous passing cases and adds 165.
Environment: Python 3.13.14, Torch 2.13.0+cu130 on CPU, NumPy 2.5.2, pytest 9.1.1.
The final wheel gate passes from outside the checkout in isolated Python.
All 33 published runnable blocks execute; 13 caller-defined sketches remain labeled.
Generated catalog, repository links and the strict production site build pass;
Python package/utility/example/benchmark files compile.

Reproduction:

```bash
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase4/final-full-2
python scripts/validate_repository.py
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
python scripts/validate_wheel.py --reuse-dependencies
# CI / fresh numerical dependency installation:
python scripts/validate_wheel.py --cpu-torch
cd docs
npm run build
```

The published ordinary-ODE example and tensor selection/split examples run
verbatim. The strict Docusaurus build includes the new tensor/options routes and
generated structural catalog columns. Raw logs remain in `.tmp/phase4`; compact
reports are retained: [full CPU gate](evidence/phase4-full.json) and
[installed wheel](evidence/phase4-wheel.json).
