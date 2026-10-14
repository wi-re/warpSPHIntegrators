# Phase 8 — variable-step Adams backend, 14 October 2026

`VariableAdams(order, kind)` supplies six fixed-order variable-step AB/ABM
configurations at orders 3–5, supporting fixed/adaptive runs and restart. These
factories are outside the fixed catalog. A classical fixed-step AB5/AM5 PECE
record is also registered as `abm5` / Adams-Bashforth-Moulton 5 (PECE), integer
139. Earlier identifiers remain unchanged; the fixed catalog now has 140 records.
Backend completion uses standalone problems rather than waiting for frontend
integration.

## Source and numerical contracts

The pinned OrdinaryDiffEq 7.8.1 monorepo tree is checked by the Julia coefficient
oracle, with AdamsBashforthMoulton 2.2.0. It evaluates the source constant-cache
divided-difference helpers on vector-basis histories, on uniform and irregular
grids at orders 3/4/5. Mutable broadcast branches are omitted. It does not run
the upstream controller or complete solver. Normalized Lagrange integration
matches the source's predictor, corrector and difference weights.

AB uses a p-point predictor and an order-(p-1) comparison, with exponent p.
Variable ABM uses the source order-(p-1) predictor and order-p corrector, with
the source same-order difference estimate at exponent p+1. The public error sign
is main minus comparison, opposite the source raw ABM difference sign. The
comparison is not advertised as a higher-order companion. Local difference
control does not guarantee an equal global error tolerance.

DP5 provides a high-order startup state. Its estimate compares that state with
the Adams predictor available from the currently populated history; exponents
ramp 2,3,... before reaching the steady method. `IntegrationResult.estimate_order`
lets the runner use this actual exponent on accepted and rejected trials. The
controller respects the method's maximum step growth. This avoids an optimistic
high-order starter repeatedly choosing a step too large for the warm low-order
formula, rejecting it, discarding history and starting again.

The registered fixed ABM5 uses classical AB5 predictor and AM5 corrector weights,
with no embedded estimate, and a DP5 starter. It is an explicit repository
startup/cache policy, not a full upstream ABM54 solver-equivalence claim.
The oracle verifies its weights; a separately assembled nonlinear one-step map
and two-call warm RHS count match the implementation. Existing constant AB/ABM/AM
drivers now invalidate changed-dt history **before** formula consumption, including
direct calls outside the runner.

## History, lifecycle, gradients and restart

Variable methods opt into StepHistory.pushed(preserve_dt=True, context=...).
The context holds method/layout/particle-ID metadata using tensors and primitives;
it is explicitly passed on each append so a different stepper cannot accidentally
preserve another method's context. Histories carry derivatives at accepted start
points. Incompatible context, time/layout discontinuity, particle-ID changes or
step ratios outside the declared range reset before consuming derivatives.
Callers explicitly restart for changed RHS parameters or derivative discontinuities.
Warm initialize hooks must preserve integrated values and particle identity.

The driver clones input state/history, computes trial error before acceptance,
and returns unchanged accepted history on a nonfinite trial. The runner owns
rejection rollback and external transactions. Extra integrated fields use the
same full-state Adams weights. Copied values are refreshed at the corrected
physical endpoint, so variable AB/ABM use two/three RHS calls after startup;
the fixed legacy-style ABM5 path uses two and retains the family's last-predictor
copied-field convention. The extra variable-driver endpoint call is explicit
work, not a source FSAL saving claim.

Portable TensorState checkpoints add optional primitive/tensor history context
to the existing weights-only payload. Existing files lacking the field load with
context=None. Nonprimitive context objects are rejected. In-memory restart
preserves gradients, including initial state and RHS parameters across history.
Custom systems retain application-owned portable serialization.

Tests cover source weights, nonlinear nonautonomous alternating-grid order,
adaptive rejection and refinement, startup exponent/rate changes, ratio and
explicit resets, portable fixed/adaptive restart, input/history/gradient
ownership, foreign/discontinuous histories, copied endpoint fields and particle
IDs, primitive serialization, fixed ABM5 map/cost and changed-dt reset of the
existing constant families. Cross-family gates cover the new fixed registration.

## Reproduction

The final full CPU gate passes **7,256 tests, 626 skipped**, zero failures/errors,
in 330.05 seconds of pytest time (331.3 seconds with orchestration). Eight
work-stealing workers use one numerical thread each; the stored fingerprint
matches the final implementation. All 35 focused Adams cases pass, along with
the registry/cross-family coverage for fixed ABM5. Explicit family-size guards
now pin the ten-member explicit multistep family and total 140 registrations.
All 41 published Python blocks execute, with 13 labeled sketches. Catalog,
repository links and production documentation build pass. The installed 0.6.1
wheel passes outside the checkout, including 133 tensor drivers and portable
variable-Adams continuation. Numerical dependencies are shared but project
imports are isolated. Editable test metadata remains historical 0.5.0 while
the source/package version remains 0.6.1.

The known vector consumer has 36 fixed/alternating-grid cases, including 16,384
components, and 18 adaptive tolerance runs. On the 80-step alternating grid,
variable AB5/ABM5 use 180/256 full RHS calls with four startup steps, versus 560
calls for fixed-coefficient Adams restarting on every size change. Their maximum
absolute errors are 2.06e-8/5.52e-9; DP5 is more accurate at 6.03e-11 and uses
560 calls. This records an accuracy/work trade, not a blanket speed advantage.
At requested local rtol=1e-8 the ABM3/4/5 adaptive endpoint errors are about
3.48e-7/1.85e-7/1.21e-7, explicitly illustrating the local/global distinction.

```bash
julia --startup-file=no scripts/coefficients/wave10/variable_adams.jl /path/to/OrdinaryDiffEq/OnlvZ
python benchmarks/variable_adams.py
python -m pytest -q tests/test_variable_adams.py tests/test_multistep.py tests/test_run.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/adams-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/adams-wheel
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
python scripts/validate_repository.py
cd docs
npm run build
```

Evidence: [pinned coefficient oracle](../../scripts/coefficients/wave10/variable_adams.json),
[known vector/adaptive consumers](evidence/variable-adams.json),
[full CPU gate](evidence/variable-adams-full.json),
[installed wheel](evidence/variable-adams-wheel.json).
