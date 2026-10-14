# Phase 8 — first wave-9 batch, 14 October 2026

The first reviewed expansion batch registers **Gauss-Legendre 3, order 6**, and
**Radau IIA s=3, order 5**. The catalog grows from 137 to 139, with the same
11 families and 47 estimate emitters. Existing enum values and legacy two-stage
coefficients remain unchanged. Higher-stage wave-9 methods and waves 10–16 are
not declared complete: each needs its own source, driver, cost and property gates.

## Source and companions

The Julia oracle verifies the entire cached OrdinaryDiffEq 7.8.1 monorepo tree
`bae9b8e5f1d00e51c554648e5094e215164a19f8` before evaluating source. Bundled
`lib/OrdinaryDiffEqFIRK` is version 2.8.1; it is not a separately installed/pinned
FIRK dependency in the historical Julia manifest. The recorded tableau source
SHA-256 is `124e136a71000fb276cb72204f52f8bd78fb2e890fdeb14205b5a1a77b5b3e51`.
The oracle uses Julia stdlibs: Golub-Welsch quadrature evaluates the unchanged
Gauss constructor; the Radau inverse-stage transform independently reconstructs
A. A dense analytic-Jacobian Newton solver checks nonlinear autonomous and
time-dependent one-step states/errors against the Python matrix-free driver.

Gauss's same-node companion has order two, weights `(-5/6,8/3,-5/6)` and
estimate exponent three. Its zero null-stage weight adds no RHS call. The driver
now removes the unused null weight before applying same-node weights.

The Radau source's `e` weights act on stage increments and then undergo a
shifted-Jacobian filter. They are not embedded solution weights. This package
derives the unsmoothed solution companion `(-1,b-e@A)`, of order three with
estimate exponent four and one extra null RHS call. Its order conditions,
nonlinear branch accuracy, estimate rates, gradients and adaptive consumer are
verified separately. Upstream smoothing/re-estimation/controller behavior is
not reproduced; the propagated solution remains the order-five collocation map.

Both methods retain first-stage-reuse refusal, optional warm guesses, copied-field
lifecycle and implicit-function gradients. Adaptive control reads the actual
estimate exponent. A conservative low-order estimate does not make adaptive
Gauss automatically more efficient than fixed steps.

## Accuracy, geometry and useful work

CPU float64, one numerical thread; Python 3.13.14, Torch 2.13.0+cu130,
Intel Core Ultra 9 285K. The default FD/Newton/GMRES floor obscures order-five/six
errors near 1e-10. Reference convergence and GL3 energy tests use JVP,
linear tolerance 1e-13 and scaled Newton threshold 1e-9. No registered order
assertion or production default is relaxed. Gauss's forced problem uses a coarser
grid because its error is already 4e-12 at h=0.1. Unsupported precision-floor
trajectories are not used to claim completed tolerance-converged runs.

| Method | Oscillator | Forced | Kepler | Damped |
|---|---:|---:|---:|---:|
| Gauss-Legendre 3 | 6.002 | 6.002 | 6.000 | 5.998 |
| Radau IIA s=3 | 5.001 | 5.000 | 4.999 | 4.996 |

Gauss satisfies the collocation symplectic identity and time-reversal identities;
strict-solve phase-area and long-horizon energy tests verify the implementation.
Radau is stiffly accurate, L-stable and dissipative. Existing nonlinear stiff,
reuse, lifecycle, field, adaptive and gradient suites include the registrations.
Fourteen focused source/companion/consumer cases provide independent checks.

The measured oscillator consumer (`q'=v,v'=-4q`, endpoint 2) shows a concrete
accuracy/work benefit, counting actual RHS calls including Krylov probes and
companions:

| Method/step | Endpoint L1 error | RHS calls |
|---|---:|---:|
| Gauss s=2, h=0.05 | 1.15e-6 | 600 |
| Gauss s=3, h=0.2 | 3.33e-7 | 270 |
| Radau s=2, h=0.05 | 1.21e-4 | 600 |
| Radau s=3, h=0.2 | 1.24e-5 | 280 |

This is a known-solution niche, not a universal speedup. The additional cubic
decay consumer `y'=-3y^3, y(0)=1`, exact `1/sqrt(1+6t)`, records nonlinear
refinement, work and timing. Its coarse h=0.2 points are pre-asymptotic; the
registry-wide Kepler gate separately verifies the full nonlinear orders.

## Unchanged spatial bar on registered drivers

48 fresh-process configurations retest the selected shifted spatial ILU and
full sparse LU reference for s=2/3, reaction/diffusion and N=64/256/1024.
These are the actual registered methods, including companion RHS work. Problem,
h=0.05, three steps, two repetitions, restart=30, GMRES budget=120, Newton
budget=6 and tolerances match Phase 6. The original bar still requires zero
tolerance failures, <=120 Krylov iterations per step, <=3x corresponding s=2
time/RHS work, <=256 MiB additional native RSS, and <=1e-7 direct-reference
agreement through N>=1024. Both selected ILU and reference LU pass every case.
Unselected ILU variants from the prior follow-up are not retested or relabeled.

At N=1024 diffusion, shifted ILU gives:

| Method | Krylov/step | RHS/step | Median seconds | Reference difference |
|---|---:|---:|---:|---:|
| Gauss s=3 | 10 | 39 | 0.0272 | 2.72e-8 |
| Radau s=3 | 10 | 31 | 0.0198 | 2.40e-9 |

Maximum additional native peak RSS is 155.0 MiB. This clears the first s=3
registration batch, not general unstructured/multifield SPH assembly or higher
stage counts. Those retain their separate gates.

## Wave 10/11 source audit and next batches

The reproducible declaration inventory records nine package versions, algorithm
names, aliases, source order expressions and hashes. It handles macro-generated
types rather than relying on explicit-struct grep. Main corrections:

- Fixed-step AB/ABM support non-autonomous RHS; variable coefficients concern
  timestep/history ratios. Pinned ABM54 is a fifth-order fixed predictor/corrector,
  not an adaptive 5(4) pair. MEBDF2 is modified **extended** BDF, and QNDF uses
  quasi-constant timesteps rather than meaning quasi-Newton.
- AMF is an **Approximate Matrix Factorization** Rosenbrock-W wrapper with split
  operators, not Adams–Moulton. It needs a factorization/operator contract.
- ExponentialRK contains ETDRK/EPIRK/Exprb members hidden behind macros.
  StabilizedRK includes ESERK and TSRKC recurrences as well as ROCK/RKC/RKL/RKG.
  Stage selection, recurrence/history and Krylov costs need driver-specific work.
- Tsit5DA is a HybridExplicitImplicitRK instance for mass-matrix DAEs. Generalized
  alpha needs weighted second-order residual/time and parameter semantics; it is
  not a plain RK tableau on the current Newmark driver.
- CompositeAlgorithm is algorithm switching, not Phase 7 physical composition.
  Map/extrapolation/controller transitions need their own run/restart design.

Generalized-alpha is a source-confirmed candidate for a structural toy consumer;
no implementation or advantage over Newmark is claimed by this inventory. General
RKN, matrix/multirate/Taylor interfaces and low-storage screening remain separate
batches. Low-storage names alone do not prove map equivalence or an allocation gain.

## Validation

The final full CPU gate passes **7,096 tests, 621 skipped**, with no failures
or errors, in 316.39 seconds (317.6 seconds including orchestration). Eight
work-stealing workers each use one numerical thread. The source/tests/config
fingerprint matches the final implementation; no source changes follow the gate.
Existing benchmark case callers also pass with the new registered-driver flag
defaulting to false when omitted.

All 37 published Python blocks execute; 13 caller-defined sketches remain labeled.
Catalog generation, repository links/disposition coverage and the production
documentation site pass. The independently built installed wheel passes every
module/export and all 139 registrations, including 132 tensor drivers.
The wheel includes actual runs of both new tensor drivers outside the checkout.
Local wheel validation reuses numerical dependencies; CI retains fresh installs.
Source version and isolated wheel are 0.6.1; historical editable installation
metadata still reads 0.5.0 and is recorded explicitly in the full gate.

```bash
julia --startup-file=no scripts/coefficients/wave9/oracle.jl /path/to/OrdinaryDiffEq/OnlvZ
python scripts/coefficients/phase8_inventory.py /path/to/OrdinaryDiffEq/OnlvZ --output /tmp/phase8-inventory.json
python benchmarks/firk_registration.py --output-dir .tmp/phase8/firk-accuracy
python benchmarks/implicit_scale.py --spatial --registered --selected-only --output-dir .tmp/phase8/wave9-scale
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/wave9-full-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/wave9-wheel-final
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
python scripts/validate_repository.py
cd docs
npm run build
```

Evidence: [Julia oracle](../../scripts/coefficients/wave9/oracle.json),
[accuracy/work](evidence/phase8-firk-accuracy.json),
[registered spatial scale gate](evidence/phase8-wave9-scale.json),
[source inventory](evidence/phase8-source-inventory.json),
[full CPU gate](evidence/phase8-wave9-full.json),
[installed wheel](evidence/phase8-wave9-wheel.json).
