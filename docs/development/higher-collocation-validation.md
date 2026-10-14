# Phase 8 — higher-stage collocation backend, 14 October 2026

The remaining Gauss/Radau backend stage configurations now use the parameterized
`Collocation(family, stages)` factory over the existing block JFNK driver. Stages
2–9 are supported: Gauss propagated orders 4–18, Radau orders 3–17. The four
fixed catalog records and their existing estimators remain unchanged, so the
registered count stays 139. Frontend/PESPH integration is separate application
work; understood backend ODE and spatial-field problems qualify this batch.

## Coefficients, companions and accuracy

The shifted Legendre basis integrates cardinal polynomials without a high-degree
monomial inverse. Gauss uses the pinned FIRK generator's same-node companion:
quadrature moments through order s-1, with moment s set to zero. The companion
order is s-1 and estimate exponent s. Radau uses a derived unsmoothed companion
of order s and exponent s+1: append a null stage of weight -1 and add Lagrange
extrapolation weights L_j(0) to b_j. This adds one RHS evaluation. It is not the
upstream filtered/re-estimated controller or an adaptive-order implementation.

The Julia oracle checks the complete OrdinaryDiffEq 7.8.1 monorepo tree
`bae9b8e5f1d00e51c554648e5094e215164a19f8`, FIRK 2.8.1. It evaluates the pinned
Gauss constructor and Radau construction through the collocation matrix before
its eigenbasis transform. Coefficient arithmetic uses BigFloat, with Julia
stdlib Golub–Welsch Float64 quadrature nodes converted to BigFloat. The Radau
companion is independently derived in that oracle. Both autonomous and forced
nonlinear one-step states/estimates come from a separate dense Julia Newton solve.
All sixteen family/stage configurations match these independent references.

Tests verify collocation C(s), quadrature B(p), companion moments and a nonzero
first omitted moment, scaled linear-series coefficients through propagated
order, left-half-plane stability, Radau stiff accuracy/L-stability and Gauss
symmetry/symplectic identities. C(s) and B(p) establish the collocation nonlinear
order; orders 17/18 are not claimed from a float64 refinement fit below the solve
floor. Actual block-driver nonlinear steps match Julia; gradient/restart/adaptive
checks exercise the new factory. The existing s=3 main maps and estimates match
the fixed registrations. Initial states are cloned by the factory.

An additional known-solution accuracy/work sweep uses a frequency-4 oscillator
and cubic decay at h=.5/.25/.125 for s=2/3/4/5/7/9. It instruments all RHS calls
and records strict-solve precision-floor outcomes rather than treating them as
convergence to a requested tolerance. Order fits are omitted for affected pairs
or errors below 1e-11; very coarse pairs can be preasymptotic. This is an
observational consumer comparison separate from the tolerance-required spatial
gate. High stage count does not promise measurable order above a float64 floor.

## Spatial backend and a resolved scaling limitation

A larger-stage run exposed loss of real reconstruction in the existing global
eigenvector shifted solver. That helper remains useful for well-conditioned stage
bases. The new `stage_schur_preconditioner(build_operator, decompose)` works in an
orthogonal **real Schur** basis instead. It back-substitutes over one/two-stage
blocks and the upper stage couplings, diagonalizing only each local two-stage
block. The operator factory returns a shifted spatial solve and a matching frozen
Jacobian action. The decomposition provider returns real `(T,Q)` with A=Q*T*Q.T.
The backend validates the decomposition, operator layouts and finite real output,
and caches only within a Newton iterate/context. No SciPy runtime dependency is
added. The reference benchmark supplies optional SciPy real Schur and sparse ILU;
other backends can provide sparse, multigrid or matrix-free callbacks.

The 60-case CPU sweep covers both families at s=3/4/5/7/9, reaction and periodic
reaction-diffusion fields, and N=64/256/1024. Each case runs in a fresh process;
three cases run concurrently with one numerical thread each. The gate requires
every step to meet nonlinear tolerance, at most 120 Krylov iterations per step,
at most 256 MiB additional peak process RSS, and at most three times the s=3
RHS work **per stage**. Stage count necessarily adds work; this normalization
keeps an accidental nonlinear-solver regression distinct from adding stages.
Times are retained with a concurrent-contention qualification. The cases use
concrete backend stencil assembly and require no frontend. They do not qualify
an arbitrary spatial Jacobian or a native Warp implementation.

## Reproduction

The full CPU gate passes **7,185 tests, 621 skipped**, zero failures/errors, in
336.39 seconds of pytest time (337.6 seconds with orchestration), with eight
work-stealing workers and one numerical thread per worker. Its fingerprint
matches the final source/tests/config. The 52 new factory/Schur cases pass;
including existing spatial-preconditioner cases gives 61 passing targeted tests.
The 60-case spatial gate passes, with at most 10 Krylov iterations, 167.4 MiB
additional peak RSS and 1.0 times the s=3 work per stage. All 40 published Python
blocks execute, 13 sketches remain labeled, and catalog, repository links and
production site build pass. The installed 0.6.1 wheel passes outside the checkout,
sharing numerical dependencies while isolating project imports. The full test
environment retains historical installed metadata 0.5.0; source version is 0.6.1.

```bash
julia --startup-file=no scripts/coefficients/wave9/higher_stages.jl /path/to/OrdinaryDiffEq/OnlvZ
python benchmarks/higher_collocation.py
python benchmarks/collocation_accuracy.py
python -m pytest -q tests/test_collocation_factory.py tests/test_spatial_preconditioner.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/higher-full
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/higher-wheel
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
python scripts/validate_repository.py
cd docs
npm run build
```

Evidence: [pinned constructor/nonlinear oracle](../../scripts/coefficients/wave9/higher_stages.json),
[spatial scale](evidence/higher-collocation-scale.json),
[known accuracy/work](evidence/higher-collocation-accuracy.json),
[full CPU gate](evidence/higher-collocation-full.json),
[isolated installed wheel](evidence/higher-collocation-wheel.json).
