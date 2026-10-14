# Spatial solver follow-up — 14 October 2026

The preferred **shifted spatial ILU** passes the unchanged Phase 6 work/time/
memory/tolerance bar on every reaction/diffusion configuration through N=1024.
The original local stage-block approximation still fails diffusion. The result
removes that measured 1D spatial-solve blocker; it does not register higher-stage
methods or claim a general unstructured/multifield SPH solver.

## Implementation and correctness

`stage_shifted_preconditioner` diagonalizes the small real RK stage matrix and
uses an application-owned factory to solve `(I-h*lambda*J_frozen) v=b` for each
stage eigenvalue. Complex conjugate systems reconstruct a real correction.
The full nonlinear Newton map retains each stage's own Jacobian; only the
preconditioner freezes a common Jacobian. Eigenbasis conditioning, input/output
layout, finiteness and real reconstruction are checked. Cached spatial factors
last for one Newton iterate/context. The public adapter requires Torch only;
it preserves the existing implicit-function adjoint path.

The benchmark's periodic 1D stencil has an explicit sparse spatial Jacobian.
Full coupled sparse LU is the reference. Full sparse ILU and shifted LU/ILU
provide alternatives. SciPy/SuperLU is declared in the benchmark extra, not a
new core dependency. The shifted provider handles complex factors explicitly.
These implementations follow [SciPy's LU interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)
and [ILU interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spilu.html).
The small-stage transform follows the shifted-solve strategy described in
[OrdinaryDiffEq's FIRK documentation](https://docs.sciml.ai/OrdinaryDiffEq/stable/implicit/FIRK/);
this is an eigendecomposition-based preconditioner, not a reproduction of its
complete W-transform driver or adaptive estimator.

Nine new tests verify the transform against dense spatial blocks, cache scope,
bad solver/basis rejection, nonlinear true-residual convergence, linear-flow
state gradients and the formerly failing N=1024 case. Four sparse-provider tests
skip explicitly when the optional SciPy dependency is absent. Dense reference
and generic Torch adapter tests remain in the core test environment.

## Unchanged bar and new measurements

120 fresh-process configurations compare two-/three-stage Gauss/Radau,
reaction/diffusion and N=64/256/1024 with local stage blocks, full sparse LU/ILU,
and shifted spatial LU/ILU. Problem, h=0.05, three steps, two repetitions,
restart=30, linear budget=120, Newton budget=6 and tolerances match Phase 6.
The same bar requires zero tolerance failures, <=120 total Krylov iterations
per step, <=3x corresponding s=2 median time/RHS work, and <=256 MiB additional
native peak RSS through N>=1024. Agreement within 1e-7 of full sparse LU is an
additional correctness check, not a relaxed threshold.

At N=1024 on diffusion:

| Scheme | Local stage block | Full sparse LU | Shifted spatial ILU |
|---|---|---|---|
| Gauss s=3 | 720 Krylov iterations, fails | 3 iterations, 0.016 s median | 10 iterations, 0.028 s median |
| Radau s=3 | 720 Krylov iterations, fails | 3 iterations, 0.010 s median | 10 iterations, 0.021 s median |

Direct LU, shifted LU and shifted ILU pass every size/problem gate. Full coupled
ILU converges but has one Radau/N=64 diffusion case outside the additional
reference-agreement check, so that variant is not selected. Each candidate's
verdict is retained separately; the selected shifted-ILU verdict is not an
aggregate claim that every attempted preconditioner passed. Maximum additional
peak RSS across the sweep is about 159 MiB, below the original bar. Setup and
factor nonzero counts are recorded alongside native process memory, residuals,
RHS work, Krylov work and time. Matrix-free RHS/JVP evaluation remains unchanged.

Hardware/environment matches Phase 6: Intel Core Ultra 9 285K, CPU float64,
Python 3.13.14, Torch 2.13.0+cu130 and one numerical thread. Direct/ILU sparse
cost on this 1D cyclic stencil does not establish scalable 3D SPH factorization.
Multigrid remains an option for spatial cases where sparse factors cease to be
competitive; the benchmark does not require an unmeasured multigrid implementation
once the selected sparse spatial approach clears the bar.

## Wave 9 and Phase 7 disposition

Wave 9's measured scale prerequisite is cleared **for these recorded cases**.
The s=3 prototypes remain unregistered because companions/estimate rates,
pinned-source transcription and complete catalog gates are still outstanding.
Spatial/field-aware assembly for actual downstream SPH remains application work.

The downstream Phase 7 audit finds no executable pressure-entropy PESPH-A
prototype: `warpSPH/src/warpSPH/schemes/builder.py` still raises NotImplementedError
for PESPH, and CompSPH integrates internal energy while entropy remains constant
metadata. The package can build and validate splitting on declared oscillator/
PDE consumers, but production SPH symplectic/energy claims remain gated on the
entropy-carrying implementation and its unsplit baseline. No downstream files
are changed by this follow-up.

## Validation

**6,972 passed, 609 skipped** in 309.14 seconds, eight work-stealing
CPU processes and one numerical thread each. The final code/config fingerprint
matches the full gate. Nine new focused tests pass; no catalog registrations
change.

Installed-wheel validation exercises the generic shifted adapter without SciPy.
Published examples, catalog, local links and the production site are checked.
Reports retain the local shared-dependency wheel qualification from Phase 6.

```bash
python benchmarks/implicit_scale.py --spatial --output-dir .tmp/spatial-followup/scale
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/spatial-followup/full
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/spatial-followup/wheel-final
python scripts/validate_doc_examples.py
python scripts/validate_repository.py
cd docs
npm run build
```

Evidence: [scale sweep](evidence/spatial-followup-scale.json),
[full CPU gate](evidence/spatial-followup-full.json),
[installed wheel](evidence/spatial-followup-wheel.json). Raw case solutions remain
under `.tmp/spatial-followup`; the report records solution checksums and direct
reference agreement.
