# Exponential operator maps and source qualifications

`ExponentialRK` adds fourteen qualified configurations without changing the 142
fixed registrations. Semilinear Lawson/exponential Euler and ETDRK2/3/4 and
Hochbruck–Ostermann 4 require a constant linear accessor. Exp4, EPIRK4s3A/B,
EPIRK5P1/P2, EXPRB53s3 and Exprb32/43 freeze the full Jacobian and augment time.
ETDRK2 is the same mathematical map as the registered ETD2RK; this execution path
adds robust matrix-exponential actions, explicit action budgets and operator AD.

## Source and mathematical gates

The [source harness](../../scripts/coefficients/wave11/exponential.jl) runs actual
OrdinaryDiffEq 7.8.1 constant-cache kernels and cached operator constructors from
pinned tree bae9b8e5f1d00e51c554648e5094e215164a19f8. Exact dense augmented-block
exponentials replace source Krylov routines; analytic Jacobians and supplied FSAL
make the scope independent of the full SciML runtime. All 120 one-step controls
and manufactured refinement trajectories remain in the
[oracle](../../scripts/coefficients/wave11/exponential.json).

A second [formal trace](../../scripts/coefficients/wave11/exponential_programs.jl)
executes those kernels with symbolic scalar operator/rate inputs.
[The compiler](../../scripts/coefficients/wave11/export_exponential.py) collects
linear operator-valued coefficients rather than manually transcribing each
method. Runtime definitions contain coefficients and stage nodes, separately
from full-state execution. Cached semilinear formulas and autonomous time-
augmented linearized formulas match the independent dense source maps.

The pinned **HochOst4 Krylov branch** has a cross-stage phi-term discrepancy:
its measured order is 3.140, versus 3.997 for the cached branch used here.
Pinned **EPIRK5s3** reaches only 2.038 even with time augmentation and remains
unavailable. Unaugmented Exp4/EPIRK/Exprb kernels generally lose nonautonomous
order; the oracle retains these failures alongside the augmented controls.
No unchanged nonautonomous-source equivalence is claimed.

## Execution and gradients

The driver supports floating integrated quantities sharing a dtype/device,
owned stages, isolated JVP/time probes, endpoint refresh, raw embedded defects,
adaptive runs and portable TensorState restart. It uses affine tensor-ODE
semantics. A custom finalize opts into `exponential_compatible` and consumes
`ExponentialQuadrature`: an initial operator increment and the operator-weighted
rate increments, rather than scalar weights over physical rates. There is no
FSAL, source controller/cache transfer or mass-matrix claim.

For at most `dense_limit=64` flattened entries, differentiable matrix
exponentials retain complete operator/Jacobian derivatives. Nonlinear state and
operator-parameter gradients match finite differences; no frozen-J Hessian
approximation is made. Above the limit, constant operators use Arnoldi and
transpose-action adjoints. Tracked operator derivatives above the bound raise
explicitly. Increasing the dense bound is an explicit memory/cost decision.
Large actions require invariant-space closure or agreement with a smaller
projected dimension; `phi_dim` is a budget, not a guarantee. An insufficient
budget produces an unsuccessful step. This is a numerical action check, not a
proof for arbitrary nonnormal spectra. No GPU coverage is inferred.

## Understood consumer

[Reaction–diffusion measurements](evidence/exponential-backend.json), produced by
[the benchmark](../../benchmarks/exponential_backend.py), cover 36 CPU float64
cases at 16/128/1,024 entries, including a nonlinear manufactured exp(-t)
trajectory. At 1,024 entries and h=.1, ETDRK2/3/4 errors are
6.36e-4 / 4.51e-6 / 4.34e-8. Times are .371/.741/.966 seconds; the existing ETD2RK
map reaches the same second-order error in .060 seconds with fewer actions and
nonlinear calls. This is a useful accuracy option, not a universal speed win.
At 16 entries EPIRK5P1 reaches 4.08e-9 versus Exprb43's 1.84e-7 at the same h;
both take roughly .069 seconds in this single run. First-use AD preparation is
included, and comparisons have unequal errors.

Focused exponential and bounded-experiment gates pass **123 tests**. Full-suite,
installed-wheel and documentation gates are recorded with the completed batch.

## Completed batch gates

The full local CPU gate passes **8,778 tests, 639 skipped, 157 warnings**
in 329.77 seconds with eight workers and one numerical thread per worker.
The [full report](evidence/exponential-backlog-full.json) fingerprint matches
the runtime/tests/configuration. The [installed-wheel gate](evidence/exponential-backlog-wheel.json)
passes all modules and adds exponential/Taylor smokes outside the checkout.
All 58 published runnable blocks execute; 13 sketches remain labeled.
Catalog generation, repository links/anchors and the production site build pass.
No new CUDA coverage or source-runtime equivalence is inferred from CPU gates.
