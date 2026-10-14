# Parallel DIRK numerical map with serial execution

`ParallelDIRK()` exposes the pinned PDIRK44 four-stage, fourth-order map outside
the 142 fixed registrations. It is an ordinary first-order ODE method, not a
partitioned position/velocity integrator. This corrects the original expansion
plan's classification. The package executes stages serially through the existing
DIRK/JFNK driver; no internal threading or parallel-speed claim is made.

## Source and map

The [Julia oracle](../../scripts/coefficients/wave12/pdirk.jl) pins OrdinaryDiffEq
7.8.1 tree bae9b8e5f1d00e51c554648e5094e215164a19f8, PDIRK 2.4.1. It evaluates
the actual tableau constructor and constant-cache step body. Both source
execution branches run, with threading macros replaced by ordinary loops and
an independent analytic-Jacobian Newton stage solver. It does not run the full
SciML nonlinear solver, scheduler, initialization or cache extrapolation policy.
All source hashes and eight nonlinear autonomous/forced map records are in
[the result](../../scripts/coefficients/wave12/pdirk.json).

In source stage order (k11,k12,k21,k22), the ordinary RK representation is

```
A = [[1/2,   0,   0,   0],
     [  0, 2/3,   0,   0],
     [-5/2,5/2, 1/2,   0],
     [-5/3,4/3,   0, 2/3]]
b = [-1, 3/2, -1, 3/2]
c = [1/2, 2/3, 1/2, 1/3]
```

The first two stages are independent; the last two depend only on the first
pair. Source output weights reorder from (b1,b2,b3,b4) over (k11,k21,k12,k22)
to the vector above. The package matches both source branches within 2e-13.
All eight rooted-tree conditions through order four are checked independently,
alongside nonlinear/time-dependent trajectory refinement. Negative weights
give no SSP guarantee; no companion estimator or FSAL reuse is claimed.

## Stability

Exact symbolic elimination gives

```
R(z) = (z+3)*(9*z²-20*z+12) / ((z-2)²*(2*z-3)²).
```

Its poles are positive real (2 and 3/2); there are no closed left-half-plane
poles. On the imaginary axis, denominator magnitude squared minus numerator
magnitude squared is x⁶(16x²+119), nonnegative for all real x. Together with
R(infinity)=0, this proves A- and L-stability. Solver tolerances still determine
how accurately the implemented map realizes those properties. The direct
tableau rational function is also checked at real/complex test points.

## State, solver and runtime boundaries

The factory owns its input snapshot and initializes once. It suppresses the
inner DIRK finalize until the candidate passes assessment, refreshes the physical
endpoint, and finalizes once with four RK weights plus a zero endpoint weight.
Copied fields therefore come from the actual endpoint rather than the final
c=1/3 stage. Coupled extra quantities use the same RK weights. Failed stage or
endpoint trials cannot finalize/commit the input state.

Existing JFNK/preconditioner options and per-stage diagnostics are retained.
Actual RHS counts include stage refresh, endpoint refresh and solver probes;
solver diagnostics are a separate count. Fixed stepping, portable tensor
restart, accepted algorithm switching and physical composition are covered.
Solver objects/options remain caller-owned and must be resupplied on restart.
Tests compare state and RHS-parameter gradients with finite differences through
nonlinear implicit stages. There is no estimator, adaptive mode or claim of
upstream nonlinear-cache/extrapolation equivalence.

## Known consumer

The [benchmark](../../benchmarks/parallel_dirk_backend.py) records 32 CPU float64
cases with N=64/4096, fixed h=.125/.0625, and independent reaction rates:
1–sqrt(10) on a nonlinear forced exp(-rate*t) trajectory, or 1–10000 on stiff
linear decay. All cases finish successfully. Diagonal shifted preconditioning
is supplied for DIRK and stage-block preconditioning for Radau. The complete
[report](evidence/parallel-dirk-backend.json) keeps errors, actual RHS calls,
solver diagnostics, Krylov counts and wall times.

At N=4096, nonlinear forcing and sixteen steps of size .0625:

| Method | Actual RHS calls | Max error |
|---|---:|---:|
| Parallel DIRK, serial | 790 | 4.80e-6 |
| SDIRK2 | 351 | 2.16e-4 |
| Radau IIA s=2 | 422 | 1.34e-5 |
| ESDIRK4(3)6L[2]SA | 880 | 1.75e-7 |

PDIRK is another compact L-stable order-four map with two independent stage
groups. It improves fixed-grid error over the lower-order comparisons, with
more solves/work; the existing six-stage ESDIRK4 is much more accurate here
for a similar RHS budget and also supplies an estimator. These measurements
do not establish an equal-error speed advantage or a new default method.
Stage-parallel scheduling would need its own lifecycle, solver ownership and
cost gates in a future execution backend.

## Validation

Focused DIRK, parallel-DIRK, physical-composition and algorithm-switch suite:
**474 passed** (302.55 seconds). The full CPU gate reports
**8,477 passed, 639 skipped, 157 warnings**, 326.01 seconds with eight workers
and one numerical thread per worker. The [full report](evidence/parallel-dirk-full.json)
fingerprint matches the final runtime/tests/configuration. The
[installed wheel](evidence/parallel-dirk-wheel.json) passes outside the checkout,
including the new factory. All 54 published Python blocks execute; 13 sketches
remain explicitly labeled. Catalog and repository checks pass (120 Markdown
pages, three notebooks, 1,067 local links, 142 registered dispositions), and the
Docusaurus production build passes. No unavailable device/graph capability is
inferred from CPU results.
