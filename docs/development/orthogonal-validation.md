# ROCK/SERK/RKMC increment backends

`OrthogonalRK` supplies ROCK2/4, SERK2 and RKMC2 outside the fixed 142-method
catalog. Fixed configurations round up to supported source counts; spectral
binding selects counts for each attempted step from a supplied negative-real
bound. It rejects insufficient caps rather than silently truncating stability
requirements. Full derivative stages and AD graphs are retained, so this is
not a low-storage or native-Warp implementation.

## Source and coordinate qualification

The [Julia oracle](../../scripts/coefficients/wave11/orthogonal_stabilized.jl)
pins OrdinaryDiffEq 7.8.1 tree bae9b8e5f1d00e51c554648e5094e215164a19f8,
StabilizedRK 2.6.0. It evaluates actual cache structs, coefficient constructors,
degree selectors and constant-cache step bodies. Spectral estimation is bypassed
in favor of fixed supplied counts. This does not execute the full source
controller, mutable-cache paths or native storage schedule.

It records 120 autonomous/forced/zero-rate steps: original kernels and the same
kernels applied to increments, with F(t,anchor+increment), then adding the anchor.
That is an exact mathematical coordinate change, whose binary64 evaluation
preserves zero rates. Source original affine sums can drift: SERK2 reaches
2.27e-13 on the initial sampled counts. Translated controls give exactly zero
drift. The package directly executes the recurrence in those coordinates.
All physical stage states, times, outputs and raw defects are compared; high
qualified counts include ROCK2 200, ROCK4 124/142, SERK2 250 and RKMC2 1000.
The [source result](../../scripts/coefficients/wave11/orthogonal_stabilized.json)
retains every evaluation and all actual coefficients. An
[export/check script](../../scripts/coefficients/wave11/export_orthogonal.py)
reproduces the runtime data, including source-computed RKMC roots for every
integer count 3–1000.

The pure recurrence program is shared by execution and scalar coefficient
inspection. It derives final/error derivative weights without a dense RK
execution tableau. A scalar first-order clock avoids summing many rounded
coefficient entries at high counts. Independent source traces, general RK
order conditions through four (and ROCK4's order-three companion), nonlinear
time-dependent refinement and full-stage quadrature reconstruction qualify it.

## Real-axis stability and source exceptions

Fixed-step source kernels confirm these counterexamples:

| Method/count | z | R(z) |
|---|---:|---:|
| ROCK4 / 133 | -8.254930664062499 | 1.00458099 |
| ROCK4 / 152 | -8.401918017786704 | 1.00773811 |
| ROCK4 / 152, outer interval | -7974.05622644043 | 1.01489981 |
| SERK2 / 150 | -17760.8 | 7.46119082 |
| SERK2 / 250 | -49600.8 | -1745.25273861 |

The two largest ROCK4 source counts have unstable interior bands near -8,
not merely an inaccurate outer endpoint. Counts 133/152 are rejected; the
largest supported ROCK4 count is 142. Other source configurations are retained.
SERK2's largest source selector radii also overreach. Package sizing applies
.95/.9 margins to the inverted ROCK4/SERK2 selectors; ROCK2/RKMC use the
conservative inverse with s-1. These are supplied negative-real spectral
bounds, not internal eigenvalue or nonlinear/nonnormal stability certificates.

The [offline spectral screen](../../benchmarks/orthogonal_stability.py) checks
all **1,103** supported configurations: 46 ROCK2, 48 ROCK4, 11 SERK2 and
998 RKMC2 counts. The table-driven families combine dense grids with every
numerically recovered polynomial critical point, evaluated by the recurrence.
RKMC2 is grid sampled. A coarse uniform grid alone missed the ROCK4 / 152
interior band. The [screen report](evidence/orthogonal-stability.json) retains
per-configuration bounds, maxima and sampling details. These float64 numerical
gates are not exact-arithmetic proofs.

## Runtime and quantities

Each step owns its input, initializes once, evaluates consistently timed stages
and refreshes the physical endpoint for copied fields. Finalization occurs only
after successful assessment. All floating integrated fields join the same affine
recurrence, including coupled quantities with distinct shapes. Final weights
are ordinary derivative quadratures, paired with actual stages, so custom hooks
need no fictitious component weights. Tests reconstruct every integrated field.
The declared fields follow affine tensor ODE updates; nonlinear custom update
projectors, constraints and mass matrices are outside this contract.

Zero-rate fields remain unchanged at large and small offsets in float32/64.
State/parameter gradients use ordinary differentiable recurrence operations.
Adaptive rejection/refinement, portable tensor continuation, switching and
physical composition are tested. The source's unused extra SERK2 initial
evaluation is omitted; a step costs s+1 full RHS entries including endpoint.
ROCK4's estimate exponent is four; the other three source comparisons have
exponent two. No source error-norm scaling, FSAL or dense output is claimed.

## Understood heat consumer

The [benchmark](../../benchmarks/orthogonal_backend.py) uses periodic diffusion
with a symmetric negative-real Laplacian. The initial temperature contains
constant, slow, middle and Nyquist Fourier modes, so smoothing and every
semi-discrete reference are known analytically. There are 39 CPU float64 cases
at N=64/1024/4096, requested h=.01/.005, and T=.1. RK4 reduces h for stability;
the other methods choose stages. First-use coefficient preparation is included.
All cases finish. The [complete report](evidence/orthogonal-backend.json) includes
work, timing, mean drift and extrema.

At N=4096, requested h=.01, the package uses 30/48/40/61 stages for
ROCK2/ROCK4/SERK2/RKMC2. ROCK2 uses 310 RHS calls, max error 5.41e-7 and .0229
seconds; SERK2 uses 410 calls, error 4.53e-7 and .0288 seconds. Stability-limited
RK4 uses 10,712 calls and .987 seconds, with error 2e-15. Both new order-two
methods meet the tested 1e-6 target much more cheaply in this case; these are
not equal-error speed rankings. RKG2 gives error 3.33e-7 at 640 calls/.0497
seconds; RKL2 error is 2.26e-5 at 380 calls/.0298 seconds.

ROCK4 retains the unresolved highest-frequency mode and gives error 2.79e-3
despite formal fourth order: stability and fast-mode damping are distinct.
RKMC2 gives error 1.05e-7 at 620 calls/.0448 seconds. The observed mean drifts
are small, but no nonlinear SSP/TVD, positivity, SPH conservation or general
performance claim follows from this normal linear consumer.

## Gates

Focused recurrence, previous stabilized drivers, switching and composition:
**275 passed**. The full CPU gate reports
**8,624 passed, 639 skipped, 157 warnings**, 341.91 seconds with eight workers
and one numerical thread per worker. The [full report](evidence/orthogonal-full.json)
fingerprint matches the final runtime/tests/configuration. A fresh Julia execution
reproduces all 120 step controls, 998 root pairs and five instability controls.
The coefficient export check passes. The [installed wheel](evidence/orthogonal-wheel.json)
passes outside the checkout, including spectral-bound ROCK4. All 56 published
Python blocks execute; 13 sketches remain labeled. Catalog and repository links
pass, and Docusaurus builds successfully. Device/graph coverage remains limited
to available hardware and the suite's recorded skips.
