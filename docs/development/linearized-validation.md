# Full-JVP Rosenbrock backend

`Rosenbrock(method)` supplies Rosenbrock23/32 and RosenbrockW6S4OS outside the
142 fixed registrations. All use the full prepared RHS Jacobian, matrix-free
JVP actions and linear GMRES stages. No mass matrix, approximate-W substitution,
FSAL or source cache/retry/controller equivalence is claimed. W6S4OS uses its
pinned coefficients; the pair has an explicitly qualified time-term correction.

## Source execution and time-term correction

The [Julia oracle](../../scripts/coefficients/wave11/rosenbrock.jl) pins
OrdinaryDiffEq 7.8.1 tree bae9b8e5f1d00e51c554648e5094e215164a19f8, Rosenbrock
2.7.0 and RosenbrockTableaus 2.4.2. It executes actual constant-cache step bodies
with exact analytic J/f_t and dense stage solves, identity mass and ordinary
broadcast in place of source annotations. It supplies initial FSAL values and
captures raw defects instead of applying the source's residual normalization.
The differentiation helper is inspected/hashed to establish that dT is the
unscaled partial time derivative. This is not full SciML solver execution or
an executed mutable-cache equivalence claim.

On y'=t from y(0)=0, the source pair uses gamma=1/(2+sqrt(2)) and yields
k1=gamma*h, k2=h/2 and k3=(2-2*gamma)*h. Rosenbrock23's main result is exact,
but its defect is h²*(1-gamma)/6. Rosenbrock32 adds that defect to the exact
answer, giving a local O(h²) error and first-order time-dependent trajectories.
At h=.2, the source high result is .024714045207910318 versus exact .02;
the source defect is .004714045207910317. This is an execution-path finding,
not an intrinsic limitation of the mathematical Rosenbrock pair.

Autonomous augmentation, t'=1 and y'=f(t,y), identifies the correction:
the third-stage partial-time term must be h*gamma*f_t rather than h*f_t.
The oracle records original kernels and a control changing only that expression.
Corrected y'=t results/defects agree with .5*h²/zero to roundoff at three grids.
All twelve original nonlinear maps and eight corrected pair maps remain in
[the source record](../../scripts/coefficients/wave11/rosenbrock.json).
Package maps and raw defects match the corrected pair / original W6S4OS within
2e-13. A separate test compares every method with the autonomous augmented system,
using differentiable time as an actual state coordinate.

## Shared stage representation

With M=I-h*gamma*J, the common Rodas-form increment equation is

```
M*K_i = h*gamma*f(t+c_i*h, y+sum(A_ij*K_j))
        + gamma*sum(C_ij*K_j) + h²*gamma*d_i*f_t
y_new = y + sum(b_i*K_i)
```

For the pair the change from source slopes is K1=h*gamma*k1,
K2=h*gamma*(k2-k1), K3=h*gamma*k3-2*K1-c32*K2. The corrected d vector is
(gamma,0,-gamma); the uncorrected source would give d3=1-2*gamma. The
third-minus-second raw defect has exponent three for both propagated orders.
W6S4OS coefficients export directly from source, with a reproducible
[export/check script](../../scripts/coefficients/wave11/export_rosenbrock.py).
Nonlinear/time-dependent order and independent time-polynomial defects are tested.

`RosenbrockQuadrature` carries solved flat increments and their weights, paired
with the physical RHS stages and a zero endpoint contribution. These increments
are already h-scaled and differ from raw rates. A custom finalize opts into
`rosenbrock_compatible=True` and can reconstruct the complete integrated update
without pretending that ordinary RK weights over raw forces would be valid.

## Stability and measured limits

Rosenbrock23 has
R(z)=(1+(1-2*gamma)*z)/(1-gamma*z)². Its pole is positive, its stiff limit is
zero, and denominator magnitude squared minus numerator magnitude squared on
the imaginary axis is gamma⁴*x⁴. It is A- and L-stable.

Rosenbrock32's stiff limit is (2+2*sqrt(2))/3=1.609475708. It has a bounded
stiff region and is not A-stable; |R(i)| is about 1.00134948. W6S4OS has a
rounded stiff limit about 7e-15 and damps large negative-real modes, but
|R(i)| is about 1.00003963, so it is not A-stable either. A near-zero stiff
limit alone does not establish L-stability. Tests retain these counterexamples
rather than assigning a common stability label to the family.

## Gradients and ownership

Forward stages use GMRES and independently assess the true unpreconditioned
residual, including with left preconditioning. The implicit stage adjoint
retains derivatives through J via differentiable JVP: state Hessian terms and
RHS-parameter dependence are included. This differs from the legacy ROS3P
frozen-operator adjoint. Nonlinear state/parameter gradients match finite
differences at the numerical tolerance used in the tests.

The input is snapshotted. JVP and central time probes operate on isolated
prepared bases so they cannot contaminate actual-stage lifecycle accumulators.
Initialize/finalize run once, only successful candidates finalize, and the
physical endpoint supplies copied fields. Coupled extra quantities participate
in the same flat stage system. Tests reconstruct quadratures, verify probe
isolation and coupled quantities, exercise nonfinite/failed-solve rollback,
adaptive rejection/refinement, portable tensor restart, switching and physical
composition. Float32 uses a suitable explicit linear tolerance. Unsupported
AD operations fail rather than silently substituting a different Jacobian.

Default f_t uses central differences; `ft_eps` controls spacing and an exact
callback can replace it. Accuracy below solver/time-probe floors is not promised.
The callback receives an owned preprocessed initial stage and returns the
prepared RHS's partial-time update. Solver options and callbacks must be
resupplied on restart; they are application-owned.

## Known consumer and cost

The [benchmark](../../benchmarks/linearized_backend.py) records 32 CPU float64
cases at N=64/4096 and h=.125/.0625, with rates 1–sqrt(10) on a nonlinear
forced exp(-rate*t) trajectory, or 1–10000 on stiff linear decay. Shifted
diagonal preconditioning is supplied to every method. Actual RHS calls include
JVP/time probes and endpoint refresh; per-solver diagnostics are separate.
The [complete report](evidence/linearized-backend.json) records every case.

At N=4096, nonlinear forcing and sixteen steps of size .0625:

| Method | Actual RHS calls | Max error | seconds |
|---|---:|---:|---:|
| Rosenbrock23 | 333 | 2.04e-4 | .149 |
| Rosenbrock32 | 333 | 6.56e-7 | .149 |
| Rosenbrock-W6S4OS | 618 | 1.90e-7 | .289 |
| ESDIRK4(3)6L[2]SA | 880 | 1.75e-7 | .382 |

W6S4OS gives a similar-error work/time benefit on this nonlinear consumer.
On stiff linear decay ESDIRK4 is faster (.074 versus .089 seconds). All stage
solves finish successfully, but Rosenbrock32's unstable fastest modes grow to
about 1,445: linear-solver convergence does not imply a stable numerical map.
These are finite-grid/horizon observations, not universal speed rankings or
SPH/native-Warp performance claims.

## Validation

Focused Rosenbrock, wrapper and physical-composition suite: **101 passed**.
The full CPU gate reports **8,517 passed, 639 skipped, 157 warnings**, 318.83
seconds with eight workers and one numerical thread per worker. The
[full report](evidence/linearized-full.json) fingerprint matches the final runtime,
tests and configuration. A fresh Julia execution reproduces every tracked
original/corrected source record. The coefficient export check passes. The
[installed-wheel gate](evidence/linearized-wheel.json) passes outside the checkout,
including adaptive corrected-time Rosenbrock32. All 55 published Python blocks
execute, with 13 sketches labeled. Catalog and repository-link checks pass, and
the Docusaurus production build succeeds. CPU results do not imply unavailable
device/graph coverage.
