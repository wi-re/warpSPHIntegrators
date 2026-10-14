# Phase 8 — explicit extrapolation backend

The unregistered Extrapolation factory supplies Euler orders 2–8 and even
midpoint orders 2–10. It has a fixed target order with adaptive timestep and
uses full-state affine quadratures. The fixed catalog remains 142. Source pin:
OrdinaryDiffEq 7.8.1 tree bae9b8e5f1d00e51c554648e5094e215164a19f8,
Extrapolation 2.5.0. A stdlib Julia oracle executes serial AitkenNeville and
ExtrapolationMidpointDeuflhard constant-cache bodies, with generic exact-rational
source coefficient construction. Twenty-four autonomous/forced nonlinear
two-field records cover all twelve configurations. Main states and comparison
errors agree within 3e-14; midpoint column coefficients agree independently.
The one-column midpoint case uses the degenerate recurrence and identity column
weight rather than a production adaptive constructor. Source hashes, scaffolding
and controller qualifications are recorded in the oracle JSON.

Euler columns use Romberg subdivisions 1,2,4,… and cancel inverse-step powers;
midpoint uses even harmonic subdivisions 2,4,6,… and cancels even powers. Public
column coefficients are derived using Python Fraction, then converted to float.
The source midpoint recurrence is unsmoothed. Comparison drops the coarsest
column; order-two midpoint compares with Euler. Estimator exponents are p for
Euler, p-1 for midpoint above order two and 2 at order two. Source adaptive-order,
work controllers and threaded columns are not reproduced. Source implicit Euler
extrapolation uses frozen-Jacobian linear corrections, so it has a separate
solver/operator contract rather than being treated as repeated nonlinear BE.

The initial derivative is shared. Unused Euler column endpoint derivatives are
omitted; one physical final endpoint refresh remains. Full work per step is
2+sum(n-1): midpoint 4/6/8/10 costs 6/11/18/27 RHS entries, while Euler order
eight costs 249. There is no FSAL saving, internal parallelism or low-storage
claim; returned stage tensors/autograd graphs consume storage proportional to
the total column work. Increment accumulation preserves zero-rate offsets
exactly and reduces cancellation in state/error combinations.

All 62 focused tests pass: source maps/coefficient weights/stage times, nonlinear
forced convergence at every order, independent scalar stability polynomials
through claimed order, defect exponents, actual work, large zero-rate offsets,
adaptive refinement and rejection, backward nonautonomous stepping, parameter
gradients, portable continuation and Phase 7 physical composition. A rich
mechanics/quantity consumer reconstructs every integrated field from the public
affine weights, checks lifecycle counts and copied endpoint fields, and rejects
nonfinite trials before finalize. Custom finalizers opt in explicitly and cannot
project integrated values/time/IDs. Both order-two main maps equal registered
explicit midpoint; equivalence is recorded without duplicate registration.
The pre-final composition/runner subset has 113 passing cases.

The known-consumer benchmark completes 108 fixed and 48 adaptive cases: N=64/4096,
manufactured nonlinear reaction with exact exp(-rate*t), real planar rotations
with exact cos/sin, h=.2/.1/.05 and adaptive tolerances 1e-5/1e-8/1e-11. Seven
extrapolation configurations are compared with RK4/DP5. All runs complete. These
are explicit nonstiff tests, not evidence for arbitrary stiff/SPH dynamics.

At N=4096 and adaptive rtol=1e-8, midpoint order ten uses 351 RHS calls and
1.50e-11 maximum reaction error; DP5 uses 511 and 4.40e-11. On rotations the
same method uses 378 calls with 3.08e-8 error, versus DP5's 770/2.45e-8. This
establishes a useful high-order accuracy/work trade on understood problems.
Euler order eight uses 1,494 reaction calls for 6.08e-11 error and is less
economical. Low-order midpoint estimates can be conservative: order four needs
14,136 rotation calls at this tolerance. Tolerances control local comparisons,
not final global errors. Fixed-grid fine/high-order results can hit roundoff.
No universal performance or conservation guarantee follows from these examples.

The final full CPU gate passes **7,851 tests, 639 skipped**, zero failures/errors,
in 312.69 seconds of pytest time (313.9 seconds with orchestration). Eight workers
use one numerical thread each; its fingerprint matches final source, tests and
shard/config files. The extrapolation category belongs to exactly one CI shard.
The installed 0.6.1 wheel passes imports/exports, 135 fixed tensor drivers and
adaptive extrapolation/quadrature callbacks outside the checkout with isolated
project imports and shared numerical dependencies. All 46 published Python
blocks execute, with 13 labeled sketches. Catalog, repository links and production
site build pass. Editable metadata remains historical 0.5.0; source/wheel are 0.6.1.

Evidence: [consumer measurements](evidence/extrapolation-backend.json),
[full CPU gate](evidence/extrapolation-full.json),
[installed wheel](evidence/extrapolation-wheel.json).

Reproduce:

```bash
python -m pytest -q tests/test_extrapolation.py
python benchmarks/extrapolation_backend.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/extrapolation-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/extrapolation-wheel
JULIA_DEPOT_PATH=/tmp/integrators-julia-oracle:$HOME/.julia julia --startup-file=no scripts/coefficients/wave13/extrapolation.jl "$HOME/.julia/packages/OrdinaryDiffEq/OnlvZ"
```
