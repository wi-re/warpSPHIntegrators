# Structural toy consumer — 14 October 2026

The user requested an understood demo consumer to evaluate a phase-8 method and
show its behavior. This batch supplies a two-mass structural problem, a dense
linear generalized-alpha prototype, a registered Newmark baseline, a known modal
solution and a four-panel measured gallery. It does not register generalized-alpha
or change the 139-method catalog. The package runtime source remains unchanged.

## Consumer and meaning

Two masses m=1 each have a ground spring k=1 and share k=799.5. The unit mass
matrix and symmetric stiffness matrix `[[800.5,-799.5],[-799.5,800.5]]` have
in-phase/out-of-phase frequencies 1/40 rad/s. Initial modal amplitudes are 1/0.05,
with zero velocity and no external force or physical damping. Exact modal motion
is `A*cos(omega*t)`; exact modal energy is constant. This removes the need for
an unknown SPH closure or a numerical reference integrator.

The behavior run has dt=0.1 and endpoint 12 seconds: omega*dt=0.1/4. The slow
mode is resolved and the fast mode is underresolved. Numerical damping can remove
that fast component while preserving most slow motion; it cannot represent a
physically important fast mode accurately without refinement.

At rho=0.5 the prototype retains 99.947% of slow energy and suppresses the fast
component to the numerical floor. The slow phase lag is 0.0149 rad versus
0.0100 rad for undamped Newmark. At rho=0, slow energy retention is 98.778% and
lag 0.0526 rad: stronger damping has a measurable accuracy cost. At rho=0.8,
fast energy retention is 0.458%, giving an intermediate filter. These are modal
diagnostics for this problem, not claims of physical dissipation or better total
trajectory accuracy at every frequency.

## Prototype and checks

`examples/structural_modes.py` carries displacement, velocity and **algorithmic
acceleration**. It evaluates a weighted M/C/K equilibrium and solves one small
dense system for the next acceleration, then applies Newmark's kinematic formulas.
State and stiffness gradients remain in Torch's graph; input states are preserved.
The prototype is constant-matrix, linear, unforced and fixed-step. It is not a
production driver, sparse solver, adaptive estimate or restart format.

The actual registered Newmark driver evaluates the same physical force. At rho=1
the prototype matches its displacement trajectory within 5.02e-14 and retains
each modal energy to roundoff. Dense-prototype and matrix-free Newmark timings
are recorded with an explicit qualification; they do not establish a backend
speed advantage. A fair performance comparison would use equivalent solve paths.

Resolved refinement uses frequencies 1/3, endpoint 2 and h=0.1 down to 0.00625.
Measured final orders are 1.987/1.998/2.000 for rho=0/0.5/1. Refreshing acceleration
from the physical RHS each step instead of carrying it drops the damped variants
to 0.992/0.991. The rho=1 anchor remains second order. This negative control
catches a real history mistake rather than just confirming coefficients.

Augmented amplification acts on `(q,h*v,h^2*a)`, including the algorithmic mode.
The sweep spans omega*dt=1e-3 to 1e6. At the upper end, computed radii are
0.000100/0.500078/0.800078/1.000000 for target rho=0/0.5/0.8/1. Finite-frequency
damping differs from the asymptotic target. Near rho=1 and extremely large
omega*dt, clustered eigenvalues produce a numerical radius excess of about
3.3e-6 in the raw augmented eigensolve; the exact Newmark physical block has
unit-modulus eigenvalues, and trajectory parity/energy provide independent anchors.
The boundedness test uses the well-conditioned finite range through omega*dt=1e3;
the complete raw sweep, including the excess, is retained in evidence.

Thirteen focused tests cover exact modes/energy, actual Newmark parity, weighted
equilibrium, caller ownership, resolved orders/reset controls, augmented radii,
parameter anchors/guards, selective damping and state/stiffness gradients.

## Source audit: why a consumer matters

The pinned OrdinaryDiffEq 7.8.1 monorepo contains Newmark 2.4.0. Its source
constructs `beta=(1/2+alpha_f-alpha_m)^2/4`, whereas the classical scheme uses
`beta=(1+alpha_f-alpha_m)^2/4`. At rho=1 this is 1/16 versus 1/4; the demo's
isolated wrong-beta amplification radius at omega*dt=4 is 5.828, failing the
undamped Newmark anchor. The cached gamma<=1 guard also rejects classical rho=0,
whose gamma is 1.5. Its acceleration-refresh behavior is explicitly described
as first order for the damped parameterizations in the source's own docstring.

The classical prototype follows [Chung & Hulbert (1993)](https://doi.org/10.1115/1.2900803)
and the independently documented [PETSc TSALPHA2 parameter/acceleration implementation](https://petsc.org/release/src/ts/impls/implicit/alpha/alpha2.c.html).
PETSc's alpha parameters weight the new state; taking their complements gives
the old-state convention here. The source hashes, findings and isolated-control
qualification are retained in [the source audit](evidence/structural-source-audit.json).
This is not a full Julia solver comparison or an upstream issue submission.

Any production driver must carry accepted acceleration through rejection,
checkpoint/restart and gradient ownership, handle initial consistency and step
changes, and define time-dependent forces and extra integrated quantity updates.
Those are the next implementation gates; none is waived by this linear demo.

## Validation and reproduction

The full CPU gate passes **7,109 tests, 621 skipped**, with zero failures/errors,
in 305.27 seconds (306.5 seconds with orchestration). Eight work-stealing workers
use one numerical thread each. Its source/tests/config fingerprint matches the
final tested implementation. Thirteen focused consumer cases also pass.

All 38 published Python blocks execute; 13 sketches remain labeled. Generated
catalog, repository links and documentation-site build pass. The scientific
gallery was rendered and visually inspected. Package runtime/exports are
unchanged, so the prior installed-wheel result remains the runtime baseline;
this demo is explicitly executed from the checkout rather than installed as a
new package API. The full gate retains the historical editable metadata 0.5.0;
source/package version remains 0.6.1.
No new runtime dependency, public scheme or package version is introduced.

```bash
python examples/structural_modes.py
MPLCONFIGDIR=/tmp/structural-matplotlib python benchmarks/structural_modes.py
python -m pytest -q tests/test_structural_demo.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/structural-full
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
python scripts/validate_repository.py
cd docs
npm run build
```

Evidence: [demo measurements](evidence/structural-demo.json),
[source audit](evidence/structural-source-audit.json),
[full CPU gate](evidence/structural-full.json),
[gallery](../assets/structural-modes.png).
