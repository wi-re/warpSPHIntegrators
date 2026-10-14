# Phase 8 — NDF difference-history backend

`NDF(order=1..5)` is a parameterized fixed-target-order backend outside the
142-method fixed catalog. It carries owned backward differences and uses the
pinned quasi-constant QNDF warm kernel, R*U polynomial history transform and
local defect. The source pin is OrdinaryDiffEq 7.8.1, tree
bae9b8e5f1d00e51c554648e5094e215164a19f8, BDF 2.4.5. The stdlib Julia oracle
executes QNDFConstantCache's warm step and unchanged U!, calc_R!, update_D!
helpers with independent dense Newton scaffolding. Thirty records cover
orders 1–5, ratios 0.5/1/1.5 and autonomous/forced nonlinear two-field systems.
State, defect, transformed and accepted differences agree to 2e-13; rerunning
the promoted script reproduces the tracked output. Source hashes are in its JSON.
This establishes warm-kernel equivalence, not full upstream integrator equivalence.

Startup uses p order-preserving implicit RK steps: TR-BDF2 for p=1/2,
ESDIRK324L2SA for p=3, KenCarp4 for p=4 and KenCarp5 for p=5. Their per-result
estimator exponents are 2/3/3/4/5; warm exponents are p+1. Startup values are
interpolated onto a virtual uniform grid before forming differences, in increment
form to preserve large zero-rate offsets exactly. NDF5 has kappa=0 and its
constant-step main map equals BDF5. It is not registered as a duplicate.
The accepted history, startup and error/controller semantics remain distinct.

Arbitrary step changes need care. For p=5, zero-RHS perturbation propagation
with alternating ratios 5/3 and 3/5 has two-step spectral radius 1.1492012386.
It can amplify a computational mode despite ratios below two. Holding six
equal steps between changes gives radius 0.022821293 for this counterexample.
The backend holds at least p+1 accepted equal-size steps and uses a 1–1.5 growth
deadband; rejected trials can shrink immediately. Rapid changes from direct
callers restart through the implicit starter before consuming history. This is
a bounded engineering policy, not a general variable-grid stability proof or
the upstream adaptive-order controller. Fixed-step characteristic-root tests
cover negative-real decay for all orders and show orders 3–5 are not fully A-stable.

The 76 focused tests include source maps, constant/blockwise-grid convergence,
defect exponents on exact polynomial histories, tolerance refinement/rejection,
owned restart and gradient propagation, actual RHS work counts, protected
beta*h preconditioner context, zero-rate offsets, full-state coupled quantities,
endpoint copied fields, lifecycle counters and cache invalidation. Invalid
state values/layout/IDs/time/ratios/context restart before use. Changed RHS
parameters require explicit restart_history. Failed trials retain accepted
memory; warm initialize and finalize cannot project integrated values or IDs.

NDFQuadrature exposes the full-state affine step. Bootstrap reuses the starter
weights with one zero-weight endpoint refresh; warm steps use the history
combination and one beta-weighted endpoint derivative. Custom finalize opts
into this contract explicitly. DIRK's internal lifecycle suppression lets the
outer NDF driver initialize/finalize exactly once. Default DIRK behavior is
unchanged. Portable tensor checkpoints allow cache-only null-stage entries and
retain difference tensors/counters in the existing weights-only context format.
Old tensor-stage payloads still load. In-memory restart retains gradients.

The known-consumer benchmark contains 128 fixed-grid cases: N=64/4096,
nonlinear manufactured reaction with exact exp(-rate*t) and stiff linear decay
with rates up to 10,000, constant/blockwise steps and 24/48 steps to T=1.
NDF1–5 are compared with BDF3/BDF5/KenCarp5 using JVP and exact diagonal forward
preconditioning. Every case is finite with tolerance-converged solves.
At N=4096, 48 blockwise reaction steps, NDF3 uses 272 RHS entries and has
5.86e-6 maximum error; BDF3 uses 496 and has 8.98e-6 error. NDF5 uses 400 and
has 1.68e-7 error; BDF5 uses 1,472 and has 4.95e-8 error. NDF retains its five
startup steps while fixed BDF5 needs 32 across grid changes. KenCarp5 is much
more accurate on this grid but uses 2,064 RHS entries. These are equal-grid
tradeoffs, not equal-accuracy speed claims. Diagonal preconditioning is a toy
operator contract, not proof of spatial SPH scaling.

Fifteen adaptive cases use a known nonlinear nonautonomous scalar solution at
rtol 1e-3/1e-5/1e-7. Error decreases for every order as tolerance tightens.
The estimates control local defects; they do not bound final global error.
Short high-order runs can finish entirely in implicit RK startup: all three
order-five T=2 cases do so. The separate T=4 controller test requires warm
steps and verifies held accepted steps. Benchmark records expose warm counts
so startup cannot masquerade as multistep efficiency.

Adaptive order, upstream rejection/order selection, Nordsieck/JVODE drivers,
DAEs/mass matrices and dense interpolation remain separate extensions.

The final full CPU gate passes **7,630 tests, 639 skipped**, zero failures/errors,
in 313.74 seconds of pytest time (314.9 seconds with orchestration). Eight workers
use one numerical thread each; the stored fingerprint matches final source,
tests and shard/config files. The existing implicit/runner subset has 531 passing
cases. Repository links and disposition paths pass.

The installed 0.6.1 wheel passes all imports/exports, 135 fixed tensor drivers
and a warm cache-only portable NDF restart outside the checkout with isolated
project imports and shared numerical dependencies. All 44 published Python
blocks execute, with 13 labeled sketches. Catalog and production site build pass.
Editable test metadata is historical 0.5.0; source/wheel versions are 0.6.1.

Evidence: [consumer measurements](evidence/ndf-backend.json),
[full CPU gate](evidence/ndf-backend-full.json),
[installed wheel](evidence/ndf-backend-wheel.json).

Reproduce with the warp Python environment:

```bash
python -m pytest -q tests/test_ndf.py
python benchmarks/ndf_backend.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/ndf-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/ndf-wheel
JULIA_DEPOT_PATH=/tmp/integrators-julia-oracle:$HOME/.julia julia --startup-file=no scripts/coefficients/wave10/ndf.jl "$HOME/.julia/packages/OrdinaryDiffEq/OnlvZ"
```
