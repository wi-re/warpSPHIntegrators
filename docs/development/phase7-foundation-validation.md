# Phase 7 — known toy splitting consumers, 14 October 2026

Phase 7 completes the package-level splitting scope on two executable consumers:
the damped oscillator with its known solution and periodic viscous Burgers with
a controlled unsplit RK4 reference. The user chose these simple validation targets
while PESPH is being built. Its missing application implementation therefore does
not block this phase; production SPH thermodynamic/energy claims remain separate.
Implementation commit: `78d6a25`. Reports were captured before committing; their
source fingerprint identifies the tested implementation independently of HEAD.

## Contracts and implementation

`ConservativeDissipativeRHS` declares both autonomous physical parts and supplies
their sum for ordinary integrators. Nested `SemilinearRHS` retains its capabilities
through the tensor adapter. `check_contracts` verifies additivity; structural
checks require zero dissipative position rate and, for geometric sub-integrators,
probe conservative force dependence on velocity, drift and extra integrated fields.
Perturbed velocities catch the cubic-velocity-at-zero false positive. An evolving
thermal quantity prevents certification even when force is velocity-independent.
These finite local probes do not prove a global Hamiltonian or a variational SPH force.

Parameterized `compose`, `LieTrotter` and `Strang` accept existing one-step schemes.
Each physical part advances one outer step in total; autonomous component clocks
are reset rather than summed. Outer initialize/finalize run once. Inner preprocess,
postprocess and copied fields remain active, with optional initialize_substep.
A custom finalize must explicitly support grouped component results and coefficients;
compatible-energy Butcher-weight finalizers cannot silently consume those groups.
The caller and each recorded substep remain isolated. Failed solves propagate
without finalization or acceptance. Precision floors are handled by the runner's
policy after all subflows finish, so opt-in cannot accept an unfinished split.

The fixed runner accepts bound recipes, output, accepted callbacks and in-memory
restart; consumers resupply the same parameters/operator/options. Adaptive split
estimates, dense output, events, FSAL and multistep components are unsupported.
Implicit D uses the existing JFNK/IFT machinery. `LinearExponentialFlow` supplies a
dense small-system Torch matrix exponential with state and matrix-parameter gradients,
checks declared linear action and preserves frozen mechanical configuration. Sparse
or matrix-free exponential application assembly is not inferred from this dense path.

`Yoshida4` uses [Yoshida's fourth-order coefficients, Eq. 2.11](https://doi.org/10.1016/0375-9601(90)90092-3).
It requires a supported symmetric conservative method and a static exact linear D
with nonpositive symmetric matrix part. Adjacent exact D flows are combined.
Every negative D step must satisfy `abs(tau)*norm(D,2) <= negative_step_limit`
(default 0.5); strong damping is rejected before initialization. This bounds
anti-diffusive amplification, not arbitrary composite instability. It does not
support generic nonlinear negative D flows or promise unconditional stiff stability.

No fixed registrations change: the catalog remains 137. Existing geometric one-step
schemes fit this interface; future general RKN/Composite catalog work has a driver
contract, but no new RKN family or arbitrary map/extrapolation semantics are claimed.

## Measured behavior

CPU float64, one numerical thread, Python 3.13.14, Torch 2.13.0+cu130 and the
Intel Core Ultra 9 285K environment from Phase 6. Fixed h refinement, completed
endpoints and solver tolerance are recorded in the linked benchmark evidence.

| Consumer/method | Expected order | Last measured order |
|---|---:|---:|
| Oscillator Lie–Trotter, Verlet/midpoint | 1 | 1.117 |
| Oscillator Strang, Verlet/exact damping | 2 | 2.003 |
| Oscillator restricted Yoshida4, Verlet/exact damping | 4 | 4.013 |
| Burgers Lie–Trotter, RK4/midpoint | 1 | 1.011 |
| Burgers Strang, RK4/midpoint | 2 | 2.000 |
| Burgers Strang, RK4/exact diffusion | 2 | 2.000 |
| Burgers Strang, RK4/backward Euler control | 1 | 0.996 |

Burgers uses n=32, nu=0.1, endpoint 0.1, reference unsplit RK4 h=0.0001 and
h=0.01/0.005/0.0025. Newton's scaled threshold 1e-7 and GMRES 1e-12 make
solver error smaller than discretization errors without precision-floor endpoint
truncation. Accuracy measurements require successful completed runs. Mass agrees
to numerical roundoff. Separate nonlinear/linear implicit and exponential tests
preserve state and parameter gradients through their respective solver paths.

For `q'=v, v'=-4q-0.5v`, Strang/Verlet/exact D has determinant `exp(-0.5h)`
to roundoff at h=0.5 (defect 1.1e-16). Yoshida4 agrees within 2.1e-10;
unsplit RK4 has a 0.00281 defect at that same step. This is the constant-damping
conformal diagnostic, not symplecticity of a dissipative composite. The energy
change plus trapezoidal viscous loss has defects 0.02060/0.005161/0.001291 at
h=0.1/0.05/0.025, giving order 1.999. Backward/forward weak-damping composition
also returns the oscillator to solver/roundoff accuracy.

The viscous-CFL case uses n=64, nu=1, endpoint 0.05 and h=0.001/0.005/0.01:
8.192/40.96/81.92 times the explicit diffusion step bound. Implicit midpoint
and exact diffusion remain bounded; the explicit control is unstable. Finite
numerical completion is recorded separately from boundedness. This demonstrates
diffusion stability, not removal of transport CFL restrictions or accuracy at any h.

Fixed/static exact-flow merging uses 21 D flows instead of 40 over 20 Strang
steps, with a final difference 2.22e-15. This whole-interval benchmark changes
lifecycle boundaries, so it is not enabled as a runner optimization. Variable
steps, nonlinear approximations, synchronized samples and callbacks need separate
staggered-state ownership before merging could be used transparently.

## Validation and remaining application work

The final full CPU gate passes **6,993 tests, 609 skipped** in 302.90 seconds
(304.26 seconds including orchestration), with eight work-stealing workers and
one numerical thread each. Its source/tests/config fingerprint matches the final
implementation. No source changes follow that gate.

The isolated installed-wheel gate passes all modules/exports, all 137 catalog
registrations and actual Strang/Yoshida4 runs outside the checkout. The local
wheel environment shares numerical dependencies; fresh dependency installation
remains covered by CI. Its installed version is 0.6.1. The editable test environment
still reports historical installed metadata 0.5.0, as in prior phase records;
the source version and isolated-wheel version are both 0.6.1.

All 37 published Python blocks execute; 13 caller-defined sketches remain labeled.
Generated catalog, repository links and the production documentation site pass.
The 21 focused splitting cases pass, including order controls, structural failures,
lifecycle/ownership, rollback, restart, parameter gradients, nested autonomy,
bounded negative steps, contraction and energy-budget convergence.

PESPH-A and a comparable ordinary-integrator baseline remain later application
work. The audit in the [spatial follow-up](spatial-followup-validation.md) records
the missing frontend prototype. No downstream repositories were changed. Pairwise
compatible-energy work maps, topology/frozen-operator ownership and actual SPH
conservation must be tested there; constant-damping toy evidence cannot certify them.
Suzuki comparisons, general RKN/catalog expansion, sparse exponential flows and
adaptive splitting remain explicit follow-ons rather than hidden scope here.

```bash
python benchmarks/splitting_validation.py --output-dir .tmp/phase7/benchmarks-final
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase7/full-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase7/wheel-final
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
python scripts/validate_repository.py
cd docs
npm run build
```

Evidence: [toy measurements](evidence/phase7-splitting.json),
[full CPU gate](evidence/phase7-full.json),
[installed wheel](evidence/phase7-wheel.json).
