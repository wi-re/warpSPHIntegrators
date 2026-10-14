# Phase 8 — MEBDF2 and CNLF2 backend

MEBDF2 is represented by a three-stage SDIRK tableau. Its stages solve backward
Euler at t+h, t+2h, then t+h with a modified base. Despite the name it is a
one-step method: there is no previous-step memory. The order-two map is stiffly
accurate and L-stable, with R(z)=(1−2z+z²/2)/(1−z)³. Callers must allow the
second stage's RHS evaluation beyond the accepted endpoint.

CNLF2 means Crank–Nicolson–Leapfrog. It solves
y[n+1]=y[n−1]+h(I[n−1]+I[n+1])+2hE[n]. Its IMEX-Euler cold start matches the
source. That single O(h²) startup defect preserves global order two when history
is threaded; restarting on every step gives first order. The two-step implicit
limit is A-stable but has a parasitic mode and no L-stability. With hλ=−100,
y0=1, the second accepted value is −99/101. Pure explicit leapfrog supports an
imaginary interval, not negative-real dissipative spectra. It is retained for
compatible split oscillatory problems, not as a general replacement for SBDF2.

The Julia oracle checks the pinned OrdinaryDiffEq 7.8.1 monorepo tree and executes
unchanged constant-cache step bodies from BDF 2.4.5 and IMEXMultistep 2.2.0.
Minimal cache/integrator scaffolding and independent dense Newton increment
solves replace the upstream solver. Autonomous and forced coupled nonlinear
steps match for MEBDF2 and for both startup/warm CNLF2. This is a step-formula
oracle, not an equivalence claim about the complete upstream controller.

CNLF2 owns input/history snapshots and validates method, layout, particle IDs,
timestep and time continuity before using memory. Incompatible history restarts
before consumption. Warm initialize hooks cannot change integrated values or
IDs. Changed RHS parameters require restart_history=True. Failed solves preserve
accepted memory. Endpoint RHS evaluation refreshes copied fields. Full quantities
use the same split formula. Tensor history has a portable weights-only restart;
in-memory trajectories retain gradients. Neither method provides an estimator.
The two new drivers default to linear tol=1e-12 and scaled Newton tol=1e-7,
avoiding an unattainable outer threshold paired with a looser absolute linear
solve. Explicit solver/solver_opts overrides remain available; older drivers'
default policies are unchanged.

The shared convergence helper now threads registered multistep history, so order
tests measure the actual backbone and its finite startup. The historical TVD
classifier still uses repeated cold calls: its CNLF2 verdict describes backward
Euler startup, not the warm leapfrog map. MEBDF2 fails its TVD sweep already at
CFL 0.25 despite its A/L stability; these are different properties.
Warm high-order implicit measurements use JVP/strict solve tolerances to separate
solver noise from discretization. Adams corrector grids include two finer halvings
to reach the asymptotic regime. Order assertions and production coefficients are
unchanged; the previous generic net measured DP5 startup repeatedly for these
history methods. Dedicated history tests remain in place too.

The final full CPU gate passes **7,349 tests, 639 skipped**, zero failures/errors,
in 312.94 seconds of pytest time (314.1 seconds including orchestration). Eight
workers use one numerical thread each; the stored fingerprint matches the final
implementation. All 18 focused cases and 979 convergence/driver cases pass.
All 552 copied-field cases pass after making the test instrumentation follow
the accepted owned system's lifecycle counter. A warm CNLF2 oscillator retains
two parity energy levels (1 and 1/1.04), separately from its dissipative cold start.
MEBDF2's beyond-endpoint stage is explicitly qualified in the stage-time net.

The isolated installed 0.6.1 wheel passes all imports/exports, 135 tensor drivers
and portable CNLF2 continuation. Numerical dependencies are shared; project
imports are isolated outside the checkout. All 41 runnable documentation blocks,
13 labeled sketches, catalog, repository links and the production site build pass.
Editable test metadata remains historical 0.5.0; source and wheel remain 0.6.1.

The 32 measured cases use N=64/1024, h=0.1/0.05 and nonlinear forced reaction or
rate-100 decay. All reach t=1 with tolerance-converged steps using JVP,
linear tol=1e-15 and scaled Newton tol=1e-8. A preliminary linear tol=1e-12
could plateau on the tiny stiff tail; the final options resolve that mismatch
without allowing stagnation. At N=1024,h=0.1 MEBDF2 uses 126 RHS calls and leaves
1.87e-13 decay error; SDIRK2 uses 86 calls and leaves 1.22e-7. At h=0.05 SDIRK2
also reaches a ~5e-15 tail at lower work. On smooth nonlinear forcing SDIRK2 is
both cheaper and more accurate. The MEBDF2 result is a specific large-step damping
trade, not general superiority. CNLF2 uses 53 calls but leaves 0.367 decay error
at h=0.1, compared with trapezoidal's 0.0173. Those persistent modes are expected.

Consumer evidence: [actual driver measurements](evidence/implicit-extensions.json).
Final gates: [full CPU report](evidence/implicit-extensions-full.json) and
[installed wheel](evidence/implicit-extensions-wheel.json).

```bash
julia --startup-file=no scripts/coefficients/wave10/implicit_extensions.jl /path/to/OrdinaryDiffEq/OnlvZ
python benchmarks/implicit_extensions.py
python -m pytest -q tests/test_implicit_extensions.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/phase8/implicit-final
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/phase8/implicit-wheel
```

Evidence: [source step oracle](../../scripts/coefficients/wave10/implicit_extensions.json).
