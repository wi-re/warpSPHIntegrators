# Backend backlog decisions and bounded experiments

This closes the Phase 8 **candidate review**, with explicit implemented,
equivalent, prototype, excluded and deferred outcomes. It does not claim that
all upstream algorithms/controllers were ported. The remaining production
projects below are backend work; none is waiting for a frontend or PESPH case.

[Individual source outcomes](../../scripts/coefficients/final-backlog/audit.json)
cover 35 candidates with source hashes, reasons and revisit conditions.
The [audit script](../../scripts/coefficients/final-backlog/audit.py) verifies
that each named declaration exists in the inspected pinned files. The
[actual-kernel oracle](../../scripts/coefficients/final-backlog/source_controls.jl)
and [results](../../scripts/coefficients/final-backlog/source_controls.json)
separately execute Nordsieck mathematics and IRKC. Source inspection alone is
not presented as a numerical equivalence gate.

## History and solver ownership

The actual Nordsieck Pascal translation, step rescaling and Adams correction
coefficients agree with the tensor representation at orders 1–12. Warm
polynomial exactness, order and owned failure tests pass. This is useful shared
mathematics for future native histories, not a complete AN5/JVODE startup or
controller. AN5 uses Tsitouras dense-interpolant startup and limited functional
iterations; JVODE adds order/wait/rewind state. Its BDF selector shares Adams
coefficient preparation and only has special BDF memory handling in order
changes. The warm Adams prototype is not labeled a BDF implementation.

**NordsieckBDF exists** in the pinned BDF package. It budgets nonlinear error
against local error and reuses/refreshed Jacobians through retry state. Native
propagated history is distinct from NDF's resampled differences. Keep it deferred
until the solver-cache refresh and error-budget contract is implemented and
measured against AdaptiveNDF. Exact source QNDF retry/discontinuity behavior and
full VCABM 1–12 startup/FSAL policy are likewise distinct from the already
implemented bounded package controllers. They retain separate source names and
explicit revisit conditions. Merely raising the Adams maximum would be unsafe:
DP5 startup limits high-order accuracy and the current monomial reconstruction
has condition number about 2.46e14 at order twelve on a uniform grid. This
conditioning observation is not a failure claim about source divided differences.

Approximate matrix factorization (AMF) now has a bounded ordered-factor probe.
For two factors its difference from I-shift*(J1+J2) is exactly
shift²*J1*J2. Factor order matters when operators do not commute. Replacing the
Rosenbrock matrix changes the method; the full-JVP backend does not silently do
so. Structured update/solve/transpose and gradient ownership need their own
contract before production W-method/order gates. Tsit5DA/hybrid DAE methods
remain excluded from the ordinary ODE catalog for their mass-matrix/problem
contract, rather than being confused with the existing Tsitouras explicit pair.

## New tensor experiments

The [bounded functions](../../src/warpSPHIntegrators/experiments/tensor.py) are
outside scheme lookup and the full-state `solve` protocol. They accept tensors
and f(t,y); they do not imply SPH lifecycle, quantity rerun or rollback support.

| Experiment | Qualified behavior | Remaining boundary |
|---|---|---|
| Frozen-Jacobian Euler/midpoint extrapolation | Nonlinear time order 2/4/6 and complete state/parameter AD; source-style linear corrections | Dense <=64; no source adaptive order/work controller, divergence policy or full-state quadrature |
| RK4 interaction picture | Fourth order and exact linear limit | Inverse exponentials overflow for stiff diffusion; source RKIP is not Rosenbrock |
| Two-step ETD2 | Exponential-Euler start, second-order warm nonlinear trajectory, parameter AD | Caller owns fixed-h nonlinear history; this is distinct from one-step ETDRK2 |
| Matrix ODE | Exponential midpoint 2, commutator Magnus 4, Cayley midpoint 2 on rotating noncommuting operators | Dense <=64, no source-name equivalence or matrix-free operator protocol |
| Multirate slow/fast forcing | Noncommuting/time-dependent order two, two slow +4*r fast calls | Fast microstep stability, macro interpolation and future rollback/adjoints |
| Taylor total-JVP | Order 3/4, tensor time, full state/parameter gradients | Six/ten RHS invocations per step plus nested AD; backend must support higher derivatives |
| Carpenter–Kennedy 2N54 | Native two-register map agrees with qualified expansion | Tensor no-grad only; full-stage lifecycle/finalize/rerun cannot discard stages |

The actual pinned **IRKC** constant-cache body with independent dense nonlinear
solves reproduces the translation but measures orders .961, 1.000 and 1.046 at
degrees 2/4/10 on the retained nonlinear/time split control. Even a simple linear
autonomous split shows the same order-one behavior in the translation. Its
nonlinear-failure checks are commented out in source. The prototype raises on
failed solves and remains an audit specimen; no second-order factory is added.
An upstream correction or separately derived recurrence needs a fresh full gate.

## Cost and allocation evidence

[Twenty-two experiments](evidence/backend-experiments.json) come from
[the reproducible benchmark](../../benchmarks/backend_experiments.py). At 1,024
entries over sixteen steps Taylor3/4 use 96/160 RHS invocations and take
.036/.144 seconds, with errors 4.13e-6/5.18e-8. Nested-JVP cost does not establish
a reason to add adaptive Taylor order now.

At 4,096 entries and ratio sixteen, the FFT-slow/local-fast multirate example
uses 32 slow and 1,024 fast calls in .0098 seconds, versus 1,024/1,024 and .0396
seconds for all-microstep RK4. Errors are 1.20e-4 versus 3.95e-12. This measures
saved expensive calls and the accuracy tradeoff; it is not an equal-error speedup.

For 262,144 entries, the CPU profiler's tracked allocation high water is
18,874,368 bytes for full-stage ExpandedRK and 12,582,912 for the native tensor
recurrence. Inputs, allocator reservation, RSS, profiler and external memory are
excluded. A tensor-only no-grad path can save memory; the existing SPH stage
contract cannot claim that saving. Keep this prototype for the separate native
backend project instead of silently weakening finalize/AD semantics.

The review therefore finishes with bounded useful backends and measured
prototypes, plus explicit future controller/operator projects. The following
package/API refactor is not held up by pretending these deferred source policies
are complete implementations.

## Completed batch gates

The full local CPU gate passes **8,778 tests, 639 skipped, 157 warnings**
in 329.77 seconds with eight workers and one numerical thread per worker.
The [full report](evidence/exponential-backlog-full.json) fingerprint matches
the runtime/tests/configuration. The [installed-wheel gate](evidence/exponential-backlog-wheel.json)
passes all modules and adds exponential/Taylor smokes outside the checkout.
All 58 published runnable blocks execute; 13 sketches remain labeled.
Catalog generation, repository links/anchors and the production site build pass.
No new CUDA coverage or source-runtime equivalence is inferred from CPU gates.
