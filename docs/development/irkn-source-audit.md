# Phase 8 — improved Nyström source-path audit

The pinned IRKN3/IRKN4 candidates are **improved explicit history methods** for
fixed-step, velocity-independent second-order ODEs. They are not implicit RKN
methods. This audit records source-path failures before a production port; it
adds no numerical driver, public factory or fixed catalog entry.

## What executes independently

`scripts/coefficients/wave12/irkn_audit.jl` pins OrdinaryDiffEq 7.8.1 tree
bae9b8e5f1d00e51c554648e5094e215164a19f8, RKN 2.2.0. It evaluates the actual
IRKN coefficient constructors and cache struct definitions, then executes the
unchanged source bootstrap and warm step bodies. The stdlib harness mirrors
source allocation aliases and accepted uprev/uprev2/FSAL advancement. Broadcast
annotations become ordinary Julia broadcast, while muladd/thread annotations
are removed. It does not execute the complete SciML integrator or its initialization
probes, events and rejection machinery. Hashes cover tableaus, caches, steps,
algorithm declarations and order metadata.

The scalar consumers start at q=1, v=.7 and run to T=1 on 20/40/80/160 steps:

| Consumer | Acceleration | Exact displacement |
|---|---|---|
| Autonomous oscillator | -q | cos(t)+.7 sin(t) |
| Time-only forcing | t | 1+.7t+t³/6 |
| Forced oscillator | -q+.1t | cos(t)+.6 sin(t)+.1t |

There are 48 trajectories: two source methods, three consumers, four grids and
source/corrected-history diagnostic variants. Direct acceleration/velocity calls
and source-reported counters are recorded separately. Counter conventions are
not assumed to equal actual work. Refinement fits use the last adjacent pair
with both errors above 1e-11; near-floor fits are omitted and all raw errors remain.

## Findings and dispositions

| Pinned path | Finding | Disposition |
|---|---|---|
| IRKN3 constant cache | Step destructures k₂, but the source constant-cache type has no such field. The actual call raises FieldError before stepping. | Excluded as a usable pinned execution path |
| IRKN4 constant cache | The RKN source supplies its coefficient cache but no matching constant-cache step body; minimal-harness dispatch raises MethodError. | Excluded as a usable pinned execution path; no claim about every external fallback |
| IRKN3 mutable cache | Autonomous refinement is about order three; time-dependent full-startup trajectories converge at order two. | Deferred as an unrestricted nonautonomous port until startup/history is corrected and qualified |
| IRKN4 mutable cache | Autonomous refinement is about order four; time-dependent full-startup trajectories converge at order two. | Same source/startup qualification gate |

On q''=t at 160 steps, source IRKN3 has position/velocity errors
6.45e-6 / 6.51e-6; IRKN4 has 1.13e-5 / 1.14e-5. Observed velocity order is
2.000 for both. On the forced oscillator the position/velocity fits are about
2.012/2.002 for IRKN3 and 1.993/2.018 for IRKN4. On the autonomous oscillator,
IRKN3 fits 2.998/2.962, and IRKN4 fits 3.988/4.021. These are measured source-path
behaviors, not claims that the mathematical methods cannot achieve their orders.

## Isolating startup history

The source seed stores an old-position acceleration at t+c1*h in the slot later
used as the old physical acceleration. IRKN4 also seeds its c2 substage at the
c1 time and uses a different acceleration anchor in its first old substage.
These matter on a time-dependent problem even though an autonomous oscillator
can hide them.

A diagnostic control changes only the history after the first bootstrap: cache
physical old/end accelerations and evaluate old substages at their own positions
and c1/c2 times. Subsequent source kernels remain unchanged. Both controls then
solve q''=t to roundoff on all four grids, and forced-oscillator errors fall
substantially. IRKN3's forced component fits approach order three; IRKN4's small
corrected errors reach cancellation/roundoff regimes, so fine-grid rates are not
used to certify its formal order. This isolates the initial-history contribution
without claiming a complete repaired upstream implementation.

A package implementation still needs a derivation and actual nonlinear/time-
dependent consumer gates for the corrected seed, owned position/velocity/cache
quadratures, constant-step invalidation, UID/layout/value/time guards, lifecycle
and passive-quantity policy, state/parameter gradients and restart behavior.
None of those is supplied by a source-only bootstrap diagnostic. Do not copy the
broken constant-cache branch or quietly patch coefficients and call it source
equivalence. Retain autonomous usefulness as a candidate rather than excluding
the complete IRKN algorithms.

## Verification

The oracle asserts both constant-cache failures, second-order source forcing
refinement and the corrected-seed linear-force reference. A fresh execution
reproduces the tracked JSON. All analytic q/v references and stored errors can
also be checked independently without Julia. No runtime or test implementation
changes are made in this audit. The last numerical backend gate remains the
adaptive-Adams batch; the intervening SciPy declaration change has its own focused
pRRK/spatial and wheel-metadata checks.

[Tracked source results](../../scripts/coefficients/wave12/irkn_audit.json).

```bash
JULIA_DEPOT_PATH=/tmp/integrators-julia-oracle:$HOME/.julia julia --startup-file=no scripts/coefficients/wave12/irkn_audit.jl "$HOME/.julia/packages/OrdinaryDiffEq/OnlvZ"
python scripts/validate_repository.py
```
