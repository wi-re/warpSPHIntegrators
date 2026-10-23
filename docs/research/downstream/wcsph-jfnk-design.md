# WCSPH acoustic JFNK integration design

Reconciled 23 October 2026 from the August–September design and experiments.
The library supplies JFNK, implicit differentiation and the relevant stiff
families. Open acoustic integration, boundary and surface work belongs to
warpSPH/warpSPHCore and requires a fresh status check there before scheduling.
The [dated experiment ladder](wcsph-jfnk-2026-08-24-to-09-04.md) preserves A/B and
E1–E1.10 results; [the library queue](../../../REPOSITORY_PLAN.md) owns only package
work. Historical downstream paths below are investigation pointers.

## Ownership and motivation

The target is implicit treatment of continuity, EOS and pressure force so the
acoustic CFL need not set the step. Advective, viscous, acceleration, accuracy
and nonlinear-solve limits still apply. Higher sound speed and a larger stable
step are hypotheses to measure; the historical zero-dissipation acoustic-core
result did not establish an end-to-end speedup or stability of forced turbulence.
The recorded DFSPH correction and neighbor-list ablations remain separate
application findings.

| Work | Disposition | Activation and acceptance |
|---|---|---|
| Phase A library solver/wave bridge | Completed within recorded scope | JFNK/GMRES, FD/JVP actions and wave-reference agreement have dated evidence. Current library solver defaults are described in maintained docs, not the original opt-in narrative. |
| Phase B and E1–E1.10 | Historical downstream completion | Reproduce the relevant current case before adopting old timestep or throughput claims. Preserve instability, cost and negative findings. |
| Phase C acoustic pressure JVP | Downstream, status unverified | Inspect current operators, derive missing actions and validate actual residual/JVP composition. |
| Phase D real periodic WCSPH run | Downstream, status unverified | Compare accuracy, stability and measured work at equal physical time against the current explicit baseline. |
| E2 mDBC | Downstream, historically scoped | Establish ghost geometry/state/configuration and the boundary-operator derivative contract. |
| E3 free surface | Downstream, historically scoped | Establish surface state/detection, active branches and pressure derivative/FD policy. |

## Acoustic derivative contract

The historical code inspection decomposed the proposed subsystem as follows:

| Piece | Historical observation | Required current verification |
|---|---|---|
| Continuity | Divergence was among wrapped value-JVP operators. | Check the actual continuity composition, tangents and configured operator path. |
| EOS | Pointwise Torch formulas needed no additional neighbor-kernel JVP. | Confirm current EOS implementation, density tangent and branch behavior. |
| Pressure force | A custom surface-aware force was the identified gap; a generic Gradient action did not replace it. | Inspect the real kernel/variant, density/pressure/geometry dependencies and source registration before deriving an action. |

The fixed-density/geometry symmetric pressure sum is linear in pressure, but
the full acoustic state also varies density, EOS and potentially correction
weights. Relaunching on pressure tangents alone does not establish the complete
state JVP. Specify which quantities are frozen by the Newton residual and include
every varying dependency. Validate directional products against independent
finite differences and reverse-mode/gradcheck where supported, then register the
action through the owning Core operator mechanism. An exact-JVP claim applies
only when every operation in the chosen residual has that action.

The old pointers are `schemes/deltaSPH.py`, `modules/momentum/inconsistent.py`,
`modules/eos/weaklyCompressible.py` and `modules/pressure/wp_surfaceAware.py` in
warpSPH, with derivative registrations in warpSPHCore. The old Antuono branch
used free-surface state while the conservative pressure variant did not; that
distinction must be checked against current source rather than assumed.

Numerical density/velocity diffusion and physical viscosity are distinct terms.
Setting densityDelta or inviscidAlpha to zero in the old experiment did not
authorize removing physical viscosity from a Taylor–Green decay reference.
Check the meanings of viscidNu and stabilization coefficients and measure whether
implicit acoustic damping changes any empirical stabilization floor.

## Periodic run and cost gate

The original target was the existing periodic weakly-compressible Taylor–Green
case, with kinetic-energy decay exp(-4*nu*k²*t) in its stated convention.
Confirm that convention and current case before reusing the formula. Wire the
declared acoustic residual through a generic implicit scheme, first using the
known Backward Euler route, then qualify higher-order or IMEX choices separately.
Keep any explicit remainder and its stability restrictions visible.

Compare complete endpoints over the same physical interval: density/pressure and
velocity accuracy, energy decay, failed solves/retries, true residuals, RHS calls,
matvec/JVP work, setup cost, wall time and memory. Compare both like-for-like
steps and the best stable/accurate explicit trajectory. A larger timestep with
hundreds of times greater step cost need not improve total work.

The historical wave test compared hand-eliminated CG with generic DIRK/JFNK
FD and JVP paths. Preserve that independent reference and the current frontend
physics suite. The [0.7 migration record](../../development/package-refactor-validation.md)
already identifies a separate pre-existing FD/JVP tolerance failure and unavailable
CUDA checks; do not relabel them as successful acoustic integration evidence.

## Boundaries and free surface

For mDBC, the historical simple velocity policies used directional
Interpolate/FluidToGhost plus pointwise algebra. That was a plausible reuse of
existing JVP machinery, not a tested derivative for every mode. The density
path used moving-least-squares reconstruction with local solves and neighbor
threshold/fallback branches, a distinct derivative problem. Gravity corrections,
ghost indices/offsets, materials/kinds and boundary generation must enter the
state/configuration contract. Start from a simple wall and a declared derivative
or FD policy before promoting general boundary support.

Free-surface integration additionally needs detection schemes, normals, dilation
and active surface branches. Inspect existing surface gradcheck tools before
inventing a new derivative path. A fixed-topology/action policy inside Newton
must still be consistent with boundary/surface updates between accepted steps.
The [E2/E3 historical scoping](wcsph-jfnk-2026-08-24-to-09-04.md#e2--mdbc-bounded-domain--scoped-not-started)
retains operator names, configuration and geometry details. These are distinct
projects, not automatic extensions of the periodic acoustic core.

## Scope exclusions and package handoff

Global operator-JVP coverage and Hessian-based Newton are not prerequisites for
a specific inexact-Newton acoustic residual. Unsupported higher derivatives and
surface switches need explicit scope rather than a claim that all Warp AD is
absent. The old exclusions of library BDF/FIRK/IMEX and implicit differentiation
are historical: those package capabilities now exist, but do not by themselves
complete this application.

A package handoff should identify a missing solver/state/operator capability,
its minimal reproducer, source/derivative contract and cost target. Production
SPH wiring, mDBC, surface detection and application scaling stay in their owners.
