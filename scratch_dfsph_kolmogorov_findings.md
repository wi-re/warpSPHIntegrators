# DFSPH (incompressible) vs. the JFNK/compressible-acoustic-core Kolmogorov study

Scratch findings file for a supervising session to fold into `JFNK_PLAN.md`.
Answers the question the project owner asked directly: does this codebase's
existing, already-registered incompressible SPH solver (`scheme=
'divergenceFree'`, DFSPH, `warpSPH/src/warpSPH/schemes/dfsph.py`) already
deliver -- for free -- some or all of what the JFNK effort (Phases B/E1/
E1.5/E1.6/E1.7 on the *compressible* acoustic core) is chasing, on the same
2D forced-Kolmogorov-flow shear-instability problem?

All numbers below are from a new standalone script,
`warpSPH/scripts/probe_kolmogorovIncompressible.py`, run on this machine's
GPU (`NVIDIA RTX PRO 6000 Blackwell`), `nx=24` and `nx=128`, `L=2.0`, `k=4`,
`xi=1.0` (matching `cases/kolmogorov.py`'s own defaults and `JFNK_PLAN.md`
E1.6/E1.7's own probes). `cases/kolmogorov.py` and `cases/tgv.py` were **not
modified** -- see "What was built" below.

## Answer, up front

**Both things are true, and they are not in tension.** DFSPH has no acoustic
CFL by construction (confirmed by reading the code, not just by outcome --
see "no acoustic term" below) and, on this probe, it also does not need
explicit physical viscosity to keep the Kolmogorov shear instability's
turbulent cascade *statistically bounded* -- a genuinely different answer
than the compressible+JFNK story, where `JFNK_PLAN.md` E1.5 found a real
`nu>=0.03` stability threshold below which every run eventually diverges.
But DFSPH is **not free of a timestep constraint or of failure modes** --
it pays elsewhere:

1. It still needs a properly bounded `dt` (an advective-CFL-like ceiling),
   just not one scaled by sound speed. Removing that ceiling reproduces the
   same kind of explicit-integrator blow-up JFNK exists to avoid on the
   compressible side, just via a different route (large steps at the
   near-zero-velocity spin-up transient, not an acoustic wave).
2. At the larger, more turbulent `nx=128` scale, a run **with** physical
   viscosity (`nu` from `alpha=0.01`, the real case's own default) diverged
   at step 720 while the **zero-viscosity** run at the same resolution
   stayed bounded for the full 1000-step budget -- the opposite of what the
   compressible study found, and traced (tentatively) to particle-disorder/
   density-void artifacts rather than a viscous-shear mechanism (see below).
3. Its cost is a **fixed** per-step tax, not a convergence-driven one: the
   shipped default relaxed-Jacobi solvers pin at their `maxIterations` caps
   (32 for the divergence-free projection, 64 for the constant-density
   correction in `finalize`) on every single step of every run in this
   probe -- i.e. the historical default configuration does not actually
   converge to its own tolerance on this problem, it just always spends the
   full budget. That is a different cost *character* than JFNK/GMRES's
   state-dependent iteration count (E1.6: iteration cost scales with local
   turbulence intensity and `dt` multiplier), not obviously cheaper or more
   expensive in an apples-to-apples sense, but structurally different.

So: DFSPH sidesteps the *acoustic* stiffness problem entirely, by
construction -- that part of the JFNK payoff really is "free" with existing,
already-registered code, for anyone who can accept an incompressible
formulation. It does **not** obviously sidestep the *physical* shear
instability's practical consequences at production scale (nx=128) the way
the headline "no acoustic CFL" framing might suggest -- it just fails
differently (particle-disorder/density-void territory, not viscous
blow-up), and its own pressure-solve cost is non-trivial and non-adaptive
under the shipped defaults.

## What was built (and what was *not* touched)

New file: `warpSPH/scripts/probe_kolmogorovIncompressible.py`. Bypasses the
`Case`/CLI machinery (matching `sample/acousticCore.py`'s own precedent),
building the system directly from:

- `warpSPH.sample.weaklyCompressible.setupBasicWeaklyCompressibleInitialState`
  -- the exact sampling `cases/tgv.py`'s own `buildSystem` calls, for the
  same `scheme='divergenceFree'`.
- The Kolmogorov `v_x = xi*sin(k*pi*y)` forcing closure, re-derived inline
  (it is a nested closure inside `cases/kolmogorov.py`'s own
  `initialConditions`, not an importable standalone function). The
  Perlin-noise symmetry-breaking term is dropped, matching `JFNK_PLAN.md`
  Phase E1's own precedent -- a small position jitter (`jitter=0.01`,
  `shuffleParticles(..., 0, jitterAmount=jitter)`, the same helper
  `cases/tgv.py`'s own relax step uses) is enough to seed the instability
  here too.
- `IncompressibleSPHConfig.boundaryConditions` (`BoundaryCondition(type=
  dynamic, sdf=..., forcingFunctions=[forcing])`), the same field
  `WeaklyCompressibleSPHConfig` uses, routed through the same
  `modules/boundaryConditions/bcs.py:computeForcing` both schemes share.
  The `sdf` here is a trivial "everywhere inside" function (`d=-1`
  constant) rather than importing `cases/weaklyCompressible.py`'s
  `domainFluidSdf`, to keep the script fully self-contained.

`cases/kolmogorov.py` and `cases/tgv.py` were **not edited**. Nothing else
in `warpSPH/src` was changed. This is the only new/modified file in
`warpSPH/`.

## Troubleshooting narrative (found running this, not guessed)

1. **Device bug in the script's own setup, not in production code.**
   `buildDomainDescription(l, dim, periodic, device, dtype)`'s own default
   for `device` is `'cpu'`; the first draft of this script passed
   `device=None` through explicitly (mirroring `probe_kolmogorovContinuation.
   py`'s pattern for its own domain construction), which *overrides* that
   default with literal `None` rather than resolving it, since Python
   default-argument resolution only applies when the argument is *omitted*.
   Every kernel in the first smoke test compiled and ran "on device 'cpu'"
   despite a GPU being present (2-8s one-time compile per kernel, ~900ms/step
   steady state at `nx=24`, 576 particles). Fixed by resolving
   `device = cuda:0 if available else cpu` explicitly before calling
   `buildDomainDescription`, matching `warpSPH.runner.runner.buildContext`'s
   own convention. After the fix: same run, ~60-100ms/step steady state on
   GPU (dominated by kernel-launch overhead at this tiny particle count, not
   compute).

2. **`stepResult.stages[0].aux`'s shape, and a real dead-mutation bug found
   in existing (untouched) production code while trying to read solver
   iteration counts out of it.** `dfsph_step` returns `(update, adjacency,
   currentState, (errors, pressures))`; `warpSPHIntegrators.euler.
   integrateSemiImplicitEuler` binds everything after `update` into `r1`,
   which becomes `StageResult.aux` -- confirmed by direct inspection to be a
   3-tuple `(adjacency, currentState, (errors, pressures))`, i.e. exactly
   `solveDivergenceFree`'s own iteration history from inside `dfsph_step`
   itself. Separately, `IncompressibleSystem.finalize`
   (`systems/incompressible.py`) runs a *second* pressure solve
   (`solveIncompressible`, the constant-density position correction) and
   appears to try to surface its own iteration count by reassigning
   `returnValues[-1] = (..., (errors_incomp, pressures_incomp))` -- but
   `returnValues` there is `[r1]`, and reassigning `returnValues[-1]`
   rebinds *the list slot* to a new tuple object; it does not, and cannot,
   mutate the original `r1` tuple that `stages[0].aux` already holds a
   reference to (tuples are immutable). So that second solve's iteration
   count is silently unreachable through the integrator's own return value
   -- the only place it is ever surfaced is a `print()` gated on
   `verbose=True` inside `finalize` itself. This is a real, verifiable
   latent bug in existing (untouched) production code, not something this
   probe caused -- noted here for the supervising session's awareness, not
   fixed (out of scope: `cases/tgv.py`/`schemes/dfsph.py`/
   `systems/incompressible.py` are all off-limits per this task's own
   instructions, and the bug is cosmetic -- it only affects introspection,
   not the actual physics). Worked around in the probe script by redirecting
   stdout and regex-parsing the `verbose=True` print instead of patching
   production code.

3. **A naive "advective CFL from current `vMax`" `dt` formula is degenerate
   at `v~0`, and exposes a real DFSPH stability wall when its `maxDt` safety
   ceiling is removed.** See finding #2 below -- this is a genuine physics
   finding, not just a script bug, but it was *found* by first writing a
   naive `dt = cflFactor*h/max(vMax, floor)` timestep picker, discovering it
   silently relied on a `maxDt=0.1` cap to stay sane during the from-rest
   spin-up transient, and only then testing what happens with that cap
   relaxed.

## Findings, with numbers

### 1. `nx=24` toy scale: DFSPH stays bounded at zero dissipation, unlike the compressible core

Natural CFL-respecting `dt` (`cflFactor=0.3`, `maxDt=0.1`, advective-only
since `nu=0` disables the viscous term), `xi=1.0`, `k=4`, `jitter=0.01`,
576 particles:

| `nu` | steps run | outcome | settled KE | settled `vMax` | settled `rhoStd` |
|---|---|---|---|---|---|
| 0.0 | 900 (`t=81.4s`) | **bounded** | ~0.45-0.94 (fluctuating) | ~0.9-1.4 | ~0.013-0.038 |
| 0.001 | 300 (`t=25.7s`) | bounded | ~0.45-0.56 | ~1.1-1.3 | ~0.017-0.028 |
| 0.00417 (`alphaToNu(0.01, c_s=10, h)`) | 300 (`t=27.3s`) | bounded | ~0.38-0.82 | ~0.81-1.4 | ~0.019-0.034 |
| 0.005 | 300 (`t=28.3s`) | bounded | ~0.39-0.50 | ~0.95-1.1 | ~0.030 |
| 0.01 | 300 (`t=29.6s`) | bounded | ~0.31-0.34 | ~0.69-0.82 | ~0.024-0.028 |
| 0.03 | 300 (`t=30.0s`) | bounded | ~0.25 | ~0.57-0.60 | ~0.007 |

Every value tested, **including exactly zero**, stays in a statistically
steady, bounded, fluctuating turbulent-like state -- no divergence at any
`nu`, and the `nu=0` run was extended to 900 steps (`t~81s` simulated,
~9x longer than the longest bounded compressible-core run in `JFNK_PLAN.md`
E1.5's own nu-sweep) specifically to rule out "just hasn't diverged yet."
The expected physical trend is visible too (higher `nu` -> lower saturated
KE/`vMax`/`rhoStd`, more damped cascade), so this is not a case of the
forcing/instability failing to engage -- the flow visibly transitions and
saturates, it just never blows up.

This directly contradicts a naive expectation that DFSPH would need the
same kind of `nu>=0.03`-style threshold `JFNK_PLAN.md` E1.5 found for the
compressible+JFNK core. The most likely explanation, from reading the code
(not proven by a targeted ablation, stated as a hypothesis): DFSPH's
default `solverConfig.integrateRho=False` means density is **recomputed
from a plain SPH summation every step**, not integrated via a stiff
continuity equation the way the compressible core's `drho/dt` is -- this
recomputation is itself a strong implicit regularizer (no memory of
accumulated high-frequency density error survives from one step to the
next), on top of whatever numerical damping the relaxed-Jacobi projection
itself contributes. **This changes what "zero dissipation" even means for
this scheme**, exactly as the task asked to check explicitly: `nu=0` here
still means "no *explicit* physical/artificial viscosity term," but it
does not mean "no dissipation of any kind" -- DFSPH's pressure-projection
pipeline appears to carry real numerical dissipation that this rudimentary
compressible acoustic core (Phase B, `f_acoustic_core`, built to have *no*
implicit regularization at all outside JFNK's own L-stable damping) simply
does not have.

### 2. DFSPH is *not* timestep-constraint-free -- it swaps the acoustic term for an advective one that still needs a ceiling

At `nx=24`, `nu=0`, raising `maxDt` from `0.1` to `1.0` (so the naive
`dt = cflFactor*h/vMax` formula is no longer capped during the low-velocity
spin-up transient) reproduces exactly the kind of blow-up the acoustic-CFL
argument predicts for the compressible core, just for a different reason:

| `cflFactor` (`maxDt=1.0`) | outcome |
|---|---|
| 0.3 (same as the bounded run above) | survives, but with a **violent early transient**: `KE` spikes to ~6700 (`vMax~237`) by step 15 before self-limiting as `vMax` grows and `dt` shrinks back down; settles to `KE~20-27`, `vMax~5-7` by step 150 -- an order of magnitude hotter than the `maxDt=0.1` run's settled state, not a comparable trajectory |
| 0.9 (3x) | **diverges to NaN by step 9** |
| 1.5 (5x) | **diverges to NaN by step 3** |
| 3.0 (10x) | **diverges to NaN by step 4** |

The mechanism: at `t=0` the flow is at rest (`vMax~0`), so a pure
`vMax`-based advective-CFL formula wants `dt -> infinity`; every existing
case in this codebase (`cases/tgv.py`'s fixed `dt=1e-3`,
`SimulationConfig`'s own class default `maxDt=1e-2`) sidesteps this by
never deriving `dt` from `vMax` alone -- they use a fixed or tightly
ceilinged value instead. So DFSPH really does eliminate the specific
*acoustic* CFL term (confirmed structurally too -- see below), but it is
not "free of a `dt` constraint" in the more general sense the plan's
framing might suggest: it substitutes an advective/viscous CFL that is
just as capable of blowing up an explicit-Euler-style step if pushed past
its own natural scale, particularly at the low-velocity/spin-up transient
where a velocity-based formula is degenerate. This is the DFSPH-side
analogue of what JFNK exists to fix on the compressible side, just solved
here the mundane way (pick a sane fixed/ceilinged `dt`) rather than via an
implicit solver, since there is no stiff acoustic *mode* to make implicit
in the first place.

**Structural confirmation that there is no acoustic term at all**: grepped
`schemes/dfsph.py` and `systems/incompressible.py` for `fixedSoundSpeed`/
`dt_acousticConstraint`/`soundspeeds` -- the `soundspeeds` per-particle
field and the `fluid.fixedSoundSpeed`/`dt_acousticConstraint` config fields
all exist structurally on `IncompressibleState`/`IncompressibleSPHConfig`
(inherited from the shared shape with `WeaklyCompressibleState`/
`WeaklyCompressibleSPHConfig`) but **are never read anywhere** in
`dfsph_step` or `IncompressibleSystem.finalize`'s actual physics. They are
dead weight for this scheme, not a hidden acoustic constraint quietly
still in effect.

### 3. `nx=128`, `xi=1.0`, `k=4` (E1.6/E1.7's own parameters): a saturated state comparable to the compressible core, and a genuinely surprising divergence with viscosity *on*

Spin-up from rest, `cflFactor=0.3`, `maxDt=0.1` (the safe ceiling from
finding #2), 16384 particles, 1000-step budget:

| `nu` | steps completed | wall time | outcome | settled KE | settled `vMax` |
|---|---|---|---|---|---|
| 0.0 | 1000 (`t=7.92s`) | 186.2s (~186ms/step) | **bounded**, full budget | ~2.57-2.65 | ~2.2-2.6 |
| `alphaToNu(0.01, c_s=10, h)` | 720 (`t=6.39s`) | 149.0s (~207ms/step) | **diverged to NaN at step 720** | (peaked ~5.3 at `t~2.8s`, settling toward ~2.4-2.5 before divergence) | (~2.1-2.8 before divergence) |

The `nu=0` run's `vMax` band (`~2.2-2.6`, `KE~2.57-2.65`) lands right in
the range `JFNK_PLAN.md` E1.6 measured for the *compressible* core at the
same `nx=128`/`xi=1.0`/`k=4` (`vMax~2.1-2.9`, `KE~2.65-2.8`) -- a
genuinely cross-validating result: two completely different formulations
(compressible EOS+pressure-gradient vs. incompressible divergence-free
projection) land on essentially the same saturated turbulent velocity
scale for this flow, which is exactly what you would hope for if both are
resolving the same physics reasonably.

The `alpha=0.01` run's divergence is the opposite of what the
compressible+JFNK story would predict (there, more viscosity strictly
helped). Reading the trajectories side by side, the more likely culprit is
**particle disorder, not a viscous-shear mechanism**: both runs show real
density excursions well before either one finishes -- `rhoMin` dropping as
low as `0.699` (`nu=0`, step ~675) and `0.871` (`alpha=0.01`, step ~625),
i.e. 13-30% below rest density in localized regions, consistent with SPH
particle clustering/void formation under sustained shear. Neither run has
`shiftProperties.active` on (this probe left it at its `cases/tgv.py`-
matching default, `False`); real `cases/kolmogorov.py` runs the
compressible `deltaSPH` scheme, which this probe did not touch, and may
rely on other machinery (density diffusion, different kernel/support
choices) this rudimentary DFSPH setup doesn't carry. **Stated as a
hypothesis, not proven** -- this probe did not run a controlled ablation
isolating shifting or the density-excursion mechanism, so whether the
`alpha=0.01` run's divergence is genuinely viscosity-linked (e.g. the
explicit velocity-diffusion Laplacian term interacting badly with an
already-disordered neighborhood) or simply which-side-of-a-chaotic-
bifurcation-you-land-on given otherwise-identical forcing is not resolved
here.

**Cost**: `solveDivergenceFree` (`solverConfig.divergenceFreeSolver`,
default `maxIterations=32`, `tolerance=2.5e-3`) pinned at exactly `32`
iterations on every sampled step in every run at both resolutions;
`finalize`'s `solveIncompressible` (`solverConfig.pressureSolver`, default
`maxIterations=64`, `tolerance=5e-4`) pinned at exactly `64` on every
sampled step. I.e. the shipped default relaxed-Jacobi solvers never
converged to their own tolerance on this flow, in either the `nx=24` toy
runs or the `nx=128` production-scale runs -- every step pays the full
fixed iteration budget (96 relaxed-Jacobi sweeps/step total, always,
regardless of local state).

**Wall-clock vs. the compressible core's own numbers**: `nx=128` reached
`t=7.92s` of simulated time in `186s` wall time (1000 steps, `dt` shrinking
from `0.1` down to `~0.006-0.008` as the flow saturates). `JFNK_PLAN.md`
E1.6 reports the *compressible* core's own plain-explicit `RK4` spin-up
reaching `t=8s` at the same `nx=128` in `24.6s` wall time (4266 steps,
fixed `dt=0.001875`). **This is not a controlled apples-to-apples
benchmark** (different schemes, different kernel-cache warm state,
different code paths, single run each, this machine only) -- but the
direction is worth stating plainly: DFSPH trades many cheap acoustic-CFL-
limited explicit steps for few, much larger, but *individually much more
expensive* implicit-projection steps (~4.3x fewer steps to cover the same
simulated duration, but ~7.5x more wall-clock time overall), and on this
probe the net wall-clock cost came out **higher**, not lower, than plain
explicit `RK4` on the compressible core at the same resolution and
duration -- before JFNK's own added Newton/GMRES cost on top of that `RK4`
baseline is even counted. A fair total-cost comparison against JFNK
specifically (rather than against plain `RK4`) was not attempted here and
would need one; `JFNK_PLAN.md` E1.6/E1.7 do not report an absolute
wall-clock number for the JFNK branch itself, only iteration-count/`dt`-
multiplier findings, so there is nothing directly comparable to quote yet.

## Caveats, stated plainly

- One probe, one machine, one set of resolutions (`nx=24`, `nx=128`), one
  domain size/wavenumber (`L=2.0`, `k=4`), one forcing amplitude
  (`xi=1.0`), one random seed for the symmetry-breaking jitter. Kolmogorov
  flow is chaotic; exact step counts at which anything happens (or doesn't)
  are this probe's own realization, not universal constants -- consistent
  with how `JFNK_PLAN.md`'s own E1.6/E1.7 caveat their compressible-core
  numbers.
- No dt-multiplier "how far past DFSPH's own natural CFL can you push it
  while still tracking a reference" sweep was run at `nx=128` (the
  `maxDt`-removal stress test in finding #2 was only run at `nx=24`) --
  the E1.6-style forced-vs-decaying, large-`dt`-vs-small-`dt`-reference
  comparison structure was not replicated for DFSPH at production scale.
- `shiftProperties.active=False` throughout (matching `cases/tgv.py`'s own
  default) -- whether particle shifting would resolve the `nx=128`
  density-void/divergence issue in finding #3 was not tested.
- The `nx=128`/`alpha=0.01` divergence is one run; it was not repeated with
  a different seed to check whether it is a robust finding or a
  single-realization chaotic accident. Given how directly it contradicts
  the naive "more viscosity = more stable" prior, this is the single
  finding in this writeup most worth an independent rerun before leaning
  on it.
- Wall-clock comparisons against `JFNK_PLAN.md` E1.6's own numbers are
  cross-codepath, cross-session, single-run, and not controlled for kernel-
  cache warm state -- treat the *direction* (DFSPH's per-step cost is
  substantial and non-adaptive) as the finding, not the specific ratio.
- The "dead-mutation bug" in `IncompressibleSystem.finalize` (troubleshooting
  item 2) is reported here as an observation for the supervising session's
  awareness, not verified against the module's git history/intent -- it may
  be known/accepted dead code rather than an oversight; this probe did not
  check.

## Files

- `warpSPH/scripts/probe_kolmogorovIncompressible.py` — new, final working
  version (confirmed via multiple successful runs at `nx=24` and `nx=128`).
  Only new/modified file in `warpSPH/`; `cases/kolmogorov.py` and
  `cases/tgv.py` are untouched, and nothing else under `warpSPH/src`
  changed.
