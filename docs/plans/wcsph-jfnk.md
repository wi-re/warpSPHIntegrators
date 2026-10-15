# JFNK plan: wave equation bridge → WCSPH acoustic subsystem

> Ownership/status, 14 October 2026: the library JFNK solver is implemented.
> A/B/E1–E1.10 completion is historical downstream evidence; C/D/E2/E3 remain open
> and require verification in warpSPH / warpSPHCore before scheduling.
> [Current task list](../../REPOSITORY_PLAN.md) owns package work.

Historical design for the stiff-solver rung (NOTES §3.4; now implemented here) as a `NonlinearSolver` (S4, done 2026-08-24) usable by the DIRK driver,
validated first against a case that needs none of the still-missing frontend JVP
derivations, then used on a real consumer: making WCSPH's acoustic subsystem
(continuity + EOS + pressure force) implicit so `dt` is set by the advective CFL
instead of the acoustic one, letting the sound speed be chosen purely for
weak-compressibility accuracy rather than as a timestep tax. Target and validation
scenario both resolved 2026-08-24 (see Phases B/C). **Phase A done, 2026-08-24**
(A1-A6, see that section) — the JFNK core exists and is validated against the wave
equation. **Phase B done, 2026-08-25, cost sweep added 2026-09-04** (steps 1-6, see
that section) — the rudimentary zero-dissipation WCSPH acoustic core exists, and its
own hypothesis test found the positive result: JFNK stays bounded at `20×` the
acoustic CFL with zero dissipation where explicit `RK4` and the registry's own
Picard(2) default both diverge — at a real cost, `~762` ms/step there versus `RK4`'s
own `2.87` ms/step at its native `1×` (step 6). **Phase
E1 (forcing/Kolmogorov) done, 2026-08-25** (see that section) — a genuinely different
finding: the forced flow's own shear instability eventually beats every solver at
zero dissipation, JFNK included, though it measurably outlasts explicit/Picard at
the same `dt`. **E1.8/E1.9 done, 2026-08-26**: refreshing the neighbor list inside
the implicit solve does not raise E1.6's turbulent-flow `dt`-multiplier ceiling (a
clean negative result — that ceiling is a genuine Newton/GMRES accuracy limit, not
neighbor staleness); and the codebase's existing incompressible solver (DFSPH)
removes the acoustic-CFL constraint too, for free, but does not obviously dodge the
Kolmogorov instability's practical consequences at production scale — it fails
differently (particle disorder, not viscous blow-up) and pays a substantial,
non-adaptive inner-solve cost rather than JFNK's converged, state-dependent one.
**E1.10 done, 2026-08-26**: E1.9's own production-scale DFSPH divergence was a real,
fixable bug, not a fundamental limit — `IncompressibleSystem.finalize`
(`warpSPH/src/warpSPH/systems/incompressible.py`) computed but silently discarded
the velocity correction its own implicit particle-shifting mechanism needs to stay
kinematically consistent; enabling it (one line) turns the step-720 divergence into
a run that survives 1600+ steps tolerating deeper density disorder than the one
that used to kill it, with no regression to the existing incompressible-TGV test.
Phase E2 (mDBC) and E3 (free surface) scoped but not started, sized
honestly in that section rather than guessed. Phases C-D not started. Cross-repo:
`warpSPHIntegrators` (the solver itself), `warpSPH`
(the wave-equation bridge case, the Phase B acoustic core, and, later, the full
WCSPH integration), `warpSPHCore` (closing the JVP gap in Phase C).

## Why

The original motivation concerned fixed-count Picard iteration. This package
now supplies JFNK as the default implicit solver, with FixedPointSolver available
explicitly. The following cost claims describe the historical downstream experiments. Picard is not a stiff solver at any iteration count (`tests/test_dirk.py`
measures `-9999` at 2 iterations and `10^39` at 20, same stiffness, same tableau);
that limitation is why NOTES.md's ladder scoped a rung-2 JFNK from the start, but it
was not yet implemented at the original August motivation; Phase A subsequently built it.

Separately, `NOTES.md` §3.4 was corrected 2026-08-24: `warpSPHCore` has a real,
narrowly-scoped forward-mode JVP layer (six operators + Covariance — Density,
Interpolate, Gradient, Divergence, Curl, Laplacian — composable across a single
`torch.autograd.forward_ad.dual_level()`), not the flat "warp has no forward mode"
this doc used to say. That makes an *exact*-JVP matvec possible for warp-native
states for the first time — previously torch-only — but only for an `f` built
entirely from that operator set.

`warpSPH/tests/test_implicitWaveEquation.py` is exactly that case, and turns out to
already be closer to plugged into this library than expected:

- Its stage matvec is *one* `warpOperationJVP(Laplacian, ...)` call — nothing else,
  confirmed by reading the module (`_laplacianMatvec`, lines 91-102). No custom
  kernels, no operator outside the wrapped set.
- `WaveSystemv3`/`WaveSystemStatev3` (`warpSPH/src/warpSPH/systems/waveSystem.py`)
  **already fully implement this library's protocol** — `initializeNewState`,
  `apply_position_update`/`apply_velocity_update`/`apply_state_update`, `finalize`,
  the works. `u`/`v` are tagged `position`/`velocity` "purely so
  `warpSPHIntegrators` treats the pair with its position/velocity integrators;
  neither is a position or a velocity... That reuse is the point" (the module's own
  docstring). No adapter needs writing for Phase A.
- It already has a working, tested reference answer: a hand-rolled CG solve
  (`_conjugateGradientSolve`, explicitly noted as written from scratch because "no CG
  solver existed in this repo to reuse"), validated against the explicit scheme at
  small `dt` and against the closed-form standing-wave solution's convergence rate.
  That is a real regression oracle to check a generic JFNK-through-DIRK path against,
  not a toy with no ground truth.
- It currently bypasses this library's registry entirely — a hand-rolled Python loop
  calling `implicitBackwardEulerStep` directly, not `get_scheme(...)`. Wiring it
  through `get_scheme('Backward Euler (implicit)')` is itself part of the payoff:
  it proves the generic DIRK driver reaches the same answer as a problem-specific,
  hand-eliminated solve.

(Separately, `warpSPH/WAVE_EQUATION_PLAN.md` is a different, unstarted effort — turning
the wave-equation *demo* into a registered, CLI-runnable `Case`. Unrelated to this plan
and not a dependency either direction; `test_implicitWaveEquation.py` already runs
standalone via `pytest`.)

## Completed experiment records

<a id="phase-a--jfnk-core--wave-equation-validation--done-2026-08-24"></a>

[Phase A — JFNK core + wave-equation validation — **DONE 2026-08-24**](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#phase-a--jfnk-core--wave-equation-validation--done-2026-08-24)

<a id="phase-b--rudimentary-wcsph-acoustic-core-no-surface-treatment-no-dissipation"></a>

[Phase B — rudimentary WCSPH acoustic core (no surface treatment, no dissipation)](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#phase-b--rudimentary-wcsph-acoustic-core-no-surface-treatment-no-dissipation)

<a id="phase-e--scope-extension-ladder-for-the-rudimentary-core"></a>

[Phase E — scope-extension ladder for the rudimentary core](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#phase-e--scope-extension-ladder-for-the-rudimentary-core)

<a id="e1--forcing-kolmogorov-flow--done-2026-08-25"></a>

[E1 — forcing (Kolmogorov flow) — **Done, 2026-08-25**](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#e1--forcing-kolmogorov-flow--done-2026-08-25)

<a id="e15--dissipation-operators-in-jvp-format--done-2026-08-25"></a>

[E1.5 — dissipation operators in JVP format — **done, 2026-08-25**](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#e15--dissipation-operators-in-jvp-format--done-2026-08-25)

<a id="e16--spin-up--snapshot-probe-at-normal-velocity-scale--done-2026-08-25"></a>

[E1.6 — spin-up + snapshot probe at "normal" velocity scale — **done, 2026-08-25**](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#e16--spin-up--snapshot-probe-at-normal-velocity-scale--done-2026-08-25)

<a id="e17--does-a-genuinely-weakly-compressible-ma01-regime-restore-jfnks-margin--done-2026-08-25"></a>

[E1.7 — does a genuinely weakly-compressible (`Ma~0.1`) regime restore JFNK's margin? — **done, 2026-08-25**](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#e17--does-a-genuinely-weakly-compressible-ma01-regime-restore-jfnks-margin--done-2026-08-25)

<a id="e18--does-refreshing-adjacency-inside-the-implicit-solve-raise-e16s-dt-multiplier-ceiling--done-2026-08-26"></a>

[E1.8 — does refreshing adjacency inside the implicit solve raise E1.6's `dt`-multiplier ceiling? — **done, 2026-08-26**](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#e18--does-refreshing-adjacency-inside-the-implicit-solve-raise-e16s-dt-multiplier-ceiling--done-2026-08-26)

<a id="e19--does-the-codebases-existing-incompressible-solver-dfsph-already-deliver-this-for-free--done-2026-08-26"></a>

[E1.9 — does the codebase's existing incompressible solver (DFSPH) already deliver this for free? — **done, 2026-08-26**](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#e19--does-the-codebases-existing-incompressible-solver-dfsph-already-deliver-this-for-free--done-2026-08-26)

<a id="e110--root-causing-and-fixing-e19s-finding-3-divergence--done-2026-08-26"></a>

[E1.10 — root-causing and fixing E1.9's finding-3 divergence — **done, 2026-08-26**](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#e110--root-causing-and-fixing-e19s-finding-3-divergence--done-2026-08-26)

<a id="e2--mdbc-bounded-domain--scoped-not-started"></a>

[E2 — mDBC (bounded domain) — **scoped, not started**](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#e2--mdbc-bounded-domain--scoped-not-started)

<a id="e3--free-surface--scoped-not-started-a-lot-more-work-per-your-own-framing"></a>

[E3 — free surface — **scoped, not started, "a lot more work" per your own framing**](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md#e3--free-surface--scoped-not-started-a-lot-more-work-per-your-own-framing)

## Phase C — close the JVP gap for the acoustic subsystem

`warpSPHCore` (derivation), `warpSPH` (wiring). **Target resolved 2026-08-24, per your
call**: not "whichever WCSPH term," specifically the *acoustic* subsystem — continuity
+ EOS + pressure-gradient force — made implicit so `dt` is governed by the advective
CFL (fluid velocity) rather than the acoustic one (`C·h/c_s`, confirmed literally that
formula, no `|v|` term, in `modules/timestep/weaklyCompressible.py:75`), which is what
currently forces `c_s` and `dt` into direct tension: `setupWeaklyCompressibleTimestep`
(`modules/timestep/weaklyCompressible.py:108`) today back-solves `c0` *from* a target
`dt`, i.e. picks the sound speed to fit the timestep budget rather than the accuracy
you actually want. Decoupling them is the entire point — `c_s` becomes free to set as
high as weak-compressibility demands once it stops taxing `dt`.

**Scoped by reading `deltaSPH_step` directly, not assumed** (this session's research
against `warpSPH/src/warpSPH/schemes/deltaSPH.py` and the modules it calls):

| Piece | Operator | JVP status |
|---|---|---|
| Continuity (`drho/dt = -rho·div(v)`) | `Divergence` (`modules/momentum/inconsistent.py:24-38`) | **Already wrapped** — one of the six. |
| EOS (`p = f(rho)`, Tait/isothermal/polytropic/Murnaghan) | none — pointwise scalar function of local `rho`, no neighbor sum (`modules/eos/weaklyCompressible.py:37-59`) | **No warpSPHCore JVP needed at all** — differentiates through plain torch autograd like any elementwise op. |
| Pressure-gradient force | `computePressureForceSurfaceAware` (`modules/pressure/wp_surfaceAware.py`) — a **custom kernel**, called unconditionally, no plain-`Gradient` path exists in the scheme | **The one gap.** |

So the acoustic subsystem needs exactly **one** new derivation, not a list. And it
looks tractable, not speculative-new-math: with `PressureForceScheme.conservative`
(the scheme's default) or any non-`Antuono` variant, the free-surface mask is ignored
entirely regardless of whether a free surface is present (`wp_surfaceAware.py:98-108`
— only the `Antuono` branch reads it), and the force reduces to the classical
momentum-conserving symmetric SPH pressure gradient,
`sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad(W_ij)` — **exactly linear in the pressure
field** for fixed positions/densities/adjacency, the same shape as Tier-1's existing
value-JVP operators ("relaunch the same kernel on tangent arrays," `warpier_adjoint.md`).
This is a Tier-1-style derivation, not a Tier-2 geometry-tangent one — no reason to
expect it harder than the five already-wrapped value-JVP operators were.

**Steps:**
1. Hand-verify the linearity claim above against the kernel's actual code (not just
   the classical formula — confirm `wp_surfaceAware.py`'s implementation matches),
   before writing any JVP code.
2. Derive and register a `JVPSpec` for `computePressureForceSurfaceAware`, gradchecked
   against `torch.autograd.gradcheck` the same way the existing six were (per the
   repo's own `gradcheck` skill/scripts).
3. Register it exactly like the existing `OperatorSpec`s so A3's `dual_jvp` primitive
   picks it up with zero JFNK-side changes.

**Density/velocity diffusion (the δ-SPH `computeDensityDiffusion`/
`computeVelocityDiffusion` terms) are not on this list, deliberately, for the first
pass.** They're called unconditionally in `deltaSPH_step` but their magnitude is
fully coefficient-controlled — `densityDelta=0` and `inviscidAlpha=0` zero them out
without touching scheme code (`configurations/moduleConfigurations/
weaklyCompressibleDiffusionParams.py`). Two things to resolve, not assumed here:
`inviscidAlpha` (artificial/numerical stabilization, by its name) vs. `viscidNu`
(physical kinematic viscosity, by its name) may be separately controlled — if so,
`viscidNu` likely needs to *stay* nonzero for the TGV case below, since the analytic
decay solution `KE(t)=KE(0)·exp(-4νk²t)` depends on real viscosity, only
`inviscidAlpha`/`densityDelta` (numerical stabilization) are candidates to zero. The
TGV case's own docstring flags `MIN_STABLE_ALPHA=0.01` as an empirical stability
floor below which artificial viscosity "stops being reliably stable" for the
*explicit* baseline — whether the same floor applies once the acoustic part is
implicit is an open empirical question this phase should answer, not assume either
way; the implicit treatment may itself supply enough numerical damping to relax it,
or may not.

## Phase D — JFNK over an actual WCSPH run

`warpSPH`. **Validation scenario resolved 2026-08-24**: `cases/tgvWeaklyCompressible.py`
already exists — periodic, boundary-free, 2D weakly-compressible Taylor-Green Vortex,
no external forcing, no free surface, no mDBC, with an analytic decay solution
(`KE(t)=KE(0)·exp(-4νk²t)`) already built into the case for comparison. This is
exactly the "no special treatment needed" scenario asked for, already built — Phase D
does not need to construct a new case from scratch, only wire the implicit path into
an existing one. (It is not currently exercised by `tests/test_physics.py`, which only
runs the *incompressible* TGV case — adding a pytest fixture for the weakly-compressible
one is part of this phase, not a prerequisite blocking it.)

Steps:
1. Split `deltaSPH_step`'s RHS so the acoustic subsystem (continuity + EOS + pressure
   force, Phase C) can be solved implicitly via a DIRK scheme
   (`get_scheme('Backward Euler (implicit)')` first, matching Phase A's proven
   pattern, before trying a higher-order tableau) with `solver=JFNKSolver()`, while
   whatever stays explicit (viscosity if `MIN_STABLE_ALPHA`'s floor turns out to still
   apply, anything Phase C left out) continues on its current path. This is a real
   scheme-level change to how `deltaSPH_step` is called for the implicit path, not
   just a new option flowing through unchanged — sized once Phase C is further along
   and the split's exact shape is clearer.
2. Set `c_s` far higher than the explicit baseline's timestep-constrained value would
   allow (the entire motivation), and confirm `dt` can grow to the *advective* CFL
   limit — `computeTimestep`'s viscous/acceleration terms (`modules/timestep/
   weaklyCompressible.py:72,82`) still apply as a floor; only the acoustic term
   (`dt_c = C·h/c_s`) is what implicit treatment removes from the `torch.min`.
3. Compare against the existing explicit TGV baseline on three axes: correctness (the
   decay rate matches the same analytic `KE(t)` curve, within the case's own existing
   tolerance), stability at the larger `dt` (the actual payoff — nothing before this
   phase demonstrates it; Phase A only proves the machinery is correct, not that it's
   faster/better on a real case), and cost (Krylov iterations + matvec cost per step
   vs. the many more, cheaper explicit steps the acoustic CFL currently forces).

## Explicitly not doing (yet)

- **Fully implicit RK, BDF, IMEX/ARK** (`NOTES.md` §3.6) — separate, larger driver
  architecture changes, not needed for JFNK itself and not requested here.
- **Differentiating through a converged JFNK solve** (the implicit function theorem
  adjoint, `NOTES.md` §3.4's own note that this "only becomes worth its complexity
  when iterating to a tolerance... Phase 3 and later"). Build the forward solve
  first; an IFT adjoint is real, separate work gated on an actual training/adjoint
  use case wanting it, not a default extension of Phase A.
- **HVP-based second-order Newton** — still a confirmed dead end outside Density
  (`warpSPHCore`'s own tracking, item 5: both generic composition routes fail on a
  hard PyTorch limitation). Inexact Newton (matvec only, no Hessian) sidesteps
  needing this by construction; not a gap JFNK needs closed.
- **Deriving JVP for every WCSPH operator up front.** Phase C is deliberately scoped
  to the acoustic subsystem's one gap (`computePressureForceSurfaceAware`), not
  density/velocity diffusion, surface detection, mDBC, or anything else
  `deltaSPH_step` touches that the acoustic-implicit path doesn't need.

## Verification

- `pytest tests/` in `warpSPHIntegrators` — the new JFNK machinery's own unit tests
  (flatten/unflatten round-trips, FD vs. exact-JVP matvec agreement on a case with a
  known Jacobian, GMRES convergence on a hand-built symmetric and non-symmetric test
  operator) should not touch anything registered this session; full suite should stay
  at 1380 passed / 114 skipped plus whatever this phase adds.
- `pytest warpSPH/tests/test_implicitWaveEquation.py` (extended, not replaced) — the
  three-way agreement (hand-rolled CG, JFNK+FD-matvec, JFNK+JVP-matvec) and the
  Picard-fails/JFNK-succeeds case from A6.
- Phase B's new tests (`warpSPH/tests/test_acousticCore.py`,
  `test_acousticCoreStability.py`, **done 2026-08-25**): the foundation checks
  (finite/bounded derivatives, momentum conservation, continuity sign, DIRK/JFNK
  wiring) plus the actual hypothesis test, which found — and asserts — that the
  explicit/zero-dissipation rudimentary core measurably destabilizes (`RK4` and
  Picard(2) both diverge) while the well-converged-JFNK version stays bounded over
  a `4×`-longer run at the same `dt`, zero dissipation. Full `warpSPH` suite stays
  exit 0 alongside them, run 3× to confirm the diverge-detection thresholds aren't
  flaky under GPU reduction-order nondeterminism.
- Phase E1's new test (`warpSPH/tests/test_acousticCoreForcing.py`, **done
  2026-08-25**, 3 tests, confirmed non-flaky over 3 repeated full-file runs): forcing
  sanity (correlation check), smooth finite growth at small `dt`, and the comparative
  finding that `JFNKSolver` outlasts (but does not avoid) divergence relative to
  `RK4`/Picard(2) under sustained zero-dissipation forcing at `20×` the acoustic CFL.
- Phase D, once reached: `warpSPH/tests/test_physics.py`'s existing suite stays green
  with the new implicit path wired in, plus whatever new stability/cost comparison
  the chosen scenario needs.
