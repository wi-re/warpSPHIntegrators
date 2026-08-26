# JFNK plan: wave equation bridge → WCSPH acoustic subsystem

Plan for building the stiff-solver rung (`NOTES.md` §3.4's ladder rung 2/3, never
implemented) as a `NonlinearSolver` (S4, done 2026-08-24) usable by the DIRK driver,
validated first against a case that needs none of the still-missing frontend JVP
derivations, then used on a real consumer: making WCSPH's acoustic subsystem
(continuity + EOS + pressure force) implicit so `dt` is set by the advective CFL
instead of the acoustic one, letting the sound speed be chosen purely for
weak-compressibility accuracy rather than as a timestep tax. Target and validation
scenario both resolved 2026-08-24 (see Phases B/C). **Phase A done, 2026-08-24**
(A1-A6, see that section) — the JFNK core exists and is validated against the wave
equation. **Phase B done, 2026-08-25** (steps 1-5, see that section) — the rudimentary
zero-dissipation WCSPH acoustic core exists, and its own hypothesis test found the
positive result: JFNK stays bounded at `20×` the acoustic CFL with zero dissipation
where explicit `RK4` and the registry's own Picard(2) default both diverge. **Phase
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

`FixedPointSolver` (Phase 2, `dirk.py`) is a fixed-count Picard iteration — the right
default for the non-stiff regime, and it is *the only* nonlinear solver this library
has. Picard is not a stiff solver at any iteration count (`tests/test_dirk.py`
measures `-9999` at 2 iterations and `10^39` at 20, same stiffness, same tableau);
that limitation is why NOTES.md's ladder scoped a rung-2 JFNK from the start, but it
was never built this session — Phase 2 shipped rung 1 only.

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
  calling `implicitBackwardEulerStep` directly, not `getIntegrator(...)`. Wiring it
  through `getIntegrator('Backward Euler (implicit)')` is itself part of the payoff:
  it proves the generic DIRK driver reaches the same answer as a problem-specific,
  hand-eliminated solve.

(Separately, `warpSPH/WAVE_EQUATION_PLAN.md` is a different, unstarted effort — turning
the wave-equation *demo* into a registered, CLI-runnable `Case`. Unrelated to this plan
and not a dependency either direction; `test_implicitWaveEquation.py` already runs
standalone via `pytest`.)

## Phase A — JFNK core + wave-equation validation — **DONE 2026-08-24**

`warpSPHIntegrators` (new solver machinery) + `warpSPH` (validation tests only, no
production code changes). All six steps landed as originally scoped, plus one
real finding along the way (see A3). `warpSPHIntegrators/tests/` gained
`test_jfnk.py` (19 tests: flatten/unflatten, GMRES on hand-built operators, FD/JVP
agreement, JFNK-through-DIRK on the stiff oscillator, generic-driver coverage on
other tableaus); full suite verified at **1399 passed, 114 skipped** (was 1380/114
at the top of this phase — 19 new, zero regressions, zero changes to anything
registered before this phase). `warpSPH/tests/test_implicitWaveEquation.py` gained
6 more tests (12 total in that file); full `warpSPH` suite verified green
(exit 0, no failures) alongside it.

**A1. Flatten/unflatten bijection over `integrated`-tagged fields.** **Done** —
`fields.py`: `integrated_field_names`, `flatten_integrated`, `unflatten_integrated`,
plus the shared primitive both that and A3's dual-seeding need,
`replace_integrated_fields(template, replacements)` ("swap in new tensors for the
named `integrated` fields, clone everything else unchanged" — the same shape as
`unflatten_integrated`'s flat-slice replacements and A3's dual-tensor replacements,
so it's one function instead of two near-duplicates). Field order is whatever
`dataclasses.fields` returns, which is stable per-class, so flatten/unflatten always
agree on it without recording names anywhere.

**A2. Generic FD matvec.** **Done** — `jfnk.py`'s `fd_matvec(step, Y, y_flat, G_y,
eps=None)`, closed over an already-computed `y_flat`/`G_y` so the outer Newton loop
pays for exactly one extra `step` evaluation per Krylov iteration, matching NOTES.md
S3.4's own accounting. `eps` defaults to the standard Knoll-Keyes scaled step
(`sqrt(eps_machine) * (1+‖y‖)/‖v‖`).

**A3. Generic exact-JVP matvec.** **Done, with one real finding** — `jfnk.py`'s
`jvp_matvec(step, Y)`, exactly as planned (`dual_level()`, seed `v` onto `Y`'s
integrated fields via `make_dual`, run `step` once, read tangents back off the
result via `unpack_dual`). Does **not** catch `NotImplementedError`, confirmed by
using it successfully end-to-end (the wave equation's `f` is entirely the wrapped
Laplacian, so this never fires here, but nothing here would swallow it if it did).

**The finding**: the naive "wrap every `integrated` field as dual, unconditionally"
version crashes — not on anything conceptually wrong, but on a real
`torch.autograd.forward_ad` incompatibility in `warpSPHCore`'s bridge, confirmed by
isolating a single `warpOperation(Laplacian, ...)` call: when **one** dual tensor
reaching `StateAwareWarpFunction.apply()` has an all-zero tangent while **another**
dual tensor in the *same* call has a live one, PyTorch's own internal bookkeeping
trips `RuntimeError: ... INTERNAL ASSERT FAILED ... expected both tensor and its
forward grad to be floating point or complex` — before `.jvp()` is even reached, so
it isn't `warpSPHCore`'s `hasLiveTangent` convention failing to apply, it's PyTorch
itself. This is not a rare edge case: it is exactly what happens on the wave
equation's own validation initial condition (`v(0) = 0`), because `du/dt = v = 0`
identically at the very first Newton iterate, so `G(y0)`'s `u`-block is exact zero
while its `v`-block isn't — the first real Krylov vector GMRES ever builds already
has this shape. Fixed at the right layer, not worked around: `jvp_matvec` now skips
`make_dual` for any field whose tangent slice is identically zero and leaves that
field primal instead — exact by linearity (a zero tangent contributes zero to the
JVP either way), not an approximation, and it composes correctly with fields that
*do* stay dual (e.g. `u`'s output still correctly picks up a nonzero tangent from a
live `v`, via the `du/dt = v` coupling, even though `u` itself wasn't wrapped).
Separately, `gmres` (A4) also never calls `matvec` on the literal zero vector
(`x0 = 0`'s implied first residual) — a legitimate GMRES shortcut in its own right
(`matvec(0) == 0` exactly, `matvec` always being a Jacobian-vector product), which
happens to sidestep the *whole-vector* version of the same underlying bug too.

**A4. GMRES.** **Done** — `jfnk.py`'s `gmres(matvec, b, x0=None, tol=1e-8,
maxiter=None, restart=30)`. Restarted GMRES(m), Arnoldi + Givens rotations (Saad's
formulation), operating on flat vectors from A1. Cross-checked against
`warpSPH/modules/incompressible/krylov.py`'s GMRES conceptually, not imported, per
the original plan.

**A5. `JFNKSolver`.** **Done** — `jfnk.py`. Each outer iteration: evaluate
`step(Y)` once, check `norm(Y, step(Y)) < tol` (defaulting to a plain flat relative
norm, `_default_flat_norm`, when the caller supplies none — `state_norm`'s
Hairer-Wanner convention, what `dirk.py` passes, is a different scale, so
`JFNKSolver`'s own default has to stand on its own), else take one Newton
correction via GMRES and loop. Verified converging in one correction (plus one
verifying `step` call) for the wave equation's exactly-linear stage system, both
matvec modes. `matvec=`/`tol=`/`max_iterations=`/`gmres_tol=`/`gmres_maxiter=`/
`gmres_restart=`/`fd_eps=` all settable at construction or per-call via `**opts`,
matching `FixedPointSolver`'s `opts.get(..., self.x)` pattern. Opt-in only, exactly
as resolved 2026-08-24: not registered anywhere, `FixedPointSolver` is still every
DIRK scheme's default.

**A6. Validation.** **Done** — `test_implicitWaveEquation.py` extended (not
replaced) with:
- `test_jfnkThroughGenericDIRKAgreesWithHandRolledCG[fd/jvp]`:
  `getIntegrator('Backward Euler (implicit)')(...) solver=JFNKSolver(matvec=...)`
  reproduces `implicitBackwardEulerStep`'s CG answer to `rtol=1e-4, atol=1e-5` for
  both matvec modes — confirming the generic `2N`-dimensional coupled `(u,v)` solve
  and the hand-eliminated `N`-dimensional CG solve land on the same fixed point,
  which was most of the point of this check.
- `test_picardDivergesWhereJFNKStaysBoundedOnAStiffStep`: at `dt=2.0` (vs. the
  case's own CFL-scaled `dt≈0.006` at `nx=32`), Picard(20) blows up past `1e10`
  while JFNK (both matvecs) lands below `1.0` (well under the initial amplitude of
  1, consistent with L-stable damping) — the wave equation's own stiffness doing
  the same job `k=1e6` does for the oscillator probe in `test_dirk.py`.
- `test_exactJVPMatvecAgreesWithFDAndUsesNoMoreGMRESIterations`: agreement to FD's
  own truncation tolerance (`rtol=1e-3, atol=1e-4`) **and** a direct GMRES
  iteration-count comparison (`iters_jvp <= iters_fd`) — measuring the concrete
  payoff, not just asserting agreement.
- `test_jfnkWorksWithOtherDIRKTableausOnTheWaveEquation[Implicit Midpoint/SDIRK2 x
  fd/jvp]`: the cheap bonus, confirmed — same driver, same solver, no new code.

## Phase B — rudimentary WCSPH acoustic core (no surface treatment, no dissipation)

`warpSPH`. Added 2026-08-24, per your suggestion: a minimal weakly-compressible
scheme — continuity + EOS + pressure force, *no* artificial or physical viscosity, no
surface-aware pressure treatment, no gravity, no boundaries — as a second validation
rung between the wave equation (Phase A, linear, one operator) and the real
`deltaSPH_step` acoustic subsystem (Phase C, needs one new JVP derivation). This one
needs **zero new derivation**, found while grounding this addition: swap
`computePressureForceSurfaceAware` for `computePressureForceSymmetric`
(`warpSPH/src/warpSPH/modules/pressure/symmetricForce.py`) and the pressure force
*is already* `-warpOperation(..., WarpOperation.Gradient, gradientMode=
GradientScheme.Symmetric, ...) / densities` — literally the wrapped `Gradient`
operator, nothing custom. Continuity is already `Divergence` (Phase C's table); the
EOS is already pointwise. So this rudimentary core is expressible *today*, entirely
from the six-operator JVP-wrapped set, with no Phase C dependency — it can be built
and validated in parallel with Phase C, or before it, without blocking on any new
derivation landing first.

**The hypothesis this phase tests, in your words: "it should theoretically be
feasible, with an implicit solver of sufficient accuracy, to run the simulation
without dissipation."** Explicit WCSPH schemes need artificial viscosity/diffusion
for numerical stability in practice — the TGV case's own `MIN_STABLE_ALPHA=0.01`
(Phase C) is exactly that floor. The question this phase asks directly: is that
floor a property of *explicit* WCSPH specifically, or of weakly-compressible SPH in
general? An L-stable implicit treatment of the acoustic part damps exactly the
high-frequency modes artificial viscosity is usually patching over; if that's
sufficient on its own, a well-converged (not the fixed-2-iteration Picard default —
see the note on `FixedPointSolver`'s `tol=`/`norm=` path, §3.6) implicit solve should
stay stable with every dissipation term at zero, where the explicit scheme provably
would not.

**Steps:**
1. Build the rudimentary step function (new, small — not a config toggle on
   `deltaSPH_step`, since it's meant to have no surface/mDBC/diffusion code paths to
   reason about at all, not just zeroed coefficients): `dx/dt = v`, `drho/dt =
   -rho·Divergence(v)`, `p = EOS(rho)`, `dv/dt = -pressureForce_warp(p, rho)` (or the
   equivalent direct `Gradient` call), nothing else. Same protocol
   `WaveSystemv3` already demonstrates satisfying — a state/system pair implementing
   `BaseState`/`BaseIntegrationSystem` directly, no fluid-scheme machinery needed.
   **Done, 2026-08-25.** `warpSPH`: `configurations/acousticCoreConfig.py`
   (`AcousticCoreConfig` — kernel/supportMode plus the two `GradientScheme`s
   actually needed, `continuityGradientMode=Difference` matching
   `computeMomentum` and `pressureGradientMode=Symmetric` matching
   `computePressureForceSymmetric`, and the isothermal-EOS constants
   `restDensity`/`soundSpeed`), `systems/acousticCore.py` (`AcousticCoreState`/
   `AcousticCoreSystemUpdate`/`AcousticCoreSystem`, field-for-field the same
   shape as `WaveSystemv3` but with `positions`/`velocities`/`densities` all
   `integrated` — matching `WeaklyCompressibleState`'s three physical fields
   with every surface/mDBC/shifting field it also carries dropped, since this
   core has no code paths for any of them), `schemes/acousticCore.py`
   (`f_acoustic_core` — continuity via `warpOperation(..., Divergence, ...)`,
   the EOS via `isoThermalEOS` reused directly from
   `modules/eos/weaklyCompressible.py` (pointwise, no warpSPHCore operator per
   the plan's own Phase C table), pressure force via
   `warpOperation(..., Gradient, gradientMode=Symmetric, ...) / densities`,
   adapted from `computePressureForceSymmetric` to call the operator directly
   since that function is typed against `CompressibleState`). Registered in
   `configurations/__init__.py`/`systems/__init__.py`; deliberately **not**
   registered in `schemes/__init__.py`/`schemes/builder.py` — that module's own
   docstring says nothing there should be called by name without going through
   `buildScheme`'s `SchemeBundle` registry, and this core has no `Case`/CLI
   consumer (yet) to justify one, so `f_acoustic_core` is imported directly from
   its submodule, exactly how the wave-equation implicit test already imports
   `f_wave_equation`.

   `tests/test_acousticCore.py` (5 tests, all green, full `warpSPH` suite still
   exit 0 alongside them) validates the foundation itself, not yet the Phase B
   hypothesis: finite/bounded derivatives; `dx/dt == v` exactly; the symmetric
   pressure force conserves momentum to `~1e-4` relative
   (`sum_i m_i dv_i/dt ≈ 0` on the periodic domain, a real check on the
   operator wiring, not just "it runs"); the continuity sign convention is
   correct (verified against a genuinely periodic `sin(kx)`/`sin(ky)` velocity
   field, after an initial version of this check used a field linear in
   absolute position that was discontinuous across the periodic seam and
   produced the wrong local sign — fixed by switching the probe field, not the
   scheme); and — the actual point of building this on `warpSPHIntegrators`'
   protocol at all — one backward-Euler step through
   `getIntegrator('Backward Euler (implicit)')` converges to a finite,
   positive-density state with both `JFNKSolver` and `FixedPointSolver`,
   confirming `AcousticCoreState`/`AcousticCoreSystem` actually satisfy the
   generic DIRK driver's protocol rather than merely resembling it.

2. A periodic, boundary-free initial condition — reuse `cases/tgvWeaklyCompressible.py`'s
   sampling/domain setup (or a decaying-random field, your original suggestion) rather
   than building sampling from scratch, but drive it through the new minimal step
   function instead of `deltaSPH_step`. **Done, 2026-08-25** — `warpSPH`:
   `sample/acousticCore.py`'s `buildPeriodicVortexAcousticCoreSystem(nx, dim, L, uMag,
   rho0, soundSpeed, cflFactor, device, dtype)`, a rotational (Taylor-Green-family)
   periodic vortex on uniform density, built from `sampleParticles` on a periodic
   `DomainDescription` — the same primitives `cases/tgvWeaklyCompressible.py` uses,
   without that case's `Case`/shuffle/`RunContext` machinery. Not a `Case` itself (no
   `Case`/CLI consumer exists for this core, matching step 1's own reasoning) — lives
   in `sample/` rather than `systems/`, mirroring `sample/waveSystem.py`'s "last stage
   of the pipeline, not an entry point" framing; callers (tests, the notebook below)
   import it directly. Both `tests/test_acousticCore.py` and
   `tests/test_acousticCoreStability.py` build on this one helper rather than each
   duplicating sampling/domain setup.
3. Run three ways at the same `c_s`/resolution: (a) explicit, zero dissipation — the
   expected-to-fail control; (b) `getIntegrator('Backward Euler (implicit)')` +
   `FixedPointSolver` (the Phase 2 default, 2 iterations) — tests whether Picard's own
   accuracy is enough or whether this needs real convergence; (c) same DIRK scheme +
   `JFNKSolver` with a tight tolerance — the actual hypothesis under test.
   **Done, 2026-08-25** — `tests/test_acousticCoreStability.py`, `nx=24`, `dt = 20×`
   the acoustic CFL (`0.2·h/c_s`, i.e. `4·h/c_s`, well past the `C·h/c_s` term
   `modules/timestep/weaklyCompressible.py` uses to floor `dt` today).
4. Validation criterion is **stability, not decay-rate matching** — this is different
   from Phase D's TGV check. Zero-viscosity 2D Euler TGV doesn't decay (the analytic
   `KE(t)` solution Phase D compares against is a *viscous* result), so the bar here
   is bounded energy over a long run (no blow-up, no secular drift) for (c), contrasted
   against (a)'s expected blow-up — not agreement with any closed-form curve. **Done,
   2026-08-25** — the three tests assert exactly this: (a)/(b) assert `diverged=True`
   within a bounded step budget (non-finite state, or density past `100×` rest density
   — added after finding Picard's own blow-up timing is not bit-exact run to run,
   GPU reduction order, so waiting on literal `inf`/`nan` alone was flaky; see the
   test's own docstring); (c) asserts `diverged=False` over `40` steps — `4×` the
   budget the other two need to fail — plus `max|rho| < 1.1·rho0`, `std(rho) < 0.01`
   (no clumping/pairing runaway), `vMax < 1.0` (order the initial vortex amplitude,
   not growing).
5. Record the finding either way. If the hypothesis holds, it's a genuine result
   worth carrying into Phase D's design (maybe dissipation-free WCSPH becomes a real
   option, not just a stability nice-to-have); if it doesn't, that's equally useful to
   know before Phase D spends effort on the full scheme. **Done, 2026-08-25 — the
   hypothesis holds, on this probe.** At `dt = 20×` the acoustic CFL, zero
   dissipation, `nx=24`: explicit `RK4` overflows to `nan` within `10` steps (`rhoMax`
   already at `~1e34` by step 9 in the exploratory run this test is drawn from) and
   `FixedPointSolver` (Picard(2), the registry's own shipped default for every
   implicit DIRK scheme) independently diverges too — same qualitative failure, just
   slower and less numerically clean (`rhoMax` past `1e30` by step ~27 rather than
   overflowing outright, run-to-run-variable in exactly which step, per point 4 above)
   — while `JFNKSolver` (`matvec='fd'`, `tol=1e-6`) stays bounded for at least `150`
   steps in the exploratory run (`rho` within `0.3%` of `rho0` throughout, `std(rho)`
   fluctuating in a `~7e-6`–`~7e-4` band with no secular growth after an initial
   settling by step ~40, `vMax` gently decaying `0.049→0.031` — consistent with
   L-stable damping of the fastest modes, not drift). So on this periodic-vortex
   probe, the acoustic-CFL/stability tax explicit (and under-converged-Picard) WCSPH
   pays is decoupled from whether artificial dissipation is present, once the
   acoustic part is actually solved to convergence rather than integrated explicitly
   or Picard-iterated a fixed couple of times — carried into Phase D's design per
   this point's own framing. **Caveat, stated plainly**: this is one probe (one
   resolution, one `dt` multiplier, `150` steps ≈ several vortex periods at this
   `dt`, one initial condition) on the *rudimentary* core (no shifting, no surface
   treatment) — it demonstrates the acoustic-stiffness half of the hypothesis
   cleanly, not a proof that zero-dissipation WCSPH never destabilizes at any
   resolution/duration/IC, and it says nothing about the tensile/pairing instability
   particle shifting (not built here) normally guards against, which is a spatial
   discretization concern the *time*-integration scheme doesn't touch either way.
   `examples/weaklyCompressible/acousticCore_implicit_vs_explicit.ipynb` (new) walks
   through this comparison interactively, plots included, mirroring
   `examples/wave/waveCase_implicit_vs_explicit.ipynb`'s structure (no numbered-`Case`
   prefix, since — like `naca.ipynb` in the same directory — this isn't a registered
   `Case`).

## Phase E — scope-extension ladder for the rudimentary core

`warpSPH`. Added 2026-08-25, per your instruction while you're away from approvals:
grow Phase B's rudimentary acoustic core's own feature set incrementally, each rung
fully validated with JFNK before adding the next, rather than jumping straight to
`deltaSPH_step`. **Complementary to, not a dependency of, Phases C/D** — those close
the JVP gap and integrate JFNK with the *real* scheme; this phase keeps testing JFNK
against progressively richer physics on the small, fully-controlled core Phase B
already validated, in the order you gave: **forcing (Kolmogorov) → mDBC (bounded
domain) → free surface (a lot more work)**. Each step below is sized from actually
reading the relevant `warpSPH` modules, not guessed.

### E1 — forcing (Kolmogorov flow) — **Done, 2026-08-25**

The cheap one: a steady sinusoidal body acceleration, no new state fields, no new
`warpSPHCore` operator. `configurations/acousticCoreConfig.py`: `AcousticCoreConfig`
gained `forcingAmplitude`/`forcingWavenumber` (default `0.0`/`4.0` — opt-in, `0.0`
recovers Phase B exactly). `schemes/acousticCore.py`: `f_acoustic_core` gained
`dv_x/dt += forcingAmplitude * sin(forcingWavenumber * pi * y)` on `y` wrapped via
`getPeriodicPositions` (`cases/kolmogorov.py`'s own convention), omitting that case's
`BoundaryCondition`/`forcingFunctions` indirection (`forcing / mass` there, a direct
acceleration here — same net physics, no machinery to route through) and its
symmetry-breaking `y`-noise term (not needed to test JFNK's Newton iteration against
a *driven* flow, which is this rung's whole point). `sample/acousticCore.py`:
`buildPeriodicVortexAcousticCoreSystem` threads both params straight through; pass
`uMag=0.0` alongside a nonzero `forcingAmplitude` for a quiescent-start,
purely-forcing-driven system.

**The finding, recorded in `tests/test_acousticCoreForcing.py`'s own module
docstring (3 tests, confirmed non-flaky over 3 repeated runs)**: this is a
genuinely *different* answer from Phase B's, not a repeat of it. Phase B's
periodic vortex is a smooth rotational flow with no instability mechanism, so a
well-converged JFNK solve stayed bounded *indefinitely* at `20×` the acoustic CFL
with zero dissipation. The Kolmogorov base flow `v_x = xi·sin(k·pi·y)` is instead a
textbook **linearly unstable shear flow** (the Meshalkin-Sinai/Kolmogorov-flow
instability) — with zero dissipation, nothing damps the exponentially growing
perturbation mode, so **every** solver tested eventually diverges once that mode
reaches nonlinear/numerical blow-up. Confirmed *not* an artifact of the fixed
acoustic-CFL-scaled `dt` becoming advectively invalid as the forced flow
accelerates: a velocity-tracking adaptive `dt` (`min(dt_acoustic, C·h/vMax)`,
recomputed every step) was tried and diverged at essentially the same step count
regardless. `JFNKSolver` does not prevent this — it isn't supposed to; the growing
mode is a real hydrodynamic instability, orthogonal to the acoustic stiffness JFNK
actually addresses — but it measurably **delays** it: at `dt = 20×` the acoustic
CFL, `RK4` and `FixedPointSolver` (Picard(2), the registry default) both diverge by
step 5-6, `JFNKSolver` by step ~13, identically across 3 repeated runs (unlike Phase
B's own Picard(2) comparison, this timing is *not* GPU-reduction-order-flaky). This
is exactly consistent with why real `cases/kolmogorov.py` both seeds
symmetry-breaking noise on purpose (to trigger transition deliberately) and relies
on δ-SPH's artificial/physical viscosity to keep the resulting turbulent cascade
numerically bounded — dissipation is physically load-bearing against a real
instability here, a different concern from Phase B's acoustic-stiffness story. The
two findings are complementary: JFNK's acoustic-`dt` payoff is real, but it is not a
general substitute for dissipation on every flow, only on ones (like Phase B's
vortex) with no other instability mechanism at zero dissipation.

### E1.5 — dissipation operators in JVP format — **done, 2026-08-25**

Direct response to E1's own finding: dissipation is physically load-bearing
against the Kolmogorov shear instability, so the natural next step is giving
`f_acoustic_core` real density/velocity dissipation that JFNK's exact-JVP
matvec can still differentiate through, rather than only ever falling back to
FD. `warpSPH`: `configurations/acousticCoreConfig.py` gained
`densityDiffusionCoefficient`/`velocityDiffusionCoefficient` (both `0.0`,
opt-in, matching `forcingAmplitude`'s own pattern) plus `laplacianMode`/
`laplacianGradientMode`; `schemes/acousticCore.py`'s `f_acoustic_core` gained
two terms, each a single `warpOperation(..., WarpOperation.Laplacian, ...)`
call — `densityDiffusionCoefficient * h * soundSpeed * Laplacian(densities)`
on `drho/dt` (a plain Fickian form, the same prefactor shape
`modules/deltaSPH/densityDiffusion.py` uses for the real scheme's own
delta-SPH term, minus its flux/renormalization machinery) and
`velocityDiffusionCoefficient * Laplacian(velocities)` on `dv/dt` (the
classic Brookshaw/Morris SPH-viscosity Laplacian — confirmed
`modules/deltaSPH/velocityDissipation.py`'s own docstring: the real scheme's
velocity term already *is* "a Laplacian operation over
`currentState.velocities`", just wrapped in a custom kernel with an
artificial/physical-viscosity coefficient switch this rudimentary core
doesn't need). Zero new `warpSPHCore` derivation either way — `Laplacian` is
already one of the six value+geometry JVP-wrapped operators.
`sample/acousticCore.py`'s `buildPeriodicVortexAcousticCoreSystem` threads
both coefficients through, matching `forcingAmplitude`'s own precedent.

**The finding, found building this, not guessed, and fixed at the root
rather than routed around**: the exact-JVP path crashed with a
kernel-argument dtype mismatch (`computeSPHLaplacianBrookshawJVP_Kernel
argument 'queryValues' type mismatch: expected array(ndim=1,
dtype=float32), got array(ndim=1, dtype=vec2f)`) under the obvious first
choice, `laplacianMode=LaplacianScheme.Brookshaw` (`OperationProperties`'
own default). Root cause, confirmed by reading `wp_laplacianJVP.py`'s own
docstring: the Laplacian geometry-JVP's `Brookshaw`/`Naive` schemes were
scalar-field-only (fixed `scalar_t` kernel arguments), so neither could
differentiate `Laplacian(velocities)`, a vector field — only `Dot`/`Default`
were generically `Any`-typed over scalar or vector fields. First fix
attempt was a workaround (default to `Default`); **superseded same day** by
fixing `warpSPHCore` itself, per your instruction that the real fix should
be "fairly straightforward in logic": `q_ij = (fj-fi)*B_ij` and every
downstream op in both schemes are elementwise scalar-times-field regardless
of whether the field is scalar or vector, so generalizing
`computeSPHLaplacianBrookshawJVP_*`/`computeSPHLaplacianNaiveJVP_*`
(`_Func_i`/`_Func_Adjacency`/`_Kernel`) from `scalar_t` to `Any`-typed
arguments/arrays — plus the public wrapper's zero-tangent-default/
`outputDtype` plumbing (`torch.zeros_like(queryValues)`/
`castTorchToWarpAsBuiltins(queryValues).dtype` in place of the old
scalar-shaped `zerosScalar(n)`/hardcoded `scalar_t`) — needed no formula
change, only relaxed type annotations, exactly the `Any`-typing Dot/Default
already used. Verified with new gradcheck-style tests
(`warpSPHCore/tests/operations/test_forward_mode_geometry_jvp_laplacian_
brookshaw.py`/`..._naive.py`, `*_matches_jacobian_reference_vectorField_2d`,
16 new tests total): both schemes now match a reverse-mode-Jacobian
reference on a genuine `(n, dim)` vector field, not just scalar fields as
before. Full `warpSPHCore` suite re-run clean (419 passed, 1 pre-existing
unrelated failure in `test_field_abstraction.py` confirmed present before
this change too, 1 skipped). `AcousticCoreConfig.laplacianMode` now defaults
back to the eponymous `Brookshaw` (`OperationProperties`' own library-wide
default), not the `Default`-scheme workaround. Verified end-to-end: a
one-step `getIntegrator('Backward Euler (implicit)')` +
`JFNKSolver(matvec=...)` run with both dissipation coefficients and
`forcingAmplitude` all nonzero converges to a finite, positive-density state
for **both** `matvec='fd'` and `matvec='jvp'` with `Brookshaw` live. Existing
`test_acousticCore*.py` suite (11 tests) still green, dissipation off by
default.

**Both follow-ups now done, 2026-08-25 — `warpSPH/tests/test_acousticCoreDissipation.py`
(4 tests):**

- **The dissipation-vs-Kolmogorov hypothesis, resolved with a genuinely
  different answer from E1's own finding.** E1 measured JFNK *delaying* the
  Kolmogorov shear instability's divergence at zero dissipation but never
  avoiding it. An exploratory sweep of `velocityDiffusionCoefficient`
  (`nx=24`, `dt=20x` the acoustic CFL, `JFNKSolver(matvec='fd')`, 80-step
  budget) found a real stability **threshold**, not just a continuation of
  the delay trend: `nu<=1e-2` all diverge by step 13-16 (E1's own order),
  `nu=0.015/0.02/0.025` push divergence later but still fail within budget
  (steps 20/25/27), and `nu>=0.03` **never diverges** in 80 steps -- confirmed
  not a budget artifact (still bounded at 200 steps) and not GPU-reduction-
  order flakiness (3 repeated 80-step runs at `nu=0.03` all stayed bounded).
  Consistent with the real physics: Kolmogorov flow has a genuine viscous
  stability threshold (a critical Reynolds number below which the linear
  instability is fully damped, not merely slowed), and `nu=0.03` is past it
  for this probe. `densityDiffusionCoefficient` alone (swept at
  `0.05`/`0.1`/`0.2`, no velocity diffusion) stayed in the same "still
  diverges" band with no monotonic trend -- expected, since the shear
  instability is a velocity/vorticity mechanism the density-diffusion term
  doesn't directly damp. `test_smallVelocityDiffusionStillDivergesLikeE1`/
  `test_sufficientVelocityDiffusionAvoidsTheKolmogorovDivergence` assert the
  two sides of this threshold.

- **The FD-vs-JVP numerical-agreement check, attempted and immediately
  finding a real, pre-existing issue unrelated to dissipation.** The first
  attempt mirrored Phase A's `test_exactJVPMatvecAgreesWithFDAndUsesNoMoreGMRESIterations`
  verbatim (`assert_close(jv_fd, jv_jvp, rtol=1e-3, atol=1e-4)` on the whole
  flattened backward-Euler stage matvec) and failed: `17%` of the flattened
  vector mismatched, `~0.0047` max absolute difference (`nx=24`, `dt=1e-3`).
  Isolated by disabling dissipation entirely (down to Phase B's own already-
  validated quiescent, unforced, zero-dissipation core) -- **the disagreement
  reproduces at the same order with dissipation off**, so it is not a
  Laplacian/dissipation bug: a pre-existing characteristic of `fd_matvec`'s
  single global Knoll-Keyes step size `h` applied to this multi-field,
  multi-scale nonlinear stage system (positions ~O(1), velocities ~O(0.05),
  densities ~O(1), `soundSpeed=10`, all sharing one flat vector and one `h`),
  concentrated in the velocity slice specifically. Confirmed directly:
  dissipation-on vs. dissipation-off at an otherwise-identical particle
  configuration and probe vector gives essentially unchanged disagreement
  (`615` bad entries / `0.0047453` max diff without dissipation vs. `609` /
  `0.00475144` with). Re-scoped the test to what's actually meaningful given
  this finding: `test_dissipationDoesNotDegradeFDvsJVPAgreementRelativeToBaseline`
  checks the *differential* claim (dissipation doesn't make agreement worse
  than the pre-existing baseline), not an absolute tolerance neither matvec
  mode was ever shown to meet on this scheme. The exact-JVP path's own
  correctness is what `warpSPHCore`'s operator-level gradcheck tests already
  establish (`torch.autograd.functional.jacobian` ground truth, not a coarse
  fixed-`h` FD probe); `test_jfnkThroughGenericDIRKConvergesWithBothDissipationTermsLive`
  separately confirms both matvec modes still reach a consistent, physically
  sane converged fixed point despite FD's per-entry noise. **Not chased
  further this pass**: whether `fd_matvec` should use a per-field (rather
  than global) step size is a generic `warpSPHIntegrators` question, not
  specific to this scheme or to dissipation, and out of scope here.

### E1.6 — spin-up + snapshot probe at "normal" velocity scale — **done, 2026-08-25**

Direct follow-up to your own question: does E1.5's `nu=0.03` stability
threshold (found on a toy sweep driven from rest, saturating at only
`vMax~0.3-0.5`, `Ma~0.03-0.05`) say anything about the "normal" `v~1`,
`Ma=0.1` operating regime real WCSPH targets? Answered by the method you
specified rather than by extrapolating: spin up from rest at *realistic*
parameters (`cases/kolmogorov.py`'s own `xi=1.0`/`k=4`/`alpha=0.01`, `nx=128`
matching its own resolution) using a small, acoustic-CFL-respecting `dt`
(`RK4`, not the earlier `20x` stress-test `dt` -- explicit can't survive
that regardless of physics) until the flow is genuinely turbulent, snapshot
it, then branch into continued-forcing vs. decaying, comparing `JFNKSolver`
at a large `dt` against the same small-`dt` `RK4` as a reference. New
scripts: `warpSPH/scripts/probe_kolmogorovSpinup.py` (spin-up + periodic
`.pt` snapshots) and `probe_kolmogorovContinuation.py` (load a snapshot,
run both branches at both matvec modes, compare against the reference).

**Finding 1 -- the real turbulent velocity is *higher* than the naive
estimate, not lower.** Spin-up (`nx=128`, `dt=0.001875`, `4266` steps for
`t=8s`, `24.6s` wall time -- cheap) shows textbook Kolmogorov transition:
laminar linear growth to `t~1.3s` (`rhoStd~1e-5`), instability onset and a
transient overshoot peaking at `KE~5.0`/`vMax~3.3` (`t~3s`), then relaxation
into a statistically-steady turbulent state by `t~4.5-8s`
(`KE~2.65-2.8`, **`vMax` fluctuating `2.1-2.9`**, `rhoStd~0.008-0.012`).
At `soundSpeed=10`, that is **`Ma~0.21-0.29`** -- already past the `Ma=0.1`
rule of thumb, not below it. `xi=1.0` is a forcing *amplitude*, not the
flow's own characteristic velocity; the turbulent cascade amplifies well
past it. So this probe already covers (and exceeds) the "normal" regime the
question asked about, using the real case's own parameters, not a
rescaled toy.

**Finding 2 -- JFNK's large-`dt` margin shrinks sharply once the flow is
actually turbulent, and the mechanism is forcing-specific, not generic
advective CFL.** Loaded the settled snapshot at `t=6.0s`
(`snap_nx128_003200.pt`, `KE~2.71`, `vMax~2.2-2.3`) and swept
`dt`-multiplier (relative to the acoustic CFL) for both branches:

| multiplier | RK4 (reference dt) | JFNK `matvec='fd'` | JFNK `matvec='jvp'` |
|---|---|---|---|
| 1-3 | tracks fine | tracks fine (both branches) | tracks fine (both branches) |
| 5 | **explodes** (`KE~3447` by step 3) | forced branch drifts (`vMax` 2.23->3.18 by step 2); decaying branch fine | **tracks fine, both branches, 5 steps** (`vMax` stays `~2.2-2.25`) |
| 10 | diverges outright by step 2 | forced: `vMax` jumps to `5.03` in **1 step**; decaying: fine to `vMax~2.2` at 2 steps | both branches eventually diverge by step 3-4 (forced: `vMax`->14 by step 3; decaying: `vMax`->190 by step 3) |

Reproduced deterministically (bit-identical across reruns -- not GPU-
reduction-order flakiness). Reading the table: **multiplier `<=3` is safe
for either matvec mode; `mult=5` is where `matvec='jvp'` earns its keep**
(stays accurate where FD has already started drifting, and where plain
`RK4` has already caught fire); **`mult=10` is past what either matvec mode
can reliably hold** in this fully-developed turbulent state. This is a
*much* smaller safe margin than the `~20x` seen in E1/E1.5's quiescent,
low-velocity sweep -- the earlier finding does not transfer to genuinely
turbulent flow.

Also notable: at `mult=5`/FD, the **forced** branch degrades while the
**decaying** branch (identical state, identical `dt`, `forcingAmplitude=0`)
stays accurate -- forcing an already-chaotic state is evidently harder for
Newton to track at a given `dt` than letting it relax, consistent with decay
being a monotonically-smoothing process while sustained forcing keeps
re-injecting energy into whatever locally-marginal configuration the
turbulence has produced. This gap closes with the exact JVP (both branches
hold to `mult=5`), suggesting FD's own single-global-step-size inaccuracy
(this plan's own E1.5 finding) is a real contributor to the forced branch's
earlier failure, not a fundamental forcing-vs-JFNK incompatibility --
though `mult=10` shows even the exact JVP is not immune once `dt` is pushed
far enough into this state's chaotic sensitivity.

**Bottom line, answering the original question directly**: at genuinely
"normal" operating conditions (`v~2-3`, `Ma~0.2-0.3`, the real turbulent
velocity this probe measured, not the naive `v~1` guess), JFNK still beats
explicit `RK4` by a real, useful margin (`RK4` already destroys itself by
`mult=5`; JFNK with the exact JVP matvec is still tracking the reference
cleanly there) -- but the payoff is a `~5x` `dt` multiplier here, not the
`~20x` the quiescent low-velocity sweep suggested. The dissipation
coefficient itself (`alpha=0.01`, the real case's own default) was not the
limiting factor in this probe at all -- both branches stayed numerically
healthy (`rhoStd` bounded, no clustering) up to `mult=5`; what limits `dt`
at production velocity scale is Newton/GMRES's own accuracy against a
genuinely chaotic, multi-scale nonlinear state, not the dissipation
threshold E1.5 characterized on the toy sweep.

**Caveats, stated plainly**: one realization, one snapshot time, no
explicit symmetry-breaking noise (unlike `cases/kolmogorov.py`'s own Perlin
`noiseLevel=0.01` term -- this core's own instability presumably seeds from
floating-point asymmetry alone, since it visibly transitions regardless);
a genuinely chaotic system means precise multiplier thresholds are
indicative of this state/parameters, not universal constants; only
`nx=128` at this domain/wavenumber was probed, not a resolution sweep.
Scripts are left in `warpSPH/scripts/` for reuse against other snapshots,
durations, or multipliers.

### E1.7 — does a genuinely weakly-compressible (`Ma~0.1`) regime restore JFNK's margin? — **done, 2026-08-25**

Direct follow-up to your own instruction: spin up `xi=0.5` (half `cases/
kolmogorov.py`'s own forcing) to see if that lands closer to `Ma=0.1`, and
separately re-run `xi=1.0` at half the timestep as a convergence check on
E1.6's own saturated-velocity measurement. All snapshots kept (`session
scratchpad/kolmogorovSpinup/`, not `/tmp` directly, so they survive):
`snap_nx128_xi1.0_*.pt` (original, `dtFactor=1`), `snap_nx128_xi0.5_*.pt`
(new), `snap_nx128_xi1.0_*.pt` dt-halved (`dtFactor=0.5`, distinct filenames
by run).

**`dt`-convergence check, `xi=1.0`**: halving `dt` (`0.000937` vs.
`0.001875`) reproduces E1.6's own transition and saturated state closely --
peak `KE~4.5-4.9` at `t~3.0-3.4s` (both runs), settled `KE~2.3-2.4`/
`vMax~2.0-2.5` by `t~6-8s` (vs. the original's `KE~2.65-2.8`/`vMax~2.1-2.9`,
same band within run-to-run chaotic variability). **E1.6's saturated
velocity was not a timestep artifact.**

**`xi=0.5` does *not* halve the saturated velocity.** Settled state
(`t~6.5-8.5s`) sits at `vMax~1.7-2.0` -- only `~15-25%` lower than `xi=1.0`'s
`~2.1-2.9`, not the `~50%` a linear force-balance estimate
(`xi_eq = A/(nu k^2)`, laminar-equilibrium reasoning) would predict. The
saturated turbulent velocity is a genuinely nonlinear property of this
flow, only weakly sensitive to the forcing amplitude -- halving the forcing
does not get you to `Ma=0.1` here; it takes you to `Ma~0.17-0.20`, still
above the rule of thumb. (The `xi=0.5` run also showed visible
intermittency -- `vMax` dipping to `~1.5` around `t~6.5s` then climbing back
past `2.4` by `t~9.5s` -- 2D Kolmogorov flow's own known bursting behavior,
not a settling artifact; the snapshot used below (`t=6.75s`, `vMax=1.88`)
sits in a comparatively quiet window, not necessarily *the* equilibrium.)

**But that modest velocity reduction is enough to fully restore JFNK's
large-`dt` margin.** Repeating E1.6's exact `dt`-multiplier sweep on the
`xi=0.5` snapshot (`vMax=1.88`, `Ma=0.19`), `matvec='fd'` (no need to even
reach for the exact JVP this time):

| multiplier | RK4 (reference `dt`) | JFNK `matvec='fd'` |
|---|---|---|
| 3 | fine | fine, both branches |
| 5 | `rhoStd` growing (`1.7e-2`, early warning) | **fine, both branches** |
| 10 | diverges by step 2 | **fine, both branches** |
| 20 | diverges by step 1 | **fine, both branches** |

No degradation at all through `mult=20`, forced or decaying, plain FD --
matching E1's own quiescent-state margin, not E1.6's degraded `~3-5x` one.
`RK4` still fails exactly where the acoustic-CFL argument predicts
(velocity-independent, as expected -- it's a wave-stability constraint, not
an advective one), so JFNK's payoff over explicit is, if anything, *larger*
here than in E1.6's higher-velocity case.

**Reading the two findings together**: E1.6's `mult<=5` ceiling was real but
narrower than it might have looked -- it is specifically tied to *how
intense* the turbulence is at the instant JFNK takes its large step, not to
turbulence being present at all. A `~20%` reduction in the local velocity
scale (`vMax` `2.2` -> `1.9`) was enough to erase the degradation entirely
in this probe. This cuts both ways for practical guidance: a run that stays
in a genuinely weakly-compressible band (`Ma` closer to `0.1-0.15` than
`0.2-0.3`) can likely keep something much closer to E1's original `~20x`
margin; a run that runs hotter (`Ma~0.2-0.3`, which is what this core's own
`xi=1.0`/`k=4` naturally saturates to, not a contrived extreme) should
expect the smaller `~3-5x` margin E1.6 found, and reach for the exact JVP
matvec there specifically since that is where it was shown to earn its
keep. Neither number is universal -- both are this probe's own
domain/wavenumber/resolution, not a general law -- but the *mechanism*
(Newton/GMRES's per-step accuracy budget shrinking as local velocity/
nonlinearity grows, independent of the acoustic subsystem JFNK was built to
fix) is the transferable finding.

**Not done**: pinning down xi=0.5's own intermittent bursting (is the
`vMax` dip around `t~6.5s` a recurring cycle or a one-off relaxation?) and
finding the actual `xi` needed to land the saturated state at `Ma=0.1`
specifically (would need a proper sweep, not two points) -- neither was
asked for here and both are follow-up work if wanted.

### E1.8 — does refreshing adjacency inside the implicit solve raise E1.6's `dt`-multiplier ceiling? — **done, 2026-08-26**

Direct follow-up to E1.6's own open question: within one real DIRK/JFNK
step, every stage, every outer Newton iteration, and every inner GMRES
matvec (`fd` and `jvp` alike) reuses the *same* `AdjacencyList`, confirmed
by tracing `f_acoustic_core`'s own docstring plus
`AcousticCoreSystem.initializeNewState`/`finalize` — adjacency is rebuilt
once, by the test/script driver, only *after* a full real step completes
(`test_acousticCoreStability.py`'s `_runSteps`,
`probe_kolmogorovSpinup.py`'s per-step loop). The real production scheme,
`deltaSPH_step`, instead calls `buildVerletList(..., priorNeighborhood=
adjacency, ...)` at the top of *every* RHS evaluation. At `mult=5-10`
(E1.6's regime), particles can move a non-trivial fraction of `h` within
one implicit step — so is E1.6's `mult<=5` turbulent-flow ceiling partly a
neighbor-staleness artifact, or purely a Newton/GMRES accuracy limit?

**Answer: purely a Newton/GMRES accuracy limit — a clean negative result,
and rebuilding is mildly counterproductive where it would need to help
most.** New sibling function `f_acoustic_core_rebuildAdjacency`
(`schemes/acousticCore.py`, `f_acoustic_core` itself **unmodified**) calls
`buildVerletList` at the top of every invocation, mirroring
`deltaSPH_step`'s exact call shape, then runs the identical physics on the
fresh adjacency. Re-ran E1.6's own `dt`-multiplier sweep
(`snap_nx128_003200.pt`, `t=6.0s`, `vMax~2.2`, both branches, both matvec
modes) baseline (frozen) vs. this variant (rebuild):

| mult | branch/matvec | baseline (frozen) | rebuild (per-call) |
|---|---|---|---|
| 1-3 | either branch, either matvec | fine | indistinguishable from baseline |
| 5 | forced, `fd` | drifts: `vMax` 2.23→3.18→177.5→diverges step 5 | **worse**: `vMax` 2.23→34.0→11362→diverges step 4 |
| 5 | decaying, `fd` | fine, all 5 steps | indistinguishable |
| 5 | forced, `jvp` | fine, all 5 steps | indistinguishable |
| 10 | forced, either matvec | diverges by step 3-4 | diverges **one step earlier, harder** (e.g. `fd`: peak 77707 vs. baseline's 304) |
| 10 | decaying, `fd` | drifts late, fine to step 3 | diverges by step 4 |

Bit-identical on a repeat of the sharpest cell — not GPU-reduction-order
noise. Everywhere the frozen baseline already held, rebuilding changed
nothing material; everywhere the baseline was already marginal or failing,
rebuilding made it fail faster and harder, never better. **A plausible but
unverified mechanism**: once different Newton iterates within one implicit
step correspond to meaningfully different particle configurations, letting
the discrete adjacency itself change *between* those iterates adds a second
source of inconsistency (the linearization target discretely shifting
underneath Newton, on top of an already-thin accuracy margin) rather than
correcting for staleness — offered with appropriate uncertainty, not
independently confirmed by a separate diagnostic.

Checked directly, not assumed: `buildVerletList` does **not** crash on
forward-mode dual-wrapped positions (`matvec='jvp'`) — `warp`'s
`wp.from_torch()` bridge reads through to the primal buffer regardless, so
no scoping to `matvec='fd'`-only was needed, contrary to this ablation's
own starting expectation (kept the explicit `unpack_dual(...).primal`
extraction anyway as the semantically-correct thing to do). Cost: 150-400
extra `buildVerletList` calls per real step (Newton iterations x (1 +
GMRES iterations), no stage multiplier since Backward Euler is 1-stage),
1.1x-2.4x wall-clock overhead — most of each call is the cheap
"still-valid" check, not a full rebuild, so the multiplier is smaller than
the raw call-count ratio suggests. Given zero stability upside anywhere
tested and a real cost plus a mild downside exactly where it would need to
help, **the guidance from E1.6/E1.7 stands unchanged**: reach for the exact
JVP matvec and/or a lower-`Ma` operating point to buy back margin at large
`dt`, not a within-step adjacency refresh. New script:
`warpSPH/scripts/probe_kolmogorovAdjacencyRebuild.py`. Existing
`test_acousticCore*.py` (15 tests) re-verified green — no regression to
`f_acoustic_core`. One snapshot/resolution/realization, 5-step short-horizon
comparisons only — same scope limits as E1.6/E1.7.

### E1.9 — does the codebase's existing incompressible solver (DFSPH) already deliver this for free? — **done, 2026-08-26**

A comparison the plan owner asked for directly, and a fair question given
E1-E1.8's whole throughline: incompressible SPH has no acoustic mode and no
acoustic CFL *by construction*, using existing, already-registered code
(`scheme='divergenceFree'`, `schemes/dfsph.py`, exercised today only by the
incompressible `cases/tgv.py`/`test_physics.py`). Does it already deliver
some or all of what this JFNK effort is chasing on the same forced-
Kolmogorov shear-instability problem, no new solver required? `kolmogorov.py`
is hardcoded to `scheme='deltaSPH'`, so this combination had never been run —
genuine troubleshooting, not a rerun of an existing path. New standalone
script (bypassing `Case`/CLI machinery, matching `sample/acousticCore.py`'s
own precedent, `cases/kolmogorov.py`/`cases/tgv.py` **not modified**):
`warpSPH/scripts/probe_kolmogorovIncompressible.py`. Full writeup:
`scratch_dfsph_kolmogorov_findings.md` (this session's scratchpad); key
findings folded in below.

**Troubleshooting, found running this, not guessed**: (a) a domain-builder
default (`device='cpu'`) silently defeated by passing `device=None`
explicitly rather than omitting it — fixed, ~10x steady-state speedup once
actually on GPU; (b) a real, verifiable **latent bug in existing, untouched
production code** found while trying to read solver iteration counts back
out of a step result: `IncompressibleSystem.finalize`
(`systems/incompressible.py`) reassigns `returnValues[-1] = (...)` trying to
surface its own (second) pressure solve's iteration count, but `returnValues`
elsewhere is `[r1]` and this rebinds the list slot, not the tuple `r1`
already referenced by `StageResult.aux` — that solve's iteration count is
silently unreachable except via a `verbose=True` print. Noted for awareness,
not fixed (out of scope, cosmetic — introspection only, not physics); worked
around in the probe by parsing the verbose print instead of touching
production code.

**Finding 1 — DFSPH does *not* need explicit dissipation for the Kolmogorov
instability to saturate rather than blow up, unlike the compressible core
(E1.5's `nu>=0.03` threshold).** `nx=24`, `xi=1.0`, `k=4`, natural CFL-
respecting `dt`: **every** `nu` tested, including exactly `0.0`, stayed
bounded and statistically steady — the `nu=0` run was extended to 900 steps
(`t~81s` simulated, ~9x E1.5's own longest nu-sweep run) specifically to
rule out "hasn't diverged yet," with the expected physical trend still
visible (higher `nu` → lower saturated KE/`vMax`/`rhoStd`). **This changes
what "zero dissipation" means for this scheme, not the underlying physics**:
DFSPH's default `integrateRho=False` recomputes density from a plain SPH
summation every step rather than integrating a stiff continuity equation —
a hypothesis (not independently ablated) for a strong built-in numerical
regularizer the acoustic core's `f_acoustic_core` was deliberately built
without, on top of whatever damping the relaxed-Jacobi projection itself
contributes.

**Finding 2 — DFSPH is not timestep-constraint-free; it swaps the acoustic
term for an advective one that still needs a ceiling.** Structurally
confirmed no acoustic term exists in `dfsph_step`'s actual physics at all
(`soundspeeds`/`fixedSoundSpeed`/`dt_acousticConstraint` are present on the
shared state/config shape but never read). But removing the safety `maxDt`
ceiling and driving `dt` from a pure `vMax`-based advective-CFL formula
reproduces the same *kind* of explicit blow-up the acoustic-CFL argument
predicts on the compressible side, for a different reason: at `t=0` the
flow is at rest, so a `vMax`-based formula is degenerate (wants `dt→∞`) —
`cflFactor` multipliers of `3x/5x/10x` over the safe baseline all diverged
to NaN within 3-9 steps at `nx=24`. Every existing case in this codebase
sidesteps this with a fixed or tightly-ceilinged `dt`, never deriving it
from `vMax` alone.

**Finding 3 — at production scale (`nx=128`, E1.6/E1.7's own parameters), a
genuinely cross-validating result plus a genuinely surprising one.** The
zero-viscosity run's saturated `vMax~2.2-2.6`/`KE~2.57-2.65` lands right in
the band E1.6 measured for the *compressible* core at the same resolution/
forcing (`vMax~2.1-2.9`/`KE~2.65-2.8`) — two unrelated formulations
(EOS+pressure-gradient vs. divergence-free projection) landing on
essentially the same saturated turbulent velocity scale, a good physics
cross-check. But the run *with* the case's own default physical viscosity
(`alpha=0.01`) **diverged at step 720** while the zero-viscosity run stayed
bounded for the full 1000-step budget — the opposite of the compressible
story, where more viscosity strictly helped. Both runs show real localized
density excursions (`rhoMin` down to `0.70`-`0.87`, 13-30% below rest
density) before failing, pointing tentatively at particle-disorder/
density-void formation rather than a viscous-shear mechanism — **stated as
a hypothesis, not proven**; no controlled ablation (shifting on/off, a
repeat with a different seed) isolated the cause, and this is the single
finding in this rung most worth an independent rerun given how directly it
contradicts the naive "more viscosity = more stable" prior. **Root-caused
and fixed, 2026-08-26 — see E1.10.**

**Cost**: the shipped default relaxed-Jacobi solvers (`maxIterations=32`
divergence-free, `64` constant-density) pinned at their *caps* on every
single sampled step at both resolutions in every run — i.e. never actually
converged to their own tolerance on this flow, always paying the full fixed
96-sweep-per-step budget, a different cost character than JFNK/GMRES's
state-dependent iteration count. Wall-clock, stated with caveats (cross-
codepath, cross-session, single run each, not a controlled benchmark):
`nx=128` DFSPH reached `t=7.92s` simulated in `186s` wall time (1000 steps,
`dt` shrinking `0.1→~0.006-0.008` as the flow saturates) vs. the
compressible core's own plain-explicit `RK4` spin-up reaching `t=8s` at the
same resolution in `24.6s` (E1.6, 4266 fixed-`dt=0.001875` steps) — ~4.3x
fewer steps, but ~7.5x more wall-clock, i.e. **higher** net cost than plain
explicit `RK4` on this probe, before JFNK's own Newton/GMRES cost on top of
that `RK4` baseline is even counted. No absolute JFNK-branch wall-clock
number exists yet in E1.6/E1.7 to compare against directly (they report
iteration-count/multiplier findings, not wall time) — a real comparison this
rung couldn't complete, not a result showing JFNK loses on cost.

**Reading E1.8 and E1.9 together, answering the motivating question
plainly**: the *acoustic*-stiffness half of this plan's premise is real but
not unique to JFNK — DFSPH removes that specific constraint too, for free,
with already-registered code, for anyone able to accept an incompressible
formulation. What DFSPH does *not* obviously do is dodge the Kolmogorov
flow's physical consequences at production scale — it fails differently
(particle disorder, not viscous blow-up; more viscosity can hurt, not help)
and its own inner-solve cost is a substantial, non-adaptive, always-at-cap
tax rather than a converged, state-dependent one. Neither solver is a free
lunch on this problem; they trade different constraints for different
failure modes and different cost profiles, which is itself the honest
answer — not evidence either approach is simply superior.

**Caveats, stated plainly**: one probe, one machine, one resolution pair
(`nx=24`/`128`), one domain/wavenumber/forcing amplitude, one jitter seed —
same scope limits E1.6/E1.7 already carry. No `dt`-multiplier sweep (the
E1.6-style forced-vs-decaying, large-vs-small-`dt`-reference structure) was
run for DFSPH at `nx=128`, only the `maxDt`-removal stress test at `nx=24`.
Particle shifting was off throughout (matching `cases/tgv.py`'s own
default) — finding 3's density-void divergence was root-caused and fixed
without it; see E1.10.

### E1.10 — root-causing and fixing E1.9's finding-3 divergence — **done, 2026-08-26**

Direct follow-on to E1.9's own most-uncertain result: at `nx=128`, the run
*with* physical viscosity diverged (step 720) while zero-viscosity stayed
bounded, with density excursions pointing tentatively at particle disorder.
The project owner's own prior experience with this codebase supplied the
lead directly: this incompressible scheme's pressure-projection pipeline is
*supposed* to act as an implicit particle-shifting mechanism (restoring
uniform density is what a shifting technique does) — so instability
traceable to particle disorder suggests that mechanism isn't doing its job
correctly, not that it's insufficient by nature. Reading
`warpSPH/src/warpSPH/systems/incompressible.py`'s `IncompressibleSystem.
finalize` turned up a concrete candidate, found by direct code reading:

```python
dx = dt**2 * dvdt_incomp                            # position correction (live)
proj_vel = torch.einsum('nij, ni -> nj', gradVel, dx)  # Taylor velocity correction
self.state.positions += dx
# self.state.velocities -= proj_vel                 # <- commented out
```

`solveIncompressible`'s constant-density pressure solve returns an
acceleration (`dvdt_incomp`) applied every step as a **position** shift —
the standard "IISPH shifting via extra pressure solve" trick. `gradVel` and
`proj_vel = ∇V·Δx` (the standard first-order correction any shifting
technique needs to keep a particle's carried velocity consistent with its
new location) were already computed, unconditionally, every step — and then
silently discarded. This is distinct from, and unconditionally active
regardless of, the separate `schemeConfig.shiftProperties.active`-gated
block earlier in the same function — confirmed to be inert dead debug
scaffolding from an earlier session's own experiment (a `dx` variable
reused/overwritten before ever being applied), left untouched per the
project owner's explicit instruction, since the incompressible scheme isn't
expected to need that *explicit* WCSPH-style shift in the first place.

**Tested before touching production code**: a runtime monkeypatch
(`inspect.getsource` + text-patch + re-`exec` + class-attribute rebind, new
script `warpSPH/scripts/probe_kolmogorovIncompressibleVelCorrection.py`,
reusing E1.9's own `probe_kolmogorovIncompressible.py` directly) enabled the
correction without editing the file, confirmed live via a smoke test, then
compared on the exact `nx=128`/`alpha=0.01` case:

| run | steps | outcome | worst `rhoMin` before/at failure |
|---|---|---|---|
| baseline (unpatched) | 720/1000 | **diverges to NaN at step 720** (reproduces E1.9 exactly, same seed) | 0.843 |
| patched (correction live) | 1000/1000 | **survives full budget** | 0.709 (recovers, no cascade) |
| patched, extended | 1600/1600 | **still survives**, deeper excursions than baseline ever saw | 0.439 (recovers) |

Not just a delayed failure: the patched run tolerates *more severe* density
disorder than the excursions that killed the baseline, and recovers from
them rather than cascading to NaN.

**Applied to production, one line**: `systems/incompressible.py`,
`IncompressibleSystem.finalize` — uncommented `self.state.velocities -=
proj_vel`. Confirmed via `git diff` this is the only change; the
`shiftProperties.active`-gated block is untouched. **Validated, and
independently re-verified this session** (not just trusted from the
agent's own report): `pytest tests/test_physics.py -k tgv` → 3 passed (the
existing incompressible-TGV regression test, the same scheme this fix
touches); full `tests/test_physics.py` → 60 passed; full `warpSPH` suite →
clean, `EXIT=0`. The `nx=128`/`nu=0` case that already worked before the
fix still works after it (1000/1000 steps, no divergence) — no regression
to the case E1.9 already validated.

**Caveats, stated plainly**: one seed, one machine, one resolution actually
compared before/after (`nx=128` — `nx=48`/`nx=64` never reproduced the
failure in the first place, so there was nothing to fix there). The
1600-step extension is one additional data point, not a systematic
long-horizon sweep (5000+ steps untested). *Why* the correction matters as
much as it does was not independently instrumented (e.g. no direct
measurement of `proj_vel`'s magnitude relative to the velocity field) — the
kinematic-inconsistency-compounding-under-shear reading is the natural one
given the code and consistent with the result, not proven beyond that.
Whether this same class of bug affects `WeaklyCompressibleSystem` (the
`deltaSPH`/compressible scheme `cases/kolmogorov.py` actually runs in
production) was not checked — that class has its own, different `finalize`
and applies its shift via a separate, already-live code path (confirmed
working in E1.9's own cross-validation against the compressible core), not
`IncompressibleSystem`'s `solveIncompressible`-based one this fix touches.

**E1.9's probe script promoted to a first-class `Case`, 2026-08-26**: the
one-off `probe_kolmogorovIncompressible.py` (CLI/`Case`-machinery-free by
design, matching every other probe in this plan) is not the same thing as
this codebase's normal way of running a scenario — no registry entry, no
`warpsph-run` CLI access, no snapshot/continuation support for further
sweeps. `warpSPH/src/warpSPH/cases/kolmogorovIncompressible.py` (new,
registered in `cases/__init__.py`'s `CASE_MODULES`) is the proper
`Case`, built the way `tgv` (scheme mechanics, mass normalisation) and
`kolmogorov` (the forcing/Perlin-noise symmetry-breaking physics, reused
verbatim rather than the probe's jitter-only shortcut) already are — not a
rename of the probe. One genuinely new piece it needed that neither
reference case has: `DFSPH` has no acoustic term, but this flow's velocity
scale changes by roughly an order of magnitude between its quiescent start
and saturated turbulent state, so a fixed `dt` (`kolmogorov`'s own
convention, safe there because `deltaSPH`'s acoustic-CFL `dt` doesn't
depend on the flow state) would be wrong here — `kolmogorovIncompressibleTimestep`,
a `case.timestep` hook mirroring the probe's own validated advective+viscous-CFL
`pickDt` formula, fills that gap. Validated end-to-end through the real
protocol, not just import-tested: `test_runner.py`/`test_physics.py`
(78 tests total) unaffected; a programmatic `run(...)` and the actual
`warpsph-run kolmogorovIncompressible` CLI entry point both produce sane,
adaptively-timestepped trajectories at `nx=24` and `nx=128` (dt growing
from `0.001` to `0.06-0.1` as the flow spins up, matching the mechanism's
own justification above).

### E2 — mDBC (bounded domain) — **scoped, not started**

Sized by reading `modules/mdbc/velocity.py` and `modules/mdbc/density2025.py`
directly, not assumed. Two genuinely different pieces hide under one name:

- **Boundary-particle velocity** (`computeBoundaryVelocities`): dispatches per
  boundary material to `zero`/`constant`/`noSlip`/`freeSlip`/`extended` policies.
  The first four are built entirely from `warpOperation(..., Interpolate,
  operationMode=OperationDirection.FluidToGhost)` (a directional Shepard-normalized
  gather) plus elementwise vector algebra (mirror/reflect across the ghost normal)
  — and `Interpolate` is one of the six JVP-wrapped operators. Checked directly in
  `wp_interpolateJVP.py`: the JVP kernel applies `checkDirectionality_j` generically
  for any `operationMode` other than `TrueAllToToAll`, the same mechanism
  `FluidToGhost` would use — so these four policies look JVP-differentiable in
  principle, not just FD-only, though nothing has verified that empirically yet
  (no `gradcheck` run against this specific direction/mode combination).
- **Boundary-particle density** (`computeMdbcDensity`, the "modified" in mDBC):
  genuinely harder. Uses `interpolateLiuLiu`, a moving-least-squares reconstruction
  that is **not** one of the six wrapped operators and has no JVP path at all today
  — FD matvec only, or a new, nontrivial derivation (MLS involves a per-point small
  linear solve, not a single kernel-weighted sum the way the six Tier-1 operators
  are). Also gravity-aware (a hydrostatic pressure correction along the ghost
  normal) and gated by neighbor-count thresholds with multiple fallback tiers
  (plain Shepard density, then rest density) — real physics, not a simplification
  opportunity.
- Both need **ghost-particle geometry that doesn't exist yet** for this core:
  boundary/ghost `kinds`/`materials`, `ghostIndices`/`ghostOffsets`, and a region/SDF
  or rigid-body-based generation step (`rigidBody/ghostParticles.py` in the real
  scheme). `AcousticCoreState` has none of this — Phase B deliberately kept it to
  the three physical fields.
- The real modules are typed against `WeaklyCompressibleSPHConfig`/
  `SimulationConfig` (`schemeConfig.fluid.restDensity`, `.gravityConfig`, ...), not
  `AcousticCoreConfig` — reusing them verbatim needs either conforming this core's
  config to that shape (defeats Phase B's "as small as `WaveEquationConfig`" design)
  or a thin adapter layer.

**Why not started this session**: this is new state-layout work plus a real
uncertain derivation (MLS JVP), not a same-shape extension of what Phase B/E1
already validated — rushing it without a chance for you to review the design would
risk leaving a half-working boundary treatment in a codebase you can't currently
check on. **Suggested minimal first cut, when picked up**: a single flat wall
(simplest case, one `RegionType.Boundary` with `BCType.zero`, no gravity, no
`extended`/MLS policy) rather than the full ghost-particle system at once — that
alone would validate the new state fields and the `Interpolate`-`FluidToGhost` JVP
claim above empirically, before touching `computeMdbcDensity`'s harder MLS path.

### E3 — free surface — **scoped, not started, "a lot more work" per your own framing**

The biggest lift of the three, and not just because of the JVP gap. `Phase C`'s own
table already found the one piece tied to JFNK directly: `computePressureForceSymmetric`
(what Phase B/C use) ignores the free-surface mask entirely, but the real
`computePressureForceSurfaceAware`'s `Antuono` branch reads it — so free-surface
support here means either extending Phase C's pressure-force JVP derivation to cover
that branch too (messier: the Antuono correction is a per-particle case split on
surface/non-surface status, not a single linear formula), or accepting FD-matvec-only
runs whenever free-surface treatment is active. Separately, and larger: surface
*detection* itself is substantial existing machinery this core has none of --
`modules/surfaceDetection/` alone has at least three detection schemes (color-field,
Maronne, Barecasco — `colorFieldDetection.py`/`maronneDetection.py`/
`barecascoDetection.py`), normal computation, and a dilation/expansion pass
(`wp_dilate.py`), configured via `SurfaceDetectionConfig`
(`configurations/moduleConfigurations/surfaceDetection.py`). A `scripts/
gradcheck_surfaceDetection.py` already exists, suggesting at least partial gradient
support was checked for this machinery at some point — worth reading before assuming
a JVP derivation has to start from zero, but not read yet this session. No sizing
beyond this pointer attempted here; scoping this properly is its own session's work,
not a paragraph.

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
   (`getIntegrator('Backward Euler (implicit)')` first, matching Phase A's proven
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
