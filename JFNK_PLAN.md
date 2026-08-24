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
equation; Phases B-D not started. Cross-repo: `warpSPHIntegrators` (the solver
itself), `warpSPH` (the wave-equation bridge case and, later, the WCSPH
integration), `warpSPHCore` (closing the JVP gap in Phase C).

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
2. A periodic, boundary-free initial condition — reuse `cases/tgvWeaklyCompressible.py`'s
   sampling/domain setup (or a decaying-random field, your original suggestion) rather
   than building sampling from scratch, but drive it through the new minimal step
   function instead of `deltaSPH_step`.
3. Run three ways at the same `c_s`/resolution: (a) explicit, zero dissipation — the
   expected-to-fail control; (b) `getIntegrator('Backward Euler (implicit)')` +
   `FixedPointSolver` (the Phase 2 default, 2 iterations) — tests whether Picard's own
   accuracy is enough or whether this needs real convergence; (c) same DIRK scheme +
   `JFNKSolver` with a tight tolerance — the actual hypothesis under test.
4. Validation criterion is **stability, not decay-rate matching** — this is different
   from Phase D's TGV check. Zero-viscosity 2D Euler TGV doesn't decay (the analytic
   `KE(t)` solution Phase D compares against is a *viscous* result), so the bar here
   is bounded energy over a long run (no blow-up, no secular drift) for (c), contrasted
   against (a)'s expected blow-up — not agreement with any closed-form curve.
5. Record the finding either way. If the hypothesis holds, it's a genuine result
   worth carrying into Phase D's design (maybe dissipation-free WCSPH becomes a real
   option, not just a stability nice-to-have); if it doesn't, that's equally useful to
   know before Phase D spends effort on the full scheme.

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
- Phase B's new test: the explicit/zero-dissipation rudimentary core measurably
  destabilizes (bounded-energy check fails, or blows up outright) while the
  well-converged-JFNK version stays bounded over the same run — both outcomes are
  useful results, but the test should assert whichever one was actually found, not
  the hoped-for one.
- Phase D, once reached: `warpSPH/tests/test_physics.py`'s existing suite stays green
  with the new implicit path wired in, plus whatever new stability/cost comparison
  the chosen scenario needs.
