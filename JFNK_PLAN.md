# JFNK plan: wave equation bridge → WCSPH acoustic subsystem

Plan for building the stiff-solver rung (`NOTES.md` §3.4's ladder rung 2/3, never
implemented) as a `NonlinearSolver` (S4, done 2026-08-24) usable by the DIRK driver,
validated first against a case that needs none of the still-missing frontend JVP
derivations, then used on a real consumer: making WCSPH's acoustic subsystem
(continuity + EOS + pressure force) implicit so `dt` is set by the advective CFL
instead of the acoustic one, letting the sound speed be chosen purely for
weak-compressibility accuracy rather than as a timestep tax. Target and validation
scenario both resolved 2026-08-24 (see Phases B/C). Not started — no code changes
have been made yet. Cross-repo: `warpSPHIntegrators` (the solver itself), `warpSPH`
(the wave-equation bridge case and, later, the WCSPH integration), `warpSPHCore`
(closing the JVP gap in Phase B).

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

## Phase A — JFNK core + wave-equation validation

`warpSPHIntegrators` (new solver machinery) + `warpSPH` (validation tests only, no
production code changes).

**A1. Flatten/unflatten bijection over `integrated`-tagged fields.** New primitive
(likely `fields.py`, alongside `state_norm`/`state_difference`): a state's integrated
fields, concatenated into one flat tensor and back. Purely mechanical — the field
metadata (`integrated()`) already names everything needed; no new tagging. This is
what GMRES's basis vectors, dot products, and linear combinations operate on.

**A2. Generic FD matvec.** `Jv ≈ (G(Y+εv) - G(Y)) / ε` where `G(Y) = Y - step_fn(Y)`
is exactly the residual shape `NonlinearSolver.solve`'s `step_fn` already produces —
no new abstraction, this composes directly with what Phase 0/2 built. Works for *any*
`f`, no capability gate; ships as the default matvec. Never form the dense Jacobian
(NOTES.md's own caution, still correct) — one extra `step_fn` evaluation per Krylov
iteration, not one per unknown.

**A3. Generic exact-JVP matvec.** A `dual_jvp(step_fn, Y, v)` primitive: opens
`torch.autograd.forward_ad.dual_level()`, seeds `v` (unflattened back onto `Y`'s
integrated fields) as the tangent, runs `step_fn(Y)` once, reads the tangent off the
result. **Must not wrap the call in a try/except that would swallow
`NotImplementedError`** — confirmed (this session's research) that
`StateAwareWarpFunction.jvp` already raises exactly that, loudly, the moment a dual
tensor reaches an operator with no registered `JVPSpec` (`operator_spec.py`'s own
docstring: "a dual-tensor argument reaching this kernel's launch raises a clear error
rather than silently returning a tangent-free dual output"). This is the safety
property asked for — it is already guaranteed by `warpSPHCore`'s existing design, not
something this rung needs to build; the only way to lose it is to add code here that
catches the exception and falls back silently. Don't.

**A4. GMRES.** No symmetry assumption — matches NOTES.md's original rung-2 choice,
and the wave-equation module's own docstring notes its stage operator is symmetric
*only* for constant `c`/zero `damping`, so CG isn't a safe generic default the way it
is for that one restricted case. New, self-contained implementation operating on the
flat vectors from A1 — `warpSPH/modules/incompressible/krylov.py` already has a GMRES,
but at a different abstraction level (SPH pressure fields directly, not an arbitrary
tagged state); worth reading for a correctness cross-check, not importing.

**A5. `JFNKSolver`.** Implements the `NonlinearSolver` protocol (S4) —
`solve(step_fn, y0, norm, **opts)`. An outer inexact-Newton loop; for an exactly
linear `f` like the wave equation it converges in one outer iteration (Newton on a
linear residual is exact), same shape it will need for a genuinely nonlinear `f`
later. `opts` selects the matvec (`'fd'` default, `'jvp'` opt-in) and GMRES tolerance/
iteration cap. **Opt-in only** (resolved 2026-08-24, per your call): `FixedPointSolver`
stays the registry default for every DIRK scheme; `JFNKSolver` is a `solver=`
override a caller reaches for deliberately, never something the registry picks
automatically. It's real machinery (flatten/unflatten, GMRES, a matvec choice) for a
problem `FixedPointSolver` already handles well in the non-stiff regime — no reason
to pay for it, or to reconsider any scheme's `dissipation`/`stability` registration,
when nobody asked for it.

**A6. Validation** (extends `test_implicitWaveEquation.py`, doesn't replace it):
- `getIntegrator('Backward Euler (implicit)')(system, dt=dt, f=f_wave_equation, solver=JFNKSolver())`
  must reproduce `implicitBackwardEulerStep`'s existing CG answer to solver
  tolerance, for both matvec modes. Note this is a genuinely different linear system
  under the hood than the CG reference solves — CG solves the hand-eliminated
  `N`-dimensional equation for `u` alone; going through the generic DIRK driver
  solves the full `2N`-dimensional coupled `(u, v)` stage system, since a generic
  solver has no way to know the problem-specific algebraic elimination is available.
  Both are solving for the same fixed point, so they should agree to tolerance — that
  agreement is itself most of the point of this check.
- JFNK must succeed somewhere Picard measurably fails — mirror
  `tests/test_dirk.py::test_picard_diverges_on_a_stiff_problem_regardless_of_tableau_stability`
  with a `dt`/resolution combo chosen so the wave equation's own stiffness (`dt·ω`
  scaling with resolution here, not a tunable `k` like the oscillator probe) pushes
  Picard into its measured failure mode.
- Exact-JVP vs FD matvec: should agree to FD's own truncation tolerance; the concrete
  case for rung 3 existing at all is fewer GMRES iterations and no `ε` to tune for
  the same accuracy — measure both, don't just assert agreement.
- Cheap bonus, since the driver is generic: also exercise Implicit Midpoint and
  SDIRK2 with `solver=JFNKSolver()` on the same problem — broadens coverage for
  free once A1-A5 exist.

## Phase B — close the JVP gap for the acoustic subsystem

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

## Phase C — JFNK over an actual WCSPH run

`warpSPH`. **Validation scenario resolved 2026-08-24**: `cases/tgvWeaklyCompressible.py`
already exists — periodic, boundary-free, 2D weakly-compressible Taylor-Green Vortex,
no external forcing, no free surface, no mDBC, with an analytic decay solution
(`KE(t)=KE(0)·exp(-4νk²t)`) already built into the case for comparison. This is
exactly the "no special treatment needed" scenario asked for, already built — Phase C
does not need to construct a new case from scratch, only wire the implicit path into
an existing one. (It is not currently exercised by `tests/test_physics.py`, which only
runs the *incompressible* TGV case — adding a pytest fixture for the weakly-compressible
one is part of this phase, not a prerequisite blocking it.)

Steps:
1. Split `deltaSPH_step`'s RHS so the acoustic subsystem (continuity + EOS + pressure
   force, Phase B) can be solved implicitly via a DIRK scheme
   (`getIntegrator('Backward Euler (implicit)')` first, matching Phase A's proven
   pattern, before trying a higher-order tableau) with `solver=JFNKSolver()`, while
   whatever stays explicit (viscosity if `MIN_STABLE_ALPHA`'s floor turns out to still
   apply, anything Phase B left out) continues on its current path. This is a real
   scheme-level change to how `deltaSPH_step` is called for the implicit path, not
   just a new option flowing through unchanged — sized once Phase B is further along
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
- **Deriving JVP for every WCSPH operator up front.** Phase B is deliberately scoped
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
- Phase C, once reached: `warpSPH/tests/test_physics.py`'s existing suite stays green
  with the new implicit path wired in, plus whatever new stability/cost comparison
  the chosen scenario needs.
