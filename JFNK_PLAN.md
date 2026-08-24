# JFNK plan: wave equation bridge → WCSPH

Plan for building the stiff-solver rung (`NOTES.md` §3.4's ladder rung 2/3, never
implemented) as a `NonlinearSolver` (S4, done 2026-08-24) usable by the DIRK driver,
validated first against a case that needs none of the still-missing frontend JVP
derivations, then used to motivate and scope closing that gap for a real consumer
inside `warpSPH`. Not started — no code changes have been made yet. Cross-repo:
`warpSPHIntegrators` (the solver itself), `warpSPH` (the wave-equation bridge case and,
later, the WCSPH integration), `warpSPHCore` (closing the JVP gap in Phase B).

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
iteration cap.

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

## Phase B — close the JVP gap for whichever WCSPH sub-problem gets targeted

`warpSPHCore` (derivation), `warpSPH` (identifying and wiring the target).

**Open question, needs an answer before this phase can be scoped at all: which part
of WCSPH is the actual implicit target?** `deltaSPH_step` (`warpSPH/src/warpSPH/
schemes/deltaSPH.py`) composes far more than the six wrapped operators — density/
velocity diffusion, momentum, surface-aware pressure force, free-surface detection,
mDBC boundary handling, forcing/Dirichlet enforcement, gravity, weakly-compressible
EOS. Deriving JVP for all of it speculatively repeats exactly what this codebase has
declined every previous time it came up (`warpSPHCore`'s own tracking, item 6: "has
been declined every time it came up" without a concrete consumer) — Phase A creates
one, but only for whichever specific term actually needs implicit treatment, not the
whole RHS at once. The precedent to follow is item 2's own finding: IISPH's pressure
sub-problem needed only Gradient+Divergence, a small slice of the full incompressible
solve, not everything solveIncompressible touches.

Once that sub-problem is named:
1. Read off exactly which operators it touches (likely a strict subset of the list
   above — e.g. if it's the diffusion terms, probably just `computeVelocityDiffusion`/
   `computeDensityDiffusion`'s own kernels, not surface detection or mDBC).
2. Derive JVP for those, Tier-2-style (`warpier_adjoint.md`'s existing methodology),
   hand-verified against the operator's own math before any code, then gradchecked —
   same rigor the six existing operators went through, no shortcut for being newer.
3. Register the new `JVPSpec`s the same way the existing ones are (`OperatorSpec`),
   so A3's `dual_jvp` primitive picks them up with zero JFNK-side changes — the whole
   point of A3 being generic rather than wave-equation-specific.

Not sized further here — the actual effort depends entirely on which sub-problem gets
picked, and could range from "one linear operator, a day" (if it looks like IISPH's
pressure equation) to "several genuinely nonlinear kernels, real derivation work"
(if it's the full momentum equation with artificial viscosity).

## Phase C — JFNK over an actual WCSPH run

`warpSPH`. Wire the Phase B sub-problem into `deltaSPH_step` via a DIRK scheme
(`getIntegrator(...)`, Phase A's now-proven pattern) with `JFNKSolver` as the solver.
Compare against the existing explicit baseline on three axes: correctness (no
regression on `tests/test_physics.py`'s existing cases), stability at a larger `dt`
than the explicit CFL limit allows (the actual payoff of going implicit — nothing
here has demonstrated that yet, including Phase A, which only proves the machinery
works, not that it's faster/better on a real case), and cost (Krylov iterations plus
matvec cost vs. the extra force evaluations an explicit substep budget would need for
the same accuracy). Not scoped further until Phase B's target is chosen — this phase
is entirely downstream of that decision.

## Open questions for you

1. Which WCSPH term is the actual implicit target — viscosity, density diffusion,
   pressure/EOS coupling, something else? Blocks Phase B from being scoped precisely.
2. Should `JFNKSolver` become a registry-level *default* for any scheme (replacing
   `FixedPointSolver` where a scheme is registered `stability='L'`, say), or stay an
   explicit `solver=` override the caller opts into? Affects whether `dissipation`/
   `stability` flags on the DIRK schemes need re-examining once a real stiff-solver
   option exists alongside Picard.
3. Any WCSPH scenario already known to want a larger-than-CFL `dt`? Would sharpen
   Phase C's target case rather than picking one arbitrarily.

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
  to whatever sub-problem gets named, not the whole scheme — see Phase B's own
  framing above.

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
