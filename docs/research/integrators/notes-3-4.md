> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.4 Newton without forward-mode AD

**Corrected 2026-08-24 — the table below originally read `warp: ✗ no forward mode`,
full stop. That is true of the bare `warp` engine and still is (`warpier_core.md`:
"Warp 1.15 has no forward-mode AD of any kind"), but it is the wrong fact to lead
with for this library's actual downstream, `warpSPHCore`, which has since built a
real — if narrowly scoped — JVP capability on top of warp's reverse-mode kernels.
Verified directly against `warpSPHCore`'s current source before writing this
correction (not assumed from an earlier session's memory):**

- **It is a hand-derived tangent-propagation layer, not engine-level forward mode.**
  `StateAwareWarpFunction` implements PyTorch's native `.jvp()` extension hook; each
  wrapped operator's `OperatorSpec` carries a hand-derived `JVPSpec` (relaunching the
  same kernel on tangent arrays for value tangents — exact by linearity — or a
  separate hand-derived geometry-JVP kernel for position/support/mass tangents).
  Composable: chaining two wrapped operators inside one
  `torch.autograd.forward_ad.dual_level()`, feeding the first's dual output as the
  second's input, propagates the tangent through correctly with no manual wiring —
  confirmed by `warpSPHCore`'s own
  `tests/operations/test_forward_mode_dual_wrapper.py`. What still does not work is
  *nested* dual levels (needed for a Hessian action, not a Jacobian action) — a hard
  PyTorch limitation reproduced on bare `x**3`, not a `warpSPHCore` gap.
- **Scoped to exactly six operators plus Covariance**: Density, Interpolate,
  Gradient, Divergence, Curl, Laplacian, Covariance — with `GradientScheme`/
  `LaplacianScheme` variants and CRK/renorm correction paths (Laplacian only for
  Brookshaw/Dot/Default, never Naive). **Nothing outside that set has any JVP path**
  — the momentum equation, mDBC, surface detection, and every other production
  kernel were never attempted (`warpSPHCore`'s own residual-open-problems tracking,
  item 4). An `f` built partly from wrapped operators and partly from anything else
  gets exact JVPs for the wrapped part and nothing for the rest — there is no partial
  credit, and no automatic detection that a term was skipped.
- **This changes the practical shape of rung 3 below** (an exact-JVP matvec is no
  longer torch-only — it now also reaches warp-native SPH states, for an `f` that
  qualifies), **without changing rung 1's argument for existing at all** (an `f` with
  any unwrapped term still has no exact-JVP path, full stop, and needs FD).

The question "can I run `f` under forward-mode AD to get the Jacobian action?" splits
into two needs that are usually conflated, and that have **asymmetric, and now more
nuanced,** backend support.

| Need | What it requires | torch | warp |
|---|---|---|---|
| **Solving** the stage equation with Newton: `(I − dt·a_ii·J)·δ = −G` | Jacobian-*vector* products `J·v` — **forward mode** (`jvp`) | `torch.func.jvp` ✓ | engine: ✗ no forward mode. `warpSPHCore`: ✓, but only for `f` built entirely from the six operators + Covariance above — see the correction. |
| **Differentiating through** a converged solve, for training | *vector*-Jacobian products `Jᵀ·λ` via the implicit function theorem — **reverse mode** (`vjp`) | ✓ | `wp.Tape` ✓ |

So warp has exactly the mode the *gradient* needs and, for an arbitrary `f`, still
lacks exactly the mode the *solve* needs — the asymmetry the original write-up led
with is still real for the general case. It is only for the specific, bounded case of
an `f` assembled from `warpSPHCore`'s wrapped operators that the solve side gets a
matching capability now, not the general one implied by "warp gained forward mode."

**It does not block anything even in the general case, because Newton does not
actually need AD.** An inexact Newton needs the *residual* to be exact — and it is,
it is just `f` — while the matvec `J·v` only has to be good enough to produce a
descent direction. A finite-difference directional derivative
`J·v ≈ (f(Y+εv) − f(Y))/ε` costs one extra RHS evaluation, needs no AD of any kind,
and works identically under torch and warp, for *any* `f`, wrapped-operator or not.
This is why FD stays the generic fallback (rung 2) rather than something the JVP
finding replaces.

[verified] Backward Euler with a purely finite-difference Jacobian, no autodiff:
bounded and correctly L-stably damped at every stiffness up to `dt·ω = 1000`, where
Picard diverges past `dt·ω = 1` (table in §3.2). FD accuracy is not the limiting
factor.

**Recommended solver ladder**, cheapest first:

1. **Fixed-count Picard (2 iterations).** Non-stiff. No AD, no norm, no branching.
   Unrolls to a fixed-depth autograd graph, so it is differentiable in both backends
   by construction, graph-capturable, and deterministic. **This covers the primary use
   case and is what Phase 2 ships as the default** — though not the *only* thing
   `FixedPointSolver` can do: it already accepts `tol=`/`norm=` for a bounded,
   convergence-checked variant (see the note appended to §3.6's DIRK section), which
   is the right choice whenever CUDA-graph capture and the fixed-unroll-depth
   semantics for training aren't actually load-bearing for the caller — a purely
   forward simulation, or one whose gradients come from a separately-managed adjoint
   rather than backprop through this solve, needs neither.
2. **JFNK with FD matvecs.** Stiff, backend-agnostic, no capability gate — works for
   *any* `f`. Needs the flatten/unflatten bijection over integrated fields
   (mechanical — the field metadata already names them) plus GMRES. **Built
   2026-08-24 — `docs/plans/wcsph-jfnk.md` Phase A, `JFNKSolver` in `jfnk.py`.** Opt-in only
   (`solver=JFNKSolver()`), validated against the implicit wave-equation example:
   reproduces the hand-rolled CG reference to solver tolerance through the generic
   DIRK driver, and succeeds (bounded, correctly L-stably damped) at a stiffness
   (`dt=2.0` vs. the case's own CFL-scaled `dt≈0.006`) where Picard(20) blows up
   past `1e10` — the same signature `test_dirk.py`'s oscillator probe measures.
3. **Exact-JVP matvecs.** `torch.func.jvp` for torch states (a torch-only fast path,
   as originally scoped); `warpSPHCore`'s dual-tensor composition for warp-native
   states, but **only** when `f` is built entirely from the six wrapped operators +
   Covariance (§3.4's correction) — for anything else this rung does not exist, and
   the implementation must say so loudly. **Every custom (non-wrapped) operator in
   `warpSPHCore` shares the same `StateAwareWarpFunction`/`OperatorSpec` dispatch
   path as the wrapped ones** — a JVP-based matvec that walks that dispatch path must
   treat "no `JVPSpec` registered" as a hard error for that call, not a silent
   skip/zero-fill, or a caller composing a not-yet-wrapped op into `f` would get a
   matvec that is quietly wrong (some terms present, some silently dropped) instead
   of a matvec that fails to build at all. A speed/robustness optimisation over rung
   2 for the `f`s that qualify, never a substitute for it. **Built 2026-08-24
   alongside rung 2 — `jvp_matvec` in `jfnk.py`**, opt-in via
   `JFNKSolver(matvec='jvp')`. One finding along the way, not anticipated by this
   section: seeding *every* `integrated` field as a dual tensor unconditionally hits
   a real `torch.autograd.forward_ad` internal-assertion bug in `warpSPHCore`'s
   bridge whenever one field's tangent is exactly zero while another's isn't in the
   *same* `dual_level()` call — not a rare edge case, it's exactly the shape of the
   wave equation's own `v(0)=0` initial condition (`du/dt=v=0` identically at the
   first Newton iterate). Fixed at the JVP-seeding layer: skip `make_dual` for a
   field whose tangent slice is identically zero and leave it primal, which is exact
   by linearity (not an approximation) and still lets a live field's tangent
   propagate correctly to every output, including one for a field that was itself
   left primal. See `docs/plans/wcsph-jfnk.md` Phase A3 for the full account.
4. **User-supplied `solve_linear`.** An ISPH code already owns a pressure-projection
   solve and will always beat a generic Krylov method. This should be the contract;
   1–3 are the fallbacks.

**The JFNK cost model, per stage (2026-09-08).** One `JFNKSolver.solve` call costs,
in right-hand-side evaluations:

- **Residual evaluations.** One per outer Newton iteration plus one final
  evaluation that verifies the last correction. That count is what
  `SolveDiagnostics.rhs_evaluations` reports. A converged solve typically spends a
  few of the default 20-correction budget on smooth stage systems; the count grows
  with the stage system's nonlinearity and with `dt` past the non-stiff regime
  (measured: mean corrections `3.1 → 11.6` as the `dt` multiplier over the
  acoustic CFL goes `1× → 30×` on `docs/plans/wcsph-jfnk.md` Phase B step 6's acoustic core).
- **GMRES matvecs.** Each correction solves one linear system with restarted GMRES
  (default restart 30), one matvec per Krylov iteration; the total across all
  corrections is `SolveDiagnostics.gmres_iterations`. **Each matvec is itself one
  extra `step` evaluation** — the perturbed state in `matvec='fd'` mode, the
  dual-level state in `matvec='jvp'` mode — so the total number of `step`
  evaluations a stage solve actually performs is
  `rhs_evaluations + gmres_iterations`, not `rhs_evaluations` alone. The base
  residual `G(Y)` is shared across every Krylov iteration of one correction
  (that is the whole point of the `y_flat`/`G_y` arguments to `fd_matvec`), which
  is what keeps a matvec at one evaluation rather than two.
- **FD versus exact-JVP matvec.** Same asymptotic cost (one `step` evaluation
  each); the difference is accuracy, not count. The FD step is Knoll-Keyes scaled
  and carries its own truncation noise, concentrated in small-magnitude fields of a
  multi-scale flat vector (E1.5's finding); the exact JVP is exact by linearity
  for the qualifying `f`s and measurably uses no more GMRES iterations
  (`test_jfnk.py`'s `iters_jvp <= iters_fd` check). On the turbulent-flow probe
  (`docs/plans/wcsph-jfnk.md` E1.6) that accuracy difference is the thing that keeps the
  large-`dt` forced branch tracking the reference where the FD matvec starts
  drifting — reach for `matvec='jvp'` there, when the `f` qualifies.
- **Measured scale** (acoustic core, `nx=24`, Backward Euler,
  `docs/plans/wcsph-jfnk.md` Phase B step 6): wall clock `13.75 → 1218.69` ms/step from
  `1×` to `30×` the acoustic CFL, against plain explicit RK4's `2.87` ms/step at
  its native `1×`. JFNK is not a wall-clock win at a `dt` where explicit is still
  stable; it is the only bounded option past explicit's own limit, and it stays
  roughly flat per unit of simulated time (ms per `dt`-unit) through `10×`.

**Why a preconditioner is the scale risk (roadmap Phase 2) — landed 2026-09-09.** Everything above
holds at the resolutions measured; the part that does not obviously hold at
particle scale is the *unpreconditioned* GMRES iteration count. The stage operator
`I − dt·a_ii·J` of a diffusion-like term has eigenvalue spread that grows with
resolution (wavenumber content up to `~h^{-2}`), and GMRES without a
preconditioner needs iteration counts that grow with that spread; the FD matvec's
noise floor rises with particle count for the same reason (more summed round-off
per evaluation's reductions — `JFNKSolver`'s stagnation tracking exists because
the float32 floor is resolution-dependent, ~1.5e-4 at 16K particles,
~2.3e-3–2.8e-3 at 1M, per its own docstring). A diagonal or block-diagonal
preconditioner — classically, the explicit-Euler step of the same stage operator
applied inside GMRES — is what should make the Krylov count stop growing with
`N`. That hook now exists (roadmap Phase 2, landed 2026-09-09):
`gmres(..., preconditioner=, preconditioning='right'|'left')` and
`JFNKSolver(preconditioner=, preconditioning=, preconditioner_context=)`, with
the caller-facing contract `preconditioner(v, state, context) -> vector` —
`state` is the current Newton iterate and `context` merges
`preconditioner_context` under `{'state': Y}`, so a preconditioner that
approximates `J_G(Y)^{-1}` can read the iterate's own fields. `identity_` and
`diagonal_preconditioner` are the shipped examples (`tests/test_preconditioner.py`);
the no-preconditioner path is untouched and a supplied identity is *bitwise*
identical to none in both modes. Two measured findings from landing it: (a) a
*scalar* diagonal does not help a uniform-coefficient operator — it only rescales
the spectrum, leaving the condition number unchanged (1D Laplacian, `dt=50`:
147 → 146 right) — while a *per-DOF* diagonal is a strong preconditioner exactly
when the stiffness varies by particle, which is the realistic SPH case (per-
particle `c`/damping): on `A = I + dt·(diag(K) + εL)` with `K ∈ [1, 100]`,
`n=200`, GMRES drops 76 → 32 (right) / 6 (left); the size sweep in
`benchmarks/jfnk_preconditioner_benchmark.py` holds 9–13 iterations preconditioned
against 104–173 unpreconditioned through `n=1024` (`images/jfnk_preconditioner_`
`benchmark.png`, 9–12× total stage-map evaluations). (b) On the real SPH operator
(`warpSPH`'s wave equation, backward-Euler 2N stage), the block-lower-triangular
factor of the stage Jacobian — `L^{-1}b = [b_u, (b_v + dt·c²·Lap(b_u))/(1+dt·damping)]`,
one `warpOperationJVP` apply, no dense matrix — cuts the stage solve 41 → 17
(`matvec='fd'`) and 33 → 7 (`matvec='jvp'`) at `nx=32`, with the preconditioned
solve agreeing with the unpreconditioned one and the hand-eliminated CG reference
(`warpSPH/tests/test_implicitWaveEquation.py`, Step 7). The unpreconditioned path
remains the default, and the per-step numbers above stay the baseline a new
resolution should be compared against.

**Caution: the probe's Jacobian does not scale, and this matters a lot for SPH.**
`newton_probe.py` (§3.9) builds a *dense* Jacobian one column at a time — one extra
RHS evaluation per unknown — which is fine for the probe's 3–6-variable oscillator and
is why it can afford `max_iterations=20`. For a real state that is completely
intractable: an SPH system has `N` particles times several integrated fields each, so
`N` is 10⁶–10⁷ and a dense per-column FD Jacobian would cost 10⁶–10⁷ extra force
evaluations *per Newton iteration*. Never build one.

The distinction that has to survive from probe to implementation is between **forming
`J`** (fine for a handful of unknowns, never do it above that) and **applying `J` to a
single vector** (`J·v ≈ (f(Y+εv) − f(Y))/ε`, one extra RHS evaluation regardless of
`N`). Rung 2 above — JFNK — only ever needs the latter: GMRES calls the matvec once
per Krylov iteration, not once per unknown, so its cost scales with the number of
Krylov iterations (typically single digits to a few dozen for a well-conditioned stage
system), not with particle count. Keep the two operations named differently in the
implementation (e.g. `jacobian_column` vs `jacobian_vector_product`) so the dense form
the probe uses for a fast correctness check cannot be copy-pasted into the
particle-scale path by accident.

**On differentiating through the solve.** With a *fixed* iteration count you should
simply unroll — 2 extra RHS evaluations in the graph — and for ML that is arguably the
correct semantics anyway, since it is the gradient of what is actually computed at
inference rather than the gradient of an idealised converged solution. The implicit
function theorem adjoint only becomes worth its complexity when iterating to a
tolerance, i.e. Phase 3 and later. Since IFT needs reverse mode, it works in both
backends when it is needed.
