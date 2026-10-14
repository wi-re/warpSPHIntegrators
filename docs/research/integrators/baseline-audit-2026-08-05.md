> Historical audit (August–September 2026). Counts, line numbers and open-work
> statements describe that date. [Current architecture](../../development/architecture.md)
> and [execution plan](../../../REPOSITORY_PLAN.md) supersede its task statuses.

# `warpSPHIntegrators` — architecture notes and open work

Analysis date: 2026-08-05, against `6b9e1d0` (v0.5.0), conda env `warp`
(Python 3.14.6, torch 2.13.0+cu130, warp 1.15.0, numpy 2.4.6). Updated 2026-08-07 for
`e327d8a`, which renamed the package (§2.4) — distribution and import name are both
`warpSPHIntegrators` now, and every path below points at `src/warpSPHIntegrators/`.

Every claim marked **[verified]** was reproduced by running the code. For §0's
numbers the probe is [benchmarks/step_reuse_convergence.py](../../../benchmarks/step_reuse_convergence.py),
described under [Reproducing the results](notes-index.md#reproducing-the-results); for §3's numbers
the probes are described in [§3.9](notes-index.md#39-the-probes-behind-these-numbers).

This document originally carried a defect-by-defect audit — misapplied `priorStep`
reuse, wrong stage times, inconsistent cloning defaults, a broken embedded-pair path,
wrong registry metadata, and README/`pyproject.toml` drift. Every item in it was fixed
for v0.5.0 and is now held down by the 960 tests in `tests/`; the write-ups were removed
from this document to keep it focused on what is still open. `git log -- NOTES.md` has
the full history if the reasoning behind a fix is needed again.

---

## 0. Status

**Fixed in v0.5.0:** `priorStep` first-stage reuse now knows which schemes it is valid
for (`warpSPHIntegrators.reuse`: `step_reuse_order` / `supports_step_reuse` / `is_fsal`, plus
three genuinely FSAL tableaus — Bogacki–Shampine 3(2), Dormand–Prince 5(4), Cash–Karp
5(4) — so reuse is lossless where it matters); every scheme evaluates its stages at
the correct time, including stage 0; the embedded-pair path returns a real error
estimate (`IntegrationResult.error`); cloning is behavior-driven and dispatched by
value type, so `wp.array` and containers are actually copied instead of aliased;
registry metadata (order, `dissipation`, enum-name lookup) is correct; and assorted
hygiene issues (dead code, duplicate definitions, an unreachable `@torch.jit.script`)
are gone.

**Headline numbers** (`python benchmarks/step_reuse_convergence.py --all`):

- Every registered scheme reaches its claimed order on an autonomous, a
  non-autonomous, and a nonlinear problem. Before, six schemes collapsed to order 1
  on anything time-dependent.
- Dormand–Prince 5(4) under reuse is **bit-for-bit identical** to Dormand–Prince
  without reuse, at one fewer right-hand-side evaluation per step. Same for
  Bogacki–Shampine 3(2).
- Every reuse prediction now matches measurement for every scheme, on both problems.

### Still open

- **Warp backend** (§2.1, §2.2) — cloning is fixed and the RK weight loop no longer
  over-clones (9 → 5 clones for RK4, the theoretical minimum). What's left is
  substantive: buffer pooling, a fused `axpy` kernel, graph capture, and the
  torch-autograd vs `wp.Tape` decision.
- **SPH-awareness** (§2.3) — adaptive `dt` (the embedded-pair machinery it needs is
  now in place; it needs the driving loop and a step-rejection path) and particle
  masking are still open. The neighbour-list reuse policy this section used to flag as
  missing turned out to be a non-issue for the actual downstream simulation — see §3.0.
- **Packaging** (§2.4) — **done in `e327d8a`.** The top-level package is now
  `warpSPHIntegrators`, matching the distribution name; nothing is left open here
  beyond the version bump that should accompany the break.
- **Multistep and implicit** (§3) — the remaining entries in the README's "Known
  Limitations". Both are much cheaper here than for a general-purpose library, because
  the surrounding simulation does not resort particles, holds `dt` constant, and
  carries its neighbour list through the state (§3.0). **All three phases — 0 (§3.5),
  2 (§3.6, DIRK), and 1 (§3.6, multistep) — are done as of 2026-08-24.** Fourteen new
  schemes are registered: seven DIRK (Backward Euler, Implicit Midpoint, Trapezoidal,
  SDIRK2, landed 2026-08-24; TR-BDF2, ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA landed
  2026-09-09 with SUNDIALS ARKODE's published coefficients) and seven explicit
  multistep (Adams-Bashforth 2-5, Adams-Bashforth-Moulton 2-4 PECE), every one
  verified to reach its claimed convergence order empirically. **Implicit midpoint's headline symplectic
  property needs a caveat the original scoping missed**: at the shipped
  `FixedPointSolver` default (2 fixed Picard iterations), its long-run energy
  behaviour measures dissipative, not symplectic — the textbook bound only returns
  with a more-converged solver (§3.6 has the full finding). The multistep schemes
  landed with zero comparable surprises — the "reuse `_weighted_update`, bootstrap
  from Dormand-Prince, thread `StepHistory`" design didn't need new machinery that
  could itself be wrong, and the full suite (1103 → 1380 passing tests) went green on
  the first run after fixing one pre-existing test's exclusion criteria. What
  remains open in this area — BDF6 and the BDF3-5 cold-start ceiling, the coupled-block
  s=3 pair (Gauss-Legendre / Radau IIA s=3) and Lobatto IIIA-IIIB, the remaining Phase 12
  families (generalized-alpha / HHT-alpha, SSPRK(5,4), high-order symplectic composition,
  one-step RKN), and the Phase 9/10 warm-start and cold-start-gradient items — is tracked
  with per-item triggers in `docs/plans/implicit.md` (open-items-only since 2026-09-15);
  each is individually scoped (see the sections cited above) and gated on a concrete
  downstream need, per the recommendation at the end of §3.8; none of it is a groundwork
  gap the way Phase 0 was. (High-order IMEX/ARK landed as
  Phase 5 on 2026-09-09 — `ark.py`, see §3.6. Preconditioned JFNK landed as Phase 2
  on 2026-09-09 — left/right-preconditioned `gmres`, the 3-arg
  `preconditioner(v, state, context)` hook on `JFNKSolver`, identity/diagonal
  examples, and the size-sweep benchmark; see §3.4. Higher-order BDF and true
  Adams-Moulton landed as IMPLICIT_ROADMAP Phase 4 on 2026-09-09 — BDF4/BDF5
  (A(α) 73.35°/51.84°, state-snapshot history) and the JFNK-corrected
  Adams-Moulton AM2-AM4 (derivative history, optional AB predictor), see §3.6.
  The broadened nonlinear/stiff benchmark suite landed as IMPLICIT_ROADMAP
  Phase 8 on 2026-09-09 — five new problem factories (stiff Prothero–Robinson
  in both signs, stiff damped oscillator, van der Pol, Robertson kinetics,
  semi-discrete diffusion), the damped-oscillator amplification matrices in
  `stability.py`, `tests/test_benchmarks.py` (52 tests), and
  `docs/assets/stiff_benchmark_suite.png`; see §3.10.
  First-stage reuse for the stiffly accurate DIRK tableaus with an explicit first
  stage landed as IMPLICIT_ROADMAP Phase 10 on 2026-09-10 — `reuse.dirk_reuse_analysis`
  plus the driver's `_dirk_reuses_first_stage` accept `priorStep` losslessly for
  Trapezoidal, TR-BDF2, and both ESDIRKs (order preserved; one explicit RHS eval/step
  saved within JFNK noise, `tests/test_dirk.py`); see §3.6's Phase 10 note. RKC/RKL
  stabilised explicit landed as IMPLICIT_ROADMAP Phase 12 on 2026-09-11 — `rkc.py`
  (RKC1 / RKC2 / RKL2: per-step stage count from `stage_count`, real-axis stability
  K(s), no nonlinear solver), which on the diffusion benchmark reaches a fixed error
  with fewer total RHS evaluations than the implicit baselines under the default JFNK;
  see §3.13. Adaptive step control landed as IMPLICIT_ROADMAP Phase 11 on
  2026-09-14 — `estimate_error_norm` / `propose_dt` helpers plus DP5 dense output,
  multistep refused explicitly, `tests/test_adaptive.py`; see §3.17. The coupled
  fully implicit RK pair landed as IMPLICIT_ROADMAP Phase 6 on 2026-09-15 —
  `fullyimplicit.py` + `fields.BlockState`: Gauss-Legendre 2 (order 4, A-stable,
  symplectic by measurement, not L-stable) and Radau IIA s=2 (order 3, L-stable,
  stiffly accurate), each with a null-stage order-2 companion, matrix-free JVP
  matvec, and the 256-DOF scale gate; see §3.18. The IMEX linear multistep family
  landed as IMPLICIT_ROADMAP Phase 12 on 2026-09-15 — `imexmultistep.py`:
  SBDF2/SBDF3 (BDF2/BDF3 implicit backbone + Lagrange endpoint extrapolation) and
  CNAB2 (trapezoidal backbone + AB2 increment), orders 2/3/2, pure-implicit limits
  and effective-explicit-limit stability pinned; see §3.19. Shu's order-4 SSP method
  SSPRK(10,4) landed 2026-09-15 — the last of Phase 12's named explicit families,
  since the bundled SSP/TVD set previously stopped at order 3, exactly where the
  explicit-SSP stage barrier bites — and the roadmap's four housekeeping items
  closed the same day (redundant `nonLagrangian` flag removed, dead `midPoint`
  import, orphaned README ABM row, Current-Baseline JFNK list); see §2.14, §3.20.)
- **A finding, not a defect:** Leap Frog, Velocity Verlet, PEFRL and VEFRL are only
  second/fourth order for a **separable** Hamiltonian, i.e. a force depending on
  position alone. With a velocity-dependent force — artificial viscosity, drag, any
  real SPH momentum equation — all four drop to first order. This is a property of the
  schemes rather than a defect here, but it is a sharp edge for SPH specifically. It is
  measured by `tests/test_convergence.py` on the `damped` problem and documented in
  the README scheme table. Symplectic Euler is not affected.

### 2.14 Housekeeping pass — done 2026-09-15

The four open housekeeping items on IMPLICIT_ROADMAP closed:

- **`nonLagrangian` removed.** `IntegrationScheme.nonLagrangian` (the metadata
  flag in [util.py](../../../src/warpSPHIntegrators/util.py)) agreed with `dissipation`
  for every registered scheme, so it carried no information, and nothing in the
  codebase read either flag. The copy-paste inconsistencies the flag's values
  had encoded were already reconciled when `dissipation` was re-derived from
  the symplectic sets, so the flag was simply deleted: the `IntegrationScheme`
  constructor loses its sixth positional argument and every registration drops
  the trailing boolean. The README's "removal candidate" bullet for the flag is
  retired, and the contributor example in the README no longer passes it.
- **Dead `midPoint` import removed.** `integration.py` imported `midPoint` but
  never registered it; the two `Midpoint` registrations build on butcher.py's
  `midpoint` tableau. No other `midPoint` references remain (grep-verified).
- **Orphaned README row relocated.** The Adams-Bashforth-Moulton 2–4 (PECE)
  row sat in the *Implicit* Multistep (BDF) table — an explicit method
  misplaced under BDF. It now sits in the Explicit Multistep table, directly
  after Adams-Bashforth 2–5.
- **Current-Baseline JFNK list updated.** It named DIRK, Newmark, BDF, implicit
  Adams-Moulton, and IMEX Euler, omitting ARK (which does default to
  `JFNKSolver`). It now names ARK, the coupled block RK pair (Gauss-Legendre 2,
  Radau IIA s=2), and IMEX multistep (SBDF2/SBDF3/CNAB2) as well.

---

## 1. What this library actually is

Despite the name, **there is not one line of NVIDIA Warp in it**
(`grep -rn 'warp\|wp\.' src/` → no hits). It is a pure-PyTorch, allocation-based
library of *explicit* ODE time-integration schemes, extracted from diffSPH
(`8c213c8 initial commit of integrators from diffSPH`). The "warp" in the name is
aspirational — it describes where this is meant to go, not what it does.

The design goal is a **solver-agnostic integrator driver**: the integrators never
touch `position`/`velocity`/`density` by name. They only know how to say "advance
component X by `c·dt` times derivative Y, optionally blended with a reference
state", and the *user's* system object decides what that means. That is a good
separation and it is the main asset here.

### 1.1 The four layers

| Layer | File | Role |
|---|---|---|
| **Field metadata** | [fields.py](../../../src/warpSPHIntegrators/fields.py) | `dataclasses.field` wrappers that stamp `metadata['behavior']` (`integrated` / `constant` / `copied` / `ephemeral` / `custom`) and `metadata['tags']` onto state fields. Drives generic cloning and tag-based lookup. |
| **Update specs** | [specs.py](../../../src/warpSPHIntegrators/specs.py) | Frozen dataclasses describing *what* an update does, without saying to which field: `ComponentUpdateSpec(derivative_dt, blend)`, `PositionUpdateSpec(+current_velocity_dt, +update_velocity_dt)`, `StateBlend(self_scale, reference_state, reference_weight)`. |
| **Dispatch** | [util.py](../../../src/warpSPHIntegrators/util.py), [protocol.py](../../../src/warpSPHIntegrators/protocol.py) | `applyPositionUpdate` / `applyVelocityUpdate` / `applyQuantityUpdate` / `applyStateUpdate` try the typed `apply_*_update` methods, else fall back to legacy `integratePosition` / `integrateVelocity` / … . Lifecycle hooks `initialize` / `preprocess` / `postprocess` / `finalize`. |
| **Schemes** | [butcher.py](../../../src/warpSPHIntegrators/butcher.py), [verlet.py](../../../src/warpSPHIntegrators/verlet.py), [ruth.py](../../../src/warpSPHIntegrators/ruth.py), [tvd.py](../../../src/warpSPHIntegrators/tvd.py), [euler.py](../../../src/warpSPHIntegrators/euler.py) | 23 registered schemes, all built from the four `apply*` primitives. Registry + lookup in [integration.py](../../../src/warpSPHIntegrators/integration.py). |

### 1.2 The generic update kernel

Everything bottoms out in two functions at [fields.py:267-310](../../../src/warpSPHIntegrators/fields.py#L267-L310):

```
update_component:   X ← self_scale·X + reference_weight·X_ref + Σᵢ dtᵢ·kᵢ.dX
update_position:    X ← (same) + current_velocity_dt·V   (semi-implicit drift)
                            + update_velocity_dt·k.dV     (Verlet correction)
```

Fields are located by **tag**, not by name: `get_tagged_attr(state, tag='position')`.
The system object is located via `role='reference_state'`. This is what lets one
`update_component` call serve every scheme.

### 1.3 One step, end to end (RK4)

```
initializeSystem(state, dt)                       → user.initialize()
currentState = state.initializeNewState()         → clone: constant+integrated copied, rest → None
k0, r0 = updateStep(...)                          → user.preprocess(); f(); user.postprocess()
for each row of the Butcher `a` matrix:
    currentState = state.initializeNewState()
    currentState = updateStateEuler(currentState, kᵢ, a·dt, copyState=False)   ← accumulate
    currentState.t = state.t + c·dt
    kⱼ, rⱼ = updateStep(...)
new_state = state.initializeNewState()
for each b:  new_state = updateStateEuler(new_state, kᵢ, b·dt)                 ← accumulate
new_state.t = state.t + dt
finalizeSystem(new_state, state, dt, rs, ks, b)
return IntegrationResult(state=new_state, stages=[StageResult(aux, update), ...])
```

`priorStep` (a `StageResult` from the previous step) can be passed in to skip the `k0`
evaluation. Whether that costs convergence order is a property of the tableau, not of
the caller — computed automatically by `warpSPHIntegrators.reuse` (`step_reuse_order`,
`supports_step_reuse`) and carried on the registered `IntegrationScheme`.

---

## 2. Open work: Warp backend and SPH-awareness

### 2.1 Allocation-per-stage fights Warp's execution model

The cheap fix already landed: the final weight-accumulation loop
(`butcher._weighted_update`) passes `copyState=False` and clones once up front, so RK4
takes 5 state clones per step — the theoretical minimum — rather than 9.

What's left is the substantive part. The whole design is **functional and
allocating**: `update_component` builds
`value = value * s; value = value + w * ref; value = value + dt * delta` — three fresh
tensors per component per stage. That is idiomatic for autograd-through-time, but it
is the opposite of what Warp wants (preallocated buffers, in-place kernel writes,
CUDA-graph capture). At SPH scale (10⁶–10⁷ particles) the allocation traffic alone
will dominate.

**Recommended direction:** keep the current spec/tag/protocol layer — it is
backend-agnostic and worth preserving — and add an out-of-place-vs-in-place switch at
the `update_component` level:
- a `WarpState` base whose `initializeNewState` pulls buffers from a per-step pool
  instead of allocating;
- an `axpy`-style Warp kernel (`x = s*x + w*ref + Σ dtᵢ·kᵢ`) with a variadic-`k`
  variant, so the whole accumulation is one launch;
- fixed stage counts per scheme so the pool is sized once and the step is
  graph-capturable. §3.4's fixed-iteration-count implicit driver is aligned with this
  rather than in tension with it — both want a step shape known in advance.

### 2.2 No differentiability story for Warp

The README's headline claim is "fully differentiable". Warp's `wp.Tape` has a
completely different gradient model from torch autograd (explicit tape, adjoint
kernels). A `torch`+`warp` hybrid needs `wp.to_torch`/`wp.from_torch` at the boundary,
or `warp.autograd`. **Decide which one is authoritative before writing the Warp
backend** — retrofitting is much worse than choosing up front.

§3.4 answers the narrower question of what an *implicit stage solve* needs: warp has
reverse-mode AD (what differentiating through a converged solve wants) but not
forward-mode (what a Newton solve's Jacobian-vector product would naively want), and
that asymmetry turns out not to matter because finite-difference directional
derivatives drive Newton just as well with no AD of any kind. That resolves the
implicit-solver corner of this question; the general "what backs `.backward()` for a
Warp state" decision above is still open.

**The torch half of the headline claim was false until 2026-09-10, and nothing caught
it because nothing tested it.** No test in the suite ever called `.backward()` through
a step — the only gradient coverage checked that `requires_grad` survived cloning — so
"fully differentiable" went unverified for the entire implicit family. It did not
hold: `gmres` builds its Hessenberg factor, its Givens rotations and its
back-substitution vector by *in-place element writes*, and recording that on the
autograd tape makes torch raise "one of the variables needed for gradient computation
has been modified by an inplace operation". Every DIRK, Newmark, IMEX and ARK scheme
hit it on step 1; BDF2+/AM hid it one step longer, because their explicit
Dormand-Prince cold start has to hand over to the implicit path before the solver runs
at all — which is exactly the kind of gap a "does it run?" smoke test cannot see.

Unrolling the solver would have been the wrong fix even if the tape had accepted it:
it differentiates *the path to* the fixed point rather than the fixed point, and under
`matvec='fd'` it would differentiate a divided difference. `JFNKSolver.solve` now runs
the whole iteration under `no_grad` and re-attaches gradients by the implicit function
theorem — with `G(y, θ) = y − step(y, θ)` and `G(y*, θ) = 0`, a cotangent `g` arriving
at `y*` is mapped to `λ = (I − Jᵀ)⁻¹ g` by one more matrix-free GMRES, this one driven
by ordinary reverse-mode VJPs instead of the forward matvec. Verified against the
closed-form amplification matrices of backward Euler, trapezoidal, implicit midpoint,
BDF1 and IMEX Euler on the linear oscillator: **agreement to 1.1e-16, and unchanged
across a `newton_tol` sweep from 1.0 to 1e-6**. That tolerance-independence is the
signature of the method — a gradient taken through the iterations would move as the
iteration count moved. `tests/test_gradients.py` (57 tests) pins all of it, including
that the forward trajectory stays bit-for-bit identical when gradients are off.

One implementation trap worth recording, because it fails *silently*: the adjoint's
`Jᵀv` products cannot be taken on the same graph the backward pass is traversing. A
hook on `out` that calls `torch.autograd.grad(out, ...)` re-enters the node the engine
is already holding and **deadlocks** — no exception, no timeout, just a hang. The fix
is to evaluate `step` a second time to give the adjoint solve a graph of its own,
which is why the differentiable path costs two extra `step` evaluations rather than
one.

### 2.3 Nothing in the API is SPH-aware

- **CFL / adaptive `dt`.** `dt` is a caller-supplied constant. Real SPH recomputes
  `dt` from `min(h/c_s, sqrt(h/|a|), ...)` every step. The embedded-pair machinery
  this needs is finished (`IntegrationResult.error`, three registered pairs); what's
  missing is the driving loop and a step-rejection path.
- **Particle count changes** (inflow/outflow/refinement) mid-step. All clone paths
  assume fixed shapes. Not a concern for the current downstream — see §3.0 — but a
  real gap for a general-purpose user.
- **`integrateSpecies` / fluid-vs-boundary masking.** The only masking support in the
  codebase is the `fluid_only` flag on `integrated()`, which is never read by
  anything. Boundary particles must not be integrated; that has to be a first-class
  concept, and §3.7's Phase 3 implicit work needs it — an unmasked boundary makes the
  stage system singular.
- **Neighbour-list reuse policy.** This used to be listed here as missing. It isn't:
  the actual downstream simulation already carries the neighbour list through the
  state and revalidates it cheaply rather than rebuilding it — see §3.0.

### 2.4 Packaging — done

This used to read: the distribution name (`sphWarpIntegrators`) and the import name
(`integrators`) differ, and `integrators` is an extremely collision-prone top-level
name on PyPI.

Both are now `warpSPHIntegrators` (`e327d8a`): the package directory moved from
`src/integrators/` to `src/warpSPHIntegrators/`, with the distribution renamed to
match and tests, scripts, README and `pyproject.toml`'s `package-data` key updated.
`import integrators` no longer works — that is the breaking change this was always
going to be, so the version bump that ships it should be the one that says so
(`pyproject.toml` still reads `0.5.0`). Remaining chores:

- `src/sphWarpIntegrators.egg-info/` is a stale build artefact from the old name and
  should be deleted (it is not tracked).
- `dist/` still holds wheels built under the old name.

---

## 3. Multistep and implicit methods

Scoping for the two remaining entries under README "Known Limitations". Everything
marked **[verified]** was measured by one of the five probes in
[§3.9](notes-index.md#39-the-probes-behind-these-numbers).
