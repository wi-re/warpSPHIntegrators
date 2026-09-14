# `warpSPHIntegrators` — architecture notes and open work

Analysis date: 2026-08-05, against `6b9e1d0` (v0.5.0), conda env `warp`
(Python 3.14.6, torch 2.13.0+cu130, warp 1.15.0, numpy 2.4.6). Updated 2026-08-07 for
`e327d8a`, which renamed the package (§2.4) — distribution and import name are both
`warpSPHIntegrators` now, and every path below points at `src/warpSPHIntegrators/`.

Every claim marked **[verified]** was reproduced by running the code. For §0's
numbers the probe is [scripts/step_reuse_convergence.py](scripts/step_reuse_convergence.py),
described under [Reproducing the results](#reproducing-the-results); for §3's numbers
the probes are described in [§3.9](#39-the-probes-behind-these-numbers).

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

**Headline numbers** (`python scripts/step_reuse_convergence.py --all`):

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
  the first run after fixing one pre-existing test's exclusion criteria. What remains
  open in this area — fully implicit RK, BDF6, and adaptive `dt` — is each
  individually scoped in §3.6/§3.4
  and gated on a concrete downstream need, per the recommendation at the end of §3.8;
  none of it is a groundwork gap the way Phase 0 was. (High-order IMEX/ARK landed as
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
  `images/stiff_benchmark_suite.png`; see §3.10.
  First-stage reuse for the stiffly accurate DIRK tableaus with an explicit first
  stage landed as IMPLICIT_ROADMAP Phase 10 on 2026-09-10 — `reuse.dirk_reuse_analysis`
  plus the driver's `_dirk_reuses_first_stage` accept `priorStep` losslessly for
  Trapezoidal, TR-BDF2, and both ESDIRKs (order preserved; one explicit RHS eval/step
  saved within JFNK noise, `tests/test_dirk.py`); see §3.6's Phase 10 note. RKC/RKL
  stabilised explicit landed as IMPLICIT_ROADMAP Phase 12 on 2026-09-11 — `rkc.py`
  (RKC1 / RKC2 / RKL2: per-step stage count from `stage_count`, real-axis stability
  K(s), no nonlinear solver), which on the diffusion benchmark reaches a fixed error
  with fewer total RHS evaluations than the implicit baselines under the default JFNK;
  see §3.13.)
- **A finding, not a defect:** Leap Frog, Velocity Verlet, PEFRL and VEFRL are only
  second/fourth order for a **separable** Hamiltonian, i.e. a force depending on
  position alone. With a velocity-dependent force — artificial viscosity, drag, any
  real SPH momentum equation — all four drop to first order. This is a property of the
  schemes rather than a defect here, but it is a sharp edge for SPH specifically. It is
  measured by `tests/test_convergence.py` on the `damped` problem and documented in
  the README scheme table. Symplectic Euler is not affected.

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
| **Field metadata** | [fields.py](src/warpSPHIntegrators/fields.py) | `dataclasses.field` wrappers that stamp `metadata['behavior']` (`integrated` / `constant` / `copied` / `ephemeral` / `custom`) and `metadata['tags']` onto state fields. Drives generic cloning and tag-based lookup. |
| **Update specs** | [specs.py](src/warpSPHIntegrators/specs.py) | Frozen dataclasses describing *what* an update does, without saying to which field: `ComponentUpdateSpec(derivative_dt, blend)`, `PositionUpdateSpec(+current_velocity_dt, +update_velocity_dt)`, `StateBlend(self_scale, reference_state, reference_weight)`. |
| **Dispatch** | [util.py](src/warpSPHIntegrators/util.py), [protocol.py](src/warpSPHIntegrators/protocol.py) | `applyPositionUpdate` / `applyVelocityUpdate` / `applyQuantityUpdate` / `applyStateUpdate` try the typed `apply_*_update` methods, else fall back to legacy `integratePosition` / `integrateVelocity` / … . Lifecycle hooks `initialize` / `preprocess` / `postprocess` / `finalize`. |
| **Schemes** | [butcher.py](src/warpSPHIntegrators/butcher.py), [verlet.py](src/warpSPHIntegrators/verlet.py), [ruth.py](src/warpSPHIntegrators/ruth.py), [tvd.py](src/warpSPHIntegrators/tvd.py), [euler.py](src/warpSPHIntegrators/euler.py) | 23 registered schemes, all built from the four `apply*` primitives. Registry + lookup in [integration.py](src/warpSPHIntegrators/integration.py). |

### 1.2 The generic update kernel

Everything bottoms out in two functions at [fields.py:267-310](src/warpSPHIntegrators/fields.py#L267-L310):

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
[§3.9](#39-the-probes-behind-these-numbers).

### 3.0 Constraints from the surrounding simulation

These are properties of the warpSPH simulation this library is written for. None of
them is derivable from this repository, and every one of them removes work that a
general-purpose ODE library would have to do. They are recorded here because the plan
below is only correct under them — if any stops holding, re-read §3.7.

| Constraint | Why it is that way | What it buys |
|---|---|---|
| **Particles are never re-sorted.** No inlets or outlets either. | Deliberate performance trade to keep file I/O simple, and because ML bindings need to diff two states by elementwise comparison rather than by matching identities. | Particle index `i` means the same particle at every step. A step history is therefore **valid indefinitely** — this dissolves the largest objection to multistep. |
| A `uid` integer tensor is carried in the state. | Lets particles be restored to their origin ids if that ever changes. | The cheap guard that turns "indices moved" from a silent wrong answer into an automatic restart. |
| **`dt` is constant.** Where adaptivity exists it is applied *around* whole steps, not within them. | Networks would otherwise have to generalise across `dt`. | Fixed multistep coefficients are correct as written. No variable-step coefficient regeneration, no Nordsieck / fixed-leading-coefficient machinery. |
| **The neighbour list is carried through the state**, moved over in `initializeNewState`, with a cheap velocity-Verlet-style validity check and a rebuild only when it fails. | Already the right design. | The "one RHS evaluation = one neighbour rebuild" assumption is **false here**. An implicit iteration costs one rebuild plus N cheap checks, not N rebuilds. This was the single largest cost objection to implicit methods and it does not apply. |
| Gradients: torch has forward *and* reverse mode; warp currently has **reverse only**. | `wp.Tape` is a reverse-mode tape. | Decides the solver design — see §3.4. |

### 3.1 Headline: both are cheaper than they look, and only one wall is left

The stage machinery already generalises further than the registry uses it.

- **A diagonally-implicit Runge–Kutta driver needs no new state algebra at all.**
  [verified] A ~60-line DIRK loop written against nothing but the existing public
  helpers — `initializeNewState`, `applyStateUpdate`, `explicit_step`, `updateStep`,
  `initializeSystem`, `finalizeSystem` — reaches full order on all three problems:

  | tableau | oscillator | forced | damped |
  |---|---|---|---|
  | Backward Euler (1) | 0.97 | 1.01 | 0.96 |
  | Implicit midpoint (2) | 2.00 | 2.00 | 2.00 |
  | Trapezoidal / Crank–Nicolson (2) | 2.00 | 2.00 | 2.00 |
  | SDIRK2, L-stable, γ=1−√2/2 (2) | 2.00 | 2.00 | 2.00 |

  The reason it falls out for free is that a DIRK stage equation
  `Y_i = y^n + dt·Σ_{j<i} a_ij k_j + dt·a_ii·f(Y_i)` has *exactly* the shape
  `butcher._weighted_update` already builds. Iterating it is a loop around code that
  exists.

- **The `b`-weight accumulation already accepts a list of updates with a list of step
  sizes** — `fields._resolve_delta` / `_accumulate` handle it, and nothing in the
  registry uses that path. That is precisely `y^{n+1} = y^n + dt·Σ_j β_j k^{n-j}`, so
  an Adams–Bashforth step is one `applyStateUpdate` call. [verified] AB2–AB4 and
  ABM2–ABM4 (PECE) all reach nominal order.

- **Two fixed Picard iterations are enough for a second-order implicit tableau.**
  [verified] With no convergence test and no early exit — a fixed, data-independent
  iteration count, which is what CUDA-graph capture and deterministic ML training both
  need:

  | iterations | implicit midpoint | SDIRK2 |
  |---|---|---|
  | 1 | 1.03 / 1.00 / 1.04 | 1.01 / 1.01 / 1.02 |
  | **2** | **2.01 / 2.00 / 2.02** | **2.00 / 2.00 / 2.01** |
  | 4 | 2.00 / 2.00 / 2.00 | 2.00 / 2.00 / 2.00 |

  (oscillator / forced / damped.) Each iteration buys one order, so `p` iterations
  suffice for order `p` from a trivial predictor. This matters more than it looks —
  see §3.4.

The remaining wall, and the only one §3.0 does not remove:

- **The library never sees the state as a vector, and never sees `f`'s Jacobian.**
  `f` is a black box returning a tagged update object. There is no `norm`, no `dot`,
  no flatten. A fixed-point stage solve needs none of those — which is why the probe
  works — but a *stiff* solve needs all of them.

The wall this section used to lead with — "one RHS evaluation is one neighbour
rebuild, so §2.3's neighbour-list item is a hard prerequisite" — **does not apply**,
because the adjacency is carried through the state and revalidated cheaply (§3.0).
Implicit iteration costs force evaluations, not neighbour searches.

### 3.2 What a fixed-point solve can and cannot do

Implicit methods exist for stiffness. A Picard iteration
`Y^{m+1} = y^n + dt·a_ii·f(Y^m) + …` converges only when `|dt·a_ii·L| < 1`, which is
the step restriction implicit methods are supposed to remove. Backward Euler —
unconditionally stable in exact arithmetic — with the probe's fixed-point solve, on
stiff oscillators at `dt = 0.1` [verified]:

| `k` | `dt·ω` | Picard(20) | Newton, FD Jacobian |
|---|---|---|---|
| 4 | 0.2 | 4.7e-01 | 4.7e-01 |
| 1e2 | 1.0 | 1.0e+00 (stalled, no damping) | 9.8e-04 |
| 1e4 | 10 | **diverged** (3.7e+239) | 3.8e-15 |
| 1e6 | 100 | **diverged** (nan) | 9.8e-17 |
| 1e8 | 1000 | **diverged** (nan) | 1.0e-18 |

Backward Euler is L-stable, so on a highly oscillatory undamped problem it should damp
hard towards zero and stay bounded; the Newton column does exactly that at every
stiffness, and is still bounded at `dt·ω = 1000`. Picard is already wrong at
`dt·ω = 1` — it stalls without damping at all — and blows up past that. So the
practical Picard limit is *tighter* than the textbook `|dt·a_ii·L| < 1`.

The conclusion is not "implicit needs Newton", it is **two different regimes with two
different answers**:

- **Non-stiff, which is the normal regime here.** A fixed 2-iteration Picard gives
  full order (§3.1), needs no norm, no Jacobian, no convergence test, no
  data-dependent control flow. That is Phase 2 and it is genuinely cheap.
- **Stiff.** Needs Newton, and Newton needs §3.4 — but that turns out to be much less
  of an obstacle than it first appears.

### 3.3 What implicit *does* buy, even without stiffness

**Implicit midpoint is strictly better than Velocity Verlet for SPH.** It is
symplectic, A-stable, symmetric, second order — and unlike the four splitting schemes
flagged in §0, it keeps second order for a velocity-dependent force. [verified], on
`oscillator` at `dt=0.05`, maximum relative energy error over the whole run:

| scheme | T=20 | T=160 | order on `damped` |
|---|---|---|---|
| Implicit midpoint | 3.1e-15 | 6.9e-15 | **2.00** |
| RK4 | 5.5e-06 | 4.4e-05 | 4.00 |
| Velocity Verlet | 2.5e-03 | — | **1.0** |

Bounded to machine precision over an 8× longer run, versus RK4's secular growth. For
any SPH momentum equation with artificial viscosity or drag — i.e. all of them — this
is the symplectic scheme the library currently does not have, at one nonlinear solve
per step with a *single* stage. It is the single highest-value item in this section.

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
   2026-08-24 — `JFNK_PLAN.md` Phase A, `JFNKSolver` in `jfnk.py`.** Opt-in only
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
   left primal. See `JFNK_PLAN.md` Phase A3 for the full account.
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
  acoustic CFL goes `1× → 30×` on `JFNK_PLAN.md` Phase B step 6's acoustic core).
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
  (`JFNK_PLAN.md` E1.6) that accuracy difference is the thing that keeps the
  large-`dt` forced branch tracking the reference where the FD matvec starts
  drifting — reach for `matvec='jvp'` there, when the `f` qualifies.
- **Measured scale** (acoustic core, `nx=24`, Backward Euler,
  `JFNK_PLAN.md` Phase B step 6): wall clock `13.75 → 1218.69` ms/step from
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
`scripts/jfnk_preconditioner_benchmark.py` holds 9–13 iterations preconditioned
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

### 3.5 Shared groundwork (prerequisite for both) — DONE 2026-08-24

| | Item | Where | Status |
|---|---|---|---|
| **S1** | `state_norm(state, rtol, atol)` — weighted RMS over integrated fields, Hairer–Wanner style, plus `state_difference`. `butcher._error_estimate` already builds a difference, so generalise rather than duplicate. | `fields.py` | **Done.** Both added; `_error_estimate` refactored to compute the embedded pair's two full solutions via the existing `_weighted_update` path and difference them with `state_difference`, rather than hand-linear-combining raw stage derivatives — equivalent for the additive update semantics every registered scheme uses (all 906 previously-passing tests still pass, including the embedded-pair suite), and correct for a nonlinear `apply_state_update` too, which the old approach silently assumed away. `state_difference` preserves a system wrapper (`get_reference_state` still works on its output) rather than unwrapping to the bare state. |
| **S2** | `StepHistory` — an ordered container of `(t, dt, derivative projection)`, carried on `IntegrationResult` and accepted as a kwarg, exactly as `priorStep` is today. `priorStep` becomes the one-entry degenerate case; do **not** ship two overlapping reuse mechanisms. | new `history.py`, `specs.py`, `butcher.py` | **Done**, with one deliberate correction to the original sketch: `history=` is wired through `RungeKuttaB` as **pure bookkeeping** (it only populates `IntegrationResult.history`) and does **not** auto-derive `priorStep`. A first draft did auto-derive it, which silently turned on first-stage reuse — and its order cost (`reuse.py`) — for every caller that merely wanted history threaded, on non-FSAL tableaus too, with none of `integration._with_reuse_guard`'s warnings, since those only fire when `priorStep` itself is in `kwargs`. Caught by `test_testing_run_threads_history_without_changing_the_trajectory` (RK4 drifted by ~1e-4 with `history=True` alone). Reuse still opts in exactly as before: pass `priorStep=history.as_prior_step()` explicitly alongside `history=`. |
| **S2g** | Two cheap history guards. **(a)** record `dt` in each entry and restart if it changes. **(b)** record the identity of the `uid` tensor (`data_ptr` + shape) and restart if it moves. | `history.py` | **Done**, inside `StepHistory.pushed`. |
| **S3** | `IntegrationScheme` metadata: `implicit: bool`, `steps: int`, `stiffly_accurate: bool`, `stability: 'A'\|'L'\|'A(α)'\|None`, `startup_order: int`. | `util.py` | **Done** — five fields added, all defaulted so every existing positional registration in `integration.py` is untouched and every registered scheme reads back the plain-explicit-RK defaults (`implicit=False`, `steps=1`, ...). `reuse.py`'s "multistep wants its own answer" is **not yet done** — left for Phase 1, which is the first thing that will actually have a multistep tableau to answer for. |
| **S4** | `NonlinearSolver` protocol: `solve(step, y0, norm, **opts) -> (y, converged, iterations)`. Ship `FixedPointSolver` with a **fixed iteration count** as the default (§3.1: 2 iterations for order 2). | new `solvers.py` | **Done**, with the callable renamed `step` (from the sketch's `residual`): what `FixedPointSolver` and every rung of the §3.4 ladder actually need is "the next iterate as a function of the current one" (`Y -> g(Y)`), not a signed residual, and naming it what it is avoids an `F(Y) = 0` framing nothing here uses yet. `converged=True` with no `tol` means "completed its fixed schedule", the only claim fixed-count Picard makes; `tol`+`norm` gives an early-exit variant using `state_norm`/`state_difference` from S1. |
| **S5** | Generalise `testing.run` / `conftest.ALL_SCHEMES`. Both assume a stateless one-step callable `scheme(system, dt, f)`. | `testing.py` | **Partially done.** `testing.run` grew an opt-in `history=` kwarg that threads a `StepHistory` alongside (or instead of) `priorStep`-based reuse, verified inert for every currently-registered scheme (`test_history_kwarg_is_inert_for_schemes_that_do_not_use_it`, parametrized over all 27). `conftest.ALL_SCHEMES`/`order_of` untouched — they still assume one-step, no-starter schemes, which is still true of everything registered; that generalisation is Phase 1's own prerequisite once a multistep scheme exists to register, not groundwork with no consumer yet. |

Verified 2026-08-24: full suite green, `954 passed, 54 skipped` (was `906 passed, 54 skipped`
at the top of this session — no regressions, 48 new tests in `tests/test_groundwork.py`).
`__init__.py` exports the new public surface (`state_difference`, `state_norm`,
`HistoryEntry`, `StepHistory`, `NonlinearSolver`, `FixedPointSolver`, `SolveResult`).

The `preprocess` caching policy that a general-purpose implicit driver would normally
need as a blocker is **not needed** here — the simulation already carries and
revalidates adjacency through the state (§3.0), by a better mechanism than a
library-level cache would be. The one thing to check when Phase 2 lands is that the
*implicit driver* reuses a single stage buffer across iterations rather than calling
`initializeNewState` per iteration as the probe does, so the adjacency is carried once
rather than re-cloned each time. That is also what §2.1's buffer pooling wants, so it
is aligned work, not a detour.

### 3.6 Valid schemes and what each costs

Effort is *marginal*, on top of the groundwork and the driver its group needs.
"tableau only" means the scheme is a data entry in `getButcherTableau` plus a
registry line — the same one-line cost that adding Dormand–Prince was.

#### Diagonally implicit RK (sequential 1-stage solves) — all seven shipped (four 2026-08-24, the rest 2026-09-09)

The probe driver is ~60 lines; budget ~150 for a registered one with the solver
protocol, verbose output and scheme metadata wired in. **Landed as `dirk.py`
(~340 lines including the 2026-09-09 tableaus) + seven registry entries**, reusing
`butcher.py`'s `_weighted_update`,
`_error_estimate` and `finalizeSystem` machinery directly rather than duplicating it,
per the design sketched here.

| Scheme | Order | Stages | Stability | Symplectic | Status |
|---|---|---|---|---|---|
| **Implicit midpoint** (Gauss–Legendre s=1) | 2 | 1 | A | **yes, exactly-solved** | **Done.** Registered `dissipation=True` — see the finding below. |
| **Backward Euler** | 1 | 1 | L | no | **Done.** |
| **Trapezoidal / Crank–Nicolson** (Lobatto IIIA-2) | 2 | 2 | A, not L | symmetric | **Done.** Explicit first stage (`a11=0`) exercises the driver's non-Picard branch. |
| **SDIRK2** (Ellsiepen, γ=1−√2/2) | 2 | 2 | L | no | **Done.** |
| **TR-BDF2** | 2(3) | 3 | L | no | **Done 2026-09-09.** Embedded pair is SUNDIALS ARKODE's published (2, 3) pair. |
| **ESDIRK3(2)4L[2]SA** (Kennedy–Carpenter) | 3(2) | 4 | L | no | **Done 2026-09-09.** Coefficients from SUNDIALS ARKODE v7.9.0; verified three ways (below). |
| **ESDIRK4(3)6L[2]SA** | 4(3) | 6 | L | no | **Done 2026-09-09.** Same source and verification as ESDIRK3(2)4. |

**TR-BDF2 and both ESDIRKs landed 2026-09-09**, once the transcription concern
below was resolved by a public, byte-for-byte reproducible coefficient source:
SUNDIALS ARKODE v7.9.0's `src/arkode/arkode_butcher_dirk.def` (the Kennedy–Carpenter
tableaus, entries `ARKODE_TRBDF2_3_3_2`, `ARKODE_ESDIRK324L2SA`,
`ARKODE_ESDIRK436L2SA`). The original concern still stands as the reason: both are
embedded, higher-stage tableaus whose published coefficients are easy to transcribe
wrong in a way a smoke test would not obviously catch — an order-2/3 convergence
measurement looks the same whether the low-order embedded weights are exactly right
or merely close, since only the *high-order* weights drive the propagated solution.
Each of the three was therefore verified beyond convergence: (1) the embedded
weights satisfy the low branch's order conditions exactly (e.g. TR-BDF2's `d`:
sum(d) = 1, d·c = 1/2, d·c² = 1/3); (2) one step expanded against the exact Taylor
solution via a symbolic Taylor model gives the claimed main and embedded order
(exact for the 4-stage ESDIRK3(2)4; numeric local-error rates against five
closed-form ODEs for the 6-stage ESDIRK4(3)6, where the exact-√2 arithmetic is too
slow symbolically); and (3) the stability function, computed exactly as a rational
function, is A-stable (|R(z)| ≤ 1 on the left half-plane, on a sweep fine enough
that grid artifact is ruled out) and L-stable (|R(-100)| = 2.65e-2 for
ESDIRK3(2)4, 7.57e-2 for ESDIRK4(3)6, 4.41e-2 for TR-BDF2; decay ~ C/|z|). One
naming note: the published names say "L[2]", but the exact stability functions
decay as O(1/|z|), so these register `stability='L'` and no O(z⁻²) claim is made.
The four 2026-08-24 tableaus were each verified two ways: by hand against their own
order conditions before writing any code, and empirically via `testing.convergence`
after (all four reach their claimed order to within 0.01 on `oscillator`/`forced`/
`damped` — `tests/test_dirk.py`).

**`reuse.py` was deliberately *not* touched, and the DIRK schemes deliberately do not
expose `.butcherTableau`.** That attribute is what `reuse._tableau_of` looks for to
run the explicit-scheme substitution analysis, whose reasoning assumes a stage is a
plain function of already-known states — not true for an implicit stage, whose value
depends on `dt` through the very solve reuse would try to skip. Leaving the attribute
off makes `step_reuse_analysis` correctly fall through to "no tableau, no recorded
reuse behaviour" rather than silently misapplying the explicit-tableau formula to an
implicit one. DIRK did not implement first-stage reuse at all then; `priorStep` was
rejected with the same warning every other non-reuse scheme in this library gives.

**Update (2026-09-10, IMPLICIT_ROADMAP Phase 10):** that "not yet" is now implemented
for the four stiffly accurate tableaus with an explicit first stage — Trapezoidal,
TR-BDF2, `ESDIRK3(2)4L[2]SA`, `ESDIRK4(3)6L[2]SA`. Each satisfies both halves of the
implicit FSAL condition: `a[0,0] == 0` / `c[0] == 0` (the first stage is explicit) *and*
`c[-1] == 1` / `a[-1] == b` (stiff accuracy), so the previous step's last stage *is*
`y^{n+1}` and its converged derivative is exactly `f(t^{n+1}, y^{n+1})` — the value this
step's first stage needs. Reuse is therefore lossless, saving one (cheap, explicit) RHS
evaluation per step. The split mirrors the design note above: `dirk.py` decides reuse
from the tableau alone (`_dirk_reuses_first_stage`, so the call-time check cannot drift
from the registration), while `reuse.py` gained a separate `dirk_reuse_analysis` — *not*
an extension of the explicit-tableau formula, which still does not apply to implicit
stages. It gates on the registered `stiffly_accurate` flag (that metadata's first real
consumer) with the same tableau check as a drift guard, and `step_reuse_analysis`
dispatches to it via `.dirkTableau` (now copied onto the reuse-guarded wrapper alongside
`.butcherTableau`). SDIRK2 (`a[0,0] != 0`, implicit first stage) and the single-stage
backward Euler / implicit midpoint still reject `priorStep` with the standard warning.
Measured (`tests/test_dirk.py`): reuse preserves the measured order on both problems, the
reuse run agrees with the no-reuse run to ~1e-11 on `oscillator` and exactly on `forced`
(the gap is JFNK's last-bit sensitivity to the reused initial guess, not a correctness
issue), and the saving is one explicit evaluation per reusable step within JFNK
iteration noise. **A subtlety for the Phase 8 cost panels:** those panels count
`SolveDiagnostics.rhs_evaluations`, which the DIRK driver records as `None` for explicit
stages, so the 4.50/step (TR-BDF2) and 10.72/step (ESDIRK4) figures do *not* include the
explicit first stage and do not move under reuse. The "4.50 → ~3.50" saving is a raw
`f`-call count, not a change in that panel number.

**A finding significant enough to change a registration flag, not just an
implementation detail:** implicit midpoint is the textbook symplectic Gauss-Legendre
s=1 method *only when its stage equation is solved to convergence*. The shipped
default — `FixedPointSolver(iterations=2)`, per this section's own "2 iterations
reach full order" measurement — leaves a real, uncorrected residual in the stage
equation, and that residual's effect on long-run energy conservation is not the same
question as its effect on local convergence order. Measured directly
(`tests/test_dirk.py::test_implicit_midpoint_energy_drift_bound_recovers_with_more_iterations`):
at the shipped default, max relative energy error grows secularly, ~9x over an
8x-longer run (`oscillator`, `dt=0.05`) — the same signature `test_hamiltonian.py`
uses to detect a *dissipative* scheme, not a symplectic one. Configuring the solver
with enough iterations to actually converge (~16, confirmed by a Cauchy check on the
iterate sequence) recovers the textbook bound: drift → ~1e-15, flat with `T`, matching
this section's own originally-cited numbers almost exactly. **Both things are true at
once and were conflated in this section's first draft**: "implicit midpoint is
symplectic" (a property of the exact method) and "the shipped 2-iteration default
gives you that property for free" (false — it gives you the *convergence order* for
free, not the *qualitative long-run energy behaviour*, which is arguably the scheme's
main selling point per §3.3). Registered `dissipation=True` to describe measured
default behaviour honestly; a caller who wants the textbook property back passes
`solver=FixedPointSolver(iterations=...)` explicitly, at the cost of more force
evaluations per step. This same distinction likely also affects the L-stable
tableaus' *stability* claims under Picard specifically — see the stiff-divergence
finding two paragraphs down — though stability (staying bounded) turned out to be a
smaller casualty here than symplecticity (staying bounded *and* non-dissipative) was.

**Note on the fixed-count default specifically (2026-08-24):** `FixedPointSolver`
already accepts `tol=`/`norm=` for a bounded, convergence-checked variant instead of
running a hard-coded iteration count — this was built into it from the start (§3.5
S4), not added for this note. "Ship a fixed count" is the right *default* because it
is what CUDA-graph capture and a fixed-unroll-depth training graph need (§3.7 pain
point 12), but neither of those is a universal requirement: a purely forward
simulation, or one whose gradients come from a separately-managed adjoint rather than
backprop through this solve, doesn't need either, and can pass a much larger
`iterations` cap with a real `tol` today to recover the textbook property above
without waiting on JFNK. The energy-drift recovery measured above used a bare
iteration-count sweep, not this `tol`-based path specifically; confirming the two give
the same answer is a small, concrete piece of follow-up if it matters to a caller
before `JFNK_PLAN.md`'s work lands.

**A second finding, matching an existing §3.2 claim exactly rather than contradicting
it:** L-stability is a property of the *exact* method, not of a truncated Picard
solve — already stated in §3.2 for backward Euler specifically, and now confirmed for
the registered DIRK path directly (`test_picard_diverges_on_a_stiff_problem...`,
oscillator at `k=1e6`, `dt·ω=100`): the fixed 2-iteration default returns a finite but
badly wrong answer (`x≈-9999`, not the ~0 an L-stable method should give), and running
more Picard iterations at the same stiffness makes the divergence explicit and
exponential (`10^4` at 2 iterations → `10^39` at 20), matching this section's own
`3.7e+239`-at-`dt·ω=10` figure in shape. This is not a new problem Phase 2 introduced;
it is the reason the ladder in §3.4 treats Picard as strictly the non-stiff rung and
gates JFNK on "a downstream that is actually stiff" rather than on "a downstream that
uses an L-stable tableau" — the tableau's stability class only pays off once the stage
equation is actually being solved, which Picard alone does not guarantee.

**One implementation bug caught by the existing test suite, not a design gap:** the
Picard loop's `step_fn` returns a *fresh clone* as the next iterate (via
`updateStateEuler(..., copyState=True)`), so the object that actually ran
`preprocess()` — and therefore carries `copied`-behaviour fields like a summation
density — was being discarded rather than passed to `finalizeSystem` as
`lastStageSystem`. `tests/test_copied_fields.py`'s existing parametrized suite (12
tests, extended for free the moment the four DIRK schemes joined the `scheme`
fixture) caught this immediately: `density` arrived as `None`, the exact `copied`→
`ephemeral` regression that section originally guarded against for the explicit path.
Fixed by capturing the actual evaluated stage buffer in `step_fn`'s closure instead of
using the solver's returned iterate for anything but the stage's own derivative.

**A third, narrower finding, in the pre-existing test suite rather than the new
code:** `test_hamiltonian.py`'s growth-ratio heuristic ("dissipative ⟹ energy error
grows >2x over an 8x-longer run") assumed every dissipative scheme dissipates
*slowly enough to keep growing* — true for the six RK schemes it was written against,
false for an L-stable scheme fast enough to hit its own floor (all initial energy
gone) before the short horizon even ends. Backward Euler saturates near 100% relative
energy loss well inside `T_SHORT`, so its growth ratio reads ≈1.0 — "bounded", the
same signature a symplectic scheme gives, for the opposite reason. Added a
`SATURATES_EARLY` exclusion (mirroring the file's existing `UNSTABLE` one) rather than
weakening the heuristic itself, since the heuristic is still correct for every scheme
slow enough for it to apply to.

`reuse.py`'s `tableau_reuse_analysis` was **not** extended to understand "stiffly
accurate" (`a[-1] == b`) as an implicit FSAL analogue — correctly so, since DIRK reuse
was not implemented at all this round (see above), so there is nothing yet for that
extension to serve.

**Update (2026-09-10, Phase 10):** it now is, as a *separate* `dirk_reuse_analysis`
rather than an extension of the explicit-tableau formula — the substitution argument
still does not apply to implicit stages, so the implicit case is decided on the DIRK
terms (`a[0,0] == 0` + stiff accuracy, gated by the registered `stiffly_accurate`
flag) instead. See the Phase 10 note above for the mechanism and the measured results.

#### Fully implicit RK (needs a *coupled* s·N-unknown solve — a different solver shape)

| Scheme | Order | Stages | Stability | Symplectic | Marginal effort |
|---|---|---|---|---|---|
| **Gauss–Legendre s=2** | 4 | 2 | A | **yes** | 4–6 d (block solve) |
| **Gauss–Legendre s=3** | 6 | 3 | A | **yes** | +1 d after the above |
| **Radau IIA s=2 / s=3** | 3 / 5 | 2 / 3 | L, stiffly accurate | no | +2 d — the gold standard for stiff ODEs |
| **Lobatto IIIA–IIIB pair** | 2s−2 | s | A | partitioned-symplectic | 3 d, and only pays off for separable Hamiltonians |

Not recommended until something downstream demands it. The coupled solve is a genuine
step up in solver machinery, not more tableau data — and its dense/block Jacobian is
the same scale trap as §3.4's caution, worse: an `s·N × s·N` system rather than
`N × N`.

#### Linear multistep — Adams-Bashforth/-Moulton shipped 2026-08-24

| Scheme | Order | Evals/step | History | Implicit | Status |
|---|---|---|---|---|---|
| **Adams–Bashforth 2–5** | k | **1** | k−1 updates | no | **Done.** |
| **ABM predictor–corrector (PECE)** | k | 2 | k−1 updates | no (fixed corrections) | **Done**, for orders 2-4. |
| **Adams–Moulton 2–4** as a true corrector | k | solve | k−1 updates | yes | **Done 2026-09-09** (IMPLICIT_ROADMAP Phase 4): `multistep.AdamsMoulton` — the DIRK/JFNK solver applied to the multistep residual exactly as feared, with an optional matching-AB predictor; AM2 = trapezoidal (A-stable), AM3/4 bounded region; beats same-order PECE by ~10x on the stiff nonlinear relaxation (`tests/test_am.py`). |
| **BDF1–2** | 1, 2 | solve | k states | yes, A-stable | **Done** (`bdf.py`, JFNK-closed; BDF1 = backward Euler). |
| **BDF3–5** | 3–5 | solve | k states | yes, A(α)-stable only (86.03° / 73.35° / 51.84°) | **Done 2026-09-09** (BDF3 with the JFNK work; BDF4/5 in IMPLICIT_ROADMAP Phase 4 — coefficients from the exact order conditions, zero-stability and the A(α) cone angles measured in `tests/test_bdf.py`; history carries state snapshots, `f` only at the new grid time, so full order on non-autonomous problems). |
| **BDF6** | 6 | solve | 6 states | yes, A(α)-stable only, smaller cone than BDF5 | not done — the A(α) cone keeps shrinking with order, so the stiffness payoff keeps getting worse; gated on a downstream that actually needs order 6 |
| **Störmer–Cowell / multistep Nyström** (`x'' = f(x)`) | k | 1 | k states | no | not done |
| **Gauss–Jackson** (8th-order Störmer–Cowell) | 8 | 1 | 8 | no | not done — niche, orbital mechanics |

**Implemented as `multistep.py`** (~215 lines): `y^{n+1} = y^n + dt·Σ_j β_j k^{n-j}` reuses
`butcher._weighted_update` directly, exactly the free lunch this section predicted — `ks` doesn't care
whether its entries came from this step's stages or past steps' `StepHistory` entries. Verified
empirically for every order (`tests/test_multistep.py`): AB2-5 and ABM2-4 all reach their claimed order
on `oscillator`/`forced`/`damped` to within 0.05. Bootstraps from Dormand-Prince 5(4) (order 5, at or
above every one of these) for the first `order - 1` steps, taking `result.stages[0].update` — the
FSAL-irrelevant fact that *every* explicit RK scheme's first stage sits at `f(t^n, y^n)` — as this
step's history entry.

**A design question this section didn't anticipate, resolved during implementation: what happens if a
caller never threads `history=`?** Unlike every one-step scheme, where `history=`/`priorStep=` are pure
opt-in bookkeeping, a multistep scheme's past derivatives exist *only* if the caller carries
`IntegrationResult.history` forward. Resolved by making the fallback safe rather than merely
documented: with insufficient history, these schemes keep re-running the Dormand-Prince starter
forever — verified bit-for-bit identical to calling `DormandPrince` directly
(`test_without_history_threading_falls_back_to_the_starter_exactly`). Since DP5 is higher order than
any of these, a caller who forgets to thread history gets a *correct but more expensive* trajectory,
never a silently wrong one — the same "safe but expensive" category as Item 3's DIRK stiff-Picard
finding, not the "silently wrong" category the `history=`-auto-deriving-`priorStep` bug from Phase 0 was.

AB's one evaluation per step is the real prize here: **one force evaluation per step
at order 4**, against RK4's four, with fixed coefficients that are exactly correct
under constant `dt` and a history that never expires under stable indexing (§3.0).
Both of the things that normally make multistep painful are absent. Note that no
linear multistep method is symplectic for a general Hamiltonian (Tang, 1993) —
measured directly for all seven (`dissipation=True`, ~7-9x energy-error growth over
an 8x-longer run, matching every other non-symplectic scheme registered here);
symmetric LMMs applied to `x''=f(x)` do show good long-time energy behaviour, but they
are subject to parasitic-root instability, so don't market them as symplectic.

#### IMEX / additive RK — the right answer for SPH, and the most work

**ARK3(2)4L[2]SA** and **ARK4(3)6L[2]SA** (Kennedy & Carpenter) pair an ESDIRK
tableau for the stiff terms with an ERK tableau for the rest: viscosity, surface
tension or the pressure term implicit, advection explicit. This is what production
stiff-SPH actually wants, and it sidesteps §3.2 — the implicit part is the part with a
tractable, often *linear* operator.

**Landed 2026-09-09 (Phase 5)** as `ark.py`, an additive (two-half) Runge-Kutta
driver. No protocol change was needed: it reuses the `IMEXRHS(explicit=..., implicit=...)`
split that IMEX Euler already shipped (an ordinary RHS stays fully implicit — the pure
ESDIRK limit), and each half is a plain `updateStep` evaluation, so `updateStep` still
returns one update object per callback.

| Scheme | Order | Stages | Stability | FSAL? | Status |
|---|---|---|---|---|---|
| **ARK3(2)4L[2]SA** | 3 | 4 | L[2] | no (implicit half SA) | **Done.** Coefficients from SUNDIALS ARKODE v7.9.0 (ERK + DIRK pair). |
| **ARK4(3)6L[2]SA** | 4 | 6 | L[2] | no (implicit half SA) | **Done.** Same source; the implicit half is a *different* ESDIRK design than the standalone ESDIRK4(3)6. |

Key findings, all verified in `tests/test_ark.py` (and numerically against the SUNDIALS
C driver, `arkode_arkstep.c`):

- **The propagated update is the additive `b`-weighted combination, not the last
  stage.** An ARK method is a *pair*; the two halves are not symmetric under stiff
  accuracy. The implicit (DIRK) half is stiffly accurate (`b_imp == A_i[last]`) but the
  explicit (ERK) half is not (`b_exp != A_e[last]`), so the combined method is **not**
  FSAL and `y^{n+1} != z_last`. SUNDIALS's own `IsStifflyAccurate` check (requires both
  halves) falls through to the additive update. Using the last stage instead drops the
  method to first order (measured); the additive combination keeps the full 3/4. This is
  why both schemes register `stiffly_accurate=False` (this codebase's flag tracks FSAL)
  and reject `priorStep`.
- **`b_exp == b_imp` elementwise for both pairs.** The "two weight vectors" reduce to a
  single vector (summing to 1) applied to `f_exp + f_imp`; the additive form is written
  per half so the driver stays general.
- **The ARK-context tableaus differ from the standalone ESDIRKs.** ARK3(2)4's implicit
  `a/b/c` equal the registered ESDIRK3(2)4, but the ARK pair shares one embedded `d`
  across both halves, and that `d` is *not* the standalone ESDIRK3(2)4's. ARK4(3)6's
  implicit half is an entirely different 6-stage ESDIRK (nodes 83/250, 31/50, 17/20 vs
  the standalone's (2−√2)/4, 5/8, 26/25), so its coefficients are carried in full.
- **Copied-field ownership when both callbacks are active.** Each callback runs
  `updateStep` on its own state object (one preprocess→f→postprocess each): the implicit
  callback owns the stage buffer (the JFNK solve and post-convergence re-eval run on it,
  the `dirk.DIRK` convention) and the explicit callback runs on a throwaway clone of the
  converged buffer. The final state's copied fields come from the implicit buffer.
- **Two-parameter IMEX stability** `R(z_exp, z_imp)` is computed exactly
  (`stability.imex_stability_function`); the pure limits recover the component stability
  functions, the implicit axis is L-stable, and a slice figure is generated by
  `scripts/stability_gallery.py` (`images/imex_stability_slices.png`).

### 3.7 Pain points and limitations

Ordered by what actually survives the §3.0 constraints. The three that used to head
this list — particle resorting, variable `dt`, and neighbour rebuild cost — are all
designed away by the surrounding simulation, and are kept here only as the conditions
under which the plan stops being valid.

**Multistep — what remains**

1. **Startup order caps the whole method.** [verified] Self-starting AB — ramping the
   order up as history accumulates — measures order **2.0 for AB3 and AB4 alike**,
   because the single AB1/Euler startup step contributes `O(dt²)` globally. With a
   Dormand–Prince starter for the first k−1 steps: AB3 → 2.97, AB4 → 3.99, ABM4 →
   4.03. So a high-order starter is not a refinement, it is the difference between
   AB4 and AB2. **Done 2026-08-24**: the starter (Dormand-Prince 5(4), unconditionally
   -- order 5, at or above every shipped order) is hard-coded inside
   `AdamsBashforth`/`AdamsBashforthMoulton`, deliberately *not* a `starter=` parameter
   a caller could override, which is what "must be registered scheme metadata, not
   caller policy" meant in practice: a caller-suppliable starter is exactly the
   footgun this pain point describes, one call away (pick a lower-order one, or one
   whose first stage isn't at `t^n`, and the whole run silently caps at the starter's
   own order instead of this scheme's).
2. **Backprop-through-time depth grows by `k`.** This is the one that bites the
   primary use case. History links step *n* to step *n−k* in the autograd graph, so a
   k-step method deepens BPTT by a factor of k on top of however many steps are already
   unrolled for training. Needs a documented detach policy — and note the inversion
   against implicit methods, which cost extra graph *width* per step but no extra depth
   (§3.4). If training-time memory is the binding constraint, that inversion may matter
   more than the RHS-evaluation count that motivated multistep in the first place.
3. **Memory.** k derivative-shaped buffers at 10⁶–10⁷ particles. Store the projection
   onto tagged derivative fields, not the user's whole update object (S2).
4. **`copied` / `ephemeral` semantics.** History entries were computed against a
   *previous* step's adjacency. Since that adjacency is carried and revalidated rather
   than rebuilt (§3.0), this is now a question about `lastStageSystem` and
   `copy_finalized_fields` bookkeeping rather than a correctness hazard — but read it
   before writing the code, not after.

**Multistep — dormant, guard rather than solve**

5. **Resorting would invalidate history silently.** Indices are stable today by
   deliberate design, so `k^{n-1}` stays meaningful indefinitely. If that ever changes,
   history becomes garbage with no error raised — a silent wrong answer, the same
   class of failure the value-dispatched cloning fix closed for state aliasing. The
   `uid` tensor makes the guard cheap (S2g-b): check its identity, restart on mismatch.
   Guard, don't build a permutation-tracking system for a case that does not exist.
6. **Variable `dt` would break the fixed coefficients.** Constant `dt` makes fixed
   β coefficients exactly correct. "For the most part" is doing real work in that
   sentence, so record `dt` per history entry and restart on change (S2g-a) — 0.5 d
   instead of the ~2 d of divided-difference coefficient regeneration a CFL-driven
   code would need.

**Implicit — what remains**

7. **The fixed-point default is not a stiff solver.** [verified] It stalls at
   `dt·ω = 1` and diverges past it (§3.2). Ship it as the default, because it covers
   the actual regime at 2 iterations per stage, but document the limit in the same
   breath and do not let the README imply otherwise.
8. **Boundary particles must be excluded from the solve** or the stage system is
   singular. §2.3's masking item becomes a prerequisite for Phase 3+, not a parallel
   track. (Phase 2's fixed-count Picard has no linear system and so no singularity —
   another reason to ship that first.)
9. **Convergence failure needs a step-rejection path**, the same machinery adaptive
   `dt` (§2.3) needs. Do not build two. Only applies once iteration counts stop being
   fixed, i.e. Phase 3+.
10. **Cost per step is honestly higher.** With adjacency carried through the state
    (§3.0) an implicit step is ~2–3× an explicit one at 2 Picard iterations, not the
    ~50× it would be if every iteration rebuilt neighbours. That is the real price of
    implicitness and it buys A-stability and symplecticity (§3.3).

**Implicit — resolved by §3.4, recorded so it is not re-litigated**

11. **Neither warp's lack of engine-level forward-mode AD, nor its bare existence in
    `warpSPHCore` for six operators, blocks or unblocks the solver on its own.** FD
    directional derivatives drive Newton to `dt·ω = 1000` with no AD at all
    [verified], and a fixed-count Picard needs no Jacobian action whatsoever — this
    still holds regardless of which operators `warpSPHCore` has since grown JVPs for.
    Exact JVPs (`torch.func.jvp` for torch states; `warpSPHCore`'s dual-tensor
    composition for warp states, scoped to six operators + Covariance — §3.4's
    2026-08-24 correction) are a fast path over FD, not a gate, and not a substitute
    for FD's genericity. The thing that *does* need care at scale is the FD matvec
    staying a directional derivative rather than a dense Jacobian — see §3.4's caution
    — and, for the JVP fast path specifically, failing loudly rather than silently
    when `f` includes an operator outside the wrapped set.
12. **Warp graph capture is fine with a fixed iteration count.** A data-dependent
    iteration count breaks capture (§2.1); the recommended default does not have one.
    Decide alongside §2.2, but the constraint points the same way as ML determinism
    does, which is a rare piece of luck.

### 3.8 Phased plan

Each phase is independently shippable and independently useful.

| Phase | Content | Effort | Gate |
|---|---|---|---|
| **0** | S1–S5 + S2g groundwork (§3.5) — **done 2026-08-24** | 5–6 d | none — S1 also unblocks §2.3's adaptive `dt` |
| **2** | DIRK driver + fixed-count `FixedPointSolver` + all seven DIRK tableaus — **done** (four 2026-08-24; TR-BDF2 + both ESDIRKs 2026-09-09, §3.6) | 4–5 d | Phase 0 |
| **1** | Explicit multistep: AB2–5 + ABM PECE, DP5 starter, `dt`/`uid` guards — **done 2026-08-24** | 2–3 d | Phase 0 |
| **3** | JFNK with FD matvecs + user `solve_linear` hook + particle masking | 1–1.5 wk | Phase 2, **and** a downstream that is actually stiff |
| **4** | BDF1–6 + true Adams-Moulton 2–4 (IMPLICIT_ROADMAP Phase 4) — **BDF4–5 and AM2–4 done 2026-09-09** (BDF1–3 already landed with the JFNK driver); BDF6 stays gated | 3–4 d | Phase 3 |
| **5** | IMEX / ARK, split right-hand side — **done 2026-09-09** (`ark.py`, ARK3(2)4L[2]SA + ARK4(3)6L[2]SA from SUNDIALS ARKODE v7.9.0, §3.6) | 1 wk | Phase 3 + a downstream with a split RHS |
| **6** | Fully implicit: Gauss–Legendre, Radau IIA | 1–1.5 wk | demand-driven; symplectic order 4 is the draw |

Phases 1 and 2 are listed out of numeric order deliberately: Phase 1's old gate —
"measure the resort cadence first" — is void, because there is no resorting (§3.0),
but Phase 2 has the stronger standalone case, so it should land first.

**Recommendation: Phase 0 → 2 → 1, ~2 weeks total, then stop and reassess.**
**All three are done as of 2026-08-24 (§3.5, §3.6). Stop and reassess, as planned:
Phases 3-6 stay gated on a downstream that needs them (see below), none appeared this
session, and nothing here was building a solver contract with no user in the meantime.**

- **Phase 2 is the headline, with one caveat confirmed after implementing it.**
  Implicit midpoint is symplectic, A-stable, and holds order 2 for velocity-dependent
  forces — which nothing else registered does (§3.3) — *when its stage equation is
  solved to convergence*. At the fixed 2-Picard-iteration default this section
  recommends (no norm, no Jacobian, no branching, a fixed-depth graph that is
  differentiable in both backends by construction and CUDA-graph-capturable), that
  headline property does not actually hold: measured energy drift grows secularly,
  the signature of a dissipative scheme, not a symplectic one (§3.6 has the numbers).
  The order-2 accuracy claim is unaffected and was verified to hold exactly at the
  2-iteration default; only the *qualitative long-run energy behaviour* needs more
  iterations than the "ship 2" recommendation to recover. Four of six planned
  tableaus landed (backward Euler, implicit midpoint, trapezoidal, SDIRK2); TR-BDF2
  and both ESDIRKs landed 2026-09-09 once SUNDIALS ARKODE's published tableaus made
  the embedded-pair transcription risk checkable (§3.6).
- **Phase 1 landed cleanly, with no comparable caveat.** AB4 at one force evaluation
  per step against RK4's four, with exactly-correct fixed coefficients and a history
  that never expires — verified for every order (AB2-5, ABM2-4) against
  `oscillator`/`forced`/`damped`, and the full suite went from 1103 to 1380 passing
  tests with only one pre-existing test needing its exclusion criteria widened (not a
  bug in the new code). Pain point 2's BPTT-depth tradeoff (k-step history deepens
  backprop-through-time by k) is still unmeasured and still a question about a
  training setup this library doesn't have visibility into, not something resolved by
  landing the schemes — weigh it before choosing a multistep scheme in a training loop
  specifically, same as before.
- **Phases 3–6 stay gated on a downstream that is actually stiff.** Nothing here needs
  them yet, and building a solver contract with no user is the same mistake §2.2 warns
  about for gradients. §3.4 records the design so the decision does not have to be
  re-derived when a user appears.

**Testing.** Every phase extends the existing net rather than replacing it: the four
problems in `testing.py` already separate the failure modes (`forced` catches stage
times, `damped` catches the separable-Hamiltonian assumption, `kepler` catches
linear-only errors). Add a stiff problem for Phase 2+ (`stiff_oscillator`, `k=1e4`)
and assert that A-stable schemes stay bounded at `dt·ω = 10` while explicit ones do
not — that test is what stops §3.2 from being quietly forgotten. Phase 1 additionally
needs a test that a `dt` change or a `uid` change forces a restart rather than
consuming stale history; that is the only place a silent wrong answer can enter.

### 3.9 The probes behind these numbers

Five scripts, written against the installed v0.5.0 and using only public helpers.
They are not committed — they are evidence for the tables above — but they are the
starting point for Phases 1 and 2 and worth promoting to
`scripts/implicit_multistep_probe.py` if this work is picked up.

| probe | establishes |
|---|---|
| `dirk_probe.py` | A DIRK driver over existing primitives reaches order 1/2/2/2 for backward Euler, implicit midpoint, trapezoidal, SDIRK2 on `oscillator` / `forced` / `damped` (§3.1) |
| `multistep_probe.py` | The list-of-updates path already expresses Adams–Bashforth; AB2–4 and ABM2–4 run; implicit midpoint's energy drift is bounded at 1e-15 over T=160 (§3.1, §3.3) |
| `starter_probe.py` | Self-starting AB caps at order 2; a Dormand–Prince starter recovers 2.97 / 3.99 / 4.03 (§3.7 pain point 1) |
| `stiff_probe.py` | The Picard stage solve diverges at `dt·ω ≳ 10`, i.e. exactly where implicit methods are needed (§3.2) |
| `newton_probe.py` | 2 fixed Picard iterations suffice for order 2 (§3.1); Newton on a **dense, per-column finite-difference** Jacobian — no AD, forward or reverse — stays bounded and L-stably damped to `dt·ω = 1000` (§3.2, §3.4). The dense form is a toy-scale correctness check only — see §3.4's caution before building anything from it. |

### 3.10 Phase 8: the nonlinear/stiff benchmark suite — done 2026-09-09

The roadmap's Phase 8 (stability and nonlinear benchmark suite) is complete.
What landed:

- **Five new problem factories in `testing.py`** (the `PROBLEMS` registry now has
  nine). `stiff_relaxation_problem(rate, forcing, sign)` is Prothero–Robinson in
  both signs: the stable variant (eigenvalue `−rate`, exact solution the moving
  target `s(t)`) and the unstable-PR variant (`sign=−1`, eigenvalue `+rate`, exact
  still `s(t)`). `stiff_damped_oscillator_problem(omega, c)` separates high
  frequency (`ω=50, c=1`) from true dissipative stiffness (`ω=1, c=50`) and has
  the closed-form exact solution in all three damping regimes. `van_der_pol_problem(mu)`,
  `robertson_problem()`, and `diffusion_problem(n)` round it out; the last starts on
  Laplacian eigenvectors 1 and 5 so the *excited* spectrum — not the full one — is
  what the stability boundary has to contain.
- **Damped-oscillator amplification matrices** in `stability.py`
  (`damped_oscillator_amplification_matrix`, `damped_oscillator_spectral_radius`)
  for leapfrog, velocity Verlet, symplectic Euler, and Newmark (β = 1/4 and 1/6),
  cross-checked against the registered schemes' actual one-step maps
  (basis-state numerical matrices agree to 1e-8/1e-10) and against the undamped
  Phase 6 results at `ζ = 0` (1e-12).
- **`tests/test_benchmarks.py`** — 52 tests, all small and deterministic (25 s):
  the registry, PR in both signs, the three damping regimes, van der Pol at
  μ = 2 and μ = 10, Robertson invariants and scheme agreement, diffusion, three
  explicit-wall tests, and the amplification-matrix checks.
- **`scripts/stiff_benchmark_suite.py` → `images/stiff_benchmark_suite.png`** —
  error vs `dt` on the four stiff benchmarks, per-step JFNK cost panels, and a
  van der Pol energy-vs-time panel (bounded bands for the settling orbits, the
  diverging orbits leave the capped axis within `t < 2`); all parameters and
  solver settings are stated in the figure caption.
- **`scripts/oscillator_stability_gallery.py`** gained the damping-ratio figure
  (`images/oscillator_stability_damped_newmark_verlet.png`): `log10(ρ)` over
  `(h·ω, ζ)` for the five methods with the `ρ = 1` boundary contoured.

Findings worth keeping:

- **Robertson needs a bootstrap, and BE has a hard ceiling on it.** The initial
  quasi-steady layer (width ~1e-3) is not crossable from `y(0) = (1, 0, 0)` at
  `dt ≥ 0.02`; ten `dt = 0.001` steps put the solution on the QSS manifold and the
  main `dt` takes over. Backward Euler at `dt ≥ 2` then converges *every* JFNK
  solve to an unstable discrete fixed point of the stiff quadratic and the map
  diverges ~2×/step — BE inaccuracy at `dt` above the fast scale (fast eigenvalue
  ~2.2e3), not a solver failure. `dt = 0.2` is strictly positive for all components
  over T = 100.
- **ESDIRK6's stability function has a pole at `z = 1`.** The stiffly-accurate
  `γ = 1` stage puts a pole in `R(z)`, so on the *unstable* PR (`z = +5`) ESDIRK6
  diverges (|x| ~ 9e29 in five steps) while A-stability says nothing about the
  positive real axis. DP5 amplifies the unstable mode by `R(+5) = 117.5` per step.
  Only BE/BDF-type methods track the unstable PR, where BE is the natural choice
  (`R(z) → 1/2` as `z → +∞`).
- **Damping reshapes the Verlet-family stability pictures in opposite directions.**
  Velocity Verlet at `h·ω = 2.5` (undamped eigenvalues −0.25 and −4.0, `ρ = 4`)
  has a stable window `0.5055 < ζ < 0.8` — at `ζ = 0.6` the pair is complex with
  `|λ| = √det = 0.5`. Leapfrog's damping term sits at the *old* velocity, so it is
  itself an explicit update unstable for `c = 2ζ·h·ω > 2`: no damping ratio
  rescues it. Newmark (average acceleration) is neutrally stable undamped for
  every `h·ω` and becomes asymptotically stable for any `ζ > 0`.
- **Van der Pol at μ = 10 has a discrete-attractor wall.** ESDIRK6 tracks the
  amplitude-~2 cycle at `dt ≤ 0.2` (errors 0.53 / 1.22 / 3.4 vs a `dt = 0.01`
  reference at `dt = 0.05 / 0.1 / 0.2`) and diverges at `dt = 0.4` — the energy
  overflows by `t ≈ 2` and the state reaches NaN well before T = 50; DP5 at
  `dt = 0.5` leaves the O(1) band within a few steps. At μ = 2, by contrast,
  DP5 tracks the cycle at `dt = 0.05` — the explicit restriction is not active
  until μ grows.
- **Per-step JFNK cost (PR, rate = 100, `dt = 0.01`, 100 steps, FD matvecs,
  GMRES tol 1e-8):** GMRES iterations per step are exactly the stage-solve counts
  (1.00 / 0.99 / 2.00 / 5.00 for BE / BDF2 / TR-BDF2 / ESDIRK6) with zero
  line-search backtracks; RHS evaluations per step are 3.40 / 2.96 / 4.50 / 10.72.
  ESDIRK6's per-step cost is ~3× BE's, so its accuracy advantage on these
  benchmarks is bought at a real price. (This metric counts only the implicit
  stages' `rhs_evaluations`; the DIRK explicit first stage is recorded as `None` and
  is excluded, so Phase 10's first-stage reuse — which saves that explicit evaluation
  — does not move these numbers. See §3.6's Phase 10 note.)

### 3.11 Phase 13: TVD / SSP classification of the registered schemes — done 2026-09-10

The roadmap's Phase 13 (nonlinear stability for the explicit side) is complete.
`stability.py` answers the parabolic question — "does this scheme stay bounded on
`y' = λy`?"; this phase measures the hyperbolic one — "does one step increase
total variation?" — for **every** registered scheme, not just the TVD/SSP-named
ones, whose defining property was previously asserted by name alone.

What landed:

- **Two hyperbolic problem factories in `testing.py`** (the `PROBLEMS` registry
  now has eleven). `advection_problem(n, c, L, ic)` is 1D periodic first-order
  upwind advection `u_t + c u_x = 0`, `(Au)_i = c(u_{i-1} − u_i)/h`: the model
  problem for the classifier, because every Fourier mode is an exact eigenmode
  with eigenvalue `(c/h)(e^{−2πim/n} − 1)` and every RK step is a circulant map
  on the grid. `ic='mode'` starts on two eigenmodes (exact semi-discrete
  solution known); `ic='step'` is a periodic square wave (TV = 4) for the
  total-variation checks. `burgers_problem(n)` is semi-discrete inviscid Burgers
  with local Lax-Friedrichs fluxes on the smooth entropy wave `−sin(2πx)` —
  the *nonlinear* hyperbolic problem.
- **`tvd_analysis.py`** — the general TVD classifier, with two measurements:
  1. `convex_combination_cfl` (the **SSP coefficient** `r`): the stage-value
     maps and the final map of the scheme's tableau, evaluated on the upwind
     model over all `n` modes at once (`stage_and_final_maps`), must be convex
     combinations of periodic shifts (elementwise non-negative circulant first
     row, row sum 1) for every `μ' ≤ μ`. A convex-combination step is TVD by
     construction, so `r` is a *rigorous* TVD CFL for every scheme that
     exposes its tableau (explicit RK, DIRK; ARK in its pure-implicit limit;
     the hand-written TVD RK2/3 through their algebraically identical Butcher
     equivalents, pinned to the actual drivers by `tests/test_tvd.py`).
  2. `measure_tvd_cfl` (the **measured per-step TVD CFL**): run the registered
     driver on the step initial condition and require the per-step total-
     variation increase to stay within `1e-8 · TV0` over a post-burn-in window
     (burn-in 10, window 40 — the multistep cold-start TV jump is shared by
     all self-starting multistep schemes and is not a property of the step
     map). This one sees the actual driver, so it also classifies the
     schemes that expose no tableau (BDF/Adams family, semi-implicit Euler).
- **`scripts/tvd_classifier.py` → the verdict table below** — the maintained
  fact; re-run the script to refresh it (the numbers are measured, not
  asserted by name).
- **`tests/test_tvd.py`** — 26 tests: the published `r = 1` of the TVD-named
  schemes, the landmark coefficients, the TVD-driver-vs-tableau cross-check,
  per-step TV / positivity / Burgers assertions at CFL 1, the stage-level
  separation (below), and the full-registry verdict table.

The measured landscape (n = 64, step IC, CFL grid 0.25 … 5, convex-combination
scan to CFL 4):

| scheme | SSP r | measured TVD CFL |
| --- | --- | --- |
| Forward Euler, Midpoint, Heun 2, Ralston 2, Heun 3, Ralston 3, Wray 3, SSP RK3, Bogacki-Shampine, EPEC, EPEC Modified, **TVD RK2, TVD RK3** | 1 | 1 |
| RK3 | 0.5 | 1 |
| RK4 | 2/3 | 1.25 |
| RK4 (alternative) | 1/3 | 1.25 |
| Cash-Karp 5(4) | 5/12 | 1.5 |
| **Nystrom 5th order, Dormand-Prince 5(4)** | **0** | 1.5 |
| Backward Euler (implicit) | unconditional (Fejer kernel) | 5 (tested limit) |
| Implicit Midpoint, Trapezoidal | 2 | 5 (tested limit) |
| SDIRK2, TR-BDF2 | 1 + √2 | 5 (tested limit) |
| ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA, ARK3(2)4L[2]SA, ARK4(3)6L[2]SA | 0 | 5 (tested limit) |
| BDF1, IMEX Euler, Semi-Implicit Euler | (no tableau) | 5 (tested limit) |
| BDF2-5, Adams-Bashforth 2-5, ABM 2-4 (PECE), Adams-Moulton 2-4 (implicit) | (no tableau) | 1.5 |
| Leap Frog, Symplectic Euler, Velocity Verlet, PEFRL, VEFRL, Newmark | n/a — second-order-system schemes; the TVD question does not apply | |

Findings worth keeping:

- **TVD is strictly broader than SSP, and the registry shows it.** RK4's
  convex-combination limit is `r = 2/3` (stage 4's circulant entry
  `(μ²/4)(2 − 3μ)`), yet its *per-step* TVD CFL measures 1.25: the final map
  (the shared fourth-order stability function) is TVD past the point where the
  internal stages leave the convex hull. Same pattern, stronger: Nystrom 5th
  order and Dormand-Prince 5(4) have `r = 0` — not a convex-combination step
  at *any* CFL — yet are per-step TVD up to CFL 1.5.
- **Per-step TV cannot separate same-order schemes; the stage level can.**
  Classical RK3 at CFL 1 produces *zero* per-step TV increase on the step
  advection and on Burgers, identical to TVD RK3 — all third-order RK methods
  share the same final map, which is a convex combination up to CFL 1. The
  separation is at the stage level: RK3's stage-3 map is `1 + ζ + ζ²`, whose
  circulant row at CFL 1 is `[1, −1, 1]` (entry `μ(1 − 2μ) = −1`), while
  TVD RK3's stage rows are `[0, 1]` and `[3/4, 0, 1/4]`. The test runs the
  registered drivers from a delta initial condition, so each stage state *is*
  its circulant row (`tests/test_tvd.py::test_classical_rk3_stage_leaves_the_convex_hull`).
- **Nystrom 5th order, Dormand-Prince, ESDIRK3/4 (and the ARK limits) have
  `r = 0` exactly, not "tiny".** For the explicit ones this was checked by
  exact rational arithmetic: Nystrom's stage-5 entry at circulant offset 3 is
  `7.995e-18·μ³ − 0.1185·μ⁴` (negative for every `μ` above `~1e-16` — the
  method's notoriously ill-scaled coefficients), and Dormand-Prince's stage-5
  and stage-6 entries at offset 4 have lowest terms `−0.0465·μ⁴` /
  `−0.0509·μ⁴` (negative for all `μ > 0`). A naive tol-based scan reports
  where the (genuinely negative) entry's magnitude happens to cross the
  tolerance (0.0054 for Nystrom), which looks like a small-but-real
  coefficient; `convex_combination_cfl` now probes below the candidate
  boundary (a genuine zero-crossing grows in magnitude as the probe moves
  away from the root, an everywhere-negative entry shrinks like `(1/2)^d` per
  halving) and returns `0.0` in the latter case. Cash-Karp's `5/12` is a
  genuine crossing and survives the probe.
- **Implicit Midpoint and Trapezoidal are *not* unconditionally TVD — both
  have `r = 2`.** Their stage map is the unconditionally positive Poisson
  kernel, but the propagated map is the shared Cayley factor
  `(1 + ζ/2)/(1 − ζ/2)`, a convex combination only for `μ ≤ 2`. The earlier
  "implicit midpoint is unconditionally TVD" reading of its A-stability was
  wrong: A-stability bounds `|R(z)|`, TVD requires the convex-combination
  structure, and the final map is what propagates.
- **The L-stable implicits are TVD to the top of the sweep** (CFL 5): BE
  (unconditionally, by the Fejer kernel), SDIRK2 / TR-BDF2 (whose `r = 1+√2`
  bounds the stages, not the final map), and the ESDIRK/ARK pairs — whose
  stages leave the convex hull immediately (`r = 0`) but whose stiff-accurate
  final maps damp so strongly that no per-step TV increase is measurable.
- **The multistep family is TVD to a measured CFL of 1.5** (BDF2-5, AB2-5,
  ABM 2-4 PECE, implicit AM2-4), diverging by CFL 2 (per-step TV increase
  `~4e25 · TV0` — the solution itself leaves the O(1) band). BDF1 = BE is
  unconditional. The measured cold-start TV jump (`+1.333e-2 · TV0`, shared by
  every self-starting multistep scheme and gone after five steps) is why the
  sweep uses a burn-in.

### 3.12 Phase 14: the structured RHS interface — done 2026-09-10

Phase 7's Rosenbrock-W and exponential integrators need a third kind of RHS
structure — a linear-operator action `L·v` and the nonlinear remainder
`N = f − L·y` — and there was nowhere to put it. Rather than add a second
NamedTuple beside `IMEXRHS`, the RHS input surface collapsed to exactly two
shapes: **a plain callable** (fully implicit, or just evaluated) or **a typed
`RHS`** whose declared capabilities cover every split any registered or
planned scheme asks for. `IMEXRHS` folds in as one shape of the latter.

What landed:

- **`rhs.py`** — the interface. `RHS` is a *concrete* class, not a bare
  structural `Protocol`, so a plain function never `isinstance`-matches it.
  `__call__(state, dt, *args, **kwargs)` is the combined right-hand side
  `f = f_E + f_I = L·y + N` — always defined, the ground truth — and
  `provides` is a `frozenset` over `{"explicit", "implicit", "linear",
  "nonlinear"}`. All four optional accessors share the same `(state, dt,
  *args, **kwargs) -> update[, aux]` contract as the combined callable, so a
  driver routes each one through `updateStep` exactly as `ark.py` already
  routes the two `IMEXRHS` halves. `linear(state, …)` reads the integrated
  field(s) it acts on from `state` and returns `L·(field)`; to apply `L` to
  an *arbitrary* vector (a Krylov iterate, a preconditioner probe) the caller
  hands it a state built with `unflatten_integrated`.
  - `IMEXRHS(explicit=, implicit=)` is now a thin constructor returning an
    `RHS` with `provides={"explicit","implicit"}` and a summing `__call__`
    (the old NamedTuple in `specs.py` is gone).
  - `SemilinearRHS(linear=L, nonlinear=N)`, and the `SemilinearRHS(linear=L,
    combined=f)` form, which synthesizes `N = f − L·y` inside `resolve`.
  - `RHS(**parts)` general constructor accepting any subset, including the
    overlap case — the same stiff operator behind both `implicit` and
    `linear`, where the additive halves define `f` and the semilinear parts
    are a view of it.
- **Driver resolution (`resolve`)** — synthesizes what can be synthesized
  (`implicit` ← the combined `f`; `explicit` ← absent; `nonlinear` ← `f − L·y`
  when a `linear` is declared, else the combined `f`), and requires what
  cannot (`linear`). A plain callable resolves to `explicit=None, implicit=f,
  linear=None, nonlinear=f` — bit-identical to the old `isinstance`
  else-branch, and the bit-identity is pinned by the golden master below. When
  a scheme needs a split the RHS does not carry, it fails **before the solve**
  with a `TypeError` naming the missing capability and what the given RHS
  actually provides. `ark.py` and `imex.py` moved from `isinstance(f,
  IMEXRHS)` to this check; nothing downstream constructs an `IMEXRHS`, so the
  reshape had no external contract to preserve.
- **`check_contracts`** — the three split contracts, checked on demand (tests
  or a harness; *not* on the production step path): additivity is disjoint
  (`explicit + implicit == f`); `linear` is linear and homogeneous (`L·0 = 0`,
  spot-checked `L(a+b) ≈ L(a)+L(b)` on random vectors built through
  `unflatten_integrated`); and any two declared views agree (`linear·y +
  nonlinear == f`).
- **`viscousBurgers` problem factory** (the `PROBLEMS` registry now has twelve)
  — semi-discrete viscous Burgers `u_t + (u²/2)_x = ν u_xx` as the canonical
  semilinear benchmark: `ν u_xx` is a constant-coefficient linear operator
  `L` with a cheap matrix-free action, `(u²/2)_x` is the nonlinear remainder
  `N`. Shipped as a `SemilinearRHS`, it doubles as the Phase 7 accuracy/cost
  problem — exponential integrators are conventionally demonstrated on exactly
  this equation. Distinct from the existing inviscid `burgers_problem`
  (Lax-Friedrichs, the §3.11 TVD/SSP classifier's nonlinear problem).
- **`tests/test_rhs.py`** — 19 tests pinning the four validation gates:
  (1) *every registered scheme* reproduces the pre-refactor golden master on a
  bare callable, bit for bit — `tests/data/bitident_golden.json`, captured
  *before* the refactor, dt = 0.1 for 3 steps on the one-DOF oscillator (exact
  in float, so the comparison is strict equality); (2) `IMEXRHS` (now a
  constructor) and a hand-built `RHS` with the same halves give identical ARK /
  IMEX Euler trajectories; (3) a `linear`-consuming scheme raises the specific
  capability error on a plain `f` and on an `IMEXRHS`, not a mid-solve shape
  mismatch; (4) the `nonlinear` synthesized from `f − L·y` matches an
  independently supplied `N` on viscous Burgers to round-off. Plus the
  `check_contracts` pass/fail cases, the resolution rules, and an end-to-end
  run integrating viscous Burgers through plain RK4 with the mass conserved to
  round-off.

Suite: 2346 → 2365 passing.

### 3.13 Phase 12: RKC / RKL — stabilized explicit super-timestepping — done 2026-09-11

The roadmap's "strongest omission": an explicit method that attacks parabolic
stiffness with no nonlinear solver at all. RKC (relaxed Chebyshev) and RKL (relaxed
Lobatto/Legendre) spend `s` matrix-free right-hand-side evaluations per step to buy a
real-axis stability interval that grows like `O(s²)` — so `dt` can be set by the
physics (the convective scale), not by the stiffest diffusive eigenvalue. For SPH
viscosity that is frequently the right answer over implicit, and it attacks exactly the
regime the Phase 2 preconditioning work and the `diffusion_problem(n)` benchmark exist
for.

Construction. Each method's one-step stability polynomial is a scaled, shifted
orthogonal polynomial in `z = dt·λ`: `R_s(z) = T_s(1 + z/s²)` for RKC1, and
`a_s + b_s·Φ_s(1 + w₁z)` for RKC2 / RKL2 (`Φ_s` the Chebyshev `T_s` or Legendre
`P_s`; the constant offset `a_s` is what lifts the order from 1 to 2). The real-axis
stability intervals are

    RKC1  K = 2s²            RKC2  K = 2(s²−1)/3           RKL2  K = (s²+s−2)/2

and all three are realised by a three-term (plus a cached `Y_0`) recurrence, so a step
costs exactly `s` right-hand-side evaluations and a few state buffers — no linear
solve, no matrix. The stage weights are chosen so that every *intermediate* stage
polynomial is itself bounded by 1 on `[−K, 0]` (no stage overshoots). RKL2 is the
scheme of Meyer, Balsara & Aslam, *J. Comput. Phys.* 257 (2014) 594–626 (Eqs. 15–19);
RKC2 is the identical construction with `T_j` in place of `P_j`; RKC1 is the order-1
Chebyshev limit (Ruuth, *J. Comput. Phys.* 169 (2001) 162–175). The full recurrence
and coefficient rules live in the `rkc.py` module docstring.

What landed:

- **`rkc.py`** — one shared core `integrateRelaxedChebyshev(state, dt, f, family, …)`
  and three thin drivers, `RKC1` / `RKC2` / `RKL2` (enum 52/53/54; registry order
  1/2/2, explicit, `dissipation=True`, no reuse). `stage_count(dt·|λ_max|, family)`
  returns the smallest admissible stage count, clamped to the family's minimum (1 for
  RKC1, 3 for RKC2/RKL2); the driver takes `s=` directly or `lambda_max=` and picks
  `s` — neither, and it raises. `priorStep` is refused with the standard warning.
- **Generic-suite skips, with reasons** (`conftest.py` `NEEDS_STAGE_COUNT`): the stage
  count is a *per-step* parameter the shared one-step tests do not provide, and the
  stability analysis is on the real axis, which the oscillator's imaginary eigenvalues
  do not exercise (RKC1 is marginally unstable on that problem for any `s`). Three
  tests that iterate the registry directly were adjusted for it: `test_rhs.py` skips
  schemes absent from the pre-Phase-14 bit-identical golden; `test_hamiltonian.py`
  skips the family in the oscillator area-property test; and `tvd_analysis.py`
  classifies it as "not applicable (parabolic super-timestepping)" — the hyperbolic
  TVD/SSP question does not apply to a method built for parabolic terms.
- **`tests/test_rkc.py`** (35 tests) — where the family is actually exercised:
  registered order on the diffusion, the stability boundary on a multi-mode random IC,
  the stage-count rule (smallest admissible, tight at every `K(s)` boundary up to
  `s = 43`, `s²` growth), `s=` ↔ `lambda_max=` equivalence, minimum clamping, the
  missing-`s` error, the `priorStep` warning, exactly `s` RHS evaluations per step,
  stage times inside `[t, t+dt]` opening at `t`, no caller-state mutation, and a
  semilinear-RHS (viscous Burgers) run that conserves mass to round-off.

Verification. The recurrences were validated *before* the driver was written (scalar
stability polynomial to round-off, intermediate-stage boundedness, order on the
semilinear diffusion); the tests pin the same through the real machinery. Measured on
the semi-discrete diffusion (n = 32, `s = 12`, T = 0.02):

- **Order**: 0.930 (RKC1), 2.137 (RKC2), 2.126 (RKL2). The order is independent of
  the stage count; `s` only sets the stability interval.
- **Stability boundary** (random multi-mode IC, n = 16, `s = 8`): the run is bounded
  just inside and blows up just outside the predicted interval — the measured blow-up
  sits at `dt·|λ_max| / K(s) = 1.007–1.011`, i.e. the `K(s)` prediction holds to about
  one percent.

The benchmark (the roadmap gate). `scripts/rkc_benchmark.py` →
`images/rkc_benchmark.png`: total RHS evaluations (finite-difference JFNK matvecs
included) to reach max error ≤ 1e-2 on `diffusion_problem`, T = 0.1:

| n (|λ_max|) | RKC1 | RKC2 | RKL2 | TR-BDF2 | BE | BDF2 |
|-------------|------|------|------|---------|-----|------|
| 16 (1146)   | 32   | 40   | 48   | 89      | 268 | 224  |
| 32 (4346)   | 64   | 80   | 88   | 147     | 399 | 896  |
| 64 (16890)  | 128  | 144  | 168  | 271     | 710 | unstable |

Two findings:

1. **The family is the cheapest way to a fixed error at every n.** RKC/RKL reach
   1e-2 with fewer total RHS evaluations than any implicit baseline — the implicit
   cost is dominated by the nonlinear solve (finite-difference Jacobian matvecs in
   the default JFNK), which RKC/RKL do not have. Order-1 RKC1 is cheapest overall
   (fewest stages); RKC2/RKL2 add an order of accuracy at a modest stage-count
   premium (12–50% more total evaluations depending on `n`).
2. **BDF2's practical stability on this benchmark is solver-limited, not
   method-limited.** BDF2 is A-stable, yet its default JFNK (finite-difference
   matvec, no preconditioner) fails from the first step at coarse `dt` (n = 16,
   dt = 0.05: max|u| 1.44 → 1336, with z ≈ 11.8 deep inside the A-stable region), at
   n = 32 for dt ≥ 0.001563, and at n = 64 at every `dt` tested. The benchmark marks
   such runs unstable and reports BDF2's n = 64 as "no stable point reaches the
   target." That solver ceiling — not the stability region — is the wall the Phase 2
   preconditioner work is meant to lower on the implicit side, and it is the regime
   RKC/RKL sidestep entirely.

Caveats. The stage times are the autonomous-domain approximation (stage `j` at
`t + dt·j/s`); for a non-autonomous right-hand side this is an approximation. RKL2
and RKC2 prefer an *odd* stage count — for even `s` the smallest-wavelength mode is
only weakly damped (Meyer et al. 2014); the driver clamps to the minimum but does not
force parity, so pass `s=` (or a slightly larger `lambda_max=`) when you want odd.

Suite: 2365 → 2400 passing (the 35 new tests are `tests/test_rkc.py`).

### 3.14 Phase 7: ROS3P (Rosenbrock-W) — done 2026-09-11

The roadmap's Phase 7 first family: a **Rosenbrock-W** (linearly-implicit, no
outer Newton loop) method for semilinear stiffness. Each stage is one `gmres`
solve against a *frozen* operator `W ≈ ∂f/∂u` held fixed across the step, so a
step is three linear solves with no nonlinear iteration — the contrast with the
JFNK DIRK/BDF/ARK drivers, which re-linearize inside a Newton loop at every
stage.

Construction. ROS3P is the 3-stage, order-3, A-stable (not L-stable)
Rosenbrock-W method of Lang & Verwer, *BIT* 41(4) (2001) 731–738, in the general
Rosenbrock-W one-step form (2.2):

    (I − τγ W) K_i = F(tn + a_i τ, un + τ Σ_{j<i} a_ij K_j)
                    + τ W Σ_{j<i} g_ij K_j + τ γ_row_i ∂_t F(tn, un)
    un+1 = un + τ Σ b_i K_i

with `γ = 1/2 + √3/6 = 0.788675…`, nodes `a = (0, 1, 1)`, `a21 = a31 = 1`,
`a32 = 0`, `g21 = −1`, `g31 = −γ`, `g32 = 1/2 − 2γ`, `γ_row = (γ, γ−1,
1/2−2γ)`, `b = (2/3, 0, 1/3)`, embedded `b_hat = (1/3, 1/3, 1/3)`. The order-3
condition `γ² − γ + 1/6 = 0` holds in `Q(√3)` (verified symbolically); the
stability function gives `R(∞) = 1 − √3 ≈ −0.732` (A-stable, not L-stable). The
three stage *states* collapse to two distinct RHS points — `(tn, un)` and
`(tn+τ, un+τ K1)` (stage 3 reuses stage 2's state/time) — so a step costs two
`f` evaluations (plus one more for `∂_t F` when `f_t='fd'`). The common
per-stage operator is `M = I − τγ W` (identical for all three stages).

What landed:

- **`rosenbrock.py`** — the dedicated driver `integrateROS3P` (its own
  coefficient structure, not a DIRK tableau; enum 55, order 3, stability `A`,
  `stiffly_accurate=False` so `priorStep` is refused). The `w=` selector picks
  the frozen operator: `'jvp'` (default, exact Jacobian via
  `jfnk.jvp_matvec`), `'fd'` (finite-difference Jacobian via
  `jfnk.fd_matvec`), or `'linear'` (the Phase 14 `linear` accessor). `f_t='fd'`
  (default) finite-differences `∂_t F` one-sided, reusing the stage-1
  evaluation, at a `dt`-scaled step `machine_eps^(1/3)·max(1, |dt|)` (the
  stage-time-shift invariant: the probe time `tn + eps` translates by exactly
  `dt` step to step, which is what the generic driver tests require);
  `f_t='none'` treats the RHS as autonomous. Gradient: the Phase 9 frozen-map
  re-attachment generalized to three solves — each stage solve `K_i = M^{-1} b_i`
  re-attached as a constant linear map whose adjoint is the transposed operator
  (a `gmres` transpose solve), each frozen-operator term `W x` re-attached with
  adjoint `W^T` (a VJP graph). The frozen `W` is held constant in the adjoint (its
  `dW/dy` is a Hessian, dropped) — exact on a linear RHS (`J_N = 0`), a few
  percent on nonlinear ones.

Three non-obvious fixes, each pinned by a test:

1. **The stage-1 `∂_t F` term is `τ·γ_row_1·ft`, not `τ·γ·γ_row_1·ft`.** The
   operator's `τ·γ` (`tau_gamma`) is a different quantity from the stage's
   `γ_row_1`; conflating them injected an extra `γ` and dropped the order.
   Pinned by the non-autonomous (forced) order test.
2. **The stage buffer must come from `initializeNewState`, not
   `unflatten_integrated`.** The latter *clones* non-integrated fields (a shared
   stage counter, a neighbour list), so a `preprocess` that mutates one in place
   would miss the caller's copy and `copied` fields would carry the wrong
   stage's value. `f_flat` now builds the buffer from
   `state.initializeNewState(*args, **kwargs)` and sets the integrated fields in
   place — the same contract the DIRK/BDF drivers use (`bdf._copy_integrated`).
   Pinned by `tests/test_copied_fields.py` (ROS3P now passes all four).
3. **The `W` action runs the full step machinery (`updateStep`), not the bare
   RHS, and the `WK1`/`Wq` RHS-side applications are costed honestly.** The
   Jacobian is of the *full* right-hand side (preprocess + f + postprocess),
   which needs the `copied` fields `preprocess` fills; the frozen point is the
   step start `(tn, un)`, so each Krylov probe re-runs `preprocess`. And `WK1` /
   `Wq` are full `f` evaluations for `w='jvp'/'fd'` (they run
   `combined_counted`) but cheap `linear_fn` applications for `w='linear'` —
   the accounting `w_op = 1 if w in ('jvp','fd') else 0` counts them only in the
   former case, so ROS3P's `rhs_evaluations` is honest against the JFNK
   baselines (which count every full-`f` evaluation too).

The `w='linear'` order drop. On a *genuinely semilinear* problem `w='linear'`
drops the nonlinear Jacobian `J_N` from the frozen operator, so it is a
**first-order** method there (verified on viscous Burgers: measured order 1,
not 3); it is order 3 only when the RHS is linear in the state (`J_N = 0`). It
is the cheap low-order option, not a stand-in for the full Jacobian — documented
on the scheme and in the demo notebook.

Embedded-pair degeneracy. For a linear autonomous `y' = Ay` the frozen operator
`W = A` makes the two embedded stages agree exactly (`K1 == K2`), so the error
estimate `τ(K1−K2)/3` is identically 0; the embedded order is therefore measured
on a non-autonomous (forced) problem (rates ~3.05/3.03/3.01), not the
oscillator.

Verification (viscous Burgers n = 64, ν = 0.01, T = 0.4, vs RK4 `dt = 2e-3`).
`tests/test_rosenbrock.py` (18 tests): order 3 on viscous Burgers for both
`w='jvp'` and `w='fd'`; `w=`-selector agreement; `w='linear'` order-1 semilinear
/ order-3 linear; `f_t='fd'` retains order on a forced RHS while `f_t='none'`
drops it; A-stable-not-L (a `rate = 1000` decay over `dt = 1` lands at
`|1−√3| ≈ 0.732`); embedded order 2 via the propagated `y_hat`; stage-3 reuses
stage-2's `StageResult`; the two distinct stage times; `priorStep` refused; no
caller-state mutation; and gradient parity (exact on linear to 1e-4/1e-3 for
jvp/fd, FD-limited to < 0.25 on the semilinear problem). TVD verdict
`(None, 5.0, True)` — not TVD-classified (no Butcher tableau exposed) but stable
to the tested CFL.

The benchmark (the Phase 7 gate). `scripts/rosenbrock_benchmark.py` →
`images/rosenbrock_benchmark.png`; the same table is §10 of
`viscous_burgers_demo.ipynb`. Sine IC, `dt = 0.02` (20 steps), T = 0.4,
work-units = `rhs_evaluations + gmres_iterations` (the JFNK invariant; for
`ROS3P (jvp/fd)` the GMRES matvecs are full Jacobian sweeps already inside
`rhs_evaluations`, so its total is just `rhs_evaluations`):

| scheme (order)   | RHS evals | GMRES iters | total | rel L2 err |
|------------------|-----------|-------------|-------|------------|
| ROS3P (W=jvp) (3) | 706 | 586 | 706 | 2.04e-4 |
| ROS3P (W=fd) (3)  | 715 | 595 | 715 | 2.04e-4 |
| ROS3P (W=linear) (1) | 60 | 783 | 843 | 7.76e-3 |
| ARK3(2)4L[2]SA (3) | 240 | 553 | 793 | 2.63e-4 |
| ESDIRK3(2)4L[2]SA (3) | 300 | 688 | 988 | 7.25e-5 |

Convergence (Gaussian IC, `w='jvp'`): measured orders 2.70, 2.71, 2.82, 2.90 on
`dt = 0.08/0.04/0.02/0.01/0.005` → order 3 (gate (a) met). **Gate (b) met on
total work-units:** among the order-3 methods `ROS3P (W=jvp)` is the cheapest
(706 < ARK3 793 < ESDIRK3 988) and also beats ESDIRK3 on GMRES iterations (586 <
688). Two honest caveats, recorded in the notebook: (i) the additive ARK3
split's GMRES iterations are cheap *linear-operator* applications (the IMEX
split hands the stiff diffusion to the `linear` part), whereas ROS3P's are full
Jacobian sweeps — so ARK3 is cheaper per iteration in FLOPs even though its
work-unit count is higher; (ii) `ROS3P (W=linear)` is cheapest in full-`f`
evaluations (60) but order 1 on this semilinear PDE and pays in the most GMRES
iterations (783).

New tests: 18 in `tests/test_rosenbrock.py`, plus the ROS3P rows added to
`tests/test_embedded.py`, `tests/test_copied_fields.py`, and `tests/test_tvd.py`.

---

### 3.15 Phase 7: ETD2RK (exponential integrator) — done 2026-09-12

The roadmap's Phase 7 **exponential-integrator** sub-phase, and the first
exponential method in the registry. An exponential integrator is the natural home
for a semilinear PDE `y' = L·y + N(y)` with a *stiff linear* part `L` and a *mild*
nonlinear remainder `N`: instead of solving a nonlinear system (DIRK/BDF) or a
frozen-operator linear system (Rosenbrock), it carries `L` **exactly** through the
matrix functions `exp(hL)` and the `phi_k` functions, and only *quadratures* the
mild nonlinear part. This is the strongest fit for the Phase 14 `SemilinearRHS`
split of the three Phase 7 families, and it is the method that consumes the `linear`
accessor most directly.

**Method.** The base unsplit **ETD2RK** (2-stage, **order 2**, **L-stable** for the
linear part), in the form of **Sarumi, arXiv:2601.06849** (unsplit ETD2RK, eq. (2.5));
the `phi` functions per **Caliari & Ostermann, *Appl. Numer. Math.* 59 (2009) 568–581,
eq. (2.4)** (`phi_k(z) = (e^z − P_{k−1}(z))/z`, `P_{k−1}` the Taylor polynomial):

```
y_{n+1} = exp(hL)·y_n + h·phi_1(hL)·N(t_n, y_n)
         + h·phi_2(hL)·( N(t_n + h, w) − N(t_n, y_n) )
w       = exp(hL)·y_n + h·phi_1(hL)·N(t_n, y_n)        (the first stage)
```

Two stages (`w`, `y_{n+1}`), **two `N` evaluations per step** (`N_n`, `N_w`), and
**three matrix-function actions per step** (`exp(hL)`, `phi_1(hL)`, `phi_2(hL)`), each
applied to a *different* vector. `L` is integrated **exactly** (the linear part has no
truncation error — see the verification below); only `N` is quadratured, which is what
makes the method order 2. (The newly-added literature PDF is the *dimension-split,
Sylvester-based* 2D extension of ETD2RK — it factors the 2D diffusion operator as a
Sylvester matrix product to cut the Krylov cost. Our benchmark `L = nu·u_xx` is already
a single 1D operator, so the Sylvester split has nothing to buy here; ETD2RK on a 1D
`L` is the in-scope, in-repo piece.)

**Matrix-free `phi_k(hL)v`.** `exponential.py` builds `krylov_phi(matvec, v, k, m=30,
tol=1e-10)`: one **Arnoldi** build of `m` steps on the operator (the same Givens-free
Hessenberg core as `jfnk.gmres`, with the residual early-stop), then `phi_k(H)·e_1` of
the small dense `(m, m)` Hessenberg matrix by a **short Taylor-vector series**
(`term_{j+1} = H·term_j / (j + k + 1)`, `k = 0` gives `exp`). No dense `L`, no matrix
exponential, no eigendecomposition. One build per distinct right-hand-side vector
(three per step: `exp(hL)·y_n`, `phi_1(hL)·(h·N_n)`, `phi_2(hL)·(h·M)`), because the
build is the expensive part and the `phi` of the Hessenberg is cheap. The total
`L`-matvec (Krylov iteration) count is reported as the step's Krylov-iteration count in
the `SolveDiagnostics` (the `gmres_iterations` field). The `phi` series is a plain
Taylor sum (no scaling-and-squaring), so it is accurate for modest `||h·L||`; the
benchmark's `||h·L|| ~= 3.3` is comfortably in range, and an over-stiff step would need
scaling-and-squaring (a future extension, noted in the module docstring).

**Gradient (in-scope, the point of a differentiable exponential integrator).** Because
`L` is **constant** (independent of `y`), the three matrix-function actions are *fixed*
linear maps; each is re-attached to the autograd tape with its **transpose** as the
adjoint — the transpose `phi_k(hL)^T` actions computed by the same Krylov machinery on
`L^T`. There is no unrolled Krylov and, unlike the Rosenbrock frozen-`W` (which drops
`dW/dy`, a Hessian, giving an O(dt) gradient), **no structural approximation** — the
gradient is **exact up to the Krylov tolerance**. Scope note: the transpose reuses the
forward `linear` accessor because the benchmark `L = nu·u_xx` (periodic central
Laplacian) is **self-adjoint**; a non-self-adjoint `L` would need an `L^T` accessor
(a future extension). This is the cleanest gradient story in the Phase 7 set: it is
exact (up to the Krylov/FD tolerance) on *both* the linear and the semilinear problem,
whereas ROS3P's is O(dt) (a few percent) on the semilinear one.

**Registration.** `etd2rk = 56` in the enum; `IntegrationScheme(integrateETD2RK,
'ETD2RK', ..., order 2, dissipation True, fsal True, implicit=False, steps=1,
stiffly_accurate=False, stability='L')`. It **requires the `linear` accessor**:
`resolve(f, scheme_name='ETD2RK', need_linear=True)` raises a `TypeError` naming the
`linear` part before the solve if the RHS is a plain callable. Not stiffly accurate, so
no first-stage reuse. No embedded pair, so `error is None`.

**Verification** (all pinned in `tests/test_exponential.py`, 10 tests):

- **Order 2 on viscous Burgers** (vs the fine-`dt` RK4 reference, Gaussian IC):
  relative L2 error `[2.34e-01, 2.15e-02, 5.09e-03, 1.24e-03, 3.06e-04]` at
  `dt = [0.08, 0.04, 0.02, 0.01, 0.005]`, measured orders `[3.44, 2.08, 2.04, 2.02]`
  (the coarsest pair is pre-asymptotic; the asymptotic order is **2**, as advertised).
- **Exact on a purely linear problem** (`N = 0`, `x' = L·x`): the error is at the
  Krylov tolerance (~1e-9), **not** `O(h^2)` — the linear part is carried exactly, the
  point of an exponential integrator.
- **L-stable**: a stiff linear mode (`x' = -rate·x`, `rate = 10`, `h = 1`, `rate·h = 10`)
  is damped by `exp(-10) ~= 4.5e-5` (killed, well under 1e-3), in contrast to ROS3P's
  A-stability ceiling (`R(∞) = 1 − √3 ~= -0.732`) which only dampens.
- **Two stages / two `N` evals per step**, the Krylov count is reported, `error is None`.
- **Gradient parity** (autograd vs central finite difference): the linear problem agrees
  to ~1e-9 (the Krylov tolerance) and the semilinear problem to ~1e-5 (the FD error
  floor) — in contrast to ROS3P's O(dt) (a few percent) on the same semilinear problem.
- **`SemilinearRHS` required**: a plain callable is rejected with a `TypeError` before
  the solve; a `priorStep` is rejected with a `RuntimeWarning`.

**Cost** (sine IC, `dt = 0.02`, 20 steps; `SolveDiagnostics` summed over the two
stages) — the roadmap's cost gate is assessed on this:

| family      | scheme  | dt   | steps | `N` evals | `L`-matvecs | total | rel L2 err |
|-------------|---------|------|-------|-----------|-------------|-------|------------|
| implicit    | BDF2    | 0.02 | 20    | 200       | 193         | 393   | 3.16e-3    |
| implicit    | TR-BDF2 | 0.02 | 20    | 400       | 175         | 575   | 7.81e-4    |
| exponential | ETD2RK  | 0.02 | 20    | 40        | 1742        | 1782  | 2.33e-3    |

ETD2RK's 40 `N` evaluations are few (two per step), but its cost is dominated by the
**1742 `L`-matvec Krylov iterations**: each of the three per-step builds runs to the
full `m = 30` because `||h·L|| ~= 3.3` does not drive the Arnoldi residual below
`1e-10` early. On the work-unit metric ETD2RK (1782) is **not** competitive with the
order-2 bars (BDF2 393, TR-BDF2 575). **FLOP caveat:** each `L`-matvec is a cheap
`nu·u_xx` stencil, whereas each JFNK `rhs_evaluation` / GMRES iteration is a full
`N + L` (Jacobian-vector) evaluation, so the work-unit metric **overstates** ETD2RK's
true FLOP cost. The roadmap's exponential **cost gate is therefore assessed at
`exprb32`** (order 3) — the method that should carry the family's cost case — before
the exponential method set expands; the **order gate is met** (order 2). (Assessed in
§3.16: **met** — EXPRB32 is the cheapest order-3 method on the fixed-error work-unit
metric, 640 < ROS3P 706 < ESDIRK3 715 < ARK3 793.)

**Carve-outs.** Because ETD2RK needs the `linear` accessor, the generic *plain-
callable* problems (oscillator, forced, kepler, advection) cannot run it as registered.
`conftest.NEEDS_SEMILINEAR_RHS = {'ETD2RK'}` skips those in the `scheme`-fixture
tests; `tvd_analysis.SEMILINEAR_ONLY_IDENTIFIERS = {'etd2rk'}` records its TVD/SSP
verdict as "not applicable (structured semilinear RHS required)" (the model advection
problem has no `linear` part; Rosenbrock-W is *not* in that set — it falls back to the
combined `f`'s Jacobian); and `test_hamiltonian`'s direct-`IntegrationSchemes` area test
skips it the same way. Its order, driver contract, and gradient are covered in
`tests/test_exponential.py` on the semilinear viscous-Burgers problem instead.

**New tests / artifacts:** 10 in `tests/test_exponential.py`; ETD2RK rows in
`tests/test_tvd.py`, `tests/conftest.py`, `tests/test_hamiltonian.py`;
`scripts/exponential_benchmark.py` → `images/exponential_benchmark.png`; the ETD2RK
section of `viscous_burgers_demo.ipynb`.

### 3.16 Phase 7: EXPRB32 (exponential Rosenbrock) — done 2026-09-13

The second exponential method in the registry, and the one that carries the
exponential family's cost case — the roadmap's order-3 exponential Rosenbrock method,
which closes the Phase 7 cost gate (§3.15 deferred it here). Unlike ETD2RK it needs
no semilinear split: it freezes the **full** right-hand-side Jacobian, so it runs on
plain callables as registered (the generic `scheme`-fixture problems, including the
TVD/SSP advection sweep) and on `SemilinearRHS` just as well.

**Method.** **EXPRB32** (2-stage, **order 3**, **L-stable**, with an embedded order-2
estimator), **Hochbrueck, Ostermann & Schweitzer, *SIAM J. Numer. Anal.* 47(1) (2009)
786–803** (`literature/080717717.pdf`), in the frozen-operator reformulation of
§2.3/§6.5 with `F_n = f(t_n, y_n)` and `J_n = D_y f(t_n, y_n)` frozen at the step
start:

```
U2      = y_n + h·phi_1(h·J_n)·F_n   [+ h²·phi_2(h·J_n)·f_t     (non-autonomous)]
D2      = f(t_n + h, U2) − F_n − J_n·(U2 − y_n)   [− h·f_t]
y_{n+1} = U2 + 2h·phi_3(h·J_n)·D2
```

Two stages; **four full-`f` evaluations per step** with the default `f_t='fd'`
(`F_n`, the `f_t` probe, `f(t_n+h, U2)`, and the frozen-`J_n` application
`J_n·(U2 − y_n)`) and three with `f_t='none'`; **up to three `phi_k(h·J_n)` Krylov
builds per step** (`phi_1` and, when `f_t` is active, `phi_2` in stage 1; `phi_3` in
stage 2 — skipped when `f_t = 0` and, on linear problems, when `D2` is at rounding).
The embedded estimate is the order-2 exp-Rosenbrock–Euler stage `û = U2`, so the
estimate is the corrector's correction itself,
`y_{n+1} − û = 2h·phi_3(h·J_n)·D2` — no extra evaluation.

**Full Jacobian, no semilinear split.** `J_n` is the exact Jacobian action
(`jfnk.jvp_matvec` over the full step machinery — preprocess + `f` + postprocess, so
the action is the Jacobian of the *full* right-hand side, `w='jvp'` default) or
finite differences (`w='fd'`); every `phi_k(h·J_n)v` is applied **matrix-free** by the
same `krylov_phi` Arnoldi machinery as ETD2RK. **There is deliberately no
`w='linear'` option**: a `J = L`-only frozen operator is order 3 only for a linear
right-hand side and **first order** on a genuinely nonlinear problem (the same order
drop ROS3P's `w='linear'` documents, §3.14), so it is documented, not shipped. The
non-autonomous term is a one-sided finite difference by default (`f_t='fd'`, step
`eps_machine^(1/3)·max(1, |dt|)` — the `_default_ft_eps` shared with ROS3P);
`f_t='none'` for autonomous right-hand sides saves one `f` evaluation and the `phi_2`
build (the probe returns `f_t = 0` bit-exactly and the build is skipped).

**Gradient.** Frozen-`J_n` re-attachment: each `phi_k(h·J_n)` action is re-attached
with the transpose `phi_k(h·J_n^T)` (the same Krylov machinery on `v -> dt·J_n^T v`,
`J_n^T` from one VJP graph at `(t_n, y_n)` — the ROS3P `wT` pattern), and
`J_n·(U2 − y_n)` with `J_n^T`. On a linear right-hand side the frozen operator *is*
the exact constant operator, so the gradient is **exact up to the Krylov tolerance**
(measured parity 7.2e-10). On a semilinear one, every frozen-operator occurrence in
the step map carries a factor of `dt` (`U2 − y_n = O(dt)`, and the `phi` inputs are
`O(dt)`), so the error in the dropped `dJ_n/dy` terms is **O(dt²)·||dJ/dy||** — one
power of `dt` better than the Rosenbrock frozen-`W` adjoint (O(dt) on semilinear);
the measured parity on viscous Burgers (4.6e-8 at `dt = 0.02`) sits at the
finite-difference floor. The `D2` zero guard on linear problems is gradient-safe: the
adjoint chain through `D2` vanishes on a linear graph.

**Registration.** `exprb32 = 57` in the enum; `IntegrationScheme(integrateEXPRB32,
'EXPRB32', IntegrationSchemeType.exprb32, 3, True, True, implicit=False, steps=1,
stiffly_accurate=False, stability='L')`. No `need_linear` — plain callables and
`SemilinearRHS` both resolve to the combined `f`. Not stiffly accurate (the last
stage is `U2`, not the step), so no first-stage reuse. The embedded (3, 2) pair means
`error` carries the estimate (inverse of ETD2RK's `error is None`).

**Verification** (all pinned in `tests/test_exponential.py`, 10 tests; the generic
`scheme`-fixture tests run unmodified on EXPRB32):

- **Order 3 on viscous Burgers** (fine-`dt` RK4 reference, Gaussian IC): relative L2
  error `[2.72e-03, 3.50e-04, 4.41e-05, 5.55e-06, 6.97e-07]` at
  `dt = [0.08, 0.04, 0.02, 0.01, 0.005]`, measured orders `[2.96, 2.99, 2.99, 2.99]`.
  The error constant is ~13x smaller than ROS3P's at the same `dt` (4.4e-5 vs
  5.6e-4 at `dt = 0.02`) — that is what buys the cost gate below.
- **Exact on a purely linear right-hand side**: `J_n` is constant, `D2` vanishes,
  the step is `exp(h·J_n)·y_n` applied through the Krylov `phi_1` build (the `phi_3`
  build skipped by the zero guard), and the error is at the Krylov tolerance — not
  `O(h^3)`.
- **L-stable**: a stiff linear mode (`rate = 10`, `h = 1`, `rate·h = 10`) is damped
  by `exp(-10) ~= 4.5e-5` (killed, well under 1e-3) — the exponential signature, in
  contrast to the Rosenbrock A-stability ceiling `1 − √3 ≈ −0.732`.
- **Driver contract**: two stages; full-`f`-class evaluations per step are
  `[2, 2]` per stage with `f_t='fd'` and `[1, 2]` with `f_t='none'` (the Krylov
  builds are counted separately, in `gmres_iterations`); `error` is not None; a
  plain callable runs as-is (the inverse of ETD2RK's `TypeError`); `priorStep` is
  rejected with a `RuntimeWarning`; the caller state is not mutated.
- **Embedded estimate**: `O(h^3)` local error — it shrinks 8.75x under `dt -> dt/2`
  on viscous Burgers (a (3, 2) pair is active on a non-autonomous problem; on a
  linear autonomous one `D2` vanishes and the estimate degenerates to exactly zero).
- **Gradient parity** (autograd vs central FD): linear 7.2e-10 (the Krylov
  tolerance), semilinear 4.6e-8 (the FD floor; the O(dt²) frozen-`J_n` error is
  below it at `dt = 0.02`).
- **TVD/SSP verdict** (`tests/test_tvd.py`): no tableau, so no SSP coefficient; on
  the model advection problem (linear, autonomous) the step is the exact per-mode
  exponential `e^{μ(e^{iθ}−1)}` — a contraction at every CFL — so the sweep finds no
  TV increase up to the CFL cap: `(None, 5.0, True)`.

**Cost** (sine IC, `T = 0.4`; `SolveDiagnostics` summed over the two stages) — the
roadmap's exponential **cost gate** is assessed here. At the same `dt = 0.02` (20
steps), EXPRB32 is the *most expensive* stiff method on the work-unit metric
(80 `f` evals + 1200 JVP Krylov iters = 1280; both per-step builds run to the full
`m = 30`) — but it also reaches 1.43e-5, ~8x below the best order-3 bar. The gate's
"to a fixed error" half is what the order-3 error constant buys: each order-3 method
runs the `dt` ladder and uses the **largest** `dt` that reaches the target error
(`rel L2 ≤ 1e-3`, sine IC):

| scheme               | dt   | steps | rel L2 err | total (work units) |
|----------------------|------|-------|------------|--------------------|
| **EXPRB32 (jvp)**    | 0.04 | 10    | 1.22e-04   | **640**            |
| ROS3P (jvp)          | 0.02 | 20    | 2.04e-04   | 706                |
| ESDIRK3(2)4L[2]SA    | 0.04 | 10    | 5.21e-04   | 715                |
| ARK3(2)4L[2]SA       | 0.02 | 20    | 2.63e-04   | 793                |

**The gate is met: EXPRB32 is the cheapest order-3 method on the work-unit metric at
fixed error (640 < 706 < 715 < 793)**, and it lands the smallest error of the four at
its own `dt` (1.22e-4). The comparison is fair in cost class: each of EXPRB32's
Krylov iterations is a **full Jacobian JVP** — a forward-mode AD sweep through the
full right-hand side, the same cost class as a full `f` evaluation and as ROS3P's
frozen-Jacobian GMRES sweeps (which sit in ROS3P's `rhs_evaluations`) — in contrast
to ETD2RK's cheap `nu·u_xx` `L`-matvecs, where the metric overstates cost (§3.15).
Two honest caveats. First, the 9% margin over ROS3P (706) is thin: it is a
work-unit win in the same cost class, not a FLOP blowout (a JVP sweep carries a
small forward-mode overhead, which if anything widens the gap slightly, but the
margin is what it is). Second, at the same `dt = 0.02` EXPRB32 is the most expensive
stiff method here (1280) — the gate is the fixed-error one, and the ~13x smaller
order-3 error constant is exactly what turns that around. This closes the family's
open cost gate: ETD2RK's order-2 work-unit count (1782, §3.15) was never a clean bar
(cheap `L`-matvecs) and stays documented as such; the exponential family's cost case
now stands on EXPRB32.

**Carve-outs.** Because EXPRB32's step is the *exact exponential flow* on a linear
autonomous right-hand side (constant `J_n`, vanishing `D2`), its error on the
oscillator / damped problems is at the roundoff floor and no order is measurable
(`measured_order` → None). `conftest.EXACT_ON_LINEAR_PROBLEMS = {'EXPRB32'}` skips
those in the generic order tests (the order is pinned on `forced`, `kepler`, and
viscous Burgers instead), in `test_hamiltonian`'s oscillator area test (the exact
flow is area-preserving trivially, so the defect is at rounding; EXPRB32 is
deliberately **not** in `LINEARLY_SYMPLECTIC` — its non-symplectic character on
nonlinear problems is pinned by the kepler dissipation-flag test, which passes with
growth > 2), and in the two embedded-pair order tests (the (3, 2) estimate degenerates
to zero on linear autonomous problems, so they are measured on `forced`). No carve-out
for the TVD/SSP sweep: EXPRB32 runs on the plain-callable advection problem (full-
Jacobian fallback — unlike ETD2RK).

**New tests / artifacts:** 10 in `tests/test_exponential.py` (the ETD2RK helpers
`_run_burgers` / `_grad_parity_error` generalized with a `scheme` parameter, the
ETD2RK tests unchanged); an EXPRB32 verdict `(None, 5.0, True)` in
`tests/test_tvd.py`; `tests/conftest.py` (`EXACT_ON_LINEAR_PROBLEMS`,
`LINEAR_AUTONOMOUS_PROBLEMS`), `tests/test_hamiltonian.py`, `tests/test_embedded.py`;
EXPRB32 rows + the fixed-error gate table in `scripts/exponential_benchmark.py` →
`images/exponential_benchmark.png`; the EXPRB32 sections of
`viscous_burgers_demo.ipynb` (§9/§10/§11/§12).

## 3.17 Adaptive step control and dense output (roadmap Phase 11)

Phase 11 lands as **helpers, not a solver** — the decision the roadmap's open
contract item asked for. The library exposes two functions in
`src/warpSPHIntegrators/adaptive.py`, and the caller drives the accept/reject loop:

- `estimate_error_norm(result, rtol=1e-3, atol=1e-6)` — the embedded-pair
  difference as a dimensionless number: `state_norm(result.error,
  reference=result.state)`, the weighted-RMS convention the DIRK driver already
  uses ("< 1.0 means at the tolerance") — the roadmap's [?] error-norm item,
  resolved by reusing `fields.state_norm` rather than introducing a second
  scale. Raises a `ValueError` for a scheme that emits no estimate.
- `propose_dt(error_norm, dt, order, *, target=1.0, safety=0.9, growth_min=0.2,
  growth_max=5.0)` — the classic predictive controller,
  `dt * clamp(safety * (target/error_norm)**(1/order), growth_min,
  growth_max)`; a zero error proposes `dt * growth_max`. The controller itself
  never knows whether a step was rejected; "never grow on a rejection" is a
  property of the *loop* (the SciPy convention) and the demo loops apply it.

The demo loop — step, measure, accept/reject, propose; ~15 lines — lives in
`scripts/adaptive_benchmark.py::run_adaptive` and
`tests/test_adaptive.py::run_adaptive`, deliberately not in the library.

**Why helpers, not a driver.** §3.8's "no solver contract without a downstream"
caution: an adaptive driver would own the loop, the output times, and the event
handling, and none of that has a downstream here yet. The helpers are useful
on their own, and the loop they leave to the caller is small.

**The estimate emitters and their `q`.** Ten registered schemes emit
`IntegrationResult.error` (the roadmap's "eight" predates ROS3P and EXPRB32):
the (p, p−1) embedded pairs Bogacki-Shampine 3(2), Dormand-Prince 5(4), and
Cash-Karp 5(4); the SUNDIALS ARKODE v7.9.0 built-in estimates TR-BDF2,
ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA, ARK3(2)4L[2]SA, and ARK4(3)6L[2]SA; and
ROS3P and EXPRB32's published embedded pairs. The controller needs the power
`q` that the estimate scales as `h^q`: `q = scheme.order` for nine of the ten,
and **`q = order + 1 = 3` for TR-BDF2** — its published SUNDIALS pair (both
branches order 2) makes the estimate the propagated branch's true local error,
`O(h^3)`. The `estimate_order` helper in the two demo loops pins the exception.

**Multistep: refused explicitly.** Adaptive `dt` and the multistep family are
mutually exclusive, by design rather than accident: `StepHistory` restarts on
any `dt` change (pinned in
`tests/test_groundwork.py::test_step_history_restarts_on_dt_change`), so every
step of a variable-`dt` run would throw away the history a multistep scheme
exists to build — and the family emits no estimate to drive a controller in the
first place, so `estimate_error_norm` raises for it. Variable-step BDF/Adams
coefficients (the CVODE/LSODA route) remain the future work if a downstream
asks for it.

**Dense output: Shampine's 1986 quartic.** DP5's published continuous extension
(Shampine, "Some Practical Runge-Kutta Formulas", *Mathematics of Computation*
46(173) 1986, 135–150 — the optimum-`c_6` variant, as implemented in SciPy's
`RK45`) is

    y(θ) = y_n + h · Σ_{i=1..7} P_i(θ) k_i,   P_i(θ) = p₁θ + p₂θ² + p₃θ³ + p₄θ⁴

(no constant term; the `7 × 4` coefficient matrix `_DP5_DENSE_P` is transcribed
verbatim from `scipy/integrate/_ivp/rk.py`, verified against the local SciPy
1.18.0 source). The stage mapping is 1:1 in order with `RungeKuttaB`'s
`k1..k7` (SciPy's `K[i]` is the repo's `k_{i+1}`; `k7` is the FSAL stage at
`c = 1`, whose row is `b` itself). Verified:

- `P(1) == b` (max diff 7e-16) and `P(0) == 0`, so the interpolant is **exact
  at both step endpoints** — `θ = 1` re-applies the propagated weights through
  `butcher._weighted_update` and is bit-for-bit the step's returned state, and
  `θ = 0` returns the input state itself;
- the continuous order conditions hold **through order 4 only** (an 8-point
  Vandermonde fit of the extension on a linear problem matches `(θz)^m/m!` for
  `m ≤ 4` and deviates at `m = 5` by ~1.8e-4), so the interior interpolation
  error is `O(dt^5)`.

Measured on the linear oscillator (one step, queried at `θ = 0.4`, against the
analytic solution): rates 4.832 / 4.961 / 4.990 / 4.998 / 4.999 as `dt` halves
from 0.4 to 0.0125 — fifth-order interior accuracy, one order below the
propagator, exactly what the order conditions predict. Dense output is
**DP5-only**: the other nine emitters have no published continuous extension,
and `dormand_prince_dense_output` refuses their stage vectors (a seven-stage
check) rather than guess.

**The gate (van der Pol, μ = 10, T = 5, DP5, dt0 = 0.1** — deliberately far
outside the Phase 8 explicit wall, fast eigenvalue ~ μ(1 + x²) = O(10–20)):

| run | endpoint error | accepted steps |
|---|---|---|
| reference floor (dt = 5e-4 vs 2.5e-4) | 8.8e-14 | — |
| fixed dt = 0.016 | 1.3e-06 | 312 |
| fixed dt = 0.008 | 2.0e-08 | 625 |
| adaptive rtol = 1e-4 | 7.2e-05 | 54 (18 rejected) |
| adaptive rtol = 1e-5 | 3.7e-06 | 75 (17 rejected) |
| adaptive rtol = 1e-6 | 3.0e-08 | 107 (19 rejected) |

Both roadmap gates hold. Step rejection triggers where the wall is (17–19
rejections out of 72–126 attempts), and the stiff problem completes in
materially fewer steps under control than at the fixed `dt` needed for the same
final error: 75 vs 312 (4.2×) at rtol = 1e-5, and 107 vs 625 (5.8×) at rtol =
1e-6. The `tests/test_adaptive.py` gate pins the former with a margin
(`accepted < 0.75 ×` the coarsest matching fixed step count). No fixed-ladder
point diverged; dt = 0.016 is already deep inside the wall.

**New tests / artifacts:** 24 in `tests/test_adaptive.py` (controller math,
growth clamps, the zero-error and `target` paths, argument validation; the norm
convention and the `ValueError` for estimate-less schemes; finite-positive
norms on all ten emitters; the loop contract on the oscillator — accepted steps
at the target, rejections occur, the endpoint reached; the van der Pol gate;
dense-output endpoint exactness, order-4 interior rates, the half-step
cross-check, and the validation errors); `scripts/adaptive_benchmark.py` →
`images/adaptive_benchmark.png` (error vs step count on the ladder, the
controller in flight, the interior interpolation rate).

---

## Reproducing the results

### `scripts/step_reuse_convergence.py` (committed)

A reusable tool for the first-stage reuse study behind §0's headline numbers. It
predicts the retained order from the tableau and measures it, so the two can be
compared:

```bash
conda activate warp

# one scheme in detail: error table, both orders, prediction vs measurement
python scripts/step_reuse_convergence.py --scheme RK4
python scripts/step_reuse_convergence.py --scheme 'SSP RK3' --problem forced

# every registered scheme, one line each
python scripts/step_reuse_convergence.py --all

# log-log convergence plot with reference slopes
python scripts/step_reuse_convergence.py --scheme RK4 --plot rk4_reuse.png
```

`--problem oscillator` is autonomous, `--problem forced` (`x'' = cos t`) is not — the
gap between them is what would expose a stage-time bug in a scheme's tableau or
hand-rolled logic, which is why `test_convergence.py` runs every scheme on both.

Sample output showing the trade concretely:

```
Scheme   : RK4 (registered order 4)
Reuse    : c_s = 1 but a[-1] != b; a[-1] is a method of order 2, so Y_s = y^{n+1} + O(dt^3)
           predicted order with reuse: 3 -- loses 1 order(s)

       dt    error (no reuse)      p     error (reuse)      p   reuse / no reuse
  0.10000          1.1612e-04     --        7.4343e-04     --               6.4x
  0.05000          7.0943e-06   4.03        1.0852e-04   2.78              15.3x
  0.02500          4.3710e-07   4.02        1.4367e-05   2.92              32.9x
  0.01250          2.7103e-08   4.01        1.8405e-06   2.96              67.9x

measured order   no reuse: 4.01
                    reuse: 2.96   (predicted 3: MATCH)
```

versus the midpoint scheme, where reuse is free apart from a fixed 2.5× constant:

```
Scheme   : Midpoint (registered order 2)
Reuse    : c_s = 0.5 != 1 ...; b[0] == 0, so the stale k0 only enters through the stage equations
           predicted order with reuse: 2 -- lossless
measured order   no reuse: 2.01
                    reuse: 2.00   (predicted 2: MATCH)
  >> Reuse is safe for this scheme: convergence order is retained.
```

### `scripts/adaptive_benchmark.py` (committed)

Phase 11's adaptive-control and dense-output benchmark (§3.17): the fixed-dt
ladder against a fine-dt reference plus the adaptive runs on van der Pol
μ = 10 (the roadmap gates — rejections trigger at the explicit wall, and the
accepted step count falls under control), the controller's dt / error norm in
flight, and DP5 dense output's interior interpolation rate on the linear
oscillator.

```bash
conda activate warp

OMP_NUM_THREADS=4 python scripts/adaptive_benchmark.py   # -> images/adaptive_benchmark.png
```

### `tests/` (committed)

`pytest` from the repository root, inside the `warp` environment. Last run 2026-08-07
against `e327d8a`: **906 passed, 54 skipped in 119 s** out of 960 collected, no
failures. The skips are almost all per-scheme opt-outs from parametrised tests —
a scheme that does not implement reuse, or loses no order under it, or is unstable on
an oscillatory problem — plus the CUDA-gated cases in `test_backend_dispatch.py`.

| file | what it pins down |
|---|---|
| `test_convergence.py` | every scheme reaches its registered order on `oscillator` (autonomous), `forced` (time-dependent) and `kepler` (nonlinear); errors decrease monotonically; the four splitting schemes lose order on `damped` and nothing else does |
| `test_step_reuse.py` | measured reuse order matches `scheme.reuse_order` exactly — neither optimistic nor pessimistic; FSAL reuse is bit-for-bit free; Forward Euler refuses; the four non-reusing schemes warn, ignore, and do not leak `priorStep` into the RHS; the warning fires once and names the achieved order |
| `test_state.py` | `nograd()` on non-tensor fields; untagged-field agreement between the two clone paths; stage times translate by `dt` each step; `t` stays a Python `float`; the update helpers do not advance time; `getIntegrator` accepts all three spellings; schemes do not mutate the caller |
| `test_embedded.py` | the error estimate reaches the caller, scales as `dt^p`, brackets the true error, and the *high-order* branch is the one propagated; tableau row sums and the FSAL property |
| `test_hamiltonian.py` | the `dissipation` flag predicts energy behaviour: symplectic schemes stay inside an `O(dt^p)` band over an 8× longer run, dissipative ones grow ~linearly |
| `test_copied_fields.py` | `copied()` fields arrive holding the *last stage's* value, `ephemeral()` ones do not survive at all |
| `test_backend_dispatch.py` | `wp.array`, lists, dicts and nested containers are cloned rather than aliased; `to(device)` reaches inside them; unknown types warn once; and no scheme hands the caller's own container back |
| `test_kwargs_passthrough.py` | caller kwargs survive every scheme — `verbose` in both states, alongside `priorStep`, and an unknown kwarg reaching the RHS intact |

The shared harness — three tagged-field reference systems, four problems with analytic
solutions, and the order/energy measurement helpers — lives in
[src/warpSPHIntegrators/testing.py](src/warpSPHIntegrators/testing.py) so that the tests and
`scripts/step_reuse_convergence.py` share one definition. It doubles as the smallest
complete worked example of the protocol.
