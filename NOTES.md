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
- **Multistep and implicit** (§3) — the two entries in the README's "Known
  Limitations". Both are much cheaper here than for a general-purpose library, because
  the surrounding simulation does not resort particles, holds `dt` constant, and
  carries its neighbour list through the state (§3.0). Recommended: ~2 weeks for
  Phase 0 → 2 → 1, headlined by **implicit midpoint**, the symplectic second-order
  scheme that — unlike everything currently registered — holds order 2 for
  velocity-dependent forces.
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

The question "can I run `f` under forward-mode AD to get the Jacobian action?" splits
into two needs that are usually conflated, and that have **opposite** backend support.

| Need | What it requires | torch | warp |
|---|---|---|---|
| **Solving** the stage equation with Newton: `(I − dt·a_ii·J)·δ = −G` | Jacobian-*vector* products `J·v` — **forward mode** (`jvp`) | `torch.func.jvp` ✓ | ✗ no forward mode |
| **Differentiating through** a converged solve, for training | *vector*-Jacobian products `Jᵀ·λ` via the implicit function theorem — **reverse mode** (`vjp`) | ✓ | `wp.Tape` ✓ |

So warp has exactly the mode the *gradient* needs and lacks exactly the mode the
*solve* needs. That is a real asymmetry, and it is worth stating clearly because it
inverts the intuition: the differentiability story is the part that ports, and the
solver is the part that does not.

**It does not block anything, because Newton does not actually need AD.** An inexact
Newton needs the *residual* to be exact — and it is, it is just `f` — while the matvec
`J·v` only has to be good enough to produce a descent direction. A finite-difference
directional derivative `J·v ≈ (f(Y+εv) − f(Y))/ε` costs one extra RHS evaluation,
needs no AD of any kind, and works identically under torch and warp.

[verified] Backward Euler with a purely finite-difference Jacobian, no autodiff:
bounded and correctly L-stably damped at every stiffness up to `dt·ω = 1000`, where
Picard diverges past `dt·ω = 1` (table in §3.2). FD accuracy is not the limiting
factor.

**Recommended solver ladder**, cheapest first:

1. **Fixed-count Picard (2 iterations).** Non-stiff. No AD, no norm, no branching.
   Unrolls to a fixed-depth autograd graph, so it is differentiable in both backends
   by construction, graph-capturable, and deterministic. **This covers the primary use
   case and is all Phase 2 ships.**
2. **JFNK with FD matvecs.** Stiff, backend-agnostic. Needs the flatten/unflatten
   bijection over integrated fields (mechanical — the field metadata already names
   them) plus GMRES.
3. **`torch.func.jvp` matvecs.** Same as 2 with exact matvecs, as a torch-only fast
   path. A speed and robustness optimisation, *not* a capability gate.
4. **User-supplied `solve_linear`.** An ISPH code already owns a pressure-projection
   solve and will always beat a generic Krylov method. This should be the contract;
   1–3 are the fallbacks.

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

### 3.5 Shared groundwork (prerequisite for both)

| | Item | Where | Effort |
|---|---|---|---|
| **S1** | `state_norm(state, rtol, atol)` — weighted RMS over integrated fields, Hairer–Wanner style, plus `state_difference`. `butcher._error_estimate` already builds a difference, so generalise rather than duplicate. Stable indexing (§3.0) means this is a plain elementwise reduction with no identity matching, which is also exactly what the ML bindings want. **Adaptive `dt` (§2.3) needs the identical primitive.** | `fields.py` | 1 d |
| **S2** | `StepHistory` — an ordered container of `(t, dt, derivative projection)`, carried on `IntegrationResult` and accepted as a kwarg, exactly as `priorStep` is today. `priorStep` becomes the one-entry degenerate case; do **not** ship two overlapping reuse mechanisms. Store a *projection* onto the tagged derivative fields, not the whole user update object, which in a real SPH run carries far more than derivatives. | `specs.py`, all schemes | 1–2 d |
| **S2g** | Two cheap history guards, replacing ~2 d of variable-step machinery. **(a)** record `dt` in each entry and restart if it changes — this is what makes fixed multistep coefficients honest under §3.0's "constant *for the most part*". **(b)** record the identity of the `uid` tensor (`data_ptr` + shape, or a generation counter) and restart if it moves. Both turn a silent wrong answer into an automatic, visible restart. | `specs.py` | 0.5 d |
| **S3** | `IntegrationScheme` metadata: `implicit: bool`, `steps: int`, `stiffly_accurate: bool`, `stability: 'A'\|'L'\|'A(α)'\|None`, `startup_order: int`. `reuse.py` must not choke on schemes with no tableau *and* no `HANDROLLED_REUSE` entry — it currently returns `None` with a reason, which is right, but the multistep case wants its own answer. | `util.py`, `integration.py`, `reuse.py` | 1 d |
| **S4** | `NonlinearSolver` protocol: `solve(residual, y0, norm, **opts) -> (y, converged, iterations)`. Ship `FixedPointSolver` with a **fixed iteration count** as the default (§3.1: 2 iterations for order 2). Pluggable from the start — rung 4 of the §3.4 ladder is the one that matters long-term. When rung 2 (JFNK) is built, keep its matvec a directional finite difference, never a dense Jacobian — see the caution in §3.4. | new `solvers.py` | 1–2 d |
| **S5** | Generalise `testing.run` / `conftest.ALL_SCHEMES`. Both assume a stateless one-step callable `scheme(system, dt, f)`. A multistep scheme needs history threading and a starter, so every existing test breaks the day one is registered unless this lands first. | `testing.py`, `tests/conftest.py` | 1 d |

**~5–6 engineer-days** for someone with this codebase in context. The `preprocess`
caching policy that a general-purpose implicit driver would normally need as a
blocker is **not needed** here — the simulation already carries and revalidates
adjacency through the state (§3.0), by a better mechanism than a library-level cache
would be. The one thing to check is that the *implicit driver* reuses a single stage
buffer across iterations rather than calling `initializeNewState` per iteration as the
probe does, so the adjacency is carried once rather than re-cloned each time. That is
also what §2.1's buffer pooling wants, so it is aligned work, not a detour.

### 3.6 Valid schemes and what each costs

Effort is *marginal*, on top of the groundwork and the driver its group needs.
"tableau only" means the scheme is a data entry in `getButcherTableau` plus a
registry line — the same one-line cost that adding Dormand–Prince was.

#### Diagonally implicit RK (sequential 1-stage solves)

The probe driver is ~60 lines; budget ~150 for a registered one with the solver
protocol, verbose output and scheme metadata wired in.

| Scheme | Order | Stages | Stability | Symplectic | Marginal effort | Worth it? |
|---|---|---|---|---|---|---|
| **Implicit midpoint** (Gauss–Legendre s=1) | 2 | 1 | A | **yes** | tableau only | **yes — §3.3** |
| **Backward Euler** | 1 | 1 | L | no | tableau only | yes, as the reference/fallback |
| **Trapezoidal / Crank–Nicolson** (Lobatto IIIA-2) | 2 | 2 | A, not L | symmetric | tableau only | yes — cheap, and the classic pair with BDF2 |
| **SDIRK2** (Ellsiepen, γ=1−√2/2) | 2 | 2 | L | no | tableau only | yes — L-stability matters for real stiffness |
| **TR-BDF2** | 2(3) | 3 | L | no | tableau only | yes — stiffly accurate, embedded estimate, and the embedded path already works |
| **ESDIRK3(2)4L[2]SA** (Kennedy–Carpenter) | 3(2) | 4 | L | no | tableau only | yes — explicit first stage is FSAL-shaped, so `reuse.py` handles it |
| **ESDIRK4(3)6L[2]SA** | 4(3) | 6 | L | no | tableau only | later — same driver, more coefficients |

The `a` matrix stops being strictly lower triangular. `reuse.py`'s
`tableau_reuse_analysis` reads `a[-1]` and `c[-1]` and will need to understand
"stiffly accurate" (`a[-1] == b`) as the implicit analogue of FSAL — a small, natural
extension of code that already exists.

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

#### Linear multistep

| Scheme | Order | Evals/step | History | Implicit | Marginal effort |
|---|---|---|---|---|---|
| **Adams–Bashforth 2–5** | k | **1** | k−1 updates | no | 1–2 d for the whole family |
| **ABM predictor–corrector (PECE)** | k | 2 | k−1 updates | no (fixed corrections) | +1 d |
| **Adams–Moulton 2–4** as a true corrector | k | solve | k−1 updates | yes | +1 d after the DIRK solver |
| **BDF1–2** | 1, 2 | solve | k states | yes, A-stable | **2 d** — constant `dt` (§3.0) removes the variable-coefficient work entirely |
| **BDF3–6** | 3–6 | solve | k states | yes, A(α)-stable only | +1 d; BDF7+ is not zero-stable, do not offer it |
| **Störmer–Cowell / multistep Nyström** (`x'' = f(x)`) | k | 1 | k states | no | 3 d — fits `PositionUpdateSpec` well; Störmer–Verlet is its 2-step case |
| **Gauss–Jackson** (8th-order Störmer–Cowell) | 8 | 1 | 8 | no | 1 wk — niche, orbital mechanics |

AB's one evaluation per step is the real prize here: **one force evaluation per step
at order 4**, against RK4's four, with fixed coefficients that are exactly correct
under constant `dt` and a history that never expires under stable indexing (§3.0).
Both of the things that normally make multistep painful are absent. Note that no
linear multistep method is symplectic for a general Hamiltonian (Tang, 1993);
symmetric LMMs applied to `x''=f(x)` do show good long-time energy behaviour, but they
are subject to parasitic-root instability, so don't market them as symplectic.

#### IMEX / additive RK — the right answer for SPH, and the most work

**ARK3(2)4L[2]SA** and **ARK4(3)6L[2]SA** (Kennedy & Carpenter) pair an ESDIRK
tableau for the stiff terms with an ERK tableau for the rest: viscosity, surface
tension or the pressure term implicit, advection explicit. This is what production
stiff-SPH actually wants, and it sidesteps §3.2 — the implicit part is the part with a
tractable, often *linear* operator.

It needs a split right-hand side (`f_explicit`, `f_implicit`), which is a **protocol
change**: `updateStep` returns one update object today. Effort **1 wk** on top of a
working DIRK driver and Newton solver. Gate it on a downstream that has the split.

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
   AB4 and AB2. The starter must be registered scheme metadata, not caller policy.
   **This is now the only structural obstacle to multistep, and it is a solved
   problem** — the FSAL pairs added in v0.5.0 are exactly the right starters.
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

11. **Warp's lack of forward-mode AD does not block the solver.** FD directional
    derivatives drive Newton to `dt·ω = 1000` with no AD at all [verified], and a
    fixed-count Picard needs no Jacobian action whatsoever. `torch.func.jvp` is a
    fast path, not a gate. The thing that *does* need care at scale is the FD matvec
    staying a directional derivative rather than a dense Jacobian — see §3.4's caution.
12. **Warp graph capture is fine with a fixed iteration count.** A data-dependent
    iteration count breaks capture (§2.1); the recommended default does not have one.
    Decide alongside §2.2, but the constraint points the same way as ML determinism
    does, which is a rare piece of luck.

### 3.8 Phased plan

Each phase is independently shippable and independently useful.

| Phase | Content | Effort | Gate |
|---|---|---|---|
| **0** | S1–S5 + S2g groundwork (§3.5) | 5–6 d | none — S1 also unblocks §2.3's adaptive `dt` |
| **2** | DIRK driver + fixed-count `FixedPointSolver` + implicit midpoint, backward Euler, trapezoidal, SDIRK2, TR-BDF2, ESDIRK3(2) | 4–5 d | Phase 0 |
| **1** | Explicit multistep: AB2–5 + ABM PECE, FSAL starter, `dt`/`uid` guards | 2–3 d | Phase 0 |
| **3** | JFNK with FD matvecs + user `solve_linear` hook + particle masking | 1–1.5 wk | Phase 2, **and** a downstream that is actually stiff |
| **4** | BDF1–6, fixed coefficients | 3–4 d | Phase 3 |
| **5** | IMEX / ARK, split right-hand side | 1 wk | Phase 3 + a downstream with a split RHS |
| **6** | Fully implicit: Gauss–Legendre, Radau IIA | 1–1.5 wk | demand-driven; symplectic order 4 is the draw |

Phases 1 and 2 are listed out of numeric order deliberately: Phase 1's old gate —
"measure the resort cadence first" — is void, because there is no resorting (§3.0),
but Phase 2 has the stronger standalone case, so it should land first.

**Recommendation: Phase 0 → 2 → 1, ~2 weeks total, then stop and reassess.**

- **Phase 2 is the headline.** Implicit midpoint is symplectic, A-stable, and holds
  order 2 for velocity-dependent forces — which nothing currently registered does
  (§3.3). At a fixed 2 Picard iterations it needs no norm, no Jacobian, no branching,
  and unrolls to a fixed-depth graph, so it is differentiable in both backends by
  construction and CUDA-graph-capturable. Five more tableaus come along for one
  registry line each.
- **Phase 1 is now unconditionally worth doing**, where before it was gated on an
  unmeasured number. AB4 at one force evaluation per step against RK4's four, with
  exactly-correct fixed coefficients and a history that never expires. The one thing
  to weigh first is pain point 2 in §3.7: k-step history deepens BPTT by k, so if
  training memory is already the binding constraint, the win is smaller than the
  evaluation count suggests. That is a question about the training setup, not about
  this library, and it is worth answering before spending the 2–3 days.
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
