# `sphWarpIntegrators` — architecture notes, defects, and improvement plan

Analysis date: 2026-08-05, against `ba43c6e` (v0.4.5), conda env `warp`
(Python 3.14.6, torch 2.13.0+cu130, warp 1.15.0, numpy 2.4.6).

Every claim marked **[verified]** was reproduced by running the code; the probe
scripts are described in [Reproducing the results](#reproducing-the-results).

---

## 0. Status — what has been fixed (v0.5.0)

Sections 2.1 through 2.16 below are the **original analysis**, kept as written so the
reasoning behind each fix stays on record. Everything in P0, P0.5 and P1 of the plan in
§5 is now done, and `tests/` (680 tests) is what keeps it done.

| § | Defect | Status |
|---|---|---|
| 2.1 | `priorStep` reuse valid nowhere in particular, with no way to tell | **fixed** — `integrators.reuse` promoted into the library; `step_reuse_order` / `supports_step_reuse` / `is_fsal`; `reuse_order` + `fsal` carried on `IntegrationScheme`; warns once, never hard-errors |
| 2.1a | No FSAL tableau exists | **fixed** — Bogacki–Shampine 3(2), Dormand–Prince 5(4), Cash–Karp 5(4) added; FSAL falls out of the tableau automatically |
| 2.2 | Wrong stage times → order collapse on time-dependent RHS | **fixed** — every scheme now measures full order on `forced` |
| 2.3 | Stage 0 pinned to `t=0` | **fixed** — every scheme sets stage-0 time from `initialState.t` |
| 2.4 | `velocityVerlet` crashes on `priorStep` | **fixed** — one `unpack_prior_step` helper, used by all three Verlet schemes and the Butcher path |
| 2.5 | `nograd()` raises on non-tensor fields | **fixed** — `_op` precedence |
| 2.6 | `clone()` / `initializeNewState()` disagree on untagged fields | **fixed** — single `field_behavior()` resolver, default `constant` |
| 2.7 | `copied` documented but never copied | **fixed** — wired into `finalizeSystem` via `lastStageSystem` |
| 2.8 | Embedded-pair path broken and unused | **fixed** — `b_` leak gone, `IntegrationResult.error` returns the estimate, three pairs registered |
| 2.9 | `priorStep` leaks into the user's RHS | **fixed** — `reject_prior_step` in every scheme that cannot reuse |
| 2.10 | `TVDRK3` finalize arity | **fixed** — passes the equivalent Butcher weights |
| 2.11 | `copy.deepcopy` in tvd/ruth | **fixed** — `initializeNewState` everywhere |
| 2.12 | Schemes mutate the caller's state; skip `initialize` | **fixed** — all schemes evaluate on stage buffers and call `initializeSystem` |
| 2.13 | Registry metadata wrong or arbitrary | **fixed** — Semi-Implicit Euler is order 1; `getIntegrator` accepts all three spellings; `dissipation` now means "energy drifts secularly" and is tested |
| 2.14 | `integrateDensity` diffSPH-ism | **fixed** — removed |
| 2.15 | Time bookkeeping split between two owners | **fixed** — the update helpers no longer touch `t` |
| 2.16 | Hygiene | **fixed** — `integrateQ`, `is_multiple_return_values`, duplicate `verbosePrint`, duplicate `IntegrationSchemes`, the `torch.jit.script` enum decorator, unreachable returns and the absolute import are gone; `StateBlend` validates; `t` stays a `float` |
| 3 | README / `pyproject.toml` out of sync | **fixed** for everything except the package rename |
| 4.1 | State machinery torch-only, silently aliases everything else | **fixed** — registry-based `clone_value` / `empty_value` / `move_value`; handlers for torch tensors, `wp.array` (duck-typed, no warp import), and list/tuple/dict recursion; unknown types warn once instead of aliasing silently |

**Headline numbers after the fixes** (`python scripts/step_reuse_convergence.py --all`):

- Every registered scheme reaches its claimed order on an autonomous, a
  non-autonomous, and a nonlinear problem. Before, six schemes collapsed to order 1
  on anything time-dependent.
- Dormand–Prince 5(4) under reuse is **bit-for-bit identical** to Dormand–Prince
  without reuse, at one fewer right-hand-side evaluation per step. Same for
  Bogacki–Shampine 3(2). That is what §2.1a was for.
- Every reuse prediction now matches measurement for every scheme, on both problems.

### Still open

- **P2 (Warp)** — §4.1 is done. §4.2's cheap half is done: the RK weight loop no longer
  clones per non-zero `b` (9 → 5 clones for RK4, the theoretical minimum). Still open:
  §4.2's substance — `update_component` still builds three fresh tensors per component
  per stage, with no buffer pool, no fused `axpy`, no graph capture — and §4.3, the
  torch-autograd vs `wp.Tape` decision, which should be made before the backend is
  written rather than retrofitted.
- **P3 (SPH features)** — adaptive `dt`, particle masking, neighbour-list reuse policy.
  Untouched, but §2.8 is finished, so the embedded-pair machinery adaptive `dt` needs
  is now in place.
- **P4** — the top-level package is still named `integrators`, which is
  collision-prone on PyPI. Renaming is a breaking change and was left alone.
- **A new finding, not in the original analysis:** Leap Frog, Velocity Verlet, PEFRL and
  VEFRL are only second/fourth order for a **separable** Hamiltonian, i.e. a force
  depending on position alone. With a velocity-dependent force — artificial viscosity,
  drag, any real SPH momentum equation — all four drop to first order. This is a
  property of the schemes rather than a defect here, but it is a sharp edge for SPH
  specifically. It is measured by `tests/test_convergence.py` on the new `damped`
  problem and documented in the README scheme table. Symplectic Euler is not affected.

---

## 1. What this library actually is

Despite the repository name, **there is not one line of NVIDIA Warp in it**
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
| **Field metadata** | [fields.py](src/integrators/fields.py) | `dataclasses.field` wrappers that stamp `metadata['behavior']` (`integrated` / `constant` / `copied` / `ephemeral` / `custom`) and `metadata['tags']` onto state fields. Drives generic cloning and tag-based lookup. |
| **Update specs** | [specs.py](src/integrators/specs.py) | Frozen dataclasses describing *what* an update does, without saying to which field: `ComponentUpdateSpec(derivative_dt, blend)`, `PositionUpdateSpec(+current_velocity_dt, +update_velocity_dt)`, `StateBlend(self_scale, reference_state, reference_weight)`. |
| **Dispatch** | [util.py](src/integrators/util.py), [protocol.py](src/integrators/protocol.py) | `applyPositionUpdate` / `applyVelocityUpdate` / `applyQuantityUpdate` / `applyStateUpdate` try the typed `apply_*_update` methods, else fall back to legacy `integratePosition` / `integrateVelocity` / … . Lifecycle hooks `initialize` / `preprocess` / `postprocess` / `finalize`. |
| **Schemes** | [butcher.py](src/integrators/butcher.py), [verlet.py](src/integrators/verlet.py), [ruth.py](src/integrators/ruth.py), [tvd.py](src/integrators/tvd.py), [euler.py](src/integrators/euler.py) | 23 registered schemes, all built from the four `apply*` primitives. Registry + lookup in [integration.py](src/integrators/integration.py). |

### 1.2 The generic update kernel

Everything bottoms out in two functions at [fields.py:267-310](src/integrators/fields.py#L267-L310):

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

`priorStep` (a `StageResult` from the previous step) can be passed in to skip the
`k0` evaluation. **This is the single most broken feature in the library — see §2.1.**

---

## 2. Confirmed defects

### 2.1 🔴 CRITICAL — `priorStep` reuse is applied to schemes where it is not valid, with no way to tell **[verified]**

`priorStep` feeds the *last stage* of step *n* in as the *first stage* `k0` of
step *n+1*, halving RHS evaluations. **The feature itself is legitimate and worth
keeping** — it is standard practice in the SPH literature (CRKSPH does exactly
this for its second-order scheme), and the measurements below confirm it is free
for the midpoint/Heun family. The defect is that its validity is a property of
the *tableau*, and the library currently offers it as an unconditional caller
option with no predicate, no diagnostic, and no FSAL tableau to use it with.

**How much order survives is computable from the tableau.** Reuse substitutes
`k0` with a derivative evaluated at `(tⁿ + c_s·dt, Y_s)` instead of
`(tⁿ⁺¹, yⁿ⁺¹)`. Two things set the damage:

- If `c_s = 1`, the times agree and `Y_s` differs from `yⁿ⁺¹` by `O(dtq⁺¹)`, where `q` is the order of the method whose weights are the last row of `a`. If `c_s ≠ 1` the times disagree at `O(dt)` and `q = 0`. When additionally `a[-1] == b`, the stage *is* `yⁿ⁺¹`: the tableau is **FSAL** and reuse is exact.
- That `O(dtq⁺¹)` perturbation of `k0` reaches the update directly if `b[0] ≠ 0` (costing one power of `dt`), or only through the stage equations if `b[0] = 0` (costing two).

Retained order = `min(p, q + 1)`, or `min(p, q + 2)` when `b[0] = 0`. Measured
against that prediction — harmonic oscillator (`k=4, m=1`), `T=2`,
`dt ∈ {0.1, …, 0.0125}`, float64, via
[scripts/step_reuse_convergence.py](scripts/step_reuse_convergence.py) `--all`:

| Scheme | order | no reuse | reuse | predicted | verdict |
|---|---|---|---|---|---|
| Midpoint / EPEC | 2 | 2.01 | 2.00 | 2 | **safe** (`b[0]=0`) |
| Heun 2nd / EPEC Modified | 2 | 2.01 | 2.00 | 2 | **safe** (`c_s=1`, `a[-1]` is order 1) |
| Ralston 2nd | 2 | 2.01 | 1.04 | 1 | degraded |
| RK3 | 3 | 2.99 | 2.02 | 2 | degraded |
| Heun 3rd / Ralston 3rd / Wray 3rd / SSP-RK3 | 3 | 2.99 | 0.98–1.00 | 1 | degraded |
| RK4 | 4 | 4.01 | 2.96 | 3 | degraded |
| RK4 (alt) | 4 | 4.01 | 2.01 | 2 | degraded |
| Nyström 5th | 5 | 4.99 | 0.99 | 1 | degraded |
| Leap Frog | 2 | 2.00 | 1.01 | 1 | degraded |
| Symplectic Euler | 2 | 2.00 | 1.01 | 1 | degraded |
| Forward Euler | 1 | 1.03 | **0.00** | 1 | degraded **+ unstable** |
| Velocity Verlet | 2 | 2.00 | **TypeError** | — | broken (§2.4) |
| PEFRL / VEFRL / TVD-RK2/3 / Euler | — | ok | unaffected | — | reuse silently ignored |

The prediction matches every measurement. The one exception is instructive:
Forward Euler is predicted order 1 but measures **0.00**, because reuse turns a
one-stage tableau into the lagged two-step method `yⁿ⁺¹ = yⁿ + dt·f(yⁿ⁻¹)`, which
is formally consistent but has no stability region on the imaginary axis. So the
predicate bounds *consistency*; reuse also changes *stability*, and a one-stage
scheme must simply refuse it.

Note that "safe" still costs a constant factor: midpoint's error is a uniform
**2.5× larger** under reuse at every step size. That is the real CRKSPH trade —
half the RHS evaluations for a fixed error constant, order intact — and it is a
perfectly good deal. What is not a good deal is SSP-RK3 on a time-dependent
problem, where reuse costs two orders and the error at `dt=0.0125` is
**22 000× larger**.

This matters today because [integrators.ipynb](integrators.ipynb) cell 5 drives
every plot in the README with `priorStep = result.stages[-1]` unconditionally, so
**the published convergence images in `images/` show degraded schemes** for
everything except the midpoint/Heun family.

**Fix — keep the feature, make it self-diagnosing:**
1. Promote `reuse_analysis()` from the script into the library as
   `integrators.step_reuse_order(scheme) -> int | None` plus
   `integrators.supports_step_reuse(scheme) -> bool`. It is ~40 lines and needs
   only the tableau. Callers can then ask before they opt in.
2. Add `reuse_order` / `fsal` to `IntegrationScheme` so the registry carries the
   answer, and record it by hand for the non-tableau schemes (Verlet family)
   which currently have no way to express it.
3. Warn once (not per step) when `priorStep` is supplied to a scheme whose
   `step_reuse_order < order`, naming the order that will actually be achieved.
   Do **not** hard-error — accepting a known order loss for half the RHS cost is
   a legitimate choice the caller is entitled to make.
4. Make the four schemes that silently ignore `priorStep` say so (§2.9).
5. Add genuinely FSAL tableaus so reuse is available at no cost at all — see §2.1a.

### 2.1a Add FSAL tableaus

No registered tableau satisfies `c_s = 1 ∧ a[-1] = b`, so lossless reuse is
currently impossible at any order above 2. Worth adding, in rough priority:

| Tableau | Order | Stages | Why |
|---|---|---|---|
| **Bogacki–Shampine 3(2)** | 3(2) | 4 (3 effective under FSAL) | Cheapest useful FSAL pair; the embedded 2nd-order estimate also feeds adaptive `dt` (§2.8, §4.4). `scipy`'s `RK23`. |
| **Dormand–Prince 5(4)** | 5(4) | 7 (6 effective) | The workhorse FSAL pair; `scipy`'s `RK45`, MATLAB's `ode45`. Direct replacement for the Nyström-5 slot, which loses 4 orders under reuse. |
| **Cash–Karp 5(4)** | 5(4) | 6 | Not FSAL, but a well-conditioned embedded pair for step control if DP5 is overkill. |
| **SSP-RK3 with FSAL variant** | 3 | 4 | If the TVD property is needed *and* reuse is wanted; plain SSP-RK3 loses 2 orders under reuse. |

These are all explicit tableaus that drop straight into `getButcherTableau`, and
`reuse_analysis` will confirm the FSAL property automatically. Adding them also
completes the embedded-pair path (§2.8) and unblocks adaptive stepping (§4.4),
so it is the highest-leverage single change in this document.

### 2.2 🔴 Wrong stage times → order collapse for any time-dependent RHS **[verified]**

Several schemes evaluate `f` at the correct *state* but the wrong *time*. Harmless
for autonomous problems, fatal for SPH with time-varying inlets, moving boundaries,
prescribed body forces, or ramped viscosity. Measured with `x'' = cos(t)`:

| Scheme | stage times passed to `f` (t₀=1.0, dt=0.1) | should be | order (t-dep) |
|---|---|---|---|
| Symplectic Euler | `1.0, 1.0` | `1.0, 1.05` | 2 → **1** |
| Leap Frog | `1.0, 1.05` | `1.0, 1.1` (state is the *full-step* position) | 2 → **1** |
| Velocity Verlet | `1.0, 1.05` | `1.0, 1.1` (ditto) | 2 → **1** |
| TVD-RK2 | `1.0, 1.0` | `1.0, 1.1` | 2 → **1** |
| TVD-RK3 | `1.0, 1.0, 1.0333` | `1.0, 1.1, 1.05` | 3 → **1** |
| VEFRL | `1.0, 1.0, 1.0, 1.0, 1.0` | staggered | 4 → **1** |
| PEFRL | staggered correctly | — | 4 ✓ |

Specifics:
- [verlet.py:96](src/integrators/verlet.py#L96) — `symplecticEuler` never sets `halfState.t`. Add `halfState.t = state.t + dt/2`.
- [verlet.py:43](src/integrators/verlet.py#L43) — `leapFrog` sets `halfState.t = state.t + 0.5*dt`, but `halfState` already holds the **full-step** position `xⁿ + dt·v + ½dt²·a`. It is `yⁿ⁺¹`, not a half state. Should be `state.t + dt`; the variable name is also misleading.
- [verlet.py:154](src/integrators/verlet.py#L154) — same in `velocityVerlet`: the position drift is a full `dt`, so the evaluation point is `t + dt`.
- [tvd.py:41](src/integrators/tvd.py#L41) — `y_2_3.t = state.t + dt/3` is doubly wrong. In Shu–Osher SSP-RK3, `y⁽¹⁾` sits at `t+dt` and `y⁽²⁾` at `t+dt/2`. And `y_1_3.t` is **never set at all** (`updateStateEuler` does not advance `t`), so stage 2 is evaluated at `tⁿ`.
- [tvd.py:75](src/integrators/tvd.py#L75) — `TVDRK2`: `state1.t` never set.
- [ruth.py:75-132](src/integrators/ruth.py#L75-L132) — `VEFRL` sets no stage times at all; only `finalState.t` at the end.

### 2.3 🔴 Stage 0 is evaluated at the wrong time whenever the user's `initializeNewState` drops `t` **[verified]**

`RungeKuttaB` sets `currentState.t` for stages 1..s-1 ([butcher.py:62](src/integrators/butcher.py#L62)) but **never for stage 0**. It inherits whatever `initializeNewState()` returned. The README's own example ([README.md:85-91](README.md#L85-L91)) and the notebook ([integrators.ipynb](integrators.ipynb) cell 4) both construct the new system *without* forwarding `t`:

```python
return HarmonicOscillatorSystem(state=state.initializeNewState())   # t defaults to 0.0
```

Observed RK4 stage times over three steps with that exact pattern:

```
step 0: [0.0,  0.05, 0.05, 0.1]   ← correct
step 1: [0.0,  0.15, 0.15, 0.2]   ← stage 0 should be 0.1
step 2: [0.0,  0.25, 0.25, 0.3]   ← stage 0 should be 0.2
```

Stage 0 is pinned to `t=0` forever. **Fix:** set `currentState.t = initialState.t`
explicitly before the first `updateStep` in every scheme, and fix the README/notebook
examples to forward `t`. Better: stop trusting the user to carry `t` — have the
integrator own it end to end (the protocol docstring at [protocol.py:202](src/integrators/protocol.py#L202) already claims it does).

### 2.4 🟠 `velocityVerlet` crashes on `priorStep`, and would swap `k`/`r` if it didn't **[verified]**

[verlet.py:149](src/integrators/verlet.py#L149):

```python
k0, r0 = updateStep(...) if priorStep is None else priorStep
```

`StageResult` is `(aux, update)`, so unpacking binds `k0 = aux`, `r0 = update` — the
reverse of `leapFrog`/`symplecticEuler`, which carefully do `k0, r0 = priorStep.update, priorStep.aux`.
Result: `TypeError: must be called with a dataclass type or instance` downstream.
The same three-branch `StageResult` / `Tuple` / raise block is copy-pasted verbatim
in `leapFrog` and `symplecticEuler`; **extract it into one `_unpack_prior_step()` helper** so this class of drift cannot recur.

### 2.5 🟠 `BaseState.nograd()` raises on any non-tensor field **[verified]**

[fields.py:187](src/integrators/fields.py#L187):

```python
def _op(value, detach=False):
    return value.detach().clone() if detach else value.clone() if isinstance(value, torch.Tensor) else value
```

Python parses this as `A if detach else (B if isinstance(...) else C)` — the
`isinstance` guard only protects the **non-detach** branch. With `detach=True`,
`value.detach()` is called on floats, ints, strings, `None`:

```
AttributeError: 'float' object has no attribute 'detach'
```

Every realistic SPH state carries scalar parameters, so `nograd()` is effectively
unusable. Fix:

```python
def _op(value, detach=False):
    if not isinstance(value, torch.Tensor):
        return value
    return value.detach().clone() if detach else value.clone()
```

### 2.6 🟠 `clone()` and `initializeNewState()` disagree on untagged fields **[verified]**

Two different defaults for the same missing metadata:

- [fields.py:198](src/integrators/fields.py#L198) `_clone_state` → `metadata.get(BEHAVIOR_KEY, 'constant')`
- [fields.py:210](src/integrators/fields.py#L210) `_state_initialize` → `metadata.get(BEHAVIOR_KEY, 'copied')`

A plain `x: torch.Tensor = None` field therefore survives `clone()` but is silently
**nulled** by `initializeNewState()`:

```
clone()              -> tensor([0., 1., 2.])
initializeNewState() -> None
```

Forgetting one `constant(...)` decorator turns a tensor into `None` mid-step, and the
failure surfaces far from its cause. Pick one default (`'constant'` is the safe one)
and share a single resolver.

### 2.7 🟠 `copied` fields are documented but never actually copied **[verified]**

`copied()` promises "copied from last substep in finalize", and `_state_finalize`
([fields.py:219](src/integrators/fields.py#L219)) implements exactly that — but
**`_state_finalize` is never called anywhere in the package**. `copied` behaves
identically to `ephemeral` (nulled in `initializeNewState`, never restored).
Either wire it into the `finalize` path or delete the behavior and its docs.

### 2.8 🟠 Embedded-tableau path is broken and currently unused **[verified]**

[butcher.py:85-103](src/integrators/butcher.py#L85-L103) supports `b` as a tuple of
weight vectors (the embedded/error-estimator form), but:

1. `finalizeSystem(new_state, initialState, dt, rs, ks, b_, ...)` at line 101 uses `b_` **leaked from the loop at line 87**, so every state is finalized with the *last* weight vector regardless of which `b_` produced it.
2. Only `new_states[-1]` is returned; the lower-order estimate is computed and thrown away, so no error estimate ever reaches the caller — the whole point of an embedded pair.
3. No embedded scheme is registered, so this is untested dead code.

Worth finishing rather than deleting: fix the `b_` leak, return both estimates
(add an `error` field to `IntegrationResult`), and register the pairs from §2.1a.
The FSAL tableaus that make step reuse lossless *are* embedded pairs, so this
work and §2.1a are the same change, and together they unblock adaptive `dt` (§4.4).

### 2.9 🟡 Schemes that ignore `priorStep` leak it into the user's RHS **[verified]**

`TVDRK3`, `TVDRK2`, `PEFRL`, `VEFRL` never `kwargs.pop('priorStep')`, so it flows
straight through into `f(state, dt, **kwargs)` and into `preprocess`/`postprocess`:

```
[ok] TVDRK3 priorStep leak: kwargs reaching RHS: ['priorStep']
[ok] PEFRL  priorStep leak: kwargs reaching RHS: ['priorStep']
```

Any RHS with a strict signature raises `TypeError` for these four schemes only.

### 2.10 🟡 `TVDRK3` calls `finalizeSystem` with the wrong arity **[verified]**

[tvd.py:59](src/integrators/tvd.py#L59): `finalizeSystem(finalState, state, dt, rs, ks, *args, **kwargs)` — the `weights` positional is missing, so `*args` slides into that slot. With empty `*args` the user's `finalize` sees `weights=()`; with any positional args it sees garbage. Every other scheme passes an explicit list. Add `[1/3, 2/3]` or at minimum `[]`.

### 2.11 🟡 Inconsistent cloning strategy: `initializeNewState` vs `copy.deepcopy`

`butcher`/`verlet`/`euler` use `state.initializeNewState()` (metadata-driven, cheap-ish).
`tvd`/`ruth` use `copy.deepcopy(state)` ([tvd.py:32](src/integrators/tvd.py#L32), [tvd.py:47](src/integrators/tvd.py#L47), [tvd.py:81](src/integrators/tvd.py#L81), [ruth.py:28](src/integrators/ruth.py#L28) and 5 more). `deepcopy` copies *everything* including `constant` fields and ephemeral scratch, ignores the behavior metadata entirely, and will be catastrophic (or will silently fail) for warp arrays, CUDA graphs, and captured neighbour lists. Standardise on `initializeNewState()`.

### 2.12 🟡 `TVDRK3` / `TVDRK2` / `VEFRL` evaluate `f` on the **caller's** state object

[tvd.py:24-25](src/integrators/tvd.py#L24-L25), [ruth.py:85-86](src/integrators/ruth.py#L85-L86) pass the incoming `state` directly to `preprocessSystem` and `f`, rather than a fresh `initializeNewState()` buffer as the Butcher/Verlet paths do. Since SPH `preprocess` typically writes neighbour lists and scratch buffers into the state, **the caller's state gets mutated**. Also, these four schemes never call `initializeSystem`, so the `initialize` lifecycle hook is skipped entirely for them.

### 2.13 🟡 Registry metadata is wrong or arbitrary

- [integration.py:64](src/integrators/integration.py#L64) — Semi-Implicit Euler registered as **order 2**; measured **1.05**. It is a first-order method.
- `dissipation` / `nonLagrangian` flags look copy-pasted: `symplecticEuler` is `(True, True)` while `leapFrog` and `velocityVerlet` are `(False, False)`, and `PEFRL` is `(False, False)` while `VEFRL` is `(True, True)`. Nothing in the code reads these flags. Either justify them or drop them.
- `RungeKutta2`, `midPoint`, and `EPEC` are the same tableau under three names; `heunsMethod` and `EPECmodified` likewise. Harmless but it inflates the "23 schemes" count to ~19 distinct ones and makes `getPreferredScheme` ambiguous.
- [integration.py:83](src/integrators/integration.py#L83) — `getIntegrator` compares `scheme.identifier == integrator` (enum vs string, never true), while `getIntegrationEnum` correctly uses `.identifier.name`. So `getIntegrator('RK4')` ✓, `getIntegrator(IntegrationSchemeType.rungeKutta4)` ✓, but `getIntegrator('rungeKutta4')` ✗ `ValueError`.

### 2.14 🟡 `symplecticEuler` has a leaked diffSPH-ism

[verlet.py:119-123](src/integrators/verlet.py#L119-L123):

```python
if hasattr(finalState, 'integrateDensity'):
    finalState.integrateDensity(k1, dt, **kwargs)
    applyQuantityUpdate(finalState, k1, explicit_step(dt), densitySwitch=True, **kwargs)
```

Density is updated **twice** (once via the bespoke method, once via the generic
path), and `densitySwitch=True` is injected into the user's `apply_quantity_update`
signature for this one scheme only. This is a solver-specific hook that has no
business in a generic integrator; it should be expressed through the field-behavior
metadata instead.

### 2.15 🟡 Time bookkeeping is split between two owners

`protocol.py` states emphatically that "Time management is exclusively handled by
the integrator", yet [util.py:37](src/integrators/util.py#L37)
`updateStateSemiImplicitEuler` does `systemState.t = systemState.t + dt` while its
sibling `updateStateEuler` does not. `integrateSemiImplicitEuler` then overwrites
`newState.t` anyway, masking it — but both helpers are publicly exported, so a user
composing them gets double-advanced time. Pick one owner (the integrator) and make
the helpers time-agnostic.

### 2.16 🟢 Minor / hygiene

- **No tests. None.** `find . -name 'test*'` → empty. For a numerics library where every defect above is a silent accuracy loss, this is the root cause of §2.1–§2.3.
- [util.py:5](src/integrators/util.py#L5) — `from integrators.fields import get_tagged_attr` is an **absolute** import inside a relative-import package; it breaks under vendoring or rename. Should be `from .fields import ...`. (It is also unused in that module.)
- `verbosePrint` defined twice, identically ([util.py:18](src/integrators/util.py#L18), [fields.py:251](src/integrators/fields.py#L251)).
- `IntegrationSchemes = []` defined in both [util.py:138](src/integrators/util.py#L138) (dead) and [integration.py:39](src/integrators/integration.py#L39) (real). `__init__` imports both names; ordering saves it.
- `integrateQ` ([util.py:140](src/integrators/util.py#L140)) is 55 lines of deprecated dead code with zero callers. `is_multiple_return_values` ([util.py:197](src/integrators/util.py#L197)) has zero callers and calls `func` for its side effects. Delete both.
- `initializeSystem` / `preprocessSystem` / `postprocessSystem` / `finalizeSystem` each have an unreachable `return` after their `with record_function(...)` block.
- `@torch.jit.script` on the `IntegrationSchemeType` Enum ([enums.py:5](src/integrators/enums.py#L5)) is a no-op at the Python level (`type(...)` is still `enum.EnumType`) but forces a torch import + TorchScript compilation for a module that is otherwise pure stdlib. Drop it.
- `StateBlend` allows `reference_state` set with `reference_weight=None`, which then raises deep inside `update_component` on `None * tensor`. Validate in `__post_init__`.
- `fields.py` imports `numpy` and re-imports `dataclass` at line 250 without using either.
- Stage times become `np.float64` (`currentState.t = initialState.t + c*dt` with `c` from a numpy array), so `system.t` silently changes dtype mid-run. Cast to `float`.

---

## 3. Documentation is out of sync with the code

The README will actively mislead anyone onboarding:

| README says | Reality |
|---|---|
| `pip install diffSPH_integrators` ([README.md:20](README.md#L20)) | package is `sphWarpIntegrators`, imports as `integrators` |
| `blend=StateBlend.IMPLICIT` ([README.md:295](README.md#L295), [README.md:301](README.md#L301)) | `StateBlend` has no such member; it is a 3-field dataclass. This example raises `AttributeError`. |
| `custom(apply_fn=custom_apply)` ([README.md:386](README.md#L386)) | `custom()` accepts no `apply_fn`; it goes into `**extra` metadata and is never read |
| `apply_my_behavior_update` is dispatched ([README.md:449](README.md#L449)) | no such dispatch exists — only the four fixed `apply_*_update` names |
| "Symplectic Euler | Order 1" ([README.md:216](README.md#L216)) | registry says order 2, measured 2.00 (autonomous) |
| "RK2 (Heun)… Forward Euler | Error O(h²)" | conflates local truncation error with global order; the table's `Error` column is inconsistent with the `Order` column |
| Example `initializeNewState` drops `t` ([README.md:85](README.md#L85)) | triggers the §2.3 stage-0 bug |
| Title: "Differentiable ODE Integration with **PyTorch**" | fine today, but contradicts the repo name and the stated Warp direction |

`pyproject.toml` is also stale: `description = "A Fully differentiable SPH Solver."`
(it is not a solver), `keywords = ["sph","radius","pytorch"]` (`radius` is a leftover
from the neighbour-search package), no `warp-lang` dependency, and
`package-data "*" = ["*.*"]` which is a blanket glob that will happily ship
`__pycache__` if it is present at build time.

The distribution name (`sphwarpintegrators`) and the import name (`integrators`)
differ, and **`integrators` is an extremely collision-prone top-level name** on
PyPI. Rename the package directory to `sph_warp_integrators` (or `warpintegrators`)
before this gets wider use.

---

## 4. What has to change to actually be Warp-based

This is the largest gap and it is architectural, not a list of bugs.

### 4.1 The state machinery is torch-only and silently aliases everything else **[verified]**

`_op` and `_empty_like` special-case `torch.Tensor` and pass every other type through
**by reference**:

```
warp array aliased after initializeNewState():  True
warp array aliased after clone():               True
list-of-tensors aliased after clone():          True
```

So a state holding `wp.array` fields is **not cloned** — every RK stage writes into
the same buffer as the initial state. Stage 2 would read stage-1-corrupted data and
the results would be wrong with no error raised anywhere. Same for any `list`/`dict`
of tensors.

**Fix:** replace the `isinstance(value, torch.Tensor)` checks with a small
registry-based `clone_value` / `empty_value` dispatch (torch tensor, `wp.array`,
list/tuple/dict recursion, `None`, plain scalars), so adding a backend is one
registration rather than an edit to two functions.

### 4.2 Allocation-per-stage fights Warp's execution model **[verified]**

Measured state clones per step:

| Scheme | `initializeNewState` calls | `copy.deepcopy` calls | theoretical minimum |
|---|---|---|---|
| RK4 | **9** | 0 | ~5 |
| Symplectic Euler | 3 | 0 | 3 |
| TVD-RK3 | 1 | 2 | 3 |
| PEFRL | 0 | **4** | 4 |

RK4's four surplus clones come from the final weight-accumulation loop
([butcher.py:79](src/integrators/butcher.py#L79)) calling `updateStateEuler` with the
default `copyState=True` — a full state clone per non-zero `b` — while the *stage*
loop at line 61 correctly passes `copyState=False`. Trivially fixable: pass
`copyState=False` there too and clone once up front.

More fundamentally, the whole design is **functional and allocating**:
`update_component` builds `value = value * s; value = value + w * ref; value = value + dt * delta`
— three fresh tensors per component per stage. That is idiomatic for
autograd-through-time, but it is the opposite of what Warp wants (preallocated
buffers, in-place kernel writes, CUDA-graph capture). At SPH scale (10⁶–10⁷
particles) the allocation traffic alone will dominate.

**Recommended direction:** keep the current spec/tag/protocol layer — it is
backend-agnostic and worth preserving — and add an out-of-place-vs-in-place switch
at the `update_component` level:
- a `WarpState` base whose `initializeNewState` pulls buffers from a per-step pool instead of allocating;
- an `axpy`-style Warp kernel (`x = s*x + w*ref + Σ dtᵢ·kᵢ`) with a variadic-`k` variant, so the whole accumulation is one launch;
- fixed stage counts per scheme so the pool is sized once and the step is graph-capturable.

### 4.3 No differentiability story for Warp

The README's headline claim is "fully differentiable". Warp's `wp.Tape` has a
completely different gradient model from torch autograd (explicit tape, adjoint
kernels). A `torch`+`warp` hybrid needs `wp.to_torch`/`wp.from_torch` at the
boundary, or `warp.autograd`. **Decide which one is authoritative before writing the
Warp backend** — retrofitting is much worse than choosing up front.

### 4.4 Nothing in the API is SPH-aware

For an SPH driver the following are missing and each interacts with the stage loop:
- **CFL / adaptive `dt`.** `dt` is a caller-supplied constant. Real SPH recomputes `dt` from `min(h/c_s, sqrt(h/|a|), ...)` every step. Needs the embedded-pair machinery (§2.8) finished, plus a step-rejection path.
- **Neighbour-list reuse policy.** `preprocess` is called once per stage. For SPH you want "rebuild the list on stage 0, reuse for stages 1..s" — there is no way to express that today.
- **Particle count changes** (inflow/outflow/refinement) mid-step. All clone paths assume fixed shapes.
- **`integrateSpecies` / fluid-vs-boundary masking.** The only masking support in the codebase is inside the deprecated `integrateQ` (`integrateSpecies`, `species` args) and the `fluid_only` flag on `integrated()` — which **is never read by anything**. Boundary particles must not be integrated; that has to be a first-class concept.
- **Density: continuity vs summation.** The `integrateDensity` hack (§2.14) is a symptom of this being unmodelled.

---

## 5. Prioritized plan

**P0 — correctness (the library is currently producing wrong answers)**
1. Write a convergence test suite first: one autonomous + one time-dependent + one Hamiltonian problem with analytic solutions, asserting measured order ≥ claimed order − 0.15 for every registered scheme, and asserting that the order measured *with* reuse matches `step_reuse_order(scheme)`. [scripts/step_reuse_convergence.py](scripts/step_reuse_convergence.py) already does the measurement and the prediction; turning it into `tests/` is mostly adding asserts. This pins down §2.1–§2.3 and §2.13 permanently.
2. Promote `reuse_analysis()` into the library as `step_reuse_order()` / `supports_step_reuse()`; carry the answer on `IntegrationScheme`; warn once when reuse is requested where it costs order (§2.1). Keep the feature enabled — do not gate it behind a hard error.
3. Fix stage times in `symplecticEuler`, `leapFrog`, `velocityVerlet`, `TVDRK2/3`, `VEFRL` (§2.2). Note these also change each scheme's reuse analysis, so do it before recording `reuse_order` by hand.
4. Set `currentState.t = initialState.t` for stage 0 everywhere; fix the README/notebook `initializeNewState` examples (§2.3).
5. Fix `_op` precedence (§2.5) and unify the untagged-field default (§2.6).
6. Fix `velocityVerlet`'s `priorStep` unpacking; extract the shared `_unpack_prior_step` helper (§2.4).
7. Regenerate `images/` — either with reuse off, or per-scheme with reuse only where `supports_step_reuse` is true, and say which in the caption.

**P0.5 — FSAL tableaus (unblocks reuse, embedded pairs, and adaptive dt at once)**
8. Add Bogacki–Shampine 3(2) and Dormand–Prince 5(4) to `getButcherTableau` and the registry (§2.1a). Verify the FSAL property falls out of `step_reuse_order` automatically.
9. Finish the embedded-pair path so the second `b` vector is returned as an error estimate rather than discarded (§2.8).

**P1 — consistency**
10. `pop('priorStep')` in all schemes and report it as unsupported rather than swallowing it (§2.9); `initializeSystem` in all schemes (§2.12).
11. Replace `copy.deepcopy` with `initializeNewState` in `tvd`/`ruth`; stop mutating the caller's state (§2.11, §2.12).
12. Fix `TVDRK3`'s `finalizeSystem` arity (§2.10); fix Semi-Implicit Euler's registered order and `getIntegrator`'s enum-name lookup (§2.13).
13. Either wire up `copied` or delete it (§2.7); remove the `integrateDensity` hack (§2.14).
14. Delete `integrateQ`, `is_multiple_return_values`, duplicate `verbosePrint`, duplicate `IntegrationSchemes`, the `torch.jit.script` enum decorator, unreachable returns, absolute import (§2.16).

**P2 — Warp**
15. Backend-dispatch `clone_value`/`empty_value` so `wp.array` and containers are actually copied (§4.1) — do this *before* any Warp state exists, or the aliasing bug will be blamed on the integrators.
16. `copyState=False` in the RK weight loop (§4.2, 9 → 5 clones for RK4).
17. Buffer pooling + fused `axpy` kernel + graph capture (§4.2). Step reuse (§2.1) compounds here: on an FSAL tableau it removes a whole kernel launch *and* a whole state buffer per step.
18. Decide the torch-autograd vs `wp.Tape` gradient story (§4.3).

**P3 — SPH features**
19. Adaptive `dt` driven by the embedded pair from P0.5; particle masking as a first-class concept; neighbour-list reuse policy (§4.4).

**P4 — packaging/docs**
20. Rewrite README (§3); fix `pyproject.toml` metadata + add `warp-lang`; rename the top-level package off `integrators`; add CI running the P0 test suite.

---

## Reproducing the results

### `scripts/step_reuse_convergence.py` (committed)

The step-reuse study of §2.1, as a reusable tool. It predicts the retained order
from the tableau and measures it, so the two can be compared:

```bash
conda activate warp

# one scheme in detail: error table, both orders, prediction vs measurement
python scripts/step_reuse_convergence.py --scheme RK4
python scripts/step_reuse_convergence.py --scheme 'SSP RK3' --problem forced

# every registered scheme, one line each (this produced the §2.1 table)
python scripts/step_reuse_convergence.py --all

# log-log convergence plot with reference slopes
python scripts/step_reuse_convergence.py --scheme RK4 --plot rk4_reuse.png
```

`--problem oscillator` is autonomous, `--problem forced` (`x'' = cos t`) is not —
the gap between them is what exposes the stage-time bugs of §2.2.

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

The ad-hoc probes that produced the original measurements have been rewritten as a
pytest suite. `pytest` from the repository root, inside the `warp` environment; ~100 s
for 680 tests.

| file | what it pins down |
|---|---|
| `test_convergence.py` | every scheme reaches its registered order on `oscillator` (autonomous), `forced` (time-dependent) and `kepler` (nonlinear); errors decrease monotonically; the four splitting schemes lose order on `damped` and nothing else does (§2.2, §2.3) |
| `test_step_reuse.py` | measured reuse order matches `scheme.reuse_order` exactly — neither optimistic nor pessimistic; FSAL reuse is bit-for-bit free; Forward Euler refuses; the four non-reusing schemes warn, ignore, and do not leak `priorStep` into the RHS; the warning fires once and names the achieved order (§2.1, §2.9) |
| `test_state.py` | `nograd()` on non-tensor fields; untagged-field agreement between the two clone paths; stage times translate by `dt` each step; `t` stays a Python `float`; the update helpers do not advance time; `getIntegrator` accepts all three spellings; schemes do not mutate the caller (§2.3, §2.5, §2.6, §2.12, §2.13, §2.15) |
| `test_embedded.py` | the error estimate reaches the caller, scales as `dt^p`, brackets the true error, and the *high-order* branch is the one propagated; tableau row sums and the FSAL property (§2.8) |
| `test_hamiltonian.py` | the `dissipation` flag predicts energy behaviour: symplectic schemes stay inside an `O(dt^p)` band over an 8× longer run, dissipative ones grow ~linearly (§2.13) |
| `test_copied_fields.py` | `copied()` fields arrive holding the *last stage's* value, `ephemeral()` ones do not survive at all (§2.7) |
| `test_backend_dispatch.py` | `wp.array`, lists, dicts and nested containers are cloned rather than aliased; `to(device)` reaches inside them; unknown types warn once; and no scheme hands the caller's own container back (§4.1) |
| `test_kwargs_passthrough.py` | caller kwargs survive every scheme — `verbose` in both states, alongside `priorStep`, and an unknown kwarg reaching the RHS intact |

The shared harness — three tagged-field reference systems, four problems with analytic
solutions, and the order/energy measurement helpers — lives in
[src/integrators/testing.py](src/integrators/testing.py) so that the tests and
`scripts/step_reuse_convergence.py` share one definition. It doubles as the smallest
complete worked example of the protocol.
