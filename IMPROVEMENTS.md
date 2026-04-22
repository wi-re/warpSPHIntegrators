# Integrators Library — Issues & Improvement Plan

## Completed Refactorings

### ✅ Return Type Refactoring: Raw tuples → Named tuple `IntegrationResult`

**What changed:**
- All integrators now return `IntegrationResult(state=..., stages=[...])` instead of raw tuples `(state, rs, ks)`
- Each stage is represented as a `StageResult(aux=..., update=...)` named tuple instead of parallel lists
- This eliminates tuple indexing guessing and provides self-documenting code

**Benefits:**
- No need to remember tuple positions: `result.state` instead of unpacking/indexing
- Accessing last stage is natural: `result.stages[-1]` returns the `StageResult` with both `aux` and `update` paired
- Used as `priorStep` pattern becomes obvious: `result.stages[-1]` captures both the auxiliary and update values together

**Files updated:**
- `specs.py` — Added `StageResult` named tuple and refactored `IntegrationResult`
- `integration.py` — Updated exports
- All integrators: `butcher.py`, `verlet.py`, `tvd.py`, `ruth.py`, `euler.py`
- `integrators.ipynb` — Updated `runIntegrator` to use new format

**Notebook usage example:**
```python
result = integrator.function(state, dt=dt, f=f_func, config={})
nextState = result.state
last_stage_aux = result.stages[-1].aux  # Easy access to last auxiliary value
last_stage_update = result.stages[-1].update  # Easy access to last k-value
```

---

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done

---

## High Severity

### H1 — ✅ Inconsistent integrator return arity
**File:** `butcher.py`, `verlet.py`, `tvd.py`, `ruth.py`, `euler.py`

All integrator functions conditionally return `(state, rs, ks)` or `(state, ks)` based on whether any aux values are non-None:
```python
if any([t is not None for t in rs]):
    return finalState, rs, ks
return finalState, ks
```
Callers cannot safely unpack the result without inspecting it first. The notebook works around this by indexing `values[-1][0]` which is fragile.

**Fix:** Always return all three values `(state, rs, ks)`. Callers that don't need aux can ignore them. Requires updating all call sites in the notebook (cell 6: `nextSystem, values, updates = integrator.function(...)`).

- [x] `butcher.py` — `RungeKuttaB`
- [x] `verlet.py` — `leapFrog`, `symplecticEuler`, `velocityVerlet`
- [x] `tvd.py`
- [x] `ruth.py`
- [x] `euler.py`
- [ ] Update notebook unpack in `runIntegrator`

---

### H2 — `leapFrog` calls `finalizeSystem` with wrong positional arguments
**File:** `verlet.py`, function `leapFrog`

`leapFrog` calls:
```python
finalizeSystem(finalState, dt, rs, ks, [0.5, 0.5], *args, **kwargs)
```
But `symplecticEuler` and `RungeKuttaB` call:
```python
finalizeSystem(finalState, initialState, dt, rs, ks, weights, *args, **kwargs)
```
`leapFrog` is passing `dt` (a float) where `initialState` is expected. If `finalize` ever inspects `initialState`, it will silently receive the wrong value.

**Fix:** Add `initialState` as the second argument in the `leapFrog` call.

- [x] `verlet.py` — `leapFrog`

---

### H3 — `ks[i]` undefined in tuple-`b` Butcher branch
**File:** `butcher.py`, `RungeKuttaB`

In the `isinstance(butcherTableau.b, tuple)` branch (used for EPEC-like dual-output schemes), this line appears before the `enumerate` loop that defines `i`:
```python
new_state = updateStateEuler(new_state, ks[i], b * dt, **kwargs)  # i not yet defined
```
This will raise `NameError` at runtime for any scheme with a tuple `b`. This code path is currently unreachable due to none of the registered schemes using tuple `b`, but is still broken.

**Fix:** Remove the stray pre-loop `updateStateEuler` line. The loop below already handles all stages correctly.

- [x] `butcher.py` — `RungeKuttaB` tuple-`b` branch

---

## Medium Severity

### M1 — `IntegrationScheme.__call__` silently drops `*args, **kwargs`
**File:** `util.py`, `IntegrationScheme`

```python
def __call__(self, state, dt, f):
    return self.function(state, dt, f)
```
All scheme functions accept `*args, **kwargs` (for `verbose`, `config`, `priorStep`, etc.). This shorthand is unusable for any real call — the notebook already bypasses it with `.function(...)` directly.

**Fix:** Forward `*args, **kwargs` in `__call__`.

- [x] `util.py` — `IntegrationScheme.__call__`

---

### M2 — Double `t` increment risk with typed protocol
**File:** `butcher.py`, `verlet.py`; user systems

Integrators set `new_state.t = initialState.t + dt` after calling `applyStateUpdate`. If the user's `apply_state_update` also increments `t` (as `HarmonicOscillatorSystem` does), `t` is advanced correctly only because the integrator overwrites it afterwards. This is an implicit ordering dependency — any system that reads `t` inside `apply_state_update` will see a stale or wrong value.

**Fix:** Document that typed `apply_*_update` methods must **not** advance `t`; `t` is always managed by the integrator. Alternatively, remove the `t` update from `HarmonicOscillatorSystem.apply_state_update` in the notebook.

- [ ] Document in `protocol.py` docstrings
- [ ] Remove `t` update from notebook `apply_state_update`

---

## Low Severity

### L1 — `integrateQ` is dead code in `util.py`
**File:** `util.py`

`integrateQ` (with `integrateSpecies`, `species`, `verletValue`, etc.) is no longer called by any integrator — the typed `apply_*_update` path is used instead. It remains exported and pollutes the public API surface.

**Fix:** Deprecate or remove. If SPH-species-masked integration is still needed, move it to a dedicated utility module.

- [ ] `util.py` — mark `integrateQ` deprecated or remove

---

### L2 — `PositionUpdateSpec` doesn't validate `derivative_dt` when `current_velocity_dt` is set
**File:** `specs.py`, `PositionUpdateSpec.__post_init__`

`semi_implicit_position_step` correctly hardcodes `derivative_dt=0.0`, but callers constructing `PositionUpdateSpec` manually can pass a non-zero `derivative_dt` alongside `current_velocity_dt`, which would silently apply both a derivative step and a velocity drift.

**Fix:** Add a `__post_init__` check: if `current_velocity_dt is not None`, assert `derivative_dt == 0.0`.

- [ ] `specs.py` — `PositionUpdateSpec.__post_init__`

---

### L3 — `config={}` threaded through all call sites but never consumed
**File:** All integrators, notebook

`config` is passed as a kwarg through every `integrator.function(...)` call and forwarded via `**kwargs` at every level, but nothing in the library consumes it. It is a legacy artifact.

**Fix:** Remove from the notebook call site. Consider accepting but ignoring it with a deprecation warning, or formally documenting it as a user-extension hook.

- [ ] Notebook — remove `config={}` from `runIntegrator`
- [ ] Decide: deprecate or document as extension hook

---

## Corrections to Prior Analysis

- **Item 7 (BaseState.initializeNewState missing):** Not an issue. `BaseState` implements `initializeNewState` generically via `_state_initialize`, driven by field behavior metadata. `HarmonicOscillatorState` inherits this correctly.
