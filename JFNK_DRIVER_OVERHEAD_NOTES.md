# JFNK/DIRK driver overhead notes (2026-08-24)

## Context

Follow-on from `warpSPHCore`'s `warpier_jvp_dual_argument_pruning_plan.md`, which closed out its
own investigation into `warpSPHCore`'s forward-mode-AD dispatch overhead (a real ~3.7x per-matched-
launch-pair cost, about half architecturally inherent, half CPU-side overhead — see that doc) after
finding it explains only about half of the real wave-equation benchmark's ~1.9-2x
`sdirk2_jfnk_jvp_1e-6` vs `sdirk2_jfnk_fd_1e-6` `msPerRhs` gap
(`warpSPH/benchmarks/wave/bench_performance.py`). This doc picks up the other half: what's actually
happening inside this repo's own DIRK (`dirk.py`) / JFNK (`jfnk.py`) driver code during a real step.

Not a phased implementation plan yet — findings and a prioritized punch list, pending direction on
which of these (if any) are worth acting on. GMRES/Newton numerics are correctness-sensitive; none
of this has been changed.

## Measurement

`torch.profiler` around 10 timed real steps of each scheme (`nx=128`, `warp` conda env, same case
`buildWaveCase` builds for the benchmark), driving `integrator.function(state=state, f=ctx.
stepFunction, dt=dt, config=ctx.config, schemeConfig=ctx.schemeConfig, solver=solver)` directly
(bypassing `runScheme`'s own timing/counting wrapper, so absolute call counts differ slightly from
`bench_performance.py`'s own numbers — read the *ratios* below, not the absolute counts).

Self-CPU-time-total, 10 steps: **jvp 2.162s, fd 1.208s** (ratio 1.79x, same order as the official
benchmark's `msPerRhs` ratio). `StateAwareWarpFunction` (the actual SPH kernel launches, all of it
`warpSPHCore`'s side): **56.66ms/step (jvp) vs 13.73ms/step (fd)** — a 4.13x difference that alone
accounts for **~45%** of the total per-step gap (about 43ms out of ~95ms). The **other ~55%** (about
52ms/step) is everything outside `StateAwareWarpFunction` — this repo's own code.

## What's in that other ~55%

Both schemes run through the *same* `gmres()`/`DIRK()` code paths — `JFNKSolver.solve` only swaps
which `matvec` closure GMRES calls (`fd_matvec` vs `jvp_matvec`, `jfnk.py`), so any cost here that's
identical between the two schemes is a genuine total-benchmark cost, not specifically a jvp
penalty — but a few pieces *are* asymmetric and worth flagging separately.

### 1. `gmres()`'s Arnoldi loop is O(k) tiny, serial, syncing tensor ops per Krylov iteration (`jfnk.py:200-238`)

Every Krylov iteration inside a restart cycle does, in a plain Python `for` loop:
- `H[i, k] = torch.dot(w, V[i])` for `i` in `range(k+1)` — a fresh, tiny `torch.dot` kernel launch
  each, no batching across `i`
- `w = w - H[i, k] * V[i]` — same, per `i`
- Givens rotation bookkeeping: `float(denom)`, `float(H[k+1,k])`, `float(g[k+1])` — each a
  **GPU→CPU sync** (`float()` on a 0-d CUDA tensor calls `.item()`)

Real-benchmark profile evidence: `aten::dot` (2276 calls / 10 steps in the fd-only profile — GMRES's
own op, present whether the matvec is fd or jvp), `cudaLaunchKernel` (52-61k calls),
`cudaMemcpyAsync` (41-44k calls) dominate self-CPU time and are **not** concentrated in
`StateAwareWarpFunction` at all. This is classic "many tiny ops, dispatcher overhead dominates"
territory — the Arnoldi process is inherently a sequence of rank-1 updates, so full batching isn't
free, but at minimum the O(k) `torch.dot`/`w -= ...` pairs *could* be replaced with two batched ops
(`H[:k+1, k] = V_stacked @ w`, `w -= V_stacked.T @ H[:k+1, k]`) — one matmul instead of `k` dots and
`k` axpys, which is the standard "modified vs. classical Gram-Schmidt as a matmul" trick. Would need
care around GMRES's numerical stability story (modified Gram-Schmidt's better conditioning is
partly *why* it's written as a sequential loop) before treating this as a free win.

### 2. `fd_matvec`'s `_fd_epsilon` recomputes a call-invariant norm every matvec (`jfnk.py:43-54`)

```python
def _fd_epsilon(y, v):
    ...
    v_norm = float(torch.linalg.norm(v))   # sync -- v changes every call, this one is real
    ...
    y_norm = float(torch.linalg.norm(y))   # sync -- y does NOT change across the whole GMRES loop
    return math.sqrt(eps_machine) * (1.0 + y_norm) / v_norm
```

`y` is the Newton iterate the *entire* GMRES solve for this correction is linearizing around — it's
identical across every Krylov iteration of the loop, yet `y_norm` (and its sync) is recomputed from
scratch on every single matvec call. `JFNKSolver.solve` already computes `y_flat` once per Newton
iteration and passes it into `fd_matvec(step, Y, y_flat, G_y, eps=fd_eps)` — the fix is `y_norm`
becomes an optional third argument computed once by the caller (`solve`, once per Newton iteration)
and threaded through, instead of recomputed inside `_fd_epsilon` every matvec. Cheap, structural,
no numerics change — this is the fd-side asymmetric analogue of `warpSPHCore`'s Fix 2 in the sibling
plan (same anti-pattern: a value that doesn't change across a hot loop gets a fresh host sync every
iteration anyway).

### 3. `jvp_matvec`'s own per-field zero-tangent check re-pays the exact sync warpSPHCore's Fix 2 batched away (`jfnk.py:124`)

```python
for name in names:                                    # e.g. ['u', 'v']
    ...
    if bool(tangent.abs().max() > 0):                  # sync, once per field, per matvec call
        replacements[name] = fwAD.make_dual(value, tangent)
```

This is the *same* `.abs().max() > 0` boolean-sync pattern `warpSPHCore`'s
`project_jvp_hasLiveTangent_sync_batching` fix batched into one round-trip inside the bridge — here
it's unfixed, in this repo's own code, paying one sync per integrated field (2 for the wave system:
`u`, `v`) per `jvp_matvec` call. Same fix shape applies: stack the per-field `.abs().max()` reductions
and pay one `.tolist()` for the whole matvec call instead of `len(names)` separate `bool()` casts.
Smaller in absolute terms than items 1-2 (few fields, not few dozen structural tensors), but the
same class of waste and cheap to fix once flagged.

### 4. `updateStateEuler(..., copyState=True)` clones the *whole* state, not just integrated fields (`dirk.py:127`, referenced from `util.py`)

Every `step_fn(Y)` call (i.e. every matvec's `step()` re-evaluation, for both fd and jvp) ends with
`updateStateEuler(base_state, k, a_ii * dt, copyState=True, **kwargs)` — not yet traced to its
`util.py`/`fields.py` implementation in this pass to confirm exactly what `copyState=True` clones
(whole-state vs. integrated-only), flagged here as the next thing to check before assuming it's a
problem — if it does clone non-integrated fields (positions, supports, masses, densities, kinds)
that are never touched by the wave-equation RHS, that's real avoidable per-matvec-call cost, equally
paid by fd and jvp.

## What's *not* asymmetric (rules out some hypotheses)

- GMRES's own arithmetic (item 1) runs identically regardless of matvec choice — it is not why jvp
  costs more than fd, even though it's a large piece of the total per-step cost either way.
- `flatten_integrated`/`unflatten_integrated`/`replace_integrated_fields` (`fields.py`) run on every
  matvec call for both schemes too — not separately profiled yet, worth a pass if items 1-4 don't
  close the remaining gap on their own.

## Suggested priority (not yet actioned, pending direction)

1. Item 2 (`_fd_epsilon`'s redundant sync) — smallest, safest, clearly-scoped fix; only touches
   `fd_matvec`, no numerics change, cheap to validate against the existing implicit-wave-equation
   test suite.
2. Item 3 (`jvp_matvec`'s per-field sync) — same shape, only touches `jvp_matvec`, cheap.
3. Item 4 — needs one read of `fields.py`'s `copyState` implementation before it's actionable at
   all; may turn out to already be minimal.
4. Item 1 (GMRES batching) — biggest potential win by far (it's the largest single cost center in
   both profiles), but the riskiest: touches core Krylov-solver numerics, needs a real
   accuracy/convergence check (`tests/` in this repo, plus `bench_accuracy.py` in `warpSPH`) before
   landing, not just a perf number.
