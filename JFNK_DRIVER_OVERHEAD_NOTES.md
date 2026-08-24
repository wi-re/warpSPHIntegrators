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

**Update (same day, second follow-on): the major Newton-convergence finding below is fixed.** See
"Fix landed" at the end of this doc.

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

## Suggested priority (updated 2026-08-24, same-day follow-on)

1. **Items 2/3 (the two redundant syncs) — done, committed (`c0db835`).** Full test suite green
   (1399 passed). Benchmark effect is within noise at this scale, as expected for fixes this small
   — real, correctness-preserving, but not the dominant cost.
2. **Item 1 (GMRES batching) — attempted, reverted, wrong premise.** Implemented CGS2 (classical
   Gram-Schmidt + one reorthogonalization pass, replacing the sequential modified-Gram-Schmidt
   inner loop with two matmuls against the stacked Krylov basis) plus `torch.where`-based sync
   elimination for the two scalar edge-case branches. Validated *correct* — full test suite green,
   and `bench_accuracy.py` errors bit-for-bit identical before/after on the real wave-equation case
   — but **measured slower** on the real benchmark (e.g. nx=128 jvp: 1.574→1.602 ms/RHS, fd:
   0.818→0.859 ms/RHS). Root cause, traced directly: this benchmark's GMRES calls almost never use
   more than 2-3 Krylov iterations (traced 300 real calls: mean `total_iters`=2.34, max 6) — the
   O(k²) sequential-loop cost this item targeted never actually gets large enough to matter, so
   CGS2's own overhead (`torch.stack` rebuilding the growing basis matrix every iteration, plus 4
   matmuls) is pure loss here. CGS2 may still be worth it for a problem that genuinely needs deep
   Krylov subspaces (large `restart`, ill-conditioned Jacobian) — this benchmark just isn't that
   problem. **Reverted** to the original sequential MGS loop; not landed.
3. **Item 4 (`updateStateEuler` copyState cost)** — still open, not investigated further this pass.

## Major finding (same-day follow-on, unplanned): Newton never converges early — it always burns its full iteration budget

While instrumenting *why* GMRES's own iteration count stays so low (looking for item 1's
regression), traced `JFNKSolver.solve`'s outer Newton loop directly on a real
`sdirk2_jfnk_jvp_1e-6` run (`nx=128`): **every single stage-solve used all 15 Newton corrections +
1 final check = 16 `step()` evaluations, with zero early exits, across 20 traced solves.** This
alone accounts for the great majority of this scheme's ~100 evals/step (16 outer evals + ~15×2.3
GMRES-driven evals ≈ 50/solve × 2 DIRK stages ≈ 100 — matches `bench_performance.py`'s own
`fEvalsPerStep` almost exactly).

**Root cause, confirmed by printing every iteration's `norm_fn` value against `tol`:**

```
newton iter 0: norm_fn=159.14,     tol=1e-06, converged=False
newton iter 1: norm_fn=0.1278,     tol=1e-06, converged=False   <- already WRMS-converged (< 1.0)!
newton iter 2: norm_fn=0.003356,   tol=1e-06, converged=False
newton iter 3: norm_fn=0.0001528,  tol=1e-06, converged=False   <- at float32's noise floor
newton iter 4: norm_fn=0.0001502,  tol=1e-06, converged=False
newton iter 5: norm_fn=5.972e-06,  tol=1e-06, converged=False   <- oscillating at the noise floor
...  (11 more iterations, all still oscillating in the 1e-4 to 6e-6 band, never < 1e-6 for long)
```

`dirk.py`'s `_default_norm` (`fields.py`'s `state_norm`, the Hairer-Wanner weighted-RMS norm every
DIRK call always supplies to `solver.solve`) is designed so that **`< 1.0` means converged** — it's
already normalized by `atol + rtol*|reference|`. `JFNKSolver`'s own `tol` (`1e-6` for this scheme,
from the registry key's name) is designed for `JFNKSolver`'s *own* default norm
(`_default_flat_norm`, a raw `||diff|| / max(||y||, 1)` relative-residual with no `atol`/`rtol`
scaling baked in) — a completely different convention, where `1e-6` is a sensible "tight" threshold.
Because `dirk.py` *always* supplies its own `norm` (overriding `JFNKSolver`'s default) but *never*
supplies a matching `tol` in `solver_opts`, the two mismatched conventions get compared directly:
`state_norm(...) < 1e-6` is asking for six-decimal-digit convergence *in an already-normalized
[0,1]-ish quantity* — past float32's noise floor for this problem, so it's essentially never
satisfied, and Newton exhausts its budget chasing an unreachable threshold on *every single solve*
even though the state was physically converged (WRMS < 1, even < 0.01) after 1-2 corrections.

**This means roughly 85-90% of the `step()` evaluations this whole investigation (this doc, and
`warpSPHCore`'s `warpier_jvp_dual_argument_pruning_plan.md`) has been trying to make individually
cheaper are, on this evidence, unnecessary** — a correct fix here plausibly dwarfs every
micro-optimization above combined (order-of-magnitude fewer evals/step, not a percentage
improvement). **Not fixed this pass** — deliberately left alone rather than unilaterally changed,
because the correct fix is a real judgment call with several shapes, not a clear-cut bug patch:

- Have `dirk.py` pass a `tol` in `solver_opts` compatible with its own WRMS norm's convention
  (something near `1.0`, e.g. matching `FixedPointSolver`'s implicit "no norm-based early exit
  unless caller opts in" posture) instead of leaving `JFNKSolver.tol` (tuned for a different norm)
  to leak through unchanged.
- Or decouple Newton's own convergence tolerance from `JFNKSolver.tol` entirely (today `gmres_tol`
  already defaults to `tol`, i.e. GMRES's *inner* linear-solve tolerance and Newton's *outer*
  convergence tolerance are the same number by default, even though they're conceptually different
  and, per this finding, need different scaling when a WRMS-style caller norm is in play).
- Or leave `JFNKSolver`'s contract as-is and have every *caller* that supplies a WRMS-convention
  norm supply a correspondingly-scaled `tol` explicitly, documenting the pairing requirement rather
  than changing solver code at all.

Whichever shape, this changes the *number of Newton iterations `sdirk2_jfnk_*` schemes actually
run* — i.e. a real behavior change to a shipped, tested integration scheme, not a transparent perf
cleanup, and worth a deliberate decision (and a fresh `bench_accuracy.py` comparison once decided)
rather than a same-session drive-by fix.

## Fix landed (same day, second follow-on): shape 2 — decoupled Newton's own tolerance

Per explicit direction ("no consumer depends on the current convergence behavior, fixing it now is
the best spot"): implemented the second option above. `JFNKSolver` gained a `newton_tol` parameter
(constructor + `solve`'s `**opts`), independent of `tol` (which stays GMRES's own inner linear-solve
tolerance — unchanged, still `1e-6`-style values, unrelated to whatever `norm` convention a caller
uses). Both convergence checks (`for` loop's early exit, and the final post-budget check) now
compare against `newton_tol` instead of `tol`. `newton_tol` defaults to `None` → falls back to
`tol` when unset, so `JFNKSolver` used standalone with no custom `norm` (`_default_flat_norm`'s own
convention, which `tol` was already tuned for — e.g. `test_jfnk_solver_converges_on_a_trivial_
linear_fixed_point`) is completely unaffected.

`dirk.py` (the only caller supplying a mismatched-convention `norm`) now defaults
`solver_opts['newton_tol']` — `solver_opts = {'newton_tol': 1e-3, **kwargs.get('solver_opts', {})}`,
a caller's own explicit value still wins. **`1e-3`, not `1.0`**: `1.0` (the WRMS norm's own literal
"converged" threshold) was tried first and empirically too loose — it broke a convergence-order
regression test (measured order went *negative*, i.e. error *growing* as `dt` shrinks) and an
exact-stage-solution comparison test, both because Newton's own error, uncontrolled below 1.0,
started contaminating the DIRK scheme's achieved order at the smaller `dt` values those tests
exercise. `0.01` fixed the order tests but still missed the tightest exact-comparison test's
`rel=1e-4` requirement by about 1.5x. `1e-3` passes everything — found empirically, by bisecting
against the existing test suite (which encodes the actual required accuracy, not by reasoning about
IEEE-754 error propagation from first principles), so it's a validated, not merely plausible, choice.

**Validated**: `warpSPHIntegrators` full suite — 1399 passed, 114 skipped, no regressions
(including `test_jfnk_reproduces_exact_backward_euler_where_picard_diverges` and the
convergence-order sweep across all three shipped DIRK tableaus, both of which failed at
`newton_tol=1.0`/`0.01` before landing on `1e-3`). `warpSPH`'s `test_implicitWaveEquation.py` (12
passed) and `test_bench_wave.py` (11 passed) — the only two `warpSPH` test files touching JFNK/DIRK.
`bench_accuracy.py` on the real wave-equation case: **errors bit-for-bit identical** to before this
fix at every `dt` tested (`sdirk2_jfnk_jvp_1e-6`/`sdirk2_jfnk_fd_1e-6`), confirming zero accuracy
cost — only the iteration count changed.

**Real-world effect, `bench_performance.py` (nx=32..256, `--device cuda:0`)**:

| nx  | jvp ms/step before | jvp ms/step after | jvp f/step before | jvp f/step after | fd ms/step before | fd ms/step after | fd f/step before | fd f/step after |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|
| 32  | 113.0 | **29.4** | 82.3  | **18.3** | 66.6 | **22.9** | 87.9  | **27.1** |
| 64  | 155.5 | **32.6** | 98.9  | **19.5** | 83.8 | **24.5** | 103.7 | **27.5** |
| 128 | 160.6 | **37.9** | 102.0 | **21.6** | 87.4 | **27.3** | 106.9 | **30.5** |
| 256 | 163.4 | **34.5** | 104.3 | **26.5** | 90.8 | **27.4** | 111.0 | **37.8** |

**~4-4.6x real wall-clock speedup for `sdirk2_jfnk_jvp_1e-6`, ~2.9-3.6x for `sdirk2_jfnk_fd_1e-6`**,
at every resolution tested, with bit-for-bit identical accuracy — by a wide margin the largest win
across this entire investigation (this doc's items 2/3, the reverted item 1, and everything in
`warpSPHCore`'s `warpier_jvp_dual_argument_pruning_plan.md` combined moved the needle a few percent;
this moved it 3-4.6x). jvp benefits more than fd in relative terms because jvp's per-eval cost was
already higher (see the sibling plan doc's "Correction" section) — cutting 80-90% of the eval
*count* pays off proportionally more for the more-expensive-per-eval scheme.
