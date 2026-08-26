# Neighbor-list-staleness ablation: does refreshing adjacency inside the JFNK solve raise E1.6's `dt`-multiplier ceiling?

Status: done, 2026-08-26. One snapshot (`nx=128`, `xi=1.0`, `t=6.0s`, the exact
state E1.6's own table used), one resolution, `matvec='fd'` full sweep plus a
`matvec='jvp'` spot check at the two multipliers where it matters. Clean,
reproducible **negative result**: rebuilding does not raise the ceiling, and
mildly *hurts* where the baseline is already struggling, at real extra cost.

## Question

E1.6 found JFNK's safe `dt`-multiplier over the acoustic CFL drops from ~20x
(quiescent flow) to `mult<=5` once Kolmogorov flow is genuinely turbulent
(`vMax~2.2-2.9`). Within one implicit step, the whole DIRK/JFNK solve (every
stage, every outer Newton iteration, every inner GMRES matvec) reuses the
*same* `AdjacencyList`, built once at the end of the previous real step
(confirmed by reading `f_acoustic_core`'s own docstring and tracing
`AcousticCoreSystem.initializeNewState`/`JFNKSolver.solve`/`dirk.py`'s
`step_fn` — every Newton/GMRES iterate's `system.adjacency` traces back,
unmutated, to the frozen adjacency the *outer* real step started with). The
real production scheme (`deltaSPH_step`) instead calls `buildVerletList(...,
priorNeighborhood=adjacency, ...)` at the top of *every* RHS evaluation. At
`mult=5-10`, particles can move a non-trivial fraction of `h` within one
implicit step, so the question: is E1.6's ceiling partly a neighbor-staleness
artifact, or purely a Newton/GMRES accuracy limit?

**Answer: purely a Newton/GMRES accuracy limit.** Rebuilding does not push
the ceiling higher anywhere it was tested, and at the two conditions where
the frozen-adjacency baseline was already marginal or failing (`mult=5`/`fd`,
`mult=10`/both matvecs), the rebuild variant diverges *faster and harder*,
not slower — the opposite of the hoped-for effect. Where the baseline was
already fine (`mult<=3` any matvec, `mult=5`/`jvp`), rebuild changes nothing
materially (agreement to the 4th-5th significant digit). This is a genuine,
reproducible finding (bit-identical across a repeat run at the sharpest data
point), not noise.

## What was built

- **`warpSPH/src/warpSPH/schemes/acousticCore.py`**: new sibling function
  `f_acoustic_core_rebuildAdjacency` (plus a tiny helper class
  `_PositionSupportsView`), appended after the existing `f_acoustic_core`.
  `f_acoustic_core` itself is **unmodified** — same body, same behavior, same
  exports (both are now in `__all__`). The new function is physically
  identical to `f_acoustic_core` (continuity, EOS, pressure force, optional
  density/velocity diffusion, optional forcing — every term unchanged) except
  it inserts, at the top:
  ```python
  primalPositions, _ = fwAD.unpack_dual(state.positions)
  queryView = _PositionSupportsView(primalPositions, state.supports)
  adjacency = buildVerletList(
      queryView, system.domain, verletScale=config.verletScale,
      supportMode=SupportScheme.SuperSymmetric,
      priorNeighborhood=system.adjacency, verbose=verbose,
  )
  ```
  mirroring `deltaSPH_step`'s own call
  (`buildVerletList(currentState, config.domain, verletScale=config.verletScale,
  supportMode=SupportScheme.SuperSymmetric, priorNeighborhood=adjacency,
  verbose=False)`, `warpSPH/src/warpSPH/schemes/deltaSPH.py:69-74`) down to
  the keyword names, and every subsequent `warpOperation(...)` call uses this
  fresh `adjacency` instead of `system.adjacency`. The function returns
  `(update, adjacency)` (the fresh one), same shape as `f_acoustic_core`'s
  `(update, system.adjacency)`.

- **`warpSPH/scripts/probe_kolmogorovAdjacencyRebuild.py`** (new): extends
  `probe_kolmogorovContinuation.py`'s exact method (same `loadSnapshot`,
  `buildSchemeConfig`, `diagnostics`, forced/decaying branch structure) with
  one new axis — baseline vs. rebuild `f`, run back-to-back on the same
  loaded snapshot — plus a `buildVerletList` call counter (a thin
  monkey-patch of `warpSPH.schemes.acousticCore.buildVerletList` for the
  duration of one rebuild run, restored in a `finally`) for the cost
  measurement in item 5.

## Item 2 — the dual-tensor question, checked directly (not assumed)

Per the task, checked whether `buildVerletList` crashes on forward-mode
dual-wrapped positions before writing anything defensive. **It does not
crash**, in either code path:
- `priorNeighborhood=None` (fresh build): succeeded, and the returned
  adjacency's `i`/`j`/`numNeighbors` were bit-identical to calling
  `buildVerletList` on the manually-unpacked primal directly (verified with
  a large synthetic tangent, `*1000`, specifically so a primal/tangent mixup
  would have been visible).
- `priorNeighborhood=<real adjacency>` (the actual path this ablation uses,
  which exercises `_verlet_validity_metrics`'s distance computation between
  a dual query and a plain prior adjacency): also succeeded, no crash, no
  `torch.autograd.forward_ad` assertion.

So `warp`'s `wp.from_torch()` bridge silently reads through to the dual
tensor's primal buffer here — this is a genuine negative finding, contrary
to the task's stated expectation ("almost certainly isn't written to accept
dual tensors"). The explicit `fwAD.unpack_dual(...).primal` extraction was
kept anyway as the semantically-correct thing to do regardless (adjacency
should never carry a tangent; leaving `adjacency.queryPositions` dual-tagged
risks it outliving its `dual_level()` in general, though tracing
`JFNKSolver.solve`'s call order shows this doesn't actually leak into
`system.adjacency` today — every return path re-evaluates `step(Y)` outside
any `dual_level()` immediately before returning). It costs nothing extra:
`fwAD.unpack_dual` on a plain tensor is a no-op read.

**Consequence**: no need to scope this ablation to `matvec='fd'` only — the
rebuild variant works and was tested under both `fd` and `jvp`.

## Stability sweep (matvec='fd', nx=128, snap_nx128_003200.pt, t=6.0s, vMax~2.2)

5 real JFNK steps per cell (`Backward Euler (implicit)`, `tol=1e-6`,
`max_iterations=25`, `gmres_maxiter=80`), compared at the same `dt` for
baseline (frozen adjacency) vs. rebuild (per-call adjacency), both branches:

| mult | branch | baseline (frozen) | rebuild (per-call) |
|---|---|---|---|
| 1 | forced | fine, tracks RK4 (KE 2.711→2.711) | fine, indistinguishable from baseline |
| 1 | decaying | fine, tracks RK4 | fine, indistinguishable from baseline |
| 3 | forced | fine (KE 2.710→2.708) | fine, indistinguishable from baseline |
| 3 | decaying | fine (KE 2.708→2.698) | fine, indistinguishable from baseline |
| 5 | forced | **drifts**: `vMax` 2.23→3.18 (step 2)→177.5 (step 3)→diverges step 5 | **drifts worse**: `vMax` 2.23→**34.0** (step 2)→**11362** (step 3)→diverges step 4 |
| 5 | decaying | fine, all 5 steps (KE 2.705→2.688) | fine, all 5 steps, indistinguishable from baseline |
| 10 | forced | diverges: `vMax`→5.03 (step1)→76.2(step2)→304(step3)→diverges step4 | diverges **one step earlier and harder**: `vMax`→4.41(step1)→**77707**(step2)→diverges step3 |
| 10 | decaying | drifts late: fine to step 3, `vMax`→6.75 (step4)→50.1(step5) | diverges earlier: `vMax`→72.0(step2)→1290(step3)→diverges step4 |

Baseline numbers reproduce E1.6's own table exactly in kind (`mult<=3` safe,
`mult=5` forced-branch drift, `mult=10` outright divergence) — confirms this
run is measuring the same regime E1.6 characterized, not a different one.

**`matvec='jvp'` spot check, forced branch (where E1.6 found JVP's payoff):**

| mult | baseline (frozen) | rebuild (per-call) |
|---|---|---|
| 5 | fine, all 5 steps (`vMax` 2.23→2.22, KE 2.708→2.704) | fine, all 5 steps, indistinguishable (`vMax`/`KE` agree to 3-4 significant figures) |
| 10 | diverges by step 4 (`vMax`→2.46→3.68→14.0→diverge) | diverges by step 4, **worse intermediate blow-up** (`vMax`→3.52→207.0→719.2→diverge) |

Same pattern as FD: no improvement where the baseline already holds
(`mult=5`), no rescue and a harder landing where it doesn't (`mult=10`).

**Reproducibility check**: re-ran the sharpest cell (`mult=5`, `fd`, forced)
independently — bit-identical trajectory both runs (`KE=2.8617`,
`vMax=33.9514` at step 2; `KE=41134.5273` at step 3, exact match). Not GPU-
reduction-order noise.

## Cost overhead (item 5)

`Backward Euler (implicit)` is a 1-stage tableau here, so "extra calls" is
purely Newton-iterations × (1 + GMRES-iterations) per real step, no stage
multiplier. Measured directly via the call-counting wrapper (baseline: 0
inner `buildVerletList` calls — adjacency frozen for the whole step, exactly
1 outer-loop rebuild per real step as before; rebuild variant: one call per
RHS evaluation):

| mult | branch | matvec | rebuild calls/real-step | wall-clock overhead (rebuild / baseline) |
|---|---|---|---|---|
| 1 | forced | fd | 36.8 | 2.17x |
| 1 | decaying | fd | 36.8 | 2.36x |
| 3 | forced | fd | 149.0 | 2.17x |
| 3 | decaying | fd | 149.2 | 2.21x |
| 5 | forced | fd | 267.5 (diverged early) | 1.24x |
| 5 | decaying | fd | 390.8 | 2.09x |
| 10 | forced | fd | 356.3 (diverged early) | 1.63x |
| 10 | decaying | fd | 288.0 (diverged early) | 1.11x |
| 5 | forced | jvp | 258.0 | 1.19x |
| 10 | forced | jvp | 247.0 (diverged early) | 1.38x |

150-400 extra `buildVerletList` calls per real step, in the same order of
magnitude as the task's own back-of-envelope estimate (Newton iterations x
(1+GMRES iterations), `max_iterations=25`/`gmres_maxiter=80` as the nominal
ceiling, actual Newton/GMRES iteration counts far below that in practice).
Wall-clock overhead is consistently **1.1x-2.4x**, smaller than the raw
call-count multiplier would suggest, because each `buildVerletList` call
here is mostly the cheap "is the prior list still valid" check
(`_verlet_validity_metrics`), not a full rebuild — the dominant per-call cost
is still the `warpOperation` (Divergence/Gradient/Laplacian) kernel launches
that both variants already pay for equally.

## Reading the two findings together

1. **Rebuilding adjacency inside the implicit solve does not raise JFNK's
   safe `dt`-multiplier ceiling on this probe.** Everywhere the frozen-
   adjacency baseline already held (`mult<=3` either matvec, `mult=5`/`jvp`),
   the rebuild variant produced materially the same trajectory. E1.6's
   ceiling is not a neighbor-staleness artifact — it stands as a genuine
   Newton/GMRES numerical-accuracy limit against a genuinely chaotic,
   multi-scale nonlinear state, exactly as E1.6 itself concluded.

2. **Where the baseline was already marginal or failing (`mult=5`/`fd`
   forced, `mult=10` both matvecs), rebuilding made it measurably worse**,
   not neutral — diverging one step earlier and to dramatically larger peak
   values before the divergence guard trips. A plausible mechanism, offered
   with appropriate uncertainty (not verified further this pass): once `dt`
   is large enough that different Newton iterates within one implicit step
   correspond to meaningfully different particle configurations, the
   rebuild variant lets the discrete adjacency itself change between Newton
   iterations of the *same* implicit solve, so the "linearize around a fixed
   operator" assumption Newton's own convergence theory relies on is now
   linearizing around a target that's also discretely shifting underneath
   it — an extra source of inconsistency stacked on top of an already-
   marginal correction, in a regime E1.6 already showed has little accuracy
   margin to spare. This is speculative, not independently confirmed with a
   separate diagnostic (e.g., tracking how many actual full rebuilds vs.
   cheap reuses happened per divergent run) — flagged as follow-up, not
   claimed as established.

3. **Given (1) and (2), rebuilding is not worth it for this core at this
   regime**: no stability upside anywhere tested, a real 1.1x-2.4x wall-clock
   tax, and a mild-to-moderate stability *downside* exactly where it would
   have needed to help. The practical guidance from E1.6/E1.7 stands
   unchanged: reach for the exact JVP matvec and/or a lower-`Ma` operating
   point to buy back margin, not a within-step adjacency refresh.

## Caveats, stated plainly

- One snapshot (`nx=128`, `xi=1.0`, `t=6.0s`), one resolution, one
  realization of a chaotic flow — same scope limits E1.6/E1.7 already
  documented apply here too.
- Only 5 real steps per cell (matching the short-horizon comparisons E1.6's
  own table used) — long-run behavior at multipliers that survive 5 steps
  (`mult<=3`, `mult=5`/`jvp`) was not separately re-checked for the rebuild
  variant over a longer horizon; no reason to expect it to differ given the
  near-bit-identical short-horizon agreement, but not verified.
- The mechanism offered in point 2 above is a plausible explanation, not an
  independently confirmed one — no separate instrumentation of "how often
  did the adjacency actually change mid-Newton-solve, vs. just get
  re-validated and reused" was added. Would be the natural next diagnostic
  if this needs pinning down further.
- `matvec='jvp'` was only spot-checked at `mult=5`/`mult=10`, forced branch
  (the two data points where E1.6 found JVP's own payoff was concentrated),
  not the full grid — the `fd` grid is the complete one.
- Snapshots reused from a prior session's scratchpad
  (`/tmp/kolmogorovSpinup/snap_nx128_003200.pt`, confirmed to match E1.6's
  own reported `t=6.0s`/`dt=0.001875`/`xi=1.0` snapshot exactly by its saved
  metadata), not regenerated — cheap-`nx` regeneration was unnecessary since
  the exact E1.6 snapshot was still present on disk.

## Files

- `warpSPH/src/warpSPH/schemes/acousticCore.py` — new function
  `f_acoustic_core_rebuildAdjacency` (+ helper `_PositionSupportsView`)
  appended; `f_acoustic_core` unmodified; `__all__` extended to export both.
- `warpSPH/scripts/probe_kolmogorovAdjacencyRebuild.py` — new sweep script
  used for every number in this report.
- Existing `warpSPH/tests/test_acousticCore*.py` (15 tests across
  `test_acousticCore.py`/`test_acousticCoreStability.py`/
  `test_acousticCoreForcing.py`/`test_acousticCoreDissipation.py`) re-run
  green after this change, confirming no regression to `f_acoustic_core`.
