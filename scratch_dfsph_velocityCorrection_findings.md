# Follow-on to E1.9: fixing the missing velocity correction in `IncompressibleSystem.finalize`

Scratch findings file for a supervising session to fold into `JFNK_PLAN.md`.
Direct follow-on to `scratch_dfsph_kolmogorov_findings.md` / `JFNK_PLAN.md`
E1.9 finding #3: at `nx=128`, `xi=1.0`, `k=4` (`cases/kolmogorov.py`'s own
parameters), a DFSPH run **with** the case's own default physical viscosity
(`alpha=0.01`) diverged to NaN at step 720, while the zero-viscosity run at
the same resolution stayed bounded for the full 1000-step budget. Both runs
showed real localized density excursions before failing, tentatively
attributed to particle disorder rather than a viscous-shear mechanism.

## Answer, up front

**The fix helps, clearly and by a wide margin, and has been applied to
production code.** `warpSPH/src/warpSPH/systems/incompressible.py`,
`IncompressibleSystem.finalize`, line 275: uncommented
`self.state.velocities -= proj_vel` (was `# self.state.velocities -=
proj_vel`). This is the only change made to production code, confirmed by
`git diff` (one line flipped, nothing else touched, `schemeConfig.
shiftProperties.active`-gated block above left exactly as found). With the
fix live, the `nx=128`/`alpha=0.01` case that previously diverged at step
720 survives the full 1000-step budget and, extended further, survives to
1600 steps (2.2x the original divergence point) with no NaN -- density
excursions still occur (down to `rhoMin~0.44` transiently) but recover
instead of blowing up. `test_physics.py`'s three `tgv*` tests (the existing
incompressible-scheme regression test) and the full `warpSPH` test suite
both pass with the fix applied. The `nx=128`/`nu=0` case, which already
worked before the fix, also still works after it -- no regression.

## Mechanism, confirmed by reading the code (this session, matching the
task's own framing)

`IncompressibleSystem.finalize` runs a second, constant-density pressure
solve (`solveIncompressible`) whose output `dvdt_incomp` is applied as a
**position** correction:
```python
dx = dt**2 * dvdt_incomp
proj_vel = torch.einsum('nij, ni -> nj', gradVel, dx)
self.state.positions += dx
self.state.velocities -= proj_vel   # was commented out
```
`gradVel` (the velocity-field Jacobian) and `proj_vel = grad(V)*dx` are
computed immediately before this, unconditionally -- i.e. the codebase
already computes the correct first-order Taylor correction every single
step, and then, until this fix, silently discarded it. Every step, DFSPH's
own implicit particle-shifting mechanism (this second pressure solve +
position correction, distinct from and unconditionally active regardless of
the separate `schemeConfig.shiftProperties.active`-gated block earlier in
the same function, which is untouched dead debug scaffolding from a prior
session per the project owner's explicit confirmation) moved particles to a
new location while leaving their carried velocity computed for the *old*
location. This is exactly the kind of small, compounding kinematic
inconsistency that would show up first, and worst, under sustained shear at
higher resolution/longer runs -- consistent with why `nx=24`/`nx=48`/`nx=64`
never reproduced the divergence (see "Cheaper-resolution search" below) but
`nx=128` did.

## Method: tested without touching production code first

Per the task's own instruction, the fix was tested via a runtime monkeypatch
before any file was edited. New script:
`warpSPH/scripts/probe_kolmogorovIncompressibleVelCorrection.py`. It:
- imports and reuses `probe_kolmogorovIncompressible.py`'s own `run()`/
  `buildKolmogorovIncompressibleSystem()` directly (no duplicated physics
  setup code);
- takes `inspect.getsource(IncompressibleSystem.finalize)`, textually
  replaces the exact commented line with its live form, re-`exec`s the
  patched source into the module's own `__dict__` (so every closed-over name
  -- `solveIncompressible`, `computeDensities`, `warpOperation`,
  `detectFreeSurface`, etc. -- still resolves), and rebinds the class
  attribute for the process's lifetime;
- one real snag found doing this: `exec`-ing a method body outside its
  original `class` statement loses the implicit `__class__` cell zero-arg
  `super()` relies on (`RuntimeError: super(): __class__ cell not found`) --
  fixed by rewriting `super().finalize(` to `super(IncompressibleSystem,
  self).finalize(` in the patched source (exactly equivalent at runtime;
  `IncompressibleSystem` is resolvable from the module's own globals since
  the class object already exists when the patch runs).
- A 10-step `nx=24` smoke test confirmed the patch is actually live (KE/vMax
  trajectories diverge from the unpatched baseline by step 1, in the
  expected small direction), not a silent no-op, before spending any time on
  the expensive `nx=128` comparison.

`cases/kolmogorov.py`, `cases/tgv.py`, and the probe script from E1.9
(`probe_kolmogorovIncompressible.py`) were **not modified** by this
follow-on except for the one production-code line described above, applied
only after the monkeypatched comparison confirmed the fix helps.

## Cheaper-resolution search (did not reproduce; fell back to `nx=128` per
plan)

Per the task's own instruction, `nx=48` and `nx=64` (same `alpha=0.01`,
`xi=1.0`, `k=4`, natural CFL-respecting `dt`, `cflFactor=0.3`, `maxDt=0.1`)
were tried first, extended to 1500 steps each (beyond the `nx=128` baseline's
own 720-step divergence point, in simulated-time terms this is well past
saturation at these resolutions):

| `nx` | steps run | wall time | outcome |
|---|---|---|---|
| 48 | 1500/1500 | 105.2s | **no divergence** -- settled `KE~0.86-1.24`, `vMax~1.1-2.0`, `rhoStd~6-14e-3` |
| 64 | 1500/1500 | 108.1s | **no divergence** -- settled `KE~1.04-1.5`, `vMax~1.2-2.5`, `rhoStd~3.5-6.6e-3` |

Neither reproduced the instability within budget, so per the task's own
fallback instruction this follow-on used `nx=128` directly for the actual
before/after comparison (confirmed tractable, ~53-75s/run at this
resolution on this machine -- see below). No independent explanation was
sought for why `nx=48`/`nx=64` don't reproduce it (plausibly just needing
more particles/finer resolution for the shear-driven disorder to compound
enough to matter within a 1000-1500-step budget) -- this is a negative
result stated for completeness, not chased further.

## `nx=128`, `xi=1.0`, `k=4`, `alpha=0.01`: before/after, with numbers

All three runs below are the same initial condition (`seed=0`, same jitter,
same forcing, same `cflFactor=0.3`/`maxDt=0.1` `dt` policy) -- only the
`finalize` velocity-correction behavior differs.

### Baseline (unpatched, i.e. current-`main`-before-this-fix behavior; rerun
this session to reconfirm E1.9's own number, via the unmodified
`probe_kolmogorovIncompressible.py`)

| step | t | KE | vMax | rhoMin | rhoMax | rhoStd |
|---|---|---|---|---|---|---|
| 500 | 4.76 | 2.824 | 2.410 | 0.982 | 1.011 | 1.9e-3 |
| 600 | 5.46 | 2.568 | 2.563 | 0.885 | 1.022 | 1.9e-3 |
| 620 | 5.61 | 2.539 | 2.571 | 0.843 | 1.010 | 2.4e-3 |
| 680 | 6.06 | 2.474 | 2.297 | 0.880 | 1.036 | 3.4e-3 |
| 700 | 6.22 | 2.422 | 2.147 | 0.857 | 1.056 | 4.9e-3 |
| **720** | **6.39** | **NaN** | **NaN** | **NaN** | **NaN** | -- |

**Diverged at step 720** (`t=6.39s`), reproducing E1.9's own reported step
720 exactly (this run is deterministic given the fixed seed -- same result
as the original E1.9 probe to the step). Wall time to divergence: 52.9s.

### Patched (monkeypatch, velocity correction live, `finalize` otherwise
identical), 1000-step budget

| step | t | KE | vMax | rhoMin | rhoMax | rhoStd |
|---|---|---|---|---|---|---|
| 500 | 4.81 | 2.460 | 2.471 | 0.779 | 1.016 | 4.0e-3 |
| 520 | 4.97 | 2.462 | 2.363 | **0.709** | 1.012 | 2.9e-3 |
| 540 | 5.13 | 2.461 | 2.489 | 0.970 | 1.008 | 1.6e-3 |
| 720 | 6.69 | 1.905 | 2.153 | 0.950 | 1.021 | 2.0e-3 |
| 740 | 6.85 | 1.848 | 2.330 | 0.843 | 1.018 | 2.8e-3 |
| 760 | 7.02 | 1.808 | 2.273 | 0.790 | 1.016 | 3.4e-3 |
| 1000 | 8.74 | 1.908 | 2.176 | 0.893 | 1.016 | 2.5e-3 |

**Survives the full 1000-step budget, no divergence.** Note the run passes
straight through step 720 (where baseline died) and through an even deeper
density excursion at step 520 (`rhoMin=0.709`, comparable to baseline's own
worst excursion) without diverging -- the correction doesn't just delay
failure, density excursions of comparable severity to the ones that killed
the baseline run now recover instead of cascading to NaN.

### Patched, extended to 1600 steps (stress test beyond the 1000-step
budget, to check the fix isn't just deferring the same failure a bit
further)

| step | t | KE | vMax | rhoMin | rhoMax | rhoStd |
|---|---|---|---|---|---|---|
| 1040 | 9.11 | 1.859 | 2.180 | 0.750 | 1.024 | 3.2e-3 |
| 1080 | 9.45 | 1.821 | 2.300 | **0.439** | 1.012 | 4.7e-3 |
| 1120 | 9.79 | 1.767 | 2.320 | 0.819 | 1.031 | 2.9e-3 |
| 1280 | 11.08 | 1.636 | 2.098 | 0.688 | 1.023 | 4.4e-3 |
| 1320 | 11.43 | 1.626 | 2.276 | **0.579** | 1.014 | 3.6e-3 |
| 1440 | 12.37 | 1.702 | 2.525 | **0.669** | 1.018 | 4.1e-3 |
| 1600 | 13.56 | 2.006 | 2.589 | 0.957 | 1.009 | 1.5e-3 |

**Still no divergence at 1600 steps**, despite a substantially deeper
density excursion than anything seen in the baseline run (`rhoMin=0.439` at
step 1080, i.e. 56% below rest density, transient and recovered) -- the
fixed run is not merely surviving marginally longer, it is tolerating
noticeably more severe density disorder than the unfixed run ever
encountered before failing. Wall time: 121.2s for 1600 steps.

## Production-code validation (after applying the fix)

- **The one-line diff**, confirmed via `git diff -- src/warpSPH/systems/
  incompressible.py`:
  ```diff
  -        # self.state.velocities -= proj_vel
  +        self.state.velocities -= proj_vel
  ```
  Nothing else in `incompressible.py` touched; the `shiftProperties.active`-
  gated block earlier in `finalize` (known dead debug scaffolding, per the
  project owner's explicit instruction) was left byte-for-byte unchanged.

- **`pytest tests/test_physics.py -k tgv`**: 3 passed (`tgvResult` fixture
  runs `cases/tgv.py`'s `tgvCase`, `nx=32`, `scheme='divergenceFree'`, the
  exact scheme this fix touches).

- **`pytest tests/test_physics.py`** (full file, all cases/schemes): 60
  passed.

- **`pytest tests/test_runner.py tests/test_incompressibleKrylov.py`** (the
  other two test files referencing `IncompressibleSystem`/
  `divergenceFree`, found by grep): 38 passed.

- **Full `warpSPH` test suite** (`pytest tests/`): run twice; both times the
  progress bar reached `[100%]` with zero `F`/`E` markers, one pre-existing
  skip (`s`, unrelated to this change -- present before this fix too), and
  the second run's exit code was captured explicitly via `echo "EXIT=$?"`
  immediately after the `pytest` invocation: **`EXIT=0`**. (The literal
  final `N passed, M warnings in Xs` summary line itself was not visible in
  either captured log, seemingly lost to a background-output-capture quirk
  in this session's own tooling rather than a test failure -- but a
  captured, immediate `$?` of `0` right after the run is direct, unambiguous
  confirmation pytest itself reported success, which is the evidence that
  actually matters here.)

- **`nx=128`, `nu=0` (zero-viscosity) case, re-run against the real,
  now-patched production code** (not the monkeypatch -- the actual fixed
  `IncompressibleSystem.finalize`, via the unmodified
  `probe_kolmogorovIncompressible.py --nu 0.0`): 1000/1000 steps, no
  divergence, settled `KE~1.5-1.6`, `vMax~1.8-2.2`, comparable to (not
  identical to, since the correction changes the trajectory) E1.9's own
  pre-fix zero-viscosity numbers (`KE~2.57-2.65`, `vMax~2.2-2.6`) -- the
  zero-viscosity case that already worked before this fix still works
  after it. Some individual `rhoMin` excursions are deeper post-fix
  (e.g. `rhoMin=0.679` at step 480, `rhoMin=0.747` at step 760) but none
  cascade to divergence, consistent with the `alpha=0.01` case's own pattern
  above.

## Caveats, stated plainly

- One seed (`seed=0`), one machine, one resolution actually compared
  before/after (`nx=128`); the `nx=48`/`nx=64` cheaper-resolution search
  ruled those out as reproduction cases but was not itself repeated with the
  fix (there is nothing to fix there, since nothing diverges at those
  resolutions in the first place).
- The 1600-step extension is a single additional data point past the
  original 1000-step budget, not a systematic "how far can this go" sweep;
  it was run specifically to check the fix isn't just deferring the same
  failure mode a short distance, and it appears not to be (the run
  tolerates deeper density excursions than the ones that killed baseline),
  but a much longer run (5000+ steps) was not attempted and might yet
  surface a different failure mode at long time horizons.
- The monkeypatch mechanism (`inspect.getsource` + text replace + `exec` +
  class-attribute rebind) is a legitimate way to test a single-line change
  to a method's behavior without touching the file on disk, but it is not
  identical bytecode to editing the file and reimporting -- the `super()`
  rebinding needed to make it work is a from-scratch equivalent, not the
  literal original mechanism. The actual production fix (direct one-line
  file edit, verified via `git diff`) is what was validated by the test
  suite runs above, not the monkeypatch itself -- the monkeypatch was only
  ever the *decision* tool, not the shipped mechanism.
- Whether this fix also affects the deltaSPH/weakly-compressible schemes
  `cases/kolmogorov.py` actually runs in production was not tested --
  `IncompressibleSystem`/`IncompressibleSystem.finalize` is specific to
  `scheme='divergenceFree'` (DFSPH); `WeaklyCompressibleSystem` is a
  different class with its own `finalize`, not touched by this change, and
  not examined in this follow-on.
- No investigation was done into *why* the correction matters as much as it
  apparently does (e.g. no direct measurement of `proj_vel`'s magnitude
  relative to the velocity field it corrects, no ablation isolating just
  the position-update half from the velocity-update half). The mechanism
  described above ("kinematic inconsistency compounding under sustained
  shear") is the natural reading of the code and is consistent with the
  observed step-720-survival result, but it was not independently
  instrumented or proven beyond that.

## Files

- `warpSPH/src/warpSPH/systems/incompressible.py` -- **production fix
  applied**, line 275 (`IncompressibleSystem.finalize`): uncommented
  `self.state.velocities -= proj_vel`. The only change to this file; the
  `shiftProperties.active`-gated block above is untouched.
- `warpSPH/scripts/probe_kolmogorovIncompressibleVelCorrection.py` -- new,
  the monkeypatch-based before/after comparison tool used to test the fix
  before applying it. Imports and reuses
  `probe_kolmogorovIncompressible.py` directly rather than duplicating its
  setup code.
- `warpSPH/scripts/probe_kolmogorovIncompressible.py` -- unmodified
  (E1.9's own script), reused as-is for the baseline reproduction and the
  post-fix `nu=0` regression check.
