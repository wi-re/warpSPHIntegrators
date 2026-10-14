# Phase 1 validation — 14 October 2026

The Phase 1 implementation and documentation are validated and included in the
bundled wave 8 commit. Wave 8 is **unreleased**; package metadata remains at 0.6.1.
The starting point was commit `711aba8` plus the existing local wave 8 changes.
Those coefficient arrays, enum additions and exports were preserved exactly;
the runtime correction in this phase is SSPSDIRK2's geometric classification.

## Gates restored

- Drift-field and pRRK tests now belong to CI categories. Collection rejects
  an uncategorized file, duplicate category assignments and cases selected by
  zero or multiple CI shards. CI and collection use `.github/test-shards.json`.
  An actual uncategorized-file collection probe exits with pytest usage code 4.
- All 45 estimate emitters receive adaptive-loop checks selected from driver
  and tableau metadata. TR-BDF2 and Gauss-Legendre 2 use estimate exponent 3.
  ROS3P/EXPRB32 use the forced problem to activate their estimates; moderate
  tolerances keep high-order pairs above the float64 estimator floor.
- Registry metadata selects 53 JFNK implicit and 78 ordinary explicit
  registrations for state-gradient checks, with runs long enough to pass
  multistep startup. ROS3P, ETD2RK and EXPRB32 retain their dedicated operator
  parity tests; RKC1/RKC2/RKL2 have new diffusion gradient checks with an explicit
  stage count. Every registered driver has coverage across these suites.
- All 36 DIRKs additionally check captured RHS-parameter gradients against
  strict-solve finite differences. These tests include differentiable state
  roots, which activate the existing JFNK IFT path; they do not establish a
  parameter-only gradient contract for detached initial states.
- Every DIRK passes the nonlinear semilinear Burgers reference/refinement check.
  A/L-stable additions also pass nonlinear stiff relaxation at rate 100, dt 0.1.
  SFSDIRK4–8 retain their bounded-region qualification.
- The TVD inventory covers all 137 registrations. Completeness is a cheap
  separate check; measurements are parameterized per scheme for xdist.
  Wave 8 verdicts use the existing step-advection CFL grid through 5 and
  all-stage convex-combination scan cap 4. These finite model measurements
  do not establish general SSP or unconditional TVD.

## Numerical failures resolved

The audit's 26 failures were stale FSAL/TVD inventories, three Kepler
convergence cases (including one monotonicity failure), and SSPSDIRK2's
geometric classification. FSAL expectations now include the 20 new eligible
DIRKs, and direct tests check explicit first rows and matching endpoint weights.

The standard Kepler grid is 0.1, 0.05, 0.025, 0.0125 over T=2.
Strict probes use `tol=gmres_tol=newton_tol=1e-14, fd_eps=1e-6`:

| Scheme | Default standard-grid order | Strict standard-grid order | Default coarse-grid order |
|---|---:|---:|---:|
| ESDIRK4(3)7L[2]SA | 3.394 | 3.998 | 4.009 |
| ESDIRK6(5)9L[2]SA | 5.003, nonmonotone | 5.998 | 5.966 |
| SFSDIRK6 | 3.835 | 3.999 | 3.990 |

The default grids are recorded in `tests/conftest.py`; strict standard-grid
regressions remain in `tests/test_dirk.py`. The order tolerance is unchanged.
Monotonicity now uses the same justified grids as order measurement.

SSPSDIRK2 satisfies the symplectic RK condition with exactly zero defect.
Its tableau is two implicit-midpoint half-steps and
`R(z)=((1+z/4)/(1-z/4))**2`, with stiff limit **+1**. Its metadata now says
`dissipation=False`; tableau, amplification, phase-area, nonlinear Kepler-form
and long-run energy tests support that decision.

The first fresh full run exposed four further failures:

| Failure | Measured default-solve noise | Resolution |
|---|---:|---|
| FSAL equality, KenCarp5 / oscillator | 1.63e-11 | Strict stage solve |
| FSAL equality, ESDIRK5(4)I8L[2]SA / oscillator | 1.20e-10 | Strict stage solve |
| FSAL equality, ESDIRK6(5)9L[2]SA / oscillator | 1.50e-11 | Strict stage solve |
| Linear phase area, SSPSDIRK2 | 7.36e-7 | Strict stage solve |

Those exact-method checks now use `tol=gmres_tol=newton_tol=1e-12,
fd_eps=1e-6`. Equality and geometry tolerances are unchanged. Default-solve
reuse order, cost, nonlinear geometry and energy tests remain active.

## Environment and full result

CPU, Linux x86_64; Python 3.13.14, PyTorch 2.13.0+cu130, NumPy 2.5.2,
pytest 9.1.1. CUDA is unavailable on this host; the gate runner also disables
CUDA explicitly. Numerical worker threads are capped at one.
This local run does not claim execution of the other CI Python versions.

| Run | Result | Wall time |
|---|---|---:|
| First full sweep, eight default-scheduled workers | 6,320 passed, 605 skipped, 4 failed | 567.87s |
| Focused corrected FSAL/geometry sweep, four workers | 247 passed, 8 skipped | 53.72s |
| Final full gate, eight work-stealing workers | **6,459 passed, 605 skipped, zero failures** | **270.23s** |
| Docusaurus production build | Passed | Client compile 8.21s on the initial build |

The two full sweeps differ in added coverage as well as scheduling and numerical
fixes; their timing difference is not an isolated scheduling benchmark.
Skips retain explicit special-RHS, stage-count, precision-floor, geometric and
optional-backend reasons. The final gate contains 7,064 collected cases.

## Shard balance and reproduction

The initial full sweep's aggregate testcase durations were 563.5s for geometry,
491.3s for convergence, 307.6s for DIRK, 255.8s for TVD and 158.1s for reuse.
These aggregates informed moving geometry beside the cheap specialized families
and reuse beside core/solvers. All six shards use two work-stealing workers.
Three shard jobs ran concurrently on this local host (two workers per job),
so these process wall times include shared-host contention; they are not GitHub
runner estimates. Every shard passed:

| Shard | Marker expression | Passed | Skipped | Process wall time |
|---|---|---:|---:|---:|
| shard-1 | `core or solvers or rhs or prrk or reuse` | 3,191 | 536 | 136.08s |
| shard-2 | `dirk or fullyimplicit or newmark or ark` | 540 | 0 | 166.23s |
| shard-3 | `bdf or am or multistep or imex` | 198 | 0 | 34.51s |
| shard-4 | `rkc or rosenbrock or exponential or geometry` | 477 | 33 | 301.04s |
| shard-5 | `convergence or gradients` | 1,570 | 36 | 267.71s |
| shard-6 | `stability` | 483 | 0 | 140.42s |

The shard JUnit case IDs partition the full run **exactly once**: 7,064 cases,
6,459 passed, 605 skipped, zero failures/errors. Total local shard sweep wall
time was 403.82s. The source/configuration fingerprint in the
shard report is `9e47854848f42ae5c6ca1d3de8e15419f2f3f0cebb59175ca55267a4e3044d09`.

The maintained gate runner retains JSON environment metadata, failure names,
process timings, full logs and JUnit reports. Its fingerprint includes numerical
source, tests and CI configuration, including untracked new files.

```bash
conda activate warp
python scripts/validate_test_gate.py --mode full --workers 8 \
  --output-dir .tmp/phase1/final-full
python scripts/validate_test_gate.py --mode shards --workers 2 --parallel-shards 3 \
  --compare-full .tmp/phase1/final-full/full.xml \
  --output-dir .tmp/phase1/final-shards
cd docs
npm run build
```

Raw local evidence is retained under `.tmp/phase1/`: the starting-tree diff,
Kepler/default-vs-strict probes, all 29 new TVD verdicts, focused reruns, both
full sweeps, shard reports and documentation-build output. Existing wave 8
extraction and upstream-oracle files remain under `.tmp/wave8_dirk/`; their
promotion is part of the later repository organization phase.
