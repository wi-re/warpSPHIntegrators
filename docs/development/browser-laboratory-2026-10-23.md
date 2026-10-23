# Browser laboratory validation — 2026-10-23

The standalone `web/` app exports the actual registered/parameterized coefficient
sources, implements numeric CPU/WGSL drivers, and is copied into the Docusaurus
`laboratory/` route by the documentation build. No Python core files were changed.
Existing unrelated conserved-Lagrangian/projection work in the checkout was retained.

Coverage: 142 registered numeric maps, 131 extra parameterized configurations,
17 toy systems; 242 configurations generate WGSL for the small oscillator layout.
The in-app coverage report and `web/README.md` explicitly list the unported
controllers, IRKN history, runtime/lifecycle/AD contracts and GPU budgets.

Validation performed:

- 553 independent Python driver fixtures: 420 registered warm-history runs over
  oscillator/forced/damped problems, plus 133 parameterized/structured exponential
  runs. Browser comparisons pass a 1e-9 state-norm bound. The registered fixtures'
  measured maximum difference was 3.16e-11 before the uniform-clock cleanup.
- All 273 configurations execute on the oscillator CPU smoke test. Cross-family
  and non-autonomous refinement recover expected resolved orders. Exact exponential
  flows and high orders stop reporting slopes at their rounding/solve/reference floor.
- Adaptive rejection, analytic Jv/Jᵀv duality and finite-difference comparisons,
  matrix-free Newton–GMRES, stage-level SSP diagnostics, nonlinear Kepler geometry,
  structural high-frequency damping and scaled phi actions pass targeted tests.
- Dawn/Tint compiles all 242 eligible oscillator configurations. Dawn's executing
  Vulkan backend runs more than 340 method/problem combinations and batched
  multistep histories against the float64 browser engine. A null adapter is used
  only for compilation, not execution. This is numerical validation, not a GPU
  speedup benchmark or a broad hardware/browser qualification.
- Desktop browser inspection verified trajectory, convergence, Kepler geometry,
  SSP/TVD, stability, analytic adjoints, matrix-free viscous Burgers, coverage,
  copied setup URLs and a JSON download with four methods/301 samples each.
  The in-app browser has no adapter in this environment; its CPU fallback was
  verified. GPU execution was validated separately with Dawn/Vulkan.
- JavaScript syntax checks, source coefficient/reference consistency checks,
  repository path/import validation and the production Docusaurus build pass.

Useful commands (from repository root):

```bash
npm --prefix web run check
npm --prefix web test
python web/tools/export_catalog.py --check
python web/tools/export_reference.py --check
python scripts/validate_repository.py
npm --prefix docs run build
```

For executable WGSL checks, supply `DAWN_NODE_MODULE` as described in the browser
README. Without it, the optional compiler/execution tests are explicitly skipped.
The documentation CI runs browser syntax/CPU gates and checks the Python exports.

## Follow-up: descriptions, playback and reliable defaults

- Added per-problem descriptions and initial-state/reference information,
  synchronized progressive reveal for time histories and error/energy plots,
  and linked scheme highlighting from the legend, selected list and results.
- Stability maps now sample 512 × 512 by default, adjustable to 2048 × 2048.
  Source-derived stability polynomials/rational functions are compiled once per
  map; tests compare them with independent direct stage solves.
- Fixed structural GPU Newmark/generalized-alpha failures caused by cancellation
  when evaluating the two-mass stiffness matrix in float32. The known orthogonal
  modal basis gives the equivalent direct linear acceleration solve. Long runs
  (T=12, h=0.1/0.05/0.02/0.01 and rho-infinity=0/0.5/1) now pass against CPU
  trajectories, with energy preservation checked for nondissipative settings.
- Presets reset all solver options. Every one of the 17 problem defaults and six
  demos completes in CPU regression checks. Auto retries failed GPU trajectories
  on CPU and reports the reason; explicit GPU mode retains the failure.
- All 25 tests pass with Dawn/Vulkan enabled, including the full shader/compiler
  suite and structural regression.

## Follow-up: categorical method browser

Replaced the flat native method dropdown with a modal catalog: eight numerical
categories, exact-family and order filters, full-text search and method cards
showing stages/embedded metadata and compatibility reasons. All 273 catalog
configurations remain browsable. Selected methods open the same dialog in
replacement mode, preserving comparison slots and rejecting duplicates. Hover
and keyboard focus synchronize the right-hand inspector with path highlighting.
Browser checks cover adding, replacing, filtering, incompatible cards, keyboard
close/focus return and inspector synchronization; existing CPU checks pass.

## GitHub Pages and history-family expansion

Published the laboratory through the existing documentation workflow at
https://fluids.dev/warpSPHIntegrators/laboratory/. The first deployment is commit
`2d85c5996c27b7ff97df3681d7a8fb3004559df5`; build/deploy workflow 39926276921 passed.

Added seven source-derived maps: fixed-target NDF1–5 and corrected IRKN3/4.
NDF preserves the Python DIRK startup, backward-difference warm update, bounded
R(ratio)U resampling, invalidation/restart rules and quasi-constant CPU adaptive
defect policy. IRKN preserves corrected physical startup nodes and cached warm
acceleration recurrences. Accepted trial memory is never mutated. Both families
have real fixed-step WGSL drivers. The canonical reduced two-form diagnostic
rejects these hidden-history methods. Variable-order AdaptiveNDF remains separate.

Python parity covers 1,872 individual history endpoints for oscillator, forced
motion, Kepler and damped mechanics, with fixed, resampled, rapidly changed and
explicitly restarted histories. CPU checks also cover measured convergence,
contract rejection, adaptive stiff relaxation, transform inversion and state/time
invalidation. The catalog now has 280 configurations, 249 eligible oscillator
shaders. Existing parameterized fixtures increased to 140 (560 total short runs).

Eleven source-declared explicit FSAL tableaus now reuse accepted endpoint rates
on both engines. Tests compare cached/uncached trajectories and counters, ensure
adaptive rejection cannot poison the cache, and invalidate changed state/time.
Implicit FSAL and general lifecycle caches remain explicitly unported.

The expanded suite passes all 32 tests with Dawn/Vulkan enabled: 249 eligible
oscillator shaders compile, the cross-problem execution gate remains green,
and batched long NDF/IRKN histories plus exact explicit-FSAL work counts pass.
Browser inspection confirms all four History methods demo results complete and
the coverage page exposes the new families without classifying variable-order
controllers or general lifecycle contracts as implemented. Production docs build
and clean-checkout Python coefficient/reference checks pass.
