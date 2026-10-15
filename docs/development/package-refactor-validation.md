# Package/API refactor and user documentation pass

The 0.7 import migration implements the requested separation between state,
coefficients/configuration, numerical drivers, solvers, orchestration, adapters,
physical wrappers and numerical analysis. The root exports exactly `get_scheme`,
`solve`, `TensorState` and `tensor_rhs`. This is a local breaking change; no
release, push or remote CI run is part of this pass.

## Boundaries and preserved numerical work

Generic explicit, DIRK, coupled and additive RK use frozen definition records
with owned read-only coefficient arrays. Records retain callable identity
equality/hash behavior, including use as dictionary keys. The registry binds those definitions;
coefficients do not import registry selection or numerical drivers. Sixteen
parameterized configurations delegate actual stepping to dedicated driver
functions. Pure coefficient builders, matrix-function evaluation and stateful
execution helpers have distinct homes. `state` and `solvers` do not depend on run
orchestration. A structural `Stepper` plus optional accepted-control protocols
replaces the growing concrete factory whitelists in solve, switching and physical
composition. History restrictions on physical subflows remain enforced.

[The numerical AST audit](../../scripts/validate_refactor.py) compares **42 actual
numerical functions** against pre-refactor local commit `7d12f0b`, including all
sixteen factory step bodies, the generic RK engines/coefficients, corrected RKC
recurrences and GMRES/FD/JVP/implicit adjoints. Imports/docstrings and only the
outer function name are normalized. Numerical statements, constants, arguments,
calls and control flow match exactly. [The report](evidence/package-numerical-ast.json)
is narrower than a whole-program proof; immutable binding, warnings, dispatch,
imports and ownership are separately tested.

## Stored data and extension checks

Five architecture/migration tests enforce the small root, read-only definitions,
layer dependencies, structural extension dispatch and actual stored data.
All 142 numeric identifiers and names match the baseline. A real pre-refactor
enum pickle resolves through the narrow `enums` storage bridge to the current
class. A real pre-refactor adaptive Adams tensor checkpoint continues to the
same endpoint within 2e-14. Portable identifiers and tensor/primitive format are
unchanged; no blanket compatibility facade is retained.

An unregistered structural method runs through solve, algorithm switching and
physical composition without editing a class list. Extra exponential tests
reconstruct coupled quantities using operator quadrature and check isolated
probe/lifecycle ownership and physical endpoint copied fields.

## Supported consumers

The refreshed audit includes Python and notebook imports in warpSPH, Core,
Cases and Plotting. Wildcard integrator imports are replaced with focused imports;
actual solver/state/tableau/deferred-clock contracts are retained. The new
warpSPH declaration requires integrator >=0.7.0. Existing unrelated Core and
Cases work is preserved; changes inside the previously untracked Cases tree
remain in that working tree rather than staging unrelated case migration work. The tracked
consumer migrations are local commits `6791cf2` in warpSPH and `a89d20e` in
warpSPHCore. Plotting has no direct integrator usage and requires no edit.

[The consumer validator](../../scripts/validate_consumers.py) checks import
bindings, imports actual configuration/system/runner/export modules, builds and
roundtrips a configuration enum, runs the real BaseSystem typed updates and
finalizer on an oscillator with in-memory restart, and checks its deferred tensor
clock on CPU. It uses writable temporary Numba/Warp/Matplotlib caches. Research
notebooks are syntax/import inspected, not executed as fresh GPU experiments.

The real downstream generic DIRK standing-wave comparison passes with both FD
and JVP actions in its declared float32 setup. A separate FD-versus-JVP tolerance
test fails at one of 64 entries: absolute discrepancy .0004009455442428589.
The identical test and discrepancy reproduce with both pre-refactor source
checkouts. It remains a pre-existing downstream tolerance issue, not a new passing
claim or a reason to loosen the test here. Setting global float64 does not turn
that test into a float64 gate: its builder hardcodes float32 and conflicts with
float64 Warp kernels. No CUDA device is available in this environment.

## Documentation work

The overview and quick start lead with known decay and damped-oscillator
behavior. The API reference explains focused imports, explicit enum selection,
coefficient inspection, factory assumptions and the difference between step and
run results. The migration guide documents the deliberate removals and storage
bridge. The README retains its full friendly-name/enum scheme list, explanations,
plots and benchmark data. Family pages add practical introductions and remove
internal phase/wave labels from user explanations; solver guidance distinguishes
physical RHS work from Krylov counters and precision-floor exits. Notebook
introductions explain the understood problem and recorded-output limitations.

All site Python blocks are now included in the runnable-example validator;
caller-dependent sketches remain labeled. The site landing page leads to the
quick start and method selection, and explains implicit solver work without
assuming a universal RHS/Krylov cost identity.

## Final local gates

- **8,785 passed, 639 skipped, 157 warnings**, eight workers and one numerical
  thread each; 334.388 seconds including orchestration. The recorded runtime/test
  fingerprint matches the final source tree. [Full report](evidence/package-full.json).
- **Installed 0.7.0 wheel passes** outside the checkout with isolated Python:
  all modules/exports, exact source-module inventory, typing marker, 135 fixed
  tensor drivers, advanced state, adaptive sampling/events, portable restart,
  stiff bootstrap, preconditioning, splitting and generalized-alpha continuation.
  [Wheel report](evidence/package-wheel.json). The editable development environment
  still reports its old 0.5.0 distribution metadata; the wheel checks the actual
  new version. Building from a fresh staged source tree prevents stale flat
  modules in an old `build/lib` from entering the wheel.
- **42 numerical function AST comparisons** match the pre-refactor baseline.
- **57 published Python blocks execute**, with 12 explicitly labeled sketches.
  Catalog generation and promoted-source manifest checks pass; all 142 current
  disposition constructors resolve. Repository checking resolves 1,801 integrator
  import bindings and validates local documentation/notebook links. Historical
  source revisions/hashes are retained alongside current implementation paths.
- Exponential and orthogonal coefficient export checks pass. The RKC audit reruns
  12 baseline/corrected known-problem refinements with its original Git source
  path preserved.
- The production documentation build passes; rendered landing, overview, quick
  start and migration pages are reviewed for readable examples, navigation,
  equations and tables.
- [Consumer report](evidence/package-consumers.json): 56 Python files, 26 notebooks,
  30 distinct import bindings, eight actual module imports and three behavior
  checks. The two real downstream generic-DIRK comparisons pass; the independently
  reproduced pre-existing FD/JVP tolerance failure and unavailable CUDA checks
  remain the limits described above.

These gates complete the package/API and user documentation passes. Phase 9
backend performance work is separate; bounded prototypes and deferred source
controller contracts retain their explicit dispositions.
