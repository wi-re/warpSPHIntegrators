# Package structure and public API refactor

This design records the requested pass **after Phase 8 and before the user-facing
documentation cleanup**. The sole schedule and task list is
[the execution plan](../../REPOSITORY_PLAN.md). These are proposed boundaries,
not implemented import paths or a second catalog-expansion phase.

The user explicitly limits compatibility work for this rewrite to warpSPH and
its sibling projects. Coordinate those consumers rather than preserving every
historical root export and module path for unknown external users. Routine API
references and examples must migrate with the code; the later documentation pass
then makes the completed interface approachable.

## What the initial audit shows

At `30d78a6`, the package contains 225 entries in its explicit root `__all__`
and 386 non-private root attributes. Wildcard imports from fields, integration
and specs expose more names than the declared public set. Numerical drivers,
coefficient data, orchestration, state operations, analysis and adapters sit in
the same directory. Reducing only the length of `__init__.py` would not resolve
the dependency and ownership problems.

Read-only inspection of the local sibling checkouts on 14 October 2026 found:

| Consumer | Python files importing the library | Relevant usage |
|---|---:|---|
| warpSPH | 43 | Enum/lookup configuration; field and system protocols through wildcard imports; solver selection; deep solver/state/spec imports; tableau inspection; CUDA graph clock hooks |
| warpSPHCore | 1 | A spike/profile script imports `getIntegrator`; no integrator import found in its production Python source |
| warpSPHCases | 1 | A copied Sod resume example imports the lookup from `integration` |
| warpSPHPlotting | 0 | No direct Python import found |

These counts cover parseable `.py` files, excluding build/cache directories;
they do not prove absence of notebook, dynamic or transitive usage. The initial
wildcard audit intersects loaded AST names with current root exports, which
identifies candidates rather than resolving ambiguous names imported from other
packages. Repeat and resolve this audit at implementation time.

The important warpSPH contracts are concrete:

- Simulation configuration selects methods through `IntegrationSchemeType` and
  `getIntegrator`, and serializes enum information. Raw scheme functions are not
  the normal method-selection interface in the inspected production code.
- Particle systems use `BaseState`, `BaseIntegrationSystem`, field decorators,
  reference-state access, typed position/component updates and update helpers.
  Their finalize behavior and compatible-energy reruns depend on real stage data.
- Solver tests/benchmarks use fixed-point/Newton solvers, diagnostics, flattening,
  FD/JVP actions, GMRES and explicit state-update helpers.
- Artificial-compressibility code accesses `getButcherTableau`; implicit tests
  access `getDIRKTableau` and its driver. Replace these with an intentional
  advanced coefficient/driver interface where needed, rather than losing them
  when removing root functions.
- Runner/CUDA graph code uses `deferHostTime`; graph plumbing and HDF5 code
  construct/import `IntegrationResult` and `StageResult`. These need explicit
  new homes and consumer checks.

## Proposed package boundaries

Names below are a working design to refine against the completed Phase 8 code.
Each package needs an explicit public surface and private implementation modules.
Avoid both a new miscellaneous `util` bucket and a second flat directory hidden
under a generic name.

| Subpackage | Owns | Boundary |
|---|---|---|
| `state` | Field behaviors, state ownership/flattening, system protocol, typed updates and result contracts | Does not depend on the catalog, concrete methods or run orchestration |
| `schemes` | Identifiers/enums, metadata, registry, validated configuration/factories and coefficient definitions grouped by family | Registry binds definitions to drivers; coefficient definitions do not import the registry |
| `drivers` | Generic explicit/DIRK/block RK, multistep, recurrence, second-order, extrapolation and map stepping implementations | Consumes contracts, coefficients and solvers; does not select methods by importing the registry |
| `solvers` | Fixed-point/Newton/Krylov methods, operator actions, preconditioners, diagnostics and implicit-gradient machinery | Independent of concrete scheme registration |
| `runtime` | Fixed/adaptive runs, accepted history/reuse, events, sampling, checkpoints and clock policy | Uses a common stepping/control contract rather than a growing list of concrete factory classes |
| `wrappers` | Physical flow compositions and algorithm switching in distinct modules | Makes the difference between composing physics and selecting algorithms explicit |
| `adapters` | Tensor ODE, split/semilinear RHS and complete-state-map adapters | Connects user callables/state representations to declared contracts |
| `analysis` | Stability, TVD/geometric diagnostics and inspection helpers | Optional tools; numerical drivers do not depend on analysis or test fixtures |

Shared history/result types must live low enough to avoid runtime/driver import
cycles. Method-specific numeric caches belong to their driver; the runtime owns
when a cache is accepted, rejected or restarted. Direct FunctionMap stepping is a
driver, while adapting a tensor callback to a StateMap is adapter work. A wrapper
must not need to reproduce each child's numerical algorithm or lifecycle policy.

Separating coefficients from execution must be real: the existing generic RK
engine should execute an explicit definition, rather than storing the tableau as
an ad hoc function attribute and using exports as the method database. Preserve
source-qualified distinctions between maps, companions, stage nodes and native
execution behavior. Do not change mathematical formulas while moving them unless
a separately reviewed numerical fix is required.

## Public API direction

Keep a handful of ordinary entry points at the root: a single canonical scheme
lookup, `solve`, `TensorState` and `tensor_rhs` are the working starting point.
Use focused imports from `schemes`, `state` and `solvers` for enums, custom SPH
systems and nonlinear-solver configuration. Final spelling and the precise root
set should be decided together, once, during this pass; avoid creating another
collection of synonymous lookup/preferred/raw-callable paths.

Fixed scheme selection should return a consistently configured callable scheme
record/object with metadata. Enums provide unambiguous identity; friendly names
remain useful for display. Decide the supported string/configuration format and
validation explicitly. Do not require ordinary users to import one raw function
per method or know which internal driver executes it.

Parameterized factories such as collocation stage selection, generalized-alpha
damping and NDF order configuration remain legitimate public capabilities. Give
them a coherent home under the scheme/family interface rather than hiding all
configuration behind unrelated magic keyword arguments. Wrappers and advanced
solver/operator hooks likewise need small documented interfaces of their own.

Remove wildcard imports from the root and supported consumers. Audit accidental
exports as well as `__all__`. Low-level scheme functions, coefficient generators,
cache details and test helpers should not become root API simply because a
registry implementation imports them. Direct driver access needed by a real
consumer can have an explicit advanced interface.

## Migration and verification

Use the closed consumer set to perform a coordinated breaking-import migration;
do not rebuild the entire old facade through permanent aliases. Temporary bridges
are justified only when needed for an identified consumer transition. Preserve
enum numeric identities and existing serialized cases/checkpoints, or provide
and exercise an explicit storage migration. Import compatibility and stored-data
compatibility are different decisions.

Before moving code, inventory consumer imports/usages and the current numerical,
lifecycle, gradient, history and clock contracts. Then establish the shared
contracts, move coefficient/driver families in reviewable batches, bind the new
registry/facade and migrate consumers. Update repository tests, examples,
notebooks, benchmarks, coefficient generators, disposition paths, documentation
generation and installed-wheel smoke checks with the actual new package layout.

The full numerical suite and source oracles retain their existing tolerances and
coverage. Add focused architectural/export checks where they enforce a meaningful
boundary, rather than merely snapshotting the old large export list. Confirm
portable restarts, adaptive rejection, child-history ownership, stage-aware
quantities/reruns and state/RHS-parameter gradients after migration.

Exercise representative warpSPH system/configuration, tableau-access and
runner/clock paths, including CPU behavior and available device/graph tests.
Record unavailable downstream dependencies or hardware as validation limits;
static import inspection alone is not a completed integration gate. Frontend
physics additions remain separate from adapting existing consumers to the new API.
