# Moving to the 0.7 API

Version 0.7 separates scheme configuration, coefficient definitions, numerical
execution and run control. The ordinary root API is now `get_scheme`, `solve`,
`TensorState` and `tensor_rhs`. Select fixed schemes through enums, and import
advanced capabilities from their focused package.

[example:run migration-selection]: #
```python
from warpSPHIntegrators import get_scheme
from warpSPHIntegrators.schemes import ExplicitRK

method = get_scheme(ExplicitRK.tsit5)  # Tsitouras 5(4)
assert method.identifier.name == 'tsit5'
assert method.order == 5
```

| Previous import/use | Current form |
|---|---|
| Root `getIntegrator(...)` | Root `get_scheme(...)` |
| Root raw `RungeKutta4(...)`, `Tsit5(...)`, etc. | `get_scheme(ExplicitRK.rungeKutta4)`, `get_scheme(ExplicitRK.tsit5)` |
| Root/flat-module state helpers | `warpSPHIntegrators.state` |
| Root/flat solver helpers | `warpSPHIntegrators.solvers` |
| Root checkpoint/event helpers | `warpSPHIntegrators.runtime` |
| Root split RHS helpers | `warpSPHIntegrators.adapters` |
| Root physical composition/switching | `warpSPHIntegrators.wrappers` |
| `getIntegrationEnum(choice)` | `get_scheme(choice).identifier` |
| Order-only preferred selectors | Choose an explicit enum; use `filter_schemes` for requirements |
| Flat `butcher`/`dirk` tableau helpers | `warpSPHIntegrators.schemes.coefficients` |
| Flat numerical driver calls | Explicit advanced engines in `warpSPHIntegrators.drivers` |

The old root wildcard and flat-module imports are deliberately removed.
`warpSPH`, Core profiling scripts and Cases examples/notebooks have coordinated
import migrations; new warpSPH code requires integrator 0.7.0. Update those
checkouts together. This prepares a local breaking change and does not publish a
release.

## Stored cases and checkpoints

Enum **numeric values and member names are unchanged**. A narrow `enums` module
remains so enum pickles written with its old module path load as the current
identifier classes. It is a storage bridge, not a replacement for the old root
API. Existing configuration strings retain their exact meanings.

Portable tensor checkpoints keep the same tensor/primitive format and scheme
identifiers. A real adaptive Adams checkpoint written before the refactor is
part of the migration gate. Resupply the RHS and application/solver options when
restarting; arbitrary functions and external resources are not serialized.
Custom simulation objects still use application-owned serialization.

This migration targets the known warpSPH/sibling consumer set. Keep advanced
tableau access, typed finalization/rerun data and deferred clock hooks through
their explicit interfaces; a storage alias is not a promise to restore removed
wildcard imports. The [architecture rationale](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/architecture.md#package-design-rationale)
explains those ownership boundaries and links the original consumer audit.

## Definitions and extensions

A fixed scheme record contains a definition bound to its numerical driver.
Generic explicit, DIRK, coupled and additive RK definitions hold read-only
coefficient arrays. Inspect a fixed RK with `method.definition.tableau`; use the
advanced coefficient constructors when intentionally building your own tableau.

Parameterized configurations remain in `schemes`; their stepping code lives in
`drivers`. New bound methods can implement `state.Stepper` and optional accepted-
step controls. The runner and wrappers recognize this contract rather than a
list of known factory classes. Physical composition still rejects history-bearing
subflows and unsupported negative dissipative steps.

The numerical validation keeps the existing order, stability, gradient,
quantity, rejection, history and clock gates. The refactor does not claim new GPU
graph coverage or equivalence to deferred upstream controllers. See the
[validation record](https://github.com/wi-re/warpSPHIntegrators/blob/main/docs/development/package-refactor-validation.md)
for the exact checks and consumer limits.
