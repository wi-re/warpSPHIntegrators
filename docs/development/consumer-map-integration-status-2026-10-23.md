# Consumer map integration status

As of 23 October 2026, ConservedLagrangian and ProjectedStep are implemented and
tested in the editable checkout, with coordinated application work described in
the companion warpSPH records. Their package additions remain uncommitted.
Cleanup Batch 1 accounts for these files and qualifications; it leaves their
numerical implementation and consumer working trees unchanged. The
[execution plan](../../REPOSITORY_PLAN.md) owns package scheduling; application
serialization and source/consumer commit coordination remain a separate handoff.

Batch 4 subsequently qualified a fresh installed 0.7.0 wheel and custom-state
in-memory continuation for all three conserved variants and an integrated
projection system. [The qualification record](consumer-map-qualification-2026-10-23.md)
also records explicit portable-save rejection and the constant-only runner
boundary. The source additions remain uncommitted; full application restart
serialization is not inferred from the package tests. Batch 5 repeats the
installed-wheel and focused checks in the [final cleanup validation](repository-cleanup-batch5-2026-10-23.md).

## Package files and shared edits

Paths below are relative to the repository root. The inventory records the
pre-survey implementation; Batch 4 adds five continuation/boundary cases to the
two existing test modules without changing numerical drivers.

| Work | Files | Status and contract |
|---|---|---|
| Conserved Lagrangian | `src/warpSPHIntegrators/schemes/families/conserved_lagrangian.py`, `src/warpSPHIntegrators/drivers/mechanics/conserved_lagrangian.py` | Untracked implementation files. Hancock, historical SSPRK and KDK use tagged position/conserved fields plus `conserved_velocity(Q)` and `wrap_positions(x)` system hooks. Spatial physics and compatible corrections remain system-owned. |
| Conserved-Lagrangian evidence | `tests/test_conserved_lagrangian.py`, `docs/development/conserved-lagrangian-2026-10-21.md` | Untracked test/record files. The original seven cases cover maps, lifecycle/clocks, ownership, passive slots, invalid contracts and gradients; three new cases qualify run continuation and the portable-save boundary. |
| Projection map | `src/warpSPHIntegrators/schemes/families/projected.py`, `src/warpSPHIntegrators/drivers/mechanics/projection.py` | Untracked implementation files. A `projected_map` system supplies one complete map and stage-state finalization; the auxiliary update is not integrated again. Kick/drift masking and shift timing remain explicit. |
| Projection evidence | `tests/test_projected_map.py` | Untracked test file. The original five cases cover map/lifecycle/time, masking/shift order, ownership, zero-step primitives, gradients and invalid/reuse contracts. Two new cases qualify integrated-state continuation and constant-only runner rejection. |
| Public factories | `src/warpSPHIntegrators/schemes/__init__.py` | Existing tracked edit exports both families through the lazy factory surface. `get_scheme` accepts their instances. Neither adds a fixed registry identifier. |
| Test categorization | `tests/conftest.py` | Existing tracked edit places both modules in `core`. Category and shard checks pass. |
| Root-plan additions | `REPOSITORY_PLAN.md` | The two existing 21 October paragraphs are reconciled into this record, the completion index and the current queue. Their consumer ownership and numerical qualifications are preserved. |

Both factories advertise no universal ODE order or error estimate. They reject
first-stage reuse. The conserved SSPRK position predictor deliberately differs
from ordinary Heun on position/conserved variables. Pressure-model choices and
projection spatial ordering belong to the frontend.

## Downstream evidence and limits

The [local conserved-Lagrangian record](conserved-lagrangian-2026-10-21.md)
retains the original package gate and consumer contract. Companion evidence is
available in the sibling checkout at:

- `warpSPH/docs/meshless_gizmo_validation/ARCHITECTURE_2026-10-21.md`
- `warpSPH/docs/meshless_gizmo_validation/PERFORMANCE_2026-10-21.md`
- `warpSPH/docs/incompressible/QUALIFICATION_2026-10-21.md`

The meshless records assign time staging to this package, preserve the historical
moving-position predictor, and retain measured parity/performance qualifications.
Their later combined frontend/Core cleanup has bounded numerical differences;
the earlier package-driver move is recorded as bitwise. Importing a developed-flow
snapshot for timing does not qualify a restart codec.

The incompressible record describes ProjectedStep routing and single-evaluation
source comparisons, including bitwise parity and unchanged pressure-model
defaults. It also preserves reproduced frontend failures; matching the old
result does not imply a passing physical scenario. These are dated downstream
results, inspected as evidence rather than rerun by this documentation batch.

Portable checkpoint files currently require plain TensorState and tensor-only
updates. These consumers use custom systems; application-owned serialization and
their intended in-memory continuation must be qualified explicitly. The
15 October installed-wheel record predates both families. The later Batch 4
wheel/continuation checks close that package qualification gap; application
restart serialization remains an explicit separate contract.

## Intended implementation boundaries

Keep consumer integration separate from documentation retirement:

1. A conserved-Lagrangian implementation commit should contain its two package
   files, tests and existing validation record, plus its export and category edits.
2. A projection implementation commit should contain its two package files,
   tests, a durable package-side qualification record, and its export/category
   edits. Preserve the downstream qualification link and its limits.
3. Review the shared export/category diff by family when preparing those commits.
   Coordinate the matching frontend changes in the owning repository; avoid
   staging unrelated consumer work or `web/.openai/hosting.json` with them.
4. Batch 4 validates packaging and package-level continuation and refreshes
   editable metadata to 0.7.0. Retain the unsupported custom-state portable
   serialization boundary and coordinate actual frontend codec/commit work in
   the owning repositories before claiming complete application integration.

These are proposed commit boundaries; Batch 1 stages and commits none of the
pre-existing implementation work. The repository's local-commit/manual-push
instruction remains in force.

## Batch 1 checks

The two consumer test modules and `tests/test_test_gates.py` pass together:
**17 tests passed**. Lookup returns each supplied factory instance unchanged;
all three conserved modes and the projection map retain bounded metadata. The
fixed registry remains at 142 methods.

The runtime/test/configuration SHA-256 remains
`a2c93827e5161b9ab53dc861bc6596c568b136b70e8598ee877c6241aa70c830`, matching the
[survey full-suite report](evidence/repository-survey-2026-10-23-full.json) with
8,797 passes and 639 skips. This documentation work reuses that unchanged full
gate. Pre-existing implementation, test, export and hosting files are checked
for byte preservation separately from the authorized root-plan rewrite.

Repository link/import/disposition validation, generated-catalog consistency,
`git diff --check` and the production Docusaurus build pass. Reproduce the focused
and documentation checks from the repository root in the documented environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_conserved_lagrangian.py tests/test_projected_map.py tests/test_test_gates.py
python scripts/validate_repository.py
python scripts/generate_catalog.py --check
npm --prefix docs run build
git diff --check
```
