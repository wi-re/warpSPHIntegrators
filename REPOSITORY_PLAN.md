# Repository execution plan

Updated 23 October 2026. This is the sole current task list. The
[completed-work index](docs/development/completed-work.md) owns completion and
evidence links; the [survey and cleanup sequence](docs/development/repository-survey-2026-10-23.md)
records the approved reconciliation scope. Historical schedules in specialist
plans and numerical records are superseded by this queue.

Push commits only when explicitly authorized. The 23 October cleanup handoff
authorizes committing, pushing and observing the resulting GitHub Actions runs.
Package publication is separate.

## Current baseline

The source declares version 0.7.0 and supplies 142 registered schemes plus
parameterized backends. Execution Phases 1–8, the package/API refactor and the
following documentation pass are complete within their recorded scope. Phase 8
completed candidate review, including explicit deferrals and exclusions. The
browser laboratory is implemented. Publication status is not established by the
source version or local tags.

Cleanup Batches 1–5 are complete: the current baseline and
[remaining plan dispositions](docs/development/plan-reconciliation-2026-10-23.md)
are reconciled, completed work has an evidence index, and the existing consumer
additions are accounted for in
[the consumer-map integration status](docs/development/consumer-map-integration-status-2026-10-23.md).
ConservedLagrangian and ProjectedStep pass installed-wheel and custom-state
in-memory continuation checks. Portable custom-state serialization remains an
application contract; source/frontend commit coordination remains a release
handoff with explicit boundaries in the integration status.

The [Batch 3 retirement record](docs/development/repository-cleanup-batch3-2026-10-23.md)
accounts for the removed plans, migrated references, direct migration map and
read-only sibling audit. Numerical records, source qualifications and notes
anchors are retained.

The [final cleanup validation](docs/development/repository-cleanup-batch5-2026-10-23.md)
reuses the passing full CPU gate with an unchanged runtime/test/configuration
fingerprint: 8,812 passed and 639 skipped. Repository, example, catalog, browser,
site and fresh isolated-wheel checks pass. Source, editable metadata and the
qualified wheel agree on 0.7.0. The [Batch 4 record](docs/development/repository-cleanup-batch4-2026-10-23.md)
retains consumer/release qualification, license inclusion and local hosting-state
classification. Cleanup made no numerical-driver changes.

## Current execution queue

### Establish a backend performance baseline

Profile realistic field/vector sizes on CPU and available CUDA, separating RHS,
clone/update traffic, Krylov work, synchronization and gradient memory. Retain
hardware, versions, sizes, tolerances and reproducible artifacts. Use measurements
to select a buffer-backed RK or native Warp prototype; specify buffer aliasing,
finalizer stage data, lifecycle and gradient ownership first. Evaluate fixed-
iteration graph capture separately. Completion requires measured speed/memory
benefit and numerical/state/gradient parity, with device/dtype checks.

### Measure test scaling before changing CI

Measure 1/2/4/8-worker test scaling and expensive trajectories before changing
suite structure; rebalance CI by durations while preserving exactly-once coverage
and numerical tolerances.

## Deferred projects and downstream ownership

The [method ledger](docs/plans/method-dispositions.md) and
[backend decisions](docs/development/backend-backlog-validation.md) retain source
controller, fitted-method, operator, multirate, Taylor and low-storage revisit
conditions. Candidate review completion does not schedule every upstream port.
BDF6, Lobatto, HHT-alpha and startup-gradient cost proposals retain explicit
[activation conditions](docs/development/plan-reconciliation-2026-10-23.md#implicit-roadmap-dispositions).

Production SPH splitting/PESPH, pressure-force JVP, mDBC, surface treatment and
application scaling belong to warpSPH/warpSPHCore. Verify their status in their
own repositories before scheduling; package work needs a specific integration
requirement. Backend projects do not depend on frontend availability.
