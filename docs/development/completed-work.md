# Completed work and evidence

Reconciled 23 October 2026. The library has completed the initial integration
roadmap, execution Phases 1–8, the 0.7 package/API migration and the following
documentation pass. The browser laboratory is implemented. This index records
the accepted scope and its evidence; [the execution plan](../../REPOSITORY_PLAN.md)
owns current scheduling. Consumer maps implemented in the working tree have
separate integration limits below.

## Library milestones

| Completion | Milestone | Accepted result and evidence |
|---|---|---|
| September 2026 | Original implicit/stiff roadmap | Original phases 0–14 landed. The [numbered measurement index](../research/integrators/notes-index.md) retains solver, gradient, history, adaptivity and structured-RHS records. Later extensions have separate dispositions. |
| 14 October 2026 | Execution Phase 1 | Reliable numerical/CI gates and the 137-method wave-8 baseline; [validation](phase1-validation.md). |
| 14 October 2026 | Execution Phase 2 | Executable onboarding, generated catalog and documentation gates; [validation](phase2-validation.md). |
| 14 October 2026 | Execution Phase 3 | Indexed research, specialist designs, provenance, examples, benchmarks and contributor workflow; [migration and validation](phase3-validation.md). Its temporary redirects were subsequently retired in cleanup Batch 3. |
| 14 October 2026 | Execution Phase 4 | Tensor ODE adapter, discovery/options and installed-wheel validation; [validation](phase4-validation.md). Particle masking remains application-owned after the [fluid-only removal audit](fluid-only-removal.md). |
| 14 October 2026 | Execution Phase 5 | Transactional fixed/adaptive runs, sampling, supported events and restart; [validation](phase5-validation.md). |
| 14 October 2026 | Execution Phase 6 and spatial follow-up | Implicit BDF startup and measured coupled scaling. The [initial gate](phase6-validation.md) failed mesh-refined diffusion; the [shifted spatial ILU follow-up](spatial-followup-validation.md) cleared the unchanged bar on its recorded cases. General SPH assembly remains application-specific. |
| 14 October 2026 | Execution Phase 7 | Lie–Trotter/Strang, implicit/dense exact dissipative flow and bounded Yoshida4 on known oscillator/Burgers consumers; [validation](phase7-foundation-validation.md). Production SPH compatible-energy claims remain conditional. |
| 14–15 October 2026 | Execution Phase 8 | Reviewed expansion and candidate dispositions; the catalog reaches 142 fixed methods with additional parameterized backends. See [the first three-stage collocation batch](phase8-wave9-validation.md), family records below and [final backlog decisions](backend-backlog-validation.md). Completion covers candidate review, including prototypes, exclusions and deferrals. |
| 15 October 2026 | Package/API migration | Focused packages, four root entry points, stored enum/checkpoint compatibility and supported-consumer CPU checks; [validation](package-refactor-validation.md), [migration guide](../docs/migration.md). The recorded pre-existing downstream FD/JVP tolerance failure and unavailable CUDA checks remain limits. |
| 15 October 2026 | User documentation pass | Refreshed overview, selection/family/API/solver/run guides and examples; [validation](package-refactor-validation.md#documentation-work). The [README evidence](readme-validation.md) retains the understood oscillator, generated scheme tables and measured figures. |
| 23 October 2026 | Browser laboratory and follow-ups | Browser CPU/WGSL maps, history, coefficient exports, CFL controls and versioned static assets; [laboratory validation](browser-laboratory-2026-10-23.md), [coefficient/CFL validation](browser-coefficients-2026-10-23.md), [maintained coverage](../../web/README.md). Counts in dated records describe their own snapshots. |
| 23 October 2026 | Repository cleanup Batch 1 | Current queue and completion evidence reconciled; consumer files and integration limits accounted for in [the status record](consumer-map-integration-status-2026-10-23.md). The [survey](repository-survey-2026-10-23.md) retains the approved cleanup sequence. |
| 23 October 2026 | Repository cleanup Batch 2 | [Residual proposals reconciled](plan-reconciliation-2026-10-23.md); unique source/design material retained, 60 evidence references migrated and nonregistered ledger path/anchor validation added before retirement. |
| 23 October 2026 | Repository cleanup Batch 3 | [Five specialist plans and five root redirects removed](repository-cleanup-batch3-2026-10-23.md); reference/index migration completed, 56 direct migration destinations checked and 52 notes aliases retained. |
| 23 October 2026 | Repository cleanup Batch 4 | [Contributor/release guidance and editable metadata refreshed](repository-cleanup-batch4-2026-10-23.md); installed 0.7.0 wheel/license and [consumer continuation boundaries](consumer-map-qualification-2026-10-23.md) qualify package behavior. Source/consumer commits and publication remain separate. |
| 23 October 2026 | Repository cleanup Batch 5 | [Final checks and closeout](repository-cleanup-batch5-2026-10-23.md); repository/examples/catalog/browser/site checks and a fresh installed wheel pass, preservation audit holds, and the unchanged full CPU fingerprint permits gate reuse. All five cleanup batches are complete. |

## Expansion and numerical qualifications

| Capability | Evidence |
|---|---|
| Higher collocation and structural damping | [Gauss/Radau stages 2–9](higher-collocation-validation.md), [known structural consumer](structural-demo-validation.md), [generalized-alpha](generalized-alpha-validation.md). |
| Adams and implicit history extensions | [Variable Adams](variable-adams-validation.md), [adaptive Adams](adaptive-adams-validation.md), [MEBDF2/CNLF2](implicit-extensions-validation.md), [NDF](ndf-backend-validation.md), [adaptive NDF](adaptive-ndf-validation.md). |
| Stabilized explicit recurrences | [RKL/RKG](stabilized-backend-validation.md), [ROCK/SERK/RKMC](orthogonal-validation.md), [corrected fixed RKC/RKL clocks and finalization](rkc-clock-validation.md). |
| Nyström mechanics | [Generalized VI/VD Nyström](nystrom-backend-validation.md), [IRKN source failures](irkn-source-audit.md), [corrected improved-Nyström history](improved-nystrom-validation.md). |
| Linearized and parallel DIRK | [Full-JVP Rosenbrock](linearized-validation.md), [serial execution of PDIRK44](parallel-dirk-validation.md). |
| Exponential operators | [Fourteen qualified exponential maps](exponential-validation.md), with bounded derivative/action contracts. |
| Extrapolation, switching and maps | [Explicit extrapolation](extrapolation-validation.md), [owned algorithm switching](algorithm-switch-validation.md), [complete FunctionMap transitions](function-map-validation.md). |
| Screened low-storage source maps | [ExpandedRK screening](expanded-rk-validation.md): 54 qualified main maps and 19 estimates executed with full-stage storage. Native storage reduction remains separate. |
| Remaining source candidates | [Method ledger](../plans/method-dispositions.md), [individual final-backlog audit](../../scripts/coefficients/final-backlog/audit.json), [bounded prototypes and decisions](backend-backlog-validation.md). Unverified candidates remain unverified. |

## Consumer maps implemented in the working tree

The 21 October conserved-Lagrangian and projected-map work is present as
uncommitted package files, exports and categorized tests. These are specific
consumer contracts outside fixed scheme registration, with no universal ODE
order, adaptive estimate or dense-output claim.

The [conserved-Lagrangian record](conserved-lagrangian-2026-10-21.md) describes
Hancock, historical moving-quadrature SSPRK and KDK through the normal lifecycle.
ProjectedStep performs one complete projected map; masked kick/drift primitives
and pre-drift shift evaluation preserve consumer ordering. Spatial reconstruction,
pressure ordering and physical compatibility corrections stay in the application.

The [integration status](consumer-map-integration-status-2026-10-23.md) accounts
for both implementations, shared exports/category edits, downstream records,
package qualification and separate application serialization/commit work. The
[installed-wheel and continuation checks](consumer-map-qualification-2026-10-23.md)
close the package qualification gap; application integration retains its stated
boundaries.

## Current validation baseline

The [final cleanup validation](repository-cleanup-batch5-2026-10-23.md) reuses the
[Batch 4 full CPU report](evidence/repository-cleanup-batch4-2026-10-23-full.json), which
records **8,812 passed and 639 skipped**, eight workers and one numerical thread
per worker. Ten validator and five consumer continuation/boundary cases extend the [survey baseline](evidence/repository-survey-2026-10-23-full.json)
without changing numerical runtime or category assignments. Its fingerprint
matches the final runtime/tests/configuration after packaging/test qualification.
Batch 5 reruns examples, catalogs, browser CPU checks and the production
site build. Editable metadata agrees with source version 0.7.0; earlier
reports preserve their historical 0.5.0 metadata. The
[current wheel](evidence/repository-cleanup-batch5-wheel.json) includes both
consumer maps and continuation checks, extending the older 15 October scope.

The [architecture rationale](architecture.md#package-design-rationale) and
[initial consumer audit](package-refactor-validation.md#initial-consumer-audit)
retain the refactor's unique decisions. The [specialist-plan reconciliation](plan-reconciliation-2026-10-23.md)
assigns remaining scope and triggers. The [original repository audit](repository-audit-2026-10-14.md) and dated numerical
records retain earlier counts, failures and decisions. Their measurements remain
historical evidence as current code and schedules advance.
