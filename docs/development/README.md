# Developer index

## Current work and references

- [Contributor workflow and validation](../../CONTRIBUTING.md)
- [Current execution plan](../../REPOSITORY_PLAN.md)
- [Completed work and evidence](completed-work.md)
- [Architecture and package rationale](architecture.md)
- [Detailed protocol examples and galleries](usage-reference.md)
- [Method dispositions and coefficient provenance](../plans/method-dispositions.md)
- [Method inventory and retained designs](../plans/README.md)
- [Specialist plan reconciliation and project triggers](plan-reconciliation-2026-10-23.md)
- [Source candidate inventory](source-candidate-inventory.md)
- [Operator splitting design and application contracts](operator-splitting-design.md)
- [Numerical measurement index](../research/integrators/notes-index.md)
- [Downstream experiment index and acoustic handoff](../research/downstream/README.md)
- [Examples and notebooks](../../examples/README.md)
- [Benchmarks](../../benchmarks/README.md)
- [Coefficient extraction and oracle tools](../../scripts/coefficients/README.md)

## Repository milestones

- [Initial repository audit](repository-audit-2026-10-14.md)
- [Repository survey and cleanup sequence](repository-survey-2026-10-23.md)
- [Plan retirement and reference migration](repository-cleanup-batch3-2026-10-23.md)
- [Contributor and release guidance qualification](repository-cleanup-batch4-2026-10-23.md)
- [Final repository cleanup validation](repository-cleanup-batch5-2026-10-23.md)
- [Phase 1 numerical and CI gates](phase1-validation.md)
- [Phase 2 executable onboarding and catalog](phase2-validation.md)
- [Phase 3 organization and provenance](phase3-validation.md)
- [Phase 4 tensor ODE usability and packaging](phase4-validation.md)
- [Unused fluid-only option removal](fluid-only-removal.md)
- [Phase 5 run control, outputs and restart](phase5-validation.md)
- [Phase 6 stiff startup and initial coupled scale gate](phase6-validation.md)
- [Shifted spatial solver follow-up](spatial-followup-validation.md)
- [Phase 7 known toy splitting consumers](phase7-foundation-validation.md)
- [Phase 8 first three-stage collocation registrations](phase8-wave9-validation.md)
- [0.7 API migration and user documentation](package-refactor-validation.md)
- [README examples, catalog and measurements](readme-validation.md)

## Family and runtime validation

- [Known structural consumer and source audit](structural-demo-validation.md)
- [Generalized-alpha](generalized-alpha-validation.md)
- [Higher-stage Gauss/Radau and Schur spatial backend](higher-collocation-validation.md)
- [Variable-step Adams](variable-adams-validation.md)
- [Adaptive Adams](adaptive-adams-validation.md)
- [MEBDF2 and CNLF2](implicit-extensions-validation.md)
- [NDF](ndf-backend-validation.md)
- [Adaptive NDF](adaptive-ndf-validation.md)
- [RKL/RKG stabilization](stabilized-backend-validation.md)
- [ROCK/SERK/RKMC recurrences](orthogonal-validation.md)
- [Fixed RKC/RKL clock and quadrature correction](rkc-clock-validation.md)
- [Generalized Nyström](nystrom-backend-validation.md)
- [Improved Nyström source audit](irkn-source-audit.md)
- [Corrected improved Nyström history](improved-nystrom-validation.md)
- [Parallel DIRK map on the serial solver](parallel-dirk-validation.md)
- [Full-JVP Rosenbrock](linearized-validation.md)
- [Exponential operator maps](exponential-validation.md)
- [Explicit extrapolation](extrapolation-validation.md)
- [Algorithm switching](algorithm-switch-validation.md)
- [Function maps](function-map-validation.md)
- [Expanded RK screening](expanded-rk-validation.md)
- [Remaining backend decisions and bounded experiments](backend-backlog-validation.md)

## Browser and consumer records

- [Browser laboratory](browser-laboratory-2026-10-23.md)
- [Browser coefficient exports and CFL controls](browser-coefficients-2026-10-23.md)
- [Conserved-Lagrangian implementation](conserved-lagrangian-2026-10-21.md)
- [Consumer-map integration status](consumer-map-integration-status-2026-10-23.md)
- [Consumer-map installed-wheel and continuation qualification](consumer-map-qualification-2026-10-23.md)

The [migration map](migration-map.json) maps both original and intermediate
filenames directly to retained destinations. The numerical index keeps the
numbered NOTES anchors; the retired root redirects are no longer needed for
repository navigation. Dated records retain their original counts and findings.
