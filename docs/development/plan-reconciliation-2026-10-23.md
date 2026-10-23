# Specialist plan reconciliation

Reconciled 23 October 2026 for cleanup Batch 2. The five specialist plans have
completion, conditional-project or downstream dispositions below. Their old
schedules are superseded by [the execution plan](../../REPOSITORY_PLAN.md).
Removal and reference migration were subsequently completed in
[Batch 3](repository-cleanup-batch3-2026-10-23.md); this record preserves why each
remaining proposal is closed, deferred, excluded or assigned elsewhere.

## Implicit roadmap dispositions

| Original item | Disposition | Evidence or activation condition |
|---|---|---|
| Original phases 0–14 and infrastructure | Completed | [Numbered measurements](../research/integrators/notes-index.md), [Phase 1 gates](phase1-validation.md) and [completed-work index](completed-work.md). |
| BDF3–5 startup ceiling | Completed | [Implicit startup gate](phase6-validation.md). The explicit starter remains optional; actual IMEX splits retain their qualified starter. |
| Three-stage Gauss/Radau scaling and registration | Completed for recorded scope | The original failed local-block scale gate was superseded by [shifted spatial ILU](spatial-followup-validation.md), then [registration](phase8-wave9-validation.md) and [higher-stage factories](higher-collocation-validation.md). General SPH assembly is application-owned. |
| Warm implicit first-stage guesses | Reviewed, deferred as an automatic optimization | [Phase 6](phase6-validation.md) found no consistent gain. Activate for a measured regime with reduced true solve work and unchanged lifecycle/convergence. Keep opt-in experiments. |
| Generalized-alpha | Completed within its contract | [Driver validation](generalized-alpha-validation.md); large structural/SPH scaling remains application work. |
| SSPRK54 | Completed | [Wave-1 source and measured properties](../research/integrators/notes-3-21.md). |
| Physical composition | Completed on known toy consumers | [Phase 7](phase7-foundation-validation.md); application and wider recipes are conditional items below. |
| One-step RKN | Completed for fourteen qualified VI/VD maps | [Nyström validation](nystrom-backend-validation.md). The screened RKN4 and corrected IRKN history retain separate outcomes. This does not implement multistep Störmer–Cowell or Gauss–Jackson. |
| BDF6 | Deferred | Activate for an order-six stiff consumer whose modes fit the narrower stability sector, with startup, stability, solve/work and gradient qualification. BDF5 availability alone does not justify it. |
| Lobatto IIIA–IIIB | Deferred | Require a concrete partitioned Hamiltonian problem, q/p force/velocity accessors and partitioned order/geometry/work gates. An additive coupled RK solve is not this interface. |
| HHT-alpha | Deferred | Require a consumer, explicit force/time/parameter convention and order, damping, lifecycle, restart, gradient and work gates. Generalized-alpha does not automatically complete a distinct HHT parameter family. |
| Native Warp gradients | Deferred backend project | Select gradient ownership and buffer/aliasing/lifecycle contracts, then measure a realistic consumer and verify state/parameter derivatives. Existing Torch IFT gradients or Warp RHS actions are not a native integrator backend. |
| Explicit-bootstrap gradient cost | Deferred profiling question | Profile the actual selected BDF/AM starter and differentiated trajectory. The old blanket claim that all starts use DP5 is obsolete; activate an optimization only for a demonstrated startup bottleneck with preserved gradients/order. |

## Expansion inventory dispositions

Wave numbers identify historical groups. A reviewed source failure or deferred
controller is a completed review outcome, not an implemented method.

| Wave | Disposition and remaining boundary |
|---|---|
| 1 plain SSP | Completed registrations; source/measured SSP distinctions retained in the [numbered record](../research/integrators/notes-3-21.md). |
| 2 SSP estimates | Completed qualified pairs. SSPRK932's source estimate remains unqualified; do not add one without normalization and rate gates. |
| 3 SSP history | Completed qualified drivers; preserve their step/history constraints. |
| 4 pRRK | Completed parameterized kappa/dt drivers outside fixed registration. |
| 5 low-storage SSP | Completed source screening/expanded maps; no native storage-saving execution claim. |
| 6 ordinary explicit methods | Completed qualified fixed maps. Anas5/FRK65 remain deferred runtime-parameter projects; SplitEuler is equivalent to combined-RHS Forward Euler. |
| 7 high-order explicit | Completed eleven qualified maps. PFRK87 remains deferred; 59 additional data-package constructors remain awaiting verification. [Retained source inventory](source-candidate-inventory.md) replaces the expansion plan as evidence. DP8/Feagin estimator exceptions remain explicit. |
| 8 DIRK | Completed qualified registrations and property gates; SDIRK22 remains excluded, and ARS222/232 retain public identities despite a shared implicit map. [Source/corrections](../research/integrators/notes-3-28.md). |
| 9 FIRK | Completed fixed s=3 registrations and parameterized Gauss/Radau s=2–9. Full upstream filtered/variable-order controllers remain deferred until source transition, error-budget and equal-error benefit gates. Lobatto is separately deferred. |
| 10 history and AMF | Completed bounded Adams/NDF, MEBDF2 and CNLF2 contracts. Full VCABM/QNDF/Nordsieck policies remain deferred or prototypes individually; AMF is an operator-factorization prototype, not an Adams variant. |
| 11 stabilized/linearized/exponential | Completed qualified recurrences, corrected-time Rosenbrock maps, fourteen exponential maps and generalized-alpha. Source failing configurations and DAE-specific hybrids keep ledger exclusions; AMF operator promotion retains separate gates. |
| 12 mechanics/parallel DIRK | Completed qualified RKN, corrected IRKN and serial PDIRK44 maps. PDIRK is parallel DIRK on first-order ODEs, not a partitioned-mechanics method. Internal stage threading is deferred execution work with parity/cost gates. |
| 13 extrapolation/maps/switching | Completed fixed-target explicit extrapolation, FunctionMap and AlgorithmSwitch. Source adaptive-order/work controllers, implicit extrapolation and RKIP remain deferred/prototypes with full-state, rollback, estimator and cost gates. Source IRKC is excluded from production for measured order failure. |
| 14 matrix and multirate interfaces | Matrix/explicit-multirate tensor experiments remain prototypes; implicit multirate variants remain deferred. Promotion needs an operator/macro-micro protocol, source-variant verification, rollback, derivative ownership and measured benefit. |
| 15 Taylor | Order-3/4 total-JVP experiments remain prototypes; production/adaptive-order Taylor needs supported higher derivatives, controller/order gates and an equal-error cost case. |
| 16 LowStorageRK | Completed all-60 screening: 54 qualified full-stage main maps and 19 estimates. Six main maps and one companion retain separate exclusions/deferrals. Native low-storage remains a no-grad tensor prototype until stage-aware finalization/AD and allocation benefit are qualified. |

Additional declaration inventory is preserved in the [pinned Phase 8 inventory](evidence/phase8-source-inventory.json).
ABDF2/FBDF beyond the implemented bounded histories remain **deferred with source
semantics awaiting verification**; inspect their actual controller/startup and
problem class before assigning equivalence or implementation. The old plan's
broad DAE grouping is not evidence that every BDF-package declaration is a DAE.
Störmer–Cowell/Gauss–Jackson are likewise deferred pending a consumer and a
separate multistep mechanics/history contract.

Stochastic, delay, global-error-control and discrete-event solvers remain excluded
from the ordinary ODE expansion scope. DAE-specific variants need mass-matrix/DAE
contracts. SIMD reimplementations are deferred execution optimizations rather than
new mathematical registrations. These boundaries do not forbid a separately
scoped project with its own problem contract.

The old method-count landing target and one-commit-per-wave schedule are retired.
Current registration/provenance checks and measured CI/TVD costs remain relevant.
Naming must preserve existing identifiers and source distinctions; source names
do not override verified maps or authorize incompatible renames.

## Splitting design dispositions

| Original proposal | Disposition and activation condition |
|---|---|
| A declared parts, additivity and separability checks | Completed local structural probes in the implemented wrapper; they are not global Hamiltonian certificates. Production entropy/energy closure verification belongs to the application. |
| B Lie–Trotter/Strang and lifecycle | Completed parameterized factories with one outer initialize/finalize and declared grouped child data. The original suggested cross-product of fixed registrations is retired. |
| C nonlinear implicit and exact D | Completed JFNK/IFT and dense small-system exact flow; sparse/matrix-free SPH action/assembly remains deferred until an operator contract and measured consumer justify it. |
| C2 pairwise compatible work / symmetric energy projection | Downstream research design, unimplemented; activation gates follow below. |
| D adjacent half-flow merging | Completed fixed/static exact-flow benchmark; transparent runner optimization deferred until staggered/synchronized state, samples, callbacks, variable steps and restart have explicit ownership. |
| D Strang-minus-Lie–Trotter estimate | Deferred pending estimator/controller and accepted-state qualification. |
| E bounded Yoshida4 | Completed qualified static exact-linear D recipe. Suzuki, higher-order/Blanes–Moan, complex/processed recipes and stiffness/error crossover studies remain deferred until a measured application need and symmetry/intermediate-growth gates. |
| F determinant and physical-energy budget | Completed constant-damping/toy measurements. General conformal metadata is deferred until the dimension, state-map and consumer contract is defined. No universal symplectic/dissipation flag follows. |
| Nonautonomous/multiway splits | Deferred pending clock/forcing contracts, cost case, order and rollback gates. |

C2 activation requires per-substep signed-step energy identity, symmetric-map
checks if claimed, source-reference thermal-distribution comparison, branch/AD
qualification and measured force/memory cost. Frozen partitions alone do not
prove reversible coupled physics. Adaptive split control needs an independently
verified error exponent, extra-work accounting, accepted/rejected state and a
known consumer. The [maintained design](operator-splitting-design.md) retains the
closure, geometric, energy-work and merging arguments with these limits.

The old requirement to build PESPH before any package splitting work is retired:
the toy-consumer implementation already completed package scope. Production
PESPH/compSPH/CRKSPH integration remains downstream work without blocking backend
implementation. A geometric submethod does not certify the frontend's physical
closure or compatible-energy update.

## JFNK and refactor dispositions

The [downstream acoustic design](../research/downstream/wcsph-jfnk-design.md)
preserves the Phase C pressure/continuity/EOS derivative contract, Phase D periodic
WCSPH experiment and E2/E3 boundary/surface handoff. A/B/E1–E1.10 are historical
completed experiments. C/D/E2/E3 remain downstream-owned with current status
unverified; their activation and acceptance conditions are in that handoff.
All-operator JVP work and Hessian-based Newton are outside that acoustic scope.
The old exclusions of package FIRK/BDF/IMEX and implicit differentiation are
superseded by implemented library capabilities.

The package/API refactor and its user-documentation pass are completed, including
the declared consumer import/storage/clock checks and their recorded limits.
The [architecture rationale](architecture.md#package-design-rationale),
[migration guide](../docs/migration.md) and
[initial consumer audit and final validation](package-refactor-validation.md)
preserve the unique decisions: meaningful dependency boundaries, structural
extension dispatch, closed-consumer compatibility scope, deliberate root/flat
import removal, advanced coefficient access, enum/checkpoint storage continuity
and separate numerical versus import verification. The old implementation
instructions are retired. The 21 October consumer maps have separate
[installed-wheel and in-memory continuation qualification](consumer-map-qualification-2026-10-23.md).
Their [status record](consumer-map-integration-status-2026-10-23.md) retains
application serialization and source/consumer commit boundaries.

## Ledger reconciliation and validation

All 142 registered and 242 other ledger entries retain their original statuses,
source fields, reasons and implementation/test paths. Exactly 60 evidence fields
move from the expansion plan to the new source inventory, including named
anchors. The 59 extra tableau declarations remain awaiting verification.

All 35 final-backlog candidates match their corresponding ledger entries in
status, reason, revisit condition and source-hash dictionary. These inventories
overlap; do not sum them as distinct implementation projects. The separate
nonregistered local-path check now runs in repository validation for string,
list and null forms. It checks local files/Markdown anchors, rejects malformed or
outside-root paths, and leaves historical `source` citations untouched.

The added tests cover valid heterogeneous entries, missing implementation/test/
evidence files, a removed evidence anchor, malformed field values and paths
outside the checkout. Validation results:

- Focused documentation/repository/category checks: **21 passed**. Ten new cases
  specifically qualify nonregistered path/anchor validation.
- [Full CPU gate](evidence/repository-cleanup-batch2-2026-10-23-full.json):
  **8,807 passed, 639 skipped, 157 warnings**, 331.3 seconds including subprocess
  overhead, eight workers and one numerical thread each. The final runtime/test/
  configuration fingerprint matches the report. No numerical runtime changes or
  category edits were made in this batch.
- Repository links/imports, registered and nonregistered dispositions,
  promoted-source hashes, generated catalog, published examples and the production
  documentation build pass. `git diff --check` passes.
- Byte checks preserve all ten protected pre-existing implementation/test/export/
  consumer-evidence/hosting files. Ledger comparison permits only the 60 evidence
  migrations; registered entries and all other nonregistered fields are unchanged.

Commands, using the documented conda `warp` environment for Python:

```bash
python -m pytest -q tests/test_doc_examples.py tests/test_test_gates.py
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/repository-cleanup-batch2-2026-10-23
python scripts/validate_repository.py
python scripts/generate_catalog.py --check
python scripts/validate_doc_examples.py
npm --prefix docs run build
git diff --check
```

Installed development metadata still reports 0.5.0 while source declares 0.7.0.
This batch does not qualify a fresh wheel, downstream execution or unavailable
CUDA/Dawn hardware. Remaining edits to references, migration-map destinations
and plan-file removal belong to Batch 3. No plans were deleted and no consumer
changes were staged or committed by this batch.
