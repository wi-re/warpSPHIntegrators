# Repository survey and cleanup proposal

Survey date: 23 October 2026. Baseline: `9bcbe6b` on `main`, including the
pre-existing uncommitted consumer-map work described below. This is a dated
assessment and proposed cleanup sequence; [REPOSITORY_PLAN.md](../../REPOSITORY_PLAN.md)
owns the live task list.

Cleanup Batch 1 was completed on 23 October 2026: the root plan now reflects the
current baseline and active queue, [completed work](completed-work.md) has an
evidence index, and [consumer integration status](consumer-map-integration-status-2026-10-23.md)
accounts for the existing uncommitted maps. Batch 2 then completed the
[specialist-plan reconciliation](plan-reconciliation-2026-10-23.md), preserved
unique content and migrated the 60 ledger evidence references. Batch 3 completed
[plan retirement and reference migration](repository-cleanup-batch3-2026-10-23.md),
including the ten file removals and direct migration map. Batch 4 completed
[contributor/release and consumer qualification](repository-cleanup-batch4-2026-10-23.md),
including metadata refresh. Batch 5 completed
[final validation and closeout](repository-cleanup-batch5-2026-10-23.md).
All five cleanup batches are complete. The original findings and proposed file
dispositions are retained here as the dated basis for that work.

The numerical library has completed its expansion review, package/API migration
and documentation pass. The browser laboratory is also implemented. The main
cleanup is to replace overlapping historical schedules with a short current
queue, preserve their useful evidence and designs, and remove the retired plans
and temporary redirects. Native backend performance work and specific deferred
controllers remain future projects.

## Current repository state

| Area | State and evidence |
|---|---|
| Python package | `pyproject.toml` declares 0.7.0; the root exports exactly `get_scheme`, `solve`, `TensorState`, `tensor_rhs`. State, schemes, coefficients, drivers, solvers, runtime, wrappers, adapters, analysis and experiments have separate ownership. See [architecture](architecture.md) and [migration validation](package-refactor-validation.md). |
| Registered methods | Runtime lookup contains 142 registered schemes across 11 families. Parameterized factories extend the fixed catalog. [Generated catalog](../docs/catalog.md) and [dispositions](../plans/method-dispositions.json) agree. |
| Numerical scope | Explicit/implicit RK, history methods, IMEX, exponential/stabilized methods, mechanics, compositions and maps have numerical and lifecycle gates. Full upstream controller equivalence is limited by the individual dispositions. |
| Runtime | Fixed/adaptive stepping, accepted-history ownership, rejection, sampling, supported events and portable tensor restart are implemented. Supported behavior depends on the selected family. |
| Expansion and refactor | Execution Phases 1–8, the 0.7 API refactor and the following user-documentation pass have completion records. Phase 8 completed candidate review, including justified deferrals and exclusions. See [backend decisions](backend-backlog-validation.md). |
| Browser laboratory | Committed `web/` source supplies CPU numeric drivers for 280 configurations: 142 registered plus 138 additional. There are 249 configurations capable of generating WebGPU kernels for compatible layouts. Browser capabilities and limits are maintained in [its README](../../web/README.md). This browser backend is separate from a native Warp backend for the Python library. |
| Tests and CI | Six exhaustive Python marker shards; main targets Python 3.13, PRs target 3.11–3.13. Wheel validation has its own job. Documentation CI checks examples, catalogs, browser tests and the site build. |
| Release state | The checkout version is 0.7.0 and the changelog retains an Unreleased section. The only local release tag is `v0.6.0`. Local tags and version strings do not establish the currently published PyPI version. |
| Backend limits | Python numerical stepping remains PyTorch based. Native Warp gradient/storage ownership, buffer reuse, graph capture and realistic CPU/CUDA profiling are open. CUDA is unavailable in this survey environment. |

### Existing working changes

At survey start, `main` matched the local `origin/main` tracking ref, with no
commits ahead or behind. This comparison did not fetch the remote.

Three tracked files already had modifications: the execution plan,
`schemes/__init__.py` and `tests/conftest.py`. Seven untracked files add
`ConservedLagrangian`, `ProjectedStep`, their mechanics drivers, two test modules
and the conserved-Lagrangian validation record. An untracked `web/.openai/`
directory contains hosting metadata. Preserve these changes during cleanup.

The consumer maps have implementations and tests in this checkout, but are not
part of the committed baseline. Their existing records describe coordinated
warpSPH adoption. Close their work items only after the package files, public
exports, test categories, consumer evidence and intended commit boundaries are
accounted for. A green editable-checkout test run alone does not establish their
installed-wheel or restart-codec support.

## Planning inconsistencies

1. The execution plan still opens with an update date of 14 October, says
   Phase 8 is underway and describes the catalog as unreleased beyond 0.6.1.
   Later sections declare candidate review, the refactor and documentation
   complete. Historical Phase 6 scaling failure text also survives beside the
   later successful spatial follow-up. These should become dated evidence links.
2. [The specialist-plan index](../plans/README.md) still schedules the package
   refactor and describes early expansion progress. The expansion document has
   an older opening status and a later completed-review conclusion.
3. [The implicit roadmap](plan-reconciliation-2026-10-23.md#implicit-roadmap-dispositions) declares all original phases
   complete but still treats one-step RKN as future work. Its startup-gradient
   discussion also needs qualification against the implemented implicit starters.
4. The splitting document opens with an implementation notice but retains a
   thousand-line research design and realization plan, including old flat module
   paths and statements that composition machinery is missing.
5. The package/API refactor plan says it is implemented, then continues with
   pre-migration instructions. Its useful architecture and compatibility decisions
   already have maintained homes.
6. The four root specialist-plan files and `NOTES.md` are temporary redirect and
   anchor pages. Their own retirement condition is the next documentation
   migration review. This cleanup is a suitable point to remove them after
   migrating references.
7. Current guidance has drifted: CONTRIBUTING still names `integration.py` for
   method registration and quotes a Phase 2 test baseline. The README's latest
   full-suite count predates the refactor and consumer-map records. Browser
   validation records retain earlier coverage counts; those are historical
   measurements and should retain their dates.
8. The local conda environment reports installed distribution metadata version
   0.5.0, while the imported checkout declares 0.7.0 in `pyproject.toml`. Refresh
   the editable installation and verify an isolated wheel before using local
   package metadata as release evidence. The numerical survey imports this
   checkout's `src/warpSPHIntegrators`, not an old installed source tree.

### Dispositions that must survive retirement

The JSON ledger has 142 implemented registered entries and 242 other entries:
121 implemented, 5 equivalent, 15 deferred, 19 excluded, 59 awaiting verification
and 23 prototypes. These are inventory entries, not a count of distinct new
algorithms or a mandatory implementation queue. The separate final-backlog audit
contains 35 reviewed candidates and overlaps this inventory.

**Sixty nonregistered entries cite `docs/plans/scheme-expansion.md` as evidence.**
They comprise the 59 awaiting-verification ExplicitTableaus candidates and
PFRK87. Preserve their source inventory and qualifications before deleting that
plan. The repository validator currently validates registered-method evidence
paths, but does not apply those path checks to `other_dispositions`; ordinary
Markdown link checking will not catch dangling JSON evidence references.

Legacy registered entries can also have unverified primary-source revisions.
Retain that qualification. An implemented local method and a verified upstream
source are separate facts.

## Proposed disposition of each plan

| Current file | Cleanup action | Content that survives |
|---|---|---|
| `REPOSITORY_PLAN.md` | Rewrite as the sole short current queue. Remove completed phase checklists and contradictory progress prose. | Current package status, active tasks with completion criteria, explicit deferred-project links, publishing boundary and a completion index. Preserve and reconcile the two existing consumer-map additions. |
| `IMPLICIT_ROADMAP.md` | Delete the root redirect after reference migration. | Direct links to numerical records, current family docs and remaining project dispositions. |
| `JFNK_PLAN.md` | Delete the root redirect after reference migration. | Direct downstream design/evidence links. |
| `SPLITTING_PLAN.md` | Delete the root redirect after reference migration. | Maintained splitting documentation and retained design rationale. |
| `SCHEME_EXPANSION_PLAN.md` | Delete the root redirect after reference migration. | Method ledger, coefficient provenance and dated expansion evidence. |
| `NOTES.md` | Delete the root redirect after migrating links and source/test commentary. | Keep `docs/research/integrators/notes-index.md`, its numbered anchors and all numerical records. |
| `docs/plans/package-api-refactor.md` | Retire and delete once unique rationale is reconciled. | Architecture, migration guide and package-refactor validation; preserve enum/checkpoint compatibility decisions. |
| `docs/plans/implicit.md` | Retire and delete after extracting remaining conditional items. | Original phase evidence remains in research. BDF6, Lobatto, HHT-alpha, native gradients and measured startup-cost work get explicit future triggers where still relevant. Close obsolete RKN/generalized-alpha/composition reminders. |
| `docs/plans/scheme-expansion.md` | Retire and delete after the ledger and provenance migration. | Preserve the ExplicitTableaus inventory and PFRK87 source/parameter contract in the ledger or a focused coefficient-source reference. Keep the existing candidate audit, oracle data, source pins, estimator exceptions and revisit conditions. |
| `docs/plans/splitting.md` | Extract a concise maintained design reference, then delete the old realization plan. | Keep the separability, dissipative-order, negative-step and compatible-energy arguments in `docs/development/operator-splitting-design.md` where they add detail beyond the user guide. Put production-SPH follow-ons in the downstream ownership summary. |
| `docs/plans/wcsph-jfnk.md` | Extract the unresolved downstream design, then delete it from the library plan directory. | Place source/pressure-JVP requirements and C/D/E2/E3 handoff in `docs/research/downstream/wcsph-jfnk-design.md`. Preserve existing dated A/B/E1 experiment records and mark downstream status as requiring verification. |
| `docs/plans/README.md` | Rewrite as a short guide to the live queue and disposition inventory. | Explain the difference between a scheduled task, deferred candidate, experiment and historical record. |
| `docs/plans/method-dispositions.md` and `.json` | Keep at their current paths and reconcile. | These are durable method/provenance inventories with tooling consumers. Fix stale narrative and retiring evidence paths without changing valid source decisions. |
| Dated audits, phase/family validation and research notes | Keep as dated evidence. | Retain results, failures, qualifications, plots and reproduction commands. Repair moved links without rewriting historical results as current behavior. |

Remove obsolete plans from the tree after extracting unique content. Git history
retains their old wording; a second directory containing verbatim archived plans
would preserve the same ambiguity. Keep the dated validation records that explain
why work was accepted, excluded or deferred.

## Cleanup sequence

### Batch 1 Establish the current baseline

1. Account for the conserved-Lagrangian and projection changes as their own
   implementation work. Preserve the existing diff; keep documentation retirement
   separate from numerical changes and consumer migration.
2. Add a concise completed-work index under `docs/development`, linking the
   existing Phase 1–8, refactor, documentation, consumer-map and browser records.
   Use dates and evidence links rather than copying their full task checklists.
3. Rewrite the root execution plan around the current queue described below.
   State the source version separately from publication status and remove the
   stale 0.6.1 release claim. Preserve the existing local-commit/manual-push
   instruction until superseded by the user.

Done when the root plan has one consistent status and contains only actionable
work or explicit deferred-project pointers. Completed scope has evidence links.

### Batch 2 Reconcile remaining work and evidence

1. Classify every residual task from the five specialist plans as completed,
   active, deferred with a trigger, excluded with a reason, or downstream-owned.
   Match names and source variants before treating two entries as duplicates.
2. Reconcile narrative against the method ledger and final-backlog audit. Keep
   the 59 unverified candidates unverified unless new evidence resolves them.
   Preserve the 60 expansion-plan evidence references at durable destinations.
3. Extract the splitting design and downstream JFNK handoff. Consolidate refactor
   rationale into the existing architecture/migration records. Migrate any unique
   remaining source and mathematical qualifications before removing text.
4. Extend `scripts/validate_repository.py` to check local implementation, test and
   evidence paths for nonregistered entries, accounting for their string/list/null
   forms. Check the new behavior with a missing-evidence case. Preserve historical
   source paths as source citations rather than requiring them to resolve as
   current checkout implementation files.

Done when each old task has one explicit disposition and all retained JSON
evidence targets resolve without depending on a file scheduled for deletion.

### Batch 3 Remove retired plans and redirects

1. Rewrite internal Markdown links, explicit anchors, notebook references,
   source/test commentary and navigation to the retained destinations. Review
   supported sibling-repository references before removing compatibility pages;
   leave their edits to their owning work rather than silently breaking known
   integration references.
2. Keep the migration map as a historical old-to-new record. Add the subsequent
   destinations or a documented second mapping so it no longer sends readers
   from an old path to a newly removed intermediate file.
3. Delete the four root specialist redirects, `NOTES.md`, and the five retired
   specialist plans listed above. Preserve the numerical notes index and aliases.
4. Refresh the developer, research and plan indexes. Include the newer family
   validations and the browser/consumer records currently absent from the main
   developer index.

Done when no live navigation or machine-readable evidence points at a removed
file, and one plan owns scheduling. Historical descriptions of the old layout
can remain explicitly dated.

### Batch 4 Refresh contributor and release guidance

1. Correct registration instructions for `schemes/enums.py`, `schemes/registry.py`,
   definitions, coefficients and family factories. Explain fixed registration
   versus parameterized factories and consumer maps.
2. Replace claims about the latest validation with the dated baseline and
   reproducible command. Add the committed browser/history/coefficient work to
   Unreleased changelog notes; describe uncommitted consumers according to their
   eventual integration status.
   Refresh the stale editable-install metadata and verify the wheel's version
   and import locations against `pyproject.toml` before release validation.
3. Review README, architecture, examples, family docs and site development links
   for current capabilities. Keep current browser counts in their maintained
   source/export surfaces instead of updating every historical validation count.
4. Classify `web/.openai/hosting.json` as intentional deployment metadata or local
   tool state before deciding whether to track or ignore it. Keep generated
   outputs ignored. Optional local disk cleanup can remove reproducible build
   outputs; preserve `.tmp` experiment evidence and `literature/` unless their
   contents have been reviewed and durable inputs/results retained.

Done when a contributor can follow current paths and checks from a clean checkout,
and version, publication and historical measurements are clearly distinguished.

### Batch 5 Validate the reconciled tree

Run repository paths/anchors/imports and disposition checks, generated catalog
checks, published examples, browser source/reference consistency, browser tests
and the Docusaurus production build. Exercise the installed-wheel gate if consumer
integration or packaging changes are included; the survey did not rerun it.
Use targeted validator tests for its new JSON coverage. Run the full numerical
gate when numerical, public-export, lifecycle or test-category changes accompany
the cleanup; documentation-only retirement can reuse an unchanged runtime/test
fingerprint.

Keep this as reviewable local batches. Publishing, pushing and starting a new
numerical implementation project are separate actions.

## Proposed current work queue

| Priority | Work | Completion criterion or activation trigger |
|---|---|---|
| First | Resolve the existing consumer-map integration status | Package exports, both test modules, consumer evidence and commit boundaries are accounted for; required wheel/restart limitations are explicit. |
| First | Complete plan retirement and current guidance | The five cleanup batches above pass; the root queue no longer repeats completed work. |
| Next | Establish a backend performance baseline | Profile realistic field/vector sizes on CPU and available CUDA, separating RHS, clones/updates, Krylov work, synchronization and gradient memory. Retain hardware, versions, tolerances and reproducible measurements. |
| Conditional | Native storage/Warp/graph execution | Select an observed bottleneck and specify buffer aliasing, lifecycle/finalizer stage data and gradient ownership. Demonstrate numerical/state/gradient parity and a measured benefit before expanding implementation. |
| Conditional | Test runtime and CI rebalance | Measure worker scaling and current shard durations; change scheduling only for a demonstrated imbalance, preserving exactly-once coverage and tolerances. |
| Deferred inventory | Upstream controller, operator, multirate, Taylor, low-storage and fitted-method extensions | Use existing individual source dispositions and cost/contract triggers. Candidate review completion does not schedule all remaining ports. |
| Downstream ownership | Production splitting/PESPH, pressure JVP, mDBC, free surface and application scaling | Verify status in warpSPH/warpSPHCore before scheduling there. Package work becomes active only for a specific integration requirement. |

## Survey validation

Validation uses the existing conda `warp` environment: Python 3.13.14 and Torch
2.13.0+cu130 on CPU, with Node 24.21.0 for browser/site checks. Results include
the pre-existing uncommitted package and test files.

- Repository validator: 1,838 integrator import bindings, 132 Markdown documents,
  three notebooks, 1,139 local links and 142 registered dispositions pass before
  adding this survey. Generated catalog is current.
- A separate inspection of 495 local implementation, test and evidence references
  in nonregistered ledger entries found no missing targets in the current tree.
- Published examples: 57 Python blocks execute; 12 caller-defined sketches are
  labeled.
- Browser syntax and tests: 33 pass, five optional Dawn/WebGPU tests skip.
  Export checks pass for all 280 configurations, 420 registered run fixtures,
  140 parameterized/structured fixtures and 1,872 history endpoints.
- Docusaurus production build passes. Its local update-check cache warning does
  not affect compilation.
- Full CPU gate: **8,797 passed, 639 skipped, 157 warnings**, 335.7 seconds for
  the gate subprocess, eight workers and one numerical thread per worker.
  The [retained report](evidence/repository-survey-2026-10-23-full.json) records
  HEAD, the runtime/test/configuration fingerprint, environment and command.
  Its `package` field is the stale installed metadata version 0.5.0, distinct
  from the source version. Raw output and JUnit results remain under
  `.tmp/repository-survey-2026-10-23/`.

After adding this survey and its developer-index link, the repository validator
passes for 133 Markdown documents and 1,150 local links. The runtime/tests are
unchanged by these documentation additions.

CUDA, Dawn execution, downstream integration, a fresh installed wheel and external
package publication status are outside these fresh survey checks. The existing
dated records remain the evidence for their earlier qualifications.
