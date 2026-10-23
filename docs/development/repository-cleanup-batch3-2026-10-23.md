# Plan retirement and reference migration

Completed 23 October 2026 for cleanup Batch 3. The five specialist plans and five
root redirect files are removed after the [Batch 2 reconciliation](plan-reconciliation-2026-10-23.md)
preserved unique content and migrated ledger evidence. The
[execution plan](../../REPOSITORY_PLAN.md) remains the sole work queue.

## Retired paths and retained destinations

| Removed path | Retained destination |
|---|---|
| `IMPLICIT_ROADMAP.md` and `docs/plans/implicit.md` | [Implicit scope and activation conditions](plan-reconciliation-2026-10-23.md#implicit-roadmap-dispositions); original results remain in the numerical records. |
| `SCHEME_EXPANSION_PLAN.md` and `docs/plans/scheme-expansion.md` | [Wave dispositions](plan-reconciliation-2026-10-23.md#expansion-inventory-dispositions), [source inventory](source-candidate-inventory.md) and the [method ledger](../plans/method-dispositions.md). |
| `SPLITTING_PLAN.md` and `docs/plans/splitting.md` | [Maintained splitting design](operator-splitting-design.md), with application assumptions and deferred projects. |
| `JFNK_PLAN.md` and `docs/plans/wcsph-jfnk.md` | [Downstream acoustic handoff](../research/downstream/wcsph-jfnk-design.md) and [dated experiments](../research/downstream/wcsph-jfnk-2026-08-24-to-09-04.md). |
| `docs/plans/package-api-refactor.md` | [Implemented architecture rationale](architecture.md#package-design-rationale), [migration guide](../docs/migration.md) and [initial consumer audit](package-refactor-validation.md#initial-consumer-audit). |
| `NOTES.md` | [Numerical measurement index](../research/integrators/notes-index.md); its 52 explicit historical anchors are preserved there. |

The [migration map](migration-map.json) contains 56 direct destinations. Both
original and intermediate plan filenames resolve to retained pages/sections;
no destination requires another retired redirect. Repository validation now
checks migration destinations as well as the registered/nonregistered ledger.
Historical filenames remain as map keys and in dated removal/audit descriptions.
They are not live navigation or evidence targets.

## References and indexes

Markdown links and anchors, notebook source references, Python comments/docstrings
and historical experiment citations now identify retained records. Dated audits
keep their historical findings; their links lead to current retained material.
The developer index groups current references, milestones, family validation and
browser/consumer records, including the later backend validations. Research and
plan indexes explain ownership without reproducing a schedule.

Notebook source references changed without changing code statements, outputs,
execution counts or metadata. Old phases retain their historical meaning; the
notebook reading lists now distinguish dated JFNK evidence from current package
and downstream work.

The two promoted coefficient tools with old references received comment-only
updates. Their maintained hashes and maintenance notes were refreshed in the
promotion manifest; original scratch paths and original hashes remain intact.
Coefficient/oracle data and numerical code were preserved.

## Sibling repository audit

Read-only inspection of tracked Python, notebook, Markdown and JSON sources found:

| Repository | Files with historical references | Matching lines | Navigable Markdown links to removed files |
|---|---:|---:|---:|
| warpSPH | 24 | 93 | 0 |
| warpSPHCore | 4 | 33 | 0 |
| warpSPHCases | 3 | 20 | 0 |
| warpSPHPlotting | 0 | 0 | 0 |

These are comments, docstrings and historical prose. The corresponding Python
and notebook code contains no executable filename string requiring these removed
files. The scan excludes unrelated filenames such as TGV_NOTES. Historical
citations in the owning repositories can be translated through the migration
map when those documents are next maintained. No sibling working tree was edited;
this is reference inspection, not a downstream physics or import qualification.

## Preservation and validation

The [migration evidence](evidence/repository-cleanup-batch3-migrations.json)
records retired paths, per-file hashes and sibling references. Checks establish:

- All ten retired files are absent; all 56 migration destinations resolve.
- All 53 Python files receiving reference edits preserve their executable AST
  after removing only module/class/function docstrings. Statements, imports,
  constants and control flow remain identical to the start of this batch.
- All three notebooks preserve code statements, outputs and metadata. The Julia
  reference preserves its executable lines; only a comment changes.
- All 52 numbered notes aliases survive in the retained index.
- All 32 tracked numerical JSON fixtures/oracles in the inspected test,
  coefficient and browser directories remain byte-identical to HEAD.
- The method ledger is unchanged from Batch 2, including the 60 migrated
  evidence fields, all source qualifications and candidate statuses.

Final validation passes:

- [Full CPU gate](evidence/repository-cleanup-batch3-2026-10-23-full.json):
  **8,807 passed, 639 skipped, 157 warnings**, eight workers, one numerical thread
  each, 343.1 seconds including subprocess overhead. The final runtime/test/
  configuration fingerprint matches the report.
- Repository links/imports, registered/nonregistered dispositions, all migration
  destinations and promoted-source hashes pass; generated catalog is current.
- All 57 published Python blocks execute; 12 caller-defined sketches stay labeled.
- Browser exports pass for 280 configurations, 420 registered Python fixtures,
  140 parameterized/structured fixtures and 1,872 history endpoints.
- Browser syntax and CPU checks pass: 33 tests pass, five optional Dawn checks
  skip. The production documentation build and `git diff --check` pass.

Reproduction from the repository root, using the documented Python environment:

```bash
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/repository-cleanup-batch3-2026-10-23/gate
python scripts/validate_repository.py
python scripts/generate_catalog.py --check
python scripts/validate_doc_examples.py
python web/tools/export_catalog.py --check
python web/tools/export_reference.py --check
npm --prefix web run check
npm --prefix web test
npm --prefix docs run build
git diff --check
```

Installed development metadata still reports 0.5.0 against source 0.7.0.
CUDA/Dawn execution, a new wheel and consumer restart qualification remain
outside this documentation batch. No consumer implementation work was staged
or committed. Contributor/release guidance and integration qualification remain
in cleanup Batch 4.
