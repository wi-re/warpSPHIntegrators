# Phase 3 consolidation and validation — 14 October 2026

Phase 3 preserves the numerical package paths and coefficients while organizing
plans, history, examples, plots and benchmark drivers. The migration map records
old/new paths; compatibility pointers preserve externally known plan locations
and numbered NOTES references.

## Migration scope

- One current task list in the root execution plan; specialist plans retain
  designs/source inventories with reconciled statuses.
- Current architecture extracted from the dated audit. NOTES 3.0–3.28 and
  reproduction commands have separate records, indexed with original identifiers.
- Completed downstream JFNK measurements separated from its open C/D/E2/E3
  design; downstream results are explicitly historical, not newly verified.
- Concise README, detailed protocol examples/galleries under developer docs,
  contributor environment/registration/provenance/release and multicore commands.
- Notebooks, benchmark/gallery drivers and durable assets relocated with portable
  output paths. Notebook dependencies include the widget backend used by the
  comparison notebook.
- Complete registered-method disposition ledger, ExplicitTableaus enumeration,
  parameterized-method deferrals, equivalences and estimator exceptions. Direct
  source inspection corrected the historical SplitEuler/Symplectic Euler claim:
  the source advances Forward Euler on `f1+f2`. SDIRK22 exclusion distinguishes
  its Trapezoid-returning algorithm constructor from its existing compact tableau.
- Julia environment, coefficient extraction/oracle tools, immutable extracted
  data and small Phase 1–2 run reports promoted before ignoring scratch output.
  Local scratch remains on disk.

## Validation

- Repository validation: **84 Markdown documents, 3 notebooks, 785 local links**,
  all 137 registered dispositions and promoted-file hashes pass. Deliberately
  broken-link and notebook-syntax probes were rejected, then removed.
- Direct source validation: **30 runnable Python blocks executed**, 13
  caller-dependent sketches labeled; generated catalog check passes.
- Core category: **2,837 passed, 149 skipped**, 72 warnings, **26.56 seconds**
  with four work-stealing CPU workers and one numerical-library thread per worker.
- Final catalog/example/gate checks: **286 passed, 4 skipped**, 18 warnings,
  **1.90 seconds**. The source package and numerical tests were unchanged, so
  the previous Phase 2 full numerical gate remains the baseline.
- Strict Docusaurus production build passes after navigation/path changes.
- All **26 plots remain byte-identical** to the previous commit. All 12 benchmark
  drivers import successfully from outside the checkout; the relocated RK4 reuse
  smoke run matches its predicted convergence order. All three notebook setup
  cells locate the checkout/assets from their new directories.
- Python maintenance/benchmark/example scripts compile. Promoted wave-6
  reconstruction and wave-7 assembly checks pass; all 11 wave-7 one-step outputs
  match the independent recorded Julia results at the retained 1e-12 tolerance.
  Wave-8 coefficient comparison/literal generation runs with documented source-node
  corrections; its exploratory source-discrepancy output is not presented as a
  fresh whole-family numerical gate.

Reproduction (use the documented environment):

```bash
python scripts/validate_repository.py
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q -n 4 --dist=worksteal -m core
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_catalog.py tests/test_doc_examples.py tests/test_test_gates.py
python -m compileall -q scripts benchmarks examples
cd docs
npm run build
```

Environment: Python 3.13.14, Torch 2.13.0+cu130 on CPU, NumPy 2.5.2,
pytest 9.1.1, conda environment `warp`; site Node 24.21.0. The build's local
Docusaurus update-check cache warning does not affect its successful compilation.
Notebook syntax and paths are checked; stored notebook outputs are historical.
The downstream SPH/GPU notebooks are not claimed as freshly executed experiments.

## Compatibility and source provenance

Root specialist pointers and NOTES anchor aliases are retained until the next
documentation migration review. New links target maintained paths directly.
That review occurred on 23 October: [cleanup Batch 3](repository-cleanup-batch3-2026-10-23.md)
removed the temporary root files after migrating references. Numbered aliases
remain in the numerical measurement index. The Phase 3 results above describe
the original 14 October migration.
Site developer navigation uses repository URLs; plots under `docs/assets` are
repository assets, not implicitly served as site static files.

The ledger distinguishes implementation from source verification. Expansion
waves pin OrdinaryDiffEq 7.8.1; legacy methods with no captured primary-source
revision explicitly retain that qualification. Names in the extra ExplicitTableaus
inventory do not establish equivalence or justify registration.

Machine-readable retained evidence: [Phase 1 full](evidence/phase1-full.json),
[Phase 1 shards](evidence/phase1-shards.json), [Phase 2 full](evidence/phase2-full.json).
