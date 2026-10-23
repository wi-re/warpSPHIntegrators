# Final repository cleanup validation

Cleanup Batch 5, 23 October 2026. All five approved cleanup batches are complete.
The [execution plan](../../REPOSITORY_PLAN.md) owns the current queue; the next
planned project is a measured backend performance baseline. Completed schedules
are retired, retained designs and numerical evidence have maintained homes, and
the method ledger preserves its source qualifications and deferred projects.

The [validation report](evidence/repository-cleanup-batch5-2026-10-23.json) retains
commands, timings, environment, input hashes, preservation results and limits.
The [wheel report](evidence/repository-cleanup-batch5-wheel.json) records a fresh
0.7.0 artifact installed outside the checkout. Source, editable metadata and
wheel version agree.

## Final checks

| Check | Result |
|---|---|
| Repository links, anchors, imports and provenance | All local destinations and import bindings resolve; 142 registered and 242 other dispositions, 56 direct migration destinations and promoted-source hashes pass. |
| Validator, categories and consumer tests | 38 passed, including missing evidence files/anchors, malformed and outside-root JSON paths, and consumer continuation/boundary cases. |
| Generated Python catalog | Current; 142 registered schemes. |
| Published examples | 57 Python blocks execute; 12 caller-defined sketches remain labeled. |
| Browser catalog and Python references | 280 configurations, 420 registered run fixtures, 140 parameterized/structured fixtures and 1,872 history endpoints match their sources. |
| Browser syntax and CPU tests | Pass; 33 tests passed and five optional Dawn checks skipped. |
| Production Docusaurus build | Pass, including browser asset generation. The optional Docusaurus update check cannot write its local cache; site generation succeeds. |
| Installed wheel | Pass; current modules, exports, type marker, configured license bytes and runtime smokes, plus all 17 consumer tests. |
| Publishing script and diff | Shell syntax and whitespace checks pass. |

The [Batch 4 full CPU gate](evidence/repository-cleanup-batch4-2026-10-23-full.json)
remains applicable: **8,812 passed, 639 skipped** with eight workers and one
numerical thread each. Batch 5 changes documentation only. Its final source,
tests and configuration SHA-256 is unchanged:

```text
dc6fd9db0276069f2211be35acba6fbe8dfae8db4de453fee412e312c7a87221
```

The full suite is reused by that fingerprint rather than rerun. Current local
validation uses Python 3.13.14, Torch 2.13.0+cu130 on CPU, NumPy 2.5.2,
pytest 9.1.1 and Node 24.21.0. The offline wheel gate shares numerical
dependencies while isolating the installed project and checking import paths.
It does not qualify fresh or minimum dependency resolution, CUDA or Dawn.

## Preservation and remaining ownership

The final preservation audit confirms all ten retired plan/redirect files stay
absent, all 56 migration destinations resolve and all 52 notes aliases survive.
The 53 Python files changed during reference migration retain their executable
ASTs; three notebooks retain their outputs and metadata. All 32 tracked numeric
JSON fixtures/oracles are byte-identical to the committed baseline. The 60
evidence migrations remain the only ledger changes; source fields, statuses and
the 59 awaiting-verification candidates are preserved.

Pre-existing consumer numerical implementations, public factory exports,
original consumer evidence and local hosting association retain their recorded
hashes. Batch 4's five added consumer tests remain qualified. Scratch evidence
and literature remain on disk, and sibling repositories remain unchanged.

Both consumer maps have package-level installed-wheel and in-memory continuation
qualification. Application serialization and coordinated source/frontend commits
retain their [integration boundaries](consumer-map-integration-status-2026-10-23.md).
The source version and passing local gates do not establish publication status.
Cleanup stages or commits no files and performs no push, publication or deployment.

## Reproduction

Use the documented conda `warp` environment and run from the repository root:

```bash
python scripts/validate_repository.py
python scripts/generate_catalog.py --check
python scripts/validate_doc_examples.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_doc_examples.py tests/test_test_gates.py tests/test_conserved_lagrangian.py tests/test_projected_map.py
python web/tools/export_catalog.py --check
python web/tools/export_reference.py --check
npm --prefix web run check
npm --prefix web test
npm --prefix docs run build
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/repository-cleanup-batch5-2026-10-23/wheel
bash -n scripts/publish_pypi.sh
git diff --check
```

The report's `full_gate_reused` and `gate_sha256` fields identify the full CPU
evidence. Reuse requires recomputing `gate_fingerprint()` from
`scripts/validate_test_gate.py` and checking equality; otherwise run the full gate
again. The [contributor guide](../../CONTRIBUTING.md) gives its command and the
separate shard-coverage workflow.
