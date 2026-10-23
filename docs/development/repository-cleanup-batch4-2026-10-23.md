# Contributor and release guidance qualification

Cleanup Batch 4, 23 October 2026. Contributor instructions now match the focused
0.7 package structure, source/wheel versions agree, and consumer-map packaging
and continuation boundaries have direct validation. Publication and coordinated
source/consumer commits remain separate from the cleanup checks.

## Contributor and release guidance

CONTRIBUTING now identifies coefficient builders, immutable definitions,
numerical drivers, registry/enums and parameterized family exports. It separates
fixed registration from configurable factories and consumer maps, with the
required lifecycle/state/estimate and restart qualifications. The current dated
validation record replaces the old Phase 2 baseline as contributor guidance.

The browser checks and Python coefficient/history exports join the documented
site workflow. Unreleased changelog entries cover the browser laboratory,
coefficient/CFL follow-ups, consumer maps, plan retirement and packaging fixes.
Current user docs distinguish general FunctionMap from application mechanics
maps; historical numerical records keep their original counts and metadata.

## Editable installation and publishing guard

The local conda `warp` environment had editable distribution metadata 0.5.0 while
the imported source declared 0.7.0. An offline, dependency-preserving editable
reinstall now reports 0.7.0 from both importlib metadata and the package; its
direct URL still identifies this checkout. Python, Torch and NumPy are unchanged.

```bash
python -m pip install --no-index --no-deps --no-build-isolation -e .
python -c "import importlib.metadata as m, warpSPHIntegrators as w; print(w.__file__); assert w.__version__ == m.version('warpSPHIntegrators') == '0.7.0'"
```

The publishing script's version preflight formerly recognized only double-quoted
source, while the actual initializer uses single quotes. It now accepts either.
The extracted preflight passes both quote styles and rejects a mismatched source
version in temporary directories; `bash -n scripts/publish_pypi.sh` passes.
The upload script itself was not run. Validation guidance explicitly distinguishes
its build/upload action from a dry-run check.

## Consumer packaging and restart scope

The [consumer qualification](consumer-map-qualification-2026-10-23.md) records the
fresh installed 0.7.0 wheel, 17 installed consumer cases, exact in-memory
continuation checks and clear custom-state portable-save rejection. The isolated
wheel also checks current module/export locations and license bytes. The
existing numerical implementations remain unchanged.

A constant-only projection system supports direct maps but not solve's
integrated-field contract. Run callbacks accept forwarded history/step options.
Full frontend/HDF5 restart-codec qualification is an application task, not a
claim inferred from these analytic custom-state tests. Intended two-family
commit boundaries remain in the integration status; no unrelated consumer work
is staged by cleanup.

## Optional hosting association

`web/.openai/hosting.json` associates an optional local hosting service project
with `web/dist`. The maintained static build and GitHub Pages workflow do not
read it. Its bytes are preserved on disk and the exact path is now ignored in
Git; no hosting project, connection or deployment is changed. Source hosting
instructions remain in the Git-tracked workflow and browser README.

## Validation

The [wheel report](evidence/repository-cleanup-batch4-wheel.json) passes with a
clean source staging tree and isolated project install, sharing installed
numerical dependencies. The editable consumer/category run passes 22 cases and
the installed consumer run passes 17 cases. The
[environment report](evidence/repository-cleanup-batch4-environment.json) retains
source/import/metadata agreement and preserved implementation/hosting hashes.

- [Full CPU gate](evidence/repository-cleanup-batch4-2026-10-23-full.json):
  **8,812 passed, 639 skipped, 157 warnings**, eight workers, one numerical thread
  each, 341.3 seconds including subprocess overhead. Its final runtime/test/config
  fingerprint matches this batch, and distribution metadata now reports 0.7.0.
- Repository links/imports, registered/nonregistered dispositions, migration
  destinations and promoted-source hashes pass. Generated catalog is current.
- All 57 published Python blocks execute; 12 caller-defined sketches stay labeled.
- Browser export checks pass for 280 configurations, 420 registered Python run
  fixtures, 140 parameterized/structured fixtures and 1,872 history endpoints.
- Browser syntax/CPU checks pass: 33 tests pass, five optional Dawn checks skip.
  Production Docusaurus build, publishing-script shell syntax and `git diff --check`
  pass.

Reproduction uses the contributor commands plus:

```bash
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/repository-cleanup-batch4-2026-10-23/gate
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/repository-cleanup-batch4-2026-10-23/wheel
bash -n scripts/publish_pypi.sh
git check-ignore -v web/.openai/hosting.json
git diff --check
```

No CUDA/Dawn execution, fresh dependency resolution, external publication or
production frontend restart-codec qualification is established by this CPU/offline
validation. Historical records retain their previous metadata and measurements;
the current queue now moves to final cleanup validation in Batch 5.
