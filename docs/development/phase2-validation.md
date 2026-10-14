# Phase 2 validation — 14 October 2026

Phase 1 and wave 8 were landed in `413d4c4`. Phase 2 makes onboarding and catalog
information executable, with a separate commit. The package remains unreleased
beyond version 0.6.1; no changes were pushed or published during this work.

## Delivered behavior

- The complete CPU example is `examples/harmonic_oscillator.py`. The overview
  displays that file verbatim; the generator synchronizes it and the validator
  rejects drift. The five-step README and quick-start tutorials now import the
  helpers they actually use and access real `IntegrationResult` fields.
- `scripts/validate_doc_examples.py` executes 29 published Python blocks across
  README, overview, quick start, API and solver pages, with no injected imports.
  Thirteen explicitly labeled sketches need caller-provided types, operators or
  RHS callbacks. The legacy README validator delegates to this source validator.
- `SchemeMetadata` and derived scheme properties expose stage count, companion
  order, estimate exponent, RHS requirements, optional splits, per-step options
  and dense output. The original 12-field NamedTuple layout is unchanged.
  TR-BDF2's higher-order companion and Gauss-Legendre 2's lower-order companion
  both give exponent 3; benchmark and test controllers consume the public value.
- `scripts/generate_catalog.py` produces the complete 137-entry catalog and site
  data from runtime metadata, plus maintained family counts and overview/README
  summaries. Docusaurus config and landing-page counts read the generated JSON.
  Authored mathematical qualifications remain in the family pages.
- The documentation workflow now gates pull requests with source execution,
  catalog freshness and a production site build. Links and anchors are strict.
  Deployment and Pages/OIDC write permissions are limited to main-branch runs;
  the built artifact is transferred with hidden site files preserved.

## Validation

Same CPU test environment as Phase 1: Python 3.13.14, PyTorch 2.13.0+cu130,
NumPy 2.5.2, pytest 9.1.1, one numerical thread per worker. The full merge gate
uses eight work-stealing workers and disables CUDA. The local result does not
claim execution of the other CI Python versions or of remote GitHub jobs.

| Check | Result |
|---|---|
| Focused catalog, source examples, embedded and adaptive suites | 692 passed, 8 skipped, 18.53s |
| Final full Python gate | **6,740 passed, 609 skipped, zero failures**, 264.00s |
| Isolated-mode standalone script | `t=1.00, x=-0.416147, v=-1.818595, stages=4` |
| Isolated-mode published-source validator | 29 runnable blocks pass; 13 caller-defined sketches labeled |
| Generated catalog check | Current |
| Docusaurus production build | Passes with strict links/anchors |
| Docs CI YAML and deployment conditions | Parsed; PR build and main-only deployment checks pass |

The additional four skips are the catalog's generic real-step checks for the
three stage-count methods and ETD2RK's structured RHS requirement; their
requirements and dedicated numerical suites remain tested. The complete
collection contains 7,349 cases, each assigned to exactly one CI shard by the
collection gate.

Focused regressions reject missing source imports, unclassified Python blocks,
included-source drift, stale catalog files and scheme tuple-layout changes.
The standalone trajectory is checked against the analytic oscillator solution.
Existing embedded-rate tests continue to measure the public estimate exponents.

```bash
conda activate warp
python scripts/generate_catalog.py --check
python -I examples/harmonic_oscillator.py
python -I scripts/validate_doc_examples.py
python scripts/validate_test_gate.py --mode full --workers 8 \
  --output-dir .tmp/phase2/final-full
cd docs
npm run build
```

Raw logs and the full JUnit/environment report are retained under `.tmp/phase2/`.
The initial site build caught incorrect relative links for the overview route;
the final build uses the proper docs routes and passes the strict gate.
