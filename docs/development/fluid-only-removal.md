# Unused integrated-field option — 14 October 2026

The `fluid_only` argument and automatically added metadata were removed from
`integrated`. Particle masking remains the system/RHS's responsibility.
Defaults and other options are keyword-only, so an old positional flag call
raises TypeError instead of silently becoming a field default.

## Consumer audit

Local checkouts inspected: warpSPHIntegrators (`b989f73` before this change),
warpSPH (`8255b6f`) and warpSPHCore (`5baf30e`). These identify checkout baselines;
the searches inspected local files rather than assuming downstream release state.

- warpSPHCore contains no `fluid_only` references or `integrated()` callers.
- The frontend has no callers or metadata consumers. Its only source mention
  is a comment in `systems/weaklyCompressible.py` describing the flag as dead
  metadata. That system already masks RHS updates and restores boundary state.
- An AST scan of Python/example-notebook source found 26 `integrated()` calls
  in this package, 25 in warpSPH and zero in warpSPHCore. None supplies the flag
  or more than one positional argument.
- Frontend `fluidOnly` experiment filters are separate options and do not read
  integrated-field metadata. Historical research mentions remain dated evidence.

No downstream source files were changed. This audit covers the checked local
repositories; it does not claim to enumerate external third-party callers.

## Validation

The removal is covered by regression checks rejecting old keyword/positional
calls, the full CPU numerical gate, direct published examples, generated catalog,
repository links and the production documentation build. Defaults, factory
isolation, field tags and arbitrary caller metadata were also checked directly.

```bash
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/fluid-only-removal
python scripts/validate_repository.py
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
cd docs
npm run build
```

The full gate passed: **6,742 passed, 609 skipped**, 139 warnings, **267.77
seconds**, with eight worker processes and one numerical-library thread each.
Environment: Python 3.13.14, Torch 2.13.0+cu130 on CPU, NumPy 2.5.2, pytest 9.1.1.
The two new regression cases pass; the remaining numerical suite stays green.

All 30 published Python blocks, generated catalog, 788 local links across 85
Markdown documents and three notebooks, and the strict Docusaurus build pass.
The full gate's logs and JUnit results are retained locally under
`.tmp/fluid-only-removal`; its [environment/timing report](evidence/fluid-only-removal-full.json)
is tracked with this record.
