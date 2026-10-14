# Contributing

Use [REPOSITORY_PLAN.md](REPOSITORY_PLAN.md) as the current task list. The
[developer index](docs/development/README.md) links architecture, numerical records
and specialist designs. Historical experiments retain their original dates and
section numbers; they do not define current release behavior.

## Environment

Python 3.11–3.13 are exercised by CI. From the checkout root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Choose the Torch CPU/CUDA build appropriate to your machine before installing the
editable package. The existing local validation environment is conda `warp`;
`conda run -n warp python ...` selects it without activating another shell.
Install `.[examples]` for plotting/notebooks and `.[benchmark]` for plot drivers.
Julia is needed only for the independent coefficient oracles, not normal tests.

## Validation and multicore execution

The maintained runner uses pytest-xdist worker **processes**, work stealing, CPU
execution and one numerical-library thread per worker. Start with a worker count
that fits your CPU and memory; benchmark additional workers before raising it.

```bash
python scripts/validate_test_gate.py --mode full --workers 8
python scripts/validate_test_gate.py --mode shards --workers 2 --parallel-shards 3
```

Logs, JUnit results and environment/timing JSON go under `.tmp/test-gate/`.
To compare shard coverage with a saved full gate, use separate output directories:

```bash
python scripts/validate_test_gate.py --mode full --workers 8 --output-dir .tmp/full
python scripts/validate_test_gate.py --mode shards --workers 2 --parallel-shards 3 \
  --compare-full .tmp/full/full.xml --output-dir .tmp/shards
```

The latest Phase 2 baseline was 6,740 passed / 609 skipped in 264 seconds with
eight workers. This is a local measurement, not a promised CI duration.
Concurrent shard times include shared-host contention.

For a targeted development run, limit nested threads explicitly:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q -n 4 --dist=worksteal -m 'dirk or convergence'
```

| Change | Relevant checks |
|---|---|
| Family driver/tableau | Family category, convergence, reuse and stability; embedded/adaptive/gradient checks where applicable |
| Newton/GMRES/preconditioner | `solvers`, implicit families, convergence and gradients |
| Fields, lifecycle, registry, RHS or run orchestration | Full gate |
| Published examples or catalog | Direct example validation, catalog check and site build |
| Paths/notebooks/repository documents | Repository link/path validation; compile notebook cells; source example and site checks |

Categories are assigned in `tests/conftest.py`; uncategorized test files fail
collection. `.github/test-shards.json` owns the six shard expressions, and each
case must match exactly one shard. CI uses two workers per shard; main runs
Python 3.13, PRs run 3.11, 3.12 and 3.13. Rebalance using measured durations rather
than case counts, preserving the full merge gate.

## Registering a method and recording provenance

1. Inspect the pinned primary source and record its version/revision, file and
   constructor in [the disposition ledger](docs/plans/method-dispositions.md).
   Mark unverified sources explicitly. Do not infer coefficients from a name.
2. Add the coefficients/driver, public exports, `IntegrationSchemeType`, family
   enum and registration in `integration.py`. Public names are compatibility
   contracts; record equivalent maps without silently deleting identifiers.
3. Verify node/row sums, weight normalization, main and companion order,
   stability, lifecycle, reuse and applicable state/parameter gradients. Inspect
   metadata-driven fixtures before adding redundant method-name lists.
4. Record transformations: extra derivative stages, `b_embed = b_main - btilde`,
   corrected source nodes, weight normalization or deliberately omitted estimates.
   A source error estimator is not automatically a valid embedded solution.
5. Update the authored family page, changelog and ledger, then run
   `python scripts/generate_catalog.py`. Commit generated catalog artifacts.
6. Run relevant category checks and the full numerical gate before landing.

[Coefficient oracles](scripts/coefficients/README.md) retain the pinned Julia
environment, extracted data and independent comparisons. New coefficient changes
need evidence tied to the proposed implementation, not just an old oracle run.

## Documentation and migration checks

```bash
python scripts/validate_repository.py
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
cd docs
npm ci
npm run build
```

Runnable Python fences have `[example:run scope]: #` or
`[example:include path/to/source.py]: #`; caller-dependent sketches use
`[example:skip reason]: #`. Included examples must match the saved source verbatim.
The site uses repository URLs for developer/research documents outside `docs/docs`.
Durable plots live in `docs/assets`; benchmark drivers locate them from their own
file, independently of the shell's working directory.

## Release validation

Confirm package/version/changelog consistency, generated catalog and provenance,
full CPU gate and shard partition, example/link checks, and production site build.
Retain environment, commands, timing and numerical caveats in a dated validation
record. Run `python scripts/validate_wheel.py --cpu-torch` to build and test the wheel
in a fresh environment outside the checkout before publishing; the
editable checkout is not evidence of wheel correctness. CI exercises a fresh dependency install. Offline local validation can use
`--reuse-dependencies`: it shares numerical dependencies but still installs the
project wheel into a separate venv and checks every module's import location and
installed version. This does not establish minimum numerical dependency versions. Publishing is a separate action
from creating commits and running checks.

`.tmp` contains preserved local probes and generated validation output and is
ignored after durable oracles were promoted. `literature/` is a local source cache;
keep citations and revisions tracked. `dist/`, caches and generated site output
are disposable build artifacts, not coefficient provenance.
