# Contributing

Use [REPOSITORY_PLAN.md](REPOSITORY_PLAN.md) as the current task list. The
[developer index](docs/development/README.md) links architecture, numerical records
and retained designs. Historical experiments retain their original dates and
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
Both `.[test]` (used by CI) and `.[dev]` include SciPy for independent numerical
references and spatial solver tests; the library's basic runtime does not require it.
Julia is needed only for the independent coefficient oracles, not normal tests.
Use Node 22.8 or newer for the browser checks and documentation workflow; Node
24.21.0 is the current local validation runtime.

An editable install needs refreshing after source-version or packaging metadata
changes. Re-run `python -m pip install --no-deps --no-build-isolation -e .` when
its existing dependencies already satisfy the project. Check both the imported
source and distribution metadata:

```bash
python -c "import importlib.metadata as m, warpSPHIntegrators as w; print(w.__file__); print(w.__version__, m.version('warpSPHIntegrators')); assert w.__version__ == m.version('warpSPHIntegrators')"
```

This verifies the development install; the isolated wheel gate below separately
checks artifact version, import locations and installed behavior.

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

The [current validation record](docs/development/repository-cleanup-batch5-2026-10-23.md)
links the dated full CPU and installed-wheel reports, with commands, versions
and scope. Retain the report fingerprint when reusing a gate. Timings are local
measurements rather than promised CI durations; concurrent shards share host
resources.

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
2. Choose the representation before editing the catalog. For a fixed method,
   add pure coefficients under `src/warpSPHIntegrators/schemes/coefficients`,
   bind a definition under `schemes/definitions` to an engine under `drivers`,
   and register it in `schemes/registry.py`. Stable identifiers and family enums
   live in `schemes/enums.py`. Preserve existing numeric enum values and lookup
   names; record equivalent maps without silently deleting identifiers.
   Parameterized configurations belong under `schemes/families`, with public
   exports in `schemes/__init__.py` and stepping in the appropriate driver. They
   need not add fixed registry entries. Keep the four-entry root API focused.
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

Consumer maps such as ConservedLagrangian and ProjectedStep have application
prediction/finalization contracts beyond an ordinary ODE RHS. Document their
state tags, hooks, clocks and stage data, qualify direct steps and the intended
run/restart path, and retain no-order/no-estimate boundaries where appropriate.
Run callbacks must accept forwarded step options such as history. Custom systems
use application serialization; in-memory continuation does not establish portable
TensorState checkpoint support. See the [consumer qualification](docs/development/consumer-map-qualification-2026-10-23.md).

## Documentation and migration checks

```bash
python scripts/validate_repository.py
python scripts/validate_doc_examples.py
python scripts/generate_catalog.py --check
python web/tools/export_catalog.py --check
python web/tools/export_reference.py --check
npm --prefix web run check
npm --prefix web test
npm --prefix docs ci
npm --prefix docs run build
```

Runnable Python fences have `[example:run scope]: #` or
`[example:include path/to/source.py]: #`; caller-dependent sketches use
`[example:skip reason]: #`. Included examples must match the saved source verbatim.
The site uses repository URLs for developer/research documents outside `docs/docs`.
Durable plots live in `docs/assets`; benchmark drivers locate them from their own
file, independently of the shell's working directory. Browser source lives in
`web/`; its generated assets share a content version across modules, workers and
coefficients. Optional Dawn checks need the external binding described in
`web/README.md`; ordinary browser tests explicitly skip those checks without it.
The local `web/.openai/hosting.json` association is ignored and preserved on disk.
GitHub Pages deployment is configured by `.github/workflows/docs.yml` and does
not depend on that local hosting-service project ID.

## Release validation

Confirm package/version/changelog consistency, generated catalog and provenance,
full CPU gate and shard partition, example/link checks, and production site build.
Retain environment, commands, timing and numerical caveats in a dated validation
record. Run `python scripts/validate_wheel.py --cpu-torch` to build and test the wheel
in a fresh environment outside the checkout before publishing; the
editable checkout is not evidence of wheel correctness. CI exercises a fresh dependency install. Offline local validation can use
`--reuse-dependencies`: it shares numerical dependencies but still installs the
project wheel into a separate venv and checks every module's import location and
installed version. It also runs the custom consumer-map tests against the
installed factories, including in-memory continuation and the portable-save
boundary. The wheel must contain the configured `license.md` bytes and no stale
modules from an earlier build. This does not establish minimum numerical
dependency versions. Publishing is a separate action from creating commits and
running checks.

Keep changes in Unreleased until an actual release is prepared. Confirm
`pyproject.toml`, package `__version__`, installed metadata and wheel metadata
agree, review the complete source/consumer commit set, then explicitly authorize
publication. `scripts/publish_pypi.sh` builds and uploads; it is not a dry-run
validation command. Do not run it as a substitute for the checks above.

`.tmp` contains preserved local probes and generated validation output and is
ignored after durable oracles were promoted. `literature/` is a local source cache;
keep citations and revisions tracked. `dist/`, caches and generated site output
are disposable build artifacts, not coefficient provenance.
