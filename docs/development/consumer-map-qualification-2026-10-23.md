# Consumer map packaging and continuation qualification

Qualified 23 October 2026 during cleanup Batch 4. ConservedLagrangian and
ProjectedStep factories, drivers and public exports are present in a fresh
0.7.0 wheel built from a clean source staging directory. Analytic custom-state
tests run against that installed wheel outside the checkout. This qualifies
package behavior; the companion frontend's production restart codec remains
application-owned.

## Qualified behavior

| Map | Checks and result |
|---|---|
| ConservedLagrangian hancock/ssprk/kdk | Each moving-quadrature map passes independent oscillator algebra, clocks/stage lifecycle, input ownership, passive conserved slots, state gradients and invalid/reuse contract checks. A full eight-step run equals a four-step run followed by in-memory checkpoint continuation exactly on the binary-step test grid. |
| ProjectedStep | A complete kick/drift map executes once per step, preserves physical endpoint time, masking/shift order and input ownership, and passes gradients/zero-step primitive checks. An integrated custom state continues through solve without repeating the map; the eight-step position/velocity endpoint matches independent constant-acceleration recurrence algebra. |
| Checkpoint ownership | Continuing leaves the checkpoint's saved state and caller input unchanged. The same method and application callbacks are resupplied on continuation. |
| Constant-only projection state | Direct stepping works. The runner explicitly rejects a state without integrated tensor fields. |
| Portable checkpoint files | Both custom systems raise the existing plain-TensorState TypeError. A user-defined system must provide its own serialization; in-memory continuation is not a portable-codec guarantee. |

The conserved test uses t=2, x=1 and conserved slots (mass=1, momentum=.25,
passive quantity=3), h=1/32, intermediate endpoint 2.125 and final endpoint 2.25.
The projection test starts at x=1, v=3 with constant acceleration -2, using the
same grid. One kick then drift per step gives final x=1.6796875 and v=2.5.
Binary steps permit exact full/resumed comparisons without endpoint-grid
roundoff ambiguity. These are consumer-map identities, not general ODE order
claims or production SPH conservation evidence.

The original seven conserved and five projection cases are extended by three
conserved continuation cases and two projection runner/boundary cases. The
installed-wheel gate runs all **17 consumer cases** after its module/export and
ordinary runtime smokes. The editable focused run adds five category-gate cases,
for **22 passing tests**.

## Packaging and source boundaries

The [wheel report](evidence/repository-cleanup-batch4-wheel.json) verifies version
0.7.0 and isolated project installation, with numerical dependencies shared from
the existing environment. Every imported package module resolves inside the
temporary environment. The archive contains exactly the current Python module
inventory, its typing marker and the configured license bytes, without stale
flat modules or bytecode caches.

The strengthened gate stages `license.md` and the package explicitly includes
that lowercase filename. The old default license globs and uppercase-only
staging omitted it. The source version and ordinary root exports are unchanged;
packaging and test tooling change, while both numerical drivers remain intact.
This run does not establish fresh dependency resolution or minimum dependency
versions; CI's fresh-install mode remains separate.

The [integration status](consumer-map-integration-status-2026-10-23.md) retains
the file inventory, intended two-family commit boundaries and downstream records.
Source additions remain uncommitted. No sibling work was edited or staged, and
no package was published. The earlier downstream source-parity/performance
records are not rerun as production physics or serialization tests here.

## Reproduction

From the repository root in the documented conda `warp` environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q tests/test_conserved_lagrangian.py tests/test_projected_map.py tests/test_test_gates.py
python scripts/validate_wheel.py --reuse-dependencies --output-dir .tmp/repository-cleanup-batch4-2026-10-23/wheel
```

The wheel gate runs its copied tests with isolated Python outside the checkout,
without the repository conftest. All three conserved variants and the integrated
projection callback accept the runner's forwarded history/step options. Future
changes to these contracts must repeat their family, run, gradient and wheel
qualification before claiming the same continuation behavior.
