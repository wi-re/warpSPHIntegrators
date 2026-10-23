# Conserved Lagrangian consumer, 21 October 2026

The MFM/MFV cleanup in the companion warpSPH checkout moves its existing
Hancock, moving-quadrature SSPRK and KDK recurrences into this package.
`ConservedLagrangian(temporal=...)` is a configurable Stepper available from
`warpSPHIntegrators.schemes`, accepted by `get_scheme`.

The driver uses the established `initialize`, `preprocess`, `postprocess` and
`finalize` lifecycle. A spatial system supplies tagged `position` / `conserved`
fields and `conserved_velocity(Q)` / `wrap_positions(x)` hooks. No SPH package,
EOS, vector layout, geometry, Riemann or closure implementation is imported.
Hancock evaluates an internally predicted RHS over dt. SSPRK and KDK use fresh
unpredicted evaluations (`predictionDt=0`) on their specified configurations.
The package owns stage construction, clocks, kicks, drifts and combination.
Compatible physics corrections remain system hooks, as in the CRK consumer.

The source is the previously shipped warpSPH meshless temporal implementation,
ported with its exact operation order. In particular, the SSPRK position
predictor is not ordinary Heun on `(x,Q)`. No tableau or physical model is
changed. These consumer maps have prediction/kick requirements beyond a generic
ODE RHS, so the family does not advertise a universal ODE convergence order.
No derivative reuse, adaptive error estimator or dense output is provided.

Independent oscillator algebra, input ownership, stage clocks and lifecycle,
constant mass/energy slots, and reverse-mode gradcheck are tested in
`tests/test_conserved_lagrangian.py`. The companion frontend additionally checks
saved-state parity across all temporal choices and records physical runs.
The full CPU test gate is recorded under
`.tmp/test-gate/conserved-lagrangian-2026-10-21/` (local evidence).

Companion architecture and measured results are in
`warpSPH/docs/meshless_gizmo_validation/ARCHITECTURE_2026-10-21.md` and
`PERFORMANCE_2026-10-21.md`. These changes must be carried with the companion
frontend; this record does not assert wheel/release or all restart-codec support.

Full CPU gate: **8,792 passed, 639 skipped**, 350.07 s, eight workers,
one numerical-library thread per worker. The seven new consumer tests pass,
including all three temporal maps and gradcheck. The gate report records the
installed distribution metadata version; this work is in the editable checkout.
