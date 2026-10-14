# Downstream research records

These are experiments from 26 August 2026 in `warpSPH` / `warpSPHCore` and the
integrator bridge. They were incorporated into the historical JFNK ladder;
downstream code and performance have not been reverified in this migration.

- [DFSPH/Kolmogorov comparison](dfsph-kolmogorov-2026-08-26.md): JFNK design E1.9.
- [Missing velocity correction](dfsph-velocity-correction-2026-08-26.md): E1.10;
  records a production fix in `warpSPH`, not this package.
- [Neighbor-list rebuild ablation](neighborlist-rebuild-2026-08-26.md): E1.8;
  negative result for the tested Newton/GMRES multiplier ceiling.
- [Historical WCSPH/JFNK ladder](wcsph-jfnk-2026-08-24-to-09-04.md): A/B/E1–E1.10.
- [WCSPH/JFNK design and ownership](../../plans/wcsph-jfnk.md).

Full pressure-force JVP, mDBC and free-surface work belongs in its owning
repositories; the [current execution plan](../../../REPOSITORY_PLAN.md) retains
their integration gates.
