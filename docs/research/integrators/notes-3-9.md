> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.9 The probes behind these numbers

Five scripts, written against the installed v0.5.0 and using only public helpers.
They are not committed — they are evidence for the tables above — but they are the
starting point for Phases 1 and 2 and worth promoting to
`scripts/implicit_multistep_probe.py` if this work is picked up.

| probe | establishes |
|---|---|
| `dirk_probe.py` | A DIRK driver over existing primitives reaches order 1/2/2/2 for backward Euler, implicit midpoint, trapezoidal, SDIRK2 on `oscillator` / `forced` / `damped` (§3.1) |
| `multistep_probe.py` | The list-of-updates path already expresses Adams–Bashforth; AB2–4 and ABM2–4 run; implicit midpoint's energy drift is bounded at 1e-15 over T=160 (§3.1, §3.3) |
| `starter_probe.py` | Self-starting AB caps at order 2; a Dormand–Prince starter recovers 2.97 / 3.99 / 4.03 (§3.7 pain point 1) |
| `stiff_probe.py` | The Picard stage solve diverges at `dt·ω ≳ 10`, i.e. exactly where implicit methods are needed (§3.2) |
| `newton_probe.py` | 2 fixed Picard iterations suffice for order 2 (§3.1); Newton on a **dense, per-column finite-difference** Jacobian — no AD, forward or reverse — stays bounded and L-stably damped to `dt·ω = 1000` (§3.2, §3.4). The dense form is a toy-scale correctness check only — see §3.4's caution before building anything from it. |
