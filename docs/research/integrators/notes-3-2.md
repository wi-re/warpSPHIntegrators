> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.2 What a fixed-point solve can and cannot do

Implicit methods exist for stiffness. A Picard iteration
`Y^{m+1} = y^n + dt·a_ii·f(Y^m) + …` converges only when `|dt·a_ii·L| < 1`, which is
the step restriction implicit methods are supposed to remove. Backward Euler —
unconditionally stable in exact arithmetic — with the probe's fixed-point solve, on
stiff oscillators at `dt = 0.1` [verified]:

| `k` | `dt·ω` | Picard(20) | Newton, FD Jacobian |
|---|---|---|---|
| 4 | 0.2 | 4.7e-01 | 4.7e-01 |
| 1e2 | 1.0 | 1.0e+00 (stalled, no damping) | 9.8e-04 |
| 1e4 | 10 | **diverged** (3.7e+239) | 3.8e-15 |
| 1e6 | 100 | **diverged** (nan) | 9.8e-17 |
| 1e8 | 1000 | **diverged** (nan) | 1.0e-18 |

Backward Euler is L-stable, so on a highly oscillatory undamped problem it should damp
hard towards zero and stay bounded; the Newton column does exactly that at every
stiffness, and is still bounded at `dt·ω = 1000`. Picard is already wrong at
`dt·ω = 1` — it stalls without damping at all — and blows up past that. So the
practical Picard limit is *tighter* than the textbook `|dt·a_ii·L| < 1`.

The conclusion is not "implicit needs Newton", it is **two different regimes with two
different answers**:

- **Non-stiff, which is the normal regime here.** A fixed 2-iteration Picard gives
  full order (§3.1), needs no norm, no Jacobian, no convergence test, no
  data-dependent control flow. That is Phase 2 and it is genuinely cheap.
- **Stiff.** Needs Newton, and Newton needs §3.4 — but that turns out to be much less
  of an obstacle than it first appears.
