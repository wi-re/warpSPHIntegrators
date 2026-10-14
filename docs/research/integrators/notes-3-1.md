> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.1 Headline: both are cheaper than they look, and only one wall is left

The stage machinery already generalises further than the registry uses it.

- **A diagonally-implicit Runge–Kutta driver needs no new state algebra at all.**
  [verified] A ~60-line DIRK loop written against nothing but the existing public
  helpers — `initializeNewState`, `applyStateUpdate`, `explicit_step`, `updateStep`,
  `initializeSystem`, `finalizeSystem` — reaches full order on all three problems:

  | tableau | oscillator | forced | damped |
  |---|---|---|---|
  | Backward Euler (1) | 0.97 | 1.01 | 0.96 |
  | Implicit midpoint (2) | 2.00 | 2.00 | 2.00 |
  | Trapezoidal / Crank–Nicolson (2) | 2.00 | 2.00 | 2.00 |
  | SDIRK2, L-stable, γ=1−√2/2 (2) | 2.00 | 2.00 | 2.00 |

  The reason it falls out for free is that a DIRK stage equation
  `Y_i = y^n + dt·Σ_{j<i} a_ij k_j + dt·a_ii·f(Y_i)` has *exactly* the shape
  `butcher._weighted_update` already builds. Iterating it is a loop around code that
  exists.

- **The `b`-weight accumulation already accepts a list of updates with a list of step
  sizes** — `fields._resolve_delta` / `_accumulate` handle it, and nothing in the
  registry uses that path. That is precisely `y^{n+1} = y^n + dt·Σ_j β_j k^{n-j}`, so
  an Adams–Bashforth step is one `applyStateUpdate` call. [verified] AB2–AB4 and
  ABM2–ABM4 (PECE) all reach nominal order.

- **Two fixed Picard iterations are enough for a second-order implicit tableau.**
  [verified] With no convergence test and no early exit — a fixed, data-independent
  iteration count, which is what CUDA-graph capture and deterministic ML training both
  need:

  | iterations | implicit midpoint | SDIRK2 |
  |---|---|---|
  | 1 | 1.03 / 1.00 / 1.04 | 1.01 / 1.01 / 1.02 |
  | **2** | **2.01 / 2.00 / 2.02** | **2.00 / 2.00 / 2.01** |
  | 4 | 2.00 / 2.00 / 2.00 | 2.00 / 2.00 / 2.00 |

  (oscillator / forced / damped.) Each iteration buys one order, so `p` iterations
  suffice for order `p` from a trivial predictor. This matters more than it looks —
  see §3.4.

The remaining wall, and the only one §3.0 does not remove:

- **The library never sees the state as a vector, and never sees `f`'s Jacobian.**
  `f` is a black box returning a tagged update object. There is no `norm`, no `dot`,
  no flatten. A fixed-point stage solve needs none of those — which is why the probe
  works — but a *stiff* solve needs all of them.

The wall this section used to lead with — "one RHS evaluation is one neighbour
rebuild, so §2.3's neighbour-list item is a hard prerequisite" — **does not apply**,
because the adjacency is carried through the state and revalidated cheaply (§3.0).
Implicit iteration costs force evaluations, not neighbour searches.
