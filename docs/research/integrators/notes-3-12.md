> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.12 Phase 14: the structured RHS interface — done 2026-09-10

Phase 7's Rosenbrock-W and exponential integrators need a third kind of RHS
structure — a linear-operator action `L·v` and the nonlinear remainder
`N = f − L·y` — and there was nowhere to put it. Rather than add a second
NamedTuple beside `IMEXRHS`, the RHS input surface collapsed to exactly two
shapes: **a plain callable** (fully implicit, or just evaluated) or **a typed
`RHS`** whose declared capabilities cover every split any registered or
planned scheme asks for. `IMEXRHS` folds in as one shape of the latter.

What landed:

- **`rhs.py`** — the interface. `RHS` is a *concrete* class, not a bare
  structural `Protocol`, so a plain function never `isinstance`-matches it.
  `__call__(state, dt, *args, **kwargs)` is the combined right-hand side
  `f = f_E + f_I = L·y + N` — always defined, the ground truth — and
  `provides` is a `frozenset` over `{"explicit", "implicit", "linear",
  "nonlinear"}`. All four optional accessors share the same `(state, dt,
  *args, **kwargs) -> update[, aux]` contract as the combined callable, so a
  driver routes each one through `updateStep` exactly as `ark.py` already
  routes the two `IMEXRHS` halves. `linear(state, …)` reads the integrated
  field(s) it acts on from `state` and returns `L·(field)`; to apply `L` to
  an *arbitrary* vector (a Krylov iterate, a preconditioner probe) the caller
  hands it a state built with `unflatten_integrated`.
  - `IMEXRHS(explicit=, implicit=)` is now a thin constructor returning an
    `RHS` with `provides={"explicit","implicit"}` and a summing `__call__`
    (the old NamedTuple in `specs.py` is gone).
  - `SemilinearRHS(linear=L, nonlinear=N)`, and the `SemilinearRHS(linear=L,
    combined=f)` form, which synthesizes `N = f − L·y` inside `resolve`.
  - `RHS(**parts)` general constructor accepting any subset, including the
    overlap case — the same stiff operator behind both `implicit` and
    `linear`, where the additive halves define `f` and the semilinear parts
    are a view of it.
- **Driver resolution (`resolve`)** — synthesizes what can be synthesized
  (`implicit` ← the combined `f`; `explicit` ← absent; `nonlinear` ← `f − L·y`
  when a `linear` is declared, else the combined `f`), and requires what
  cannot (`linear`). A plain callable resolves to `explicit=None, implicit=f,
  linear=None, nonlinear=f` — bit-identical to the old `isinstance`
  else-branch, and the bit-identity is pinned by the golden master below. When
  a scheme needs a split the RHS does not carry, it fails **before the solve**
  with a `TypeError` naming the missing capability and what the given RHS
  actually provides. `ark.py` and `imex.py` moved from `isinstance(f,
  IMEXRHS)` to this check; nothing downstream constructs an `IMEXRHS`, so the
  reshape had no external contract to preserve.
- **`check_contracts`** — the three split contracts, checked on demand (tests
  or a harness; *not* on the production step path): additivity is disjoint
  (`explicit + implicit == f`); `linear` is linear and homogeneous (`L·0 = 0`,
  spot-checked `L(a+b) ≈ L(a)+L(b)` on random vectors built through
  `unflatten_integrated`); and any two declared views agree (`linear·y +
  nonlinear == f`).
- **`viscousBurgers` problem factory** (the `PROBLEMS` registry now has twelve)
  — semi-discrete viscous Burgers `u_t + (u²/2)_x = ν u_xx` as the canonical
  semilinear benchmark: `ν u_xx` is a constant-coefficient linear operator
  `L` with a cheap matrix-free action, `(u²/2)_x` is the nonlinear remainder
  `N`. Shipped as a `SemilinearRHS`, it doubles as the Phase 7 accuracy/cost
  problem — exponential integrators are conventionally demonstrated on exactly
  this equation. Distinct from the existing inviscid `burgers_problem`
  (Lax-Friedrichs, the §3.11 TVD/SSP classifier's nonlinear problem).
- **`tests/test_rhs.py`** — 19 tests pinning the four validation gates:
  (1) *every registered scheme* reproduces the pre-refactor golden master on a
  bare callable, bit for bit — `tests/data/bitident_golden.json`, captured
  *before* the refactor, dt = 0.1 for 3 steps on the one-DOF oscillator (exact
  in float, so the comparison is strict equality); (2) `IMEXRHS` (now a
  constructor) and a hand-built `RHS` with the same halves give identical ARK /
  IMEX Euler trajectories; (3) a `linear`-consuming scheme raises the specific
  capability error on a plain `f` and on an `IMEXRHS`, not a mid-solve shape
  mismatch; (4) the `nonlinear` synthesized from `f − L·y` matches an
  independently supplied `N` on viscous Burgers to round-off. Plus the
  `check_contracts` pass/fail cases, the resolution rules, and an end-to-end
  run integrating viscous Burgers through plain RK4 with the mass conserved to
  round-off.

Suite: 2346 → 2365 passing.
