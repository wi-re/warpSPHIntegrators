> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

## 3.23 Scheme-expansion wave 3: SSPRKMSVS32/43 -- Shu's SSP multistep -- done 2026-10-13

The two strong-stability-preserving *multistep* methods that OrdinaryDiffEq
v7.8.1 ships (`ssprk_perform_step.jl`, the SSPRKMSVS32 / SSPRKMSVS43
perform_step blocks), from Shu, *Efficient time integration for parabolic and
hyperbolic PDEs*, J. Comput. Phys. 169 (2001) 201-228. Wave 3 of
`docs/development/plan-reconciliation-2026-10-23.md`, and the first non-Adams multistep scheme in the
registry -- a new driver, `ssprkmsvs.py`. Unlike the Adams family (which
consumes a *derivative* history), the steady-state formula consumes past
*states*, so each `HistoryEntry` carries a state snapshot **and** the
step-start derivative -- the same `HistoryEntry.state` mechanism BDF already
uses.

**Formulas** (steady state; `f^n = f(t^n, u^n)`):

    SSPRKMSVS32 (3-step, order 2):  u^{n+1} = (3/4)(u^n + 2 dt f^n) + (1/4) u^{n-2}
    SSPRKMSVS43 (4-step, order 3):  u^{n+1} = (16/27)(u^n + 3 dt f^n)
                                       + (11/27)(u^{n-3} + (12/11) dt f^{n-3})

Each bracketed term is one forward-Euler substep, so both steps are convex
combinations of stable maps -- the structure that makes them strongly
stability preserving. Taylor-expanding around `t^n` (verified in
`.tmp/wave3_probe.py` on a 2D nonlinear non-autonomous ODE against a
`solve_ivp` RK45 reference, and re-confirmed by `tests/test_multistep.py`
with history threaded): SSPRKMSVS32 matches the exact flow through `dt^2`
(local error `O(dt^3)` -> order 2), SSPRKMSVS43 through `dt^3` (local error
`O(dt^4)` -> order 3).

**SSP coefficients** (`C = 1 / <binding substep factor>`): SSPRKMSVS32's
binding substep is `u^n + 2 dt f^n` (`2 dt` -> `C = 1/2`); SSPRKMSVS43's is
`u^n + 3 dt f^n` (`3 dt` -> `C = 1/3`), its other substep `(12/11) dt`
giving `11/12` and not binding. These are *structural* properties of the
steady-state convex combination -- not what the TVD trajectory sweep measures
(see below).

**Zero-stability / startup-error decay** (roots of the first characteristic,
`np.roots`):
- SSPRKMSVS32: `rho(ζ) = ζ^3 - (3/4)ζ^2 - 1/4 = (ζ-1)(ζ^2 + (1/4)ζ + 1/4)`.
  Non-principal pair `-1/8 ± (√15/8) i`, `|ζ| = 1/2` exactly. Zero-stable;
  cold-start perturbations decay as `(1/2)^N`.
- SSPRKMSVS43: `rho(ζ) = ζ^4 - (16/27)ζ^3 - 11/27 = (ζ-1)(ζ^3 + (11/27)ζ^2 +
  (11/27)ζ + 11/27)`. Non-principal roots: a real `-0.6835` and a complex
  pair `0.1380 ± 0.7596 i` with `|ζ| = 0.7721`. Zero-stable; cold-start
  perturbations decay as `0.7721^N` (`0.7721^40 ≈ 3.2e-5`).

**Cold start: Dormand-Prince, not ODE's Heun -- and why the order survives
either way.** ODE self-starts these with a 2nd-order Heun step. The bootstrap
runs a *fixed* `k-1` steps (not `T/dt`), so a `q`-order starter accumulates
only `O(dt^{q+1})` starting error; a `p`-order LMM needs starting error
`O(dt^p)`, i.e. a starter of order at least `p-1`. Heun (`q=2`) is exactly the
minimum for the 3rd-order method -- measured order 3 with a Heun starter
(`.tmp/wave3_probe.py` rates 3.09/2.87/2.92, *not* capped at 2) -- and more
than enough for the 2nd-order one. The library's uniform Dormand-Prince 5(4)
starter (S3.7) is a higher-order superset of that minimum, so it is used here
for consistency with every other multistep scheme; the only difference from
ODE is a smaller cold-start error constant, not the order. A deliberate,
documented divergence from ODE's Heun self-start.

**The history stores state snapshots.** The formula reads `u^{n-2}`
(SSPRKMSVS32) / `u^{n-3}` and `f^{n-3}` (SSPRKMSVS43), so the driver needs
`steps` entries -- 2 for SSPRKMSVS32 (`entries[-2]`), 3 for SSPRKMSVS43
(`entries[-3]`). A load-bearing subtlety: for these methods `steps == order`
(a 3-step method of order 2, a 4-step method of order 3), *not* `order - 1`
as for the Adams family. `tests/test_multistep.py` therefore sizes its
`StepHistory` from `scheme.steps` (the value the driver itself uses as
`maxlen`), matching `tests/test_bdf.py`'s `maxlen=max(1, scheme.steps)`.
A draft that sized the history as `order - 1` left the driver one entry short
and bootstrapped forever (the convergence/cost/divergence tests all failed
before the fix).

**Fixed-step only.** No embedded estimate exists, so `result.error is None`
on every call and adaptive stepping is refused by construction (pinned by
`tests/test_multistep.py::test_ssprkmsvs_is_fixed_step_only`).

**TVD verdicts** -- `(None, 1.5, False)` for both, added to
`EXPECTED_VERDICTS`. The `1.5` is the shared Dormand-Prince *cold start's*
limit, **not** the methods' steady-state SSP bounds (`C = 1/2`, `1/3`):
`tv_per_step_increases` does not thread `history`, so the driver re-runs its
DP5 starter every step and never reaches the steady-state convex combination --
exactly the cap the whole Adams/BDF/AM family carries. The tight SSP
coefficients above are structural, not a trajectory limit.

`tests/test_multistep.py` is now 87 tests; `tests/test_family_enums.py` pins
76 members with `MultistepExplicit` at 9.

---
