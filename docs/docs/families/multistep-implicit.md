---
sidebar_label: Multistep (implicit)
---

# Implicit linear multistep — BDF & Adams-Moulton (`MultistepImplicit`) — 8 schemes

The implicit multistep family: **BDF1-BDF5** (`bdf.py`) and the
JFNK-corrected **Adams-Moulton 2-4** (`multistep.py`). One step = one
nonlinear solve (JFNK by default) for the new state, with the past states
entering **as state values** from the scheme's `StepHistory` (state-snapshot
history, threaded with `history=`).

## BDF (backward differentiation)

The order-$p$ formula, newest state first:

$$
y^n = \sum_{j=1}^{p} c_j\, y^{n-j} \;+\; \beta_p\, \mathrm{dt}\, f(t^n, y^n),
$$

with the unique weights from the BDF order conditions (`getBDFCoefficients`;
`tests/test_bdf.py` re-derives and cross-checks them):

| Order | state weights $c_j$ (newest first) | derivative weight $\beta$ |
|---|---|---|
| BDF1 | $1$ | $1$ |
| BDF2 | $4/3,\ -1/3$ | $2/3$ |
| BDF3 | $18/11,\ -9/11,\ 2/11$ | $6/11$ |
| BDF4 | $48/25,\ -36/25,\ 16/25,\ -3/25$ | $12/25$ |
| BDF5 | $300/137,\ -300/137,\ 200/137,\ -75/137,\ 12/137$ | $60/137$ |

The right-hand side is evaluated **once, at the new grid time** — the standard
formula keeps its claimed order on non-autonomous problems. BDF1 *is*
backward Euler.

**Stability.** BDF1/2 are A-stable (BDF1 L-stable); **BDF3-5 are
A($\alpha$)-stable** with cone half-angles 86.03°, 73.35°, 51.84° — stiff
oscillatory modes outside the cone are not damped (measured and pinned in
`tests/test_bdf.py`). BDF3–5 now bootstrap with order-matched L-stable
implicit RK methods: ESDIRK3(2)4L[2]SA, KenCarp4 and KenCarp5.
At rate 100, the forcing/transient checks retain orders 3–5 and remain bounded
at dt=0.1, where the historical explicit bootstrap amplifies the transient.
These starters remove the explicit startup ceiling; they do not widen BDF's
steady-state stability cone or guarantee accuracy at arbitrary stiffness.


## Adams-Moulton (implicit, JFNK-corrected)

The order-$p$ Adams-Moulton corrector, solved implicitly at the endpoint:

$$
y^{n+1} = y^n + \mathrm{dt}\,\big(\gamma_1 f^{n+1}(y^{n+1}) + \gamma_2 f^n + \cdots\big),
$$

with the corrector weights (newest first, predicted point first — the same
weights the explicit PECE family uses, see
[multistep-explicit](multistep-explicit)): AM2 $(1/2, 1/2)$ (AM2 is the
trapezoidal rule — A-stable, symmetric), AM3 $(5/12, 8/12, -1/12)$, AM4
$(9/24, 19/24, -5/24, 1/24)$. The known part of the formula can optionally be
predicted with the matching Adams-Bashforth weights (the `predictor` flag),
which gives the implicit solve a better initial guess.

**Cost.** One JFNK solve + one (optional) predictor evaluation per step —
the *implicit* counterpart of ABM: AM4 costs one solve where ABM4 costs two
evaluations. The stability regions are bounded (not A-stable, except AM2's
trapezoidal limit); the stiffly-damped alternative is the BDF/ESDIRK family.

## History, cold start, diagnostics

BDF3–5 default to `startup='implicit'`; `startup='explicit'` preserves DP5 for
comparisons. Their implicit starter has the same order as the BDF formula, so
startup errors cannot cap the formal global order. A BE warm-up by itself can
cap higher-order trajectories; it is not offered as an order-preserving mode.
BDF2 retains DP5 by default, with `startup='implicit'` selecting TR-BDF2.
BDF1 needs no bootstrap. Starter diagnostics reach the result and run API.

AM and split IMEX multistep methods retain their existing DP5 cold starts.
SBDF without an explicit part delegates to BDF, including its startup. Thread
sufficient history; BDF rejects insufficient capacity. BDF and the run API clear
history before consuming a changed timestep. A cold BDF step can involve several
implicit RK stage solves; after startup it uses one nonlinear solve per step.
Historical TVD/CFL measurements used the old startup and remain dated evidence;
current numerical tests cover the new behavior.

`priorStep` is refused across the family — the state-snapshot history *is*
the multistep's reuse mechanism.

## When to use

- **BDF1/2**: A-stable, cheap, robust — the default implicit multistep for
  mildly stiff problems.
- **BDF3-5**: more order, but the A($\alpha$) cone excludes stiff oscillations;
  use for stiff *dissipative* problems where the explicit CFL is the cost.
- **AM2-4 (implicit)**: the trapezoidal rule and its higher-order siblings —
  symmetric (AM2) or higher-order with a bounded region, one solve per step.

## References

- Hairer, Nørsett, Wanner, *Solving ODEs I* (1993), Ch. 4 — Adams; *Solving ODEs II* (1996), Ch. 4 — BDF and A($\alpha$) stability.
- Lambert (1991) — BDF order conditions and stability cones.

**Tests:** `tests/test_bdf.py` (order conditions, the cones, the cold-start
cap), `tests/test_am.py` (AM orders, the predictor option, the cost
accounting), `tests/test_stiff.py` (stiff damping vs. the explicit family),
`tests/test_tvd.py` (the measured CFL 1.5).
