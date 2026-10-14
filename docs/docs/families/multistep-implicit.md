---
sidebar_label: Multistep (implicit)
---

# Implicit multistep — BDF, Adams-Moulton and NDF

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

## NDF: variable-step backward differences

`NDF(order=1..5, max_step_ratio=2.)` binds a fixed target order outside the
142-method catalog. It supplies the pinned QNDF warm difference kernel and
polynomial timestep resampling, with an order-preserving implicit RK startup.
It supports coupled integrated fields, adaptive acceptance and portable tensor
history without requiring a frontend consumer.

[example:run implicit variable-history NDF]: #
```python
import math
import torch
from warpSPHIntegrators import NDF, TensorState, tensor_rhs, solve

rhs = tensor_rhs(lambda t, y: -y)
first = solve(NDF(3), TensorState(torch.ones(4, dtype=torch.float64)),
              .3, rhs, dt=.025)
continued = solve(NDF(3), None, .6, rhs, restart=first.checkpoint)
assert continued.successful
assert torch.allclose(continued.state.y,
                      torch.full((4,), math.exp(-.6), dtype=torch.float64),
                      atol=1e-5, rtol=0)
```

For order p, with backward differences D1 through Dp, the warm solve uses
`prediction = y + sum(Di)`, `phi = sum(H_i*Di)` and
`y_new = prediction - beta*phi + beta*h*f(t+h,y_new)`.
Here H_i is the harmonic sum and
`beta = 1 / ((1-kappa)*H_p)`; kappa is
`(-.185, -1/9, -.0823, -.0415, 0)` for orders 1–5. The local defect is
`(kappa*H_p + 1/(p+1))*(y_new-prediction)`, with exponent p+1.
Order five has the BDF5 constant-step main map. The different history,
startup, error and timestep policy are why this is a separate factory.

The accepted-step policy holds h for at least p+1 equal steps and uses a growth
deadband. Rejected trials may shrink immediately. Direct callers thread
`result.history`; changing a warm step size too soon restarts with the implicit
RK starter. Ratios beyond `max_step_ratio` also restart before memory is used.
This matters: arbitrary alternating steps can amplify a fifth-order computational
mode even when each ratio is below two. Quasi-constant blocks avoid that measured
counterexample; they are not a proof for every variable grid or nonnormal system.
The upstream variable-order QNDF controller is a separate backend extension.

The first p steps use TR-BDF2 (orders 1–2), ESDIRK324L2SA (3), KenCarp4 (4) or
KenCarp5 (5). Their estimator exponents are 2/3/3/4/5, then p+1 on warm steps.
The runner reads `result.estimate_order`. Each warm step uses one nonlinear solve
and a physical endpoint RHS refresh. Preconditioners receive the protected
`stage_coefficient=beta*h`. Diagnostics include bootstrap and refresh work.

History owns its tensors and records state values, layout, dtype, IDs and time.
Changed state values or invalid memory restart automatically. Use
`restart_history=True` after changing RHS parameters or introducing a derivative
discontinuity. Warm initialize and all finalize hooks must preserve integrated
values, time and IDs. Custom finalize opts into `ndf_compatible=True` and reads
`NDFQuadrature` on `result.quadrature`: its `known_state` and `weights` express
the full affine step. Bootstrap weights are the starter's weights plus a
zero-weight endpoint; warm weights contain only beta. Copied endpoint fields
are refreshed. Arbitrary projections require a history restart contract and
are rejected by this driver.

The implementation is for explicit ODE right-hand sides. It does not supply
mass matrices, DAEs, dense output, adaptive order or source-controller/FSAL
equivalence. Fixed-step characteristic-root checks include negative-real decay
at all five orders; orders 3–5 have boundaries entering the left half-plane and
do not have full A-stability. Local defect tolerances do not bound global error.

## References

- Hairer, Nørsett, Wanner, *Solving ODEs I* (1993), Ch. 4 — Adams; *Solving ODEs II* (1996), Ch. 4 — BDF and A($\alpha$) stability.
- Lambert (1991) — BDF order conditions and stability cones.

**Tests:** `tests/test_bdf.py` (order conditions, the cones, the cold-start
cap), `tests/test_am.py` (AM orders, the predictor option, the cost
accounting), `tests/test_stiff.py` (stiff damping vs. the explicit family),
`tests/test_tvd.py` (the measured CFL 1.5).
