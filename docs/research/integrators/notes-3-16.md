> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.16 Phase 7: EXPRB32 (exponential Rosenbrock) — done 2026-09-13

The second exponential method in the registry, and the one that carries the
exponential family's cost case — the roadmap's order-3 exponential Rosenbrock method,
which closes the Phase 7 cost gate (§3.15 deferred it here). Unlike ETD2RK it needs
no semilinear split: it freezes the **full** right-hand-side Jacobian, so it runs on
plain callables as registered (the generic `scheme`-fixture problems, including the
TVD/SSP advection sweep) and on `SemilinearRHS` just as well.

**Method.** **EXPRB32** (2-stage, **order 3**, **L-stable**, with an embedded order-2
estimator), **Hochbrueck, Ostermann & Schweitzer, *SIAM J. Numer. Anal.* 47(1) (2009)
786–803** (`literature/080717717.pdf`), in the frozen-operator reformulation of
§2.3/§6.5 with `F_n = f(t_n, y_n)` and `J_n = D_y f(t_n, y_n)` frozen at the step
start:

```
U2      = y_n + h·phi_1(h·J_n)·F_n   [+ h²·phi_2(h·J_n)·f_t     (non-autonomous)]
D2      = f(t_n + h, U2) − F_n − J_n·(U2 − y_n)   [− h·f_t]
y_{n+1} = U2 + 2h·phi_3(h·J_n)·D2
```

Two stages; **four full-`f` evaluations per step** with the default `f_t='fd'`
(`F_n`, the `f_t` probe, `f(t_n+h, U2)`, and the frozen-`J_n` application
`J_n·(U2 − y_n)`) and three with `f_t='none'`; **up to three `phi_k(h·J_n)` Krylov
builds per step** (`phi_1` and, when `f_t` is active, `phi_2` in stage 1; `phi_3` in
stage 2 — skipped when `f_t = 0` and, on linear problems, when `D2` is at rounding).
The embedded estimate is the order-2 exp-Rosenbrock–Euler stage `û = U2`, so the
estimate is the corrector's correction itself,
`y_{n+1} − û = 2h·phi_3(h·J_n)·D2` — no extra evaluation.

**Full Jacobian, no semilinear split.** `J_n` is the exact Jacobian action
(`jfnk.jvp_matvec` over the full step machinery — preprocess + `f` + postprocess, so
the action is the Jacobian of the *full* right-hand side, `w='jvp'` default) or
finite differences (`w='fd'`); every `phi_k(h·J_n)v` is applied **matrix-free** by the
same `krylov_phi` Arnoldi machinery as ETD2RK. **There is deliberately no
`w='linear'` option**: a `J = L`-only frozen operator is order 3 only for a linear
right-hand side and **first order** on a genuinely nonlinear problem (the same order
drop ROS3P's `w='linear'` documents, §3.14), so it is documented, not shipped. The
non-autonomous term is a one-sided finite difference by default (`f_t='fd'`, step
`eps_machine^(1/3)·max(1, |dt|)` — the `_default_ft_eps` shared with ROS3P);
`f_t='none'` for autonomous right-hand sides saves one `f` evaluation and the `phi_2`
build (the probe returns `f_t = 0` bit-exactly and the build is skipped).

**Gradient.** Frozen-`J_n` re-attachment: each `phi_k(h·J_n)` action is re-attached
with the transpose `phi_k(h·J_n^T)` (the same Krylov machinery on `v -> dt·J_n^T v`,
`J_n^T` from one VJP graph at `(t_n, y_n)` — the ROS3P `wT` pattern), and
`J_n·(U2 − y_n)` with `J_n^T`. On a linear right-hand side the frozen operator *is*
the exact constant operator, so the gradient is **exact up to the Krylov tolerance**
(measured parity 7.2e-10). On a semilinear one, every frozen-operator occurrence in
the step map carries a factor of `dt` (`U2 − y_n = O(dt)`, and the `phi` inputs are
`O(dt)`), so the error in the dropped `dJ_n/dy` terms is **O(dt²)·||dJ/dy||** — one
power of `dt` better than the Rosenbrock frozen-`W` adjoint (O(dt) on semilinear);
the measured parity on viscous Burgers (4.6e-8 at `dt = 0.02`) sits at the
finite-difference floor. The `D2` zero guard on linear problems is gradient-safe: the
adjoint chain through `D2` vanishes on a linear graph.

**Registration.** `exprb32 = 57` in the enum; `IntegrationScheme(integrateEXPRB32,
'EXPRB32', IntegrationSchemeType.exprb32, 3, True, True, implicit=False, steps=1,
stiffly_accurate=False, stability='L')`. No `need_linear` — plain callables and
`SemilinearRHS` both resolve to the combined `f`. Not stiffly accurate (the last
stage is `U2`, not the step), so no first-stage reuse. The embedded (3, 2) pair means
`error` carries the estimate (inverse of ETD2RK's `error is None`).

**Verification** (all pinned in `tests/test_exponential.py`, 10 tests; the generic
`scheme`-fixture tests run unmodified on EXPRB32):

- **Order 3 on viscous Burgers** (fine-`dt` RK4 reference, Gaussian IC): relative L2
  error `[2.72e-03, 3.50e-04, 4.41e-05, 5.55e-06, 6.97e-07]` at
  `dt = [0.08, 0.04, 0.02, 0.01, 0.005]`, measured orders `[2.96, 2.99, 2.99, 2.99]`.
  The error constant is ~13x smaller than ROS3P's at the same `dt` (4.4e-5 vs
  5.6e-4 at `dt = 0.02`) — that is what buys the cost gate below.
- **Exact on a purely linear right-hand side**: `J_n` is constant, `D2` vanishes,
  the step is `exp(h·J_n)·y_n` applied through the Krylov `phi_1` build (the `phi_3`
  build skipped by the zero guard), and the error is at the Krylov tolerance — not
  `O(h^3)`.
- **L-stable**: a stiff linear mode (`rate = 10`, `h = 1`, `rate·h = 10`) is damped
  by `exp(-10) ~= 4.5e-5` (killed, well under 1e-3) — the exponential signature, in
  contrast to the Rosenbrock A-stability ceiling `1 − √3 ≈ −0.732`.
- **Driver contract**: two stages; full-`f`-class evaluations per step are
  `[2, 2]` per stage with `f_t='fd'` and `[1, 2]` with `f_t='none'` (the Krylov
  builds are counted separately, in `gmres_iterations`); `error` is not None; a
  plain callable runs as-is (the inverse of ETD2RK's `TypeError`); `priorStep` is
  rejected with a `RuntimeWarning`; the caller state is not mutated.
- **Embedded estimate**: `O(h^3)` local error — it shrinks 8.75x under `dt -> dt/2`
  on viscous Burgers (a (3, 2) pair is active on a non-autonomous problem; on a
  linear autonomous one `D2` vanishes and the estimate degenerates to exactly zero).
- **Gradient parity** (autograd vs central FD): linear 7.2e-10 (the Krylov
  tolerance), semilinear 4.6e-8 (the FD floor; the O(dt²) frozen-`J_n` error is
  below it at `dt = 0.02`).
- **TVD/SSP verdict** (`tests/test_tvd.py`): no tableau, so no SSP coefficient; on
  the model advection problem (linear, autonomous) the step is the exact per-mode
  exponential `e^{μ(e^{iθ}−1)}` — a contraction at every CFL — so the sweep finds no
  TV increase up to the CFL cap: `(None, 5.0, True)`.

**Cost** (sine IC, `T = 0.4`; `SolveDiagnostics` summed over the two stages) — the
roadmap's exponential **cost gate** is assessed here. At the same `dt = 0.02` (20
steps), EXPRB32 is the *most expensive* stiff method on the work-unit metric
(80 `f` evals + 1200 JVP Krylov iters = 1280; both per-step builds run to the full
`m = 30`) — but it also reaches 1.43e-5, ~8x below the best order-3 bar. The gate's
"to a fixed error" half is what the order-3 error constant buys: each order-3 method
runs the `dt` ladder and uses the **largest** `dt` that reaches the target error
(`rel L2 ≤ 1e-3`, sine IC):

| scheme               | dt   | steps | rel L2 err | total (work units) |
|----------------------|------|-------|------------|--------------------|
| **EXPRB32 (jvp)**    | 0.04 | 10    | 1.22e-04   | **640**            |
| ROS3P (jvp)          | 0.02 | 20    | 2.04e-04   | 706                |
| ESDIRK3(2)4L[2]SA    | 0.04 | 10    | 5.21e-04   | 715                |
| ARK3(2)4L[2]SA       | 0.02 | 20    | 2.63e-04   | 793                |

**The gate is met: EXPRB32 is the cheapest order-3 method on the work-unit metric at
fixed error (640 < 706 < 715 < 793)**, and it lands the smallest error of the four at
its own `dt` (1.22e-4). The comparison is fair in cost class: each of EXPRB32's
Krylov iterations is a **full Jacobian JVP** — a forward-mode AD sweep through the
full right-hand side, the same cost class as a full `f` evaluation and as ROS3P's
frozen-Jacobian GMRES sweeps (which sit in ROS3P's `rhs_evaluations`) — in contrast
to ETD2RK's cheap `nu·u_xx` `L`-matvecs, where the metric overstates cost (§3.15).
Two honest caveats. First, the 9% margin over ROS3P (706) is thin: it is a
work-unit win in the same cost class, not a FLOP blowout (a JVP sweep carries a
small forward-mode overhead, which if anything widens the gap slightly, but the
margin is what it is). Second, at the same `dt = 0.02` EXPRB32 is the most expensive
stiff method here (1280) — the gate is the fixed-error one, and the ~13x smaller
order-3 error constant is exactly what turns that around. This closes the family's
open cost gate: ETD2RK's order-2 work-unit count (1782, §3.15) was never a clean bar
(cheap `L`-matvecs) and stays documented as such; the exponential family's cost case
now stands on EXPRB32.

**Carve-outs.** Because EXPRB32's step is the *exact exponential flow* on a linear
autonomous right-hand side (constant `J_n`, vanishing `D2`), its error on the
oscillator / damped problems is at the roundoff floor and no order is measurable
(`measured_order` → None). `conftest.EXACT_ON_LINEAR_PROBLEMS = {'EXPRB32'}` skips
those in the generic order tests (the order is pinned on `forced`, `kepler`, and
viscous Burgers instead), in `test_hamiltonian`'s oscillator area test (the exact
flow is area-preserving trivially, so the defect is at rounding; EXPRB32 is
deliberately **not** in `LINEARLY_SYMPLECTIC` — its non-symplectic character on
nonlinear problems is pinned by the kepler dissipation-flag test, which passes with
growth > 2), and in the two embedded-pair order tests (the (3, 2) estimate degenerates
to zero on linear autonomous problems, so they are measured on `forced`). No carve-out
for the TVD/SSP sweep: EXPRB32 runs on the plain-callable advection problem (full-
Jacobian fallback — unlike ETD2RK).

**New tests / artifacts:** 10 in `tests/test_exponential.py` (the ETD2RK helpers
`_run_burgers` / `_grad_parity_error` generalized with a `scheme` parameter, the
ETD2RK tests unchanged); an EXPRB32 verdict `(None, 5.0, True)` in
`tests/test_tvd.py`; `tests/conftest.py` (`EXACT_ON_LINEAR_PROBLEMS`,
`LINEAR_AUTONOMOUS_PROBLEMS`), `tests/test_hamiltonian.py`, `tests/test_embedded.py`;
EXPRB32 rows + the fixed-error gate table in `benchmarks/exponential_benchmark.py` →
`docs/assets/exponential_benchmark.png`; the EXPRB32 sections of
`examples/notebooks/viscous_burgers_demo.ipynb` (§9/§10/§11/§12).
