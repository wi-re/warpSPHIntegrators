> Historical measurement record; original section identifiers are retained.
> [Record index](notes-index.md) · [Current execution plan](../../../REPOSITORY_PLAN.md)

### 3.14 Phase 7: ROS3P (Rosenbrock-W) — done 2026-09-11

The roadmap's Phase 7 first family: a **Rosenbrock-W** (linearly-implicit, no
outer Newton loop) method for semilinear stiffness. Each stage is one `gmres`
solve against a *frozen* operator `W ≈ ∂f/∂u` held fixed across the step, so a
step is three linear solves with no nonlinear iteration — the contrast with the
JFNK DIRK/BDF/ARK drivers, which re-linearize inside a Newton loop at every
stage.

Construction. ROS3P is the 3-stage, order-3, A-stable (not L-stable)
Rosenbrock-W method of Lang & Verwer, *BIT* 41(4) (2001) 731–738, in the general
Rosenbrock-W one-step form (2.2):

    (I − τγ W) K_i = F(tn + a_i τ, un + τ Σ_{j<i} a_ij K_j)
                    + τ W Σ_{j<i} g_ij K_j + τ γ_row_i ∂_t F(tn, un)
    un+1 = un + τ Σ b_i K_i

with `γ = 1/2 + √3/6 = 0.788675…`, nodes `a = (0, 1, 1)`, `a21 = a31 = 1`,
`a32 = 0`, `g21 = −1`, `g31 = −γ`, `g32 = 1/2 − 2γ`, `γ_row = (γ, γ−1,
1/2−2γ)`, `b = (2/3, 0, 1/3)`, embedded `b_hat = (1/3, 1/3, 1/3)`. The order-3
condition `γ² − γ + 1/6 = 0` holds in `Q(√3)` (verified symbolically); the
stability function gives `R(∞) = 1 − √3 ≈ −0.732` (A-stable, not L-stable). The
three stage *states* collapse to two distinct RHS points — `(tn, un)` and
`(tn+τ, un+τ K1)` (stage 3 reuses stage 2's state/time) — so a step costs two
`f` evaluations (plus one more for `∂_t F` when `f_t='fd'`). The common
per-stage operator is `M = I − τγ W` (identical for all three stages).

What landed:

- **`rosenbrock.py`** — the dedicated driver `integrateROS3P` (its own
  coefficient structure, not a DIRK tableau; enum 55, order 3, stability `A`,
  `stiffly_accurate=False` so `priorStep` is refused). The `w=` selector picks
  the frozen operator: `'jvp'` (default, exact Jacobian via
  `jfnk.jvp_matvec`), `'fd'` (finite-difference Jacobian via
  `jfnk.fd_matvec`), or `'linear'` (the Phase 14 `linear` accessor). `f_t='fd'`
  (default) finite-differences `∂_t F` one-sided, reusing the stage-1
  evaluation, at a `dt`-scaled step `machine_eps^(1/3)·max(1, |dt|)` (the
  stage-time-shift invariant: the probe time `tn + eps` translates by exactly
  `dt` step to step, which is what the generic driver tests require);
  `f_t='none'` treats the RHS as autonomous. Gradient: the Phase 9 frozen-map
  re-attachment generalized to three solves — each stage solve `K_i = M^{-1} b_i`
  re-attached as a constant linear map whose adjoint is the transposed operator
  (a `gmres` transpose solve), each frozen-operator term `W x` re-attached with
  adjoint `W^T` (a VJP graph). The frozen `W` is held constant in the adjoint (its
  `dW/dy` is a Hessian, dropped) — exact on a linear RHS (`J_N = 0`), a few
  percent on nonlinear ones.

Three non-obvious fixes, each pinned by a test:

1. **The stage-1 `∂_t F` term is `τ·γ_row_1·ft`, not `τ·γ·γ_row_1·ft`.** The
   operator's `τ·γ` (`tau_gamma`) is a different quantity from the stage's
   `γ_row_1`; conflating them injected an extra `γ` and dropped the order.
   Pinned by the non-autonomous (forced) order test.
2. **The stage buffer must come from `initializeNewState`, not
   `unflatten_integrated`.** The latter *clones* non-integrated fields (a shared
   stage counter, a neighbour list), so a `preprocess` that mutates one in place
   would miss the caller's copy and `copied` fields would carry the wrong
   stage's value. `f_flat` now builds the buffer from
   `state.initializeNewState(*args, **kwargs)` and sets the integrated fields in
   place — the same contract the DIRK/BDF drivers use (`bdf._copy_integrated`).
   Pinned by `tests/test_copied_fields.py` (ROS3P now passes all four).
3. **The `W` action runs the full step machinery (`updateStep`), not the bare
   RHS, and the `WK1`/`Wq` RHS-side applications are costed honestly.** The
   Jacobian is of the *full* right-hand side (preprocess + f + postprocess),
   which needs the `copied` fields `preprocess` fills; the frozen point is the
   step start `(tn, un)`, so each Krylov probe re-runs `preprocess`. And `WK1` /
   `Wq` are full `f` evaluations for `w='jvp'/'fd'` (they run
   `combined_counted`) but cheap `linear_fn` applications for `w='linear'` —
   the accounting `w_op = 1 if w in ('jvp','fd') else 0` counts them only in the
   former case, so ROS3P's `rhs_evaluations` is honest against the JFNK
   baselines (which count every full-`f` evaluation too).

The `w='linear'` order drop. On a *genuinely semilinear* problem `w='linear'`
drops the nonlinear Jacobian `J_N` from the frozen operator, so it is a
**first-order** method there (verified on viscous Burgers: measured order 1,
not 3); it is order 3 only when the RHS is linear in the state (`J_N = 0`). It
is the cheap low-order option, not a stand-in for the full Jacobian — documented
on the scheme and in the demo notebook.

Embedded-pair degeneracy. For a linear autonomous `y' = Ay` the frozen operator
`W = A` makes the two embedded stages agree exactly (`K1 == K2`), so the error
estimate `τ(K1−K2)/3` is identically 0; the embedded order is therefore measured
on a non-autonomous (forced) problem (rates ~3.05/3.03/3.01), not the
oscillator.

Verification (viscous Burgers n = 64, ν = 0.01, T = 0.4, vs RK4 `dt = 2e-3`).
`tests/test_rosenbrock.py` (18 tests): order 3 on viscous Burgers for both
`w='jvp'` and `w='fd'`; `w=`-selector agreement; `w='linear'` order-1 semilinear
/ order-3 linear; `f_t='fd'` retains order on a forced RHS while `f_t='none'`
drops it; A-stable-not-L (a `rate = 1000` decay over `dt = 1` lands at
`|1−√3| ≈ 0.732`); embedded order 2 via the propagated `y_hat`; stage-3 reuses
stage-2's `StageResult`; the two distinct stage times; `priorStep` refused; no
caller-state mutation; and gradient parity (exact on linear to 1e-4/1e-3 for
jvp/fd, FD-limited to < 0.25 on the semilinear problem). TVD verdict
`(None, 5.0, True)` — not TVD-classified (no Butcher tableau exposed) but stable
to the tested CFL.

The benchmark (the Phase 7 gate). `benchmarks/rosenbrock_benchmark.py` →
`docs/assets/rosenbrock_benchmark.png`; the same table is §10 of
`examples/notebooks/viscous_burgers_demo.ipynb`. Sine IC, `dt = 0.02` (20 steps), T = 0.4,
work-units = `rhs_evaluations + gmres_iterations` (the JFNK invariant; for
`ROS3P (jvp/fd)` the GMRES matvecs are full Jacobian sweeps already inside
`rhs_evaluations`, so its total is just `rhs_evaluations`):

| scheme (order)   | RHS evals | GMRES iters | total | rel L2 err |
|------------------|-----------|-------------|-------|------------|
| ROS3P (W=jvp) (3) | 706 | 586 | 706 | 2.04e-4 |
| ROS3P (W=fd) (3)  | 715 | 595 | 715 | 2.04e-4 |
| ROS3P (W=linear) (1) | 60 | 783 | 843 | 7.76e-3 |
| ARK3(2)4L[2]SA (3) | 240 | 553 | 793 | 2.63e-4 |
| ESDIRK3(2)4L[2]SA (3) | 300 | 688 | 988 | 7.25e-5 |

Convergence (Gaussian IC, `w='jvp'`): measured orders 2.70, 2.71, 2.82, 2.90 on
`dt = 0.08/0.04/0.02/0.01/0.005` → order 3 (gate (a) met). **Gate (b) met on
total work-units:** among the order-3 methods `ROS3P (W=jvp)` is the cheapest
(706 < ARK3 793 < ESDIRK3 988) and also beats ESDIRK3 on GMRES iterations (586 <
688). Two honest caveats, recorded in the notebook: (i) the additive ARK3
split's GMRES iterations are cheap *linear-operator* applications (the IMEX
split hands the stiff diffusion to the `linear` part), whereas ROS3P's are full
Jacobian sweeps — so ARK3 is cheaper per iteration in FLOPs even though its
work-unit count is higher; (ii) `ROS3P (W=linear)` is cheapest in full-`f`
evaluations (60) but order 1 on this semilinear PDE and pays in the most GMRES
iterations (783).

New tests: 18 in `tests/test_rosenbrock.py`, plus the ROS3P rows added to
`tests/test_embedded.py`, `tests/test_copied_fields.py`, and `tests/test_tvd.py`.

---
